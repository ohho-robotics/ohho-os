from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import math
import os
import sys
import time
from typing import Any
import unittest

from ohho.cli import build_parser, main
from ohho.sim_serve import (
    DEADMAN_TIMEOUT_S,
    DEFAULT_PORT,
    SimServe,
    SimServeUnavailable,
    backend_entry,
    build_telemetry_message,
    camera_jpeg_from_ros,
    cmd_vel_rosbridge,
    gated_velocity,
    joint_command_rosbridge,
    odom_from_ros,
    parse_client_command,
    parse_velocity,
    serve_sim,
)


def _has_websockets() -> bool:
    return importlib.util.find_spec("websockets") is not None


class TestVelocityParser(unittest.TestCase):
    def test_parse_nested(self):
        vx, vy, w = parse_velocity(
            {
                "linear": {"x": 0.25, "y": -0.1},
                "angular": {"z": 0.5},
            }
        )
        self.assertAlmostEqual(vx, 0.25)
        self.assertAlmostEqual(vy, -0.1)
        self.assertAlmostEqual(w, 0.5)

    def test_parse_dotted(self):
        vx, vy, w = parse_velocity(
            {
                "linear.x": 0.15,
                "linear.y": 0.05,
                "angular.z": -0.4,
            }
        )
        self.assertAlmostEqual(vx, 0.15)
        self.assertAlmostEqual(vy, 0.05)
        self.assertAlmostEqual(w, -0.4)

    def test_parse_flat_and_xyz(self):
        vx, vy, w = parse_velocity(
            {
                "linear_x": 0.1,
                "linear_y": 0.2,
                "angular_z": 0.3,
            }
        )
        self.assertAlmostEqual(vx, 0.1)
        self.assertAlmostEqual(vy, 0.2)
        self.assertAlmostEqual(w, 0.3)

        vx, vy, w = parse_velocity({"x": 0.4, "y": -0.2, "z": 0.1})
        self.assertAlmostEqual(vx, 0.4)
        self.assertAlmostEqual(vy, -0.2)
        self.assertAlmostEqual(w, 0.1)

    def test_parse_connect_velocity_fields(self):
        vx, vy, w = parse_velocity({"linearX": 0.2, "linearY": -0.05, "angularZ": 0.3})
        self.assertAlmostEqual(vx, 0.2)
        self.assertAlmostEqual(vy, -0.05)
        self.assertAlmostEqual(w, 0.3)

    def test_parse_invalid(self):
        self.assertEqual(parse_velocity(None), (0.0, 0.0, 0.0))
        self.assertEqual(parse_velocity("fast"), (0.0, 0.0, 0.0))
        self.assertEqual(parse_velocity({}), (0.0, 0.0, 0.0))


class TestNotBuiltBackends(unittest.TestCase):
    """Assert that isaac and mujoco exit non-zero, print 'not built', and don't import anything."""

    def test_isaac_not_built(self):
        with (
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            code = main(["sim-serve", "--backend", "isaac"])
            self.assertNotEqual(code, 0)
            output = stdout.getvalue() + stderr.getvalue()
            self.assertIn("not built", output)
            self.assertIn("not-implemented", output)
        self.assertEqual(backend_entry("isaac"), "not-implemented")
        self.assertNotIn("isaac", sys.modules)
        self.assertNotIn("isaacsim", sys.modules)

    def test_mujoco_not_built(self):
        with (
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            code = main(["sim-serve", "--backend", "mujoco"])
            self.assertNotEqual(code, 0)
            output = stdout.getvalue() + stderr.getvalue()
            self.assertIn("not built", output)
            self.assertIn("not-implemented", output)
        self.assertEqual(backend_entry("mujoco"), "not-implemented")
        self.assertNotIn("mujoco", sys.modules)


@unittest.skipUnless(
    _has_websockets(), "websockets not installed — [serve] or [sim-serve] extra"
)
class TestSimServeSimBackend(unittest.TestCase):
    """Test the in-process sim:// robot over WebSocket session."""

    def test_sim_session_driving_timeout_and_estop(self):
        import websockets

        start_time = time.monotonic()

        async def drain_latest(client: Any, timeout: float = 0.05) -> dict[str, Any]:
            latest = None
            while True:
                try:
                    raw = await asyncio.wait_for(client.recv(), timeout=timeout)
                    latest = json.loads(raw)
                    timeout = 0.01
                except asyncio.TimeoutError:
                    break
            assert latest is not None
            return latest

        async def run_test():
            server = SimServe(
                backend="sim",
                robot="omnibot",
                host="127.0.0.1",
                port=0,
                rate_hz=40.0,
            )
            await server.start()
            uri = f"ws://127.0.0.1:{server.port}"

            try:
                async with websockets.connect(uri) as client:
                    # Initial state
                    init_msg = await drain_latest(client)
                    self.assertEqual(init_msg["backend"], "sim")
                    self.assertTrue(init_msg.get("no_camera"))
                    x0 = init_msg["odom"]["x"]

                    # 1. Deadman is false -> commanded velocity is zero
                    await client.send(
                        json.dumps(
                            {
                                "deadman": False,
                                "velocity": {
                                    "linear": {"x": 0.2, "y": 0.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.1)
                    msg = await drain_latest(client)
                    self.assertAlmostEqual(msg["odom"]["x"], x0, places=4)

                    # 2. Deadman is true -> robot drives and odom changes
                    for _ in range(10):
                        await client.send(
                            json.dumps(
                                {
                                    "deadman": True,
                                    "velocity": {
                                        "linear": {"x": 0.2, "y": 0.0},
                                        "angular": {"z": 0.0},
                                    },
                                }
                            )
                        )
                        await asyncio.sleep(0.04)
                    msg = await drain_latest(client)
                    x_moving = msg["odom"]["x"]
                    self.assertGreater(x_moving, x0 + 0.01)

                    # 3. 300 ms timeout -> motion halts if no message arrives
                    await asyncio.sleep(0.35)
                    msg = await drain_latest(client)
                    x_timeout = msg["odom"]["x"]

                    await asyncio.sleep(0.1)
                    msg = await drain_latest(client)
                    self.assertAlmostEqual(msg["odom"]["x"], x_timeout, places=3)

                    # 4. Estop latch -> latches until estop: false arrives
                    await client.send(
                        json.dumps(
                            {
                                "estop": True,
                                "deadman": True,
                                "velocity": {
                                    "linear": {"x": 0.2, "y": 0.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.05)
                    msg = await drain_latest(client)
                    x_estop = msg["odom"]["x"]

                    # Commanded drive messages without estop: false must NOT move robot
                    for _ in range(4):
                        await client.send(
                            json.dumps(
                                {
                                    "deadman": True,
                                    "velocity": {
                                        "linear": {"x": 0.2, "y": 0.0},
                                        "angular": {"z": 0.0},
                                    },
                                }
                            )
                        )
                        await asyncio.sleep(0.04)
                    msg = await drain_latest(client)
                    self.assertAlmostEqual(msg["odom"]["x"], x_estop, places=3)

                    # 5. Estop unlatch -> robot drives again
                    for _ in range(10):
                        await client.send(
                            json.dumps(
                                {
                                    "estop": False,
                                    "deadman": True,
                                    "velocity": {
                                        "linear": {"x": 0.2, "y": 0.0},
                                        "angular": {"z": 0.0},
                                    },
                                }
                            )
                        )
                        await asyncio.sleep(0.04)
                    msg = await drain_latest(client)
                    self.assertGreater(msg["odom"]["x"], x_estop + 0.01)
            finally:
                await server.stop()

        asyncio.run(run_test())
        elapsed = time.monotonic() - start_time
        # Must drive and complete all checks in under 3 seconds (macOS runners)
        self.assertLess(
            elapsed,
            3.0,
            f"Test took {elapsed:.2f}s, expected under 3 seconds",
        )


@unittest.skipUnless(
    os.environ.get("OHHO_ROSBRIDGE"),
    "OHHO_ROSBRIDGE not set — skipped (do not require Gazebo in default CI)",
)
class TestRos2SimServe(unittest.TestCase):
    """Test the ros2 backend over rosbridge forwarding velocity to /cmd_vel and reading /odom."""

    def test_ros2_backend_cmd_vel_and_odom(self):
        import websockets

        cmd_vel_received: list[dict[str, Any]] = []

        async def mock_rosbridge_handler(ws):
            async for raw in ws:
                msg = json.loads(raw)
                op = msg.get("op")
                topic = msg.get("topic")
                if op == "subscribe" and topic == "/odom":
                    odom_pkt = {
                        "op": "publish",
                        "topic": "/odom",
                        "msg": {
                            "pose": {
                                "pose": {
                                    "position": {"x": 3.14, "y": 1.59, "z": 0.0},
                                    "orientation": {
                                        "x": 0.0,
                                        "y": 0.0,
                                        "z": 0.0,
                                        "w": 1.0,
                                    },
                                }
                            }
                        },
                    }
                    await ws.send(json.dumps(odom_pkt))
                elif op == "publish" and topic == "/cmd_vel":
                    cmd_vel_received.append(msg.get("msg", {}))

        async def run_test():
            rb_env = os.environ.get("OHHO_ROSBRIDGE", "")
            rb_server = None
            if rb_env.startswith("ws://"):
                rb_url = rb_env
            else:
                rb_server = await websockets.serve(
                    mock_rosbridge_handler, "127.0.0.1", 0
                )
                rb_port = rb_server.sockets[0].getsockname()[1]
                rb_url = f"ws://127.0.0.1:{rb_port}"

            sim_server = SimServe(
                backend="ros2",
                rosbridge=rb_url,
                host="127.0.0.1",
                port=0,
                rate_hz=40.0,
            )
            await sim_server.start()

            try:
                client_url = f"ws://127.0.0.1:{sim_server.port}"
                async with websockets.connect(client_url) as client:
                    await asyncio.sleep(0.05)
                    raw = await asyncio.wait_for(client.recv(), timeout=1.0)
                    tele = json.loads(raw)
                    self.assertEqual(tele["backend"], "ros2")
                    self.assertAlmostEqual(tele["odom"]["x"], 3.14, places=2)
                    self.assertAlmostEqual(tele["odom"]["y"], 1.59, places=2)

                    # Send velocity command
                    cmd = {
                        "deadman": True,
                        "velocity": {
                            "linear": {"x": 0.25, "y": 0.05},
                            "angular": {"z": 0.75},
                        },
                    }
                    await client.send(json.dumps(cmd))
                    await asyncio.sleep(0.05)

                    self.assertGreater(len(cmd_vel_received), 0)
                    last_twist = cmd_vel_received[-1]
                    self.assertAlmostEqual(
                        last_twist.get("linear", {}).get("x", 0.0), 0.25
                    )
                    self.assertAlmostEqual(
                        last_twist.get("linear", {}).get("y", 0.0), 0.05
                    )
                    self.assertAlmostEqual(
                        last_twist.get("angular", {}).get("z", 0.0), 0.75
                    )
            finally:
                await sim_server.stop()
                if rb_server is not None:
                    rb_server.close()
                    await rb_server.wait_closed()

        asyncio.run(run_test())


class TestRos2ExtraCheck(unittest.TestCase):
    """When ROS 2 environment is absent, --backend ros2 reports dependency on [ros2] extra."""

    def test_ros2_unavailable_without_extra(self):
        old_distro = os.environ.pop("ROS_DISTRO", None)
        old_rb = os.environ.pop("OHHO_ROSBRIDGE", None)
        try:
            # If rclpy is installed in the test env, skip this negative test
            if importlib.util.find_spec("rclpy") is not None:
                self.skipTest("rclpy is installed in this environment")

            server = SimServe(backend="ros2")
            with self.assertRaises(SimServeUnavailable) as ctx:
                asyncio.run(server.start())
            self.assertIn("[ros2]", str(ctx.exception))
            self.assertFalse(server._running)
        finally:
            if old_distro is not None:
                os.environ["ROS_DISTRO"] = old_distro
            if old_rb is not None:
                os.environ["OHHO_ROSBRIDGE"] = old_rb


class TestWebsocketsExtraCheck(unittest.TestCase):
    """When websockets is absent, the sim backend names that dependency."""

    @unittest.skipIf(_has_websockets(), "websockets is installed")
    def test_sim_unavailable_without_websockets(self):
        server = SimServe(backend="sim")
        with self.assertRaises(SimServeUnavailable) as ctx:
            asyncio.run(server.start())
        self.assertIn("websockets", str(ctx.exception))
        self.assertFalse(server._running)


class TestTeleopSessionProtocol(unittest.TestCase):
    """Protocol decisions with no WebSocket, no rosbridge, and no Gazebo."""

    def test_missing_deadman_is_not_held(self):
        command = parse_client_command(
            {
                "velocity": {"linear": {"x": 0.2, "y": 0.0}, "angular": {"z": 0.1}},
                "joints": {"arm_shoulder_pan": 0.4, "skip": True},
            }
        )
        self.assertFalse(command.deadman)
        self.assertIsNone(command.estop)
        self.assertEqual(command.velocity, (0.2, 0.0, 0.1))
        self.assertEqual(command.joints, {"arm_shoulder_pan": 0.4})

        held = parse_client_command({"deadman": True, "estop": False})
        self.assertTrue(held.deadman)
        self.assertIs(held.estop, False)
        self.assertFalse(parse_client_command({"deadman": "false"}).deadman)
        self.assertFalse(parse_client_command({"deadman": 0}).deadman)
        self.assertTrue(parse_client_command({"deadman": "true"}).deadman)

    def test_velocity_gate_zeroes_without_deadman_link_or_fresh_command(self):
        target = (0.2, -0.05, 0.3)
        allowed = dict(
            deadman=True,
            estopped=False,
            client_count=1,
            command_window_open=True,
            now=10.0,
            last_msg_time=10.0,
        )
        self.assertEqual(gated_velocity(target, **allowed), target)
        self.assertEqual(
            gated_velocity(target, **{**allowed, "deadman": False}),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            gated_velocity(target, **{**allowed, "estopped": True}),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            gated_velocity(target, **{**allowed, "client_count": 0}),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            gated_velocity(target, **{**allowed, "command_window_open": False}),
            (0.0, 0.0, 0.0),
        )
        just_inside = allowed["last_msg_time"] + DEADMAN_TIMEOUT_S - 0.001
        self.assertEqual(
            gated_velocity(target, **{**allowed, "now": just_inside}),
            target,
        )
        at_timeout = allowed["last_msg_time"] + DEADMAN_TIMEOUT_S
        self.assertEqual(
            gated_velocity(target, **{**allowed, "now": at_timeout}),
            (0.0, 0.0, 0.0),
        )

    def test_server_message_has_backend_and_camera_or_no_camera(self):
        quiet = build_telemetry_message(
            odom={"x": 1.0, "y": 2.0, "yaw": 0.5},
            joints={"arm_shoulder_pan": 0.25},
            backend="sim",
            camera=None,
        )
        self.assertEqual(quiet["backend"], "sim")
        self.assertEqual(quiet["odom"], {"x": 1.0, "y": 2.0, "yaw": 0.5})
        self.assertEqual(quiet["joints"], {"arm_shoulder_pan": 0.25})
        self.assertTrue(quiet["no_camera"])
        self.assertNotIn("camera", quiet)

        framed = build_telemetry_message(
            odom={"x": 0.0, "y": 0.0, "yaw": 0.0},
            joints={},
            backend="ros2",
            camera="aW1hZ2U=",
        )
        self.assertEqual(framed["backend"], "ros2")
        self.assertEqual(framed["camera"], "aW1hZ2U=")
        self.assertNotIn("no_camera", framed)

    def test_ros_messages_match_connect_topics(self):
        twist = cmd_vel_rosbridge(0.2, -0.1, 0.4)
        self.assertEqual(twist["op"], "publish")
        self.assertEqual(twist["topic"], "/cmd_vel")
        self.assertEqual(twist["msg"]["linear"], {"x": 0.2, "y": -0.1, "z": 0.0})
        self.assertEqual(twist["msg"]["angular"]["z"], 0.4)

        joints = joint_command_rosbridge({"arm_shoulder_pan": 0.5})
        self.assertEqual(joints["topic"], "/arm/joint_commands")
        self.assertEqual(joints["msg"]["name"], ["arm_shoulder_pan"])
        self.assertEqual(joints["msg"]["position"], [0.5])

        half_turn = math.sqrt(2.0) / 2.0
        odom = odom_from_ros(
            {
                "pose": {
                    "pose": {
                        "position": {"x": 1.5, "y": -0.25, "z": 0.0},
                        "orientation": {"w": half_turn, "z": half_turn},
                    }
                }
            }
        )
        self.assertIsNotNone(odom)
        assert odom is not None
        self.assertAlmostEqual(odom["x"], 1.5)
        self.assertAlmostEqual(odom["y"], -0.25)
        self.assertAlmostEqual(odom["yaw"], math.pi / 2.0, places=5)

        self.assertIsNone(
            camera_jpeg_from_ros(
                "/camera/front/image_raw/compressed",
                {"format": "jpeg", "data": ""},
            )
        )
        self.assertIsNone(
            camera_jpeg_from_ros(
                "/camera/wrist/image_raw/compressed",
                {"format": "jpeg", "data": "not-front"},
            )
        )
        self.assertEqual(
            camera_jpeg_from_ros(
                "/camera/front/image_raw/compressed",
                {"format": "jpeg", "data": "aW1hZ2U="},
            ),
            "aW1hZ2U=",
        )

    def test_unimplemented_backends_do_not_import_simulators(self):
        self.assertEqual(backend_entry("isaac"), "not-implemented")
        self.assertEqual(backend_entry("mujoco"), "not-implemented")
        with self.assertRaises(SimServeUnavailable) as ctx:
            serve_sim(backend="isaac")
        self.assertIn("not-implemented", str(ctx.exception))
        with self.assertRaises(SimServeUnavailable) as ctx_mujoco:
            SimServe(backend="mujoco")
        self.assertIn("not-implemented", str(ctx_mujoco.exception))
        self.assertNotIn("isaac", sys.modules)
        self.assertNotIn("isaacsim", sys.modules)
        self.assertNotIn("mujoco", sys.modules)


def _live_odom_x(server: SimServe) -> float:
    """Integrated pose. ``Robot.telemetry`` is the 10 Hz emit cache."""
    bot = server._bot
    assert bot is not None
    return float(bot.transport.read().odom.x)


def _commanded_velocity(server: SimServe) -> tuple[float, float, float]:
    """Velocity last applied to the sim, after the deadman gate."""
    bot = server._bot
    assert bot is not None
    cmd = bot.transport._cmd
    return (float(cmd.linear_x), float(cmd.linear_y), float(cmd.angular_z))


async def _hold_deadman_until_moved(
    client: Any, server: SimServe, *, min_dx: float = 0.01, timeout: float = 2.0
) -> float:
    """Hold deadman until the base moves at least ``min_dx`` meters.

    The physics thread aims for 50 Hz. On macOS CI, after merging main, a
    fixed 200 ms burst only integrates one or two 4 mm steps. Wait until the
    base has actually moved. Sends stay inside the 300 ms deadman window.
    """
    x0 = _live_odom_x(server)
    deadline = time.monotonic() + timeout
    while True:
        await client.send(
            json.dumps(
                {
                    "deadman": True,
                    "velocity": {
                        "linear": {"x": 0.2, "y": 0.0},
                        "angular": {"z": 0.0},
                    },
                }
            )
        )
        await asyncio.sleep(0.05)
        x = _live_odom_x(server)
        if x - x0 > min_dx:
            return x
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"deadman hold moved {x - x0:.4f} m in {timeout:.1f}s"
            )


@unittest.skipUnless(
    _has_websockets(), "websockets not installed — [serve] or [sim-serve] extra"
)
class TestSimServeDeadmanEdges(unittest.TestCase):
    """Wire-level deadman rules against the in-process sim. No Gazebo."""

    def test_missing_deadman_does_not_move(self):
        import json as json_mod

        import websockets

        async def run_test():
            server = SimServe(
                backend="sim", robot="omnibot", host="127.0.0.1", port=0, rate_hz=40.0
            )
            await server.start()
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    raw = await asyncio.wait_for(client.recv(), timeout=1.0)
                    x0 = json_mod.loads(raw)["odom"]["x"]
                    for _ in range(4):
                        await client.send(
                            json_mod.dumps(
                                {
                                    "velocity": {
                                        "linear": {"x": 0.2, "y": 0.0},
                                        "angular": {"z": 0.0},
                                    }
                                }
                            )
                        )
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(0.1)
                    x1 = server._bot.telemetry().odom.x
                self.assertAlmostEqual(x1, x0, places=3)
            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_dropped_socket_stops_within_300ms(self):
        import websockets

        async def run_test():
            server = SimServe(
                backend="sim", robot="omnibot", host="127.0.0.1", port=0, rate_hz=40.0
            )
            await server.start()
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    moving = await _hold_deadman_until_moved(client, server)
                self.assertGreater(moving, 0.01)
                # Past the 300 ms bound the base must already be stopped.
                # One 50 Hz sim step is 4 mm, so the check allows that and
                # rejects a command that is still 0.2 m/s. The settle window
                # is long enough to see that drift on a slow physics wake.
                await asyncio.sleep(DEADMAN_TIMEOUT_S + 0.05)
                await asyncio.sleep(0)
                self.assertEqual(_commanded_velocity(server), (0.0, 0.0, 0.0))
                settled = _live_odom_x(server)
                await asyncio.sleep(1.0)
                later = _live_odom_x(server)
                self.assertLess(later - settled, 0.01)
            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_timeout_is_not_tied_to_telemetry_rate(self):
        """A 1 Hz broadcast must still zero velocity 300 ms after the last message."""
        import websockets

        async def run_test():
            server = SimServe(
                backend="sim", robot="omnibot", host="127.0.0.1", port=0, rate_hz=1.0
            )
            await server.start()
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    moving = await _hold_deadman_until_moved(client, server)
                    self.assertGreater(moving, 0.01)
                    # Stay connected and stop sending. 300 ms must zero the
                    # command even though the next broadcast is 1 s away.
                    await asyncio.sleep(DEADMAN_TIMEOUT_S + 0.05)
                    await asyncio.sleep(0)
                    self.assertFalse(server._command_window_open)
                    self.assertEqual(server._current_velocity(), (0.0, 0.0, 0.0))
                    self.assertEqual(_commanded_velocity(server), (0.0, 0.0, 0.0))
                    stopped = _live_odom_x(server)
                    # Still before one 1 Hz period, so a rate-tied watchdog
                    # would still be integrating 0.2 m/s across this window.
                    await asyncio.sleep(0.55)
                    later = _live_odom_x(server)
                self.assertLess(abs(later - stopped), 0.01)
            finally:
                await server.stop()

        asyncio.run(run_test())

    def test_joint_command_updates_joint_state(self):
        import json as json_mod

        import websockets

        async def run_test():
            server = SimServe(
                backend="sim", robot="omnibot", host="127.0.0.1", port=0, rate_hz=40.0
            )
            await server.start()
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    await client.send(
                        json_mod.dumps(
                            {"deadman": True, "joints": {"arm_shoulder_pan": 0.5}}
                        )
                    )
                    await asyncio.sleep(0.25)
                    raw = await asyncio.wait_for(client.recv(), timeout=1.0)
                    # Drain to the newest snapshot.
                    latest = json_mod.loads(raw)
                    while True:
                        try:
                            raw = await asyncio.wait_for(client.recv(), timeout=0.05)
                            latest = json_mod.loads(raw)
                        except asyncio.TimeoutError:
                            break
                self.assertGreater(latest["joints"].get("arm_shoulder_pan", 0.0), 0.05)
                self.assertEqual(latest["backend"], "sim")
                self.assertTrue(latest.get("no_camera"))
            finally:
                await server.stop()

        asyncio.run(run_test())


@unittest.skipUnless(
    _has_websockets(), "websockets not installed — [serve] or [sim-serve] extra"
)
class TestRos2ProtocolWithoutGazebo(unittest.TestCase):
    """Fake rosbridge on localhost. Does not open Gazebo or a real robot."""

    def test_connect_session_over_fake_rosbridge(self):
        import json as json_mod

        import websockets

        received: list[dict[str, Any]] = []
        outgoing: asyncio.Queue[str | None] = asyncio.Queue()

        async def mock_rosbridge(ws: Any) -> None:
            async def pump_in() -> None:
                async for raw in ws:
                    msg = json_mod.loads(raw)
                    received.append(msg)
                    if msg.get("op") == "subscribe" and msg.get("topic") == "/odom":
                        await ws.send(
                            json_mod.dumps(
                                {
                                    "op": "publish",
                                    "topic": "/odom",
                                    "msg": {
                                        "pose": {
                                            "pose": {
                                                "position": {
                                                    "x": 1.25,
                                                    "y": -0.5,
                                                    "z": 0.0,
                                                },
                                                "orientation": {
                                                    "x": 0.0,
                                                    "y": 0.0,
                                                    "z": 0.0,
                                                    "w": 1.0,
                                                },
                                            }
                                        }
                                    },
                                }
                            )
                        )
                    elif (
                        msg.get("op") == "subscribe"
                        and msg.get("topic") == "/joint_states"
                    ):
                        await ws.send(
                            json_mod.dumps(
                                {
                                    "op": "publish",
                                    "topic": "/joint_states",
                                    "msg": {
                                        "name": ["arm_shoulder_pan"],
                                        "position": [0.33],
                                    },
                                }
                            )
                        )

            async def pump_out() -> None:
                while True:
                    payload = await outgoing.get()
                    if payload is None:
                        return
                    await ws.send(payload)

            await asyncio.gather(pump_in(), pump_out())

        async def wait_for(predicate: Any, timeout: float = 1.0) -> None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.02)
            raise AssertionError("timed out waiting for rosbridge traffic")

        async def run_test():
            bridge = await websockets.serve(mock_rosbridge, "127.0.0.1", 0)
            port = bridge.sockets[0].getsockname()[1]
            old_rb = os.environ.get("OHHO_ROSBRIDGE")
            os.environ["OHHO_ROSBRIDGE"] = "1"
            server = SimServe(
                backend="ros2",
                rosbridge=f"ws://127.0.0.1:{port}",
                host="127.0.0.1",
                port=0,
                rate_hz=40.0,
            )
            try:
                await server.start()
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:

                    async def latest() -> dict[str, Any]:
                        raw = await asyncio.wait_for(client.recv(), timeout=1.0)
                        msg = json_mod.loads(raw)
                        while True:
                            try:
                                raw = await asyncio.wait_for(
                                    client.recv(), timeout=0.05
                                )
                                msg = json_mod.loads(raw)
                            except asyncio.TimeoutError:
                                return msg

                    async def until_pose() -> dict[str, Any]:
                        deadline = time.monotonic() + 1.0
                        last: dict[str, Any] = {}
                        while time.monotonic() < deadline:
                            last = await latest()
                            odom = last.get("odom") or {}
                            joints = last.get("joints") or {}
                            if (
                                abs(odom.get("x", 0.0) - 1.25) < 0.02
                                and abs(joints.get("arm_shoulder_pan", 0.0) - 0.33)
                                < 0.02
                            ):
                                return last
                        raise AssertionError(f"odom/joints missing: {last}")

                    tele = await until_pose()
                    self.assertEqual(tele["backend"], "ros2")
                    self.assertTrue(tele.get("no_camera"))
                    self.assertNotIn("camera", tele)
                    self.assertAlmostEqual(tele["odom"]["y"], -0.5, places=2)

                    subscribed = {
                        msg.get("topic")
                        for msg in received
                        if msg.get("op") == "subscribe"
                    }
                    self.assertIn("/odom", subscribed)
                    self.assertIn("/joint_states", subscribed)
                    self.assertIn("/camera/front/image_raw", subscribed)
                    self.assertIn("/camera/front/image_raw/compressed", subscribed)

                    await outgoing.put(
                        json_mod.dumps(
                            {
                                "op": "publish",
                                "topic": "/camera/wrist/image_raw/compressed",
                                "msg": {"format": "jpeg", "data": "d3Jpc3Q="},
                            }
                        )
                    )
                    await outgoing.put(
                        json_mod.dumps(
                            {
                                "op": "publish",
                                "topic": "/camera/front/image_raw/compressed",
                                "msg": {"format": "jpeg", "data": ""},
                            }
                        )
                    )
                    await asyncio.sleep(0.05)
                    still_quiet = await latest()
                    self.assertTrue(still_quiet.get("no_camera"))
                    self.assertNotIn("camera", still_quiet)

                    await outgoing.put(
                        json_mod.dumps(
                            {
                                "op": "publish",
                                "topic": "/camera/front/image_raw/compressed",
                                "msg": {"format": "jpeg", "data": "aW1hZ2U="},
                            }
                        )
                    )

                    async def until_camera() -> dict[str, Any]:
                        deadline = time.monotonic() + 1.0
                        last = still_quiet
                        while time.monotonic() < deadline:
                            last = await latest()
                            if last.get("camera") == "aW1hZ2U=":
                                return last
                        raise AssertionError(f"camera frame missing: {last}")

                    framed = await until_camera()
                    self.assertNotIn("no_camera", framed)

                    mark = len(received)
                    await client.send(
                        json_mod.dumps(
                            {
                                "deadman": True,
                                "velocity": {
                                    "linearX": 0.25,
                                    "linearY": 0.05,
                                    "angularZ": 0.75,
                                },
                            }
                        )
                    )
                    await wait_for(
                        lambda: any(
                            msg.get("topic") == "/cmd_vel"
                            and abs(
                                msg.get("msg", {}).get("linear", {}).get("x", 0.0)
                                - 0.25
                            )
                            < 1e-6
                            for msg in received[mark:]
                        )
                    )

                    mark = len(received)
                    await client.send(
                        json_mod.dumps(
                            {
                                "velocity": {
                                    "linear": {"x": 0.4, "y": 0.0},
                                    "angular": {"z": 0.0},
                                }
                            }
                        )
                    )
                    await wait_for(
                        lambda: any(
                            msg.get("topic") == "/cmd_vel"
                            and msg.get("msg", {}).get("linear", {}).get("x") == 0.0
                            for msg in received[mark:]
                        )
                    )
                    await asyncio.sleep(0.05)
                    zeros = [
                        msg for msg in received[mark:] if msg.get("topic") == "/cmd_vel"
                    ]
                    self.assertEqual(zeros[-1]["msg"]["linear"]["x"], 0.0)

                    mark = len(received)
                    await client.send(
                        json_mod.dumps(
                            {"deadman": True, "joints": {"arm_shoulder_pan": 0.4}}
                        )
                    )
                    await wait_for(
                        lambda: any(
                            msg.get("topic") == "/arm/joint_commands"
                            for msg in received[mark:]
                        )
                    )
                    joint_msgs = [
                        msg
                        for msg in received
                        if msg.get("topic") == "/arm/joint_commands"
                    ]
                    self.assertEqual(
                        joint_msgs[-1]["msg"]["name"], ["arm_shoulder_pan"]
                    )
                    self.assertAlmostEqual(joint_msgs[-1]["msg"]["position"][0], 0.4)

                    mark = len(received)
                    await client.send(
                        json_mod.dumps(
                            {
                                "estop": True,
                                "deadman": True,
                                "velocity": {
                                    "linear": {"x": 0.2, "y": 0.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await wait_for(
                        lambda: any(
                            msg.get("topic") == "/emergency_stop"
                            and msg.get("msg", {}).get("data") is True
                            for msg in received[mark:]
                        )
                    )
                    await asyncio.sleep(0.05)
                    twists = [
                        msg for msg in received[mark:] if msg.get("topic") == "/cmd_vel"
                    ]
                    self.assertTrue(twists)
                    self.assertEqual(twists[-1]["msg"]["linear"]["x"], 0.0)
            finally:
                await outgoing.put(None)
                await server.stop()
                if old_rb is None:
                    os.environ.pop("OHHO_ROSBRIDGE", None)
                else:
                    os.environ["OHHO_ROSBRIDGE"] = old_rb
                bridge.close()
                await bridge.wait_closed()

        asyncio.run(run_test())


class TestDefaultPort(unittest.TestCase):
    """sim-serve must not take Foxglove's port unless the caller asks for it."""

    def test_default_is_8790_not_foxglove(self):
        import inspect

        self.assertEqual(DEFAULT_PORT, 8790)
        self.assertNotEqual(DEFAULT_PORT, 8765)
        self.assertEqual(SimServe().port, 8790)
        self.assertEqual(inspect.signature(serve_sim).parameters["port"].default, 8790)
        self.assertEqual(SimServe(port=8765).port, 8765)

    def test_cli_default_and_help(self):
        args = build_parser().parse_args(["sim-serve"])
        self.assertEqual(args.port, 8790)
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit):
            build_parser().parse_args(["sim-serve", "--help"])
        help_text = buf.getvalue()
        self.assertIn("8790", help_text)
        self.assertIn("8765", help_text)
        self.assertIn("Foxglove", help_text)


if __name__ == "__main__":
    unittest.main()
