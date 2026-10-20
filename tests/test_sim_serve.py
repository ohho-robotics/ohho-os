from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from typing import Any
import unittest
from unittest.mock import patch

from ohho import capabilities as caps
from ohho.cli import main
from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass
from ohho.safety import ARM_ENV, ARM_PHRASE
from ohho.sim_serve import (
    SILENCE_S,
    SimServe,
    SimServeUnavailable,
    decide_mode_request,
    parse_velocity,
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
        self.assertNotIn("isaac", sys.modules)
        self.assertNotIn("isaacsim", sys.modules)

    def test_mujoco_not_built(self):
        # A fresh interpreter: the humanoid-mujoco tests import MuJoCo in this
        # process, and that must not hide a sim-serve import.
        script = (
            "import sys\n"
            "from ohho.cli import main\n"
            "code = main(['sim-serve', '--backend', 'mujoco'])\n"
            "imported = 'mujoco' in sys.modules\n"
            "print('IMPORTED', imported)\n"
            "raise SystemExit(0 if code != 0 and not imported else 1)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + proc.stderr
        self.assertIn("not built", output)
        self.assertIn("IMPORTED False", output)
        self.assertEqual(proc.returncode, 0)


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
                    for _ in range(5):
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
                    for _ in range(5):
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
        # Must drive and complete all checks in under 2 seconds
        self.assertLess(
            elapsed,
            2.0,
            f"Test took {elapsed:.2f}s, expected under 2 seconds",
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


def _posture_spec(
    modes: tuple[str, ...],
    robot_id: str = "walker",
    *,
    posture: bool = True,
) -> RobotSpec:
    capabilities: tuple[str, ...] = (caps.BASE_DRIVE,)
    params: dict[str, object] = {}
    if posture:
        capabilities = (caps.BASE_DRIVE, caps.HUMANOID_POSTURE)
        params[caps.HUMANOID_POSTURE] = {"modes": list(modes)}
    return RobotSpec(
        id=robot_id,
        name="Walker",
        category="humanoid" if posture else "mobile_base",
        capabilities=capabilities,
        capability_params=params,
        robot_class=RobotClass.HUMANOID if posture else RobotClass.MOBILE_BASE,
        source="test",
    )


class _PostureRobot:
    """In-memory humanoid. Records posture actions and body velocity."""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.transport = self
        self.calls: list[str] = []
        self.drives: list[tuple[float, float, float]] = []
        self.protocol = "simulated"

    def has(self, capability: str) -> bool:
        return self.spec.has(capability)

    def send_humanoid(self, action: str, **fields: object) -> None:
        del fields
        self.calls.append(action)

    def drive(self, vx: float = 0.0, vy: float = 0.0, w: float = 0.0) -> None:
        self.drives.append((float(vx), float(vy), float(w)))

    def emergency_stop(self) -> None:
        self.calls.append("estop")

    def release_stop(self) -> None:
        self.calls.append("release")


def _modes_message(
    request: str,
    *,
    deadman: bool = True,
    confirm: str | None = None,
    vx: float = 0.2,
) -> dict[str, Any]:
    modes: dict[str, Any] = {"request": request}
    if confirm is not None:
        modes["confirm"] = confirm
    return {
        "velocity": {"linear": {"x": vx, "y": 0.0}, "angular": {"z": 0.0}},
        "deadman": deadman,
        "estop": False,
        "modes": modes,
    }


class TestHumanoidModes(unittest.TestCase):
    """Capability-keyed posture requests on the sim-serve message."""

    def _server(self, spec: RobotSpec, **kwargs: Any) -> tuple[SimServe, _PostureRobot]:
        robot = _PostureRobot(spec)
        server = SimServe(robot=spec.id, **kwargs)
        server._bot = robot
        return server, robot

    def test_section_6_1_message_on_sim_does_not_need_the_phrase(self) -> None:
        spec = _posture_spec(
            ("damping", "prepare", "walk", "custom", "get_up", "lie_down")
        )
        server, robot = self._server(spec, backend="sim")
        message = {
            "velocity": {"linear": {"x": 0.2, "y": 0.0}, "angular": {"z": 0.0}},
            "deadman": True,
            "estop": False,
            "modes": {"request": "walk_mode", "confirm": ARM_PHRASE},
        }
        server._on_client_message(message)
        self.assertEqual(robot.calls, ["release", "walk_mode"])
        self.assertEqual(server._posture, "walk_mode")
        self.assertEqual(server._mode_result, {"accepted": True, "action": "walk_mode"})
        self.assertEqual(robot.drives[-1], (0.2, 0.0, 0.0))
        decision = decide_mode_request(
            spec,
            "walk_mode",
            backend="sim",
            uri="sim://",
            confirm=None,
            env={},
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "walk_mode")

    def test_request_must_be_a_declared_posture_mode(self) -> None:
        spec = _posture_spec(("prepare",))
        server, robot = self._server(spec, backend="sim")
        server._on_client_message(_modes_message("walk_mode"))
        self.assertEqual(server._mode_result["reason"], "undeclared")
        self.assertEqual(server._posture, "")
        self.assertNotIn("walk_mode", robot.calls)
        server._on_client_message(_modes_message("dance"))
        self.assertEqual(server._mode_result["reason"], "undeclared")
        server._on_client_message(_modes_message("prepare"))
        self.assertEqual(server._posture, "prepare")
        self.assertIn("prepare", robot.calls)

    def test_mode_request_is_keyed_by_capability_not_robot_id(self) -> None:
        lacking = _posture_spec((), robot_id="alpha", posture=False)
        having = _posture_spec(("prepare", "walk"), robot_id="beta")
        refused, quiet = self._server(lacking, backend="sim")
        refused.robot_id = "beta"
        refused._on_client_message(_modes_message("prepare"))
        self.assertEqual(refused._mode_result["reason"], "capability")
        self.assertEqual(quiet.calls, ["release"])
        accepted, moved = self._server(having, backend="sim")
        accepted.robot_id = "alpha"
        accepted._on_client_message(_modes_message("walk"))
        self.assertEqual(accepted._posture, "walk_mode")
        self.assertIn("walk_mode", moved.calls)

    def test_deadman_false_zeros_velocity_and_leaves_the_mode(self) -> None:
        spec = _posture_spec(("prepare", "walk", "damping"))
        server, robot = self._server(spec, backend="sim")
        server._on_client_message(_modes_message("walk_mode"))
        self.assertEqual(server._posture, "walk_mode")
        server._on_client_message(_modes_message("damp", deadman=False, vx=0.4))
        self.assertEqual(robot.drives[-1], (0.0, 0.0, 0.0))
        self.assertEqual(server._posture, "walk_mode")
        self.assertNotIn("damp", robot.calls)
        self.assertEqual(server._mode_result["reason"], "deadman")

    def test_silence_zeros_velocity_and_leaves_the_mode(self) -> None:
        spec = _posture_spec(("prepare", "walk", "damping"))
        server, robot = self._server(spec, backend="sim")
        server._on_client_message(_modes_message("prepare"))
        self.assertEqual(server._posture, "prepare")
        server._deadman = True
        server._target_vx = 0.2
        base = 1_000.0
        server._last_msg_time = base
        self.assertEqual(server.commanded_velocity(base + SILENCE_S), (0.2, 0.0, 0.0))
        stopped = server.commanded_velocity(base + SILENCE_S + 1e-4)
        self.assertEqual(stopped, (0.0, 0.0, 0.0))
        asyncio.run(server._apply_velocity(*stopped))
        self.assertEqual(robot.drives[-1], (0.0, 0.0, 0.0))
        self.assertEqual(server._posture, "prepare")
        self.assertNotIn("damp", robot.calls)
        self.assertNotIn("walk_mode", robot.calls)

    def test_robot_backend_without_the_phrase_is_refused(self) -> None:
        spec = _posture_spec(("prepare", "walk", "damping"))
        server, robot = self._server(
            spec, backend="robot", transport="serial://127.0.0.1"
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotEqual(os.environ.get(ARM_ENV), "1")
            server._on_client_message(_modes_message("walk_mode"))
            self.assertFalse(server._mode_result["accepted"])
            self.assertEqual(server._mode_result["reason"], "confirm")
            self.assertEqual(server._posture, "")
            self.assertNotIn("walk_mode", robot.calls)
            self.assertNotIn(ARM_ENV, os.environ)

    def test_robot_backend_phrase_without_arm_env_is_refused(self) -> None:
        spec = _posture_spec(("prepare", "walk"))
        server, robot = self._server(
            spec, backend="robot", transport="serial://127.0.0.1"
        )
        with patch.dict(os.environ, {ARM_ENV: "0"}, clear=True):
            server._on_client_message(_modes_message("prepare", confirm=ARM_PHRASE))
            self.assertEqual(server._mode_result["reason"], "arm_env")
            self.assertEqual(server._posture, "")
            self.assertNotIn("prepare", robot.calls)
            self.assertEqual(os.environ.get(ARM_ENV), "0")

    def test_robot_backend_sim_uri_does_not_need_the_phrase(self) -> None:
        spec = _posture_spec(("walk",))
        server, robot = self._server(spec, backend="robot", transport="sim://local")
        with patch.dict(os.environ, {}, clear=True):
            server._on_client_message(_modes_message("walk"))
        self.assertEqual(server._posture, "walk_mode")
        self.assertIn("walk_mode", robot.calls)

    def test_robot_backend_cli_requires_a_transport(self) -> None:
        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            code = main(["sim-serve", "--backend", "robot"])
        self.assertEqual(code, 2)
        self.assertIn("transport", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
