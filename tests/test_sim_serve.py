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

from ohho.cli import main
from ohho.sim_serve import SimServe, SimServeUnavailable, parse_velocity


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
            "code = main(['sim-serve', '--backend', 'mujoco', '--viewer'])\n"
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


if __name__ == "__main__":
    unittest.main()
