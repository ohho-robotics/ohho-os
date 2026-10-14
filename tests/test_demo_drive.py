"""Tests for the sim/ros2 demo drive."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from ohho import Robot
from ohho.adapters.ros2 import Ros2Transport
from ohho.demo_drive import (
    _has_live_odom,
    default_runtime_for_transport,
    run_drive,
    wait_for_odom,
)
from ohho.registry import get_spec
from ohho.runtime import NativeRuntime
from ohho.safety import SafetyGate

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_NODE = REPO_ROOT / "examples" / "fake_ros2_robot_node.py"


def _rclpy_available() -> bool:
    return importlib.util.find_spec("rclpy") is not None


class TestDemoDrive(unittest.TestCase):
    def test_default_runtime_for_transport(self) -> None:
        cases = [
            ("sim://", "native"),
            ("ros2://", "ros2"),
        ]
        for transport, expected in cases:
            with self.subTest(transport=transport):
                self.assertEqual(default_runtime_for_transport(transport), expected)

    def test_has_live_odom_through_safety_gate(self) -> None:
        from test_ros2 import FakeMsgFactory, FakeOdometry, FakeRos2Node

        spec = get_spec("omnibot")
        node = FakeRos2Node("test")
        inner = Ros2Transport(spec, "", node=node, msg_factory=FakeMsgFactory)
        inner.connect()
        gate = SafetyGate(inner, spec, auto_tick=False)
        gate.connect()
        bot = Robot(spec, gate, NativeRuntime())
        try:
            self.assertFalse(_has_live_odom(bot))
            node.inject_message("/odom", FakeOdometry(x=0.0, y=0.0, theta=0.0))
            self.assertTrue(_has_live_odom(bot))
            self.assertTrue(gate.odom_received)
        finally:
            bot.disconnect()

    def test_run_drive_sim_moves_forward(self) -> None:
        bot = Robot.connect("omnibot", transport="sim://", runtime="native")
        try:
            self.assertTrue(wait_for_odom(bot, timeout_s=2.0))
            result = run_drive(bot, seconds=1.5, vx=0.2, rate_hz=20.0)
            self.assertGreater(result.dx, 0.05)
            self.assertGreater(result.displacement, 0.05)
        finally:
            bot.disconnect()

    @unittest.skipUnless(_rclpy_available(), "rclpy not installed")
    def test_demo_drive_ros2_fake_node_odom_delta(self) -> None:
        """Integration: fake cmd_vel integrator + ohho ros2 transport."""
        env = {
            **os.environ,
            "ROS_DOMAIN_ID": "199",
            "ROS_LOCALHOST_ONLY": "1",
        }
        proc = subprocess.Popen(
            [sys.executable, str(FAKE_NODE)],
            cwd=str(REPO_ROOT),
            env=env,
        )
        try:
            time.sleep(1.5)
            demo = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples" / "demo_drive.py"),
                    "--transport",
                    "ros2://",
                    "--seconds",
                    "2",
                    "--vx",
                    "0.2",
                    "--min-displacement",
                    "0.15",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                demo.returncode,
                0,
                demo.stderr + demo.stdout,
            )
            self.assertIn("odom delta", demo.stdout)
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
