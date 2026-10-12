"""Tests for the sim/ros2 demo drive (pytest)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ohho import Robot
from ohho.demo_drive import (
    default_runtime_for_transport,
    run_drive,
    wait_for_odom,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_NODE = REPO_ROOT / "examples" / "fake_ros2_robot_node.py"


def _rclpy_available() -> bool:
    return importlib.util.find_spec("rclpy") is not None


def test_default_runtime_for_transport():
    assert default_runtime_for_transport("sim://") == "native"
    assert default_runtime_for_transport("ros2://") == "ros2"


def test_run_drive_sim_moves_forward():
    bot = Robot.connect("omnibot", transport="sim://", runtime="native")
    try:
        assert wait_for_odom(bot, timeout_s=2.0)
        result = run_drive(bot, seconds=1.5, vx=0.2, rate_hz=20.0)
        assert result.dx > 0.05
        assert result.displacement > 0.05
    finally:
        bot.disconnect()


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not installed")
def test_demo_drive_ros2_fake_node_odom_delta():
    """Integration: fake cmd_vel integrator + ohho ros2 transport."""
    env = {**os.environ, "ROS_DOMAIN_ID": "199"}
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
        assert demo.returncode == 0, demo.stderr + demo.stdout
        assert "odom delta" in demo.stdout
    finally:
        proc.terminate()
        proc.wait(timeout=10)
