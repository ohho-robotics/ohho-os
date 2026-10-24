"""Humanoid class scenario (OHH-334).

The scripted checks run on a planar stand-in so the suite does not download
meshes. The MuJoCo check needs mujoco, torch and the pinned assets. It skips
when those are absent. In GitHub Actions a missing pin is a failure on the
job that installs them.
"""

from __future__ import annotations

import importlib.util
import math
import os
import unittest
import urllib.error

from ohho import capabilities as caps
from ohho.humanoid import FallState, HumanoidMode
from ohho.humanoid_telemetry import apply_humanoid_telemetry, build_humanoid_telemetry
from ohho.registry import RobotSpec, UnknownRobot, get_spec
from ohho.robot_spec import RobotClass
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.testing.conformance import format_result, run_robot
from ohho.testing.conformance.runner import run_humanoid_walk
from ohho.transport import BaseTransport

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None


class _PlanarHumanoid(BaseTransport):
    """Integrates a body velocity. A push reports a fallen base."""

    protocol = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self._state = ConnectionState.IDLE
        self._mode = HumanoidMode.OFF
        self._fall = FallState.UPRIGHT
        self._vel = Velocity()
        self._kp = False
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._push_left = 0.0
        self._stamp = 0.0
        self.transport_estop = False

    def connect(self) -> TransportStatus:
        self._state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol, state=self._state, label="planar"
        )

    def send_velocity(self, vel: Velocity) -> None:
        self._vel = Velocity() if self.transport_estop else vel

    def send_joint_command(self, name: str, position: float) -> None:
        del name, position

    def emergency_stop(self) -> None:
        self.transport_estop = True
        self._vel = Velocity()

    def release_stop(self) -> None:
        self.transport_estop = False

    def send_humanoid(self, action: str, *args: object, **fields: object) -> None:
        del args, fields
        name = action.strip().lower()
        if name == "prepare":
            self._mode = HumanoidMode.PREPARE
            self._kp = True
        elif name == "walk_mode":
            self._mode = HumanoidMode.WALK
            self._kp = True
        elif name == "damp":
            self._mode = HumanoidMode.DAMPING
            self._kp = False
            self._vel = Velocity()
        else:
            raise ValueError(action)

    def inject_push(self, force: object, duration_s: float = 0.0) -> None:
        del force
        self._push_left = 1e-3 if duration_s == 0 else float(duration_s)

    @property
    def command(self) -> Velocity:
        return self._vel

    @property
    def base_position(self) -> tuple[float, float, float]:
        return (self.x, self.y, 1.0)

    def effective_kp(self) -> tuple[float, ...]:
        if self._kp:
            return (20.0,)
        return (0.0,)

    def read(self) -> Telemetry:
        block = build_humanoid_telemetry(
            mode=self._mode,
            vendor_mode="",
            fall_state=self._fall,
            base_rpy=(0.0, 0.0, self.theta),
            base_ang_vel=(0.0, 0.0, self._vel.angular_z),
            joints={"body": {"q": [0.0], "dq": [0.0]}},
            odom={"x": self.x, "y": self.y, "yaw": self.theta},
            battery={"fraction": 1.0, "voltage": 0.0},
        )
        sample = Telemetry(battery=1.0, custom={}, timestamp=self._stamp)
        return apply_humanoid_telemetry(sample, block)

    def step(self, dt: float | None = None) -> Telemetry:
        duration = 0.05 if dt is None else float(dt)
        if self._mode is HumanoidMode.WALK and not self.transport_estop:
            yaw = self.theta
            self.x += (
                self._vel.linear_x * math.cos(yaw) - self._vel.linear_y * math.sin(yaw)
            ) * duration
            self.y += (
                self._vel.linear_x * math.sin(yaw) + self._vel.linear_y * math.cos(yaw)
            ) * duration
            self.theta += self._vel.angular_z * duration
        if self._push_left > 0.0:
            self._push_left = max(0.0, self._push_left - duration)
            self._fall = FallState.FALLEN
        self._stamp += duration
        sample = self.read()
        self._emit_telemetry(sample)
        return sample


def _humanoid_spec() -> RobotSpec:
    return RobotSpec(
        id="humanoid-fixture",
        name="Fixture Humanoid",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.HUMANOID_LOCOMOTION,
            caps.HUMANOID_POSTURE,
        ),
        adapter="sim",
        sim_backend="humanoid-mujoco",
        max_lin=0.5,
        max_ang=1.0,
    )


class TestHeightFloor(unittest.TestCase):
    def test_a_tall_spawn_keeps_the_class_floor(self):
        from ohho.testing.conformance.runner import _MIN_HEIGHT_M, _height_floor

        self.assertEqual(_height_floor(1.05), _MIN_HEIGHT_M)
        self.assertEqual(_height_floor(0.7), _MIN_HEIGHT_M)
        self.assertEqual(_height_floor(None), _MIN_HEIGHT_M)
        self.assertEqual(_height_floor(0.0), _MIN_HEIGHT_M)

    def test_a_shorter_spawn_uses_seventy_percent(self):
        from ohho.testing.conformance.runner import _height_floor

        self.assertAlmostEqual(_height_floor(0.6), 0.42)


class TestHumanoidWalkScript(unittest.TestCase):
    def test_prepare_walk_turn_stop_estop_and_push(self):
        transport = _PlanarHumanoid()
        checks = {
            check.name: check
            for check in run_humanoid_walk(transport, _humanoid_spec())
        }
        for name in (
            "telemetry_schema",
            "prepare",
            "walk_1m",
            "turn_90",
            "stop",
            "estop_holds_prepare",
            "push_damp",
        ):
            self.assertEqual(checks[name].status, "pass", checks[name].reason)
        self.assertFalse(transport.transport_estop)
        self.assertEqual(transport._mode, HumanoidMode.DAMPING)
        self.assertEqual(transport._fall, FallState.FALLEN)
        self.assertGreaterEqual(math.hypot(transport.x, transport.y), 1.0)
        self.assertGreaterEqual(abs(transport.theta), math.pi / 2 - 0.2)


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestHumanoidMuJoCoConformance(unittest.TestCase):
    def test_registered_humanoid_passes(self):
        try:
            get_spec("booster-t2")
        except UnknownRobot:
            self.skipTest("humanoid plugin is not installed")
        from ohho_booster.assets import ensure_assets

        try:
            ensure_assets()
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise
            self.skipTest(f"pinned assets absent: {exc}")
        result = run_robot("booster-t2")
        self.assertEqual(result.status, "pass", format_result(result))
        names = {check.name: check.status for check in result.checks}
        self.assertEqual(names["prepare"], "pass")
        self.assertEqual(names["walk_1m"], "pass")
        self.assertEqual(names["turn_90"], "pass")
        self.assertEqual(names["stop"], "pass")
        self.assertEqual(names["estop_holds_prepare"], "pass")
        self.assertEqual(names["push_damp"], "pass")

    def test_booster_k1_passes(self):
        try:
            get_spec("booster-k1")
        except UnknownRobot:
            self.skipTest("humanoid plugin is not installed")
        from ohho_booster.assets import ensure_assets

        try:
            ensure_assets(robot_id="booster-k1")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                raise
            self.skipTest(f"pinned assets absent: {exc}")
        result = run_robot("booster-k1")
        self.assertEqual(result.status, "pass", format_result(result))
        names = {check.name: check.status for check in result.checks}
        self.assertEqual(names["prepare"], "pass")
        self.assertEqual(names["walk_1m"], "pass")
        self.assertEqual(names["turn_90"], "pass")
        self.assertEqual(names["stop"], "pass")
        self.assertEqual(names["estop_holds_prepare"], "pass")
        self.assertEqual(names["push_damp"], "pass")


if __name__ == "__main__":
    unittest.main()
