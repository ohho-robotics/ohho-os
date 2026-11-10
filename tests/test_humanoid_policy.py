"""HumanoidPolicy (OHH-328). Fake clock, fake transports, no hardware."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from ohho.humanoid import HumanoidMode
from ohho.humanoid_policy import (
    HARDWARE_CAPS,
    HUMANOID_MCP_TOOLS,
    MAX_JOINT_STEP_RAD,
    SIM_CAPS,
    TILT_HOLD_S,
    TILT_LIMIT_RAD,
    HumanoidCaps,
    HumanoidPolicy,
    cap_looser_reasons,
    class_profile,
    explicit_sim_target,
    humanoid_denied,
    humanoid_mcp_allowed,
    intersect_caps,
    mcp_command,
)
from ohho.registry import RobotSpec
from ohho.robot_spec import HumanoidLimits, RobotClass
from ohho.safety import (
    ARM_ENV,
    ARM_PHRASE,
    AuditLog,
    SafetyGate,
    SafetyProfile,
    normalize_action,
)
from ohho.safety_policy import SafetyPolicy, SafetyPolicyError, arming_uri, load_policy
from ohho.schema import (
    ConnectionState,
    JointReading,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ohho.transport import BaseTransport

WALL = 1_760_000_000.0


def setUpModule() -> None:
    logging.getLogger("ohho.safety").addHandler(logging.NullHandler())


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Link(BaseTransport):
    def __init__(
        self,
        protocol: str = "simulated",
        host: str = "127.0.0.1",
        sim_target: bool = False,
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.host = host
        self.sim_target = sim_target
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self.state = ConnectionState.CONNECTED

    def _rec(self, name: str, *args: object) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, *args))

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="fake")

    def read(self) -> Telemetry:
        return Telemetry()

    def send_velocity(self, vel: Velocity) -> None:
        self._rec("vel", (vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        self._rec("joint", name, position)

    def send_humanoid(self, action: str, *args: object) -> None:
        self._rec("humanoid", action, args, {})

    def emergency_stop(self) -> None:
        self._rec("estop")

    def release_stop(self) -> None:
        self._rec("release")

    def emit(self, telemetry: Telemetry) -> None:
        self._emit_telemetry(telemetry)


class KeywordLink(Link):
    """Accepts only keyword joint targets."""

    def send_humanoid(self, action: str, **fields: object) -> None:
        self._rec("humanoid", action, (), fields)


def _spec(**overrides: object) -> RobotSpec:
    body: dict[str, object] = {
        "id": "walker",
        "name": "Walker",
        "category": "humanoid",
        "capabilities": ("base.drive", "humanoid.posture", "whole_body.joint"),
        "robot_class": RobotClass.HUMANOID,
        "source": "walker-plugin",
        "joint_groups": {"left_arm": ("j0", "j1")},
        "max_lin": 0.5,
        "max_ang": 0.6,
    }
    body.update(overrides)
    return RobotSpec(**body)  # type: ignore[arg-type]


def _tele(
    mode: str = "WALK",
    fall: str = "upright",
    roll: float = 0.0,
    pitch: float = 0.0,
    timestamp: float = 100.0,
    joints: list[JointReading] | None = None,
    *,
    error_code: object = 0,
    include_error: bool = True,
    rpy: object = None,
    humanoid: bool = True,
) -> Telemetry:
    custom: dict[str, object] = {"roll": roll, "pitch": pitch}
    if include_error:
        custom["error_code"] = error_code
    if humanoid:
        block: dict[str, object] = {
            "mode": mode,
            "fall_state": fall,
            "vendor_mode": "logged-only",
        }
        if rpy is not None:
            block["base_rpy"] = rpy
        elif mode or fall:
            block["base_rpy"] = [roll, pitch, 0.0]
        custom["humanoid"] = block
    return Telemetry(timestamp=timestamp, joints=list(joints or []), custom=custom)


def _joints_at(
    *positions: float, timestamp: float = 100.0, mode: str = "CUSTOM"
) -> Telemetry:
    readings = [
        JointReading(name=f"j{index}", position=position)
        for index, position in enumerate(positions)
    ]
    return _tele(mode=mode, timestamp=timestamp, joints=readings)


class PolicyCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-humanoid-policy-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = Clock()
        self.prompts: list[str] = []
        self.answer: object = ARM_PHRASE

    def _input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return str(self.answer)

    def events(self, name: str | None = None) -> list[dict]:
        out: list[dict] = []
        for path in sorted(Path(self.tmp).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if name is None or record["event"] == name:
                    out.append(record)
        return out

    def make(
        self,
        *,
        protocol: str = "simulated",
        host: str = "127.0.0.1",
        sim_target: bool = False,
        arm: bool = True,
        profile: SafetyProfile | None = None,
        spec: RobotSpec | None = None,
        environ: dict[str, str] | None = None,
        tty: bool = False,
        link: Link | None = None,
    ) -> tuple[SafetyGate, Link]:
        inner = link or Link(protocol, host, sim_target)
        gate = SafetyGate(
            inner,
            spec if spec is not None else _spec(),
            profile=profile,
            clock=self.clock,
            environ={} if environ is None else environ,
            isatty=lambda: tty,
            input_fn=self._input,
            audit=AuditLog(self.tmp, wall_clock=lambda: WALL),
            auto_tick=False,
        )
        gate.connect()
        inner.emit(_tele(timestamp=self.clock()))
        if arm:
            self.assertTrue(gate.arm(), self.events("arm_refused"))
        inner.calls.clear()
        return gate, inner


class TestWatchdog(PolicyCase):
    def test_watchdog_zeros_velocity_and_does_not_damp(self) -> None:
        gate, inner = self.make()
        self.clock.advance(0.05)
        gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.clock.advance(0.31)
        inner.emit(_tele(mode="WALK", timestamp=self.clock()))
        gate.tick()
        self.assertFalse(gate.latched)
        self.assertEqual(inner.calls[-1][0], "vel")
        self.assertEqual(inner.calls[-1][1], (0.0, 0.0, 0.0))
        self.assertFalse(
            any(call[1] == "damp" for call in inner.calls if call[0] == "humanoid")
        )
        self.assertEqual(self.events("watchdog_trip")[-1]["kind"], "command")

    def test_policy_watchdog_swallows_a_send_failure(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        link = Link()
        link.fail.add("vel")
        with self.assertLogs("ohho.safety", level="ERROR"):
            policy.on_watchdog(link, _tele())
        self.assertEqual(link.calls, [])


class TestEstop(PolicyCase):
    def test_estop_while_walking_holds_prepare(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(mode="WALK", fall="upright"))
        gate.emergency_stop()
        self.assertTrue(gate.latched)
        humanoid = [call for call in inner.calls if call[0] == "humanoid"]
        self.assertEqual(humanoid, [("humanoid", "prepare", (), {})])
        self.assertNotIn("estop", [call[0] for call in inner.calls])
        self.assertEqual(self.events("estop")[-1]["reason"], "operator")
        self.assertEqual(self.events("damp"), [])

    def test_estop_in_prepare_holds_prepare(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(mode="PREPARE"))
        gate.emergency_stop()
        self.assertEqual(
            [call for call in inner.calls if call[0] == "humanoid"],
            [("humanoid", "prepare", (), {})],
        )

    def test_estop_in_custom_damps(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(mode="CUSTOM", fall="upright"))
        gate.emergency_stop()
        self.assertIn(("humanoid", "damp", (), {}), inner.calls)
        self.assertEqual(self.events("damp")[-1]["why"], "custom")

    def test_estop_when_fallen_damps(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        link = Link()
        action = policy.on_estop(link, _tele(mode="WALK", fall="fallen"))
        self.assertEqual(action.damp_why, "fallen")
        self.assertIn(("humanoid", "damp", (), {}), link.calls)
        self.assertNotIn("zero_torque", str(link.calls))
        self.assertNotIn("drag", str(link.calls))

    def test_estop_when_fall_state_is_unknown_does_not_damp(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        link = Link()
        action = policy.on_estop(link, _tele(mode="WALK", fall="unknown"))
        self.assertEqual(action.damp_why, "")
        self.assertEqual(
            [call for call in link.calls if call[0] == "humanoid"],
            [],
        )
        self.assertEqual(link.calls[0][1], (0.0, 0.0, 0.0))

    def test_estop_without_a_humanoid_hook_still_stops(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())

        class Bare(Link):
            send_humanoid = None  # type: ignore[assignment]

        link = Bare()
        with self.assertLogs("ohho.safety", level="ERROR"):
            action = policy.on_estop(link, _tele(mode="CUSTOM"))
        self.assertEqual(action.damp_why, "")
        self.assertEqual(link.calls[0][1], (0.0, 0.0, 0.0))

    def test_estop_records_a_velocity_failure(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        link = Link()
        link.fail.add("vel")
        with self.assertLogs("ohho.safety", level="ERROR"):
            action = policy.on_estop(link, _tele(mode="WALK"))
        self.assertTrue(action.inner_error)
        self.assertIn(("humanoid", "prepare", (), {}), link.calls)

    def test_damp_send_failure_is_not_reported_as_sent(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        link = Link()
        link.fail.add("humanoid")
        with self.assertLogs("ohho.safety", level="ERROR"):
            action = policy.on_estop(link, _tele(mode="FALLEN", fall="fallen"))
        self.assertEqual(action.damp_why, "")
        self.assertFalse(policy._damped)


class TestFallLatch(PolicyCase):
    def test_vendor_fall_states_and_unknown_latch(self) -> None:
        for fall in ("falling", "fallen", "recovering", "unknown"):
            with self.subTest(fall=fall):
                gate, inner = self.make()
                inner.emit(_tele(fall=fall, timestamp=self.clock()))
                self.assertTrue(gate.latched)
                self.assertEqual(gate.latch_reason, "fall")
                record = self.events("estop")[-1]
                self.assertEqual(record["fall_state"], fall)
                self.assertEqual(record["plugin"], "walker-plugin")
                self.assertIn("mode", record)

    def test_missing_fall_state_latches(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(humanoid=False, timestamp=self.clock()))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "fall")
        self.assertEqual(self.events("estop")[-1]["fall_state"], "unknown")

    def test_upright_does_not_latch(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(fall="upright", roll=0.1, pitch=-0.1))
        self.assertFalse(gate.latched)

    def test_tilt_latches_only_after_100ms(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(roll=0.6, timestamp=1.0))
        self.assertFalse(gate.latched)
        inner.emit(_tele(roll=0.6, timestamp=1.0 + TILT_HOLD_S - 0.01))
        self.assertFalse(gate.latched)
        inner.emit(_tele(roll=0.51, timestamp=1.0 + TILT_HOLD_S))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "tilt")
        self.assertEqual(self.events("estop")[-1]["plugin"], "walker-plugin")
        self.assertIn(("humanoid", "prepare", (), {}), inner.calls)

    def test_tilt_window_resets_when_the_robot_recovers(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(roll=0.7, timestamp=1.0))
        inner.emit(_tele(roll=0.1, timestamp=1.05))
        inner.emit(_tele(roll=0.7, timestamp=1.2))
        self.assertFalse(gate.latched)

    def test_tilt_then_fallen_damps_once(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(roll=0.8, timestamp=2.0))
        inner.emit(_tele(roll=0.8, timestamp=2.2))
        self.assertEqual(gate.latch_reason, "tilt")
        inner.calls.clear()
        inner.emit(_tele(mode="WALK", fall="fallen", timestamp=3.0))
        self.assertEqual(
            [call for call in inner.calls if call[0] == "humanoid"],
            [("humanoid", "damp", (), {})],
        )
        self.assertEqual(self.events("damp")[-1]["why"], "fallen")
        inner.calls.clear()
        inner.emit(_tele(mode="FALLEN", fall="fallen", timestamp=4.0))
        self.assertEqual(inner.calls, [])

    def test_fault_and_bad_orientation(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        fault = policy.fall_or_tilt(_tele(error_code=4))
        self.assertIsNotNone(fault)
        assert fault is not None
        self.assertEqual(fault[0], "fault")
        self.assertEqual(fault[1]["error_code"], 4)
        bad = policy.fall_or_tilt(_tele(error_code="nope"))
        assert bad is not None
        self.assertEqual(bad[0], "fault")
        tilted = policy.fall_or_tilt(_tele(rpy=[float("nan"), 0.0], error_code=0))
        assert tilted is not None
        self.assertEqual(tilted[0], "tilt")
        missing_time = policy.fall_or_tilt(
            Telemetry(
                timestamp=float("nan"),
                custom={
                    "error_code": 0,
                    "humanoid": {
                        "mode": "WALK",
                        "fall_state": "upright",
                        "base_rpy": [0.9, 0.0, 0.0],
                    },
                },
            )
        )
        self.assertIsNone(missing_time)
        upright = Telemetry(
            timestamp=1.0,
            custom={"humanoid": {"mode": "WALK", "fall_state": "upright"}},
        )
        self.assertIsNone(policy.fall_or_tilt(upright))

    def test_bad_rpy_shape_and_imu_fallback(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        policy.arming("simulated://", {})
        shaped = policy.fall_or_tilt(_tele(rpy="nope", include_error=False))
        assert shaped is not None
        self.assertEqual(shaped[0], "tilt")
        imu = Telemetry(
            timestamp=1.0,
            custom={
                "imu_rpy": [0.1, 0.1, 0.0],
                "humanoid": {"mode": "WALK", "fall_state": "upright"},
            },
        )
        self.assertIsNone(policy.fall_or_tilt(imu))
        nan_imu = Telemetry(
            timestamp=1.0,
            custom={
                "roll": float("nan"),
                "pitch": 0.0,
                "humanoid": {"mode": "WALK", "fall_state": "upright"},
            },
        )
        bad = policy.fall_or_tilt(nan_imu)
        assert bad is not None
        self.assertEqual(bad[0], "tilt")


class TestModeAudit(PolicyCase):
    def test_mode_transition_names_the_plugin(self) -> None:
        gate, inner = self.make()
        inner.emit(_tele(mode="PREPARE", timestamp=self.clock()))
        record = self.events("mode_transition")[-1]
        self.assertEqual(record["mode_from"], "WALK")
        self.assertEqual(record["mode_to"], "PREPARE")
        self.assertEqual(record["plugin"], "walker-plugin")
        self.assertEqual(record["policy"], "humanoid")

    def test_note_state_without_an_audit_fn_still_tracks_mode(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        policy.note_state(Link(), None, latched=False)
        policy.note_state(Link(), _tele(mode="WALK"), latched=False)
        policy.note_state(Link(), _tele(mode="PREPARE"), latched=False)
        self.assertEqual(policy._last_mode, HumanoidMode.PREPARE)
        link = Link()
        policy.note_state(link, _tele(mode="FALLEN", fall="fallen"), latched=True)
        self.assertTrue(policy._damped)
        policy.note_state(link, _tele(mode="FALLEN", fall="fallen"), latched=True)
        self.assertEqual(
            [call for call in link.calls if call[0] == "humanoid"],
            [("humanoid", "damp", (), {})],
        )


class TestSpeedCaps(PolicyCase):
    def test_two_meters_per_second_clamps_to_sim_cap_and_is_reported(self) -> None:
        gate, inner = self.make()
        self.clock.advance(0.3)
        inner.emit(_tele(timestamp=self.clock()))
        gate.send_velocity(Velocity(2.0, 0.8, 2.0))
        self.assertEqual(inner.calls[-1], ("vel", (0.5, 0.3, 0.6)))
        record = self.events("clamp")[-1]
        self.assertEqual(record["requested"], [2.0, 0.8, 2.0])
        self.assertEqual(record["applied"], [0.5, 0.3, 0.6])
        self.assertIn("velocity_cap", record["reasons"])
        self.clock.advance(0.3)
        inner.emit(_tele(timestamp=self.clock()))
        gate.send_velocity(Velocity(-2.0, -0.8, -2.0))
        self.assertEqual(inner.calls[-1][1], (-0.3, -0.3, -0.6))

    def test_hardware_caps_until_a_dated_walk(self) -> None:
        gate, inner = self.make(
            protocol="serial",
            environ={ARM_ENV: "1"},
            tty=True,
        )
        self.assertTrue(gate.requires_hardware_arming)
        self.assertIn(ARM_PHRASE, self.prompts[-1])
        self.clock.advance(0.3)
        inner.emit(_tele(timestamp=self.clock()))
        gate.send_velocity(Velocity(2.0, 0.8, 2.0))
        self.assertEqual(inner.calls[-1][1], (0.3, 0.2, 0.5))
        self.clock.advance(0.3)
        inner.emit(_tele(timestamp=self.clock()))
        gate.send_velocity(Velocity(-2.0, 0.0, 0.0))
        self.assertEqual(inner.calls[-1][1][0], -0.2)
        self.assertEqual(HARDWARE_CAPS.max_vx_forward, 0.3)
        self.assertEqual(HARDWARE_CAPS.max_vx_back, 0.2)
        self.assertEqual(HARDWARE_CAPS.max_vy, 0.2)
        self.assertEqual(HARDWARE_CAPS.max_wz, 0.5)

    def test_acceleration_and_non_finite(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        policy.arming("simulated://", {})
        slow = policy.clamp(
            "velocity",
            ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), 0.01),
            None,
        )
        self.assertIn("accel_cap", slow.reasons)
        self.assertLess(slow.applied[0], 0.5)
        within = policy.clamp(
            "velocity",
            ((0.1, 0.0, 0.0), (0.1, 0.0, 0.0), 0.3),
            None,
        )
        self.assertEqual(within.reasons, ())
        self.assertEqual(within.applied, (0.1, 0.0, 0.0))
        rejected = policy.clamp(
            "velocity",
            ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0), 0.3),
            None,
        )
        self.assertEqual(rejected.reject, "non_finite")
        short = policy.clamp("velocity", ((0.0, 0.0),), None)
        self.assertEqual(short.reject, "non_finite")


class TestArming(PolicyCase):
    def test_loopback_sim_arms_only_via_the_sim_target_check(self) -> None:
        gate, _inner = self.make(protocol="simulated", arm=False, tty=False)
        self.assertFalse(gate.requires_hardware_arming)
        self.assertTrue(explicit_sim_target("simulated://"))
        self.assertTrue(explicit_sim_target("sim://walker"))
        self.assertTrue(gate.arm())
        self.assertFalse(self.events("arm")[-1]["hardware"])
        self.assertEqual(self.prompts, [])

    def test_mujoco_loopback_requires_the_sim_target_flag(self) -> None:
        flagged = Link("humanoid-mujoco", "127.0.0.1", True)
        self.assertIn("sim_target=True", arming_uri(flagged))
        gate, _inner = self.make(link=flagged, arm=False, tty=False)
        self.assertFalse(gate.requires_hardware_arming)
        self.assertTrue(gate.arm())

        plain = Link("humanoid-mujoco", "localhost", False)
        refused, _inner = self.make(link=plain, arm=False, tty=False)
        self.assertTrue(refused.requires_hardware_arming)
        self.assertFalse(refused.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "env")

        remote = Link("humanoid-mujoco", "10.1.1.1", True)
        remote_gate, _inner = self.make(link=remote, arm=False, tty=False)
        self.assertTrue(remote_gate.requires_hardware_arming)
        self.assertFalse(explicit_sim_target(arming_uri(remote)))

        ipv6 = Link("humanoid-mujoco", "::1", True)
        self.assertTrue(explicit_sim_target(arming_uri(ipv6)))
        self.assertFalse(explicit_sim_target(None))
        self.assertFalse(
            explicit_sim_target("humanoid-mujoco://127.0.0.1?sim_target=true")
        )
        self.assertFalse(explicit_sim_target("serial://"))

    def test_real_uri_needs_the_phrase_from_a_mock_env(self) -> None:
        gate, _inner = self.make(protocol="serial", arm=False)
        self.assertTrue(gate.requires_hardware_arming)
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "env")
        armed, _inner = self.make(
            protocol="serial",
            environ={ARM_ENV: "1"},
            tty=True,
        )
        self.assertTrue(armed.armed)
        self.assertIn(ARM_PHRASE, self.prompts[-1])


class TestJoints(PolicyCase):
    def _limited(self) -> SafetyProfile:
        return dataclasses.replace(
            class_profile(SIM_CAPS),
            joint_limits={"j0": (-1.0, 1.0), "j1": (-0.4, 0.4)},
        )

    def test_move_joints_without_a_limit_is_refused(self) -> None:
        gate, inner = self.make()
        inner.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertFalse(gate.command("move_joints", "left_arm", (0.05, 0.0)))
        self.assertEqual(self.events("reject")[-1]["reason"], "no_joint_limit")
        self.assertFalse(any(call[0] == "humanoid" for call in inner.calls))

    def test_move_joints_refused_outside_custom(self) -> None:
        gate, inner = self.make(profile=self._limited())
        for mode in ("WALK", "PREPARE", "UNKNOWN", "DAMPING", "OFF"):
            with self.subTest(mode=mode):
                inner.emit(_tele(mode=mode, timestamp=self.clock()))
                self.assertFalse(
                    gate.command("move_joints", "left_arm", (0.1, 0.1), None, None)
                )
                self.assertEqual(self.events("reject")[-1]["reason"], "not_custom")
        self.assertFalse(any(call[0] == "humanoid" for call in inner.calls))

    def test_move_joints_refused_via_mcp_even_in_custom(self) -> None:
        self.assertNotIn("move_joints", HUMANOID_MCP_TOOLS)
        self.assertNotIn("damp", HUMANOID_MCP_TOOLS)
        self.assertNotIn("arm", HUMANOID_MCP_TOOLS)
        self.assertFalse(humanoid_mcp_allowed("move_joints"))
        self.assertFalse(humanoid_mcp_allowed("MoveJoints"))
        self.assertFalse(humanoid_mcp_allowed(""))
        self.assertFalse(humanoid_mcp_allowed("damp"))
        decision = mcp_command("move_joints", "left_arm", (0.1, 0.1), mode="CUSTOM")
        self.assertEqual(decision.reject, "mcp_refused")
        self.assertEqual(mcp_command(None).reject, "mcp_refused")
        allowed = mcp_command("prepare")
        self.assertIsNone(allowed.reject)
        self.assertEqual(allowed.applied, ("prepare",))
        self.assertIn("prepare", HUMANOID_MCP_TOOLS)
        self.assertTrue(humanoid_mcp_allowed("walk_mode"))

    def test_move_joints_in_custom_clamps_step_and_limits(self) -> None:
        spec = _spec(
            limits=HumanoidLimits(max_kp=20.0, max_tau=8.0, max_joint_step=0.1)
        )
        gate, inner = self.make(spec=spec, profile=self._limited(), link=KeywordLink())
        inner.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertTrue(gate.command("move_joints", "left_arm", (0.2, 0.5), 25.0, None))
        sent = [call for call in inner.calls if call[0] == "humanoid"][-1]
        self.assertEqual(sent[1], "move_joints")
        self.assertEqual(sent[3]["group"], "left_arm")
        self.assertEqual(sent[3]["q"], (0.1, 0.1))
        self.assertEqual(sent[3]["kp"], 20.0)
        record = self.events("clamp")[-1]
        self.assertIn("joint_step", record["reasons"])
        self.assertIn("joint_limit", record["reasons"])
        self.assertIn("kp_cap", record["reasons"])

    def test_keyword_and_positional_forwarding(self) -> None:
        gate, inner = self.make(profile=self._limited(), link=KeywordLink())
        inner.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertTrue(gate.command("move_joints", "left_arm", (0.05, -0.05)))
        self.assertEqual(inner.calls[-1][3]["q"], (0.05, -0.05))

        positional, link = self.make(profile=self._limited())
        link.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertTrue(positional.command("move_joints", "left_arm", (0.05, 0.0)))
        self.assertEqual(link.calls[-1][2][1], (0.05, 0.0))

    def test_joint_refusals(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        policy.arming("simulated://", {})
        profile = self._limited()
        policy = policy.bind(profile)
        policy.arming("simulated://", {})
        tele = _joints_at(0.0, 0.0)
        self.assertEqual(
            policy.low_level_joints(("left_arm", (0.1,)), tele).reject,
            "bad_joint_shape",
        )
        self.assertEqual(
            policy.low_level_joints(("missing", (0.1, 0.1)), tele).reject,
            "unknown_group",
        )
        self.assertEqual(
            policy.low_level_joints(("left_arm",), tele).reject, "bad_arguments"
        )
        self.assertEqual(
            policy.low_level_joints(("", (0.1, 0.1)), tele).reject, "bad_arguments"
        )
        self.assertEqual(
            policy.low_level_joints(("left_arm", (0.1, float("nan"))), tele).reject,
            "bad_arguments",
        )
        self.assertEqual(
            policy.low_level_joints(("left_arm", "nope"), tele).reject, "bad_arguments"
        )
        bare = policy.low_level_joints(("left_arm", (0.1, 0.1)), _tele(mode="CUSTOM"))
        self.assertEqual(bare.reject, "no_joint_reference")
        gained = policy.low_level_joints(("left_arm", (0.0, 0.0), 1.0, 1.0), tele)
        self.assertEqual(gained.reject, "no_kp_limit")
        kd_only = policy.low_level_joints(("left_arm", (0.0, 0.0), None, 1.0), tele)
        self.assertEqual(kd_only.reject, "no_kd_limit")
        policy.caps = dataclasses.replace(policy.caps, max_kp=5.0)
        negative = policy.low_level_joints(("left_arm", (0.0, 0.0), -1.0, None), tele)
        self.assertEqual(negative.reject, "bad_gain")
        policy._sim_target = False
        self.assertEqual(
            policy.low_level_joints(("left_arm", (0.0, 0.0)), tele).reject,
            "not_simulated",
        )
        narrowed = dataclasses.replace(
            profile, allowlist=frozenset({"move", "stop_move"})
        )
        blocked = policy.bind(narrowed)
        blocked._sim_target = True
        self.assertEqual(
            blocked.low_level_joints(("left_arm", (0.0, 0.0)), tele).reject,
            "not_allowlisted",
        )

    def test_disarmed_and_failing_joint_commands(self) -> None:
        gate, inner = self.make(profile=self._limited(), arm=False)
        inner.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertFalse(gate.command("move_joints", "left_arm", (0.05, 0.0)))
        self.assertEqual(self.events("reject")[-1]["reason"], "not_armed")

        class NoHook(Link):
            send_humanoid = None  # type: ignore[assignment]

        no_hook = NoHook()
        bare, _inner = self.make(profile=self._limited(), link=no_hook)
        no_hook.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        self.assertFalse(bare.command("move_joints", "left_arm", (0.05, 0.0)))
        self.assertEqual(
            self.events("reject")[-1]["reason"], "unsupported_by_transport"
        )

        failing, link = self.make(profile=self._limited())
        link.fail.add("humanoid")
        link.emit(_joints_at(0.0, 0.0, timestamp=self.clock()))
        with self.assertLogs("ohho.safety", level="ERROR"):
            self.assertFalse(failing.command("move_joints", "left_arm", (0.05, 0.0)))
        self.assertTrue(failing.latched)
        self.assertEqual(failing.latch_reason, "transport_error")

    def test_effort_ceiling(self) -> None:
        policy = HumanoidPolicy.for_spec(
            _spec(limits=HumanoidLimits(max_tau=4.0, max_kp=1.0))
        )
        decision = policy.clamp("effort", ("j0", 9.0), None)
        self.assertEqual(decision.applied, (4.0,))
        self.assertIn("effort_cap", decision.reasons)
        within = policy.clamp("effort", ("j0", 1.0), None)
        self.assertEqual(within.applied, (1.0,))
        self.assertEqual(within.reasons, ())
        plain = HumanoidPolicy.for_spec(_spec())
        refused = plain.clamp("effort", ("j0", 1.0), None)
        self.assertEqual(refused.reject, "no_effort_limit")
        bad = policy.clamp("effort", ("j0", float("nan")), None)
        self.assertEqual(bad.reject, "non_finite")
        limited = dataclasses.replace(
            class_profile(SIM_CAPS), effort_limits={"j0": 2.0}
        )
        bound = policy.bind(limited)
        both = bound.clamp("effort", ("j0", 3.0), None)
        self.assertEqual(both.applied, (2.0,))

    def test_reference_falls_back_to_the_last_command(self) -> None:
        policy = HumanoidPolicy(
            self._limited(),
            spec=_spec(),
            caps=dataclasses.replace(SIM_CAPS, max_kp=5.0, max_kd=1.0),
        )
        policy.arming("sim://", {})
        first = policy.low_level_joints(
            ("left_arm", (0.05, 0.0), 5.0, 0.2),
            _joints_at(0.0, 0.0),
        )
        self.assertIsNone(first.reject)
        second = policy.low_level_joints(
            ("left_arm", (0.4, 0.0), 5.0, 0.2),
            _tele(mode="CUSTOM"),
        )
        self.assertAlmostEqual(second.applied[1][0], 0.2)
        self.assertEqual(second.applied[1][1], 0.0)
        self.assertIn("joint_step", second.reasons)
        empty = HumanoidPolicy(
            self._limited(), spec=_spec(joint_groups={"left_arm": ()})
        )
        self.assertIsNone(empty._group_names("left_arm"))
        self.assertIsNone(empty._group_names("other"))


class TestPluginOverride(PolicyCase):
    def test_loosening_override_is_refused(self) -> None:
        spec = _spec()
        loose = HumanoidPolicy(
            dataclasses.replace(class_profile(SIM_CAPS), max_vx=0.8),
            spec=spec,
            caps=dataclasses.replace(SIM_CAPS, max_vx_forward=0.8),
        )

        class Plugin:
            def safety_policy(self, robot: RobotSpec) -> HumanoidPolicy:
                del robot
                return loose

        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=Plugin())
        self.assertIn("refused at load", str(raised.exception))
        self.assertIn("max_vx", str(raised.exception))

        back = HumanoidPolicy(
            class_profile(SIM_CAPS),
            spec=spec,
            caps=dataclasses.replace(SIM_CAPS, max_vx_back=0.4),
        )
        with self.assertRaises(SafetyPolicyError) as raised_back:
            load_policy(spec, plugin=_Plugin(back))
        self.assertIn("max_vx_back", str(raised_back.exception))

        enabled = HumanoidPolicy(
            class_profile(SIM_CAPS),
            spec=spec,
            caps=dataclasses.replace(SIM_CAPS, max_kp=10.0),
        )
        with self.assertRaises(SafetyPolicyError) as raised_gain:
            load_policy(spec, plugin=_Plugin(enabled))
        self.assertIn("max_kp", str(raised_gain.exception))

    def test_tighter_override_is_kept(self) -> None:
        spec = _spec()
        tight = HumanoidPolicy(
            dataclasses.replace(class_profile(SIM_CAPS), max_vx=0.2, max_vy=0.1),
            spec=spec,
            caps=dataclasses.replace(
                SIM_CAPS, max_vx_forward=0.2, max_vx_back=0.1, max_vy=0.1
            ),
        )
        loaded = load_policy(spec, plugin=_Plugin(tight))
        self.assertIs(loaded, tight)
        self.assertTrue(loaded.from_plugin)
        loaded.arming("simulated://", {})
        decision = loaded.clamp(
            "velocity",
            ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.3),
            None,
        )
        self.assertEqual(decision.applied[0], 0.2)
        self.assertEqual(loaded.looser_than(HumanoidPolicy.for_spec(spec)), [])
        self.assertEqual(loaded.looser_than(SafetyPolicy(class_profile(SIM_CAPS))), [])
        self.assertEqual(
            SafetyPolicy(class_profile(SIM_CAPS)).looser_than(loaded),
            [],
        )


class _Plugin:
    def __init__(self, policy: HumanoidPolicy) -> None:
        self.policy = policy

    def safety_policy(self, spec: RobotSpec) -> HumanoidPolicy:
        del spec
        return self.policy


class TestDenylistAndCaps(unittest.TestCase):
    def test_dances_kicks_and_drag_are_denied(self) -> None:
        policy = HumanoidPolicy.for_spec(_spec())
        for name in (
            "dance",
            "lion_dance",
            "kick",
            "shoot",
            "handshake",
            "ZeroTorqueDrag",
            "record_trajectory",
            "replay_trajectory",
            "load_custom_trained_traj",
            "front_flip",
        ):
            with self.subTest(name=name):
                normalised = normalize_action(name)
                self.assertTrue(policy.denied(normalised))
                self.assertTrue(humanoid_denied(normalised))

    def test_cap_helpers(self) -> None:
        self.assertEqual(SIM_CAPS.max_vx_forward, 0.5)
        self.assertEqual(MAX_JOINT_STEP_RAD, 0.15)
        self.assertEqual(TILT_LIMIT_RAD, 0.5)
        tighter = intersect_caps(
            SIM_CAPS, dataclasses.replace(SIM_CAPS, max_vy=0.1, max_kp=None)
        )
        self.assertEqual(tighter.max_vy, 0.1)
        self.assertIsNone(tighter.max_kp)
        both = intersect_caps(
            dataclasses.replace(SIM_CAPS, max_kp=10.0),
            dataclasses.replace(SIM_CAPS, max_kp=4.0),
        )
        self.assertEqual(both.max_kp, 4.0)
        kept = intersect_caps(dataclasses.replace(SIM_CAPS, max_kp=4.0), SIM_CAPS)
        self.assertEqual(kept.max_kp, 4.0)
        self.assertEqual(
            cap_looser_reasons(
                dataclasses.replace(SIM_CAPS, tilt_hold_s=0.2, cmd_timeout_s=0.4),
                SIM_CAPS,
            ),
            [
                "tilt_hold_s 0.2 exceeds class cap 0.1",
                "cmd_timeout_s 0.4 exceeds class cap 0.3",
            ],
        )
        lowered = HumanoidPolicy.for_spec(
            _spec(limits=HumanoidLimits(max_joint_step=0.05, max_kp=3.0))
        )
        self.assertEqual(lowered.caps.max_joint_step, 0.05)
        self.assertEqual(lowered.caps.max_kp, 3.0)
        self.assertEqual(HumanoidPolicy.for_spec(None).plugin_name, "")
        wider = HumanoidPolicy.for_spec(
            _spec(limits=HumanoidLimits(max_joint_step=1.0, max_kp=0.0))
        )
        self.assertEqual(wider.caps.max_joint_step, MAX_JOINT_STEP_RAD)
        self.assertIsNone(wider.caps.max_kp)
        with self.assertRaises(ValueError):
            HumanoidCaps(tilt_rad=0.0)
        with self.assertRaises(ValueError):
            HumanoidCaps(max_kp=-1.0)
        self.assertFalse(HumanoidPolicy.for_spec(_spec())._send(Link(), "zero_torque"))


class TestGatePluginRefusal(PolicyCase):
    def test_gate_refuses_a_loosening_plugin_before_connect(self) -> None:
        spec = _spec()
        loose = HumanoidPolicy(
            dataclasses.replace(class_profile(SIM_CAPS), max_wz=0.9),
            spec=spec,
            caps=dataclasses.replace(SIM_CAPS, max_wz=0.9),
        )

        def named(source: str) -> _Plugin:
            self.assertEqual(source, "walker-plugin")
            return _Plugin(loose)

        from unittest.mock import patch

        with patch("ohho.plugins.plugin_named", named):
            with self.assertRaises(SafetyPolicyError):
                SafetyGate(Link(), spec, audit=AuditLog(self.tmp), auto_tick=False)


if __name__ == "__main__":
    unittest.main()
