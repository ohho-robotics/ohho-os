"""Tests for ``ohho.safety`` — the SafetyGate transport wrapper (OHH-117).

Every test is deterministic: a fake monotonic clock, fake transports, injected
environment / TTY / input, and a temporary audit directory. Nothing sleeps,
nothing touches hardware, and the watchdog is driven by calling ``tick()``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from ohho import safety
from ohho.adapters.sim import SimTransport
from ohho.registry import RobotSpec, get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import (
    ARM_ENV,
    ARM_PHRASE,
    GO2_PROFILE,
    OMNIBOT_PROFILE,
    RELEASE_PHRASE,
    AuditLog,
    SafetyConfig,
    SafetyGate,
    SafetyProfile,
    is_denied,
    load_safety_config,
    normalize_action,
    profile_for,
    requires_hardware_arming,
    resolve_config,
)
from ohho.schema import ConnectionState, JointReading, Telemetry, TransportStatus
from ohho.schema import Velocity
from ohho.transport import BaseTransport

WALL = 1_760_000_000.0


def setUpModule() -> None:
    # Keep the gate's warnings out of the test output (assertLogs still works).
    logging.getLogger("ohho.safety").addHandler(logging.NullHandler())


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeTransport(BaseTransport):
    """Records every call; ``fail`` names calls that should raise."""

    def __init__(self, protocol: str = "serial") -> None:
        super().__init__()
        self.protocol = protocol
        self.state = ConnectionState.IDLE
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self.latest = Telemetry()

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="fake")

    def read(self) -> Telemetry:
        return self.latest

    def _rec(self, name: str, *args) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, *args))

    def send_velocity(self, vel: Velocity) -> None:
        self._rec("vel", (vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        self._rec("joint", name, position)

    def emergency_stop(self) -> None:
        self._rec("estop")

    def release_stop(self) -> None:
        self._rec("release")

    def emit(self, joints=None, **custom) -> Telemetry:
        t = Telemetry(joints=list(joints or []), custom=custom)
        self.latest = t
        self._emit_telemetry(t)
        return t

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def velocities(self) -> list[tuple]:
        return [c[1] for c in self.calls if c[0] == "vel"]


class FakeGo2(FakeTransport):
    def __init__(self, iface: str = "eth0", domain_id: int = 0) -> None:
        super().__init__("dds")
        self.iface = iface
        self.domain_id = domain_id

    def sport_command(self, name: str, *args) -> None:
        self._rec("sport", name, *args)


class FakeEffortArm(FakeTransport):
    def send_joint_effort(self, name: str, effort: float) -> None:
        self._rec("effort", name, effort)


class FailingAudit(AuditLog):
    def __init__(self) -> None:
        super().__init__(tempfile.gettempdir())
        self.fail = False
        self.records: list[dict] = []

    def write(self, event: str, **fields) -> bool:
        if self.fail:
            return False
        self.records.append({"event": event, **fields})
        return True


def fast(profile: SafetyProfile, **kw) -> SafetyProfile:
    """Same velocity caps, effectively no acceleration limit."""
    return replace(profile, max_accel_x=1e6, max_accel_y=1e6, max_accel_wz=1e6, **kw)


ZERO = (0.0, 0.0, 0.0)


class GateCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-safety-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()
        self.env: dict[str, str] = {ARM_ENV: "1"}
        self.tty = True
        self.answer = ARM_PHRASE
        self.prompts: list[str] = []

    def _input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        if callable(self.answer):
            return self.answer()
        return self.answer

    def make(
        self,
        inner: FakeTransport | None = None,
        spec: str | RobotSpec | None = "unitree-go2",
        profile: SafetyProfile | None = None,
        arm: bool = True,
        **kw,
    ) -> tuple[SafetyGate, FakeTransport]:
        inner = inner if inner is not None else FakeGo2()
        if isinstance(spec, str):
            spec = get_spec(spec)
        kw.setdefault("audit", AuditLog(self.tmp, wall_clock=lambda: WALL))
        kw.setdefault("auto_tick", False)
        gate = SafetyGate(
            inner,
            spec,
            profile=profile,
            clock=self.clock,
            environ=self.env,
            isatty=lambda: self.tty,
            input_fn=self._input,
            **kw,
        )
        gate.connect()
        self.state(inner)
        if arm:
            self.assertTrue(gate.arm())
        inner.calls.clear()
        return gate, inner

    def state(self, inner: FakeTransport, joints=None, **custom) -> Telemetry:
        base = {"roll": 0.0, "pitch": 0.0, "error_code": 0, "body_height": 0.3}
        base.update(custom)
        return inner.emit(joints=joints, **base)

    def events(self, name: str | None = None) -> list[dict]:
        out: list[dict] = []
        for p in sorted(Path(self.tmp).glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if name is None or rec["event"] == name:
                    out.append(rec)
        return out


# ── profiles, ceilings, config ────────────────────────────────────────────────
class TestProfiles(GateCase):
    def test_go2_defaults(self):
        p = profile_for("unitree-go2")
        self.assertIs(p, GO2_PROFILE)
        self.assertEqual((p.max_vx, p.max_vy, p.max_wz), (0.5, 0.3, 1.0))
        self.assertGreater(p.max_accel_x, 0.0)
        self.assertTrue(p.require_orientation)
        self.assertEqual(p.cmd_timeout_s, 0.3)
        self.assertEqual(p.state_timeout_s, 0.5)
        self.assertEqual(p.tilt_limit_rad, 0.6)
        self.assertEqual(p.euler_limit_rad, 0.3)
        self.assertFalse(p.estop_damp)
        self.assertEqual(
            p.allowlist,
            frozenset(
                {
                    "stand_up",
                    "stand_down",
                    "balance_stand",
                    "recovery_stand",
                    "stop_move",
                    "move",
                    "euler",
                    "sit",
                    "rise_sit",
                }
            ),
        )

    def test_omnibot_profile_has_arm_limits(self):
        p = profile_for("omnibot")
        self.assertIs(p, OMNIBOT_PROFILE)
        self.assertEqual(p.max_vx, 0.2)
        self.assertEqual(set(p.joint_limits), set(get_spec("omnibot").joint_names))
        self.assertEqual(p.allowlist, frozenset({"move", "stop_move"}))

    def test_unknown_robot_profile_is_conservative(self):
        spec = RobotSpec(
            id="diff", name="Diff", category="wheeled", capabilities=("base.drive",)
        )
        spec = replace(spec, max_lin=3.0, max_ang=5.0)
        p = profile_for("diff", spec)
        self.assertLessEqual(p.max_vx, 0.3)
        self.assertEqual(p.max_vy, 0.0)  # not holonomic
        self.assertLessEqual(p.max_wz, 1.0)
        self.assertEqual(p.allowlist, frozenset({"move", "stop_move"}))
        self.assertEqual(profile_for("nothing-at-all"), SafetyProfile())

    def test_profile_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            GO2_PROFILE.max_vx = 3.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            OMNIBOT_PROFILE.joint_limits["arm_gripper"] = (-9.0, 9.0)  # type: ignore[index]
        hash(GO2_PROFILE)

    def test_profile_validation_fails_closed(self):
        bad = [
            {"cmd_timeout_s": 0.5},
            {"cmd_timeout_s": 0.0},
            {"state_timeout_s": 0.6},
            {"tilt_limit_rad": 0.7},
            {"tilt_limit_rad": 0.0},
            {"euler_limit_rad": 0.4},
            {"max_vx": -0.1},
            {"max_wz": float("nan")},
            {"max_accel_x": float("inf")},
            {"max_joint_step": -1.0},
            {"lying_body_height": float("nan")},
            {"allowlist": frozenset({"front_flip"})},
            {"allowlist": frozenset({"teleport"})},
            {"joint_limits": {"j": (1.0, -1.0)}},
            {"joint_limits": {"j": (float("nan"), 1.0)}},
            {"effort_limits": {"j": -2.0}},
        ]
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                SafetyProfile(**kw)

    def test_allowlist_is_normalized(self):
        p = SafetyProfile(allowlist=frozenset({"StandUp", "stop-move"}))
        self.assertEqual(p.allowlist, frozenset({"stand_up", "stop_move"}))

    def test_hard_ceiling_cannot_be_raised_by_caller(self):
        gate, _ = self.make(
            profile=SafetyProfile(max_vx=3.0, max_vy=3.0, max_wz=5.0), arm=False
        )
        self.assertEqual(gate.profile.max_vx, safety.HARD_MAX_LIN)
        self.assertEqual(gate.profile.max_vy, safety.HARD_MAX_LIN)
        self.assertEqual(gate.profile.max_wz, safety.HARD_MAX_ANG)
        self.assertEqual((safety.HARD_MAX_LIN, safety.HARD_MAX_ANG), (1.5, 2.0))

    def test_config_file_raises_ceiling_and_overrides_profile(self):
        path = Path(self.tmp) / "safety.json"
        path.write_text(
            json.dumps(
                {
                    "hard_ceiling": {"max_lin": 2.0, "max_ang": 3.0},
                    "profiles": {
                        "unitree-go2": {
                            "max_vx": 1.8,
                            "max_wz": 2.5,
                            "allowlist": ["move", "stop_move"],
                            "estop_damp": True,
                        },
                        "omnibot": {
                            "joint_limits": {"arm_gripper": [0.0, 0.5]},
                            "effort_limits": {"arm_gripper": 1.0},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        cfg = load_safety_config(path)
        self.assertEqual((cfg.max_lin_ceiling, cfg.max_ang_ceiling), (2.0, 3.0))
        self.assertEqual(cfg.source, str(path))
        go2 = profile_for("unitree-go2", config=cfg)
        self.assertEqual((go2.max_vx, go2.max_vy, go2.max_wz), (1.8, 0.3, 2.5))
        self.assertEqual(go2.allowlist, frozenset({"move", "stop_move"}))
        self.assertTrue(go2.estop_damp)
        omni = profile_for("omnibot", config=cfg)
        self.assertEqual(omni.joint_limits["arm_gripper"], (0.0, 0.5))
        self.assertEqual(omni.effort_limits["arm_gripper"], 1.0)
        gate, _ = self.make(config=cfg, arm=False)
        self.assertEqual(gate.profile.max_vx, 1.8)
        self.assertEqual(gate.profile.max_wz, 2.5)

    def test_config_file_rejects_bad_content(self):
        cases = [
            "not json",
            json.dumps([1, 2]),
            json.dumps({"unknown": 1}),
            json.dumps({"hard_ceiling": {"max_lin": -1}}),
            json.dumps({"hard_ceiling": {"max_lin": "fast"}}),
            json.dumps({"hard_ceiling": {"speed": 1}}),
            json.dumps({"hard_ceiling": []}),
            json.dumps({"profiles": []}),
            json.dumps({"profiles": {"unitree-go2": []}}),
            json.dumps({"profiles": {"unitree-go2": {"max_vx": 9.0}}}),
            json.dumps({"profiles": {"unitree-go2": {"warp": 1}}}),
            json.dumps({"profiles": {"unitree-go2": {"allowlist": ["backflip"]}}}),
            json.dumps({"profiles": {"unitree-go2": {"allowlist": "move"}}}),
            json.dumps({"profiles": {"omnibot": {"joint_limits": {"j": [1]}}}}),
        ]
        path = Path(self.tmp) / "bad.json"
        for text in cases:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_safety_config(path)
        with self.assertRaises(ValueError):
            load_safety_config(Path(self.tmp) / "missing.json")

    def test_resolve_config_sources(self):
        self.assertEqual(resolve_config(environ={}), SafetyConfig())
        path = Path(self.tmp) / "c.json"
        path.write_text(json.dumps({"hard_ceiling": {"max_lin": 1.0}}), "utf-8")
        cfg = resolve_config(environ={safety.CONFIG_ENV: str(path)})
        self.assertEqual(cfg.max_lin_ceiling, 1.0)
        self.assertEqual(resolve_config(str(path), environ={}).max_lin_ceiling, 1.0)


# ── velocity + acceleration caps ──────────────────────────────────────────────
class TestVelocityCaps(GateCase):
    def test_clamps_each_axis_not_rejects(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        self.clock.advance(0.01)
        gate.send_velocity(Velocity(2.0, 1.0, 3.0))
        self.assertEqual(inner.velocities()[-1], (0.5, 0.3, 1.0))
        self.clock.advance(0.01)
        gate.send_velocity(Velocity(-2.0, -1.0, -3.0))
        self.assertEqual(inner.velocities()[-1], (-0.5, -0.3, -1.0))
        clamps = self.events("clamp")
        self.assertEqual(len(clamps), 2)
        self.assertEqual(clamps[0]["requested"], [2.0, 1.0, 3.0])
        self.assertEqual(clamps[0]["applied"], [0.5, 0.3, 1.0])
        self.assertIn("velocity_cap", clamps[0]["reasons"])

    def test_within_caps_passes_unchanged(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        self.clock.advance(0.01)
        gate.send_velocity(Velocity(0.2, -0.1, 0.4))
        self.assertEqual(inner.velocities(), [(0.2, -0.1, 0.4)])
        self.assertEqual(self.events("clamp"), [])

    def test_acceleration_limited_per_axis(self):
        gate, inner = self.make(profile=GO2_PROFILE)  # 1.0 m/s^2, 0.6, 2.0
        self.clock.advance(0.1)
        gate.send_velocity(Velocity(0.5, 0.3, 1.0))
        vx, vy, wz = inner.velocities()[-1]
        self.assertAlmostEqual(vx, 0.1, places=6)
        self.assertAlmostEqual(vy, 0.06, places=6)
        self.assertAlmostEqual(wz, 0.2, places=6)
        self.clock.advance(0.1)
        gate.send_velocity(Velocity(0.5, 0.3, 1.0))
        vx, vy, wz = inner.velocities()[-1]
        self.assertAlmostEqual(vx, 0.2, places=6)
        self.assertAlmostEqual(vy, 0.12, places=6)
        self.assertAlmostEqual(wz, 0.4, places=6)
        self.assertIn("accel_cap", self.events("clamp")[-1]["reasons"])

    def ramp(self, gate, inner, vx: float, steps: int = 2) -> None:
        for _ in range(steps):
            self.clock.advance(0.3)
            self.state(inner)
            gate.send_velocity(Velocity(vx))

    def test_braking_is_never_limited(self):
        gate, inner = self.make(profile=GO2_PROFILE)
        self.ramp(gate, inner, 0.5)
        self.assertAlmostEqual(inner.velocities()[-1][0], 0.5)
        self.clock.advance(0.05)
        gate.send_velocity(Velocity(0.2))
        self.assertAlmostEqual(inner.velocities()[-1][0], 0.2)
        self.clock.advance(0.05)
        gate.send_velocity(Velocity(0.0))
        self.assertEqual(inner.velocities()[-1], ZERO)

    def test_reversal_brakes_to_zero_then_limits(self):
        gate, inner = self.make(profile=GO2_PROFILE)
        self.ramp(gate, inner, 0.4)
        self.assertAlmostEqual(inner.velocities()[-1][0], 0.4)
        self.clock.advance(0.1)
        gate.send_velocity(Velocity(-0.5))
        self.assertAlmostEqual(inner.velocities()[-1][0], -0.1, places=6)

    def test_accel_dt_is_capped_at_command_timeout(self):
        gate, inner = self.make(profile=GO2_PROFILE)
        self.clock.advance(0.45)
        self.state(inner)
        gate.send_velocity(Velocity(0.5))
        self.assertAlmostEqual(inner.velocities()[-1][0], 0.3, places=6)

    def test_non_finite_command_rejected_and_zeroed(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        for bad in (float("nan"), float("inf"), -float("inf")):
            gate.send_velocity(Velocity(bad, 0.0, 0.0))
            self.assertEqual(inner.velocities()[-1], ZERO)
        rej = self.events("reject")
        self.assertEqual({r["reason"] for r in rej}, {"non_finite"})
        self.assertEqual(rej[0]["command"], "send_velocity")

    def test_transport_error_latches(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        inner.fail.add("vel")
        gate.send_velocity(Velocity(0.2))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "transport_error")
        self.assertIn("estop", inner.names())


# ── effort + joint caps ───────────────────────────────────────────────────────
class TestJointAndEffortCaps(GateCase):
    def omni(self, inner=None, profile=None, joints=None, arm=True):
        inner = inner if inner is not None else FakeEffortArm()
        prof = profile or replace(
            OMNIBOT_PROFILE, effort_limits={"arm_gripper": 1.5, "arm_elbow_flex": 2.0}
        )
        gate, inner = self.make(inner, "omnibot", profile=prof, arm=False)
        if joints is not None:
            self.state(inner, joints=joints)
        if arm:
            self.assertTrue(gate.arm())
        inner.calls.clear()
        return gate, inner

    def test_effort_clamped_to_limit(self):
        gate, inner = self.omni()
        gate.send_joint_effort("arm_gripper", 5.0)
        gate.send_joint_effort("arm_elbow_flex", -9.0)
        gate.send_joint_effort("arm_gripper", 0.5)
        self.assertEqual(
            [c for c in inner.calls if c[0] == "effort"],
            [
                ("effort", "arm_gripper", 1.5),
                ("effort", "arm_elbow_flex", -2.0),
                ("effort", "arm_gripper", 0.5),
            ],
        )
        self.assertEqual(len(self.events("clamp")), 2)

    def test_effort_refused_without_limit_support_or_arming(self):
        gate, inner = self.omni()
        gate.send_joint_effort("arm_wrist_roll", 0.1)  # no limit configured
        gate.send_joint_effort("arm_gripper", float("nan"))
        gate.disarm()
        gate.send_joint_effort("arm_gripper", 0.1)
        self.assertNotIn("effort", inner.names())
        reasons = [r["reason"] for r in self.events("reject")]
        self.assertEqual(reasons, ["no_effort_limit", "non_finite", "not_armed"])
        plain, plain_inner = self.omni(inner=FakeTransport())
        plain.send_joint_effort("arm_gripper", 0.1)
        self.assertEqual(self.events("reject")[-1]["reason"], "effort_unsupported")

    def test_joint_position_clamped_to_limits_and_step(self):
        joints = [JointReading("arm_gripper", 0.7), JointReading("arm_elbow_flex", 0.0)]
        gate, inner = self.omni(joints=joints)
        gate.send_joint_command("arm_gripper", 3.0)  # limit 0.8, step from 0.7
        self.assertEqual(inner.calls[-1], ("joint", "arm_gripper", 0.8))
        gate.send_joint_command("arm_elbow_flex", 1.0)  # step 0.25 from 0.0
        self.assertAlmostEqual(inner.calls[-1][2], OMNIBOT_PROFILE.max_joint_step)
        gate.send_joint_command("arm_elbow_flex", 0.1)  # within step
        self.assertAlmostEqual(inner.calls[-1][2], 0.1)
        self.assertEqual(len(self.events("clamp")), 2)

    def test_joint_reference_falls_back_to_last_command(self):
        gate, inner = self.omni(joints=[JointReading("arm_gripper", 0.0)])
        gate.send_joint_command("arm_gripper", 0.2)
        self.state(inner)  # fresh state without joint readings
        gate.send_joint_command("arm_gripper", 0.8)
        self.assertAlmostEqual(inner.calls[-1][2], 0.2 + OMNIBOT_PROFILE.max_joint_step)

    def test_joint_refusals(self):
        gate, inner = self.omni()  # no joint readings at all
        gate.send_joint_command("arm_gripper", 0.1)  # no reference
        gate.send_joint_command("tail", 0.1)  # unknown joint
        gate.send_joint_command("arm_gripper", float("inf"))
        no_limits = replace(OMNIBOT_PROFILE, joint_limits={})
        gate2, inner2 = self.omni(
            profile=no_limits, joints=[JointReading("arm_gripper", 0.0)]
        )
        gate2.send_joint_command("arm_gripper", 0.1)
        self.assertNotIn("joint", inner.names())
        self.assertNotIn("joint", inner2.names())
        self.assertEqual(
            [r["reason"] for r in self.events("reject")],
            ["no_joint_reference", "unknown_joint", "non_finite", "no_joint_limit"],
        )

    def test_joints_not_forwarded_when_unarmed_or_latched(self):
        gate, inner = self.omni(joints=[JointReading("arm_gripper", 0.0)], arm=False)
        gate.send_joint_command("arm_gripper", 0.1)
        self.assertTrue(gate.arm())
        gate.emergency_stop()
        gate.send_joint_command("arm_gripper", 0.1)
        self.assertNotIn("joint", inner.names())
        self.assertEqual(
            [r["reason"] for r in self.events("reject")], ["not_armed", "estop_latched"]
        )

    def test_joint_transport_error_latches(self):
        gate, inner = self.omni(joints=[JointReading("arm_gripper", 0.0)])
        inner.fail.add("joint")
        gate.send_joint_command("arm_gripper", 0.1)
        self.assertTrue(gate.latched)
        gate2, inner2 = self.omni()
        inner2.fail.add("effort")
        gate2.send_joint_effort("arm_gripper", 0.1)
        self.assertTrue(gate2.latched)

    def test_joint_stale_state_trips(self):
        gate, inner = self.omni(joints=[JointReading("arm_gripper", 0.0)])
        self.clock.advance(0.6)
        gate.send_joint_command("arm_gripper", 0.1)
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "watchdog_state")
        gate2, _ = self.omni()
        self.clock.advance(0.6)
        gate2.send_joint_effort("arm_gripper", 0.1)
        self.assertEqual(gate2.latch_reason, "watchdog_state")


# ── watchdogs ─────────────────────────────────────────────────────────────────
class TestWatchdogs(GateCase):
    def test_zero_dt_allows_no_acceleration(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        gate.send_velocity(Velocity(0.3))  # same instant as arming
        self.assertEqual(inner.velocities(), [ZERO])

    def test_command_watchdog_zeroes_after_300ms(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        self.clock.advance(0.01)
        gate.send_velocity(Velocity(0.3))
        self.clock.advance(0.29)
        gate.tick()
        self.assertEqual(inner.velocities(), [(0.3, 0.0, 0.0)])
        self.clock.advance(0.02)
        gate.tick()
        self.assertEqual(inner.velocities()[-1], ZERO)
        self.assertFalse(gate.latched)
        self.assertEqual(self.events("watchdog_trip")[0]["kind"], "command")
        gate.tick()  # does not spam once stopped
        self.assertEqual(len(inner.velocities()), 2)

    def test_fresh_command_resets_command_watchdog(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        for _ in range(3):
            self.clock.advance(0.2)
            self.state(inner)
            gate.send_velocity(Velocity(0.3))
            gate.tick()
        self.assertNotIn(ZERO, inner.velocities())

    def test_idle_robot_does_not_trip_command_watchdog(self):
        gate, inner = self.make()
        self.clock.advance(0.4)
        gate.tick()
        self.assertEqual(inner.calls, [])

    def test_state_watchdog_stops_and_latches_after_500ms(self):
        gate, inner = self.make()
        self.clock.advance(0.49)
        gate.tick()
        self.assertFalse(gate.latched)
        self.clock.advance(0.02)
        gate.tick()
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "watchdog_state")
        self.assertIn("estop", inner.names())
        self.assertEqual(self.events("watchdog_trip")[-1]["kind"], "state")
        self.assertEqual(self.events("estop")[-1]["reason"], "watchdog_state")
        self.assertFalse(gate.armed)

    def test_stale_state_trips_on_command(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        self.clock.advance(0.6)
        gate.send_velocity(Velocity(0.2))
        self.assertTrue(gate.latched)
        self.assertNotIn((0.2, 0.0, 0.0), inner.velocities())

    def test_state_without_orientation_is_not_fresh_for_go2(self):
        gate, inner = self.make()
        self.clock.advance(0.3)
        inner.emit(error_code=0)  # no roll/pitch
        self.clock.advance(0.3)
        gate.tick()
        self.assertTrue(gate.latched)

    def test_orientation_optional_for_wheeled_base(self):
        gate, inner = self.make(FakeTransport(), "omnibot")
        self.clock.advance(0.3)
        inner.emit()
        self.clock.advance(0.3)
        gate.tick()
        self.assertFalse(gate.latched)

    def test_tick_is_noop_when_disarmed_or_latched(self):
        gate, inner = self.make(arm=False)
        self.clock.advance(5.0)
        gate.tick()
        self.assertFalse(gate.latched)

    def test_watchdog_loop_ticks_until_stopped(self):
        gate, _ = self.make()

        class Ev:
            n = 0

            def wait(self, timeout):
                self.n += 1
                return self.n > 2

        with mock.patch.object(gate, "tick") as tick:
            gate._watchdog_loop(Ev())
        self.assertEqual(tick.call_count, 2)

    def test_watchdog_loop_error_latches(self):
        gate, inner = self.make()

        class Once:
            n = 0

            def wait(self, timeout):
                self.n += 1
                return self.n > 1

        with mock.patch.object(gate, "tick", side_effect=RuntimeError("boom")):
            gate._watchdog_loop(Once())
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "watchdog_error")

    def test_watchdog_thread_lifecycle(self):
        gate, inner = self.make(auto_tick=True, tick_period_s=0.001)
        self.assertTrue(gate._thread is not None and gate._thread.is_alive())
        gate.connect()  # idempotent: no second thread
        gate.disconnect()
        self.assertIsNone(gate._thread)
        self.assertIn("disconnect", inner.names())


# ── e-stop latch, damp, release ───────────────────────────────────────────────
class TestEstopLatch(GateCase):
    def test_estop_latches_and_blocks_motion(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        gate.emergency_stop()
        self.assertTrue(gate.latched)
        self.assertFalse(gate.armed)
        self.assertEqual(inner.names()[0], "estop")
        gate.send_velocity(Velocity(0.3))
        self.assertEqual(inner.velocities()[-1], ZERO)
        self.assertEqual(self.events("reject")[-1]["reason"], "estop_latched")
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "estop_latched")
        self.assertEqual(self.events("estop")[-1]["reason"], "operator")
        self.assertTrue(gate.status().label.endswith("safety: latched (operator)"))

    def test_zero_velocity_always_passes(self):
        gate, inner = self.make(arm=False)
        gate.send_velocity(Velocity())
        gate.emergency_stop()
        gate.send_velocity(Velocity())
        self.assertEqual(inner.velocities(), [ZERO, ZERO])
        self.assertEqual(self.events("reject"), [])

    def test_estop_never_damps_a_standing_robot(self):
        gate, inner = self.make()
        gate.emergency_stop()
        self.assertNotIn(("sport", "damp"), inner.calls)

    def test_estop_damps_only_when_lying_down(self):
        gate, inner = self.make()
        self.state(inner, body_height=0.08)
        inner.calls.clear()
        gate.emergency_stop()
        self.assertEqual(inner.calls[:2], [("estop",), ("sport", "damp")])
        self.assertEqual(self.events("damp")[-1]["why"], "lying_down")

    def test_estop_damp_flag_forces_damp_after_stop(self):
        gate, inner = self.make(estop_damp=True)
        self.assertTrue(gate.profile.estop_damp)
        gate.emergency_stop()
        self.assertEqual(inner.calls[:2], [("estop",), ("sport", "damp")])
        self.assertEqual(self.events("damp")[-1]["why"], "estop_damp")

    def test_unknown_posture_never_damps(self):
        gate, inner = self.make(FakeGo2(), profile=replace(GO2_PROFILE))
        inner.emit(roll=0.0, pitch=0.0)  # no body_height
        inner.calls.clear()
        gate.emergency_stop()
        self.assertNotIn(("sport", "damp"), inner.calls)
        g2, i2 = self.make(FakeTransport(), "omnibot")
        i2.emit(body_height=0.05)  # lying but no sport_command → nothing to damp
        g2.emergency_stop()
        self.assertEqual(i2.names(), ["estop"])

    def test_estop_falls_back_to_zero_velocity_when_inner_stop_fails(self):
        gate, inner = self.make()
        inner.fail.add("estop")
        gate.emergency_stop()
        self.assertTrue(gate.latched)
        self.assertEqual(inner.velocities(), [ZERO])
        self.assertTrue(self.events("estop")[-1]["inner_error"])
        inner.fail = {"estop", "vel", "sport"}
        self.state(inner, body_height=0.05)
        gate.emergency_stop()  # must not raise even if everything fails
        self.assertTrue(gate.latched)

    def test_release_requires_human_tty_then_rearm(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        gate.emergency_stop()
        self.tty = False
        self.assertFalse(gate.release_stop())
        self.assertTrue(gate.latched)
        self.assertEqual(self.events("release_refused")[-1]["reason"], "no_tty")
        self.tty = True
        self.answer = "yes"
        self.assertFalse(gate.release_stop())
        self.assertEqual(self.events("release_refused")[-1]["reason"], "not_confirmed")
        self.answer = EOFError()
        self.assertFalse(gate.release_stop())
        self.answer = RELEASE_PHRASE
        self.assertTrue(gate.release_stop())
        self.assertFalse(gate.latched)
        self.assertFalse(gate.armed)  # reset requires re-arming
        self.assertIn("release", inner.names())
        gate.send_velocity(Velocity(0.2))
        self.assertEqual(self.events("reject")[-1]["reason"], "not_armed")
        self.answer = ARM_PHRASE
        self.state(inner)
        self.assertTrue(gate.arm())
        self.clock.advance(0.01)
        gate.send_velocity(Velocity(0.2))
        self.assertEqual(inner.velocities()[-1], (0.2, 0.0, 0.0))

    def test_release_refused_in_ci_or_while_tilted(self):
        gate, inner = self.make()
        self.state(inner, roll=0.9)
        self.assertTrue(gate.latched)
        self.answer = RELEASE_PHRASE
        self.assertFalse(gate.release_stop())
        self.assertEqual(self.events("release_refused")[-1]["reason"], "unsafe_state")
        self.state(inner)
        self.env["CI"] = "true"
        self.assertFalse(gate.release_stop())
        self.assertEqual(self.events("release_refused")[-1]["reason"], "ci")
        del self.env["CI"]
        inner.fail.add("release")
        self.assertFalse(gate.release_stop())
        self.assertTrue(gate.latched)
        self.assertEqual(
            self.events("release_refused")[-1]["reason"], "transport_error"
        )

    def test_release_when_not_latched_is_noop(self):
        gate, inner = self.make()
        self.assertTrue(gate.release_stop())
        self.assertEqual(len(self.prompts), 1)  # only the arming prompt
        self.assertTrue(gate.armed)

    def test_release_without_confirmation_for_loopback_sim(self):
        gate, inner = self.make(FakeGo2("lo", 1))
        gate.emergency_stop()
        self.tty = False
        self.assertTrue(gate.release_stop())


# ── allowlist ─────────────────────────────────────────────────────────────────
class TestAllowlist(GateCase):
    DENIED = [
        "front_flip",
        "BackFlip",
        "LeftFlip",
        "front_jump",
        "FrontPounce",
        "dance1",
        "Dance2",
        "handstand",
        "Handstand",
        "walk_upright",
        "WalkUpright",
        "free_walk",
        "FreeBound",
        "FreeJump",
        "FreeAvoid",
        "cross_step",
        "CrossStep",
    ]

    def test_normalize_action(self):
        self.assertEqual(normalize_action("StandUp"), "stand_up")
        self.assertEqual(normalize_action(" rise-sit "), "rise_sit")
        self.assertEqual(normalize_action("BalanceStand"), "balance_stand")
        self.assertEqual(normalize_action("stop move"), "stop_move")

    def test_dangerous_actions_always_refused_and_logged(self):
        gate, inner = self.make()
        with self.assertLogs("ohho.safety", level="WARNING") as logs:
            for name in self.DENIED:
                with self.subTest(name=name):
                    self.assertTrue(is_denied(normalize_action(name)))
                    self.assertFalse(gate.command(name))
        self.assertNotIn("sport", inner.names())
        rej = self.events("reject")
        self.assertEqual(len(rej), len(self.DENIED))
        self.assertTrue(all(r["reason"] == "denied" for r in rej))
        self.assertTrue(any("front_flip" in m for m in logs.output))

    def test_unknown_actions_refused(self):
        gate, inner = self.make()
        for name in ("damp", "hello", "stretch", "scrape", "", 42, None):
            self.assertFalse(gate.command(name))
        self.assertNotIn("sport", inner.names())
        reasons = {r["reason"] for r in self.events("reject")}
        self.assertEqual(reasons, {"not_allowlisted", "invalid_action"})

    def test_allowlisted_actions_forwarded(self):
        gate, inner = self.make()
        for name in ("stand_up", "StandDown", "balance_stand", "RecoveryStand"):
            self.assertTrue(gate.command(name))
        for name in ("sit", "rise_sit"):
            self.assertTrue(gate.command(name))
        sports = [c[1] for c in inner.calls if c[0] == "sport"]
        self.assertEqual(
            sports,
            [
                "stand_up",
                "stand_down",
                "balance_stand",
                "recovery_stand",
                "sit",
                "rise_sit",
            ],
        )

    def test_euler_clamped(self):
        gate, inner = self.make()
        self.assertTrue(gate.command("euler", 0.9, -0.5, 0.1))
        self.assertEqual(inner.calls[-1], ("sport", "euler", 0.3, -0.3, 0.1))
        self.assertEqual(self.events("clamp")[-1]["command"], "euler")
        self.assertTrue(gate.command("euler", 0.1, 0.0, 0.0))
        self.assertEqual(inner.calls[-1], ("sport", "euler", 0.1, 0.0, 0.0))

    def test_move_goes_through_velocity_caps(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        self.clock.advance(0.01)
        self.assertTrue(gate.command("move", 2.0, 0.0, 0.0))
        self.assertEqual(inner.velocities()[-1], (0.5, 0.0, 0.0))
        self.assertTrue(gate.command("Move", 0.1, 0.0, 0.0))
        self.assertFalse(gate.command("move", float("nan"), 0.0, 0.0))

    def test_bad_arguments_refused(self):
        gate, inner = self.make()
        self.assertFalse(gate.command("move", 1.0))
        self.assertFalse(gate.command("euler", "a", 0.0, 0.0))
        self.assertFalse(gate.command("euler", float("inf"), 0.0, 0.0))
        self.assertFalse(gate.command("stand_up", 1.0))
        self.assertEqual(
            {r["reason"] for r in self.events("reject")}, {"bad_arguments"}
        )

    def test_stop_move_always_allowed(self):
        gate, inner = self.make(arm=False)
        self.assertTrue(gate.command("stop_move"))
        gate.emergency_stop()
        inner.calls.clear()
        self.assertTrue(gate.command("StopMove"))
        self.assertEqual(inner.calls, [("vel", ZERO), ("sport", "stop_move")])
        self.assertFalse(gate.command("stop_move", 1.0))

    def test_stop_move_on_transport_without_sport(self):
        gate, inner = self.make(FakeTransport(), "omnibot")
        self.assertTrue(gate.command("stop_move"))
        self.assertEqual(inner.calls, [("vel", ZERO)])
        inner.fail.add("vel")
        self.assertTrue(gate.command("stop_move"))  # best effort, still True

    def test_motion_actions_need_arming_and_no_latch(self):
        gate, inner = self.make(arm=False)
        self.assertFalse(gate.command("stand_up"))
        self.assertTrue(gate.arm())
        gate.emergency_stop()
        self.assertFalse(gate.command("stand_up"))
        self.assertEqual(
            [r["reason"] for r in self.events("reject")], ["not_armed", "estop_latched"]
        )

    def test_action_stale_state_trips(self):
        gate, inner = self.make()
        self.clock.advance(0.6)
        self.assertFalse(gate.command("stand_up"))
        self.assertEqual(gate.latch_reason, "watchdog_state")

    def test_action_unsupported_or_failing_transport(self):
        gate, inner = self.make(FakeTransport(), "omnibot", profile=GO2_PROFILE)
        self.assertFalse(gate.command("stand_up"))
        self.assertEqual(
            self.events("reject")[-1]["reason"], "unsupported_by_transport"
        )
        gate2, inner2 = self.make()
        inner2.fail.add("sport")
        self.assertFalse(gate2.command("stand_up"))
        self.assertTrue(gate2.latched)
        self.assertEqual(gate2.latch_reason, "transport_error")

    def test_narrowed_allowlist(self):
        prof = replace(GO2_PROFILE, allowlist=frozenset({"stop_move"}))
        gate, inner = self.make(profile=prof)
        self.assertFalse(gate.command("stand_up"))
        self.assertEqual(self.events("reject")[-1]["reason"], "not_allowlisted")

    def test_no_passthrough_of_unknown_attributes(self):
        gate, inner = self.make()
        self.assertFalse(hasattr(gate, "sport_command"))
        self.assertFalse(hasattr(gate, "send_motor"))


# ── tilt / fall / fault guard ─────────────────────────────────────────────────
class TestTiltGuard(GateCase):
    def test_roll_or_pitch_over_limit_latches(self):
        for key, val in (("roll", 0.61), ("roll", -0.61), ("pitch", 0.7)):
            with self.subTest(key=key, val=val):
                gate, inner = self.make()
                self.state(inner, **{key: val})
                self.assertTrue(gate.latched)
                self.assertEqual(gate.latch_reason, "tilt")
                self.assertIn("estop", inner.names())
                self.assertEqual(self.events("estop")[-1]["reason"], "tilt")

    def test_within_limits_is_fine(self):
        gate, inner = self.make()
        self.state(inner, roll=0.59, pitch=-0.59)
        self.assertFalse(gate.latched)

    def test_imu_rpy_list_form(self):
        gate, inner = self.make()
        inner.emit(imu_rpy=[0.0, 0.65, 1.0], error_code=0)
        self.assertTrue(gate.latched)
        gate2, inner2 = self.make()
        inner2.emit(imu_rpy=[0.1, 0.1, 3.0])
        self.assertFalse(gate2.latched)
        gate3, inner3 = self.make()
        inner3.emit(imu_rpy="bogus")  # unusable orientation → not fresh
        self.clock.advance(0.6)
        gate3.tick()
        self.assertTrue(gate3.latched)

    def test_non_finite_orientation_latches(self):
        gate, inner = self.make()
        self.state(inner, roll=float("nan"))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "tilt")

    def test_error_code_latches(self):
        gate, inner = self.make()
        self.state(inner, error_code=3)
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "fault")
        self.assertEqual(self.events("estop")[-1]["error_code"], 3)
        gate2, inner2 = self.make()
        self.state(inner2, error_code="bad")
        self.assertEqual(gate2.latch_reason, "fault")

    def test_trip_latches_even_when_disarmed(self):
        gate, inner = self.make(arm=False)
        self.state(inner, pitch=1.2)
        self.assertTrue(gate.latched)

    def test_already_latched_does_not_re_trip(self):
        gate, inner = self.make()
        self.state(inner, roll=0.9)
        n = len(self.events("estop"))
        self.state(inner, roll=0.9)
        self.assertEqual(len(self.events("estop")), n)

    def test_telemetry_forwarded_to_subscribers(self):
        gate, inner = self.make()
        seen = []
        unsub = gate.on_telemetry(seen.append)
        t = self.state(inner)
        self.assertIs(seen[-1], t)
        unsub()
        self.assertIs(gate.read(), t)


# ── hardware arming ───────────────────────────────────────────────────────────
class TestArming(GateCase):
    def test_arms_with_env_tty_and_phrase(self):
        gate, inner = self.make(arm=False)
        self.assertFalse(gate.armed)
        self.assertTrue(gate.arm())
        self.assertTrue(gate.armed)
        self.assertIn(ARM_PHRASE, self.prompts[-1])
        rec = self.events("arm")[-1]
        self.assertTrue(rec["hardware"])
        self.assertIn("ts", rec)
        self.assertEqual(rec["t"], WALL)
        self.assertEqual(rec["robot"], "unitree-go2")
        self.assertTrue(gate.arm())  # already armed: no second prompt
        self.assertEqual(len(self.prompts), 1)
        self.assertTrue(gate.status().label.endswith("safety: armed"))

    def test_phrase_compare_tolerates_case_and_spacing(self):
        gate, _ = self.make(arm=False)
        self.answer = (
            "  i am physically present,  the area is clear, the remote is in my hand. "
        )
        self.assertTrue(gate.arm())

    def test_refused_without_tty(self):
        gate, inner = self.make(arm=False)
        self.tty = False
        self.assertFalse(gate.arm())
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "no_tty")
        gate.send_velocity(Velocity(0.2))
        self.assertEqual(inner.velocities(), [ZERO])

    def test_refused_without_env(self):
        for env in ({}, {ARM_ENV: "0"}, {ARM_ENV: "true"}):
            with self.subTest(env=env):
                self.env = dict(env)
                gate, _ = self.make(arm=False)
                self.assertFalse(gate.arm())
                self.assertEqual(self.events("arm_refused")[-1]["reason"], "env")
        self.assertEqual(self.prompts, [])

    def test_refused_in_ci(self):
        for var in ("CI", "GITHUB_ACTIONS"):
            with self.subTest(var=var):
                self.env = {ARM_ENV: "1", var: "true"}
                gate, _ = self.make(arm=False)
                self.assertFalse(gate.arm())
                self.assertEqual(self.events("arm_refused")[-1]["reason"], "ci")
        self.env = {ARM_ENV: "1", "CI": "false"}
        gate, _ = self.make(arm=False)
        self.assertTrue(gate.arm())

    def test_refused_on_wrong_phrase_or_input_error(self):
        gate, _ = self.make(arm=False)
        for answer in ("yes", "", KeyboardInterrupt(), EOFError(), OSError()):
            self.answer = answer
            self.assertFalse(gate.arm())
        reasons = [r["reason"] for r in self.events("arm_refused")]
        self.assertEqual(reasons, ["not_confirmed"] * 5)

    def test_refused_when_not_connected_or_state_stale(self):
        inner = FakeGo2()
        gate = SafetyGate(
            inner,
            get_spec("unitree-go2"),
            clock=self.clock,
            audit=AuditLog(self.tmp, wall_clock=lambda: WALL),
            environ=self.env,
            isatty=lambda: True,
            input_fn=self._input,
            auto_tick=False,
        )
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "not_connected")
        gate.connect()
        self.assertFalse(gate.arm())  # never received state
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "no_fresh_state")
        self.state(inner)
        self.clock.advance(0.6)
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "no_fresh_state")

    def test_refused_if_latched_during_prompt(self):
        gate, _ = self.make(arm=False)

        def trip_then_answer():
            gate.emergency_stop()
            return ARM_PHRASE

        self.answer = trip_then_answer
        self.assertFalse(gate.arm())
        self.assertEqual(self.events("arm_refused")[-1]["reason"], "estop_latched")

    def test_refused_when_audit_unwritable(self):
        audit = FailingAudit()
        gate, _ = self.make(arm=False, audit=audit)
        audit.fail = True
        self.assertFalse(gate.arm())
        self.assertFalse(gate.armed)

    def test_audit_failure_while_armed_disarms(self):
        audit = FailingAudit()
        gate, inner = self.make(profile=GO2_PROFILE, audit=audit)
        audit.fail = True
        self.clock.advance(0.1)
        gate.send_velocity(Velocity(0.5))  # accel clamp → audit → fails
        self.assertFalse(gate.armed)
        self.assertEqual(inner.velocities()[-1], ZERO)

    def test_disarm_zeroes_and_audits(self):
        gate, inner = self.make()
        gate.disarm()
        self.assertFalse(gate.armed)
        self.assertEqual(inner.velocities(), [ZERO])
        self.assertEqual(self.events("disarm")[-1]["reason"], "operator")
        inner.fail.add("vel")
        gate.disarm()  # best effort, never raises

    def test_requires_hardware_arming_rules(self):
        cases = [
            (FakeTransport("simulated"), False),
            (FakeGo2("lo", 1), False),
            (FakeGo2("127.0.0.1", 2), False),
            (FakeGo2("lo0", 1), False),
            (FakeGo2("Loopback Pseudo-Interface 1", 1), False),
            (FakeGo2("lo", 0), True),  # DDS domain 0 is the real robot domain
            (FakeGo2("eth0", 1), True),
            (FakeGo2("", 1), True),
            (FakeGo2("lo", "1"), True),  # malformed domain: fail closed
            (FakeTransport("serial"), True),
            (FakeTransport("ros2"), True),
            (FakeTransport("composite"), True),
        ]
        for tp, expected in cases:
            with self.subTest(tp=tp.protocol, iface=getattr(tp, "iface", None)):
                self.assertEqual(requires_hardware_arming(tp), expected)

    def test_loopback_sim_arms_without_tty(self):
        self.tty = False
        self.env = {}
        gate, _ = self.make(FakeGo2("lo", 1), arm=False)
        self.assertFalse(gate.requires_hardware_arming)
        self.assertTrue(gate.arm())
        self.assertFalse(self.events("arm")[-1]["hardware"])

    def test_default_io_without_env_refuses(self):
        inner = FakeGo2()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ARM_ENV, None)
            gate = SafetyGate(
                inner,
                get_spec("unitree-go2"),
                clock=self.clock,
                audit=AuditLog(self.tmp),
                auto_tick=False,
            )
            gate.connect()
            self.state(inner)
            with mock.patch("builtins.input") as inp:
                self.assertFalse(gate.arm())
                inp.assert_not_called()

    def test_default_io_without_tty_refuses(self):
        inner = FakeGo2()
        with mock.patch.dict(os.environ, {ARM_ENV: "1"}):
            for var in safety.CI_ENV_VARS:
                os.environ.pop(var, None)
            gate = SafetyGate(
                inner,
                get_spec("unitree-go2"),
                clock=self.clock,
                audit=AuditLog(self.tmp),
                auto_tick=False,
            )
            gate.connect()
            self.state(inner)
            with mock.patch.object(sys, "stdin", io.StringIO("")):
                self.assertFalse(gate.arm())
            with mock.patch.object(sys, "stdin", None):
                self.assertFalse(gate.arm())
            fake_tty = mock.Mock()
            fake_tty.isatty.return_value = True
            with (
                mock.patch.object(sys, "stdin", fake_tty),
                mock.patch("builtins.input", return_value=ARM_PHRASE),
            ):
                self.assertTrue(gate.arm())


# ── audit log ─────────────────────────────────────────────────────────────────
class TestAuditLog(GateCase):
    def test_default_dir_and_daily_file(self):
        self.assertEqual(safety.default_audit_dir({}), Path.home() / ".ohho" / "safety")
        self.assertEqual(
            safety.default_audit_dir({safety.AUDIT_DIR_ENV: self.tmp}), Path(self.tmp)
        )
        log = AuditLog(self.tmp, wall_clock=lambda: WALL)
        day = datetime.fromtimestamp(WALL, timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(log.path_for(WALL), Path(self.tmp) / f"{day}.jsonl")
        with mock.patch.dict(os.environ, {safety.AUDIT_DIR_ENV: self.tmp}):
            self.assertEqual(AuditLog().directory, Path(self.tmp))

    def test_records_are_jsonl_with_timestamps(self):
        log = AuditLog(self.tmp, wall_clock=lambda: WALL)
        self.assertTrue(log.write("reject", reason="denied", value=float("nan")))
        self.assertTrue(log.write("arm", robot="x"))
        lines = log.path_for(WALL).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        rec = json.loads(lines[0])
        self.assertEqual(rec["event"], "reject")
        self.assertEqual(rec["t"], WALL)
        self.assertTrue(rec["ts"].endswith("+00:00"))
        self.assertIsNone(rec["value"])  # non-finite floats are not valid JSON

    def test_write_failure_returns_false(self):
        blocker = Path(self.tmp) / "file"
        blocker.write_text("x", encoding="utf-8")
        log = AuditLog(blocker / "sub", wall_clock=lambda: WALL)
        self.assertFalse(log.write("arm"))

    def test_gate_audits_every_refusal_and_clamp(self):
        gate, inner = self.make(profile=fast(GO2_PROFILE))
        gate.send_velocity(Velocity(3.0))
        gate.command("front_flip")
        gate.emergency_stop()
        gate.send_velocity(Velocity(0.1))
        kinds = [r["event"] for r in self.events()]
        for k in ("arm", "clamp", "reject", "estop"):
            self.assertIn(k, kinds)
        for rec in self.events():
            self.assertEqual(rec["protocol"], "dds")
            self.assertIn("ts", rec)


# ── wrapper plumbing + Robot integration ──────────────────────────────────────
class TestGatePlumbing(GateCase):
    def test_refuses_to_double_wrap(self):
        gate, _ = self.make(arm=False)
        with self.assertRaises(ValueError):
            SafetyGate(gate)

    def test_protocol_status_and_label(self):
        gate, inner = self.make(arm=False)
        self.assertEqual(gate.protocol, "dds")
        s = gate.status()
        self.assertEqual(s.state, ConnectionState.CONNECTED)
        self.assertEqual(s.label, "fake · safety: disarmed")
        self.assertEqual(self.events("connect")[-1]["hardware"], True)

    def test_status_forwarded(self):
        gate, inner = self.make(arm=False)
        seen = []
        gate.on_status(seen.append)
        inner._emit_status(inner.status())
        self.assertEqual(seen[-1].protocol, "dds")

    def test_disconnect_zeroes_disarms_and_closes(self):
        gate, inner = self.make()
        gate.disconnect()
        self.assertFalse(gate.armed)
        self.assertEqual(inner.calls[0], ("vel", ZERO))
        self.assertEqual(inner.calls[-1], ("disconnect",))
        self.assertEqual(self.events("disarm")[-1]["reason"], "disconnect")
        g2, i2 = self.make(arm=False)
        i2.fail.add("vel")
        g2.disconnect()  # never raises on the way out

    def test_no_spec_uses_conservative_defaults(self):
        gate, inner = self.make(FakeTransport(), spec=None, arm=False)
        self.assertEqual(gate.profile, SafetyProfile())
        self.assertEqual(self.events("connect")[-1]["robot"], "unknown")

    def test_robot_over_gated_sim_physically_clamps(self):
        spec = get_spec("unitree-go2")
        sim = SimTransport(spec)
        prof = fast(GO2_PROFILE, require_orientation=False)
        gate = SafetyGate(
            sim,
            spec,
            profile=prof,
            clock=self.clock,
            audit=AuditLog(self.tmp, wall_clock=lambda: WALL),
            environ={},
            isatty=lambda: False,
            auto_tick=False,
        )
        bot = Robot(spec, gate, NativeRuntime())
        gate.connect()
        self.assertTrue(bot.gated)
        sim._emit_telemetry(sim.read())
        self.assertTrue(bot.arm())  # simulated link: no TTY needed
        self.clock.advance(0.01)
        bot.drive(vx=2.0)
        sim.step(1.0)
        self.assertAlmostEqual(sim.read().odom.x, 0.5, places=5)
        bot.disarm()
        self.assertFalse(gate.armed)
        bot.disconnect()

    def test_robot_connect_bypasses_gate_for_sim(self):
        bot = Robot.connect("sim", "sim://", runtime="native")
        try:
            self.assertNotIsInstance(bot.transport, SafetyGate)
            self.assertFalse(bot.gated)
            self.assertTrue(bot.arm())
            bot.disarm()
        finally:
            bot.disconnect()

    def test_robot_connect_inserts_gate_for_non_sim(self):
        inner = FakeGo2()
        cfg = Path(self.tmp) / "cfg.json"
        cfg.write_text(json.dumps({"hard_ceiling": {"max_lin": 1.0}}), "utf-8")
        with (
            mock.patch("ohho.robot.resolve_transport", return_value=inner),
            mock.patch.dict(os.environ, {safety.AUDIT_DIR_ENV: self.tmp}),
        ):
            bot = Robot.connect(
                "unitree-go2",
                "dds://eth0",
                runtime="native",
                safety_config=str(cfg),
                estop_damp=True,
            )
        try:
            gate = bot.transport
            self.assertIsInstance(gate, SafetyGate)
            self.assertTrue(bot.gated)
            self.assertTrue(gate.profile.estop_damp)
            self.assertFalse(gate.armed)  # never auto-armed
            bot.drive(vx=0.4)
            self.assertEqual(inner.velocities(), [ZERO])  # refused, zero sent
            self.assertEqual(self.events("reject")[-1]["reason"], "not_armed")
        finally:
            bot.disconnect()


class TestFailClosedEdges(GateCase):
    def test_unauditable_clamps_are_not_forwarded(self):
        audit = FailingAudit()
        prof = replace(OMNIBOT_PROFILE, effort_limits={"arm_gripper": 1.0})
        gate, inner = self.make(FakeEffortArm(), "omnibot", profile=prof, audit=audit)
        self.state(inner, joints=[JointReading("arm_gripper", 0.0)])
        audit.fail = True
        gate.send_joint_command("arm_gripper", 0.7)  # step clamp → audit fails
        self.assertTrue(self.make_rearmed(gate, audit))
        audit.fail = True
        gate.send_joint_effort("arm_gripper", 5.0)  # effort clamp → audit fails
        self.assertNotIn("joint", inner.names())
        self.assertNotIn("effort", inner.names())
        audit.fail = False
        g2, i2 = self.make(audit=audit)
        audit.fail = True
        self.assertFalse(g2.command("euler", 0.9, 0.0, 0.0))
        self.assertNotIn("sport", i2.names())

    def make_rearmed(self, gate, audit) -> bool:
        audit.fail = False
        return gate.arm()

    def test_stop_move_survives_sport_failure(self):
        gate, inner = self.make()
        inner.fail.add("sport")
        self.assertTrue(gate.command("stop_move"))
        self.assertEqual(inner.velocities()[-1], ZERO)

    def test_unparseable_orientation_trips(self):
        gate, inner = self.make()
        self.state(inner, roll="tilted?")
        self.assertEqual(gate.latch_reason, "tilt")

    def test_estop_before_any_state_never_damps(self):
        inner = FakeGo2()
        gate = SafetyGate(
            inner,
            get_spec("unitree-go2"),
            clock=self.clock,
            audit=AuditLog(self.tmp, wall_clock=lambda: WALL),
            environ=self.env,
            auto_tick=False,
        )
        gate.emergency_stop()
        self.assertEqual(inner.names(), ["estop"])

    def test_broken_stdin_is_not_a_tty(self):
        broken = mock.Mock()
        broken.isatty.side_effect = ValueError("I/O operation on closed file")
        with mock.patch.object(sys, "stdin", broken):
            self.assertFalse(safety._stdin_isatty())

    def test_operator_identity_falls_back(self):
        with mock.patch("getpass.getuser", side_effect=OSError("no user")):
            self.assertEqual(safety._operator()["user"], "unknown")

    def test_audit_serialises_nested_and_odd_values(self):
        log = AuditLog(self.tmp, wall_clock=lambda: WALL)
        self.assertTrue(log.write("x", nested={"a": (1.0, float("inf"))}, obj=object()))
        rec = self.events("x")[-1]
        self.assertEqual(rec["nested"], {"a": [1.0, None]})
        self.assertIsInstance(rec["obj"], str)


class TestAgentsCannotArmOrRelease(unittest.TestCase):
    def test_agent_tool_and_skill_sources_never_arm_or_release(self):
        root = Path(safety.__file__).resolve().parent
        for module in ("brains.py", "market.py", "agent.py"):
            src = (root / module).read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertNotIn("release_stop", src)
                self.assertNotIn(".arm(", src)


class TestCliSafetyFlags(unittest.TestCase):
    def test_flags_parse_and_reach_robot_connect(self):
        from ohho import cli

        args = cli.build_parser().parse_args(
            [
                "drive",
                "unitree-go2",
                "--transport",
                "dds://eth0",
                "--arm",
                "--estop-damp",
                "--safety-config",
                "s.json",
            ]
        )
        self.assertTrue(args.arm)
        self.assertTrue(args.estop_damp)
        self.assertEqual(args.safety_config, "s.json")
        fake = mock.Mock()
        fake.gated = True
        fake.arm.return_value = False
        with mock.patch.object(cli.Robot, "connect", return_value=fake) as conn:
            self.assertIs(cli._connect(args), fake)
        conn.assert_called_once_with(
            "unitree-go2",
            transport="dds://eth0",
            runtime="auto",
            safety_config="s.json",
            estop_damp=True,
        )
        fake.arm.assert_called_once_with()

    def test_no_arm_by_default(self):
        from ohho import cli

        args = cli.build_parser().parse_args(["connect", "omnibot"])
        self.assertFalse(args.arm)
        fake = mock.Mock()
        fake.gated = False
        with mock.patch.object(cli.Robot, "connect", return_value=fake):
            cli._connect(args)
        fake.arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
