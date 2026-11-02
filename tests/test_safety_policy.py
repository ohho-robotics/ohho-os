"""Per-class SafetyPolicy (OHH-313).

Existing safety tests are unchanged. These cover class selection, the
plugin tighten/loosen rule, and the additive audit field.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ohho.plugins import loaded_plugins
from ohho.robots.omnibot.policy import OMNIBOT_PROFILE
from ohho.registry import RobotSpec, get_spec
from ohho.robot_spec import RobotClass
from ohho.safety import (
    ARM_ENV,
    ARM_PHRASE,
    GO2_PROFILE,
    AuditLog,
    SafetyGate,
    SafetyProfile,
    profile_for,
    requires_hardware_arming,
)
from ohho.safety_policy import (
    ArmPolicy,
    GroundArmPolicy,
    GroundPolicy,
    LeggedPolicy,
    SafetyPolicyError,
    arming_uri,
    load_policy,
    policy_for_class,
)
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport
from ohho import safety_policy as policy_mod

WALL = 1_760_000_000.0


def setUpModule() -> None:
    logging.getLogger("ohho.safety").addHandler(logging.NullHandler())


class _Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _Transport(BaseTransport):
    def __init__(
        self, protocol: str = "dds", iface: str = "eth0", domain_id: object = 0
    ) -> None:
        super().__init__()
        self.protocol = protocol
        self.iface = iface
        self.domain_id = domain_id
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self.state = ConnectionState.IDLE

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

    def emergency_stop(self) -> None:
        self._rec("estop")

    def release_stop(self) -> None:
        self._rec("release")

    def sport_command(self, name: str, *args: object) -> None:
        self._rec("sport", name, *args)


class _Plugin:
    def __init__(self, policy: object) -> None:
        self.policy = policy

    def safety_policy(self, spec: RobotSpec) -> object:
        del spec
        return self.policy


@contextmanager
def _installed(plugin: object, name: str = "builtin"):
    """Swap the cached plugin instance. Discovery itself is not re-run."""
    import ohho.plugins as plugins

    old = loaded_plugins()
    updated = []
    found = False
    for info in old:
        if info.name == name and info.error is None:
            updated.append(dataclasses.replace(info, plugin=plugin))
            found = True
        else:
            updated.append(info)
    if not found:
        raise AssertionError(f"no loaded plugin named {name!r}")
    plugins._CACHE = tuple(updated)
    try:
        yield
    finally:
        plugins._CACHE = old


def _arm_spec(robot_id: str = "so101") -> RobotSpec:
    return RobotSpec(
        id=robot_id,
        name="SO-101",
        category="arm",
        capabilities=("manipulation",),
        robot_class=RobotClass.ARM,
        source="",
    )


class TestPolicyByClass(unittest.TestCase):
    def test_mobile_base_uses_ground(self) -> None:
        spec = get_spec("sim")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_BASE)
        policy = load_policy(spec)
        self.assertIsInstance(policy, GroundPolicy)
        self.assertNotIsInstance(policy, GroundArmPolicy)
        self.assertEqual(policy.name, "ground")
        self.assertEqual(policy.profile, profile_for("sim", spec))

    def test_mobile_manipulator_uses_ground_and_arm(self) -> None:
        spec = get_spec("omnibot")
        self.assertEqual(spec.robot_class, RobotClass.MOBILE_MANIPULATOR)
        policy = load_policy(spec)
        self.assertIsInstance(policy, GroundArmPolicy)
        self.assertIsInstance(policy.ground, GroundPolicy)
        self.assertIsInstance(policy.arm, ArmPolicy)
        self.assertEqual(policy.name, "ground+arm")
        self.assertEqual(policy.allowlist, frozenset({"move", "stop_move"}))
        self.assertEqual(policy.profile, OMNIBOT_PROFILE)
        self.assertIn("flip", policy.denylist)
        self.assertTrue(policy.denied("front_flip"))
        self.assertFalse(policy.denied("move"))

    def test_arm_uses_arm_policy(self) -> None:
        spec = _arm_spec()
        self.assertEqual(spec.robot_class, RobotClass.ARM)
        policy = load_policy(spec)
        self.assertIsInstance(policy, ArmPolicy)
        self.assertEqual(policy.name, "arm")
        self.assertEqual(policy.profile, profile_for(spec.id, spec))
        self.assertLessEqual(policy.profile.max_vx, 0.3)

    def test_quadruped_uses_legged(self) -> None:
        spec = get_spec("unitree-go2")
        self.assertEqual(spec.robot_class, RobotClass.QUADRUPED)
        policy = load_policy(spec)
        self.assertIsInstance(policy, LeggedPolicy)
        self.assertEqual(policy.name, "legged")
        self.assertEqual(policy.profile, GO2_PROFILE)
        self.assertIn("stand_up", policy.allowlist)
        self.assertTrue(policy.profile.require_orientation)

    def test_other_quadruped_keeps_conservative_caps(self) -> None:
        spec = RobotSpec(
            id="other-quad",
            name="Other",
            category="legged",
            capabilities=("base.drive",),
            max_lin=0.2,
            max_ang=0.4,
            robot_class=RobotClass.QUADRUPED,
            source="",
        )
        policy = load_policy(spec)
        self.assertIsInstance(policy, LeggedPolicy)
        self.assertEqual(policy.profile, profile_for(spec.id, spec))
        self.assertLessEqual(policy.profile.max_vx, 0.2)
        self.assertEqual(policy.profile.max_vy, 0.0)
        self.assertNotEqual(policy.profile, GO2_PROFILE)

    def test_aerial_has_no_aerial_policy_yet(self) -> None:
        self.assertFalse(hasattr(policy_mod, "AerialPolicy"))
        spec = RobotSpec(
            id="px4-sitl",
            name="PX4",
            category="aerial",
            robot_class=RobotClass.AERIAL,
            source="",
        )
        policy = policy_for_class(spec)
        self.assertIsInstance(policy, GroundPolicy)
        self.assertEqual(policy.profile, profile_for(spec.id, spec))

    def test_humanoid_has_no_humanoid_policy_yet(self) -> None:
        self.assertFalse(hasattr(policy_mod, "HumanoidPolicy"))
        spec = RobotSpec(
            id="walker",
            name="Walker",
            category="humanoid",
            robot_class=RobotClass.HUMANOID,
            source="",
        )
        policy = policy_for_class(spec)
        self.assertIsInstance(policy, GroundPolicy)
        self.assertEqual(policy.name, "ground")


class TestPluginCaps(unittest.TestCase):
    def test_loosening_velocity_is_refused_at_load(self) -> None:
        spec = get_spec("unitree-go2")
        loose = LeggedPolicy(dataclasses.replace(GO2_PROFILE, max_vx=0.8))
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(loose))
        message = str(raised.exception)
        self.assertIn("refused at load", message)
        self.assertIn("max_vx", message)
        self.assertIn("unitree-go2", message)

    def test_disabling_orientation_requirement_is_refused(self) -> None:
        spec = get_spec("unitree-go2")
        loose = LeggedPolicy(
            dataclasses.replace(GO2_PROFILE, require_orientation=False)
        )
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(loose))
        self.assertIn("require_orientation", str(raised.exception))

    def test_loosening_allowlist_is_refused(self) -> None:
        spec = get_spec("omnibot")
        wider = GroundArmPolicy(
            dataclasses.replace(
                OMNIBOT_PROFILE,
                allowlist=frozenset({"move", "stop_move", "stand_up"}),
            )
        )
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(wider))
        self.assertIn("allowlist", str(raised.exception))

    def test_loosening_joint_range_is_refused(self) -> None:
        spec = get_spec("omnibot")
        joints = dict(OMNIBOT_PROFILE.joint_limits)
        joints["arm_gripper"] = (-1.0, 1.0)
        wider = GroundArmPolicy(
            dataclasses.replace(OMNIBOT_PROFILE, joint_limits=joints)
        )
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(wider))
        self.assertIn("arm_gripper", str(raised.exception))

    def test_new_effort_limit_is_refused(self) -> None:
        spec = _arm_spec()
        extra = ArmPolicy(SafetyProfile(effort_limits={"j1": 1.0}))
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(extra))
        self.assertIn("effort_limits", str(raised.exception))

    def test_non_policy_is_refused(self) -> None:
        spec = get_spec("sim")
        with self.assertRaises(SafetyPolicyError) as raised:
            load_policy(spec, plugin=_Plugin(object()))
        self.assertIn("refused at load", str(raised.exception))

    def test_gate_refuses_a_loosening_plugin_before_it_connects(self) -> None:
        spec = get_spec("unitree-go2")
        loose = LeggedPolicy(dataclasses.replace(GO2_PROFILE, max_vx=0.9))
        inner = _Transport()
        with _installed(_Plugin(loose)):
            with self.assertRaises(SafetyPolicyError):
                SafetyGate(inner, spec, auto_tick=False, environ={})
        self.assertEqual(inner.calls, [])

    def test_tighter_plugin_lowers_the_gate_cap(self) -> None:
        spec = get_spec("unitree-go2")
        tight = LeggedPolicy(dataclasses.replace(GO2_PROFILE, max_vx=0.25))
        tight.name = "legged-tight"
        with _installed(_Plugin(tight)):
            gate = SafetyGate(_Transport(), spec, auto_tick=False, environ={})
        self.assertEqual(gate.profile.max_vx, 0.25)
        self.assertEqual(gate.profile.max_vy, GO2_PROFILE.max_vy)
        self.assertEqual(gate._policy.name, "legged-tight")
        decision = gate._policy.clamp(
            "velocity",
            ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
            None,
        )
        self.assertLessEqual(decision.applied[0], 0.25)

    def test_explicit_profile_is_not_replaced_by_a_tighter_plugin(self) -> None:
        spec = get_spec("unitree-go2")
        tight = LeggedPolicy(dataclasses.replace(GO2_PROFILE, max_vx=0.25))
        with _installed(_Plugin(tight)):
            gate = SafetyGate(
                _Transport(),
                spec,
                profile=GO2_PROFILE,
                auto_tick=False,
                environ={},
            )
        self.assertEqual(gate.profile.max_vx, 0.5)

    def test_equal_plugin_cap_does_not_undo_a_config_raise(self) -> None:
        spec = get_spec("unitree-go2")
        same = LeggedPolicy(GO2_PROFILE)
        cfg_path = Path(tempfile.mkdtemp()) / "safety.json"
        self.addCleanup(shutil.rmtree, cfg_path.parent, True)
        cfg_path.write_text(
            json.dumps({"profiles": {"unitree-go2": {"max_vx": 0.8}}}),
            encoding="utf-8",
        )
        from ohho.safety import load_safety_config

        cfg = load_safety_config(cfg_path)
        with _installed(_Plugin(same)):
            gate = SafetyGate(
                _Transport(), spec, config=cfg, auto_tick=False, environ={}
            )
        self.assertEqual(gate.profile.max_vx, 0.8)

    def test_tighter_plugin_wins_over_a_config_raise(self) -> None:
        spec = get_spec("unitree-go2")
        tight = LeggedPolicy(dataclasses.replace(GO2_PROFILE, max_vx=0.4))
        cfg_path = Path(tempfile.mkdtemp()) / "safety.json"
        self.addCleanup(shutil.rmtree, cfg_path.parent, True)
        cfg_path.write_text(
            json.dumps({"profiles": {"unitree-go2": {"max_vx": 0.8}}}),
            encoding="utf-8",
        )
        from ohho.safety import load_safety_config

        cfg = load_safety_config(cfg_path)
        with _installed(_Plugin(tight)):
            gate = SafetyGate(
                _Transport(), spec, config=cfg, auto_tick=False, environ={}
            )
        self.assertEqual(gate.profile.max_vx, 0.4)

    def test_narrower_joint_limit_is_applied(self) -> None:
        spec = get_spec("omnibot")
        joints = dict(OMNIBOT_PROFILE.joint_limits)
        joints["arm_gripper"] = (0.0, 0.4)
        tight = GroundArmPolicy(
            dataclasses.replace(OMNIBOT_PROFILE, joint_limits=joints)
        )
        with _installed(_Plugin(tight), name="omnibot"):
            gate = SafetyGate(_Transport("serial"), spec, auto_tick=False, environ={})
        self.assertEqual(gate.profile.joint_limits["arm_gripper"], (0.0, 0.4))
        self.assertEqual(
            gate.profile.joint_limits["arm_shoulder_pan"],
            OMNIBOT_PROFILE.joint_limits["arm_shoulder_pan"],
        )
        fault = gate._policy.fall_or_tilt(
            Telemetry(custom={"roll": 0.0, "pitch": 0.3, "error_code": 0})
        )
        self.assertIsNone(fault)


class TestPolicyHooks(unittest.TestCase):
    def test_each_class_zeros_velocity_on_watchdog(self) -> None:
        for policy in (
            GroundPolicy(profile_for("sim", get_spec("sim"))),
            ArmPolicy(profile_for("so101", _arm_spec())),
            LeggedPolicy(GO2_PROFILE),
            GroundArmPolicy(OMNIBOT_PROFILE),
        ):
            with self.subTest(policy=policy.name):
                transport = _Transport()
                policy.on_watchdog(transport, Telemetry())
                self.assertEqual(transport.calls, [("vel", (0.0, 0.0, 0.0))])

    def test_estop_damps_only_when_lying_down(self) -> None:
        policy = LeggedPolicy(GO2_PROFILE)
        standing = _Transport()
        policy.on_estop(standing, Telemetry(custom={"body_height": 0.3}))
        self.assertEqual(standing.calls, [("estop",)])
        lying = _Transport()
        action = policy.on_estop(lying, Telemetry(custom={"body_height": 0.05}))
        self.assertEqual(lying.calls, [("estop",), ("sport", "damp")])
        self.assertEqual(action.damp_why, "lying_down")
        unknown = _Transport()
        policy.on_estop(unknown, Telemetry())
        self.assertEqual(unknown.calls, [("estop",)])

    def test_arming_uri_matches_requires_hardware_arming(self) -> None:
        cases = [
            _Transport("simulated"),
            _Transport("dds", "lo", 1),
            _Transport("dds", "127.0.0.1", 2),
            _Transport("dds", "lo0", 1),
            _Transport("dds", "Loopback Pseudo-Interface 1", 1),
            _Transport("dds", "lo", 0),
            _Transport("dds", "eth0", 1),
            _Transport("dds", "", 1),
            _Transport("serial"),
            _Transport("ros2"),
            _Transport("composite"),
        ]
        cases.append(_Transport("dds", "lo", "1"))
        policy = LeggedPolicy(GO2_PROFILE)
        for transport in cases:
            with self.subTest(protocol=transport.protocol, iface=transport.iface):
                uri = arming_uri(transport)
                self.assertEqual(
                    policy.arming(uri, {}),
                    requires_hardware_arming(transport, {}),
                )

    def test_fall_or_tilt_matches_today(self) -> None:
        policy = LeggedPolicy(GO2_PROFILE)
        self.assertIsNone(
            policy.fall_or_tilt(
                Telemetry(custom={"roll": 0.1, "pitch": -0.1, "error_code": 0})
            )
        )
        tilt = policy.fall_or_tilt(Telemetry(custom={"roll": 0.9, "pitch": 0.0}))
        self.assertIsNotNone(tilt)
        assert tilt is not None
        self.assertEqual(tilt[0], "tilt")
        fault = policy.fall_or_tilt(Telemetry(custom={"error_code": 3}))
        self.assertIsNotNone(fault)
        assert fault is not None
        self.assertEqual(fault[0], "fault")
        self.assertEqual(fault[1]["error_code"], 3)


class TestAuditNamesPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-policy-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _events(self) -> list[dict]:
        out = []
        for path in sorted(Path(self.tmp).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                out.append(json.loads(line))
        return out

    def _gate(self, robot_id: str) -> SafetyGate:
        spec = get_spec(robot_id)
        inner = _Transport()
        gate = SafetyGate(
            inner,
            spec,
            clock=_Clock(),
            environ={ARM_ENV: "1"},
            isatty=lambda: True,
            input_fn=lambda _prompt: ARM_PHRASE,
            audit=AuditLog(self.tmp, wall_clock=lambda: WALL),
            auto_tick=False,
        )
        gate.connect()
        return gate

    def test_go2_audit_names_legged_and_keeps_existing_fields(self) -> None:
        gate = self._gate("unitree-go2")
        gate._inner._emit_telemetry(
            Telemetry(custom={"roll": 0.0, "pitch": 0.0, "error_code": 0})
        )
        self.assertTrue(gate.arm())
        rec = self._events()[-1]
        self.assertEqual(rec["event"], "arm")
        self.assertEqual(rec["robot"], "unitree-go2")
        self.assertEqual(rec["protocol"], "dds")
        self.assertEqual(rec["policy"], "legged")
        self.assertIn("ts", rec)
        self.assertEqual(rec["t"], WALL)
        self.assertTrue(rec["hardware"])

    def test_omnibot_audit_names_ground_and_arm(self) -> None:
        gate = self._gate("omnibot")
        gate.emergency_stop()
        rec = [item for item in self._events() if item["event"] == "estop"][-1]
        self.assertEqual(rec["policy"], "ground+arm")
        self.assertEqual(rec["robot"], "omnibot")
        self.assertEqual(rec["protocol"], "dds")
        self.assertEqual(rec["reason"], "operator")
        self.assertIn("inner_error", rec)


if __name__ == "__main__":
    unittest.main()
