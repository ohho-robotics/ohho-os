"""Generic humanoid modes and Robot methods (OHH-326).

No hardware. Gated cases use a fake transport and a temporary audit log.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohho import capabilities as caps
from ohho import humanoid
from ohho.adapters.sim import SimTransport
from ohho.humanoid import HumanoidMode, is_standing, joints_permitted, parse_mode
from ohho.registry import RobotSpec, get_spec
from ohho.robot import Robot
from ohho.robot_spec import RobotClass
from ohho.runtime import NativeRuntime
from ohho.safety import HUMANOID_ACTIONS, AuditLog, SafetyGate
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport

# Names that must not appear in ohho/humanoid.py. The test file may list them.
_VENDOR = re.compile(
    r"booster|unitree|\bgo2\b|\bt2\b|px4|crazyflie|yahboom|omnibot|\bdji\b|mavlink",
    re.IGNORECASE,
)

_CALLS: tuple[tuple[str, tuple[object, ...], str, dict[str, object], str], ...] = (
    ("prepare", (), "prepare", {}, caps.HUMANOID_POSTURE),
    ("walk_mode", (), "walk_mode", {}, caps.HUMANOID_POSTURE),
    ("damp", (), "damp", {}, caps.HUMANOID_POSTURE),
    ("get_up", (), "get_up", {}, caps.HUMANOID_POSTURE),
    ("lie_down", (), "lie_down", {}, caps.HUMANOID_POSTURE),
    ("look", (0.1, -0.2), "look", {"pitch": 0.1, "yaw": -0.2}, caps.HEAD_GAZE),
    (
        "move_ee",
        ("left", (0.1, 0.2, 0.3), 1.0),
        "move_ee",
        {"side": "left", "pose": (0.1, 0.2, 0.3), "t": 1.0},
        caps.MANIPULATION_DUAL_ARM,
    ),
    (
        "gripper",
        ("right", 0.5),
        "gripper",
        {"side": "right", "cmd": 0.5},
        caps.END_EFFECTOR_GRIPPER,
    ),
    (
        "hand",
        ("left", (0.0, 0.1)),
        "hand",
        {"side": "left", "fingers": (0.0, 0.1)},
        caps.END_EFFECTOR_DEX_HAND,
    ),
)


def _spec(*capabilities: str, **kwargs: object) -> RobotSpec:
    body: dict[str, object] = {
        "id": "humanoid-test",
        "name": "Humanoid test",
        "category": "humanoid",
        "capabilities": capabilities,
        "max_lin": 0.5,
        "max_ang": 0.6,
        "robot_class": RobotClass.HUMANOID,
    }
    body.update(kwargs)
    return RobotSpec(**body)  # type: ignore[arg-type]


def _full_spec() -> RobotSpec:
    return _spec(
        caps.BASE_DRIVE,
        caps.BASE_HOLONOMIC,
        caps.HUMANOID_LOCOMOTION,
        caps.HUMANOID_POSTURE,
        caps.WHOLE_BODY_JOINT,
        caps.MANIPULATION,
        caps.MANIPULATION_DUAL_ARM,
        caps.END_EFFECTOR_GRIPPER,
        caps.END_EFFECTOR_DEX_HAND,
        caps.HEAD_GAZE,
        joint_names=("a", "b"),
    )


class RecordingTransport(BaseTransport):
    """Ungated fake. Records velocity, arm joints and send_humanoid."""

    protocol = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self.humanoid: list[tuple[str, dict]] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.joints: list[tuple[str, float]] = []
        self.latest = Telemetry()
        self.state = ConnectionState.CONNECTED

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="fake")

    def read(self) -> Telemetry:
        return self.latest

    def send_velocity(self, vel: Velocity) -> None:
        self.velocities.append((vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        self.joints.append((name, float(position)))

    def emergency_stop(self) -> None:
        return None

    def release_stop(self) -> None:
        return None

    def send_humanoid(self, action: str, **fields: object) -> None:
        self.humanoid.append((action, dict(fields)))


class RecordingGate(SafetyGate):
    """Records command() then runs the real gate."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.commands: list[tuple[object, tuple]] = []
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def command(self, action: object, *args: object) -> bool:
        self.commands.append((action, args))
        return super().command(action, *args)


def _invoke(bot: Robot) -> None:
    for name, args, _action, _fields, _cap in _CALLS:
        getattr(bot, name)(*args)
    bot.move_joints("left_arm", (0.2, -0.1), kp=4.0, kd=0.2)


class TestModes(unittest.TestCase):
    def test_mode_set_and_unknown_is_never_standing(self):
        self.assertEqual(
            {mode.name for mode in HumanoidMode},
            {"OFF", "DAMPING", "PREPARE", "WALK", "CUSTOM", "FALLEN", "UNKNOWN"},
        )
        standing = {mode for mode in HumanoidMode if is_standing(mode)}
        self.assertEqual(standing, {HumanoidMode.PREPARE, HumanoidMode.WALK})
        self.assertNotIn(HumanoidMode.UNKNOWN, standing)
        self.assertFalse(is_standing(HumanoidMode.UNKNOWN))
        self.assertFalse(is_standing(HumanoidMode.OFF))
        self.assertFalse(is_standing(HumanoidMode.DAMPING))
        self.assertFalse(is_standing(HumanoidMode.FALLEN))
        self.assertFalse(is_standing(HumanoidMode.CUSTOM))
        self.assertTrue(is_standing(HumanoidMode.PREPARE))
        self.assertTrue(is_standing(HumanoidMode.WALK))

    def test_unrecognised_values_parse_to_unknown_and_are_not_standing(self):
        for value in (
            None,
            "",
            "   ",
            "standing",
            "upright",
            "get_up",
            "walk_mode",
            "kWalking",
            "booster",
            "t2",
            1,
            0,
            True,
        ):
            mode = parse_mode(value)
            self.assertIs(mode, HumanoidMode.UNKNOWN)
            self.assertFalse(is_standing(mode))
            self.assertFalse(is_standing(value))
        self.assertFalse(is_standing("WALK"))
        self.assertFalse(is_standing("UNKNOWN"))
        self.assertIs(parse_mode("walk"), HumanoidMode.WALK)
        self.assertIs(parse_mode(" Walk "), HumanoidMode.WALK)
        self.assertIs(parse_mode(HumanoidMode.FALLEN), HumanoidMode.FALLEN)
        self.assertTrue(is_standing(parse_mode("prepare")))
        self.assertFalse(is_standing(parse_mode("custom")))

    def test_mode_of_reads_only_the_humanoid_block(self):
        self.assertIs(humanoid.mode_of(None), HumanoidMode.UNKNOWN)
        self.assertIs(humanoid.mode_of(Telemetry()), HumanoidMode.UNKNOWN)
        self.assertIs(
            humanoid.mode_of(Telemetry(custom={"mode": "WALK"})),
            HumanoidMode.UNKNOWN,
        )
        self.assertIs(
            humanoid.mode_of(Telemetry(custom={"humanoid": "WALK"})),
            HumanoidMode.UNKNOWN,
        )
        self.assertIs(
            humanoid.mode_of(Telemetry(custom={"humanoid": {"mode": "UNKNOWN"}})),
            HumanoidMode.UNKNOWN,
        )
        self.assertFalse(
            is_standing(
                humanoid.mode_of(Telemetry(custom={"humanoid": {"mode": "not-a-mode"}}))
            )
        )
        self.assertIs(
            humanoid.mode_of(Telemetry(custom={"humanoid": {"mode": "CUSTOM"}})),
            HumanoidMode.CUSTOM,
        )
        self.assertTrue(
            is_standing(
                humanoid.mode_of(
                    Telemetry(custom={"humanoid": {"mode": HumanoidMode.WALK}})
                )
            )
        )

    def test_joints_require_custom_sim_and_policy(self):
        self.assertFalse(humanoid.policy_allows_low_level_joints())
        for mode in HumanoidMode:
            allowed = joints_permitted(mode, simulated=True, policy_allows=True)
            self.assertEqual(allowed, mode is HumanoidMode.CUSTOM)
        self.assertFalse(
            joints_permitted(HumanoidMode.CUSTOM, simulated=False, policy_allows=True)
        )
        self.assertFalse(
            joints_permitted(HumanoidMode.CUSTOM, simulated=True, policy_allows=False)
        )
        self.assertFalse(
            joints_permitted(HumanoidMode.UNKNOWN, simulated=True, policy_allows=True)
        )
        self.assertTrue(
            joints_permitted(HumanoidMode.CUSTOM, simulated=True, policy_allows=True)
        )
        self.assertFalse(joints_permitted("CUSTOM", simulated=True, policy_allows=True))


class TestNoCapability(unittest.TestCase):
    def test_builtin_robots_noop_and_return_themselves(self):
        for robot_id in ("omnibot", "unitree-go2", "sim"):
            with self.subTest(robot_id=robot_id):
                spec = get_spec(robot_id)
                transport = SimTransport(spec)
                transport.connect()
                bot = Robot(spec, transport, NativeRuntime())
                self.assertIs(bot.prepare().walk_mode().damp().get_up().lie_down(), bot)
                self.assertIs(bot.look("bad", "nope"), bot)  # type: ignore[arg-type]
                self.assertIs(bot.move_ee("middle", "pose", -1), bot)  # type: ignore[arg-type]
                self.assertIs(bot.gripper(1, None), bot)  # type: ignore[arg-type]
                self.assertIs(bot.hand("left", "fingers"), bot)  # type: ignore[arg-type]
                self.assertIs(bot.move_joints("left_arm", "not-numbers"), bot)  # type: ignore[arg-type]
                transport.step(1.0)
                odom = transport.read().odom
                assert odom is not None
                self.assertAlmostEqual(odom.x, 0.0)
                self.assertAlmostEqual(odom.y, 0.0)

    def test_empty_spec_records_nothing(self):
        transport = RecordingTransport()
        bot = Robot(_spec(), transport, NativeRuntime())
        _invoke(bot)
        self.assertEqual(transport.humanoid, [])
        self.assertEqual(transport.velocities, [])
        self.assertEqual(transport.joints, [])

    def test_arm_form_still_rejects_a_group_argument(self):
        bot = Robot(_spec(caps.MANIPULATION), RecordingTransport(), NativeRuntime())
        with self.assertRaises(TypeError):
            bot.move_joints([0.1, 0.2], [0.3])  # type: ignore[misc]


class TestUngatedDispatch(unittest.TestCase):
    def _bot(self, spec: RobotSpec) -> tuple[Robot, RecordingTransport]:
        transport = RecordingTransport()
        return Robot(spec, transport, NativeRuntime()), transport

    def test_each_method_dispatches_when_the_capability_is_present(self):
        bot, transport = self._bot(_full_spec())
        _invoke(bot)
        self.assertEqual(
            transport.humanoid,
            [(action, fields) for _name, _args, action, fields, _cap in _CALLS],
        )

    def test_each_method_requires_only_its_capability(self):
        for name, args, action, fields, capability in _CALLS:
            with self.subTest(name=name):
                present, transport = self._bot(_spec(capability))
                self.assertIs(getattr(present, name)(*args), present)
                self.assertEqual(transport.humanoid, [(action, fields)])
                absent, quiet = self._bot(_spec())
                self.assertIs(getattr(absent, name)(*args), absent)
                self.assertEqual(quiet.humanoid, [])

    def test_bad_arguments_are_not_sent_when_capable(self):
        bot, transport = self._bot(_full_spec())
        with self.assertRaises(TypeError):
            bot.look("up", 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bot.look(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            bot.move_ee("middle", (0.0, 0.0, 0.0), 1.0)
        with self.assertRaises(ValueError):
            bot.move_ee("left", (0.0, 0.0, 0.0), -0.1)
        with self.assertRaises(TypeError):
            bot.move_ee("left", "here", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bot.gripper("left", "  ")
        with self.assertRaises(TypeError):
            bot.gripper("left", None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bot.hand("right", ())
        with self.assertRaises(ValueError):
            bot.move_joints("  ", (0.1,))
        self.assertEqual(transport.humanoid, [])

    def test_commands_coerce_numbers_and_sides(self):
        bot, transport = self._bot(_full_spec())
        bot.look(1, 0)
        bot.move_ee("LEFT", [0.2, 0.0, 0.4], 1)
        bot.gripper("Right", "open")
        self.assertEqual(
            transport.humanoid,
            [
                ("look", {"pitch": 1.0, "yaw": 0.0}),
                ("move_ee", {"side": "left", "pose": (0.2, 0.0, 0.4), "t": 1.0}),
                ("gripper", {"side": "right", "cmd": "open"}),
            ],
        )

    def test_missing_hook_raises_and_does_not_silently_drop(self):
        spec = _spec(caps.HUMANOID_POSTURE)
        transport = SimTransport(spec)
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        with self.assertRaises(AttributeError):
            bot.prepare()
        transport.step(0.2)
        odom = transport.read().odom
        assert odom is not None
        self.assertAlmostEqual(odom.x, 0.0)

    def test_walking_uses_drive_and_not_a_humanoid_action(self):
        bot, transport = self._bot(_full_spec())
        transport.latest = Telemetry(custom={"humanoid": {"mode": "UNKNOWN"}})
        self.assertFalse(is_standing(humanoid.mode_of(bot.telemetry())))
        self.assertIs(bot.drive(vx=0.2, vy=0.1, w=-0.05), bot)
        self.assertEqual(transport.humanoid, [])
        self.assertEqual(transport.velocities, [(0.2, 0.1, -0.05)])
        self.assertIs(bot.walk_mode(), bot)
        self.assertEqual(transport.humanoid, [("walk_mode", {})])

    def test_arm_move_joints_stays_on_the_arm_path(self):
        spec = _full_spec()
        transport = SimTransport(spec)
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        bot.move_joints([0.25, -0.25])
        for _ in range(100):
            transport.step(0.05)
        positions = {joint.name: joint.position for joint in transport.read().joints}
        self.assertAlmostEqual(positions["a"], 0.25, places=2)
        self.assertAlmostEqual(positions["b"], -0.25, places=2)
        before = dict(positions)
        bot.move_joints("left_arm", (0.9,))
        self.assertEqual(
            {joint.name: joint.position for joint in transport.read().joints},
            before,
        )


class TestLowLevelJoints(unittest.TestCase):
    def test_refused_unless_custom_simulated_and_policy_allows(self):
        transport = RecordingTransport()
        bot = Robot(_spec(caps.WHOLE_BODY_JOINT), transport, NativeRuntime())
        cases = (
            ("UNKNOWN", "simulated", False),
            ("CUSTOM", "simulated", False),
            ("CUSTOM", "serial", True),
            ("WALK", "simulated", True),
        )
        for mode, protocol, allow in cases:
            with self.subTest(mode=mode, protocol=protocol, allow=allow):
                transport.protocol = protocol
                transport.latest = Telemetry(custom={"humanoid": {"mode": mode}})
                transport.humanoid.clear()
                with patch.object(
                    humanoid, "policy_allows_low_level_joints", return_value=allow
                ):
                    self.assertIs(
                        bot.move_joints("left_arm", (0.2, -0.1), 4.0, 0.2), bot
                    )
                self.assertEqual(transport.humanoid, [])
            self.assertFalse(is_standing(HumanoidMode.UNKNOWN))

        transport.protocol = "simulated"
        transport.latest = Telemetry(custom={"humanoid": {"mode": "CUSTOM"}})
        with patch.object(
            humanoid, "policy_allows_low_level_joints", return_value=True
        ):
            self.assertIs(bot.move_joints("left_arm", [0.2, -0.1], kp=4, kd=0.2), bot)
        self.assertEqual(
            transport.humanoid,
            [
                (
                    "move_joints",
                    {"group": "left_arm", "q": (0.2, -0.1), "kp": 4.0, "kd": 0.2},
                )
            ],
        )
        self.assertFalse(is_standing(HumanoidMode.CUSTOM))


class TestGatedDispatch(unittest.TestCase):
    def test_each_method_dispatches_through_the_gate_and_is_refused(self):
        inner = RecordingTransport()
        spec = _full_spec()
        with tempfile.TemporaryDirectory() as tmp:
            gate = RecordingGate(
                inner,
                spec,
                audit=AuditLog(tmp),
                environ={},
                auto_tick=False,
            )
            bot = Robot(spec, gate, NativeRuntime())
            self.assertIsInstance(bot.transport, SafetyGate)
            self.assertFalse(gate.armed)
            with self.assertLogs("ohho.safety", level="WARNING"):
                _invoke(bot)
            self.assertEqual(
                gate.commands,
                [
                    (action, _args_of(name, args))
                    for name, args, action, _fields, _cap in _CALLS
                ]
                + [("move_joints", ("left_arm", (0.2, -0.1), 4.0, 0.2))],
            )
            self.assertEqual(inner.humanoid, [])
            self.assertEqual(inner.joints, [])
            self.assertEqual(inner.velocities, [(0.0, 0.0, 0.0)] * len(gate.commands))
            body = "".join(
                path.read_text(encoding="utf-8") for path in Path(tmp).glob("*.jsonl")
            )
            self.assertIn("not_armed", body)
            self.assertNotIn("booster", body.lower())

    def test_move_joints_stays_refused_when_the_policy_would_allow_it(self):
        inner = RecordingTransport()
        inner.latest = Telemetry(custom={"humanoid": {"mode": "CUSTOM"}})
        spec = _spec(caps.WHOLE_BODY_JOINT)
        with tempfile.TemporaryDirectory() as tmp:
            gate = RecordingGate(
                inner,
                spec,
                audit=AuditLog(tmp),
                environ={},
                auto_tick=False,
            )
            bot = Robot(spec, gate, NativeRuntime())
            with patch.object(
                humanoid, "policy_allows_low_level_joints", return_value=True
            ):
                with self.assertLogs("ohho.safety", level="WARNING"):
                    self.assertIs(bot.move_joints("left_arm", (0.1,), kp=1.0), bot)
            self.assertEqual(
                gate.commands, [("move_joints", ("left_arm", (0.1,), 1.0, None))]
            )
            self.assertEqual(inner.humanoid, [])
            self.assertFalse(is_standing(humanoid.mode_of(bot.telemetry())))


def _args_of(name: str, args: tuple) -> tuple:
    """Positional command tuple for one high-level call, after coercion."""
    if name == "look":
        return (0.1, -0.2)
    if name == "move_ee":
        return ("left", (0.1, 0.2, 0.3), 1.0)
    if name == "gripper":
        return ("right", 0.5)
    if name == "hand":
        return ("left", (0.0, 0.1))
    return args


class TestActionNames(unittest.TestCase):
    def test_module_actions_match_the_gate_set(self):
        self.assertEqual(frozenset(humanoid.ACTION_NAMES), HUMANOID_ACTIONS)
        self.assertIn("move_joints", HUMANOID_ACTIONS)
        self.assertNotIn("move_joints", {"move", "stop_move"})

    def test_humanoid_module_has_no_vendor_names(self):
        path = Path(humanoid.__file__)
        self.assertEqual(path.name, "humanoid.py")
        text = path.read_text(encoding="utf-8")
        match = _VENDOR.search(text)
        self.assertIsNone(match, match.group(0) if match else "")


if __name__ == "__main__":
    unittest.main()
