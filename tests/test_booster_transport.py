"""booster:// transport against a mock client. The real SDK is never imported."""

from __future__ import annotations

import builtins
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from ohho.adapters import resolve_transport
from ohho.adapters.errors import AdapterUnavailable
from ohho.humanoid import is_upright
from ohho.humanoid_telemetry import HUMANOID_BLOCK_KEYS
from ohho.robot import Robot
from ohho.runtime.native import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import ConnectionState, Velocity

try:
    from test_ohho_booster import (
        _Entry,
        _builtin_entry,
        _discovered,
        _ensure_plugin_path,
    )
except ImportError:  # python -m unittest tests.test_booster_transport
    from tests.test_ohho_booster import (  # type: ignore[no-redef]
        _Entry,
        _builtin_entry,
        _discovered,
        _ensure_plugin_path,
    )

_ensure_plugin_path()

from ohho_booster.factory import booster_transport  # noqa: E402
from ohho_booster.messages import (  # noqa: E402
    TOPIC_JOINT_CTRL,
    TOPIC_LOCO_API,
    CommandRefused,
    UnknownRobotModel,
    fall_state_name,
    is_supported_model,
    vendor_mode_name,
)
from ohho_booster.modes import generic_mode  # noqa: E402
from ohho_booster.plugin import plugin as plugin_factory  # noqa: E402
from ohho_booster.sdk import (  # noqa: E402
    STATE_CHANGING_RPCS,
    CommandFailed,
    SdkSession,
    SdkUnavailable,
    _load_sdk,
    open_session,
)
from ohho_booster.spec import booster_t2_spec  # noqa: E402
from ohho_booster.transport import BoosterTransport  # noqa: E402

_ADDRESS = "192.0.2.10"
_PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "ohho-booster"
_REPO = Path(__file__).resolve().parents[1]
_REFUSED = (
    "dance",
    "kick",
    "zero_torque_drag",
    "ZeroTorqueDrag",
    "record_trajectory",
    "replay_trajectory",
    "load_custom_trained_traj",
    "LoadCustomTrainedTraj",
    "ai",
    "lui",
    "soccer",
    "visual_kick",
    "lion_dance",
    "handshake",
)


class FakeSession:
    """Stand-in for the SDK session. Records calls and pushes samples."""

    def __init__(self, model: object = "T2", vendor_mode: str = "kUnknown") -> None:
        self.model = model
        self.vendor_mode = vendor_mode
        self.moves: list[tuple[float, float, float]] = []
        self.mode_changes: list[str] = []
        self.joint_cmds: list[tuple[str, list[dict[str, float]]]] = []
        self.calls: list[object] = []
        self.subscribed = False
        self.closed = 0
        self._handler = None

    def connect(self) -> object:
        self.calls.append("connect")
        return self.model

    def close(self) -> None:
        self.closed += 1

    def mode(self) -> str:
        return self.vendor_mode

    def change_mode(self, vendor_mode: str) -> None:
        self.mode_changes.append(vendor_mode)
        self.vendor_mode = vendor_mode

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        self.moves.append((vx, vy, vyaw))

    def rotate_head(self, pitch: float, yaw: float) -> None:
        self.calls.append(("rotate_head", pitch, yaw))

    def get_up(self) -> None:
        self.calls.append("get_up")

    def lie_down(self) -> None:
        self.calls.append("lie_down")

    def move_hand(self, side: str, pose: object, duration_s: float) -> None:
        self.calls.append(("move_hand", side, tuple(pose), duration_s))  # type: ignore[arg-type]

    def move_dual_hands(self, left: object, right: object, duration_s: float) -> None:
        self.calls.append(("move_dual_hands", tuple(left), tuple(right), duration_s))  # type: ignore[arg-type]

    def control_gripper(self, side: str, command: object) -> None:
        self.calls.append(("gripper", side, command))

    def control_hand(self, side: str, fingers: object) -> None:
        self.calls.append(("hand", side, tuple(fingers)))  # type: ignore[arg-type]

    def reset_odometry(self) -> None:
        self.calls.append("reset_odometry")

    def publish_joints(self, cmd_type: str, motors: object) -> None:
        copied = [dict(item) for item in motors]  # type: ignore[union-attr]
        self.joint_cmds.append((cmd_type, copied))

    def subscribe(self, callback: object) -> None:
        self.subscribed = True
        self._handler = callback

    def push(self, topic: str, msg: object) -> None:
        if self._handler is None:
            raise AssertionError("not subscribed")
        self._handler(topic, msg)


class _Bag:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Enum:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _LowCmd:
    def __init__(self) -> None:
        self.cmd_type = None
        self.motor_cmd: list[object] = []


class _MotorCmd:
    def __init__(self) -> None:
        self.mode = None
        self.q = None
        self.dq = None
        self.tau = None
        self.kp = None
        self.kd = None
        self.weight = None


class _Posture:
    def __init__(self) -> None:
        self.position = None
        self.orientation = None


class _GripperParam:
    def __init__(self) -> None:
        self.position = None
        self.force = None
        self.speed = None


class FakeSdk:
    """Enough of the vendor module for SdkSession, with no network and no wheel."""

    def __init__(
        self, model: str = "T2", mode: int = 2, change_status: object = 0
    ) -> None:
        self.model = model
        self.mode = mode
        self.change_status = change_status
        self.events: list[object] = []
        self.subs: list[object] = []
        self.publishers: list[object] = []
        self.written: list[_LowCmd] = []
        self.closed: list[str] = []
        self.client: _Client | None = None
        self.RobotMode = _Enum(
            kDamping="kDamping",
            kPrepare="kPrepare",
            kWalking="kWalking",
            kCustom="kCustom",
        )
        self.LowCmdType = _Enum(SERIAL="SERIAL", PARALLEL="PARALLEL")
        self.HandIndex = _Enum(kLeftHand=0, kRightHand=1)
        self.GripperControlMode = _Enum(kPosition="kPosition")
        self.BoosterHandType = _Enum(kInspireHand="kInspireHand")
        self.ChannelFactory = _ChannelFactory(self)
        self.LowCmd = _LowCmd
        self.MotorCmd = _MotorCmd
        self.Posture = _Posture
        self.GripperMotionParameter = _GripperParam

    def B1LocoClient(self) -> _Client:
        self.client = _Client(self)
        return self.client

    def B1LowStateSubscriber(self, cb: object) -> _Sub:
        return _Sub(self, "rt/low_state", cb)

    def B1FallDownStateSubscriber(self, cb: object) -> _Sub:
        return _Sub(self, "rt/fall_down", cb)

    def B1OdometerStateSubscriber(self, cb: object) -> _Sub:
        return _Sub(self, "rt/odometer_state", cb)

    def B1RobotStatesSubscriber(self, cb: object) -> _Sub:
        return _Sub(self, "rt/robot_states", cb)

    def B1LowCmdPublisher(self) -> _Pub:
        return _Pub(self)

    def Position(self, x: float, y: float, z: float) -> _Bag:
        return _Bag(x=x, y=y, z=z)

    def Orientation(self, roll: float, pitch: float, yaw: float) -> _Bag:
        return _Bag(roll=roll, pitch=pitch, yaw=yaw)

    def DexterousFingerParameter(self, *args: object) -> tuple[object, ...]:
        return args


class _ChannelFactory:
    def __init__(self, sdk: FakeSdk) -> None:
        self.sdk = sdk

    def Instance(self) -> _ChannelFactory:
        return self

    def Init(self, domain: int, iface: str) -> None:
        self.sdk.events.append(("Init", domain, iface))


class _Client:
    def __init__(self, sdk: FakeSdk) -> None:
        self.sdk = sdk
        self.calls: list[object] = []

    def Init(self) -> None:
        self.calls.append("Init")

    def GetRobotInfo(self) -> _Bag:
        self.calls.append("GetRobotInfo")
        fields: dict[str, object] = {"model": self.sdk.model, "name": "robot"}
        firmware = getattr(self.sdk, "firmware", None)
        if firmware is not None:
            fields["version"] = firmware
        return _Bag(**fields)

    def GetMode(self) -> _Bag:
        self.calls.append("GetMode")
        return _Bag(mode=self.sdk.mode)

    def GetStatus(self) -> _Bag:
        self.calls.append("GetStatus")
        return _Bag(
            current_mode=self.sdk.mode, current_body_control=0, current_actions=[]
        )

    def ChangeMode(self, mode: object) -> object:
        self.calls.append(("ChangeMode", mode))
        status = self.sdk.change_status
        if status in (0, None) and isinstance(mode, str):
            self.sdk.mode = {"kDamping": 0, "kPrepare": 1, "kWalking": 2, "kCustom": 3}[
                mode
            ]
        return status

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.calls.append(("Move", vx, vy, vyaw))
        return 0

    def RotateHead(self, pitch: float, yaw: float) -> int:
        self.calls.append(("RotateHead", pitch, yaw))
        return 0

    def GetUp(self) -> None:
        self.calls.append("GetUp")
        return None

    def LieDown(self) -> int:
        self.calls.append("LieDown")
        return 0

    def MoveHandEndEffector(self, posture: object, millis: int, hand: object) -> int:
        self.calls.append(("MoveHandEndEffector", posture, millis, hand))
        return 0

    def MoveDualHandEndEffector(self, left: object, right: object, millis: int) -> int:
        self.calls.append(("MoveDualHandEndEffector", left, right, millis))
        return 0

    def ControlGripper(self, param: object, mode: object, hand: object) -> int:
        self.calls.append(("ControlGripper", param, mode, hand))
        return 0

    def ControlDexterousHand(self, params: object, hand: object, kind: object) -> int:
        self.calls.append(("ControlDexterousHand", params, hand, kind))
        return 0

    def ResetOdometry(self) -> int:
        self.calls.append("ResetOdometry")
        return 0


class _Sub:
    def __init__(self, sdk: FakeSdk, topic: str, cb: object) -> None:
        self.topic = topic
        self.cb = cb
        sdk.subs.append(self)
        self.sdk = sdk

    def InitChannel(self) -> None:
        self.sdk.events.append(("InitChannel", self.topic))

    def CloseChannel(self) -> None:
        self.sdk.closed.append(self.topic)


class _Pub:
    def __init__(self, sdk: FakeSdk) -> None:
        self.sdk = sdk
        sdk.publishers.append(self)

    def InitChannel(self) -> None:
        self.sdk.events.append(("InitChannel", "rt/joint_ctrl"))

    def Write(self, cmd: _LowCmd) -> None:
        self.sdk.written.append(cmd)

    def CloseChannel(self) -> None:
        self.sdk.closed.append("rt/joint_ctrl")


def _motors(count: int = 31) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(q=float(index), dq=0.01, tau_est=0.2, temperature=30)
        for index in range(count)
    ]


def _low(motors: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        imu_state=SimpleNamespace(rpy=(0.1, -0.2, 0.3), gyro=(0.01, 0.02, 0.03)),
        motor_state_serial=motors if motors is not None else _motors(),
        motor_state_parallel=[],
    )


class TransportCase(unittest.TestCase):
    def link(
        self, model: object = "T2"
    ) -> tuple[object, FakeSession, BoosterTransport]:
        spec = booster_t2_spec()
        session = FakeSession(model)
        return spec, session, BoosterTransport(spec, _ADDRESS, session)

    def test_model_tokens_and_mode_names(self) -> None:
        for model in (
            "T2",
            "Booster T2",
            "T2_31dof",
            "T2-PRO",
            "T2 EDU",
            "  booster-t2  ",
        ):
            self.assertTrue(is_supported_model(model), model)
        for model in ("K1", "T1", "T1_7DofArm", "", None, "unknown", "T20", "not t2"):
            self.assertFalse(is_supported_model(model), model)
        self.assertEqual(vendor_mode_name(2), "kWalking")
        self.assertEqual(generic_mode(vendor_mode_name(2)), "WALK")
        self.assertEqual(vendor_mode_name(4), "kSoccer")
        self.assertEqual(generic_mode(vendor_mode_name(4)), "UNKNOWN")
        self.assertEqual(fall_state_name("IS_READY"), "upright")
        self.assertEqual(fall_state_name(1), "falling")
        self.assertEqual(fall_state_name(2), "fallen")
        self.assertEqual(fall_state_name(3), "recovering")
        self.assertEqual(fall_state_name(99), "unknown")

    def test_connect_refuses_an_unknown_model_before_subscribe(self) -> None:
        for model in ("K1", "T1", "T1_7DofArm", "", None, "unknown", "T20", "not t2"):
            with self.subTest(model=model):
                _spec, session, transport = self.link(model)
                with self.assertRaises(UnknownRobotModel):
                    transport.connect()
                self.assertFalse(session.subscribed)
                self.assertGreaterEqual(session.closed, 1)
                self.assertEqual(session.moves, [])
                self.assertEqual(session.joint_cmds, [])
                self.assertEqual(session.mode_changes, [])
                self.assertEqual(transport.status().state, ConnectionState.ERROR)

    def test_supported_models_connect(self) -> None:
        for model in ("T2", "Booster T2", "T2_31dof", "T2-PRO", "T2 EDU"):
            with self.subTest(model=model):
                _spec, session, transport = self.link(model)
                status = transport.connect()
                self.assertEqual(status.state, ConnectionState.CONNECTED)
                self.assertEqual(transport.protocol, "booster")
                self.assertNotEqual(transport.protocol, "simulated")
                self.assertTrue(session.subscribed)
                self.assertEqual(session.calls, ["connect"])

    def test_mode_changes_and_move_only_in_walk(self) -> None:
        spec, session, transport = self.link()
        transport.connect()
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.assertEqual(session.moves, [])
        bot = Robot(spec, transport, NativeRuntime())
        self.addCleanup(bot.disconnect)
        bot.prepare()
        bot.drive(0.2, 0.1, -0.2)
        self.assertEqual(session.mode_changes, ["kPrepare"])
        self.assertEqual(session.moves, [])
        bot.walk_mode()
        bot.drive(0.2, 0.1, -0.2)
        self.assertEqual(session.mode_changes, ["kPrepare", "kWalking"])
        self.assertEqual(session.moves, [(0.2, 0.1, -0.2)])
        bot.stop()
        self.assertEqual(session.moves[-1], (0.0, 0.0, 0.0))
        bot.damp()
        bot.drive(0.3, 0.0, 0.0)
        self.assertEqual(session.moves, [(0.2, 0.1, -0.2), (0.0, 0.0, 0.0)])
        transport.send_humanoid("custom")
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        session.push("rt/robot_states", SimpleNamespace(current_mode=4))
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.assertEqual(session.moves, [(0.2, 0.1, -0.2), (0.0, 0.0, 0.0)])
        session.push("rt/robot_states", SimpleNamespace(current_mode=2))
        transport.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.assertEqual(session.moves[-1], (0.2, 0.0, 0.0))
        transport.emergency_stop()
        self.assertEqual(session.moves[-1], (0.0, 0.0, 0.0))
        parked = len(session.moves)
        transport.send_humanoid("prepare")
        transport.emergency_stop()
        self.assertEqual(session.mode_changes[-1], "kPrepare")
        self.assertEqual(len(session.moves), parked)

    def test_generic_actions_map_onto_the_client(self) -> None:
        spec, session, transport = self.link()
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        self.addCleanup(bot.disconnect)
        bot.get_up()
        bot.lie_down()
        bot.look(0.1, -0.2)
        pose = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
        bot.move_ee("left", pose, 0.5)
        bot.gripper("left", "open")
        bot.hand("right", (1.0, 2.0))
        self.assertIn("get_up", session.calls)
        self.assertIn("lie_down", session.calls)
        self.assertIn(("rotate_head", 0.1, -0.2), session.calls)
        self.assertIn(("move_hand", "left", pose, 0.5), session.calls)
        self.assertNotIn(
            "gripper", [item[0] for item in session.calls if isinstance(item, tuple)]
        )
        transport.send_humanoid("gripper", side="right", cmd="close")
        transport.send_humanoid("hand", side="left", fingers=(3.0, 4.0))
        transport.send_humanoid("reset_odometry")
        self.assertIn(("gripper", "right", "close"), session.calls)
        self.assertIn(("hand", "left", (3.0, 4.0)), session.calls)
        self.assertEqual(session.calls.count("reset_odometry"), 1)

    def test_refused_actions_are_not_forwarded(self) -> None:
        _spec, session, transport = self.link()
        transport.connect()
        for action in _REFUSED:
            with self.subTest(action=action):
                with self.assertRaises(CommandRefused) as caught:
                    transport.send_humanoid(action)
                self.assertEqual(caught.exception.reason, "not_used")
        self.assertEqual(session.mode_changes, [])
        self.assertEqual(session.moves, [])
        self.assertEqual(session.joint_cmds, [])
        self.assertNotIn("reset_odometry", session.calls)

    def test_low_level_writes_only_in_custom_and_not_via_send_joint_command(
        self,
    ) -> None:
        spec, session, transport = self.link()
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        self.addCleanup(bot.disconnect)
        with self.assertRaises(CommandRefused) as caught:
            transport.send_humanoid(
                "move_joints", group="head", q=(0.11, -0.22), kp=4.0, kd=0.5
            )
        self.assertEqual(caught.exception.reason, "not_custom")
        transport.send_humanoid("walk_mode")
        with self.assertRaises(CommandRefused):
            transport.send_humanoid("move_joints", group="head", q=(0.11, -0.22))
        self.assertEqual(session.joint_cmds, [])
        transport.send_humanoid("custom")
        bot.move_joints("head", [0.11, -0.22], kp=4.0, kd=0.5)
        transport.send_joint_command("head_pitch_joint", 0.2)
        self.assertEqual(session.joint_cmds, [])
        transport.send_humanoid(
            "move_joints", group="head", q=(0.11, -0.22), kp=4.0, kd=0.5
        )
        kind, motors = session.joint_cmds[0]
        self.assertEqual(kind, "SERIAL")
        self.assertEqual(len(motors), 31)
        self.assertEqual(
            set(motors[0]), {"mode", "q", "dq", "tau", "kp", "kd", "weight"}
        )
        self.assertEqual(motors[0]["q"], 0.11)
        self.assertEqual(motors[0]["mode"], 1.0)
        self.assertEqual(motors[0]["kp"], 4.0)
        self.assertEqual(motors[0]["kd"], 0.5)
        self.assertEqual(motors[0]["weight"], 1.0)
        self.assertEqual(motors[0]["dq"], 0.0)
        self.assertEqual(motors[0]["tau"], 0.0)
        self.assertEqual(motors[1]["q"], -0.22)
        self.assertEqual(motors[1]["weight"], 1.0)
        self.assertEqual(motors[2]["weight"], 0.0)
        self.assertEqual(motors[2]["mode"], 0.0)
        transport.send_humanoid(
            "move_joints", group="head", q=(0.0, 0.0), cmd_type="parallel"
        )
        self.assertEqual(session.joint_cmds[-1][0], "PARALLEL")
        with self.assertRaises(CommandRefused):
            transport.send_humanoid("move_joints", group="missing", q=(0.0,))
        with self.assertRaises(CommandRefused):
            transport.send_humanoid("move_joints", group="head", q=(0.0,))
        self.assertEqual(len(session.joint_cmds), 2)

    def test_telemetry_waits_for_fall_and_fills_the_humanoid_block(self) -> None:
        _spec, session, transport = self.link()
        transport.connect()
        low = _low()
        session.push("rt/low_state", low)
        session.push("rt/odometer_state", SimpleNamespace(x=1.5, y=-0.25, theta=0.4))
        session.push("rt/robot_states", SimpleNamespace(current_mode=2))
        self.assertNotIn("humanoid", transport.read().custom)
        session.push("rt/fall_down", SimpleNamespace(fall_down_state="IS_READY"))
        snap = transport.read()
        block = snap.custom["humanoid"]
        self.assertEqual(block["mode"], "WALK")
        self.assertEqual(block["vendor_mode"], "kWalking")
        self.assertEqual(block["fall_state"], "upright")
        self.assertEqual(block["base_rpy"], [0.1, -0.2, 0.3])
        self.assertEqual(block["base_ang_vel"], [0.01, 0.02, 0.03])
        self.assertEqual(block["odom"], {"x": 1.5, "y": -0.25, "yaw": 0.4})
        self.assertIsNotNone(snap.odom)
        assert snap.odom is not None
        self.assertEqual(snap.odom.x, 1.5)
        self.assertEqual(snap.odom.theta, 0.4)
        self.assertEqual(block["joints"]["head"]["q"], [0.0, 1.0])
        self.assertEqual(len(block["joints"]["left_arm"]["q"]), 7)
        self.assertEqual(len(block["joints"]["right_leg"]["q"]), 6)
        self.assertEqual(block["joints"]["head"]["temp"], [30.0, 30.0])
        self.assertEqual(block["battery"], {"fraction": 0.0, "voltage": 0.0})
        self.assertIs(block["experimental"], True)
        self.assertTrue(is_upright(snap))
        parallel = _low()
        parallel.motor_state_serial = []
        parallel.motor_state_parallel = _motors()
        session.push("rt/low_state", parallel)
        self.assertEqual(
            transport.read().custom["humanoid"]["joints"]["waist"]["q"],
            [16.0, 17.0, 18.0],
        )

    def _gated(
        self,
    ) -> tuple[SafetyGate, BoosterTransport, FakeSession, tempfile.TemporaryDirectory]:
        spec, session, transport = self.link()
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        gate = SafetyGate(
            transport,
            spec,
            audit=AuditLog(tmp.name),
            auto_tick=False,
            environ={},
        )
        gate.connect()
        return gate, transport, session, tmp

    def test_connect_does_not_latch_before_a_fall_message(self) -> None:
        gate, _transport, session, _tmp = self._gated()
        self.assertFalse(gate.latched)
        session.push("rt/low_state", _low())
        self.assertFalse(gate.latched)

    def test_upright_does_not_latch_and_other_fall_states_do(self) -> None:
        cases = (
            ("IS_READY", False, False),
            (0, False, False),
            ("IS_FALLING", True, False),
            ("IS_GETTING_UP", True, False),
            ("NOPE", True, False),
            ("HAS_FALLEN", True, True),
        )
        for fall, latches, damps in cases:
            with self.subTest(fall=fall):
                gate, _transport, session, _tmp = self._gated()
                session.push("rt/fall_down", SimpleNamespace(fall_down_state=fall))
                self.assertEqual(gate.latched, latches)
                if latches:
                    self.assertEqual(gate.latch_reason, "fall")
                if damps:
                    self.assertEqual(session.mode_changes, ["kDamping"])
                else:
                    self.assertEqual(session.mode_changes, [])
                self.assertEqual(session.moves, [])

    def test_fall_while_walking_zeros_move_and_latches(self) -> None:
        gate, transport, session, _tmp = self._gated()
        transport.send_humanoid("walk_mode")
        session.push("rt/fall_down", SimpleNamespace(fall_down_state="IS_FALLING"))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "fall")
        self.assertEqual(session.moves, [(0.0, 0.0, 0.0)])
        self.assertEqual(session.mode_changes, ["kWalking"])

    def test_gate_refuses_custom_reset_and_low_level_off_sim(self) -> None:
        gate, transport, session, tmp = self._gated()
        self.assertFalse(gate.command("custom"))
        self.assertFalse(gate.command("reset_odometry"))
        self.assertEqual(session.mode_changes, [])
        self.assertNotIn("reset_odometry", session.calls)
        transport.send_humanoid("custom")
        session.push("rt/fall_down", SimpleNamespace(fall_down_state="IS_READY"))
        self.assertFalse(gate.latched)
        self.assertEqual(
            gate.command("move_joints", "head", (0.0, 0.1), 4.0, 0.5), False
        )
        self.assertEqual(session.joint_cmds, [])
        text = "".join(
            path.read_text(encoding="utf-8") for path in Path(tmp.name).glob("*.jsonl")
        )
        self.assertIn("not_simulated", text)


class SdkSessionCase(unittest.TestCase):
    def test_connect_reads_the_model_before_subscribers(self) -> None:
        sdk = FakeSdk(model="T2", mode=1)
        session = SdkSession(_ADDRESS, 4, sdk)
        self.assertEqual(session.connect(), "T2")
        self.assertEqual(session.mode(), "kPrepare")
        self.assertEqual(sdk.subs, [])
        self.assertEqual(sdk.publishers, [])
        self.assertEqual(sdk.events, [("Init", 4, _ADDRESS)])
        assert sdk.client is not None
        self.assertEqual(
            sdk.client.calls, ["Init", "GetRobotInfo", "GetMode", "GetStatus"]
        )
        self.assertNotIn("ResetOdometry", sdk.client.calls)
        session.subscribe(lambda _topic, _msg: None)
        self.assertEqual(
            [getattr(sub, "topic") for sub in sdk.subs],
            ["rt/low_state", "rt/fall_down", "rt/odometer_state", "rt/robot_states"],
        )
        self.assertEqual(sdk.publishers, [])

    def test_unknown_model_closes_without_motion_or_subscribers(self) -> None:
        sdk = FakeSdk(model="K1")
        session = SdkSession(_ADDRESS, 0, sdk)
        with self.assertRaises(UnknownRobotModel):
            session.connect()
        self.assertEqual(sdk.subs, [])
        self.assertEqual(sdk.publishers, [])
        self.assertEqual(sdk.written, [])
        assert sdk.client is not None
        self.assertNotIn("GetMode", sdk.client.calls)
        self.assertFalse(
            any(
                isinstance(call, tuple) and call[0] == "Move"
                for call in sdk.client.calls
            )
        )

    def test_client_mapping(self) -> None:
        sdk = FakeSdk(model="Booster T2", mode=0)
        session = SdkSession("192.0.2.10", 2, sdk)
        session.connect()
        assert sdk.client is not None
        session.change_mode("kWalking")
        self.assertEqual(sdk.client.calls[-1], ("ChangeMode", sdk.RobotMode.kWalking))
        self.assertEqual(session.mode(), "kWalking")
        with self.assertRaises(ValueError):
            session.change_mode("kSoccer")
        sdk.change_status = 7
        with self.assertRaises(CommandFailed):
            session.change_mode("kCustom")
        sdk.change_status = None
        session.change_mode("kCustom")
        session.move(0.1, 0.2, 0.3)
        session.rotate_head(0.2, -0.1)
        session.get_up()
        session.lie_down()
        pose = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
        session.move_hand("left", pose, 0.5)
        session.move_dual_hands(pose, pose, 1.25)
        session.control_gripper("right", "open")
        session.control_gripper("left", "close")
        session.control_hand("left", (200.0, 800.0))
        session.reset_odometry()
        names = [
            call[0] if isinstance(call, tuple) else call for call in sdk.client.calls
        ]
        self.assertIn("Move", names)
        self.assertIn("RotateHead", names)
        self.assertIn("GetUp", names)
        self.assertIn("LieDown", names)
        self.assertIn("MoveHandEndEffector", names)
        self.assertIn("MoveDualHandEndEffector", names)
        self.assertIn("ControlGripper", names)
        self.assertIn("ControlDexterousHand", names)
        self.assertIn("ResetOdometry", names)
        hand = next(
            call
            for call in sdk.client.calls
            if isinstance(call, tuple) and call[0] == "MoveHandEndEffector"
        )
        self.assertEqual(hand[2], 500)
        self.assertEqual(hand[3], 0)
        self.assertEqual(hand[1].position.x, 0.1)
        self.assertEqual(hand[1].orientation.yaw, 0.0)
        dual = next(
            call
            for call in sdk.client.calls
            if isinstance(call, tuple) and call[0] == "MoveDualHandEndEffector"
        )
        self.assertEqual(dual[3], 1250)
        gripper = [
            call
            for call in sdk.client.calls
            if isinstance(call, tuple) and call[0] == "ControlGripper"
        ]
        self.assertEqual(gripper[0][1].position, 500)
        self.assertEqual(gripper[0][1].force, 100)
        self.assertEqual(gripper[0][2], "kPosition")
        self.assertEqual(gripper[1][1].position, 100)
        fingers = next(
            call
            for call in sdk.client.calls
            if isinstance(call, tuple) and call[0] == "ControlDexterousHand"
        )
        self.assertEqual(fingers[1], [(0, 200, 500, 500), (1, 800, 500, 500)])
        self.assertEqual(fingers[3], "kInspireHand")
        motors = [
            {
                "mode": 1,
                "q": 0.1,
                "dq": 0.0,
                "tau": 0.0,
                "kp": 4.0,
                "kd": 0.5,
                "weight": 1.0,
            }
        ]
        session.publish_joints("SERIAL", motors)
        session.publish_joints("PARALLEL", motors)
        self.assertEqual(len(sdk.publishers), 1)
        self.assertEqual(sdk.written[0].cmd_type, "SERIAL")
        self.assertEqual(sdk.written[1].cmd_type, "PARALLEL")
        motor = sdk.written[0].motor_cmd[0]
        self.assertEqual(motor.mode, 1)
        self.assertEqual(motor.q, 0.1)
        self.assertEqual(motor.kp, 4.0)
        self.assertEqual(motor.kd, 0.5)
        self.assertEqual(motor.weight, 1.0)
        session.close()
        self.assertIn("rt/joint_ctrl", sdk.closed)

    def test_empty_address_does_not_import_the_sdk(self) -> None:
        with patch("ohho_booster.sdk._load_sdk") as load:
            with self.assertRaises(ValueError):
                open_session("   ")
            load.assert_not_called()
            with self.assertRaises(AdapterUnavailable):
                booster_transport(booster_t2_spec(), "", None)
            load.assert_not_called()

    def test_missing_sdk_is_adapter_unavailable(self) -> None:
        with patch("ohho_booster.sdk._load_sdk", side_effect=SdkUnavailable("missing")):
            with self.assertRaises(AdapterUnavailable):
                booster_transport(booster_t2_spec(), _ADDRESS, None)

    def test_load_sdk_checks_the_pinned_version_without_a_real_wheel(self) -> None:
        name = "booster_robotics_sdk_python"
        saved = sys.modules.get(name)

        def restore() -> None:
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

        self.addCleanup(restore)
        module = ModuleType(name)
        module.__version__ = "9.9.9"  # type: ignore[attr-defined]
        sys.modules[name] = module
        with self.assertRaises(SdkUnavailable):
            _load_sdk()
        module.__version__ = "1.6.3"  # type: ignore[attr-defined]
        self.assertIs(_load_sdk(), module)
        sys.modules.pop(name, None)
        real_import = builtins.__import__

        def blocked(mod_name: str, *args: object, **kwargs: object) -> object:
            if mod_name == name:
                raise ImportError("blocked")
            return real_import(mod_name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaises(SdkUnavailable):
                _load_sdk()

    def test_sdk_source_does_not_name_refused_clients(self) -> None:
        text = (_PLUGIN / "ohho_booster" / "sdk.py").read_text(encoding="utf-8")
        for token in (
            "ZeroTorqueDrag",
            "LoadCustomTrainedTraj",
            "RecordTrajectory",
            "ReplayTrajectory",
            "AiSubtitle",
            "LuiAsr",
            "LionDance",
            "VisualKick",
            "Dance(",
        ):
            self.assertNotIn(token, text)


class RegistryCase(unittest.TestCase):
    def test_resolve_uses_the_plugin_factory_without_the_sdk(self) -> None:
        spec = booster_t2_spec()
        session = FakeSession("T2")
        entry = _Entry("booster", "ohho_booster:plugin", plugin_factory)
        before = {
            name for name in sys.modules if name.startswith("booster_robotics_sdk")
        }
        with _discovered(lambda: [_builtin_entry(), entry]):
            with patch("ohho_booster.sdk.open_session", return_value=session) as opened:
                transport = resolve_transport(
                    f"booster://{_ADDRESS}?domain=2", spec, None
                )
            opened.assert_called_once_with(f"{_ADDRESS}?domain=2")
            self.assertIsInstance(transport, BoosterTransport)
            self.assertIs(transport._session, session)
            stub = resolve_transport(f"booster+ros2://{_ADDRESS}", spec, None)
            self.assertEqual(stub.protocol, "booster+ros2")
            with self.assertRaises(AdapterUnavailable) as caught:
                stub.connect()
            message = str(caught.exception)
            self.assertIn("open question", message)
            self.assertNotIn("humble", message.lower())
            self.assertNotIn("jazzy", message.lower())
            with self.assertRaises(AdapterUnavailable):
                stub.send_velocity(Velocity())
            simulated = resolve_transport("sim://", spec, None)
            self.assertEqual(simulated.protocol, "simulated")
            with self.assertRaises(AdapterUnavailable):
                resolve_transport("wat://nope", spec, None)
        with _discovered(lambda: [_builtin_entry()]):
            with self.assertRaises(AdapterUnavailable):
                resolve_transport(f"booster://{_ADDRESS}", spec, None)
        after = {
            name for name in sys.modules if name.startswith("booster_robotics_sdk")
        }
        self.assertEqual(before, after)

    def test_optional_extra_pins_the_sdk_and_is_not_in_all(self) -> None:
        root = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
        plugin = (_PLUGIN / "pyproject.toml").read_text(encoding="utf-8")
        pin = 'booster = ["booster_robotics_sdk_python==1.6.3"]'
        self.assertIn(pin, root)
        self.assertIn(pin, plugin)
        all_extra = root.split("all = [", 1)[1].split("]", 1)[0]
        self.assertNotIn("booster_robotics_sdk_python", all_extra)


def _rpc_names(calls: list[object]) -> set[object]:
    names: set[object] = set()
    for item in calls:
        names.add(item[0] if isinstance(item, tuple) else item)
    return names


class ReadOnlyCase(unittest.TestCase):
    def test_read_only_publishes_nothing_and_keeps_humanoid_telemetry(self) -> None:
        """Fails if a joint publish or a state-changing RPC is attempted.

        The link is in ``WALK``, so a missing guard would call ``Move``.
        """
        sdk = FakeSdk(model="T2", mode=2)
        sdk.firmware = "1.8.3"
        session = SdkSession(_ADDRESS, 0, sdk, read_only=True)
        spec = booster_t2_spec()
        transport = BoosterTransport(spec, _ADDRESS, session, read_only=True)
        transport.connect()
        self.assertEqual(session.mode(), "kWalking")
        for sub in sdk.subs:
            topic = getattr(sub, "topic", "")
            if topic == "rt/low_state":
                sub.cb(_low())  # type: ignore[attr-defined]
            elif topic == "rt/fall_down":
                sub.cb(SimpleNamespace(fall_down_state="IS_READY"))  # type: ignore[attr-defined]
            elif topic == "rt/odometer_state":
                sub.cb(SimpleNamespace(x=0.1, y=0.2, theta=0.3))  # type: ignore[attr-defined]
        block = transport.read().custom["humanoid"]
        for key in HUMANOID_BLOCK_KEYS:
            self.assertIn(key, block)
        self.assertEqual(block["mode"], "WALK")
        self.assertEqual(block["fall_state"], "upright")
        self.assertIn("head", block["joints"])
        self.assertEqual(set(block["battery"]), {"fraction", "voltage"})

        transport.send_velocity(Velocity(0.4, 0.2, 0.1))
        transport.send_joint_command("head_pitch_joint", 0.2)
        transport.emergency_stop()
        for action in (
            "prepare",
            "walk_mode",
            "damp",
            "custom",
            "get_up",
            "lie_down",
            "look",
            "move_ee",
            "gripper",
            "hand",
            "move_joints",
            "reset_odometry",
        ):
            with self.subTest(action=action):
                with self.assertRaises(CommandRefused) as caught:
                    transport.send_humanoid(action)
                self.assertEqual(caught.exception.reason, "read_only")

        refused = (
            lambda: session.change_mode("kPrepare"),
            lambda: session.move(1.0, 0.0, 0.0),
            lambda: session.rotate_head(0.1, 0.1),
            lambda: session.get_up(),
            lambda: session.lie_down(),
            lambda: session.move_hand("left", (0, 0, 0, 0, 0, 0), 0.1),
            lambda: session.move_dual_hands(
                (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), 0.1
            ),
            lambda: session.control_gripper("left", "open"),
            lambda: session.control_hand("left", (0.1,)),
            lambda: session.reset_odometry(),
            lambda: session.publish_joints(
                "SERIAL",
                [
                    {
                        "mode": 0.0,
                        "q": 0.0,
                        "dq": 0.0,
                        "tau": 0.0,
                        "kp": 0.0,
                        "kd": 0.0,
                        "weight": 0.0,
                    }
                ],
            ),
        )
        for call in refused:
            with self.assertRaises(CommandRefused):
                call()

        self.assertEqual(sdk.publishers, [])
        self.assertEqual(sdk.written, [])
        blob = repr(sdk.events)
        self.assertNotIn(TOPIC_JOINT_CTRL, blob)
        self.assertNotIn(TOPIC_LOCO_API, blob)
        assert sdk.client is not None
        self.assertTrue(STATE_CHANGING_RPCS.isdisjoint(_rpc_names(sdk.client.calls)))
        self.assertTrue(
            _rpc_names(sdk.client.calls)
            <= {"Init", "GetRobotInfo", "GetMode", "GetStatus"}
        )

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        gate = SafetyGate(
            transport,
            spec,
            audit=AuditLog(tmp.name),
            auto_tick=False,
            environ={},
        )
        gate.connect()
        gate._zero_inner()
        assert sdk.client is not None
        self.assertTrue(STATE_CHANGING_RPCS.isdisjoint(_rpc_names(sdk.client.calls)))
        self.assertEqual(sdk.publishers, [])
        self.assertEqual(sdk.written, [])

    def test_resolve_read_only_reaches_the_session(self) -> None:
        spec = booster_t2_spec()
        sdk = FakeSdk(model="T2", mode=2)
        seen: dict[str, object] = {}

        def opener(address: str, sdk_module: object = None, *, read_only: bool = False):
            del sdk_module
            seen["address"] = address
            seen["read_only"] = read_only
            return SdkSession(address.partition("?")[0], 0, sdk, read_only=read_only)

        entry = _Entry("booster", "ohho_booster:plugin", plugin_factory)
        with _discovered(lambda: [_builtin_entry(), entry]):
            with patch("ohho_booster.sdk.open_session", opener):
                transport = resolve_transport(
                    f"booster://{_ADDRESS}?domain=2", spec, None, read_only=True
                )
        self.assertEqual(seen["address"], f"{_ADDRESS}?domain=2")
        self.assertIs(seen["read_only"], True)
        self.assertIsInstance(transport, BoosterTransport)
        self.assertTrue(transport.read_only)
        self.assertTrue(transport._session.read_only)
        transport.connect()
        transport.send_velocity(Velocity(0.3, 0.0, 0.0))
        with self.assertRaises(CommandRefused):
            transport._session.publish_joints("SERIAL", [])
        self.assertEqual(sdk.publishers, [])
        assert sdk.client is not None
        self.assertTrue(STATE_CHANGING_RPCS.isdisjoint(_rpc_names(sdk.client.calls)))


if __name__ == "__main__":
    unittest.main()
