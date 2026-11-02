"""Robot flight methods: no-op without aerial, dispatch with an explicit frame."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from ohho import capabilities as caps
from ohho.adapters.sim import SimTransport
from ohho.aerial import Frame
from ohho.registry import RobotSpec, get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport


def _aerial_spec() -> RobotSpec:
    return RobotSpec(
        id="aerial-test",
        name="Aerial test",
        category="aerial",
        capabilities=(caps.AERIAL, caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        max_lin=2.0,
        max_ang=0.8,
    )


def _ground_spec() -> RobotSpec:
    return RobotSpec(
        id="ground-test",
        name="Ground test",
        category="wheeled",
        capabilities=(caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        max_lin=1.0,
        max_ang=1.0,
    )


class RecordingTransport(BaseTransport):
    """Records flight requests. Implements the ungated ``send_flight`` hook."""

    protocol = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self.flight: list[tuple[str, dict]] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.state = ConnectionState.CONNECTED

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
        self.velocities.append((vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def emergency_stop(self) -> None:
        return None

    def release_stop(self) -> None:
        return None

    def send_flight(self, action: str, **fields: object) -> None:
        self.flight.append((action, dict(fields)))


class RecordingGate(SafetyGate):
    """SafetyGate that records ``command`` and ``arm`` before the real gate runs."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.commands: list[tuple[object, tuple]] = []
        self.arm_calls = 0
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def command(self, action: object, *args: object) -> bool:
        self.commands.append((action, args))
        return super().command(action, *args)

    def arm(self) -> bool:
        self.arm_calls += 1
        return super().arm()


def _calls() -> list[tuple[str, tuple]]:
    return [
        ("arm_flight", ()),
        ("disarm_flight", ()),
        ("takeoff", (2.5,)),
        ("land", ()),
        ("rtl", ()),
        ("hold", ()),
        ("offboard_position", (1.0, 2.0, -3.0, 0.4, Frame.LOCAL_NED)),
        ("offboard_velocity", (0.1, 0.0, -0.2, 0.05, Frame.BODY_FRD)),
        ("set_yaw", (0.7, Frame.LOCAL_ENU)),
    ]


class TestAerialRobot(unittest.TestCase):
    def _pair(self, spec: RobotSpec) -> tuple[Robot, RecordingTransport]:
        transport = RecordingTransport()
        return Robot(spec, transport, NativeRuntime()), transport

    def test_builtin_ground_specs_noop(self):
        for robot_id in ("omnibot", "unitree-go2", "sim"):
            spec = get_spec(robot_id)
            self.assertNotIn(caps.AERIAL, spec.capabilities)
            transport = SimTransport(spec)
            transport.connect()
            bot = Robot(spec, transport, NativeRuntime())
            for name, args in _calls():
                self.assertIs(getattr(bot, name)(*args), bot)
            transport.step(0.1)
            odom = transport.read().odom
            assert odom is not None
            self.assertAlmostEqual(odom.x, 0.0)
            self.assertAlmostEqual(odom.y, 0.0)

    def test_spec_without_aerial_records_nothing(self):
        bot, transport = self._pair(_ground_spec())
        for name, args in _calls():
            self.assertIs(getattr(bot, name)(*args), bot)
        self.assertEqual(transport.flight, [])
        # A bad altitude is ignored with the rest of the no-op.
        self.assertIs(bot.takeoff(-5), bot)
        self.assertIs(bot.takeoff("high"), bot)  # type: ignore[arg-type]
        self.assertEqual(transport.flight, [])
        # A frame argument is still required, including on a ground robot.
        with self.assertRaises(TypeError):
            bot.set_yaw(0.0, "LOCAL_ENU")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bot.offboard_position(0.0, 0.0, 0.0, 0.0)  # type: ignore[call-arg]
        self.assertEqual(transport.flight, [])

    def test_spec_with_aerial_dispatches_each_method_and_frame(self):
        bot, transport = self._pair(_aerial_spec())
        for name, args in _calls():
            self.assertIs(getattr(bot, name)(*args), bot)
        self.assertEqual(
            transport.flight,
            [
                ("arm", {}),
                ("disarm", {}),
                ("takeoff", {"alt_m": 2.5}),
                ("land", {}),
                ("rtl", {}),
                ("hold", {}),
                (
                    "offboard_position",
                    {
                        "x": 1.0,
                        "y": 2.0,
                        "z": -3.0,
                        "yaw": 0.4,
                        "frame": Frame.LOCAL_NED,
                    },
                ),
                (
                    "offboard_velocity",
                    {
                        "vx": 0.1,
                        "vy": 0.0,
                        "vz": -0.2,
                        "yaw_rate": 0.05,
                        "frame": Frame.BODY_FRD,
                    },
                ),
                ("set_yaw", {"yaw": 0.7, "frame": Frame.LOCAL_ENU}),
            ],
        )
        for _action, fields in transport.flight:
            self.assertNotIn("force", fields)
            self.assertNotIn(21196, fields.values())

    def test_each_frame_is_forwarded_unchanged(self):
        bot, transport = self._pair(_aerial_spec())
        for frame in Frame:
            bot.offboard_position(0.0, 0.0, 1.0, 0.2, frame)
            bot.offboard_velocity(0.0, 0.0, -0.1, 0.0, frame)
            bot.set_yaw(0.2, frame)
        forwarded = [fields["frame"] for _action, fields in transport.flight]
        self.assertEqual(forwarded, [frame for frame in Frame for _ in range(3)])
        for action, fields in transport.flight:
            self.assertIsInstance(fields["frame"], Frame)
            if action == "offboard_position":
                self.assertEqual(fields["z"], 1.0)
            if action == "offboard_velocity":
                self.assertEqual(fields["vz"], -0.1)

    def test_missing_or_string_frame_raises_and_does_not_dispatch(self):
        bot, transport = self._pair(_aerial_spec())
        with self.assertRaises(TypeError):
            bot.offboard_position(0.0, 0.0, 0.0, 0.0, "LOCAL_NED")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bot.offboard_velocity(0.0, 0.0, 0.0, 0.0, "BODY_FRD")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bot.set_yaw(0.0, None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            bot.offboard_velocity(0.0, 0.0, 0.0, 0.0)  # type: ignore[call-arg]
        self.assertEqual(transport.flight, [])

    def test_arm_flight_does_not_call_arm(self):
        bot, transport = self._pair(_aerial_spec())
        self.assertTrue(bot.arm())
        self.assertEqual(transport.flight, [])
        self.assertIs(bot.arm_flight(), bot)
        self.assertEqual(transport.flight, [("arm", {})])
        self.assertTrue(bot.arm())
        self.assertEqual(len(transport.flight), 1)

    def test_disarm_flight_has_no_force_parameter(self):
        self.assertEqual(
            list(inspect.signature(Robot.disarm_flight).parameters), ["self"]
        )
        bot, transport = self._pair(_aerial_spec())
        with self.assertRaises(TypeError):
            bot.disarm_flight(force=True)  # type: ignore[call-arg]
        self.assertEqual(transport.flight, [])
        bot.disarm_flight()
        self.assertEqual(transport.flight, [("disarm", {})])

    def test_takeoff_altitude_is_positive_up_when_aerial(self):
        bot, transport = self._pair(_aerial_spec())
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bot.takeoff(bad)
        with self.assertRaises(TypeError):
            bot.takeoff("2.5")  # type: ignore[arg-type]
        self.assertEqual(transport.flight, [])
        bot.takeoff(2)
        self.assertEqual(transport.flight, [("takeoff", {"alt_m": 2.0})])

    def test_non_finite_setpoints_are_not_sent(self):
        bot, transport = self._pair(_aerial_spec())
        with self.assertRaises(ValueError):
            bot.offboard_velocity(float("inf"), 0.0, 0.0, 0.0, Frame.BODY_FRD)
        with self.assertRaises(TypeError):
            bot.set_yaw(True, Frame.LOCAL_NED)  # type: ignore[arg-type]
        self.assertEqual(transport.flight, [])

    def test_drive_stays_planar(self):
        bot, transport = self._pair(_aerial_spec())
        self.assertIs(bot.drive(vx=0.4, vy=-0.2, w=0.1), bot)
        self.assertEqual(transport.flight, [])
        self.assertEqual(transport.velocities, [(0.4, -0.2, 0.1)])
        self.assertFalse(hasattr(Velocity(0.4, -0.2, 0.1), "linear_z"))

    def test_aerial_transport_must_implement_send_flight(self):
        spec = _aerial_spec()
        transport = SimTransport(spec)
        transport.connect()
        bot = Robot(spec, transport, NativeRuntime())
        with self.assertRaises(AttributeError):
            bot.hold()

    def test_gated_calls_go_through_command_and_not_around_the_gate(self):
        inner = RecordingTransport()
        spec = _aerial_spec()
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
            with self.assertLogs("ohho.safety", level="WARNING"):
                self.assertIs(bot.arm_flight(), bot)
                self.assertIs(bot.disarm_flight(), bot)
                self.assertIs(bot.takeoff(2.5), bot)
                self.assertIs(bot.land(), bot)
                self.assertIs(bot.rtl(), bot)
                self.assertIs(bot.hold(), bot)
                self.assertIs(
                    bot.offboard_position(1.0, 2.0, 3.0, 0.4, Frame.LOCAL_NED),
                    bot,
                )
                self.assertIs(
                    bot.offboard_velocity(0.1, -0.2, 0.3, -0.4, Frame.BODY_FRD),
                    bot,
                )
                self.assertIs(bot.set_yaw(1.2, Frame.LOCAL_ENU), bot)
            before = len(gate.commands)
            with self.assertRaises(TypeError):
                bot.offboard_position(0.0, 0.0, 0.0, 0.0, "LOCAL_NED")  # type: ignore[arg-type]
            self.assertEqual(len(gate.commands), before)

            self.assertFalse(gate.armed)
            self.assertEqual(gate.arm_calls, 0)
            self.assertEqual(inner.flight, [])
            self.assertEqual(inner.velocities, [(0.0, 0.0, 0.0)] * 9)
            self.assertEqual(
                [name for name, _args in gate.commands],
                [
                    "arm",
                    "disarm",
                    "takeoff",
                    "land",
                    "rtl",
                    "hold",
                    "offboard_position",
                    "offboard_velocity",
                    "set_yaw",
                ],
            )
            by_name = {name: args for name, args in gate.commands}
            self.assertEqual(by_name["arm"], ())
            self.assertEqual(by_name["disarm"], ())
            self.assertNotIn(21196, by_name["disarm"])
            self.assertEqual(by_name["takeoff"], (2.5,))
            self.assertEqual(by_name["offboard_position"][:4], (1.0, 2.0, 3.0, 0.4))
            self.assertIs(by_name["offboard_position"][4], Frame.LOCAL_NED)
            self.assertEqual(by_name["offboard_velocity"][:4], (0.1, -0.2, 0.3, -0.4))
            self.assertIs(by_name["offboard_velocity"][4], Frame.BODY_FRD)
            self.assertEqual(by_name["set_yaw"][0], 1.2)
            self.assertIs(by_name["set_yaw"][1], Frame.LOCAL_ENU)

            body = "".join(
                path.read_text(encoding="utf-8") for path in Path(tmp).glob("*.jsonl")
            )
            self.assertIn("not_allowlisted", body)
            self.assertNotIn("21196", body)
            self.assertNotIn("force_disarm", body)


if __name__ == "__main__":
    unittest.main()
