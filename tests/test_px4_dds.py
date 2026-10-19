"""PX4 uXRCE-DDS transport against a mock ROS graph. No Gazebo and no agent."""

from __future__ import annotations

import importlib.util
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from ohho import capabilities as caps
from ohho.adapters import available_adapters, resolve_transport
from ohho.adapters.px4_dds import (
    FORCE_DISARM_PARAM2,
    PINNED_OFFBOARD_FIELDS,
    PX4_VERSION,
    VEHICLE_CMD_COMPONENT_ARM_DISARM,
    VEHICLE_CMD_DO_FLIGHTTERMINATION,
    VEHICLE_CMD_DO_SET_MODE,
    VEHICLE_CMD_NAV_LAND,
    VEHICLE_CMD_NAV_RETURN_TO_LAUNCH,
    VEHICLE_CMD_NAV_TAKEOFF,
    Px4CommandRefused,
    Px4DdsError,
    Px4DdsTransport,
    altitude_m_of,
    check_vehicle_command,
    estop_action,
    is_sim_target,
    landed_state_of,
    offboard_field_mismatch,
    parse_px4_dds_uri,
)
from ohho.adapters.unitree import UnitreeDdsTransport
from ohho.aerial import Frame
from ohho.registry import RobotSpec, get_spec
from ohho.robot import Robot
from ohho.runtime.native import NativeRuntime
from ohho.schema import Velocity

_ROOT = Path(__file__).resolve().parents[1]


class _Msg:
    def __init__(self) -> None:
        return None


class GoodOffboard(_Msg):
    @staticmethod
    def get_fields_and_field_types() -> dict[str, str]:
        return dict(PINNED_OFFBOARD_FIELDS)


class BadOffboard(_Msg):
    """Pre-1.17 shape: `actuator` instead of the two v1.17 flags."""

    @staticmethod
    def get_fields_and_field_types() -> dict[str, str]:
        fields = dict(PINNED_OFFBOARD_FIELDS)
        del fields["thrust_and_torque"]
        del fields["direct_actuator"]
        fields["actuator"] = "bool"
        return fields


class _Factory:
    def __init__(self, offboard: type) -> None:
        self.OffboardControlMode = offboard
        self.TrajectorySetpoint = _Msg
        self.VehicleCommand = _Msg
        self.VehicleStatus = _Msg
        self.VehicleLocalPosition = _Msg
        self.VehicleAttitude = _Msg
        self.BatteryStatus = _Msg
        self.VehicleLandDetected = _Msg
        self.FailsafeFlags = _Msg


class FakePublisher:
    def __init__(self, msg_type: type, topic: str, qos: object) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.published: list[object] = []

    def publish(self, msg: object) -> None:
        self.published.append(msg)


class FakeSubscription:
    def __init__(self, msg_type: type, topic: str, callback, qos: object) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.callback = callback
        self.qos = qos


class FakeTimer:
    def __init__(self, period: float, callback) -> None:
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeNode:
    """Records the DDS graph. Timers do not fire until a test asks them to."""

    def __init__(self) -> None:
        self.publishers: dict[str, FakePublisher] = {}
        self.subs: list[FakeSubscription] = []
        self.timers: list[FakeTimer] = []

    def create_publisher(self, msg_type, topic, qos):
        pub = FakePublisher(msg_type, topic, qos)
        self.publishers[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = FakeSubscription(msg_type, topic, callback, qos)
        self.subs.append(sub)
        return sub

    def create_timer(self, period, callback):
        timer = FakeTimer(period, callback)
        self.timers.append(timer)
        return timer

    def publisher(self, topic: str) -> FakePublisher:
        return self.publishers[topic]

    def inject(self, topic: str, msg: object) -> None:
        matched = [sub for sub in self.subs if sub.topic == topic]
        if not matched:
            raise AssertionError(f"no subscription for {topic}")
        for sub in matched:
            sub.callback(msg)

    def fire_timers(self) -> None:
        for timer in list(self.timers):
            if not timer.cancelled:
                timer.callback()


def _qos_bits(qos: object) -> tuple[str, str, int]:
    reliability = getattr(qos, "reliability", None)
    durability = getattr(qos, "durability", None)
    rel = getattr(reliability, "name", str(reliability)).lower()
    dur = getattr(durability, "name", str(durability)).lower()
    return rel, dur, int(getattr(qos, "depth"))


def _good() -> _Factory:
    return _Factory(GoodOffboard)


def _open(
    uri: str = "px4+dds://127.0.0.1:8888",
    *,
    factory: _Factory | None = None,
    environ: dict[str, str] | None = None,
    clock=None,
) -> tuple[Px4DdsTransport, FakeNode]:
    node = FakeNode()
    tp = Px4DdsTransport(
        get_spec("sim"),
        uri,
        node=node,
        msg_factory=factory or _good(),
        environ=environ,
        clock=clock,
    )
    tp.connect()
    return tp, node


def _commands(node: FakeNode, uri_ns: str = "") -> list:
    topic = f"/{uri_ns}/fmu/in/vehicle_command" if uri_ns else "/fmu/in/vehicle_command"
    return list(node.publisher(topic).published)


def _setpoints(node: FakeNode) -> list:
    return list(node.publisher("/fmu/in/trajectory_setpoint").published)


def _modes(node: FakeNode) -> list:
    return list(node.publisher("/fmu/in/offboard_control_mode").published)


def _prearm(
    node: FakeNode, *, passed: bool = True, nav: int = 2, armed: int = 1
) -> None:
    node.inject(
        "/fmu/out/vehicle_status",
        SimpleNamespace(
            arming_state=armed,
            nav_state=nav,
            pre_flight_checks_pass=passed,
            gcs_connection_lost=False,
            high_latency_data_link_lost=False,
        ),
    )


def _nums(values) -> list[float]:
    return [float(v) for v in values]


class TestSimTarget(unittest.TestCase):
    def test_loopback_host_is_sim_and_scheme_is_not_an_input(self):
        self.assertTrue(is_sim_target("127.0.0.1", 8888))
        self.assertTrue(is_sim_target("127.0.0.1", 1))
        self.assertTrue(is_sim_target("127.0.0.2", 9))
        self.assertTrue(is_sim_target("localhost", 8888))
        self.assertTrue(is_sim_target("::1", 8888))
        self.assertFalse(is_sim_target("192.168.1.20", 8888))
        self.assertFalse(is_sim_target("10.0.0.8", 8888))
        self.assertFalse(is_sim_target("sim", 8888))
        self.assertFalse(is_sim_target("simulated", 8888))
        self.assertFalse(is_sim_target("dds", 8888))
        self.assertFalse(is_sim_target("px4+dds", 8888))

    def test_parse_loopback_and_namespace(self):
        host, port, ns = parse_px4_dds_uri("px4+dds://127.0.0.1:8888?ns=uav1")
        self.assertEqual((host, port, ns), ("127.0.0.1", 8888, "uav1"))
        host, port, ns = parse_px4_dds_uri("px4+dds://[::1]:8888")
        self.assertEqual(host, "::1")
        self.assertTrue(is_sim_target(host, port))
        host, port, ns = parse_px4_dds_uri("px4+dds://127.0.0.1")
        self.assertEqual(port, 8888)
        self.assertEqual(ns, "")

    def test_parse_rejects_credentials_without_echoing_them(self):
        with self.assertRaises(Px4DdsError) as caught:
            parse_px4_dds_uri("px4+dds://user:secret@127.0.0.1:8888")
        self.assertNotIn("secret", str(caught.exception))

    def test_parse_rejects_foreign_scheme_namespace_and_query(self):
        with self.assertRaises(Px4DdsError):
            parse_px4_dds_uri("dds://127.0.0.1:8888")
        with self.assertRaises(Px4DdsError):
            parse_px4_dds_uri("px4+dds://127.0.0.1:8888?ns=../etc")
        with self.assertRaises(Px4DdsError):
            parse_px4_dds_uri("px4+dds://127.0.0.1:8888?arm=1")
        with self.assertRaises(Px4DdsError):
            parse_px4_dds_uri("px4+dds://127.0.0.1:99999")

    def test_resolve_selects_px4_and_leaves_dds_on_unitree(self):
        px4 = resolve_transport("px4+dds://127.0.0.1:8888", get_spec("sim"))
        self.assertIsInstance(px4, Px4DdsTransport)
        self.assertEqual(px4.protocol, "px4-dds")
        self.assertTrue(px4.sim_target)
        self.assertFalse(px4.requires_hardware_arming)
        unitree = resolve_transport("dds://eth0", get_spec("unitree-go2"))
        self.assertIsInstance(unitree, UnitreeDdsTransport)
        self.assertNotIsInstance(unitree, Px4DdsTransport)
        self.assertEqual(unitree.protocol, "dds")

    def test_non_loopback_requires_hardware_arming(self):
        tp = resolve_transport("px4+dds://192.168.1.20:8888", get_spec("sim"))
        self.assertFalse(tp.sim_target)
        self.assertTrue(tp.requires_hardware_arming)


class TestMessagePin(unittest.TestCase):
    def test_boolean_alias_matches_v1_17(self):
        fields = {
            key: ("boolean" if value == "bool" else value)
            for key, value in PINNED_OFFBOARD_FIELDS.items()
        }
        self.assertIsNone(offboard_field_mismatch(fields))

    def test_actuator_field_refuses_connect_before_publishers(self):
        node = FakeNode()
        tp = Px4DdsTransport(
            get_spec("sim"),
            "px4+dds://127.0.0.1:8888",
            node=node,
            msg_factory=_Factory(BadOffboard),
        )
        with self.assertRaises(Px4DdsError) as caught:
            tp.connect()
        text = str(caught.exception)
        self.assertIn("release/1.17", text)
        self.assertIn(PX4_VERSION, text)
        self.assertIn("actuator", text)
        self.assertIn("thrust_and_torque", text)
        self.assertEqual(node.publishers, {})
        self.assertEqual(tp.publisher_topics, [])

    def test_extra_field_refuses_connect(self):
        fields = dict(PINNED_OFFBOARD_FIELDS)
        fields["actuator"] = "bool"
        text = offboard_field_mismatch(fields)
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("v1.18", text)


class TestQosAndHeartbeat(unittest.TestCase):
    def test_qos_is_best_effort_depth_1(self):
        tp, node = _open()
        self.assertLessEqual(tp.offboard_period_s, 0.1)
        self.assertGreaterEqual(1.0 / tp.offboard_period_s, 10.0)
        self.assertEqual(node.timers[0].period, tp.offboard_period_s)
        self.assertEqual(len(node.publishers), 3)
        self.assertEqual(len(node.subs), 6)
        for pub in node.publishers.values():
            rel, dur, depth = _qos_bits(pub.qos)
            self.assertIn("best_effort", rel)
            self.assertIn("volatile", dur)
            self.assertEqual(depth, 1)
            self.assertNotIsInstance(pub.qos, int)
        for sub in node.subs:
            rel, dur, depth = _qos_bits(sub.qos)
            self.assertIn("best_effort", rel)
            self.assertIn("transient_local", dur)
            self.assertEqual(depth, 1)
        self.assertEqual(
            set(tp.subscription_topics),
            {
                "/fmu/out/vehicle_status",
                "/fmu/out/vehicle_local_position",
                "/fmu/out/vehicle_attitude",
                "/fmu/out/battery_status",
                "/fmu/out/vehicle_land_detected",
                "/fmu/out/failsafe_flags",
            },
        )

    def test_namespace_prefixes_topics(self):
        _tp, node = _open("px4+dds://127.0.0.1:8888?ns=uav1")
        self.assertIn("/uav1/fmu/in/offboard_control_mode", node.publishers)
        self.assertIn("/uav1/fmu/out/vehicle_status", [sub.topic for sub in node.subs])

    def test_heartbeat_publishes_mode_and_nan_setpoint_not_a_command(self):
        _tp, node = _open()
        self.assertEqual(_modes(node), [])
        node.fire_timers()
        self.assertEqual(_commands(node), [])
        mode = _modes(node)[-1]
        self.assertTrue(mode.position)
        self.assertFalse(mode.velocity)
        self.assertFalse(mode.acceleration)
        self.assertFalse(mode.attitude)
        self.assertFalse(mode.body_rate)
        self.assertFalse(mode.thrust_and_torque)
        self.assertFalse(mode.direct_actuator)
        setpoint = _setpoints(node)[-1]
        for value in _nums(setpoint.position) + _nums(setpoint.velocity):
            self.assertTrue(math.isnan(value))

    def test_arm_refused_until_offboard_has_been_published(self):
        _tp, node = _open()
        _prearm(node)
        with self.assertRaises(Px4CommandRefused) as caught:
            _tp.send_flight("arm")
        self.assertIn("offboard stream", str(caught.exception))
        self.assertEqual(_commands(node), [])
        node.fire_timers()
        _tp.send_flight("arm")
        cmd = _commands(node)[-1]
        self.assertEqual(cmd.command, VEHICLE_CMD_COMPONENT_ARM_DISARM)
        self.assertEqual(cmd.command, 400)
        self.assertEqual(cmd.param1, 1.0)
        self.assertEqual(cmd.param2, 0.0)
        self.assertTrue(cmd.from_external)
        self.assertEqual(cmd.target_system, 1)
        self.assertNotEqual(cmd.param2, FORCE_DISARM_PARAM2)

    def test_arm_refused_when_the_stream_goes_stale(self):
        clock = {"t": 0.0}

        def now() -> float:
            return clock["t"]

        tp, node = _open(clock=now)
        _prearm(node)
        node.fire_timers()
        clock["t"] = 1.0
        with self.assertRaises(Px4CommandRefused):
            tp.send_flight("arm")
        self.assertEqual(_commands(node), [])

    def test_arm_refused_when_prearm_has_not_passed(self):
        tp, node = _open()
        node.fire_timers()
        with self.assertRaises(Px4CommandRefused) as caught:
            tp.send_flight("arm")
        self.assertIn("pre-arm", str(caught.exception))
        _prearm(node, passed=False)
        with self.assertRaises(Px4CommandRefused):
            tp.send_flight("arm")
        self.assertEqual(_commands(node), [])


class TestEstopAndCommands(unittest.TestCase):
    def test_estop_table(self):
        self.assertEqual(estop_action("unknown", 0.0), "hold_then_land")
        self.assertEqual(estop_action("on_ground", 0.25), "disarm")
        self.assertEqual(estop_action("on_ground", 0.26), "land")
        self.assertEqual(estop_action("on_ground", None), "land")
        self.assertEqual(estop_action("in_air", 0.0), "land")
        self.assertEqual(estop_action("takeoff", 0.0), "land")
        self.assertEqual(estop_action("landing", 1.0), "land")

    def test_maybe_landed_is_not_on_the_ground(self):
        state = landed_state_of(SimpleNamespace(landed=False, maybe_landed=True), None)
        self.assertEqual(state, "in_air")
        self.assertEqual(
            landed_state_of(SimpleNamespace(landed=False), 17),
            "takeoff",
        )

    def test_estop_in_air_sends_land_not_disarm(self):
        tp, node = _open()
        node.inject(
            "/fmu/out/vehicle_land_detected",
            SimpleNamespace(landed=False, maybe_landed=False),
        )
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(
                z_valid=True, z=-2.5, xy_valid=False, dist_bottom_valid=False
            ),
        )
        tp.emergency_stop()
        cmds = _commands(node)
        self.assertEqual([cmd.command for cmd in cmds], [VEHICLE_CMD_NAV_LAND])
        self.assertEqual(cmds[0].command, 21)
        self.assertEqual(cmds[0].param2, 0.0)
        self.assertEqual(_setpoints(node), [])
        self.assertNotIn(VEHICLE_CMD_COMPONENT_ARM_DISARM, [c.command for c in cmds])
        self.assertNotIn(VEHICLE_CMD_DO_FLIGHTTERMINATION, [c.command for c in cmds])
        for cmd in cmds:
            self.assertNotEqual(cmd.param2, 21196)
            self.assertNotEqual(cmd.param2, FORCE_DISARM_PARAM2)

    def test_estop_unknown_sends_hold_then_land(self):
        tp, node = _open()
        tp.emergency_stop()
        cmds = _commands(node)
        self.assertEqual(
            [(c.command, c.param1, c.param2, c.param3) for c in cmds],
            [
                (VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0),
                (VEHICLE_CMD_NAV_LAND, 0.0, 0.0, 0.0),
            ],
        )
        self.assertEqual(_setpoints(node), [])

    def test_estop_on_ground_disarms_normally(self):
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=True))
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(z_valid=True, z=0.0, dist_bottom_valid=False),
        )
        tp.emergency_stop()
        cmd = _commands(node)[-1]
        self.assertEqual(cmd.command, 400)
        self.assertEqual(cmd.param1, 0.0)
        self.assertEqual(cmd.param2, 0.0)
        self.assertNotIn(21, [c.command for c in _commands(node)])

    def test_estop_on_ground_but_high_sends_land(self):
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=True))
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(z_valid=True, z=-4.0, dist_bottom_valid=False),
        )
        tp.emergency_stop()
        self.assertEqual([c.command for c in _commands(node)], [21])

    def test_low_distance_sensor_does_not_disarm_a_high_vehicle(self):
        local = SimpleNamespace(
            dist_bottom_valid=True,
            dist_bottom=0.1,
            z_valid=True,
            z=-8.0,
        )
        self.assertAlmostEqual(altitude_m_of(local) or -1, 8.0)
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=True))
        node.inject("/fmu/out/vehicle_local_position", local)
        tp.emergency_stop()
        self.assertEqual([c.command for c in _commands(node)], [21])

    def test_disarm_flight_in_air_publishes_nothing(self):
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=False))
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(z_valid=True, z=0.0, dist_bottom_valid=False),
        )
        with self.assertRaises(Px4CommandRefused) as caught:
            tp.send_flight("disarm")
        self.assertIn("21196", str(caught.exception))
        self.assertEqual(_commands(node), [])

    def test_disarm_flight_on_ground_is_normal(self):
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=True))
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(z_valid=True, z=-0.1, dist_bottom_valid=False),
        )
        tp.send_flight("disarm")
        cmd = _commands(node)[-1]
        self.assertEqual((cmd.command, cmd.param1, cmd.param2), (400, 0.0, 0.0))

    def test_land_is_a_vehicle_command_not_a_zero_setpoint(self):
        tp, node = _open()
        tp.send_flight("land")
        self.assertEqual([c.command for c in _commands(node)], [VEHICLE_CMD_NAV_LAND])
        self.assertEqual(_setpoints(node), [])

    def test_zero_velocity_is_hover_not_land(self):
        tp, node = _open()
        tp.send_velocity(Velocity())
        self.assertNotIn(VEHICLE_CMD_NAV_LAND, [c.command for c in _commands(node)])
        self.assertNotIn(400, [c.command for c in _commands(node)])
        setpoint = _setpoints(node)[-1]
        self.assertEqual(_nums(setpoint.velocity), [0.0, 0.0, 0.0])
        for value in _nums(setpoint.position):
            self.assertTrue(math.isnan(value))
        mode = _commands(node)[-1]
        self.assertEqual((mode.command, mode.param2), (VEHICLE_CMD_DO_SET_MODE, 6.0))

    def test_estop_blocks_a_later_zero_velocity(self):
        tp, node = _open()
        tp.emergency_stop()
        before = len(_setpoints(node))
        tp.send_velocity(Velocity(1.0, 0.0, 0.0))
        node.fire_timers()
        self.assertEqual(len(_setpoints(node)), before)

    def test_release_stop_does_not_arm_or_resume_offboard(self):
        tp, node = _open()
        node.inject("/fmu/out/vehicle_land_detected", SimpleNamespace(landed=False))
        tp.emergency_stop()
        tp.release_stop()
        commands = [c.command for c in _commands(node)]
        self.assertEqual(commands, [VEHICLE_CMD_NAV_LAND])
        self.assertNotIn(400, commands)

    def test_command_guard_blocks_termination_and_force_disarm(self):
        with self.assertRaises(Px4CommandRefused):
            check_vehicle_command(VEHICLE_CMD_DO_FLIGHTTERMINATION, 1.0, 0.0, 0.0)
        with self.assertRaises(Px4CommandRefused):
            check_vehicle_command(400, 0.0, FORCE_DISARM_PARAM2, 0.0)
        with self.assertRaises(Px4CommandRefused):
            check_vehicle_command(999, 0.0, 0.0, 0.0)
        check_vehicle_command(VEHICLE_CMD_NAV_LAND, 0.0, 0.0, 0.0)
        check_vehicle_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0, 0.0)
        check_vehicle_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 0.0, 0.0)
        self.assertEqual(VEHICLE_CMD_DO_FLIGHTTERMINATION, 185)
        self.assertEqual(FORCE_DISARM_PARAM2, 21196.0)

    def test_takeoff_rtl_and_hold(self):
        tp, node = _open()
        tp.send_flight("takeoff", alt_m=2.5)
        tp.send_flight("rtl")
        tp.send_flight("hold")
        cmds = _commands(node)
        self.assertEqual(cmds[0].command, VEHICLE_CMD_NAV_TAKEOFF)
        self.assertEqual(cmds[0].param7, 2.5)
        self.assertEqual(cmds[1].command, VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.assertEqual(
            (cmds[2].command, cmds[2].param2, cmds[2].param3), (176, 4.0, 3.0)
        )
        with self.assertRaises(ValueError):
            tp.send_flight("takeoff", alt_m=-1)

    def test_enu_position_is_converted_to_ned(self):
        tp, node = _open()
        tp.send_flight(
            "offboard_position",
            x=1.0,
            y=2.0,
            z=3.0,
            yaw=0.0,
            frame=Frame.LOCAL_ENU,
        )
        setpoint = _setpoints(node)[-1]
        self.assertEqual(_nums(setpoint.position), [2.0, 1.0, -3.0])
        self.assertAlmostEqual(setpoint.yaw, math.pi / 2.0)
        mode = _modes(node)[-1]
        self.assertTrue(mode.position)
        self.assertFalse(mode.velocity)

    def test_body_position_is_refused(self):
        tp, node = _open()
        with self.assertRaises(Px4CommandRefused):
            tp.send_flight(
                "offboard_position",
                x=1.0,
                y=0.0,
                z=0.0,
                yaw=0.0,
                frame=Frame.BODY_FRD,
            )
        self.assertEqual(_setpoints(node), [])
        with self.assertRaises(TypeError):
            tp.send_flight(
                "set_yaw",
                yaw=0.2,
                frame="LOCAL_NED",
            )

    def test_body_velocity_uses_attitude(self):
        tp, node = _open()
        half = math.sqrt(2.0) / 2.0
        node.inject(
            "/fmu/out/vehicle_attitude",
            SimpleNamespace(q=(half, 0.0, half, 0.0)),
        )
        tp.send_flight(
            "offboard_velocity",
            vx=1.0,
            vy=0.0,
            vz=0.0,
            yaw_rate=1.0,
            frame=Frame.BODY_FRD,
        )
        setpoint = _setpoints(node)[-1]
        vx, vy, vz = _nums(setpoint.velocity)
        self.assertAlmostEqual(vx, 0.0, places=6)
        self.assertAlmostEqual(vy, 0.0, places=6)
        self.assertAlmostEqual(vz, -1.0, places=6)
        self.assertAlmostEqual(setpoint.yawspeed, 0.0, places=6)

    def test_body_velocity_without_attitude_is_refused(self):
        tp, node = _open()
        with self.assertRaises(Px4CommandRefused):
            tp.send_flight(
                "offboard_velocity",
                vx=0.2,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0,
                frame=Frame.BODY_FRD,
            )
        self.assertEqual(_setpoints(node), [])


class TestTelemetry(unittest.TestCase):
    def test_drone_custom_fields(self):
        _tp, node = _open()
        node.inject(
            "/fmu/out/vehicle_status",
            SimpleNamespace(
                arming_state=2,
                nav_state=14,
                pre_flight_checks_pass=True,
                gcs_connection_lost=True,
                high_latency_data_link_lost=False,
            ),
        )
        node.inject(
            "/fmu/out/vehicle_local_position",
            SimpleNamespace(
                xy_valid=True,
                z_valid=True,
                v_xy_valid=True,
                v_z_valid=True,
                x=1.0,
                y=2.0,
                z=-2.5,
                vx=0.4,
                vy=-0.1,
                vz=0.0,
                heading=0.3,
                dist_bottom_valid=False,
            ),
        )
        node.inject(
            "/fmu/out/vehicle_attitude",
            SimpleNamespace(q=(1.0, 0.0, 0.0, 0.0)),
        )
        node.inject(
            "/fmu/out/battery_status",
            SimpleNamespace(remaining=0.5, voltage_v=16.2, current_a=3.5),
        )
        node.inject(
            "/fmu/out/vehicle_land_detected",
            SimpleNamespace(landed=False),
        )
        node.inject(
            "/fmu/out/failsafe_flags",
            SimpleNamespace(
                offboard_control_signal_lost=True,
                manual_control_signal_lost=False,
                geofence_breached=True,
                battery_warning=0,
                battery_unhealthy=False,
                battery_low_remaining_time=False,
                gcs_connection_lost=False,
            ),
        )
        tele = _tp.read()
        drone = tele.custom["drone"]
        self.assertTrue(drone["armed"])
        self.assertEqual(drone["flight_mode"], "offboard")
        self.assertEqual(drone["landed_state"], "in_air")
        self.assertEqual(drone["pose_frame"], "LOCAL_NED")
        self.assertEqual(drone["velocity_frame"], "LOCAL_NED")
        self.assertAlmostEqual(drone["pose"]["x"], 1.0)
        self.assertAlmostEqual(drone["pose"]["z"], -2.5)
        self.assertAlmostEqual(drone["pose"]["yaw"], 0.0)
        self.assertIsNone(drone["velocity"]["yaw_rate"])
        self.assertEqual(tele.battery, 0.5)
        self.assertEqual(drone["voltage_v"], 16.2)
        self.assertEqual(drone["current_a"], 3.5)
        self.assertEqual(drone["gps_fix"], "none")
        self.assertEqual(drone["failsafe"], ["offboard_loss", "geofence", "datalink"])
        self.assertTrue(drone["link_ok"])
        self.assertTrue(drone["experimental"])
        self.assertFalse(drone["flight_ready"])
        assert tele.odom is not None
        self.assertEqual(tele.odom.omega, 0.0)
        self.assertAlmostEqual(tele.odom.x, 1.0)

    def test_invalid_battery_sentinels_stay_empty(self):
        tp, node = _open()
        node.inject(
            "/fmu/out/battery_status",
            SimpleNamespace(remaining=-1.0, voltage_v=0.0, current_a=-1.0),
        )
        tele = tp.read()
        self.assertIsNone(tele.battery)
        self.assertIsNone(tele.custom["drone"]["voltage_v"])
        self.assertIsNone(tele.custom["drone"]["current_a"])
        self.assertFalse(tele.custom["drone"]["flight_ready"])


class TestHardwareArming(unittest.TestCase):
    def test_non_loopback_without_arming_is_refused(self):
        before = os.environ.get("OHHO_ARM_HARDWARE")
        node = FakeNode()
        tp = Px4DdsTransport(
            get_spec("sim"),
            "px4+dds://10.0.0.8:8888",
            node=node,
            msg_factory=_good(),
            environ={},
        )
        with self.assertRaises(Px4DdsError) as caught:
            tp.connect()
        self.assertIn("not a loopback", str(caught.exception))
        self.assertEqual(node.publishers, {})
        self.assertEqual(os.environ.get("OHHO_ARM_HARDWARE"), before)
        for value in ("", "0", "true", "yes"):
            other = Px4DdsTransport(
                get_spec("sim"),
                "px4+dds://192.168.1.50:8888",
                node=FakeNode(),
                msg_factory=_good(),
                environ={"OHHO_ARM_HARDWARE": value},
            )
            with self.assertRaises(Px4DdsError):
                other.connect()

    def test_arming_mapping_is_not_written_into_the_process_environment(self):
        before = os.environ.get("OHHO_ARM_HARDWARE")
        node = FakeNode()
        tp = Px4DdsTransport(
            get_spec("sim"),
            "px4+dds://192.168.1.20:8888",
            node=node,
            msg_factory=_good(),
            environ={"OHHO_ARM_HARDWARE": "1"},
        )
        tp.connect()
        self.assertEqual(len(node.publishers), 3)
        self.assertEqual(_commands(node), [])
        self.assertEqual(os.environ.get("OHHO_ARM_HARDWARE"), before)
        tp.disconnect()


class TestRobotDispatch(unittest.TestCase):
    def test_aerial_robot_arm_flight_reaches_the_transport(self):
        tp, node = _open()
        _prearm(node)
        node.fire_timers()
        spec = RobotSpec(
            id="px4-dds-test",
            name="PX4 DDS test",
            category="aerial",
            capabilities=(caps.AERIAL,),
        )
        bot = Robot(spec, tp, NativeRuntime())
        try:
            bot.arm_flight()
            bot.takeoff(2.5)
        finally:
            bot.disconnect()
        cmds = _commands(node)
        self.assertEqual(cmds[0].command, 400)
        self.assertEqual(cmds[0].param1, 1.0)
        self.assertEqual(cmds[1].command, 22)
        self.assertEqual(cmds[1].param7, 2.5)

    def test_ground_robot_arm_flight_does_not_publish(self):
        tp, node = _open()
        _prearm(node)
        node.fire_timers()
        bot = Robot(get_spec("sim"), tp, NativeRuntime())
        try:
            bot.arm_flight()
        finally:
            bot.disconnect()
        self.assertEqual(_commands(node), [])


class TestPackaging(unittest.TestCase):
    def test_base_dependencies_stay_empty(self):
        text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", text)
        self.assertIn("ros2 = []", text)

    def test_px4_dds_listed_only_when_both_imports_exist(self):
        present = (
            importlib.util.find_spec("rclpy") is not None
            and importlib.util.find_spec("px4_msgs") is not None
        )
        self.assertEqual("px4+dds" in available_adapters(), present)
        self.assertIn("sim", available_adapters())

    def test_docs_pin_the_versions(self):
        text = (_ROOT / "docs" / "px4-dds.md").read_text(encoding="utf-8")
        for needle in (
            "1.17.0",
            "release/1.17",
            "Jazzy",
            "Humble",
            "2.4.3",
            "v1.18",
            "best-effort",
            "transient-local",
            "volatile",
            "21196",
            "thrust_and_torque",
            "direct_actuator",
            "127.0.0.1",
            "sim_target",
            "OHH-288",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
