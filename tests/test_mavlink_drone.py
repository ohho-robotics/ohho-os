"""MAVLink backend tests (OHH-305).

A recording link stands in for MAVSDK. Nothing opens a socket, and nothing
imports mavsdk. E-stop cases fail if a disarm is sent in the air or if
param2 is the force-disarm magic.
"""

from __future__ import annotations

import enum
import importlib.util
import inspect
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ohho.adapters import AdapterUnavailable, available_adapters, resolve_transport
from ohho.adapters.mavlink_drone import (
    MAV_FRAME_BODY_NED,
    MAV_FRAME_LOCAL_NED,
    TYPEMASK_VELOCITY_YAW_RATE,
    MavlinkCommand,
    MavlinkDroneTransport,
    MavsdkLink,
    RecordingLink,
    _ACTION_METHODS,
    map_offboard_velocity,
    parse_mavlink_uri,
    position_target_fields,
    to_mavsdk_url,
)
from ohho.aerial import Frame
from ohho.drone.estop import (
    HOLD_COMMAND,
    LAND_COMMAND,
    MAV_CMD_DO_FLIGHTTERMINATION,
    MAV_CMD_NAV_LAND,
    MAV_FORCE_DISARM_MAGIC,
)
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime.native import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import Velocity

_SITL = "mavlink+udp://127.0.0.1:14540"
_MAGIC = float(MAV_FORCE_DISARM_MAGIC)


def _air(link: RecordingLink, state: str = "in_air", z: float = -2.5) -> None:
    link.sample["landed_state"] = state
    link.sample["z"] = z
    link.sample["armed"] = True


def _ground(link: RecordingLink, z: float = 0.0) -> None:
    link.sample["landed_state"] = "on_ground"
    link.sample["z"] = z
    link.sample["armed"] = True


class _Plugins:
    """Records plugin calls. kill / terminate / arm_force raise immediately."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def _record(self, name: str, payload: object = None) -> None:
        self.calls.append((name, payload))

    def arm(self) -> None:
        self._record("arm")

    def disarm(self) -> None:
        self._record("disarm")

    def set_takeoff_altitude(self, altitude: float) -> None:
        self._record("set_takeoff_altitude", altitude)

    def takeoff(self) -> None:
        self._record("takeoff")

    def land(self) -> None:
        self._record("land")

    def hold(self) -> None:
        self._record("hold")

    def return_to_launch(self) -> None:
        self._record("return_to_launch")

    def set_velocity_body(self, values: object) -> None:
        self._record("set_velocity_body", values)

    def set_position_ned(self, values: object) -> None:
        self._record("set_position_ned", values)

    def start(self) -> None:
        self._record("start")

    def kill(self) -> None:
        raise AssertionError("kill")

    def terminate(self) -> None:
        raise AssertionError("terminate")

    def arm_force(self) -> None:
        raise AssertionError("arm_force")


class TestMavlinkUri(unittest.TestCase):
    def test_px4_sitl_udp_is_mavsdk_udpin(self) -> None:
        target = parse_mavlink_uri(_SITL)
        self.assertEqual(target.mavsdk_url, "udpin://127.0.0.1:14540")
        self.assertEqual(to_mavsdk_url(_SITL), "udpin://127.0.0.1:14540")
        self.assertTrue(target.simulation_target)
        self.assertFalse(target.requires_hardware_arming)

    def test_loopback_names_are_simulation_targets(self) -> None:
        for uri in (
            "mavlink+udp://localhost:14540",
            "mavlink+udp://[::1]:14540",
            "mavlink+udp://127.0.0.2:14540",
        ):
            target = parse_mavlink_uri(uri)
            self.assertTrue(target.simulation_target, uri)
            self.assertFalse(target.requires_hardware_arming, uri)
        self.assertEqual(
            to_mavsdk_url("mavlink+udp://[::1]:14540"), "udpin://[::1]:14540"
        )

    def test_remote_udp_and_serial_require_hardware_arming(self) -> None:
        remote = parse_mavlink_uri("mavlink+udp://192.168.1.20:14550")
        self.assertEqual(remote.mavsdk_url, "udpin://192.168.1.20:14550")
        self.assertFalse(remote.simulation_target)
        self.assertTrue(remote.requires_hardware_arming)
        serial = parse_mavlink_uri("mavlink+serial:///dev/ttyACM0?baud=921600")
        self.assertEqual(serial.mavsdk_url, "serial:///dev/ttyACM0:921600")
        self.assertEqual(serial.baud, 921600)
        self.assertFalse(serial.simulation_target)
        self.assertTrue(serial.requires_hardware_arming)

    def test_serial_baud_default_and_windows_port(self) -> None:
        default = parse_mavlink_uri("mavlink+serial:///dev/ttyACM0")
        self.assertEqual(default.mavsdk_url, "serial:///dev/ttyACM0:921600")
        com = parse_mavlink_uri("mavlink+serial://COM3?baud=57600")
        self.assertEqual(com.device, "COM3")
        self.assertEqual(com.baud, 57600)
        self.assertEqual(com.mavsdk_url, "serial://COM3:57600")
        self.assertTrue(com.requires_hardware_arming)

    def test_sitl_query_allows_docker_local_only_when_asked(self) -> None:
        plain = parse_mavlink_uri("mavlink+udp://host.docker.internal:14540")
        self.assertFalse(plain.simulation_target)
        flagged = parse_mavlink_uri("mavlink+udp://host.docker.internal:14540?sitl=1")
        self.assertTrue(flagged.simulation_target)
        self.assertFalse(flagged.requires_hardware_arming)

    def test_bad_uris_raise(self) -> None:
        for uri in (
            "mavlink+udp://127.0.0.1",
            "mavlink+serial://",
            "mavlink://127.0.0.1:14540",
        ):
            with self.assertRaises(AdapterUnavailable):
                parse_mavlink_uri(uri)


class TestMavlinkResolve(unittest.TestCase):
    def test_schemes_resolve_for_px4_sitl(self) -> None:
        spec = get_spec("px4-sitl")
        udp = resolve_transport(_SITL, spec)
        self.assertIsInstance(udp, MavlinkDroneTransport)
        self.assertEqual(udp.protocol, "mavlink")
        self.assertEqual(udp.backend_id, "mavlink")
        self.assertTrue(udp.simulation_target)
        self.assertFalse(udp.requires_hardware_arming)
        serial = resolve_transport("mavlink+serial:///dev/ttyACM0?baud=921600", spec)
        self.assertIsInstance(serial, MavlinkDroneTransport)
        self.assertTrue(serial.requires_hardware_arming)
        self.assertFalse(serial.simulation_target)

    def test_px4_x500_still_refuses_mavlink(self) -> None:
        spec = get_spec("px4-x500")
        for uri in (_SITL, "mavlink+serial:///dev/ttyACM0", "sim://", None):
            with self.assertRaises(AdapterUnavailable):
                resolve_transport(uri, spec)

    def test_available_adapters_follow_mavsdk(self) -> None:
        present = importlib.util.find_spec("mavsdk") is not None
        names = available_adapters()
        self.assertEqual("mavlink+udp" in names, present)
        self.assertEqual("mavlink+serial" in names, present)

    def test_connect_without_mavsdk_does_not_open_a_socket(self) -> None:
        transport = MavlinkDroneTransport(get_spec("px4-sitl"), _SITL)
        with mock.patch.dict(sys.modules, {"mavsdk": None}):
            with self.assertRaises(AdapterUnavailable) as caught:
                transport.connect()
        self.assertIn("mavsdk", str(caught.exception).lower())
        self.assertEqual(transport.status().state.value, "error")


class TestMavlinkEstop(unittest.TestCase):
    def setUp(self) -> None:
        self.link = RecordingLink()
        self.transport = MavlinkDroneTransport(
            get_spec("px4-sitl"), _SITL, link=self.link
        )
        self.transport.connect()

    def _names(self) -> list[str]:
        return [command.name for command in self.link.commands]

    def _assert_clean(self, commands: list[MavlinkCommand] | None = None) -> None:
        commands = self.link.commands if commands is None else commands
        for command in commands:
            self.assertNotIn(
                command.name,
                {
                    "kill",
                    "terminate",
                    "arm_force",
                    "force_disarm",
                    "flight_termination",
                },
            )
            self.assertNotEqual(command.mav_command, MAV_CMD_DO_FLIGHTTERMINATION)
            self.assertNotEqual(command.mav_command, MAV_FORCE_DISARM_MAGIC)
            for value in (command.param1, command.param2, command.param3):
                self.assertGreater(abs(value - _MAGIC), 1e-3)
            if command.name == "disarm":
                self.assertEqual(command.param1, 0.0)
                self.assertEqual(command.param2, 0.0)
                self.assertEqual(command.mav_command, 400)

    def test_in_air_estop_lands_and_does_not_disarm(self) -> None:
        _air(self.link)
        self.transport.emergency_stop()
        self.assertEqual(self._names(), ["land"])
        self.assertEqual(self.link.commands[0].mav_command, MAV_CMD_NAV_LAND)
        self.assertEqual(self.link.commands[0].mav_command, LAND_COMMAND.mav_command)
        self._assert_clean()

    def test_takeoff_and_landing_estop_land(self) -> None:
        for state in ("takeoff", "landing"):
            self.link.commands.clear()
            _air(self.link, state)
            self.transport.emergency_stop()
            self.assertEqual(self._names(), ["land"], state)
            self.assertNotIn("disarm", self._names())
            self._assert_clean()

    def test_unknown_estop_holds_then_lands_and_does_not_disarm(self) -> None:
        self.link.sample["landed_state"] = "unknown"
        self.link.sample["z"] = 0.0
        self.transport.emergency_stop()
        self.assertEqual(self._names(), ["hold", "land"])
        self.assertEqual(self.link.commands[0].mav_command, HOLD_COMMAND.mav_command)
        self.assertEqual(self.link.commands[0].param2, HOLD_COMMAND.param2)
        self.assertNotIn("disarm", self._names())
        self._assert_clean()

    def test_on_ground_below_quarter_metre_disarms_normally(self) -> None:
        _ground(self.link, z=-0.2)
        self.transport.emergency_stop()
        self.assertEqual(self._names(), ["disarm"])
        self._assert_clean()

    def test_on_ground_above_quarter_metre_lands(self) -> None:
        _ground(self.link, z=-0.26)
        self.transport.emergency_stop()
        self.assertEqual(self._names(), ["land"])
        self.assertNotIn("disarm", self._names())
        self._assert_clean()

    def test_send_flight_disarm_is_refused_in_the_air(self) -> None:
        _air(self.link)
        self.transport.send_flight("disarm")
        self.transport.send_flight("disarm", force=True)
        self.assertEqual(self.link.commands, [])

    def test_send_flight_disarm_on_ground_ignores_force(self) -> None:
        _ground(self.link, z=0.0)
        self.transport.send_flight("disarm", force=True)
        self.assertEqual(self._names(), ["disarm"])
        self.assertEqual(self.link.commands[0].param2, 0.0)
        self._assert_clean()

    def test_recording_link_drops_magic_before_the_log(self) -> None:
        forbidden = MavlinkCommand(
            name="disarm",
            mav_command=400,
            param1=0.0,
            param2=_MAGIC,
        )
        with self.assertRaises(RuntimeError):
            self.link.send(forbidden)
        self.assertEqual(self.link.commands, [])
        termination = MavlinkCommand(
            name="flight_termination",
            mav_command=MAV_CMD_DO_FLIGHTTERMINATION,
            param1=1.0,
        )
        with self.assertRaises(RuntimeError):
            self.link.send(termination)
        self.assertEqual(self.link.commands, [])

    def test_gate_estop_in_air_does_not_disarm(self) -> None:
        _air(self.link)
        self.transport._emit_telemetry(self.transport.read())
        tmp = tempfile.mkdtemp(prefix="ohho-mavlink-")
        self.addCleanup(shutil.rmtree, tmp, True)
        gate = SafetyGate(
            self.transport,
            get_spec("px4-sitl"),
            audit=AuditLog(tmp),
            auto_tick=False,
            environ={},
        )
        gate.connect()
        gate.emergency_stop()
        self.assertEqual(self._names(), ["land"])
        self.assertNotIn("disarm", self._names())
        self._assert_clean()
        self.assertTrue(gate.profile.aerial)


class TestMavlinkFrames(unittest.TestCase):
    def setUp(self) -> None:
        self.link = RecordingLink()
        self.transport = MavlinkDroneTransport(
            get_spec("px4-sitl"), _SITL, link=self.link
        )
        self.transport.connect()

    def test_body_frd_velocity_is_mav_frame_body_ned(self) -> None:
        yaw_rate = math.radians(30.0)
        self.transport.send_flight(
            "offboard_velocity",
            vx=1.0,
            vy=-0.25,
            vz=-0.5,
            yaw_rate=yaw_rate,
            frame=Frame.BODY_FRD,
        )
        command = self.link.commands[-1]
        self.assertEqual(command.ohho_frame, "BODY_FRD")
        self.assertEqual(command.mav_frame, "MAV_FRAME_BODY_NED")
        self.assertEqual(command.mav_frame_id, MAV_FRAME_BODY_NED)
        self.assertEqual(command.mav_frame_id, 8)
        self.assertNotEqual(command.mav_frame_id, 12)
        self.assertAlmostEqual(command.x or 0.0, 1.0)
        self.assertAlmostEqual(command.y or 0.0, -0.25)
        self.assertAlmostEqual(command.z or 0.0, -0.5)
        self.assertAlmostEqual(command.yawspeed_deg_s or 0.0, 30.0)
        self.assertAlmostEqual(command.yaw_rate_rad_s or 0.0, yaw_rate)
        self.assertIsNone(command.yaw_deg)

    def test_local_enu_velocity_converts_to_ned_and_keeps_yaw_rate(self) -> None:
        self.transport.send_flight(
            "offboard_velocity",
            vx=1.0,
            vy=2.0,
            vz=3.0,
            yaw_rate=0.5,
            frame=Frame.LOCAL_ENU,
        )
        command = self.link.commands[-1]
        self.assertEqual(command.ohho_frame, "LOCAL_ENU")
        self.assertEqual(command.mav_frame, "MAV_FRAME_LOCAL_NED")
        self.assertEqual(command.mav_frame_id, MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(command.x or 0.0, 2.0)
        self.assertAlmostEqual(command.y or 0.0, 1.0)
        self.assertAlmostEqual(command.z or 0.0, -3.0)
        self.assertAlmostEqual(command.yaw_rate_rad_s or 0.0, -0.5)
        self.assertIsNone(command.yaw_deg)
        fields = position_target_fields(command)
        self.assertEqual(fields["message"], "SET_POSITION_TARGET_LOCAL_NED")
        self.assertEqual(fields["coordinate_frame"], MAV_FRAME_LOCAL_NED)
        self.assertEqual(int(fields["type_mask"]) & 1024, 1024)
        self.assertEqual(int(fields["type_mask"]) & 2048, 0)
        self.assertAlmostEqual(float(fields["yaw_rate"]), -0.5)
        self.assertEqual(command.type_mask, TYPEMASK_VELOCITY_YAW_RATE)

    def test_local_ned_position_and_enu_position(self) -> None:
        self.transport.send_flight(
            "offboard_position",
            x=1.0,
            y=2.0,
            z=-3.0,
            yaw=0.5,
            frame=Frame.LOCAL_NED,
        )
        ned = self.link.commands[-1]
        self.assertEqual(ned.mav_frame_id, MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(ned.x or 0.0, 1.0)
        self.assertAlmostEqual(ned.y or 0.0, 2.0)
        self.assertAlmostEqual(ned.z or 0.0, -3.0)
        self.assertAlmostEqual(ned.yaw_deg or 0.0, math.degrees(0.5))
        self.transport.send_flight(
            "offboard_position",
            x=1.0,
            y=2.0,
            z=3.0,
            yaw=0.0,
            frame=Frame.LOCAL_ENU,
        )
        enu = self.link.commands[-1]
        self.assertEqual(enu.ohho_frame, "LOCAL_ENU")
        self.assertEqual(enu.mav_frame_id, MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(enu.x or 0.0, 2.0)
        self.assertAlmostEqual(enu.y or 0.0, 1.0)
        self.assertAlmostEqual(enu.z or 0.0, -3.0)
        self.assertAlmostEqual(enu.yaw_rad or 0.0, math.pi / 2.0)

    def test_body_position_is_not_rotated(self) -> None:
        self.transport.send_flight(
            "offboard_position",
            x=1.0,
            y=0.0,
            z=0.2,
            yaw=0.0,
            frame=Frame.BODY_FRD,
        )
        command = self.link.commands[-1]
        self.assertEqual(command.mav_frame, "MAV_FRAME_BODY_NED")
        self.assertEqual(command.mav_frame_id, 8)
        self.assertAlmostEqual(command.x or 0.0, 1.0)
        self.assertAlmostEqual(command.y or 0.0, 0.0)
        self.assertAlmostEqual(command.z or 0.0, 0.2)

    def test_set_yaw_enu_converts_and_body_stays_body(self) -> None:
        self.transport.send_flight("set_yaw", yaw=0.0, frame=Frame.LOCAL_ENU)
        enu = self.link.commands[-1]
        self.assertEqual(enu.mav_frame_id, MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(enu.yaw_rad or 0.0, math.pi / 2.0)
        self.transport.send_flight("set_yaw", yaw=0.3, frame=Frame.BODY_FRD)
        body = self.link.commands[-1]
        self.assertEqual(body.mav_frame_id, MAV_FRAME_BODY_NED)
        self.assertAlmostEqual(body.yaw_rad or 0.0, 0.3)

    def test_string_frame_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.transport.send_flight(
                "offboard_velocity",
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0,
                frame="BODY_FRD",
            )
        with self.assertRaises(TypeError):
            map_offboard_velocity(0.0, 0.0, 0.0, 0.0, "LOCAL_NED")  # type: ignore[arg-type]

    def test_planar_velocity_is_body_frame_and_zero_is_not_land(self) -> None:
        self.transport.send_velocity(Velocity(0.2, -0.1, 0.3))
        command = self.link.commands[-1]
        self.assertEqual(command.name, "offboard_velocity")
        self.assertEqual(command.mav_frame_id, MAV_FRAME_BODY_NED)
        self.assertAlmostEqual(command.x or 0.0, 0.2)
        self.assertAlmostEqual(command.z or 0.0, 0.0)
        self.link.commands.clear()
        self.transport.send_velocity(Velocity())
        zero = self.link.commands[-1]
        self.assertEqual(zero.name, "offboard_velocity")
        self.assertNotEqual(zero.name, "land")
        self.assertAlmostEqual(zero.x or 0.0, 0.0)

    def test_robot_methods_reach_the_link(self) -> None:
        bot = Robot(get_spec("px4-sitl"), self.transport, NativeRuntime())
        bot.offboard_velocity(0.4, 0.0, -0.1, 0.0, Frame.BODY_FRD)
        bot.hold()
        self.assertEqual(self.link.commands[-1].name, "hold")
        self.assertEqual(self.link.commands[-2].mav_frame_id, MAV_FRAME_BODY_NED)
        bot.disconnect()


class TestMavsdkDispatch(unittest.TestCase):
    def test_plugin_map_has_no_force_disarm_methods(self) -> None:
        methods = set(_ACTION_METHODS.values())
        self.assertEqual(
            methods,
            {"arm", "disarm", "takeoff", "land", "hold", "return_to_launch"},
        )
        source = Path(
            importlib.import_module("ohho.adapters.mavlink_drone").__file__ or ""
        ).read_text(encoding="utf-8")
        self.assertNotIn(".kill(", source)
        self.assertNotIn(".terminate(", source)
        self.assertNotIn(".arm_force(", source)

    def test_body_velocity_uses_set_velocity_body_and_not_kill(self) -> None:
        plugins = _Plugins()
        raw: list[object] = []
        link = MavsdkLink(action=plugins, offboard=plugins, raw_sender=raw.append)
        link.send(
            map_offboard_velocity(1.0, 0.0, -0.5, math.radians(10), Frame.BODY_FRD)
        )
        self.assertEqual(
            plugins.calls,
            [
                ("set_velocity_body", (1.0, 0.0, -0.5, math.degrees(math.radians(10)))),
                ("start", None),
            ],
        )
        self.assertEqual(raw, [])

    def test_local_velocity_is_a_raw_yaw_rate_message(self) -> None:
        plugins = _Plugins()
        raw: list[dict[str, object]] = []
        link = MavsdkLink(action=plugins, offboard=plugins, raw_sender=raw.append)
        link.send(map_offboard_velocity(0.5, 0.0, 0.0, 0.2, Frame.LOCAL_NED))
        self.assertEqual(plugins.calls, [])
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["coordinate_frame"], MAV_FRAME_LOCAL_NED)
        self.assertAlmostEqual(float(raw[0]["vx"]), 0.5)
        self.assertAlmostEqual(float(raw[0]["yaw_rate"]), 0.2)

    def test_ned_position_uses_set_position_ned(self) -> None:
        plugins = _Plugins()
        link = MavsdkLink(
            action=plugins, offboard=plugins, raw_sender=lambda _fields: None
        )
        link.send(map_offboard_velocity(0.0, 0.0, 0.0, 0.0, Frame.LOCAL_NED))
        # velocity above is raw; position is the MAVSDK setter
        plugins.calls.clear()
        from ohho.adapters.mavlink_drone import map_offboard_position

        link.send(map_offboard_position(4.0, 0.0, -2.5, 0.0, Frame.LOCAL_NED))
        self.assertEqual(plugins.calls[0][0], "set_position_ned")
        self.assertEqual(plugins.calls[0][1], (4.0, 0.0, -2.5, 0.0))
        self.assertEqual(plugins.calls[1][0], "start")

    def test_disarm_calls_disarm_not_kill(self) -> None:
        plugins = _Plugins()
        link = MavsdkLink(action=plugins, offboard=plugins)
        link.send(
            MavlinkCommand(name="disarm", mav_command=400, param1=0.0, param2=0.0)
        )
        self.assertEqual(plugins.calls, [("disarm", None)])

    def test_magic_disarm_never_reaches_the_plugin(self) -> None:
        plugins = _Plugins()
        link = MavsdkLink(action=plugins, offboard=plugins)
        with self.assertRaises(RuntimeError):
            link.send(
                MavlinkCommand(
                    name="disarm",
                    mav_command=400,
                    param1=0.0,
                    param2=_MAGIC,
                )
            )
        self.assertEqual(plugins.calls, [])

    def test_takeoff_sets_altitude_then_takes_off(self) -> None:
        plugins = _Plugins()
        link = MavsdkLink(action=plugins, offboard=plugins)
        link.send(MavlinkCommand(name="takeoff", alt_m=2.5))
        self.assertEqual(
            plugins.calls,
            [("set_takeoff_altitude", 2.5), ("takeoff", None)],
        )


class TestMavlinkTelemetry(unittest.TestCase):
    def test_custom_drone_matches_the_schema(self) -> None:
        link = RecordingLink(
            {
                "armed": True,
                "flight_mode": "hold",
                "landed_state": "in_air",
                "x": 1.0,
                "y": 2.0,
                "z": -2.5,
                "yaw": 0.4,
                "vx": 0.1,
                "battery": 0.8,
                "voltage_v": 15.0,
                "current_a": 1.2,
                "gps_fix": 6,
                "failsafe": [],
                "timestamp": 12.0,
            }
        )
        transport = MavlinkDroneTransport(get_spec("px4-sitl"), _SITL, link=link)
        transport.connect()
        snap = transport.read()
        drone = snap.custom["drone"]
        self.assertIs(drone["experimental"], True)
        self.assertIs(drone["flight_ready"], False)
        self.assertEqual(drone["pose_frame"], "LOCAL_NED")
        self.assertEqual(drone["velocity_frame"], "LOCAL_NED")
        self.assertEqual(drone["landed_state"], "in_air")
        self.assertEqual(drone["flight_mode"], "hold")
        self.assertEqual(drone["gps_fix"], "rtk_fixed")
        self.assertIs(drone["armed"], True)
        self.assertIs(drone["link_ok"], True)
        self.assertAlmostEqual(drone["pose"]["z"], -2.5)
        self.assertAlmostEqual(snap.battery or 0.0, 0.8)
        self.assertAlmostEqual(drone["voltage_v"], 15.0)
        self.assertIsNotNone(snap.odom)
        assert snap.odom is not None
        self.assertAlmostEqual(snap.odom.x, 1.0)
        self.assertAlmostEqual(snap.odom.theta, 0.4)
        self.assertEqual(snap.timestamp, 12.0)


class _MavsdkFlightMode(enum.IntEnum):
    """Names and values from mavsdk 4.0.6 ``telemetry.FlightMode``."""

    HOLD = 3
    RETURN_TO_LAUNCH = 5


class _MavsdkLandedState(enum.IntEnum):
    """Names and values from mavsdk 4.0.6 ``telemetry.LandedState``."""

    ON_GROUND = 1
    IN_AIR = 2


class _MavsdkFixType(enum.IntEnum):
    """Names and values from mavsdk 4.0.6 ``telemetry.FixType``."""

    FIX_3D = 3


class _Mavsdk4Telemetry:
    """Records subscribe callbacks and invokes them as mavsdk 4.0.6 does.

    ``Telemetry.subscribe_*`` calls ``callback(value, user_data)``.
    ``user_data`` is the object passed to subscribe, ``None`` when omitted.
    """

    def __init__(self) -> None:
        self.subscribed: dict[str, tuple[object, object]] = {}

    def _remember(self, name: str, callback: object, user_data: object = None) -> None:
        self.subscribed[name] = (callback, user_data)

    def subscribe_armed(self, callback: object, user_data: object = None) -> None:
        self._remember("armed", callback, user_data)

    def subscribe_flight_mode(self, callback: object, user_data: object = None) -> None:
        self._remember("flight_mode", callback, user_data)

    def subscribe_landed_state(
        self, callback: object, user_data: object = None
    ) -> None:
        self._remember("landed_state", callback, user_data)

    def subscribe_position_velocity_ned(
        self, callback: object, user_data: object = None
    ) -> None:
        self._remember("position_velocity_ned", callback, user_data)

    def subscribe_attitude_euler(
        self, callback: object, user_data: object = None
    ) -> None:
        self._remember("attitude_euler", callback, user_data)

    def subscribe_attitude_angular_velocity_body(
        self, callback: object, user_data: object = None
    ) -> None:
        self._remember("attitude_angular_velocity_body", callback, user_data)

    def subscribe_battery(self, callback: object, user_data: object = None) -> None:
        self._remember("battery", callback, user_data)

    def subscribe_gps_info(self, callback: object, user_data: object = None) -> None:
        self._remember("gps_info", callback, user_data)

    def deliver(self, name: str, value: object) -> None:
        callback, user_data = self.subscribed[name]
        if not callable(callback):
            raise AssertionError(name)
        callback(value, user_data)


_SUBSCRIPTION_HANDLERS = (
    "_on_armed",
    "_on_mode",
    "_on_landed",
    "_on_position_velocity",
    "_on_attitude",
    "_on_yaw_rate",
    "_on_battery",
    "_on_gps",
)


class TestMavsdk4SubscriptionCallbacks(unittest.TestCase):
    def test_handlers_take_value_and_optional_user_data(self) -> None:
        for name in _SUBSCRIPTION_HANDLERS:
            signature = inspect.signature(getattr(MavsdkLink, name))
            parameters = list(signature.parameters.values())
            self.assertGreaterEqual(len(parameters), 3, name)
            self.assertEqual(parameters[0].name, "self", name)
            user_data = signature.parameters["user_data"]
            self.assertIsNone(user_data.default, name)
            self.assertEqual(
                user_data.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD, name
            )
            self.assertNotIn("args", signature.parameters, name)
            self.assertNotIn("kwargs", signature.parameters, name)

    def test_mavsdk4_callback_shape_fills_every_subscription(self) -> None:
        telemetry = _Mavsdk4Telemetry()
        link = MavsdkLink()
        link._bind_telemetry(telemetry)
        self.assertEqual(
            set(telemetry.subscribed),
            {
                "armed",
                "flight_mode",
                "landed_state",
                "position_velocity_ned",
                "attitude_euler",
                "attitude_angular_velocity_body",
                "battery",
                "gps_info",
            },
        )
        for _callback, user_data in telemetry.subscribed.values():
            self.assertIsNone(user_data)

        telemetry.deliver("armed", True)
        telemetry.deliver("flight_mode", _MavsdkFlightMode.HOLD)
        telemetry.deliver("landed_state", _MavsdkLandedState.ON_GROUND)
        telemetry.deliver(
            "position_velocity_ned",
            SimpleNamespace(
                position=SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-2.5),
                velocity=SimpleNamespace(north_m_s=0.1, east_m_s=-0.2, down_m_s=0.3),
            ),
        )
        telemetry.deliver(
            "attitude_euler",
            SimpleNamespace(roll_deg=0.0, pitch_deg=90.0, yaw_deg=180.0),
        )
        telemetry.deliver(
            "attitude_angular_velocity_body",
            SimpleNamespace(yaw_rad_s=0.25),
        )
        telemetry.deliver(
            "battery",
            SimpleNamespace(
                remaining_percent=0.8, voltage_v=16.2, current_battery_a=4.5
            ),
        )
        telemetry.deliver(
            "gps_info",
            SimpleNamespace(fix_type=_MavsdkFixType.FIX_3D),
        )

        sample = link.read_sample()
        self.assertIs(sample["armed"], True)
        self.assertEqual(sample["flight_mode"], "hold")
        self.assertEqual(sample["landed_state"], "on_ground")
        self.assertEqual(sample["gps_fix"], "3d")
        self.assertIs(sample["link_ok"], True)
        for key, expected in (
            ("x", 1.0),
            ("y", 2.0),
            ("z", -2.5),
            ("vx", 0.1),
            ("vy", -0.2),
            ("vz", 0.3),
            ("pitch", math.pi / 2),
            ("yaw", math.pi),
            ("yaw_rate", 0.25),
            ("battery", 0.8),
            ("voltage_v", 16.2),
            ("current_a", 4.5),
        ):
            reading = sample[key]
            self.assertIsInstance(reading, float, key)
            self.assertAlmostEqual(reading, expected, msg=key)

    def test_user_data_is_not_the_sample(self) -> None:
        link = MavsdkLink()
        link._on_armed(False, True)
        link._on_landed(_MavsdkLandedState.ON_GROUND, _MavsdkLandedState.IN_AIR)
        link._on_mode(_MavsdkFlightMode.HOLD, _MavsdkFlightMode.RETURN_TO_LAUNCH)
        sample = link.read_sample()
        self.assertIs(sample["armed"], False)
        self.assertEqual(sample["landed_state"], "on_ground")
        self.assertEqual(sample["flight_mode"], "hold")

    def test_one_argument_call_still_delivers_the_sample(self) -> None:
        telemetry = _Mavsdk4Telemetry()
        link = MavsdkLink()
        link._bind_telemetry(telemetry)
        callback, _user_data = telemetry.subscribed["landed_state"]
        self.assertTrue(callable(callback))
        callback(_MavsdkLandedState.ON_GROUND)
        self.assertEqual(link.read_sample()["landed_state"], "on_ground")

    def test_extra_positional_argument_raises(self) -> None:
        link = MavsdkLink()
        with self.assertRaises(TypeError):
            link._on_landed(_MavsdkLandedState.ON_GROUND, None, "extra")  # type: ignore[call-arg]
        self.assertEqual(link.read_sample()["landed_state"], "unknown")


if __name__ == "__main__":
    unittest.main()
