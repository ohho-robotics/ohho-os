"""Crazyflie backend (OHH-303). Mocks only: no radio, no ROS graph, no sim process.

The flight recipe is the one Crazyswarm2 ``backend:=sim`` executes
(takeoff, hover-by-takeoff, land) plus the emergency service Ohho must
call. ``cmd_vel_legacy`` is not part of that recipe.
"""

from __future__ import annotations

import builtins
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohho import capabilities as caps
from ohho.adapters import AdapterUnavailable, resolve_transport
from ohho.adapters.crazyflie import (
    EXAMPLE_CF_NAME,
    EXAMPLE_RADIO_URI,
    GO_TO_YAW_UNIT,
    HL_DURATION_FLOOR_S,
    LAND_HEIGHT_M,
    SIM_ABSENT,
    SIM_EXECUTED,
    SIM_GAPS,
    SIM_LOG_ONLY,
    CflibLink,
    CrazyflieTransport,
    CrazyflieUnsupported,
    Crazyswarm2Link,
    battery_fraction_from_voltage,
    to_crazyswarm_position,
    to_crazyswarm_velocity,
    yaw_from_quaternion,
    yaw_quaternion,
)
from ohho.adapters.sim import SimTransport
from ohho.aerial import Frame
from ohho.registry import get_spec, list_specs
from ohho.safety import ARM_ENV, ARM_PHRASE, AuditLog, SafetyGate
from ohho.schema import ConnectionState, Velocity

_WALL = 1_760_000_000.0


class MemoryLink:
    """Records commands. Does not open a socket or a radio."""

    def __init__(self, sample: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.sample_data = dict(sample or {})
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def sample(self) -> dict:
        return dict(self.sample_data)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(**fields: object) -> None:
            self.calls.append((name, fields))

        return _call


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _transport(
    kind: str = "crazyswarm2",
    address: str = EXAMPLE_CF_NAME,
    sample: dict | None = None,
) -> tuple[CrazyflieTransport, MemoryLink]:
    link = MemoryLink(sample)
    transport = CrazyflieTransport(
        get_spec("crazyflie-cf21"),
        address if kind == "crazyswarm2" else EXAMPLE_RADIO_URI,
        kind=kind,
        link=link,
        heartbeat=False,
    )
    transport.connect()
    return transport, link


def _names(transport: CrazyflieTransport) -> list[str]:
    return [name for name, _fields in transport.sent]


class TestCrazyflieRegistry(unittest.TestCase):
    def test_cf21_spec_matches_d03_caps_and_has_no_gps(self):
        spec = get_spec("crazyflie-cf21")
        self.assertEqual(spec.category, "aerial")
        self.assertEqual(spec.adapter, "crazyflie")
        self.assertIn(caps.AERIAL, spec.capabilities)
        self.assertIn(caps.BASE_DRIVE, spec.capabilities)
        self.assertNotIn(caps.MANIPULATION, spec.capabilities)
        self.assertEqual(spec.max_lin, 2.0)
        self.assertEqual(spec.max_climb, 1.0)
        self.assertEqual(spec.max_ang, 0.8)
        self.assertEqual(spec.gps_fix, "none")
        self.assertTrue(spec.requires_hardware_arming)
        self.assertTrue(spec.experimental)
        self.assertFalse(spec.flight_ready)
        self.assertIn(spec.id, {item.id for item in list_specs()})

    def test_sim_uri_stays_the_ground_sim(self):
        spec = get_spec("crazyflie-cf21")
        for uri in (None, "sim://"):
            transport = resolve_transport(uri, spec)
            self.assertIs(type(transport), SimTransport)


class TestCrazyflieUri(unittest.TestCase):
    def test_radio_and_ros_uris_resolve_without_connecting(self):
        before = set(sys.modules)
        radio = resolve_transport(
            f"crazyflie://{EXAMPLE_RADIO_URI}", get_spec("crazyflie-cf21")
        )
        ros = resolve_transport(
            f"crazyflie+ros2://{EXAMPLE_CF_NAME}", get_spec("crazyflie-cf21")
        )
        self.assertIsInstance(radio, CrazyflieTransport)
        self.assertIsInstance(ros, CrazyflieTransport)
        self.assertEqual(radio.kind, "cflib")
        self.assertEqual(ros.kind, "crazyswarm2")
        self.assertEqual(radio.address, EXAMPLE_RADIO_URI)
        self.assertEqual(ros.address, EXAMPLE_CF_NAME)
        self.assertEqual(radio.protocol, "crazyflie")
        self.assertEqual(radio.backend_id, "crazyflie")
        self.assertEqual(ros.backend_id, "crazyflie")
        self.assertIsNone(radio._link)
        self.assertIsNone(ros._link)
        self.assertEqual(radio.sent, [])
        added = set(sys.modules) - before
        self.assertFalse(
            any(name == "cflib" or name.startswith("cflib.") for name in added)
        )
        self.assertFalse(
            any(name == "rclpy" or name.startswith("rclpy.") for name in added)
        )

    def test_bad_uris_raise_before_a_link_exists(self):
        spec = get_spec("crazyflie-cf21")
        with self.assertRaises(AdapterUnavailable):
            resolve_transport("crazyflie://usb:0", spec)
        with self.assertRaises(AdapterUnavailable):
            resolve_transport("crazyflie+ros2://", spec)
        with self.assertRaises(AdapterUnavailable):
            resolve_transport("crazyflie+ros2://cf/21", spec)
        with self.assertRaises(AdapterUnavailable):
            resolve_transport("crazyflie://", spec)

    def test_dependencies_stay_empty_and_cflib_is_an_extra(self):
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("dependencies = []", text)
        self.assertIn('crazyflie = ["cflib>=0.1.18"]', text)


class TestSimContract(unittest.TestCase):
    def test_support_matrix_is_disjoint(self):
        self.assertFalse(SIM_EXECUTED & SIM_LOG_ONLY)
        self.assertFalse(SIM_EXECUTED & SIM_ABSENT)
        self.assertFalse(SIM_LOG_ONLY & SIM_ABSENT)
        self.assertIn("takeoff", SIM_EXECUTED)
        self.assertIn("land", SIM_EXECUTED)
        self.assertIn("go_to", SIM_EXECUTED)
        self.assertIn("cmd_full_state", SIM_EXECUTED)
        self.assertIn("emergency", SIM_LOG_ONLY)
        self.assertNotIn("emergency", SIM_EXECUTED)
        self.assertIn("cmd_vel_legacy", SIM_LOG_ONLY)
        self.assertIn("arm", SIM_ABSENT)
        self.assertIn("rtl", SIM_ABSENT)
        self.assertIn("pose", SIM_ABSENT)
        self.assertIn("status", SIM_ABSENT)
        for key in (
            "emergency",
            "cmd_vel_legacy",
            "arm",
            "rtl",
            "pose",
            "status",
            "go_to_from_idle",
        ):
            self.assertIn(key, SIM_GAPS)

    def test_takeoff_hover_land_and_emergency(self):
        """The local sim recipe, against a link that only records calls."""
        transport, _link = _transport()
        transport.send_flight("takeoff", alt_m=0.5)
        transport.send_flight("hold")
        transport.send_flight("land")
        self.assertEqual(_names(transport), ["takeoff", "land"])
        takeoff = transport.sent[0][1]
        land = transport.sent[1][1]
        self.assertEqual(takeoff["height_m"], 0.5)
        self.assertEqual(takeoff["duration_s"], HL_DURATION_FLOOR_S)
        self.assertEqual(land["height_m"], LAND_HEIGHT_M)
        self.assertEqual(land["duration_s"], HL_DURATION_FLOOR_S)
        self.assertNotIn("cmd_vel_legacy", _names(transport))
        self.assertNotIn("notify_setpoints_stop", _names(transport))

        transport.motor_cut()
        self.assertEqual(_names(transport), ["takeoff", "land", "emergency"])
        transport.emergency_stop()
        self.assertEqual(_names(transport).count("emergency"), 1)

        transport.send_flight("takeoff", alt_m=0.5)
        self.assertEqual(_names(transport).count("takeoff"), 1)
        self.assertIn("commands_after_emergency", transport.gaps)

    def test_telemetry_does_not_invent_airborne_or_a_gps_fix(self):
        transport, _link = _transport()
        transport.send_flight("takeoff", alt_m=0.5)
        drone = transport.read().custom["drone"]
        self.assertEqual(drone["gps_fix"], "none")
        self.assertEqual(drone["landed_state"], "unknown")
        self.assertTrue(drone["experimental"])
        self.assertFalse(drone["flight_ready"])
        self.assertTrue(drone["link_ok"])
        self.assertIsNone(drone["battery"])
        self.assertEqual(drone["pose_frame"], "LOCAL_ENU")

    def test_rtl_arm_and_drive_are_not_sim_commands(self):
        transport, _link = _transport()
        transport.send_flight("rtl")
        transport.send_flight("arm")
        transport.send_velocity(Velocity(0.2, 0.0, 0.1))
        self.assertEqual(transport.sent, [])
        self.assertIn("rtl", transport.gaps)
        self.assertIn("arm", transport.gaps)
        self.assertIn("planar_velocity", transport.gaps)

    def test_cflib_arm_is_sent_and_ros_arm_is_not(self):
        radio, _link = _transport(kind="cflib")
        radio.send_flight("arm")
        radio.send_flight("disarm")
        self.assertEqual(_names(radio), ["arm", "disarm"])
        ros, _ros_link = _transport()
        ros.send_flight("arm")
        self.assertEqual(ros.sent, [])
        self.assertIn("arm", ros.gaps)

    def test_go_to_converts_ned_and_refuses_body(self):
        self.assertEqual(GO_TO_YAW_UNIT, "rad")
        transport, _link = _transport()
        transport.send_flight(
            "offboard_position",
            x=1.0,
            y=2.0,
            z=-0.5,
            yaw=0.0,
            frame=Frame.LOCAL_NED,
        )
        fields = transport.sent[0][1]
        self.assertEqual(transport.sent[0][0], "go_to")
        self.assertAlmostEqual(fields["x"], 2.0)
        self.assertAlmostEqual(fields["y"], 1.0)
        self.assertAlmostEqual(fields["z"], 0.5)
        self.assertAlmostEqual(fields["yaw"], math.pi / 2.0)
        self.assertFalse(fields["relative"])
        self.assertGreaterEqual(fields["duration_s"], HL_DURATION_FLOOR_S)

        with self.assertRaises(CrazyflieUnsupported):
            transport.send_flight(
                "offboard_position",
                x=0.1,
                y=0.0,
                z=0.0,
                yaw=0.0,
                frame=Frame.BODY_FRD,
            )

    def test_full_state_needs_a_position_and_is_not_cmd_vel_legacy(self):
        transport, link = _transport()
        transport.send_flight(
            "offboard_velocity",
            vx=0.2,
            vy=0.0,
            vz=0.0,
            yaw_rate=0.0,
            frame=Frame.LOCAL_ENU,
        )
        self.assertEqual(transport.sent, [])
        self.assertIn("cmd_full_state_needs_pose", transport.gaps)

        link.sample_data["pose_enu"] = (1.0, 2.0, 0.5, 0.0)
        transport.send_flight(
            "offboard_velocity",
            vx=0.2,
            vy=0.0,
            vz=-0.1,
            yaw_rate=0.3,
            frame=Frame.LOCAL_ENU,
        )
        name, fields = transport.sent[-1]
        self.assertEqual(name, "cmd_full_state")
        self.assertEqual(fields["pos"], (1.0, 2.0, 0.5))
        self.assertAlmostEqual(fields["vel"][0], 0.2)
        self.assertAlmostEqual(fields["vel"][2], -0.1)
        # ENU nose-left rate becomes a nose-right body rate.
        self.assertAlmostEqual(fields["yaw_rate"], -0.3)
        self.assertNotIn("cmd_vel_legacy", _names(transport))

    def test_radio_velocity_uses_the_world_setpoint(self):
        transport, _link = _transport(kind="cflib")
        transport.send_flight(
            "offboard_velocity",
            vx=0.2,
            vy=0.0,
            vz=-0.1,
            yaw_rate=0.5,
            frame=Frame.LOCAL_ENU,
        )
        name, fields = transport.sent[0]
        self.assertEqual(name, "velocity_world")
        self.assertAlmostEqual(fields["vx"], 0.2)
        self.assertAlmostEqual(fields["yaw_rate"], -0.5)
        self.assertNotIn("cmd_full_state", _names(transport))
        self.assertNotIn("cmd_vel_legacy", _names(transport))

    def test_body_velocity_at_zero_yaw_maps_forward_to_east(self):
        vx, vy, vz, rate = to_crazyswarm_velocity(
            1.0, 0.0, 0.0, 0.2, Frame.BODY_FRD, yaw_enu=0.0
        )
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(vy, 0.0)
        self.assertAlmostEqual(vz, 0.0)
        self.assertAlmostEqual(rate, 0.2)
        right = to_crazyswarm_velocity(0.0, 1.0, 0.0, 0.0, Frame.BODY_FRD, yaw_enu=0.0)
        self.assertAlmostEqual(right[0], 0.0)
        self.assertAlmostEqual(right[1], -1.0)
        down = to_crazyswarm_velocity(0.0, 0.0, 1.0, 0.0, Frame.BODY_FRD, yaw_enu=0.0)
        self.assertAlmostEqual(down[2], -1.0)

    def test_set_yaw_does_not_jump_to_the_origin(self):
        transport, link = _transport()
        transport.send_flight("set_yaw", yaw=0.4, frame=Frame.LOCAL_ENU)
        self.assertEqual(transport.sent, [])
        self.assertIn("set_yaw_needs_pose", transport.gaps)
        link.sample_data["pose_enu"] = (0.3, 0.4, 0.5, 0.0)
        transport.send_flight("set_yaw", yaw=math.pi / 2.0, frame=Frame.LOCAL_NED)
        fields = transport.sent[-1][1]
        self.assertEqual(transport.sent[-1][0], "go_to")
        self.assertAlmostEqual(fields["x"], 0.3)
        self.assertAlmostEqual(fields["y"], 0.4)
        self.assertAlmostEqual(fields["z"], 0.5)
        self.assertAlmostEqual(fields["yaw"], 0.0)

    def test_battery_fraction_is_one_s_only(self):
        self.assertAlmostEqual(battery_fraction_from_voltage(4.2), 1.0)
        self.assertAlmostEqual(battery_fraction_from_voltage(3.0), 0.0)
        self.assertAlmostEqual(battery_fraction_from_voltage(3.6), 0.5)
        self.assertIsNone(battery_fraction_from_voltage(0.0))
        self.assertIsNone(battery_fraction_from_voltage(16.8))
        east, north, up, yaw = to_crazyswarm_position(
            1.0, 2.0, -0.5, 0.0, Frame.LOCAL_NED
        )
        self.assertAlmostEqual(east, 2.0)
        self.assertAlmostEqual(north, 1.0)
        self.assertAlmostEqual(up, 0.5)
        self.assertAlmostEqual(yaw, math.pi / 2.0)


class TestCrazyswarm2Mapping(unittest.TestCase):
    def _link(self) -> tuple[Crazyswarm2Link, "_Node"]:
        node = _Node()
        link = Crazyswarm2Link(EXAMPLE_CF_NAME, node, types=_Types())
        link.open()
        return link, node

    def test_service_names_and_takeoff_land_emergency(self):
        link, node = self._link()
        names = {client.name for client in node.clients}
        self.assertIn(f"/{EXAMPLE_CF_NAME}/takeoff", names)
        self.assertIn(f"/{EXAMPLE_CF_NAME}/land", names)
        self.assertIn(f"/{EXAMPLE_CF_NAME}/go_to", names)
        self.assertIn(f"/{EXAMPLE_CF_NAME}/emergency", names)
        topics = {pub.name for pub in node.publishers}
        self.assertIn(f"/{EXAMPLE_CF_NAME}/cmd_full_state", topics)
        self.assertIn(f"/{EXAMPLE_CF_NAME}/cmd_vel_legacy", topics)

        link.takeoff(height_m=0.5, duration_s=2.0)
        request = link._clients["takeoff"].requests[0]
        self.assertAlmostEqual(request.height, 0.5)
        self.assertEqual(request.duration.sec, 2)
        self.assertEqual(request.duration.nanosec, 0)
        self.assertEqual(request.group_mask, 0)

        link.land(height_m=0.0, duration_s=2.0)
        land = link._clients["land"].requests[0]
        self.assertAlmostEqual(land.height, 0.0)

        link.emergency()
        self.assertEqual(len(link._clients["emergency"].requests), 1)
        self.assertEqual(link.published, [])

    def test_go_to_yaw_stays_in_radians_and_full_state_is_published(self):
        link, _node = self._link()
        link.go_to(
            x=0.5, y=-0.25, z=0.5, yaw=math.pi / 4.0, duration_s=2.0, relative=False
        )
        request = link._clients["go_to"].requests[0]
        self.assertAlmostEqual(request.yaw, math.pi / 4.0)
        self.assertNotAlmostEqual(request.yaw, 45.0)
        self.assertFalse(request.relative)
        self.assertAlmostEqual(request.goal.x, 0.5)

        link.cmd_full_state(
            pos=(0.5, 0.0, 0.5),
            vel=(0.1, 0.0, 0.0),
            acc=(0.0, 0.0, 0.0),
            yaw=0.0,
            yaw_rate=0.2,
        )
        self.assertEqual(link.published, ["cmd_full_state"])
        msg = link._pubs["cmd_full_state"].messages[0]
        self.assertEqual(msg.header.frame_id, "/world")
        self.assertAlmostEqual(msg.pose.position.z, 0.5)
        self.assertAlmostEqual(msg.twist.linear.x, 0.1)
        self.assertAlmostEqual(msg.twist.angular.z, 0.2)
        self.assertAlmostEqual(msg.pose.orientation.w, 1.0)

    def test_cmd_vel_legacy_packing_is_not_a_velocity(self):
        link, _node = self._link()
        link.cmd_vel_legacy(
            roll_deg=1.0, pitch_deg=2.0, yaw_rate_deg_s=3.0, thrust=10000
        )
        msg = link._pubs["cmd_vel_legacy"].messages[0]
        self.assertAlmostEqual(msg.linear.x, 2.0)
        self.assertAlmostEqual(msg.linear.y, 1.0)
        self.assertAlmostEqual(msg.linear.z, 10000.0)
        self.assertAlmostEqual(msg.angular.z, 3.0)
        self.assertEqual(link.published, ["cmd_vel_legacy"])

    def test_pose_and_status_callbacks(self):
        link, node = self._link()
        pose_cb = next(
            cb for _typ, name, cb in node.subscriptions if name.endswith("/pose")
        )
        status_cb = next(
            cb for _typ, name, cb in node.subscriptions if name.endswith("/status")
        )
        quat = yaw_quaternion(math.pi / 2.0)
        pose_cb(_pose_msg(0.2, 0.3, 0.4, quat))
        sample = link.sample()
        self.assertAlmostEqual(sample["pose_enu"][0], 0.2)
        self.assertAlmostEqual(sample["pose_enu"][3], math.pi / 2.0, places=6)
        # IS_ARMED | IS_FLYING, 3.6 V → half a 1S pack.
        status_cb(_status_msg(3.6, 2 | 16, 0))
        sample = link.sample()
        self.assertAlmostEqual(sample["voltage_v"], 3.6)
        self.assertTrue(sample["armed"])
        self.assertTrue(sample["flying"])
        self.assertAlmostEqual(yaw_from_quaternion(*quat), math.pi / 2.0, places=6)


class TestCflibMapping(unittest.TestCase):
    def test_injected_crazyflie_does_not_open_the_radio(self):
        fake = _FakeCf()
        link = CflibLink(EXAMPLE_RADIO_URI, crazyflie=fake)
        link.open()
        self.assertFalse(link.radio_opened)
        link.takeoff(height_m=0.5, duration_s=2.0)
        link.land(height_m=0.0, duration_s=2.0)
        self.assertEqual(fake.high_level.calls[0], ("takeoff", 0.5, 2.0, None))
        self.assertEqual(fake.high_level.calls[1], ("land", 0.0, 2.0, None))
        link.emergency()
        self.assertEqual(fake.supervisor.calls, ["emergency"])
        self.assertEqual(fake.loc.calls, [])
        link.arm(armed=True)
        self.assertEqual(fake.supervisor.calls[-1], ("arm", True))
        link.velocity_world(vx=0.2, vy=0.0, vz=-0.1, yaw_rate=0.5)
        kind, vx, _vy, vz, yaw_deg = fake.commander.calls[-1]
        self.assertEqual(kind, "vel")
        self.assertAlmostEqual(vx, 0.2)
        self.assertAlmostEqual(vz, -0.1)
        self.assertAlmostEqual(yaw_deg, math.degrees(0.5))

    def test_legacy_emergency_when_supervisor_has_no_method(self):
        fake = _FakeCf()
        fake.supervisor = object()
        link = CflibLink(EXAMPLE_RADIO_URI, crazyflie=fake)
        link.emergency()
        self.assertEqual(fake.loc.calls, ["emergency"])

    def test_open_without_cflib_raises_before_the_radio(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "cflib" or name.startswith("cflib."):
                raise ImportError("blocked for the unit test")
            return real_import(name, *args, **kwargs)

        link = CflibLink(EXAMPLE_RADIO_URI)
        with mock.patch("builtins.__import__", guarded):
            with self.assertRaises(AdapterUnavailable):
                link.open()
        self.assertFalse(link.radio_opened)
        self.assertIsNone(link._cf)


class TestCrazyflieGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-crazyflie-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _gate(
        self, sample: dict | None = None, *, arm: bool
    ) -> tuple[SafetyGate, CrazyflieTransport]:
        link = MemoryLink(sample)
        transport = CrazyflieTransport(
            get_spec("crazyflie-cf21"),
            EXAMPLE_CF_NAME,
            kind="crazyswarm2",
            link=link,
            heartbeat=False,
        )
        # The phrase is passed only into this gate. The process environment
        # is not given OHHO_ARM_HARDWARE, and the link is in memory.
        gate = SafetyGate(
            transport,
            get_spec("crazyflie-cf21"),
            audit=AuditLog(self.tmp, wall_clock=lambda: _WALL),
            auto_tick=False,
            clock=FakeClock(),
            environ={ARM_ENV: "1"},
            isatty=lambda: True,
            input_fn=lambda _prompt: ARM_PHRASE,
        )
        gate.connect()
        if arm:
            self.assertTrue(gate.requires_hardware_arming)
            self.assertTrue(gate.arm())
        self.addCleanup(gate.disconnect)
        return gate, transport

    def _events(self, name: str) -> list[dict]:
        out = []
        for path in Path(self.tmp).glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["event"] == name:
                    out.append(rec)
        return out

    def test_estop_calls_emergency_and_names_the_backend(self):
        gate, transport = self._gate(arm=False)
        self.assertEqual(transport.status().state, ConnectionState.CONNECTED)
        gate.emergency_stop()
        self.assertEqual(_names(transport), ["emergency"])
        self.assertTrue(gate.latched)
        rec = self._events("estop")[-1]
        self.assertEqual(rec["backend"], "crazyflie")
        self.assertEqual(rec["flight_request"], "motor_cut")
        self.assertEqual(rec["estop_layer"], "crazyflie_motor_cut")
        self.assertNotIn("land", _names(transport))
        text = "".join(
            path.read_text(encoding="utf-8") for path in Path(self.tmp).glob("*.jsonl")
        )
        self.assertNotIn("21196", text)
        self.assertNotIn("disarm", _names(transport))

    def test_takeoff_is_forwarded_when_battery_and_link_are_known(self):
        gate, transport = self._gate(sample={"battery": 0.9}, arm=True)
        self.assertTrue(gate.command("takeoff", 0.5))
        self.assertTrue(gate.command("hold"))
        self.assertTrue(gate.command("land"))
        self.assertEqual(_names(transport), ["takeoff", "land"])
        self.assertEqual(transport.sent[0][1]["height_m"], 0.5)
        gate.emergency_stop()
        self.assertEqual(_names(transport)[-1], "emergency")
        self.assertEqual(self._events("estop")[-1]["backend"], "crazyflie")

    def test_takeoff_is_refused_without_a_battery_sample(self):
        gate, transport = self._gate(arm=True)
        self.assertFalse(gate.command("takeoff", 0.5))
        self.assertEqual(gate.last_error, "battery")
        self.assertEqual(transport.sent, [])


class _Client:
    def __init__(self, typ: object, name: str) -> None:
        self.typ = typ
        self.name = name
        self.requests: list[object] = []

    def call_async(self, request: object) -> None:
        self.requests.append(request)


class _Pub:
    def __init__(self, typ: object, name: str) -> None:
        self.typ = typ
        self.name = name
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class _Node:
    def __init__(self) -> None:
        self.clients: list[_Client] = []
        self.publishers: list[_Pub] = []
        self.subscriptions: list[tuple] = []

    def create_client(self, typ: object, name: str) -> _Client:
        client = _Client(typ, name)
        self.clients.append(client)
        return client

    def create_publisher(self, typ: object, name: str, qos: int) -> _Pub:
        pub = _Pub(typ, name)
        self.publishers.append(pub)
        return pub

    def create_subscription(self, typ: object, name: str, callback, qos: int):
        self.subscriptions.append((typ, name, callback))
        return None


class _Duration:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _Point:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Quat:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0


class _Header:
    def __init__(self) -> None:
        self.frame_id = ""
        self.stamp = None


class _Pose:
    def __init__(self) -> None:
        self.position = _Point()
        self.orientation = _Quat()


class _Twist:
    def __init__(self) -> None:
        self.linear = _Point()
        self.angular = _Point()


class _Takeoff:
    class Request:
        def __init__(self) -> None:
            self.group_mask = 0
            self.height = 0.0
            self.duration = _Duration()


class _Land:
    class Request:
        def __init__(self) -> None:
            self.group_mask = 0
            self.height = 0.0
            self.duration = _Duration()


class _GoTo:
    class Request:
        def __init__(self) -> None:
            self.group_mask = 0
            self.relative = False
            self.goal = _Point()
            self.yaw = 0.0
            self.duration = _Duration()


class _Empty:
    class Request:
        pass


class _FullState:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _Pose()
        self.twist = _Twist()
        self.acc = _Point()


class _Types:
    Empty = _Empty
    Takeoff = _Takeoff
    Land = _Land
    GoTo = _GoTo
    FullState = _FullState
    Twist = _Twist
    PoseStamped = object
    Status = object


def _pose_msg(x: float, y: float, z: float, quat: tuple[float, float, float, float]):
    pose = _Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        quat
    )

    class _Msg:
        pass

    msg = _Msg()
    msg.pose = pose
    return msg


def _status_msg(voltage: float, info: int, pm_state: int):
    class _Msg:
        pass

    msg = _Msg()
    msg.battery_voltage = voltage
    msg.supervisor_info = info
    msg.pm_state = pm_state
    return msg


class _Calls:
    def __init__(self) -> None:
        self.calls: list = []


class _FakeHL(_Calls):
    def takeoff(self, height, duration, group_mask=0, yaw=0.0) -> None:
        self.calls.append(("takeoff", height, duration, yaw))

    def land(self, height, duration, group_mask=0, yaw=0.0) -> None:
        self.calls.append(("land", height, duration, yaw))

    def go_to(self, x, y, z, yaw, duration, relative=False, linear=False, group_mask=0):
        self.calls.append(("go_to", x, y, z, yaw, duration, relative))


class _FakeCommander(_Calls):
    def send_velocity_world_setpoint(self, vx, vy, vz, yawrate) -> None:
        self.calls.append(("vel", vx, vy, vz, yawrate))

    def send_full_state_setpoint(
        self, pos, vel, acc, orientation, roll, pitch, yaw
    ) -> None:
        self.calls.append(("full", pos, vel, acc, orientation, roll, pitch, yaw))

    def send_setpoint(self, roll, pitch, yawrate, thrust) -> None:
        self.calls.append(("rpyt", roll, pitch, yawrate, thrust))


class _FakeSupervisor(_Calls):
    def send_emergency_stop(self) -> None:
        self.calls.append("emergency")

    def send_arming_request(self, armed: bool) -> None:
        self.calls.append(("arm", armed))


class _FakeLoc(_Calls):
    def send_emergency_stop(self) -> None:
        self.calls.append("emergency")


class _FakeCf:
    def __init__(self) -> None:
        self.high_level_commander = _FakeHL()
        self.commander = _FakeCommander()
        self.supervisor = _FakeSupervisor()
        self.loc = _FakeLoc()

    @property
    def high_level(self) -> _FakeHL:
        return self.high_level_commander


if __name__ == "__main__":
    unittest.main()
