"""Read-only px4+dds connect and the PX4 doctor. No Gazebo and no rclpy."""

from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ohho import capabilities as caps
from ohho import cli
from ohho.adapters import resolve_transport
from ohho.adapters.errors import AdapterUnavailable
from ohho.adapters.px4_dds import (
    PINNED_OFFBOARD_FIELDS,
    Px4CommandRefused,
    Px4DdsError,
    Px4DdsTransport,
    gps_fix_name,
)
from ohho.px4_doctor import (
    MIN_VEHICLE_STATUS_HZ,
    PX4_DOCTOR_CHECKS,
    PX4_DOCTOR_ROBOTS,
    Px4DoctorSample,
    evaluate_px4_doctor,
    live_px4_sample,
    observe_px4_graph,
    probe_xrce_agent,
)
from ohho.registry import get_spec
from ohho.safety import AuditLog, SafetyGate, requires_hardware_arming
from ohho.schema import Velocity

_URI = "px4+dds://127.0.0.1:8888"
_HARDWARE_URI = "px4+dds://10.0.0.8:8888"
_WALL = 1_760_000_000.0


class _Msg:
    def __init__(self) -> None:
        return None


class _Offboard(_Msg):
    @staticmethod
    def get_fields_and_field_types() -> dict[str, str]:
        return dict(PINNED_OFFBOARD_FIELDS)


class _BadOffboard(_Msg):
    @staticmethod
    def get_fields_and_field_types() -> dict[str, str]:
        fields = dict(PINNED_OFFBOARD_FIELDS)
        del fields["thrust_and_torque"]
        del fields["direct_actuator"]
        fields["actuator"] = "bool"
        return fields


class _Factory:
    OffboardControlMode = _Offboard
    TrajectorySetpoint = _Msg
    VehicleCommand = _Msg
    VehicleLocalPosition = _Msg
    VehicleAttitude = _Msg
    VehicleStatus = _Msg
    BatteryStatus = _Msg
    VehicleLandDetected = _Msg
    FailsafeFlags = _Msg


class _BadFactory(_Factory):
    OffboardControlMode = _BadOffboard


class _GpsFactory(_Factory):
    SensorGps = _Msg


class GraphNode:
    """Mock DDS graph. Publishers are recorded and must stay empty in read-only."""

    def __init__(self) -> None:
        self.subs: list[SimpleNamespace] = []
        self.publishers: dict[str, object] = {}
        self.publisher_calls = 0
        self.timers: list[object] = []
        self.publisher_infos: dict[str, list[object]] = {}

    def create_publisher(self, msg_type, topic, qos):
        self.publisher_calls += 1
        self.publishers[topic] = SimpleNamespace(
            topic=topic, qos=qos, msg_type=msg_type, published=[]
        )
        return self.publishers[topic]

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = SimpleNamespace(
            topic=topic, callback=callback, qos=qos, msg_type=msg_type
        )
        self.subs.append(sub)
        return sub

    def create_timer(self, period, callback):
        timer = SimpleNamespace(period=period, callback=callback, cancelled=False)
        self.timers.append(timer)
        return timer

    def get_publishers_info_by_topic(self, topic: str) -> list[object]:
        return list(self.publisher_infos.get(topic, []))

    def inject(self, topic: str, msg: object) -> None:
        matched = [sub for sub in self.subs if sub.topic == topic]
        if not matched:
            raise AssertionError(f"no subscription for {topic}")
        for sub in matched:
            sub.callback(msg)


class SubOnlyNode:
    """A node that cannot publish. Read-only connect must still succeed."""

    def __init__(self) -> None:
        self.subs: list[str] = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subs.append(topic)
        return SimpleNamespace(topic=topic)

    def create_publisher(self, *args, **kwargs):
        raise AssertionError("create_publisher was called")

    def create_timer(self, *args, **kwargs):
        raise AssertionError("create_timer was called")


def _qos_token(qos: object) -> str:
    reliability = getattr(qos, "reliability", None)
    name = getattr(reliability, "name", str(reliability))
    return str(name).lower()


def _open_readonly(
    uri: str = _URI,
    *,
    robot: str = "px4-sitl",
    factory: type = _Factory,
    node: GraphNode | None = None,
) -> tuple[Px4DdsTransport, GraphNode]:
    graph = node or GraphNode()
    transport = Px4DdsTransport(
        get_spec(robot),
        uri,
        node=graph,
        msg_factory=factory(),
        environ={},
        read_only=True,
    )
    transport.connect()
    return transport, graph


def _info(reliability: str) -> SimpleNamespace:
    return SimpleNamespace(
        qos_profile=SimpleNamespace(reliability=SimpleNamespace(name=reliability))
    )


def _healthy_sample(**overrides: object) -> Px4DoctorSample:
    base: dict[str, object] = {
        "agent_reachable": True,
        "agent_detail": "1 publisher on /fmu/out/vehicle_status_v1",
        "vehicle_status_hz": 5.0,
        "vehicle_status_detail": "/fmu/out/vehicle_status_v1 at 5.00 Hz",
        "px4_msgs_fields": dict(PINNED_OFFBOARD_FIELDS),
        "px4_msgs_detail": "fields read",
        "ros_distro": "jazzy",
        "qos_best_effort": True,
        "qos_detail": "best-effort",
        "armed": False,
        "armed_detail": "armed=false",
        "gps_fix": "3d",
        "gps_detail": "fix 3d",
        "battery_remaining": 0.8,
        "battery_voltage_v": 16.2,
        "battery_detail": "remaining 0.8",
    }
    base.update(overrides)
    return Px4DoctorSample(**base)  # type: ignore[arg-type]


class TestRegistry(unittest.TestCase):
    def test_px4_specs_are_aerial_without_an_arm(self) -> None:
        for robot_id, adapter in (("px4-sitl", "sim"), ("px4-x500", "px4-dds")):
            spec = get_spec(robot_id)
            self.assertEqual(spec.adapter, adapter)
            self.assertIn(caps.AERIAL, spec.capabilities)
            self.assertNotIn(caps.MANIPULATION, spec.capabilities)
            self.assertNotIn(caps.BASE_DRIVE, spec.capabilities)
            self.assertEqual(spec.max_lin, 2.0)
            self.assertEqual(spec.max_ang, 0.8)
            self.assertIn(robot_id, PX4_DOCTOR_ROBOTS)


class TestReadOnlyTransport(unittest.TestCase):
    def test_loopback_sitl_creates_no_fmu_in_publisher_or_command(self) -> None:
        transport, node = _open_readonly()
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.publisher_calls, 0)
        self.assertEqual(node.timers, [])
        self.assertEqual(transport.publisher_topics, [])
        self.assertEqual(transport.vehicle_commands_sent, 0)
        self.assertIsNone(transport._timer)
        self.assertTrue(
            any(sub.topic == "/fmu/out/vehicle_status_v1" for sub in node.subs)
        )
        self.assertFalse(any("/fmu/in/" in sub.topic for sub in node.subs))
        for sub in node.subs:
            self.assertIn("best_effort", _qos_token(sub.qos))
        for action in ("arm", "disarm", "takeoff", "land", "rtl", "hold"):
            with self.assertRaises(Px4CommandRefused):
                transport.send_flight(action)
        with self.assertRaises(Px4CommandRefused):
            transport.send_flight(
                "offboard_velocity",
                vx=0.1,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0,
                frame="LOCAL_NED",
            )
        with self.assertRaises(Px4CommandRefused):
            transport.send_velocity(Velocity(0.2, 0.0, 0.1))
        transport.send_velocity(Velocity())
        with self.assertRaises(Px4CommandRefused):
            transport.emergency_stop()
        transport._on_heartbeat()
        node_timers = list(node.timers)
        for timer in node_timers:
            timer.callback()
        self.assertEqual(node.publishers, {})
        self.assertEqual(transport.vehicle_commands_sent, 0)
        self.assertTrue(transport._estopped)

    def test_x500_non_loopback_read_only_still_publishes_nothing(self) -> None:
        transport, node = _open_readonly(_HARDWARE_URI, robot="px4-x500")
        self.assertFalse(transport.sim_target)
        self.assertTrue(transport.requires_hardware_arming)
        self.assertTrue(requires_hardware_arming(transport, {}))
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.timers, [])
        self.assertIn("read-only", transport.status().label)
        transport.disconnect()
        self.assertEqual(node.publisher_calls, 0)

    def test_message_mismatch_creates_no_endpoint(self) -> None:
        node = GraphNode()
        transport = Px4DdsTransport(
            get_spec("px4-sitl"),
            _URI,
            node=node,
            msg_factory=_BadFactory(),
            environ={},
            read_only=True,
        )
        with self.assertRaises(Px4DdsError) as caught:
            transport.connect()
        self.assertIn("release/1.17", str(caught.exception))
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.subs, [])
        self.assertEqual(node.timers, [])

    def test_sub_only_node_can_connect(self) -> None:
        node = SubOnlyNode()
        transport = Px4DdsTransport(
            get_spec("px4-sitl"),
            _URI,
            node=node,
            msg_factory=_Factory(),
            read_only=True,
        )
        transport.connect()
        self.assertIn("/fmu/out/vehicle_status_v1", node.subs)
        self.assertNotIn("/fmu/in/vehicle_command", node.subs)

    def test_namespace_and_gps_sample(self) -> None:
        transport, node = _open_readonly(
            "px4+dds://127.0.0.1:8888?ns=uav1", factory=_GpsFactory
        )
        topics = [sub.topic for sub in node.subs]
        self.assertIn("/uav1/fmu/out/vehicle_status_v1", topics)
        self.assertIn("/uav1/fmu/out/sensor_gps", topics)
        self.assertFalse(any("/fmu/in/" in topic for topic in topics))
        node.inject("/uav1/fmu/out/sensor_gps", SimpleNamespace(fix_type=3))
        self.assertEqual(transport.read().custom["drone"]["gps_fix"], "3d")
        self.assertEqual(node.publishers, {})

    def test_missing_gps_fix_type_is_not_reported_as_none(self) -> None:
        self.assertIsNone(gps_fix_name(None))
        self.assertIsNone(gps_fix_name(True))
        self.assertIsNone(gps_fix_name(9))
        self.assertEqual(gps_fix_name(1), "none")

    def test_gate_zero_and_refusal_do_not_publish(self) -> None:
        tmp = tempfile.mkdtemp(prefix="ohho-px4-ro-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        transport, node = _open_readonly()
        gate = SafetyGate(
            transport,
            get_spec("px4-sitl"),
            environ={},
            auto_tick=False,
            audit=AuditLog(tmp, wall_clock=lambda: _WALL),
        )
        gate.send_velocity(Velocity(0.4, 0.0, 0.2))
        with self.assertRaises(Px4CommandRefused):
            transport.send_flight("arm")
        gate.disconnect()
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.publisher_calls, 0)
        self.assertEqual(transport.vehicle_commands_sent, 0)

    def test_resolve_read_only_is_px4_only(self) -> None:
        transport = resolve_transport(_URI, get_spec("px4-sitl"), read_only=True)
        self.assertIsInstance(transport, Px4DdsTransport)
        self.assertTrue(transport.read_only)
        writable = resolve_transport(_URI, get_spec("px4-x500"))
        self.assertFalse(writable.read_only)
        with self.assertRaises(AdapterUnavailable):
            resolve_transport("sim://", get_spec("px4-sitl"), read_only=True)
        with self.assertRaises(AdapterUnavailable):
            resolve_transport(None, get_spec("px4-x500"), read_only=True)


class TestDoctorReport(unittest.TestCase):
    def test_healthy_report_has_every_check(self) -> None:
        report = evaluate_px4_doctor("px4-sitl", _URI, _healthy_sample())
        self.assertTrue(report.ok)
        self.assertEqual(
            tuple(check.name for check in report.checks), PX4_DOCTOR_CHECKS
        )
        payload = report.to_dict()
        self.assertEqual(
            [row["name"] for row in payload["checks"]], list(PX4_DOCTOR_CHECKS)
        )
        self.assertTrue(all(row["ok"] for row in payload["checks"]))
        by_name = {row["name"]: row for row in payload["checks"]}
        self.assertFalse(by_name["armed"]["value"])
        self.assertEqual(by_name["gps_fix"]["value"], "3d")
        self.assertEqual(by_name["qos"]["value"], "best_effort")
        self.assertEqual(by_name["ros_distro"]["value"], "jazzy")
        self.assertGreaterEqual(
            by_name["vehicle_status_rate"]["value"], MIN_VEHICLE_STATUS_HZ
        )
        self.assertEqual(by_name["px4_msgs"]["value"], "release/1.17")
        self.assertEqual(by_name["battery"]["value"], 0.8)

    def test_humble_and_observed_none_gps_pass(self) -> None:
        report = evaluate_px4_doctor(
            "px4-x500",
            _HARDWARE_URI,
            _healthy_sample(ros_distro="Humble", gps_fix="none", battery_remaining=0.0),
        )
        self.assertTrue(report.ok)
        by_name = {check.name: check for check in report.checks}
        self.assertEqual(by_name["ros_distro"].value, "humble")
        self.assertEqual(by_name["gps_fix"].value, "none")
        self.assertEqual(by_name["battery"].value, 0.0)

    def test_missing_observation_is_not_a_pass(self) -> None:
        report = evaluate_px4_doctor(
            "px4-sitl",
            _URI,
            _healthy_sample(
                agent_reachable=False,
                agent_detail="agent did not reply",
                vehicle_status_hz=None,
                vehicle_status_detail="topic missing",
                px4_msgs_fields=None,
                px4_msgs_detail="px4_msgs is not importable",
                ros_distro=None,
                qos_best_effort=None,
                qos_detail="QoS was not observed. Not assumed best-effort.",
                armed=None,
                armed_detail="armed is not reported as false",
                gps_fix=None,
                gps_detail="No GPS fix is reported.",
                battery_remaining=None,
                battery_detail="No fraction is reported.",
            ),
        )
        self.assertFalse(report.ok)
        for check in report.checks:
            self.assertFalse(check.ok, check.name)
        by_name = {check.name: check for check in report.checks}
        self.assertIsNone(by_name["armed"].value)
        self.assertIsNot(by_name["armed"].value, False)
        self.assertIsNone(by_name["gps_fix"].value)
        self.assertNotEqual(by_name["qos"].value, "best_effort")
        self.assertIn("not importable", by_name["px4_msgs"].detail)
        self.assertIn("unset", by_name["ros_distro"].detail)
        self.assertIn("Not assumed jazzy", by_name["ros_distro"].detail)

    def test_each_bad_value_fails_on_its_own(self) -> None:
        cases = {
            "xrce_agent": {"agent_reachable": None, "agent_detail": "not checked"},
            "vehicle_status_rate": {"vehicle_status_hz": 0.0},
            "px4_msgs": {
                "px4_msgs_fields": {"actuator": "bool"},
            },
            "ros_distro": {"ros_distro": "rolling"},
            "qos": {"qos_best_effort": False, "qos_detail": "reliable"},
            "armed": {"armed": True, "armed_detail": "arming_state is armed"},
            "gps_fix": {"gps_fix": "locked"},
            "battery": {"battery_remaining": -1.0},
        }
        for name, overrides in cases.items():
            report = evaluate_px4_doctor("px4-sitl", _URI, _healthy_sample(**overrides))
            self.assertFalse(report.ok, name)
            failed = [check.name for check in report.checks if not check.ok]
            self.assertEqual(failed, [name], failed)

    def test_bool_rate_and_battery_are_not_numbers(self) -> None:
        report = evaluate_px4_doctor(
            "px4-sitl",
            _URI,
            _healthy_sample(vehicle_status_hz=True, battery_remaining=True),
        )
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["vehicle_status_rate"].ok)
        self.assertFalse(by_name["battery"].ok)


class TestDoctorGraph(unittest.TestCase):
    def _observe(
        self,
        node: GraphNode,
        *,
        spin,
        probe=None,
        fields=None,
        ros: str | None = "jazzy",
        uri: str = _URI,
    ):
        return observe_px4_graph(
            node,
            {
                "VehicleStatus": object,
                "BatteryStatus": object,
                "SensorGps": object,
            },
            uri,
            agent_probe=probe,
            spin=spin,
            window_s=1.0,
            ros_distro=ros,
            px4_fields=dict(PINNED_OFFBOARD_FIELDS) if fields is None else fields,
            px4_detail="mock fields",
        )

    def test_mock_graph_passes_and_creates_no_publisher(self) -> None:
        node = GraphNode()
        topic = "/fmu/out/vehicle_status_v1"

        def probe(host: str, port: int) -> tuple[bool, str]:
            raise AssertionError(f"UDP probe {host}:{port} should not run")

        def spin(graph: GraphNode, window: float) -> None:
            self.assertEqual(window, 1.0)
            for _ in range(5):
                graph.inject(
                    topic,
                    SimpleNamespace(arming_state=1, nav_state=2),
                )
            graph.inject(
                "/fmu/out/battery_status_v1",
                SimpleNamespace(remaining=0.7, voltage_v=15.5),
            )
            graph.inject("/fmu/out/sensor_gps", SimpleNamespace(fix_type=3))

        node.publisher_infos[topic] = [_info("BEST_EFFORT")]
        sample = self._observe(node, spin=spin, probe=probe)
        self.assertEqual(node.publisher_calls, 0)
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.timers, [])
        for sub in node.subs:
            self.assertNotIn("/fmu/in/", sub.topic)
            self.assertIn("best_effort", _qos_token(sub.qos))
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        self.assertTrue(report.ok, [c.detail for c in report.checks if not c.ok])
        self.assertEqual(sample.vehicle_status_hz, 5.0)
        self.assertFalse(sample.armed)
        self.assertEqual(sample.gps_fix, "3d")

    def test_missing_agent_and_topics_fail_honestly(self) -> None:
        node = GraphNode()

        def spin(graph: GraphNode, window: float) -> None:
            return None

        sample = self._observe(
            node,
            spin=spin,
            probe=lambda host, port: (False, f"no reply from {host}:{port}"),
        )
        report = evaluate_px4_doctor("px4-x500", _HARDWARE_URI, sample)
        self.assertFalse(report.ok)
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["xrce_agent"].ok)
        self.assertIn("no reply", by_name["xrce_agent"].detail)
        self.assertFalse(by_name["vehicle_status_rate"].ok)
        self.assertEqual(by_name["vehicle_status_rate"].value, 0.0)
        self.assertFalse(by_name["qos"].ok)
        self.assertIn("Not assumed best-effort", by_name["qos"].detail)
        self.assertFalse(by_name["armed"].ok)
        self.assertIsNone(by_name["armed"].value)
        self.assertFalse(by_name["gps_fix"].ok)
        self.assertIsNone(by_name["gps_fix"].value)
        self.assertFalse(by_name["battery"].ok)
        self.assertTrue(by_name["px4_msgs"].ok)
        self.assertTrue(by_name["ros_distro"].ok)
        self.assertEqual(node.publisher_calls, 0)

    def test_agent_up_without_samples_does_not_invent_state(self) -> None:
        node = GraphNode()

        def spin(graph: GraphNode, window: float) -> None:
            return None

        sample = self._observe(
            node,
            spin=spin,
            probe=lambda host, port: (True, f"reply from {host}:{port}"),
        )
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        by_name = {check.name: check for check in report.checks}
        self.assertTrue(by_name["xrce_agent"].ok)
        self.assertFalse(by_name["armed"].ok)
        self.assertIsNone(sample.armed)
        self.assertIsNone(sample.gps_fix)
        self.assertIsNone(sample.battery_remaining)

    def test_reliable_publisher_qos_fails(self) -> None:
        node = GraphNode()
        topic = "/fmu/out/vehicle_status_v1"
        node.publisher_infos[topic] = [_info("RELIABLE")]

        def spin(graph: GraphNode, window: float) -> None:
            graph.inject(topic, SimpleNamespace(arming_state=1))
            graph.inject(
                "/fmu/out/battery_status_v1",
                SimpleNamespace(remaining=0.5, voltage_v=16.0),
            )
            graph.inject("/fmu/out/sensor_gps", SimpleNamespace(fix_type=3))

        sample = self._observe(
            node,
            spin=spin,
            probe=lambda *_: (_ for _ in ()).throw(AssertionError("udp")),
        )
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["qos"].ok)
        self.assertIn("not best-effort", by_name["qos"].detail)
        self.assertTrue(by_name["xrce_agent"].ok)

    def test_status_without_arming_state_is_not_false(self) -> None:
        node = GraphNode()
        topic = "/fmu/out/vehicle_status_v1"
        node.publisher_infos[topic] = [_info("best_effort")]

        def spin(graph: GraphNode, window: float) -> None:
            graph.inject(topic, SimpleNamespace(nav_state=2))
            graph.inject(
                "/fmu/out/battery_status_v1",
                SimpleNamespace(remaining=-1.0, voltage_v=0.0),
            )
            graph.inject("/fmu/out/sensor_gps", SimpleNamespace(fix_type=99))

        sample = self._observe(node, spin=spin)
        self.assertIsNone(sample.armed)
        self.assertIsNone(sample.gps_fix)
        self.assertIsNone(sample.battery_remaining)
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["armed"].ok)
        self.assertFalse(by_name["gps_fix"].ok)
        self.assertFalse(by_name["battery"].ok)

    def test_bool_arming_state_is_not_reported_disarmed(self) -> None:
        node = GraphNode()
        topic = "/fmu/out/vehicle_status_v1"
        node.publisher_infos[topic] = [_info("best_effort")]

        def spin(graph: GraphNode, window: float) -> None:
            graph.inject(topic, SimpleNamespace(arming_state=True))

        sample = self._observe(node, spin=spin)
        self.assertIsNone(sample.armed)
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["armed"].ok)
        self.assertIsNone(by_name["armed"].value)

    def test_namespaced_topic_is_what_gets_sampled(self) -> None:
        node = GraphNode()
        topic = "/uav1/fmu/out/vehicle_status_v1"
        node.publisher_infos[topic] = [_info("best_effort")]

        def spin(graph: GraphNode, window: float) -> None:
            graph.inject(topic, SimpleNamespace(arming_state=2))

        sample = self._observe(
            node,
            spin=spin,
            uri="px4+dds://127.0.0.1:8888?ns=uav1",
        )
        self.assertIn(topic, sample.vehicle_status_detail)
        self.assertTrue(sample.armed)
        self.assertEqual(node.publisher_calls, 0)


class TestLiveSample(unittest.TestCase):
    def test_missing_rclpy_does_not_probe_or_pass(self) -> None:
        def import_module(name: str):
            raise ImportError(name)

        with mock.patch("ohho.px4_doctor._find_spec", return_value=None):
            with mock.patch(
                "ohho.px4_doctor._import_module", side_effect=import_module
            ):
                with mock.patch(
                    "ohho.px4_doctor.probe_xrce_agent",
                    side_effect=AssertionError("socket"),
                ):
                    sample = live_px4_sample(_URI, environ={})
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        self.assertFalse(report.ok)
        by_name = {check.name: check for check in report.checks}
        self.assertFalse(by_name["xrce_agent"].ok)
        self.assertIn("not probed", by_name["xrce_agent"].detail)
        self.assertIn("rclpy", by_name["xrce_agent"].detail)
        self.assertFalse(by_name["vehicle_status_rate"].ok)
        self.assertIsNone(by_name["armed"].value)
        self.assertFalse(by_name["px4_msgs"].ok)
        self.assertIn("px4_msgs is not importable", by_name["px4_msgs"].detail)
        self.assertFalse(by_name["ros_distro"].ok)

    def test_msgs_can_match_while_the_graph_is_absent(self) -> None:
        module = SimpleNamespace(OffboardControlMode=_Offboard)

        def import_module(name: str):
            if name == "px4_msgs.msg":
                return module
            raise ImportError(name)

        with mock.patch("ohho.px4_doctor._find_spec", return_value=None):
            with mock.patch(
                "ohho.px4_doctor._import_module", side_effect=import_module
            ):
                with mock.patch(
                    "ohho.px4_doctor.probe_xrce_agent",
                    side_effect=AssertionError("socket"),
                ):
                    sample = live_px4_sample(_URI, environ={"ROS_DISTRO": "jazzy"})
        report = evaluate_px4_doctor("px4-sitl", _URI, sample)
        by_name = {check.name: check for check in report.checks}
        self.assertTrue(by_name["px4_msgs"].ok)
        self.assertTrue(by_name["ros_distro"].ok)
        self.assertFalse(by_name["xrce_agent"].ok)
        self.assertFalse(by_name["armed"].ok)
        self.assertIsNone(by_name["armed"].value)

    def test_udp_probe_needs_a_reply(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        try:
            ok, detail = probe_xrce_agent("127.0.0.1", port, timeout=0.05)
        finally:
            sock.close()
        self.assertFalse(ok)
        self.assertIn("did not reply", detail)
        self.assertIn("Not reported reachable", detail)

    def test_udp_probe_accepts_a_reply(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        ready = threading.Event()

        def reply() -> None:
            sock.settimeout(2)
            ready.set()
            data, addr = sock.recvfrom(16)
            self.assertEqual(data, b"\x00")
            sock.sendto(b"ok", addr)

        thread = threading.Thread(target=reply)
        thread.start()
        self.assertTrue(ready.wait(1))
        try:
            ok, detail = probe_xrce_agent("127.0.0.1", port, timeout=1.0)
        finally:
            thread.join(2)
            sock.close()
        self.assertTrue(ok)
        self.assertIn("replied", detail)


class TestCli(unittest.TestCase):
    def test_environment_doctor_still_runs(self) -> None:
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            code = cli.main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("runtimes", out.getvalue())

    def test_connect_flag_and_arm_conflict(self) -> None:
        args = cli.build_parser().parse_args(
            ["connect", "px4-x500", "--transport", _HARDWARE_URI, "--read-only"]
        )
        self.assertTrue(args.read_only)
        with mock.patch.object(
            cli.Robot, "connect", side_effect=AssertionError("connect")
        ):
            code = cli.main(
                [
                    "connect",
                    "px4-sitl",
                    "--transport",
                    _URI,
                    "--read-only",
                    "--arm",
                ]
            )
        self.assertEqual(code, 2)

    def test_connect_passes_read_only(self) -> None:
        args = cli.build_parser().parse_args(
            ["connect", "px4-sitl", "--transport", _URI, "--read-only"]
        )
        fake = mock.Mock()
        fake.gated = True
        fake.arm = mock.Mock()
        with mock.patch.object(cli.Robot, "connect", return_value=fake) as conn:
            cli._connect(args)
        self.assertTrue(conn.call_args.kwargs.get("read_only"))
        fake.arm.assert_not_called()

    def test_read_only_connect_creates_no_publisher(self) -> None:
        node = GraphNode()
        classes = {
            "OffboardControlMode": _Offboard,
            "TrajectorySetpoint": _Msg,
            "VehicleCommand": _Msg,
            "VehicleStatus": _Msg,
            "VehicleLocalPosition": _Msg,
            "VehicleAttitude": _Msg,
            "BatteryStatus": _Msg,
            "VehicleLandDetected": _Msg,
            "FailsafeFlags": _Msg,
        }
        tmp = tempfile.mkdtemp(prefix="ohho-px4-cli-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        previous = os.environ.get("OHHO_SAFETY_DIR")
        armed = os.environ.get("OHHO_ARM_HARDWARE")
        os.environ["OHHO_SAFETY_DIR"] = tmp
        os.environ.pop("OHHO_ARM_HARDWARE", None)
        try:
            with mock.patch.object(Px4DdsTransport, "_resolve_node", lambda self: node):
                with mock.patch.object(
                    Px4DdsTransport, "_load_messages", lambda self: classes
                ):
                    with mock.patch("ohho.cli.time.sleep", return_value=None):
                        code = cli.main(
                            [
                                "connect",
                                "px4-x500",
                                "--transport",
                                _HARDWARE_URI,
                                "--read-only",
                                "--runtime",
                                "native",
                            ]
                        )
        finally:
            if previous is None:
                os.environ.pop("OHHO_SAFETY_DIR", None)
            else:
                os.environ["OHHO_SAFETY_DIR"] = previous
            if armed is None:
                os.environ.pop("OHHO_ARM_HARDWARE", None)
            else:
                os.environ["OHHO_ARM_HARDWARE"] = armed
        self.assertEqual(code, 0)
        self.assertEqual(node.publishers, {})
        self.assertEqual(node.publisher_calls, 0)
        self.assertEqual(node.timers, [])
        self.assertTrue(
            any(sub.topic == "/fmu/out/vehicle_status_v1" for sub in node.subs)
        )
        self.assertFalse(any("/fmu/in/" in sub.topic for sub in node.subs))

    def test_writable_non_loopback_connect_is_refused(self) -> None:
        node = GraphNode()
        previous = os.environ.get("OHHO_ARM_HARDWARE")
        os.environ.pop("OHHO_ARM_HARDWARE", None)
        try:
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                code = cli.main(
                    [
                        "connect",
                        "px4-x500",
                        "--transport",
                        _HARDWARE_URI,
                        "--runtime",
                        "native",
                    ]
                )
        finally:
            if previous is None:
                os.environ.pop("OHHO_ARM_HARDWARE", None)
            else:
                os.environ["OHHO_ARM_HARDWARE"] = previous
        self.assertEqual(code, 2)
        self.assertIn("not a loopback", err.getvalue())
        self.assertEqual(node.publisher_calls, 0)

    def test_doctor_usage_and_json(self) -> None:
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(cli.main(["doctor", "--robot", "px4-sitl"]), 2)
            self.assertEqual(
                cli.main(
                    ["doctor", "--robot", "omnibot", "--transport", _URI, "--json"]
                ),
                2,
            )
            self.assertEqual(
                cli.main(
                    [
                        "doctor",
                        "--robot",
                        "px4-sitl",
                        "--transport",
                        "dds://lo",
                        "--json",
                    ]
                ),
                2,
            )
        sample = _healthy_sample(agent_reachable=False, agent_detail="agent missing")
        with mock.patch("ohho.cli.sample_px4_graph", return_value=sample):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = cli.main(
                    [
                        "doctor",
                        "--robot",
                        "px4-sitl",
                        "--transport",
                        _URI,
                        "--json",
                    ]
                )
        self.assertEqual(code, 1)
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(
            [row["name"] for row in payload["checks"]], list(PX4_DOCTOR_CHECKS)
        )
        self.assertFalse(payload["checks"][0]["ok"])
        healthy = _healthy_sample()
        with mock.patch("ohho.cli.sample_px4_graph", return_value=healthy):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = cli.main(
                    [
                        "doctor",
                        "--robot",
                        "px4-x500",
                        "--transport",
                        _HARDWARE_URI,
                        "--json",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out.getvalue())["ok"])

    def test_dependencies_stay_empty(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("\ndependencies = []\n", text)


if __name__ == "__main__":
    unittest.main()
