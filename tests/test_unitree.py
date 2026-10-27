import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ohho.adapters.errors import AdapterUnavailable
from ohho.adapters.unitree import (
    UnitreeDdsTransport,
    low_message_to_dict,
    merge_state_dicts,
    parse_dds_address,
    sport_message_to_dict,
    sportstate_to_telemetry,
    velocity_to_move,
)
from ohho.lidar import DEFAULT_NUM_RAYS
from ohho.nav import Navigator
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import AuditLog, SafetyGate
from ohho.schema import ConnectionState, Velocity


class TestUnitreeMapping(unittest.TestCase):
    def test_velocity_to_move(self):
        self.assertEqual(velocity_to_move(Velocity(0.5, 0.1, 0.2)), (0.5, 0.1, 0.2))

    def test_sportstate_to_telemetry(self):
        t = sportstate_to_telemetry(
            {
                "position": [1.0, 2.0, 0.0],
                "velocity": [0.3, 0.0, 0.0],
                "yaw_speed": 0.4,
                "imu_rpy": [0.0, 0.0, 1.57],
                "battery": 0.8,
            }
        )
        self.assertAlmostEqual(t.odom.x, 1.0)
        self.assertAlmostEqual(t.odom.y, 2.0)
        self.assertAlmostEqual(t.odom.theta, 1.57)
        self.assertAlmostEqual(t.odom.vx, 0.3)
        self.assertAlmostEqual(t.odom.omega, 0.4)
        self.assertAlmostEqual(t.battery, 0.8)

    def test_sportstate_handles_missing_fields(self):
        t = sportstate_to_telemetry({})
        self.assertEqual(t.odom.x, 0.0)
        self.assertIsNone(t.battery)

    def test_sportstate_maps_joints_and_gait(self):
        t = sportstate_to_telemetry(
            {
                "imu_rpy": [0.0, 0.0, 0.0],
                "joints": [{"name": "FR_hip", "position": 0.1, "velocity": 0.01}],
                "gait_type": 2,
                "foot_force": [1, 2, 3, 4],
            }
        )
        self.assertEqual(len(t.joints), 1)
        self.assertAlmostEqual(t.joints[0].position, 0.1)
        self.assertEqual(t.custom["gait_type"], 2)
        self.assertEqual(t.custom["foot_force"], [1.0, 2.0, 3.0, 4.0])

    def test_merge_state_dicts(self):
        merged = merge_state_dicts(
            {"position": [1.0, 0.0, 0.0], "imu_rpy": [0.0, 0.0, 0.5]},
            {"battery": 0.75, "joints": [{"name": "FR_hip", "position": 0.2}]},
        )
        self.assertAlmostEqual(merged["battery"], 0.75)
        self.assertEqual(merged["joints"][0]["name"], "FR_hip")
        self.assertAlmostEqual(merged["position"][0], 1.0)

    def test_low_message_to_dict(self):
        motors = [SimpleNamespace(q=0.1 + i * 0.01, dq=0.0) for i in range(12)]
        msg = SimpleNamespace(
            motor_state=motors,
            bms_state=SimpleNamespace(soc=80),
            tick=2_000_000,
        )
        d = low_message_to_dict(msg)
        self.assertAlmostEqual(d["battery"], 0.8)
        self.assertEqual(d["joints"][0]["name"], "FR_hip")
        self.assertAlmostEqual(d["joints"][0]["position"], 0.1)

    def test_connect_without_sdk_raises_clean_error(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "eth0")
        with self.assertRaises(AdapterUnavailable):
            tp.connect()

    def test_estop_safe_without_client(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        tp.emergency_stop()
        tp.send_velocity(Velocity(0.5))


class TestUnitreeStatePolling(unittest.TestCase):
    """Verify the state reader thread maps state dicts → telemetry."""

    def test_state_factory_feeds_telemetry(self):
        state = {
            "position": [1.0, 2.0, 0.0],
            "velocity": [0.3, 0.0, 0.0],
            "yaw_speed": 0.4,
            "imu_rpy": [0.0, 0.0, 1.57],
            "battery": 0.9,
        }

        def factory():
            return lambda: state

        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "eth0", state_factory=factory)
        seen = []
        tp.on_telemetry(seen.append)
        tp.connect()
        try:
            import time as _t

            _t.sleep(0.15)
            self.assertGreaterEqual(len(seen), 1)
            t = tp.read()
            self.assertAlmostEqual(t.odom.x, 1.0)
            self.assertAlmostEqual(t.odom.vx, 0.3)
            self.assertAlmostEqual(t.battery, 0.9)
        finally:
            tp.disconnect()

    def test_state_factory_none_is_safe(self):
        def factory():
            return lambda: None

        tp = UnitreeDdsTransport(get_spec("unitree-go2"), state_factory=factory)
        tp.connect()
        try:
            t = tp.read()
            self.assertEqual(t.odom.x, 0.0)
        finally:
            tp.disconnect()

    def test_telemetry_callback_fires_on_state(self):
        def factory():
            return lambda: {
                "position": [1.0, 0.0, 0.0],
                "velocity": [0.2, 0.0, 0.0],
                "yaw_speed": 0.0,
                "imu_rpy": [0.0, 0.0, 0.0],
            }

        tp = UnitreeDdsTransport(get_spec("unitree-go2"), state_factory=factory)
        seen = []
        tp.on_telemetry(seen.append)
        tp.connect()
        try:
            import time as _t

            _t.sleep(0.15)
            self.assertGreaterEqual(len(seen), 1)
        finally:
            tp.disconnect()


class _FakeSportClient:
    def __init__(self):
        self.calls = []
        self.move_code = 0
        self.stop_code = 0
        self.timeout = None

    def SetTimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def Init(self) -> None:
        return None

    def __getattr__(self, name):
        if name == "Move":
            return lambda vx, vy, vyaw: self.move_code
        if name == "StopMove":
            return lambda: self.stop_code

        def call(*args):
            self.calls.append((name, args))

        return call


class TestUnitreeSafetyHooks(unittest.TestCase):
    """Fields and commands the SafetyGate relies on (OHH-117)."""

    def test_state_exposes_orientation_and_faults(self):
        t = sportstate_to_telemetry(
            {
                "imu_rpy": [0.1, -0.2, 1.0],
                "error_code": 0,
                "body_height": 0.31,
                "mode": 1,
            }
        )
        self.assertAlmostEqual(t.custom["roll"], 0.1)
        self.assertAlmostEqual(t.custom["pitch"], -0.2)
        self.assertEqual(t.custom["error_code"], 0)
        self.assertAlmostEqual(t.custom["body_height"], 0.31)
        self.assertEqual(t.custom["mode"], 1)
        self.assertEqual(t.custom["posture"], "standing")

    def test_missing_safety_fields_are_not_invented(self):
        t = sportstate_to_telemetry({"position": [0.0, 0.0, 0.0]})
        for key in ("roll", "pitch", "error_code", "body_height", "mode"):
            self.assertNotIn(key, t.custom)

    def test_sportstate_to_dict_reads_safety_fields(self):
        msg = SimpleNamespace(
            position=[1.0, 2.0, 0.0],
            velocity=[0.1, 0.0, 0.0],
            yaw_speed=0.0,
            imu_state=SimpleNamespace(rpy=[0.1, 0.2, 0.3]),
            stamp=SimpleNamespace(sec=10, nanosec=500_000_000),
            error_code=0,
            body_height=0.3,
            mode=1,
            gait_type=2,
            foot_force=[1, 2, 3, 4],
        )
        d = sport_message_to_dict(msg)
        self.assertEqual(d["imu_rpy"], [0.1, 0.2, 0.3])
        self.assertEqual(d["error_code"], 0)
        self.assertEqual(d["body_height"], 0.3)
        self.assertEqual(d["mode"], 1)
        self.assertEqual(d["gait_type"], 2)
        bare = sport_message_to_dict(SimpleNamespace())
        self.assertNotIn("imu_rpy", bare)
        self.assertNotIn("error_code", bare)

    def test_parse_dds_address_defaults_and_explicit(self):
        self.assertEqual(parse_dds_address(""), ("lo", 1))
        self.assertEqual(parse_dds_address("lo?domain=1"), ("lo", 1))
        self.assertEqual(parse_dds_address("eth0"), ("eth0", 0))
        self.assertEqual(parse_dds_address("lo"), ("lo", 0))
        self.assertEqual(parse_dds_address("", iface_kw="enp3s0"), ("enp3s0", 0))
        with self.assertRaises(ValueError):
            parse_dds_address("lo?domain=x")

    def test_iface_and_domain_parsing(self):
        spec = get_spec("unitree-go2")
        tp = UnitreeDdsTransport(spec, "lo?domain=1")
        self.assertEqual((tp.iface, tp.domain_id), ("lo", 1))
        tp = UnitreeDdsTransport(spec, "eth0")
        self.assertEqual((tp.iface, tp.domain_id), ("eth0", 0))
        tp = UnitreeDdsTransport(spec, "", iface="enp3s0")
        self.assertEqual((tp.iface, tp.domain_id), ("enp3s0", 0))
        tp = UnitreeDdsTransport(spec, "")
        self.assertEqual((tp.iface, tp.domain_id), ("lo", 1))

    def test_stale_state_reports_error(self):
        tp = UnitreeDdsTransport(
            get_spec("unitree-go2"), state_factory=lambda: lambda: None
        )
        tp.connect()
        tp._last_msg_mono = 0.0
        tp._state = ConnectionState.CONNECTED
        with patch("ohho.adapters.unitree.time") as mtime:
            mtime.monotonic.return_value = 2.0
            mtime.time.return_value = 100.0
            st = tp.status()
        self.assertEqual(st.state, ConnectionState.ERROR)
        self.assertIsNotNone(st.error)
        tp.disconnect()

    def test_subscriber_callbacks_update_telemetry(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "lo?domain=1")
        sport_msg = SimpleNamespace(
            position=[1.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            yaw_speed=0.0,
            imu_state=SimpleNamespace(rpy=[0.05, -0.02, 0.3]),
            stamp=SimpleNamespace(sec=1, nanosec=0),
            error_code=0,
            body_height=0.3,
            mode=1,
            gait_type=1,
            foot_force=[0, 0, 0, 0],
        )
        low_msg = SimpleNamespace(
            motor_state=[SimpleNamespace(q=0.42, dq=0.0)],
            bms_state=SimpleNamespace(soc=50),
            tick=1_000_000,
        )
        self.assertFalse(tp.odom_received)
        tp._on_low_message(low_msg)
        self.assertFalse(tp.odom_received)
        self.assertAlmostEqual(tp.read().battery, 0.5)
        tp._on_sport_message(SimpleNamespace())
        self.assertFalse(tp.odom_received)
        tp._on_sport_message(sport_msg)
        tp._on_low_message(low_msg)
        self.assertTrue(tp.odom_received)
        t = tp.read()
        self.assertAlmostEqual(t.odom.x, 1.0)
        origin = UnitreeDdsTransport(get_spec("unitree-go2"), "lo?domain=1")
        origin._on_sport_message(
            SimpleNamespace(
                position=[0.0, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                yaw_speed=0.0,
            )
        )
        self.assertTrue(origin.odom_received)
        self.assertAlmostEqual(t.custom["roll"], 0.05)
        self.assertAlmostEqual(t.joints[0].position, 0.42)
        self.assertAlmostEqual(t.battery, 0.5)

    def test_move_return_code_logged(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        client = _FakeSportClient()
        client.move_code = 7
        tp._client = client
        with self.assertLogs("ohho.adapters.unitree", level="WARNING") as logs:
            tp.send_velocity(Velocity(0.1))
        self.assertTrue(any("Move" in line and "7" in line for line in logs.output))

    def test_sport_command_maps_to_client(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        client = _FakeSportClient()
        tp._client = client
        tp.sport_command("stand_up")
        tp.sport_command("euler", 0.1, 0.0, -0.1)
        tp.sport_command("rise_sit")
        tp.sport_command("damp")
        self.assertEqual(
            client.calls,
            [
                ("StandUp", ()),
                ("Euler", (0.1, 0.0, -0.1)),
                ("RiseSit", ()),
                ("Damp", ()),
            ],
        )
        with self.assertRaises(ValueError):
            tp.sport_command("front_flip")

    def test_sport_command_blocked_when_estopped(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        client = _FakeSportClient()
        tp._client = client
        tp.emergency_stop()
        client.calls.clear()
        tp.sport_command("stand_up")
        tp.sport_command("stop_move")
        tp.sport_command("damp")
        self.assertEqual(client.calls, [("Damp", ())])

    def test_sport_command_without_client_is_noop(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        tp.sport_command("stand_up")


class TestUnitreeLidarScan(unittest.TestCase):
    def test_sportstate_pointcloud_to_scan(self):
        t = sportstate_to_telemetry(
            {
                "position": [0.0, 0.0, 0.0],
                "imu_rpy": [0.0, 0.0, 0.0],
                "pointcloud": [(2.0, 0.0, 0.2)],
            }
        )
        self.assertIsNotNone(t.scan)
        self.assertEqual(len(t.scan.ranges), DEFAULT_NUM_RAYS)
        self.assertAlmostEqual(t.scan.ranges[DEFAULT_NUM_RAYS // 2], 2.0, places=3)

    def test_state_factory_feeds_scan(self):
        state = {
            "position": [0.0, 0.0, 0.0],
            "imu_rpy": [0.0, 0.0, 0.0],
            "pointcloud": [(1.5, 0.0, 0.25)],
        }

        def factory():
            return lambda: state

        tp = UnitreeDdsTransport(
            get_spec("unitree-go2"), "lo?domain=1", state_factory=factory
        )
        tp.connect()
        try:
            import time as _t

            _t.sleep(0.15)
            scan = tp.read().scan
            self.assertIsNotNone(scan)
            self.assertAlmostEqual(scan.ranges[DEFAULT_NUM_RAYS // 2], 1.5, places=3)
        finally:
            tp.disconnect()

    def test_scan_visible_through_safety_gate(self):
        state = {
            "position": [0.0, 0.0, 0.0],
            "imu_rpy": [0.0, 0.0, 0.0],
            "pointcloud": [(2.5, 0.0, 0.2)],
        }

        def factory():
            return lambda: state

        inner = UnitreeDdsTransport(
            get_spec("unitree-go2"), "lo?domain=1", state_factory=factory
        )
        spec = get_spec("unitree-go2")
        gate = SafetyGate(
            inner,
            spec,
            audit=AuditLog(None, wall_clock=lambda: 0.0),
            auto_tick=False,
            clock=lambda: 0.0,
            isatty=lambda: False,
        )
        gate.connect()
        try:
            import time as _t

            _t.sleep(0.15)
            scan = gate.read().scan
            self.assertIsNotNone(scan)
            nav = Navigator(Robot(spec, gate, NativeRuntime()))
            nav.update_map()
            self.assertGreater(nav.grid.counts()["occupied"], 0)
        finally:
            gate.disconnect()

    def test_subscriber_path_keeps_lidar_scan(self):
        """Pose subscribers and the lidar callback both feed one telemetry snapshot."""
        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "lo?domain=1")
        tp._on_lidar([(1.7, 0.0, 0.2)])
        tp._on_sport_message(
            SimpleNamespace(
                position=[1.0, 0.0, 0.0],
                velocity=[0.0, 0.0, 0.0],
                yaw_speed=0.0,
                imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
            )
        )
        t = tp.read()
        self.assertAlmostEqual(t.odom.x, 1.0)
        self.assertIsNotNone(t.scan)
        self.assertAlmostEqual(t.scan.ranges[DEFAULT_NUM_RAYS // 2], 1.7, places=3)

    def test_pointcloud_factory_publishes_scan_with_state_factory(self):
        state = {
            "position": [0.4, 0.0, 0.0],
            "imu_rpy": [0.0, 0.0, 0.0],
        }

        def state_factory():
            return lambda: state

        tp = UnitreeDdsTransport(
            get_spec("unitree-go2"),
            "lo?domain=1",
            state_factory=state_factory,
            pointcloud_factory=lambda: [(1.1, 0.0, 0.2)],
        )
        tp.connect()
        try:
            import time as _t

            _t.sleep(0.2)
            t = tp.read()
            self.assertAlmostEqual(t.odom.x, 0.4)
            self.assertIsNotNone(t.scan)
            self.assertAlmostEqual(t.scan.ranges[DEFAULT_NUM_RAYS // 2], 1.1, places=3)
        finally:
            tp.disconnect()


if __name__ == "__main__":
    unittest.main()
