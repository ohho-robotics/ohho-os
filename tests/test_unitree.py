import unittest

from ohho.adapters.errors import AdapterUnavailable
from ohho.adapters.unitree import (
    UnitreeDdsTransport,
    sportstate_to_telemetry,
    velocity_to_move,
)
from ohho.registry import get_spec
from ohho.schema import Velocity


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

    def test_connect_without_sdk_raises_clean_error(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"), "eth0")
        with self.assertRaises(AdapterUnavailable):
            tp.connect()

    def test_estop_safe_without_client(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        tp.emergency_stop()
        tp.send_velocity(Velocity(0.5))


class TestUnitreeStatePolling(unittest.TestCase):
    """Verify the DDS state reader thread maps state dicts → telemetry."""

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

    def __getattr__(self, name):
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

    def test_missing_safety_fields_are_not_invented(self):
        t = sportstate_to_telemetry({"position": [0.0, 0.0, 0.0]})
        for key in ("roll", "pitch", "error_code", "body_height", "mode"):
            self.assertNotIn(key, t.custom)

    def test_sportstate_to_dict_reads_safety_fields(self):
        from types import SimpleNamespace

        msg = SimpleNamespace(
            position=[1.0, 2.0, 0.0],
            velocity=[0.1, 0.0, 0.0],
            yaw_speed=0.0,
            imu_state=SimpleNamespace(rpy=[0.1, 0.2, 0.3]),
            battery_soc=None,
            error_code=0,
            body_height=0.3,
            mode=1,
        )
        d = UnitreeDdsTransport._sportstate_to_dict(msg)
        self.assertEqual(d["imu_rpy"], [0.1, 0.2, 0.3])
        self.assertEqual(d["error_code"], 0)
        self.assertEqual(d["body_height"], 0.3)
        self.assertEqual(d["mode"], 1)
        bare = UnitreeDdsTransport._sportstate_to_dict(SimpleNamespace())
        self.assertNotIn("imu_rpy", bare)
        self.assertNotIn("error_code", bare)

    def test_iface_and_domain_parsing(self):
        spec = get_spec("unitree-go2")
        tp = UnitreeDdsTransport(spec, "lo?domain=1")
        self.assertEqual((tp.iface, tp.domain_id), ("lo", 1))
        tp = UnitreeDdsTransport(spec, "eth0")
        self.assertEqual((tp.iface, tp.domain_id), ("eth0", 0))
        tp = UnitreeDdsTransport(spec, "", iface="enp3s0")
        self.assertEqual(tp.iface, "enp3s0")
        with self.assertRaises(ValueError):
            UnitreeDdsTransport(spec, "lo?domain=x")

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
        self.assertEqual(client.calls, [("StopMove", ()), ("Damp", ())])

    def test_sport_command_without_client_is_noop(self):
        tp = UnitreeDdsTransport(get_spec("unitree-go2"))
        tp.sport_command("stand_up")


if __name__ == "__main__":
    unittest.main()
