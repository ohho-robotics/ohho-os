"""Unit tests for the hardware self-test suite (ohho selftest)."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ohho.adapters.sim import SimTransport
from ohho.cli import main
from ohho.registry import get_spec
from ohho.schema import ConnectionState, Odometry, Telemetry, TransportStatus
from ohho.selftest import (
    CheckResult,
    SelfTestReport,
    _port_exists,
    check_camera_frames,
    check_imu_rate,
    check_serial_link,
    check_sts3215_servos,
    check_wheel_spin_encoders,
    default_hardware_transport_uri,
    default_report_path,
    format_report_text,
    profile_for_robot,
    run_selftest,
)


class TestSelfTestUnitChecks(unittest.TestCase):
    def setUp(self):
        self.spec = get_spec("omnibot")
        self.sim_tp = SimTransport(self.spec)
        self.sim_tp.connect()

    def tearDown(self):
        self.sim_tp.disconnect()

    # ── Check 1: Serial link ──────────────────────────────────────────────────

    def test_check_serial_link_sim(self):
        res = check_serial_link(self.sim_tp, sim=True)
        self.assertEqual(res.name, "serial_link")
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.measurements["protocol"], "simulated")
        self.assertEqual(res.measurements["state"], "connected")

    def test_check_serial_link_hardware_missing_pyserial(self):
        with patch.dict("sys.modules", {"serial": None}):
            res = check_serial_link(self.sim_tp, sim=False, base_port="/dev/ttyUSB0")
            self.assertEqual(res.status, "SKIP")
            self.assertIn("pyserial", res.reason)

    def test_check_serial_link_hardware_disconnected(self):
        fake_tp = MagicMock()
        fake_tp.protocol = "serial"
        fake_tp.port = "/dev/nonexistent_port_123"
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.DISCONNECTED,
            label="Offline",
        )
        mock_serial = MagicMock()
        with patch.dict("sys.modules", {"serial": mock_serial}):
            res = check_serial_link(
                fake_tp, sim=False, base_port="/dev/nonexistent_port_123"
            )
            self.assertEqual(res.status, "SKIP")
            self.assertIn("not connected", res.reason)

    def test_check_serial_link_hardware_connected(self):
        fake_tp = MagicMock()
        fake_tp.protocol = "serial"
        fake_tp.port = "/dev/ttyUSB0"
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
            label="Online",
            latency_ms=1.5,
        )
        mock_serial = MagicMock()
        with patch.dict("sys.modules", {"serial": mock_serial}):
            res = check_serial_link(fake_tp, sim=False, base_port="/dev/ttyUSB0")
            self.assertEqual(res.status, "PASS")
            self.assertIn("connected", res.reason)

    # ── Check 2: Wheel spin & encoders ────────────────────────────────────────

    def test_check_wheel_spin_safety_locked_skip(self):
        # On hardware without allow_spin, check must SKIP
        res = check_wheel_spin_encoders(self.sim_tp, sim=False, allow_spin=False)
        self.assertEqual(res.status, "SKIP")
        self.assertIn("safety lock", res.reason.lower())
        self.assertTrue(res.measurements.get("safety_locked"))

    def test_check_wheel_spin_hardware_disconnected_skip(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.DISCONNECTED,
        )
        res = check_wheel_spin_encoders(fake_tp, sim=False, allow_spin=True)
        self.assertEqual(res.status, "SKIP")
        self.assertIn("not connected", res.reason.lower())

    def test_check_wheel_spin_sim_pass(self):
        res = check_wheel_spin_encoders(self.sim_tp, sim=True)
        self.assertEqual(res.status, "PASS")
        self.assertTrue(res.measurements["all_ok"])
        wheels = res.measurements["wheels"]
        for w in ["front_left", "front_right", "rear_left", "rear_right"]:
            self.assertIn(w, wheels)
            self.assertGreater(wheels[w]["delta_ticks"], 0)
            self.assertTrue(wheels[w]["ok"])

    def test_check_wheel_spin_wheel_fail(self):
        # Mock transport where rear_right does not increment
        mock_tp = MagicMock()
        encs = {
            "front_left": 100,
            "front_right": 100,
            "rear_left": 100,
            "rear_right": 100,
        }

        def mock_get_enc():
            return dict(encs)

        mock_tp.get_wheel_encoders = mock_get_enc

        # Only increment FL, FR, RL, not RR
        def mock_send_m(fl, fr, rl, rr):
            if fl:
                encs["front_left"] += 50
            if fr:
                encs["front_right"] += 50
            if rl:
                encs["rear_left"] += 50
            # rr doesn't move!

        mock_tp.send_motor = mock_send_m
        res = check_wheel_spin_encoders(mock_tp, sim=True)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("rear_right", res.reason)
        self.assertFalse(res.measurements["all_ok"])

    # ── Check 3: IMU publishing rate ──────────────────────────────────────────

    def test_check_imu_rate_sim(self):
        res = check_imu_rate(self.sim_tp, sim=True)
        self.assertEqual(res.status, "PASS")
        self.assertGreaterEqual(res.measurements["rate_hz"], 10.0)

    def test_check_imu_rate_hardware_pass(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        fake_tp.get_imu_rate.return_value = 25.0
        res = check_imu_rate(fake_tp, sim=False, min_rate_hz=10.0)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.measurements["rate_hz"], 25.0)

    def test_check_imu_rate_hardware_low_fail(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        fake_tp.get_imu_rate.return_value = 4.5
        res = check_imu_rate(fake_tp, sim=False, min_rate_hz=10.0, sample_window_s=0.05)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("below minimum", res.reason)

    def test_check_imu_rate_hardware_zero_fail(self):
        fake_tp = MagicMock()
        fake_tp.get_imu_rate.return_value = 0.0
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        fake_tp.read.return_value = Telemetry(odom=Odometry())
        res = check_imu_rate(fake_tp, sim=False, min_rate_hz=10.0, sample_window_s=0.05)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("No IMU packets", res.reason)

    # ── Check 4: STS3215 servos ───────────────────────────────────────────────

    def test_check_sts3215_servos_sim(self):
        res = check_sts3215_servos(self.sim_tp, sim=True)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.measurements["count"], 6)
        self.assertTrue(res.measurements["all_responding"])
        servos = res.measurements["servos"]
        for s in servos:
            self.assertIn(s["id"], [1, 2, 3, 4, 5, 6])
            self.assertGreaterEqual(s["voltage"], 6.0)
            self.assertLessEqual(s["voltage"], 9.0)
            self.assertLessEqual(s["temperature"], 70.0)
            self.assertTrue(s["online"])

    def test_check_sts3215_servos_voltage_fail(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="feetech", state=ConnectionState.CONNECTED
        )
        fake_tp.read_servo_diagnostics.return_value = [
            {
                "id": 1,
                "name": "j1",
                "voltage": 5.2,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 2,
                "name": "j2",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 3,
                "name": "j3",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 4,
                "name": "j4",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 5,
                "name": "j5",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 6,
                "name": "j6",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
        ]
        res = check_sts3215_servos(fake_tp, sim=False)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("voltage 5.2V out of range", res.reason)

    def test_check_sts3215_servos_temperature_fail(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="feetech", state=ConnectionState.CONNECTED
        )
        fake_tp.read_servo_diagnostics.return_value = [
            {
                "id": i,
                "name": f"j{i}",
                "voltage": 7.4,
                "temperature": 75.0 if i == 3 else 30.0,
                "position": 0.0,
                "online": True,
            }
            for i in range(1, 7)
        ]
        res = check_sts3215_servos(fake_tp, sim=False)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("temperature 75.0°C exceeds 70.0°C", res.reason)

    def test_check_sts3215_servos_missing_fail(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="feetech", state=ConnectionState.CONNECTED
        )
        # Only 5 servos instead of 6
        fake_tp.read_servo_diagnostics.return_value = [
            {
                "id": i,
                "name": f"j{i}",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            }
            for i in range(1, 6)
        ]
        res = check_sts3215_servos(fake_tp, sim=False, expected_count=6)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("Expected 6 servos but received 5", res.reason)

    def test_check_sts3215_servos_disconnected_skip(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="feetech", state=ConnectionState.DISCONNECTED
        )
        fake_tp.read_servo_diagnostics.return_value = []
        res = check_sts3215_servos(fake_tp, sim=False)
        self.assertEqual(res.status, "SKIP")
        self.assertIn("arm bus not connected", res.reason)

    # ── Check 5: Camera frames ────────────────────────────────────────────────

    def test_check_camera_frames_sim(self):
        res = check_camera_frames(sim=True, required_frames=5)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.measurements["frames_received"], 5)
        self.assertEqual(res.measurements["width"], 640)
        self.assertEqual(res.measurements["height"], 480)

    def test_check_camera_frames_missing_cv2(self):
        with patch.dict("sys.modules", {"cv2": None}):
            res = check_camera_frames(sim=False)
            self.assertEqual(res.status, "SKIP")
            self.assertIn("OpenCV", res.reason)

    def test_check_camera_frames_capture_mock(self):
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = MagicMock()
        fake_frame.size = 640 * 480 * 3
        fake_frame.shape = (480, 640, 3)
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            res = check_camera_frames(sim=False, required_frames=3)
            self.assertEqual(res.status, "PASS")
            self.assertEqual(res.measurements["frames_received"], 3)
            self.assertEqual(res.measurements["width"], 640)
            self.assertEqual(res.measurements["height"], 480)
            mock_cap.release.assert_called_once()


class TestSelfTestReportAndRunner(unittest.TestCase):
    def test_default_report_path_format(self):
        p = default_report_path()
        self.assertTrue(re.match(r"^selftest-\d{8}-\d{6}\.json$", p))

    def test_profile_for_go2_is_legged(self):
        spec = get_spec("unitree-go2")
        profile = profile_for_robot(spec)
        self.assertEqual(profile.id, "legged_go2")
        self.assertNotIn("wheel_spin_encoders", profile.check_ids)
        self.assertNotIn("sts3215_servos", profile.check_ids)

    def test_run_selftest_go2_sim_only_legged_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "go2-sim.json")
            report = run_selftest(
                robot="unitree-go2",
                sim=True,
                out_path=out_file,
            )
            self.assertEqual(report.mode, "sim")
            self.assertEqual(report.summary["profile"], "legged_go2")
            self.assertNotIn("wheel_spin_encoders", report.checks)
            self.assertNotIn("sts3215_servos", report.checks)
            self.assertIn("dds_link", report.checks)
            self.assertEqual(report.checks["dds_link"].status, "PASS")
            self.assertEqual(report.checks["leg_motors"].status, "PASS")
            self.assertEqual(report.checks["stand_lie_cycle"].status, "SKIP")
            self.assertEqual(report.overall_status, "PASS")

    def test_default_hardware_uri_go2_and_transport_override(self):
        spec = get_spec("unitree-go2")
        uri = default_hardware_transport_uri(spec)
        self.assertTrue(uri.startswith("dds://"))
        custom = default_hardware_transport_uri(spec, transport_uri="dds://lo?domain=1")
        self.assertEqual(custom, "dds://lo?domain=1")

    def test_run_selftest_sim_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test-report.json")
            report = run_selftest(
                robot="omnibot",
                sim=True,
                allow_spin=True,
                out_path=out_file,
            )
            self.assertEqual(report.robot, "omnibot")
            self.assertEqual(report.mode, "sim")
            self.assertEqual(report.overall_status, "PASS")
            self.assertEqual(report.summary["total"], 5)
            self.assertEqual(report.summary["passed"], 5)
            self.assertEqual(report.summary["failed"], 0)
            self.assertEqual(report.summary["skipped"], 0)

            # Check that all 5 checks are present in the report
            expected_checks = {
                "serial_link",
                "wheel_spin_encoders",
                "imu_rate",
                "sts3215_servos",
                "camera_frames",
            }
            self.assertEqual(set(report.checks.keys()), expected_checks)

            # Verify saved JSON on disk
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["overall_status"], "PASS")
            self.assertEqual(data["summary"]["passed"], 5)
            self.assertIn("serial_link", data["checks"])

    def test_format_report_text(self):
        report = run_selftest("omnibot", sim=True)
        txt = format_report_text(report)
        self.assertIn("OhhO OS Hardware Self-Test Report", txt)
        self.assertIn("[PASS] serial_link", txt)
        self.assertIn("[PASS] wheel_spin_encoders", txt)
        self.assertIn("[PASS] imu_rate", txt)
        self.assertIn("[PASS] sts3215_servos", txt)
        self.assertIn("[PASS] camera_frames", txt)
        self.assertIn("Summary: 5 passed", txt)

    def test_cli_selftest_sim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "cli-report.json")
            code = main(["selftest", "--sim", "--out", out_file])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out_file))

    def test_cli_selftest_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "fail-report.json")
            # Force a check to FAIL
            with patch("ohho.selftest.check_wheel_spin_encoders") as mock_wheel:
                mock_wheel.return_value = CheckResult(
                    name="wheel_spin_encoders",
                    status="FAIL",
                    reason="Motor encoder wire disconnected",
                )
                code = main(["selftest", "--sim", "--out", out_file])
                self.assertEqual(code, 1)

    def test_run_selftest_omnibot_hardware_no_pyserial(self):
        with patch.dict("sys.modules", {"serial": None}):
            report = run_selftest(robot="omnibot", sim=False, allow_spin=False)
            self.assertEqual(report.robot, "omnibot")
            self.assertEqual(report.mode, "hardware")
            self.assertEqual(report.checks["serial_link"].status, "SKIP")
            self.assertIn("pyserial", report.checks["serial_link"].reason)
            self.assertNotEqual(report.overall_status, "FAIL")


class TestSelftestHonestyBugbotRegressions(unittest.TestCase):
    """Explicit regression tests for Bugbot review findings on PR #9 (OHH-87)."""

    # ── Finding 1: Hardware encoders are not real ticks ──────────────────────
    def test_finding_1_wheel_spin_skips_when_encoders_not_available(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        fake_tp.get_wheel_encoders.return_value = {}  # Yahboom adapter returns empty dict
        res = check_wheel_spin_encoders(fake_tp, sim=False, allow_spin=True)
        self.assertEqual(res.status, "SKIP")
        self.assertEqual(res.reason, "encoder counts not available from this adapter")

    # ── Finding 2: IMU check fails without samples ───────────────────────────
    def test_finding_2_imu_rate_bounded_wait_eventual_success(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        rates = [0.0, 0.0, 25.0]

        def get_rate():
            return rates.pop(0) if rates else 25.0

        fake_tp.get_imu_rate = get_rate
        fake_tp.get_imu_samples.return_value = []
        res = check_imu_rate(fake_tp, sim=False, min_rate_hz=10.0, sample_window_s=0.5)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(res.measurements["rate_hz"], 25.0)

    def test_finding_2_imu_rate_fails_after_wait_with_sample_count(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="serial",
            state=ConnectionState.CONNECTED,
        )
        fake_tp.get_imu_rate.return_value = 0.0
        fake_tp.get_imu_samples.return_value = []
        res = check_imu_rate(fake_tp, sim=False, min_rate_hz=10.0, sample_window_s=0.05)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("0 samples received in", res.reason)
        self.assertEqual(res.measurements["samples_received"], 0)

    # ── Finding 3: Servo diagnostics hide missing readings ───────────────────
    def test_finding_3_servos_fail_on_missing_readings_no_defaults(self):
        fake_tp = MagicMock()
        fake_tp.status.return_value = TransportStatus(
            protocol="feetech", state=ConnectionState.CONNECTED
        )
        # Servo 1 missing voltage, Servo 2 missing temp, Servo 3 missing position, Servo 4 offline
        fake_tp.read_servo_diagnostics.return_value = [
            {
                "id": 1,
                "name": "j1",
                "voltage": None,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 2,
                "name": "j2",
                "voltage": 7.4,
                "temperature": None,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 3,
                "name": "j3",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": None,
                "online": True,
            },
            {
                "id": 4,
                "name": "j4",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": False,
            },
            {
                "id": 5,
                "name": "j5",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
            {
                "id": 6,
                "name": "j6",
                "voltage": 7.4,
                "temperature": 30.0,
                "position": 0.0,
                "online": True,
            },
        ]
        res = check_sts3215_servos(fake_tp, sim=False)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("Servo 1 (j1) missing voltage reading", res.reason)
        self.assertIn("Servo 2 (j2) missing temperature reading", res.reason)
        self.assertIn("Servo 3 (j3) missing position reading", res.reason)
        self.assertIn("Servo 4 (j4) offline", res.reason)

    # ── Finding 4: Arm port flag ignored alone ────────────────────────────────
    def test_finding_4_arm_port_works_alone(self):
        with patch("ohho.adapters.resolve_transport") as mock_resolve:
            mock_resolve.return_value = SimTransport(get_spec("omnibot"))
            run_selftest("omnibot", sim=False, arm_port="/dev/custom_arm")
            self.assertTrue(mock_resolve.called)
            uri = mock_resolve.call_args[0][0]
            self.assertEqual(uri, "serial:///dev/ttyUSB0,/dev/custom_arm")

    def test_finding_4_base_port_works_alone(self):
        with patch("ohho.adapters.resolve_transport") as mock_resolve:
            mock_resolve.return_value = SimTransport(get_spec("omnibot"))
            run_selftest("omnibot", sim=False, base_port="/dev/custom_base")
            self.assertTrue(mock_resolve.called)
            uri = mock_resolve.call_args[0][0]
            self.assertEqual(uri, "serial:///dev/custom_base,/dev/ttyACM0")

    # ── Finding 5: Windows device checks always skip ──────────────────────────
    def test_finding_5_windows_com_port_detection_and_link_status(self):
        mock_serial = MagicMock()
        mock_tools = MagicMock()
        mock_list_ports = MagicMock()
        mock_port = MagicMock()
        mock_port.device = "COM3"
        mock_list_ports.comports.return_value = [mock_port]
        mock_tools.list_ports = mock_list_ports
        mock_serial.tools = mock_tools

        with patch.dict(
            "sys.modules",
            {
                "serial": mock_serial,
                "serial.tools": mock_tools,
                "serial.tools.list_ports": mock_list_ports,
            },
        ):
            self.assertTrue(_port_exists("COM3"))
            self.assertTrue(_port_exists(r"\\.\COM3"))
            self.assertFalse(_port_exists("COM99"))

            fake_tp = MagicMock()
            fake_tp.protocol = "serial"
            fake_tp.port = "COM3"
            fake_tp.status.return_value = TransportStatus(
                protocol="serial",
                state=ConnectionState.DISCONNECTED,
            )
            # Present on system but disconnected -> FAIL
            res_fail = check_serial_link(fake_tp, sim=False, base_port="COM3")
            self.assertEqual(res_fail.status, "FAIL")

            # Not present on system -> SKIP
            fake_tp.port = "COM99"
            res_skip = check_serial_link(fake_tp, sim=False, base_port="COM99")
            self.assertEqual(res_skip.status, "SKIP")

    def test_finding_5_windows_camera_probe_and_capture(self):
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch("os.name", "nt"), patch.dict("sys.modules", {"cv2": mock_cv2}):
            # Camera probe fails to open device -> SKIP with probe reason
            mock_cap.isOpened.return_value = False
            res_probe = check_camera_frames(sim=False, camera_idx=2)
            self.assertEqual(res_probe.status, "SKIP")
            self.assertIn(
                "No camera found at index 2 via OpenCV probe", res_probe.reason
            )

            # Camera device opens but fails to stream frames -> FAIL
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            res_fail = check_camera_frames(sim=False, camera_idx=2, required_frames=3)
            self.assertEqual(res_fail.status, "FAIL")
            self.assertIn("Captured only 0/3 frames", res_fail.reason)

    # ── Finding 6: Hardware mode falls back to sim ────────────────────────────
    def test_finding_6_hardware_mode_never_falls_back_to_sim(self):
        with patch(
            "ohho.adapters.resolve_transport", side_effect=RuntimeError("port busy")
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "hardware-fail.json")
                with self.assertRaises(RuntimeError) as cm:
                    run_selftest("omnibot", sim=False, out_path=out_path)
                self.assertIn(
                    "Failed to initialize hardware transport", str(cm.exception)
                )
                self.assertTrue(os.path.exists(out_path))
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["mode"], "hardware")
                self.assertEqual(data["overall_status"], "FAIL")
                self.assertEqual(data["summary"]["failed"], 1)

    def test_go2_hardware_broken_transport_fails_not_sim_pass(self):
        with patch(
            "ohho.adapters.resolve_transport", side_effect=RuntimeError("dds down")
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "go2-hw-fail.json")
                with self.assertRaises(RuntimeError):
                    run_selftest(
                        "unitree-go2",
                        sim=False,
                        transport_uri="dds://lo?domain=1",
                        out_path=out_path,
                    )
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["mode"], "hardware")
                self.assertEqual(data["overall_status"], "FAIL")
                self.assertEqual(data["checks"]["dds_link"]["status"], "FAIL")
                self.assertNotIn("wheel_spin_encoders", data["checks"])
                self.assertNotEqual(
                    data["overall_status"],
                    "PASS",
                    "broken hardware transport must not report sim PASS",
                )

    def test_finding_6_cli_hardware_all_skip_exits_code_3(self):
        all_skip_report = SelfTestReport(
            timestamp="2026-10-03T12:00:00Z",
            robot="omnibot",
            mode="hardware",
            overall_status="PASS",
            summary={
                "total": 5,
                "passed": 0,
                "failed": 0,
                "skipped": 5,
                "duration_s": 0.1,
            },
            checks={},
        )
        with patch("ohho.selftest.run_selftest", return_value=all_skip_report):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_f = os.path.join(tmpdir, "skip-report.json")
                code = main(["selftest", "--allow-spin", "--out", out_f])
                self.assertEqual(code, 3)

                code_allowed = main(
                    ["selftest", "--allow-spin", "--allow-all-skip", "--out", out_f]
                )
                self.assertEqual(code_allowed, 0)


if __name__ == "__main__":
    unittest.main()
