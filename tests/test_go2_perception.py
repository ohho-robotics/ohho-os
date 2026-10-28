"""Unit tests for Perception on Go2 (OHH-125).

Front-camera frames (VideoClient / MuJoCo render) -> VlmPerceptor -> spatio-temporal memory.
Includes tests for:
- UnitreeDdsTransport.get_camera_frame() with frame_factory and stub mode
- Safe domain != 0 loopback vs real robot domain 0 hardware safety
- Detection image_x parsing and coordinate projection from odom + bearing
- VlmPerceptor 1 Hz rate-limiting and latency/cost hooks
- CameraPerceptor end-to-end integration with SpatialMemory
- CLI `ohho look unitree-go2 --transport dds://lo?domain=1`
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ohho.adapters.unitree import UnitreeDdsTransport
from ohho.cli import main
from ohho.memory import SpatialMemory, default_memory_path
from ohho.perception import (
    CameraPerceptor,
    Detection,
    VlmCallStats,
    VlmPerceptor,
    _FakeClient,
    estimate_vlm_cost,
    parse_vlm_detections,
    project_detection,
    project_detections,
    remember_detections,
)
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import SafetyGate
from ohho.schema import Odometry, Telemetry

# Block the Unitree SDK so domain-0 and missing-SDK tests never open DDS.
_BLOCK_UNITREE_SDK = {
    "unitree_sdk2py": None,
    "unitree_sdk2py.core": None,
    "unitree_sdk2py.core.channel": None,
}

# Synthetic 1x1 JPEG frame bytes for tests
SYNTHETIC_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
    b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01"
    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    b"\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


class TestUnitreeCameraTransport(unittest.TestCase):
    def test_get_camera_frame_with_factory(self):
        spec = get_spec("unitree-go2")

        def state_factory():
            return lambda: {
                "position": [1.0, 2.0, 0.0],
                "imu_rpy": [0.0, 0.0, 0.0],
            }

        def frame_factory():
            return lambda: SYNTHETIC_JPEG

        tp = UnitreeDdsTransport(
            spec,
            "lo?domain=1",
            state_factory=state_factory,
            frame_factory=frame_factory,
        )
        tp.connect()
        try:
            frame = tp.get_camera_frame()
            self.assertEqual(frame, SYNTHETIC_JPEG)
        finally:
            tp.disconnect()

    def test_get_camera_frame_returns_none_when_unavailable(self):
        spec = get_spec("unitree-go2")

        def state_factory():
            return lambda: {"position": [0.0, 0.0, 0.0]}

        tp = UnitreeDdsTransport(spec, state_factory=state_factory)
        tp.connect()
        try:
            self.assertIsNone(tp.get_camera_frame())
        finally:
            tp.disconnect()

    def test_get_camera_frame_handles_exception_gracefully(self):
        spec = get_spec("unitree-go2")

        def broken_frame_factory():
            def getter():
                raise RuntimeError("camera connection reset")

            return getter

        tp = UnitreeDdsTransport(
            spec,
            state_factory=lambda: lambda: None,
            frame_factory=broken_frame_factory,
        )
        tp.connect()
        try:
            self.assertIsNone(tp.get_camera_frame())
        finally:
            tp.disconnect()

    def test_make_video_frame_getter_patterns(self):
        # Case 1: client.GetImageSample() -> (0, data)
        client_sample_ok = SimpleNamespace(
            GetImageSample=lambda: (0, b"frame_sample_bytes")
        )
        getter = UnitreeDdsTransport._make_video_frame_getter(client_sample_ok)
        self.assertEqual(getter(), b"frame_sample_bytes")

        # Case 2: client.GetImageSample() -> non-zero error code
        client_sample_err = SimpleNamespace(GetImageSample=lambda: (-1, b"corrupted"))
        getter_err = UnitreeDdsTransport._make_video_frame_getter(client_sample_err)
        self.assertIsNone(getter_err())

        # Case 3: client.GetImage() -> bytes
        client_get_image = SimpleNamespace(GetImage=lambda: b"frame_get_image")
        getter_img = UnitreeDdsTransport._make_video_frame_getter(client_get_image)
        self.assertEqual(getter_img(), b"frame_get_image")

        # Case 4: client method raises
        def fail():
            raise OSError("dds read error")

        client_fail = SimpleNamespace(GetImageSample=fail)
        getter_fail = UnitreeDdsTransport._make_video_frame_getter(client_fail)
        self.assertIsNone(getter_fail())

    def test_stub_dds_mode_on_domain_1(self):
        spec = get_spec("unitree-go2")
        with mock.patch.dict(os.environ, {"OHHO_STUB_DDS": "1"}):
            tp = UnitreeDdsTransport(spec, "lo?domain=1")
            status = tp.connect()
            try:
                self.assertEqual(status.state.value, "connected")
                frame = tp.get_camera_frame()
                self.assertIsNotNone(frame)
                self.assertTrue(frame.startswith(b"\xff\xd8"))
                t = tp.read()
                self.assertTrue(tp.odom_received)
                self.assertIsNotNone(t.odom)
                self.assertEqual(t.odom.x, 1.0)
                self.assertEqual(t.odom.y, 2.0)
            finally:
                tp.disconnect()

    def test_stub_dds_mode_refused_on_domain_0_hardware(self):
        """Never allow automated stubbing on domain 0 (real hardware domain)."""
        spec = get_spec("unitree-go2")
        with (
            mock.patch.dict(os.environ, {"OHHO_STUB_DDS": "1"}),
            mock.patch.dict("sys.modules", _BLOCK_UNITREE_SDK),
        ):
            tp = UnitreeDdsTransport(spec, "lo?domain=0")
            from ohho.adapters.errors import AdapterUnavailable

            with self.assertRaises(AdapterUnavailable):
                tp.connect()

    def test_stub_dds_mode_refused_on_non_loopback(self):
        """A non-loopback interface must open the SDK, even on a non-zero domain."""
        spec = get_spec("unitree-go2")
        with (
            mock.patch.dict(os.environ, {"OHHO_STUB_DDS": "1"}),
            mock.patch.dict("sys.modules", _BLOCK_UNITREE_SDK),
        ):
            tp = UnitreeDdsTransport(spec, "eth0?domain=1")
            from ohho.adapters.errors import AdapterUnavailable

            with self.assertRaises(AdapterUnavailable):
                tp.connect()


class TestSafetyGateCameraForwarding(unittest.TestCase):
    def test_gate_forwards_camera_frame(self):
        spec = get_spec("unitree-go2")
        inner = UnitreeDdsTransport(
            spec,
            state_factory=lambda: (
                lambda: {
                    "position": [0.0, 0.0, 0.0],
                    "imu_rpy": [0.0, 0.0, 0.0],
                }
            ),
            frame_factory=lambda: lambda: SYNTHETIC_JPEG,
        )
        gate = SafetyGate(inner, spec, auto_tick=False)
        gate.connect()
        try:
            self.assertEqual(gate.get_camera_frame(), SYNTHETIC_JPEG)
        finally:
            gate.disconnect()

    def test_robot_get_camera_frame(self):
        spec = get_spec("unitree-go2")
        inner = UnitreeDdsTransport(
            spec,
            state_factory=lambda: (
                lambda: {
                    "position": [0.0, 0.0, 0.0],
                    "imu_rpy": [0.0, 0.0, 0.0],
                }
            ),
            frame_factory=lambda: lambda: SYNTHETIC_JPEG,
        )
        bot = Robot(spec, inner, NativeRuntime())
        inner.connect()
        try:
            self.assertEqual(bot.get_camera_frame(), SYNTHETIC_JPEG)
        finally:
            bot.disconnect()


class TestPerceptionParsingAndProjection(unittest.TestCase):
    def test_parse_vlm_detections_with_image_x(self):
        text = """
        [
            {"label": "chair", "confidence": 0.88, "x": 0.25},
            {"label": "desk", "confidence": 0.95, "image_x": 0.75},
            {"label": "laptop", "confidence": 0.8, "box_2d": [100, 400, 200, 600]},
            {"label": "cup", "confidence": 0.7}
        ]
        """
        dets = parse_vlm_detections(text)
        self.assertEqual(len(dets), 4)

        self.assertEqual(dets[0].label, "chair")
        self.assertAlmostEqual(dets[0].image_x, 0.25)
        self.assertIn("img_x 0.25", str(dets[0]))

        self.assertEqual(dets[1].label, "desk")
        self.assertAlmostEqual(dets[1].image_x, 0.75)

        self.assertEqual(dets[2].label, "laptop")
        # box_2d [ymin, xmin, ymax, xmax] -> center x = (400 + 600)/2 / 1000 = 0.5
        self.assertAlmostEqual(dets[2].image_x, 0.5)

        self.assertEqual(dets[3].label, "cup")
        self.assertIsNone(dets[3].image_x)

    def test_project_detection_center_object(self):
        # Robot at origin facing theta = 0 (+x direction)
        odom = Odometry(x=0.0, y=0.0, theta=0.0)
        # Object at center of image (x = 0.5)
        det = Detection(label="box", confidence=0.9, image_x=0.5)
        proj = project_detection(det, odom, fov=math.radians(90), fixed_range=2.0)

        self.assertEqual(proj.label, "box")
        self.assertAlmostEqual(proj.bearing, 0.0, places=3)
        self.assertAlmostEqual(proj.range, 2.0, places=3)
        self.assertAlmostEqual(proj.x, 2.0, places=3)
        self.assertAlmostEqual(proj.y, 0.0, places=3)

    def test_project_detection_left_and_right(self):
        # Robot at (1.0, 1.0) facing theta = 0 (+x)
        odom = Odometry(x=1.0, y=1.0, theta=0.0)
        fov = math.radians(90)  # 90 degrees

        # Left object at x = 0.25: bearing = (0.5 - 0.25) * 90 deg = +22.5 deg (+pi/8)
        left_det = Detection(label="left_ob", image_x=0.25)
        proj_left = project_detection(left_det, odom, fov=fov, fixed_range=1.0)
        expected_bearing = math.radians(22.5)
        self.assertAlmostEqual(proj_left.bearing, expected_bearing, places=3)
        self.assertAlmostEqual(
            proj_left.x, 1.0 + 1.0 * math.cos(expected_bearing), places=3
        )
        self.assertAlmostEqual(
            proj_left.y, 1.0 + 1.0 * math.sin(expected_bearing), places=3
        )
        self.assertGreater(proj_left.y, 1.0)  # Left has +y

        # Right object at x = 0.75: bearing = (0.5 - 0.75) * 90 deg = -22.5 deg (-pi/8)
        right_det = Detection(label="right_ob", image_x=0.75)
        proj_right = project_detection(right_det, odom, fov=fov, fixed_range=1.0)
        self.assertAlmostEqual(proj_right.bearing, -expected_bearing, places=3)
        self.assertLess(proj_right.y, 1.0)  # Right has -y

    def test_project_detection_with_robot_heading(self):
        # Robot facing north (+y direction, theta = pi/2)
        odom = Odometry(x=2.0, y=3.0, theta=math.pi / 2)
        # Center object
        det = Detection(label="post", image_x=0.5)
        proj = project_detection(det, odom, fixed_range=1.5)
        self.assertAlmostEqual(proj.x, 2.0, places=3)
        self.assertAlmostEqual(proj.y, 4.5, places=3)

    def test_project_detections_batch(self):
        odom = Odometry(x=0.0, y=0.0, theta=0.0)
        dets = [
            Detection(label="a", image_x=0.5),
            Detection(label="b", image_x=0.5, range=3.0),
        ]
        out = project_detections(dets, odom, fixed_range=1.5)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].x, 1.5)
        self.assertAlmostEqual(out[1].x, 3.0)


class TestVlmPerceptorRateLimitAndHooks(unittest.TestCase):
    def test_estimate_vlm_cost(self):
        cost_sonnet = estimate_vlm_cost("claude-sonnet-5", 1000, 200)
        # 1000 * 2/1M + 200 * 10/1M = 0.002 + 0.002 = 0.004 (list-price estimate)
        self.assertAlmostEqual(cost_sonnet, 0.004, places=5)
        older = estimate_vlm_cost("claude-sonnet-4.5", 1000, 200)
        self.assertAlmostEqual(older, 0.006, places=5)

        cost_haiku = estimate_vlm_cost("claude-3-5-haiku", 1000, 200)
        # 1000 * 0.8/1M + 200 * 4/1M = 0.0008 + 0.0008 = 0.0016
        self.assertAlmostEqual(cost_haiku, 0.0016, places=5)

    def test_vlm_stats_hook_and_latency(self):
        stats_seen = []
        client = _FakeClient('[{"label": "dog", "confidence": 0.95, "x": 0.4}]')
        vlm = VlmPerceptor(
            client=client,
            rate_limit_hz=100.0,  # Fast for unit tests
            stats_hook=stats_seen.append,
        )
        dets = vlm.look(SYNTHETIC_JPEG)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "dog")
        self.assertEqual(len(stats_seen), 1)
        st: VlmCallStats = stats_seen[0]
        self.assertGreaterEqual(st.duration_s, 0.0)
        self.assertEqual(st.input_tokens, 120)
        self.assertEqual(st.output_tokens, 25)
        self.assertGreater(st.estimated_cost_usd, 0.0)
        self.assertEqual(vlm.last_stats, st)

    def test_vlm_rate_limiting_enforced(self):
        client = _FakeClient('[{"label": "cone"}]')
        # Rate limit at 4 Hz (0.25s interval)
        vlm = VlmPerceptor(client=client, rate_limit_hz=4.0)

        def frozen_clock() -> float:
            return 10.0

        with (
            mock.patch("ohho.perception.time.monotonic", frozen_clock),
            mock.patch("ohho.perception.time.sleep") as sleep,
        ):
            vlm.look(SYNTHETIC_JPEG)
            vlm.look(SYNTHETIC_JPEG)
        # Second call must wait for the rest of the 1/4 s interval
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args[0][0], 0.25, places=5)


class TestCameraPerceptorIntegration(unittest.TestCase):
    def test_camera_perceptor_perceive_and_remember(self):
        spec = get_spec("unitree-go2")

        def state_factory():
            return lambda: {
                "position": [2.0, 1.0, 0.0],
                "imu_rpy": [0.0, 0.0, 0.0],  # heading theta = 0
            }

        def frame_factory():
            return lambda: SYNTHETIC_JPEG

        tp = UnitreeDdsTransport(
            spec,
            state_factory=state_factory,
            frame_factory=frame_factory,
        )
        bot = Robot(spec, tp, NativeRuntime())
        got_state = threading.Event()
        tp.on_telemetry(lambda _t: got_state.set())
        tp.connect()
        try:
            if not tp.odom_received:
                self.assertTrue(got_state.wait(timeout=2.0))
            self.assertTrue(tp.odom_received)
            client = _FakeClient('[{"label": "hydrant", "confidence": 0.92, "x": 0.5}]')
            vlm = VlmPerceptor(client=client, rate_limit_hz=100.0)
            perceptor = CameraPerceptor(
                bot, vlm=vlm, fov=math.radians(90), fixed_range=2.0
            )

            dets = perceptor.look()
            self.assertEqual(len(dets), 1)
            d = dets[0]
            self.assertEqual(d.label, "hydrant")
            self.assertAlmostEqual(d.x, 4.0, places=3)  # 2.0 (robot x) + 2.0
            self.assertAlmostEqual(d.y, 1.0, places=3)  # 1.0 (robot y) + 0.0

            memory = SpatialMemory()
            n = remember_detections(memory, dets)
            self.assertEqual(n, 1)
            recalled = memory.where_is("hydrant")
            self.assertIsNotNone(recalled)
            self.assertAlmostEqual(recalled.x, 4.0, places=3)
            self.assertAlmostEqual(recalled.y, 1.0, places=3)
        finally:
            bot.disconnect()

    def test_camera_perceptor_uses_lidar_scan_range(self):
        """A forward lidar hit replaces the fixed-range assumption."""
        spec = get_spec("unitree-go2")

        def state_factory():
            return lambda: {
                "position": [2.0, 1.0, 0.0],
                "imu_rpy": [0.0, 0.0, 0.0],
                "pointcloud": [(2.0, 0.0, 0.2)],
            }

        tp = UnitreeDdsTransport(
            spec,
            "lo?domain=1",
            state_factory=state_factory,
            frame_factory=lambda: lambda: SYNTHETIC_JPEG,
        )
        bot = Robot(spec, tp, NativeRuntime())
        got_scan = threading.Event()
        tp.on_telemetry(lambda t: got_scan.set() if t.scan is not None else None)
        tp.connect()
        try:
            self.assertTrue(got_scan.wait(timeout=2.0))
            self.assertIsNotNone(bot.telemetry().scan)
            client = _FakeClient('[{"label": "hydrant", "confidence": 0.9, "x": 0.5}]')
            vlm = VlmPerceptor(client=client, rate_limit_hz=100.0)
            perceptor = CameraPerceptor(
                bot, vlm=vlm, fov=math.radians(90), fixed_range=5.0
            )
            dets = perceptor.look()
            self.assertEqual(len(dets), 1)
            self.assertAlmostEqual(dets[0].range, 2.0, places=2)
            self.assertAlmostEqual(dets[0].x, 4.0, places=2)
            self.assertAlmostEqual(dets[0].y, 1.0, places=2)
        finally:
            bot.disconnect()

    def test_camera_perceptor_returns_empty_when_no_frame(self):
        spec = get_spec("unitree-go2")
        tp = UnitreeDdsTransport(
            spec, state_factory=lambda: lambda: {"position": [0, 0, 0]}
        )
        bot = Robot(spec, tp, NativeRuntime())
        tp.connect()
        try:
            perceptor = CameraPerceptor(bot)
            self.assertEqual(perceptor.look(), [])
        finally:
            bot.disconnect()


class TestCliGo2LookCommand(unittest.TestCase):
    def test_cli_look_unitree_go2_mock(self):
        # Test: `ohho look unitree-go2 --transport dds://lo?domain=1 --mock`
        with (
            tempfile.TemporaryDirectory() as home,
            mock.patch.dict(
                os.environ,
                {"OHHO_STUB_DDS": "1", "HOME": home, "USERPROFILE": home},
            ),
        ):
            code = main(
                ["look", "unitree-go2", "--transport", "dds://lo?domain=1", "--mock"]
            )
            self.assertEqual(code, 0)

            # Verify items were remembered into unitree-go2 memory
            mem = SpatialMemory(path=default_memory_path("unitree-go2"))
            chair = mem.where_is("chair")
            table = mem.where_is("table")
            self.assertIsNotNone(chair)
            self.assertIsNotNone(table)
            self.assertGreater(chair.attrs["confidence"], 0.8)
            self.assertEqual(os.environ.get("OHHO_STUB_DDS"), "1")

    def test_cli_look_mock_scopes_dds_stub_to_connect(self):
        seen = {}

        def spy(_args):
            seen["stub"] = os.environ.get("OHHO_STUB_DDS")
            raise RuntimeError("stop after connect")

        previous = os.environ.pop("OHHO_STUB_DDS", None)
        try:
            with mock.patch("ohho.cli._connect", side_effect=spy):
                code = main(["look", "omnibot", "--transport", "sim://", "--mock"])
            self.assertEqual(code, 2)
            self.assertEqual(seen["stub"], "1")
            self.assertNotIn("OHHO_STUB_DDS", os.environ)

            os.environ["OHHO_STUB_DDS"] = "0"
            with mock.patch("ohho.cli._connect", side_effect=spy):
                code = main(["look", "omnibot", "--transport", "sim://"])
            self.assertEqual(code, 2)
            self.assertEqual(seen["stub"], "0")
            self.assertEqual(os.environ.get("OHHO_STUB_DDS"), "0")
        finally:
            if previous is None:
                os.environ.pop("OHHO_STUB_DDS", None)
            else:
                os.environ["OHHO_STUB_DDS"] = previous

    def test_cli_look_mock_refuses_unsafe_transport(self):
        refused = [
            ["look", "unitree-go2", "--transport", "dds://eth0", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://eth0?domain=1", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://lo", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://lo?domain=0", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://lo?domain=nope", "--mock"],
            ["look", "omnibot", "--transport", "serial:///dev/ttyUSB0", "--mock"],
            ["look", "omnibot", "--transport", "ros2://", "--mock"],
        ]
        previous = os.environ.pop("OHHO_STUB_DDS", None)
        try:
            for argv in refused:
                with mock.patch("ohho.cli.Robot.connect") as connect:
                    code = main(argv)
                self.assertEqual(code, 2, argv)
                connect.assert_not_called()
                self.assertNotIn("OHHO_STUB_DDS", os.environ)
        finally:
            if previous is None:
                os.environ.pop("OHHO_STUB_DDS", None)
            else:
                os.environ["OHHO_STUB_DDS"] = previous

        with (
            mock.patch.dict(os.environ, {"OHHO_MOCK_VLM": "1"}, clear=False),
            mock.patch("ohho.cli.Robot.connect") as connect,
        ):
            code = main(["look", "unitree-go2", "--transport", "dds://eth0"])
        self.assertEqual(code, 2)
        connect.assert_not_called()

    def test_cli_look_mock_allows_sim_and_loopback_stub(self):
        allowed = [
            ["look", "omnibot", "--mock"],
            ["look", "omnibot", "--transport", "sim://", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://", "--mock"],
            ["look", "unitree-go2", "--transport", "dds://lo?domain=1", "--mock"],
            [
                "look",
                "unitree-go2",
                "--transport",
                "dds://127.0.0.1?domain=2",
                "--mock",
            ],
        ]

        def stop(_args):
            raise RuntimeError("stop after connect")

        for argv in allowed:
            with mock.patch("ohho.cli._connect", side_effect=stop) as connect:
                code = main(argv)
            self.assertEqual(code, 2, argv)
            connect.assert_called_once()

        with (
            mock.patch.dict(os.environ, {"OHHO_MOCK_VLM": "1"}, clear=False),
            mock.patch("ohho.cli._connect", side_effect=stop) as connect,
        ):
            code = main(["look", "omnibot", "--transport", "sim://"])
            self.assertEqual(code, 2)
            connect.assert_called_once()

        with mock.patch("ohho.cli._connect", side_effect=stop) as connect:
            code = main(["look", "unitree-go2", "--transport", "dds://eth0"])
        self.assertEqual(code, 2)
        connect.assert_called_once()

    def test_cli_look_env_mock_stubs_loopback_dds(self):
        """OHHO_MOCK_VLM=1 on loopback DDS uses the stub only during connect."""
        seen = {}

        def spy(_args):
            seen["stub"] = os.environ.get("OHHO_STUB_DDS")
            raise RuntimeError("stop after connect")

        previous = os.environ.pop("OHHO_STUB_DDS", None)
        try:
            with (
                mock.patch.dict(os.environ, {"OHHO_MOCK_VLM": "1"}, clear=False),
                mock.patch("ohho.cli._connect", side_effect=spy),
            ):
                code = main(["look", "unitree-go2", "--transport", "dds://lo?domain=1"])
            self.assertEqual(code, 2)
            self.assertEqual(seen["stub"], "1")
            self.assertNotIn("OHHO_STUB_DDS", os.environ)

            seen.clear()
            with (
                mock.patch.dict(os.environ, {"OHHO_MOCK_VLM": "1"}, clear=False),
                mock.patch("ohho.cli._connect", side_effect=spy),
            ):
                code = main(["look", "omnibot", "--transport", "sim://"])
            self.assertEqual(code, 2)
            self.assertIsNone(seen["stub"])
            self.assertNotIn("OHHO_STUB_DDS", os.environ)
        finally:
            if previous is None:
                os.environ.pop("OHHO_STUB_DDS", None)
            else:
                os.environ["OHHO_STUB_DDS"] = previous

    def test_cli_look_unitree_go2_missing_sdk_clean_exit(self):
        # Without mock and without the SDK, exit code 2. Do not open DDS.
        with (
            mock.patch.dict(os.environ, {"OHHO_STUB_DDS": "0"}, clear=False),
            mock.patch.dict("sys.modules", _BLOCK_UNITREE_SDK),
        ):
            code = main(["look", "unitree-go2", "--transport", "dds://lo?domain=1"])
            self.assertEqual(code, 2)

    def test_camera_perceptor_skips_placeholder_and_unlocated(self):
        png = b"\x89PNG\r\n\x1a\n" + b"rest"
        seen = {}

        class _Vlm:
            def look(self, frame, media_type="image/jpeg"):
                seen["media_type"] = media_type
                return [
                    Detection(label="chair", confidence=0.9, image_x=0.5),
                    Detection(label="cup", confidence=0.4),
                ]

        class _Bot:
            spec = SimpleNamespace(id="unitree-go2")
            transport = SimpleNamespace(odom_received=False)

            def get_camera_frame(self):
                return png

            def telemetry(self):
                return Telemetry(odom=Odometry(x=9.0, y=9.0, theta=0.0))

        dets = CameraPerceptor(_Bot(), vlm=_Vlm(), fixed_range=2.0).look()
        self.assertEqual(seen["media_type"], "image/png")
        self.assertIsNone(dets[0].x)
        self.assertIsNone(dets[1].x)

        _Bot.transport = SimpleNamespace(odom_received=True)
        located = CameraPerceptor(_Bot(), vlm=_Vlm(), fixed_range=2.0).look()
        self.assertAlmostEqual(located[0].x, 11.0)
        self.assertIsNone(located[1].x)
        self.assertIsNone(located[1].y)


if __name__ == "__main__":
    unittest.main()
