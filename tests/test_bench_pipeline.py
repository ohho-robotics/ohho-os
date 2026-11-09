"""OHH-474 pipeline bench: camera frame to a published action.

The numeric checks use a virtual clock. Wall-clock pacing is not asserted.
No test opens a robot transport, and none set OHHO_ARM_HARDWARE.
"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from ohho.bench import PIPELINE_SCHEMA_ID, PIPELINE_STAGE_NAMES, validate
from ohho.bench.clock import VirtualClock
from ohho.bench.command import run_bench_command
from ohho.bench.errors import BenchHarnessError
from ohho.bench.pipeline import (
    BAG_SCHEMA_ID,
    STREAM_SCHEMA_ID,
    InProcessSink,
    PipelineConfig,
    Ros2PublishSink,
    TinyCpuPolicy,
    execute_pipeline,
)
from ohho.bench.report import render_markdown
from ohho.cli import build_parser, main
from ohho.robot import Robot


def _b64(width: int, height: int, pixel: tuple[int, int, int]) -> str:
    raw = bytes(pixel) * (width * height)
    return base64.b64encode(raw).decode("ascii")


def _write_fixture(path: Path, count: int, *, world: str = "sim") -> None:
    frames = []
    for index in range(count):
        frames.append(
            {
                "width": 2,
                "height": 2,
                "encoding": "rgb8",
                "timestamp": index / 10.0,
                "data": _b64(2, 2, ((index * 40) % 250, 10, 20)),
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": STREAM_SCHEMA_ID,
                "world": world,
                "fps": 10,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )


def _costs(**overrides: float) -> dict[str, float]:
    costs = {name: 0.001 for name in PIPELINE_STAGE_NAMES}
    costs.update(overrides)
    return costs


def _run_fixture(
    path: Path,
    *,
    hz: float = 10.0,
    warmup: int = 0,
    iters: int = 4,
    costs: dict[str, float] | None = None,
    **extra: object,
) -> dict:
    clock = VirtualClock()
    return execute_pipeline(
        PipelineConfig(
            source="fixture",
            fixture=str(path),
            hz=hz,
            warmup=warmup,
            iters=iters,
            sink="inprocess",
            robot="sim",
        ),
        clock=clock,
        costs=_costs() if costs is None else costs,
        **extra,
    )


class TestPipelineContract(unittest.TestCase):
    def test_scripted_fixture_meets_the_rate_and_the_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 4)
            document = _run_fixture(path)
        validate(document)
        self.assertEqual(document["schema"], "ohho.bench/v1")
        pipeline = document["pipeline"]
        self.assertEqual(pipeline["version"], PIPELINE_SCHEMA_ID)
        self.assertEqual(pipeline["world"], "sim")
        self.assertEqual(pipeline["source"], "fixture")
        self.assertEqual(pipeline["device"], "cpu")
        self.assertFalse(pipeline["live_robot"])
        self.assertEqual(pipeline["gpu_follow_up"], ["h2d", "d2h"])
        self.assertEqual(pipeline["dropped_frames"], 0)
        self.assertTrue(pipeline["rate_sustained"])
        self.assertEqual(pipeline["frames_presented"], 4)
        self.assertEqual(pipeline["frames_published"], 4)
        self.assertAlmostEqual(pipeline["target_hz"], 10.0)
        self.assertAlmostEqual(pipeline["achieved_hz"], 10.0, places=6)
        self.assertEqual(pipeline["jitter_ms"], document["latency"]["std"])
        self.assertAlmostEqual(document["latency"]["p50"], 7.0, places=6)
        self.assertAlmostEqual(document["latency"]["p95"], 7.0, places=6)
        self.assertAlmostEqual(document["latency"]["p99"], 7.0, places=6)
        self.assertAlmostEqual(document["latency"]["mean"], 7.0, places=6)
        names = [stage["name"] for stage in document["latency"]["stages"]]
        self.assertEqual(names, list(PIPELINE_STAGE_NAMES))
        stage_sum = sum(stage["mean"] for stage in document["latency"]["stages"])
        self.assertAlmostEqual(stage_sum, document["latency"]["mean"], places=6)
        self.assertEqual(document["environment"]["world"], "sim")
        self.assertEqual(document["environment"]["timing_source"], "virtual")
        self.assertIn("host copies", pipeline["gpu_note"])
        summary = render_markdown(document)
        self.assertIn("### Pipeline", summary)
        self.assertIn("| decode |", summary)
        self.assertIn("| publish |", summary)
        self.assertNotIn("<html", summary.lower())

    def test_overrun_drops_frames_and_clears_the_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 4)
            document = _run_fixture(
                path,
                costs=_costs(
                    decode=0.0,
                    preprocess=0.0,
                    h2d=0.0,
                    inference=0.25,
                    d2h=0.0,
                    postprocess=0.0,
                    publish=0.0,
                ),
            )
        pipeline = document["pipeline"]
        self.assertEqual(pipeline["dropped_frames"], 2)
        self.assertEqual(pipeline["frames_published"], 2)
        self.assertFalse(pipeline["rate_sustained"])
        self.assertGreater(document["latency"]["p99"], 0)

    def test_postprocess_clamps_to_the_sim_spec(self) -> None:
        sink = InProcessSink()

        class Huge:
            name = "huge"

            def infer(self, features: list[float]) -> list[float]:
                return [100.0, 100.0, 100.0]

            def digest(self) -> str:
                return "sha256:" + "ab" * 32

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 2)
            document = execute_pipeline(
                PipelineConfig(
                    source="fixture",
                    fixture=str(path),
                    hz=10,
                    warmup=0,
                    iters=2,
                    sink="inprocess",
                ),
                clock=VirtualClock(),
                costs=_costs(),
                policy=Huge(),
                sink=sink,
            )
        self.assertEqual(sink.actions, [(1.0, 1.0, 2.0), (1.0, 1.0, 2.0)])
        self.assertFalse(document["pipeline"]["live_robot"])

    def test_slo_gate_uses_pipeline_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 4)
            passed = _run_fixture(path, slos=("p95<=20ms",))
            failed = _run_fixture(path, slos=("p95<=1ms",))
        self.assertTrue(passed["gates"]["passed"])
        self.assertFalse(failed["gates"]["passed"])


class TestPipelineSources(unittest.TestCase):
    def test_ppm_fixture_decodes(self) -> None:
        rgb = bytes((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        payload = b"P6\n2 2\n255\n" + rgb
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": STREAM_SCHEMA_ID,
                        "world": "recorded",
                        "frames": [
                            {
                                "encoding": "ppm",
                                "width": 2,
                                "height": 2,
                                "data": base64.b64encode(payload).decode("ascii"),
                            },
                            {
                                "encoding": "ppm",
                                "width": 2,
                                "height": 2,
                                "data": base64.b64encode(payload).decode("ascii"),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            document = _run_fixture(path, iters=2)
        self.assertEqual(document["pipeline"]["world"], "recorded")
        self.assertEqual(document["pipeline"]["frame_origin"], "camera")
        self.assertEqual(document["pipeline"]["dropped_frames"], 0)

    def test_rosbag_fixture_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag = root / "bag"
            bag.mkdir()
            (bag / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema": BAG_SCHEMA_ID,
                        "topic": "/camera/image_raw",
                        "world": "sim",
                        "fps": 10,
                    }
                ),
                encoding="utf-8",
            )
            line = json.dumps(
                {
                    "width": 2,
                    "height": 2,
                    "encoding": "rgb8",
                    "data": _b64(2, 2, (8, 9, 10)),
                }
            )
            (bag / "frames.jsonl").write_text(
                line + "\n" + line + "\n", encoding="utf-8"
            )
            document = execute_pipeline(
                PipelineConfig(
                    source="rosbag",
                    bag=str(bag),
                    hz=10,
                    warmup=0,
                    iters=2,
                    sink="inprocess",
                ),
                clock=VirtualClock(),
                costs=_costs(),
            )
        self.assertEqual(document["pipeline"]["source"], "rosbag")
        self.assertEqual(document["pipeline"]["world"], "sim")
        self.assertEqual(document["pipeline"]["frames_published"], 2)

    def test_mcap_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mcap"
            path.write_bytes(b"not a bag")
            with self.assertRaises(BenchHarnessError) as caught:
                execute_pipeline(
                    PipelineConfig(source="rosbag", bag=str(path), warmup=0, iters=1)
                )
        self.assertIn("MCAP", str(caught.exception))

    def test_lerobot_episode_without_a_camera_is_a_state_raster(self) -> None:
        from ohho.sim_loop import record_sim_episodes

        with tempfile.TemporaryDirectory() as tmp:
            dataset = record_sim_episodes(
                robot_id="sim",
                num_episodes=1,
                steps_per_episode=4,
                fps=10.0,
                output_dir=str(Path(tmp) / "dataset"),
            )
            document = execute_pipeline(
                PipelineConfig(
                    source="lerobot",
                    dataset=dataset,
                    episode=0,
                    hz=10,
                    warmup=0,
                    iters=4,
                    sink="inprocess",
                ),
                clock=VirtualClock(),
                costs=_costs(),
            )
        self.assertEqual(document["pipeline"]["source"], "lerobot")
        self.assertEqual(document["pipeline"]["world"], "sim")
        self.assertEqual(document["pipeline"]["frame_origin"], "state-raster")
        self.assertEqual(document["pipeline"]["frames_published"], 4)
        self.assertFalse(document["pipeline"]["live_robot"])

    def test_inline_lerobot_image_is_a_camera_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            data = root / "data" / "chunk-000"
            data.mkdir(parents=True)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "repo_id": "local/demo",
                        "robot_type": "omnibot",
                        "fps": 10,
                        "total_episodes": 1,
                        "total_frames": 2,
                    }
                ),
                encoding="utf-8",
            )
            image = _b64(2, 2, (1, 2, 3))
            rows = []
            for index in range(2):
                rows.append(
                    json.dumps(
                        {
                            "observation.state": [0.1, 0.2, 0.0],
                            "action": [0.0, 0.0, 0.0],
                            "timestamp": index / 10,
                            "frame_index": index,
                            "episode_index": 0,
                            "task_index": 0,
                            "next.done": index == 1,
                            "observation.images.front": image,
                        }
                    )
                )
            (data / "episode_000000.jsonl").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
            document = execute_pipeline(
                PipelineConfig(
                    source="lerobot",
                    dataset=str(root),
                    hz=10,
                    warmup=0,
                    iters=2,
                    sink="inprocess",
                ),
                clock=VirtualClock(),
                costs=_costs(),
            )
        self.assertEqual(document["pipeline"]["frame_origin"], "camera")
        self.assertEqual(document["pipeline"]["world"], "recorded")

    def test_sim_loop_is_labelled_sim_and_publishes(self) -> None:
        sink = InProcessSink()
        document = execute_pipeline(
            PipelineConfig(
                source="sim",
                robot="sim",
                hz=10,
                warmup=0,
                iters=3,
                sink="inprocess",
            ),
            clock=VirtualClock(),
            costs=_costs(),
            sink=sink,
        )
        validate(document)
        self.assertEqual(document["pipeline"]["world"], "sim")
        self.assertEqual(document["pipeline"]["frame_origin"], "sim")
        self.assertEqual(document["pipeline"]["sink"], "inprocess")
        self.assertEqual(len(sink.actions), 3)
        self.assertTrue(
            all(abs(vx) <= 1.0 and abs(yaw) <= 2.0 for vx, _vy, yaw in sink.actions)
        )
        self.assertFalse(document["pipeline"]["live_robot"])

    def test_default_sim_sink_drives_the_simulator(self) -> None:
        document = execute_pipeline(
            PipelineConfig(source="sim", robot="sim", hz=10, warmup=0, iters=2),
            clock=VirtualClock(),
            costs=_costs(),
        )
        self.assertEqual(document["pipeline"]["sink"], "sim")
        self.assertGreater(document["pipeline"]["sim_x"], 0.0)
        self.assertEqual(document["run"]["manifest_ref"], "sim://sim")


class TestPipelineHardwareGate(unittest.TestCase):
    def test_hardware_is_refused_without_the_flag(self) -> None:
        with self.assertRaises(BenchHarnessError) as caught:
            execute_pipeline(
                PipelineConfig(source="hardware", warmup=0, iters=1),
            )
        self.assertIn("refused", str(caught.exception))
        self.assertIn("does not open a robot", str(caught.exception))

    def test_allow_hardware_still_does_not_connect(self) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("Robot.connect was called")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 2, world="recorded")
            with patch.object(Robot, "connect", boom):
                document = execute_pipeline(
                    PipelineConfig(
                        source="hardware",
                        fixture=str(path),
                        allow_hardware=True,
                        hz=10,
                        warmup=0,
                        iters=2,
                        sink="inprocess",
                    ),
                    clock=VirtualClock(),
                    costs=_costs(),
                )
        self.assertEqual(document["pipeline"]["world"], "hardware")
        self.assertEqual(document["pipeline"]["source"], "hardware")
        self.assertEqual(document["pipeline"]["recording"], "fixture")
        self.assertFalse(document["pipeline"]["live_robot"])

    def test_allow_hardware_without_a_recording_refuses_to_open_a_robot(self) -> None:
        with self.assertRaises(BenchHarnessError) as caught:
            execute_pipeline(
                PipelineConfig(
                    source="hardware",
                    allow_hardware=True,
                    warmup=0,
                    iters=1,
                )
            )
        self.assertIn("refusing to open a robot", str(caught.exception))

    def test_device_uri_is_refused(self) -> None:
        with self.assertRaises(BenchHarnessError) as caught:
            PipelineConfig(fixture="dds://eth0")
        self.assertIn("does not open a robot", str(caught.exception))


class TestPipelineSinks(unittest.TestCase):
    def test_ros2_factory_publishes_the_clamped_action(self) -> None:
        seen: list[tuple[float, float, float]] = []

        def factory(topic: str):
            self.assertEqual(topic, "/cmd_vel")

            def send(vx: float, vy: float, yaw: float) -> None:
                seen.append((vx, vy, yaw))

            return send, lambda: None

        sink = Ros2PublishSink(factory=factory)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.json"
            _write_fixture(path, 1)
            execute_pipeline(
                PipelineConfig(
                    source="fixture",
                    fixture=str(path),
                    hz=10,
                    warmup=0,
                    iters=1,
                ),
                clock=VirtualClock(),
                costs=_costs(),
                sink=sink,
            )
        self.assertEqual(len(seen), 1)
        vx, vy, yaw = seen[0]
        self.assertLessEqual(abs(vx), 1.0)
        self.assertLessEqual(abs(vy), 1.0)
        self.assertLessEqual(abs(yaw), 2.0)

    def test_ros2_without_rclpy_is_an_error(self) -> None:
        try:
            import rclpy  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("rclpy is installed")
        with self.assertRaises(BenchHarnessError) as caught:
            Ros2PublishSink()
        self.assertIn("rclpy", str(caught.exception))

    def test_tiny_policy_is_deterministic(self) -> None:
        policy = TinyCpuPolicy()
        features = [0.2] * 8
        self.assertEqual(policy.infer(features), policy.infer(features))
        self.assertTrue(policy.digest().startswith("sha256:"))


class TestPipelineCli(unittest.TestCase):
    def test_bare_bench_is_still_rejected(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["bench"])
        self.assertEqual(caught.exception.code, 2)

    def test_pipeline_cli_writes_a_valid_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "stream.json"
            out = root / "out.json"
            _write_fixture(fixture, 4)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "bench",
                    "--pipeline",
                    "--fixture",
                    str(fixture),
                    "--warmup",
                    "0",
                    "--iters",
                    "4",
                    "--hz",
                    "10",
                    "--sink",
                    "inprocess",
                    "--json",
                    str(out),
                    "--slo",
                    "p95<=20ms",
                ]
            )
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = run_bench_command(args, clock=VirtualClock())
            self.assertEqual(code, 0, stderr.getvalue())
            document = json.loads(out.read_text(encoding="utf-8"))
        validate(document)
        self.assertTrue(document["gates"]["passed"])
        self.assertEqual(document["pipeline"]["dropped_frames"], 0)
        self.assertIn("wrote", stdout.getvalue())

    def test_hardware_cli_is_refused(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["bench", "--pipeline", "--source", "hardware"])
        self.assertEqual(code, 2)
        self.assertIn("refused", stderr.getvalue())

    def test_pipeline_flags_do_not_apply_to_the_artifact_bench(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"x")
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "--artifact",
                        str(artifact),
                        "--target",
                        "x86-cpu",
                        "--hz",
                        "30",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("requires --pipeline", stderr.getvalue())

    def test_artifact_and_pipeline_together_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"x")
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "--pipeline",
                        "--artifact",
                        str(artifact),
                        "--warmup",
                        "0",
                        "--iters",
                        "1",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("Omit --artifact", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
