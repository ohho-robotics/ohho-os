"""OHH-431 bench harness: stats, stages, gates, and ohho.bench/v1 output.

ONNX Runtime is optional. When it is missing, the schema test uses the
pure-Python runner. The 5% stage-sum check always uses that runner on the
CPU clock. CUDA events are exercised with a fake event clock; a machine
without a CUDA device does not run them on a GPU.
"""

from __future__ import annotations

import copy
import json
import statistics
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout

from ohho.bench import BenchConfig, execute, validate
from ohho.bench.clock import (
    CudaEventClock,
    PerfClock,
    VirtualClock,
    clock_for_providers,
)
from ohho.bench.command import run_bench_command
from ohho.bench.compare import compare_results
from ohho.bench.errors import BenchHarnessError
from ohho.bench.gates import evaluate_gates
from ohho.bench.report import (
    log_wandb,
    push_prometheus,
    render_markdown,
    render_prometheus,
)
from ohho.bench.schema import validate_path
from ohho.bench.stats import summarize
from ohho.cli import build_parser, main

_ROOT = Path(__file__).resolve().parents[1]


def _onnx_ready() -> bool:
    try:
        import numpy  # noqa: F401
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _write_tiny_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    outputs = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "tiny", [inputs], [outputs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _burn(count: int) -> int:
    total = 0
    for i in range(count):
        total += i
    return total


class BusyRunner:
    """CPU runner whose stages are real Python work, timed by perf_counter."""

    def preprocess(self, batch: int) -> int:
        _burn(400 + batch)
        return batch

    def h2d(self, prepared: int) -> int:
        _burn(80)
        return prepared

    def inference(self, device_input: int) -> int:
        _burn(1500)
        return device_input

    def d2h(self, device_output: int) -> int:
        _burn(80)
        return device_output

    def postprocess(self, host_output: int) -> int:
        _burn(200)
        return host_output


class ScaledRunner:
    """Advances a virtual clock by a fixed stage budget times ``scale``."""

    def __init__(self, clock: VirtualClock, scale: float = 1.0) -> None:
        self.clock = clock
        self.scale = scale

    def preprocess(self, batch: int) -> int:
        self.clock.advance(0.001 * self.scale)
        return batch

    def h2d(self, prepared: int) -> int:
        self.clock.advance(0.0001 * self.scale)
        return prepared

    def inference(self, device_input: int) -> int:
        self.clock.advance(0.01 * self.scale)
        return device_input

    def d2h(self, device_output: int) -> int:
        self.clock.advance(0.0001 * self.scale)
        return device_output

    def postprocess(self, host_output: int) -> int:
        self.clock.advance(0.001 * self.scale)
        return host_output


def _stage_gap(document: dict) -> float:
    latency = document["latency"]
    total = sum(stage["mean"] for stage in latency["stages"])
    end_to_end = latency["mean"]
    if end_to_end == 0:
        return 0.0
    return abs(total - end_to_end) / end_to_end


def _latency(p95_ms: float, *, unit: str = "ms") -> dict:
    value = p95_ms if unit == "ms" else p95_ms / 1000.0
    hz = 0.0 if p95_ms == 0 else 1000.0 / p95_ms
    fields = {
        "mean": value,
        "median": value,
        "p50": value,
        "p90": value,
        "p95": value,
        "p99": value,
        "min": value,
        "max": value,
        "std": 0.0,
        "unit": unit,
        "cold_start_ms": p95_ms,
        "achievable_hz": hz,
    }
    return {
        "latency": fields,
        "throughput": {"value": hz, "unit": "Hz"},
        "telemetry": {"memory_mb_max": 10.0},
    }


class TestStats(unittest.TestCase):
    def test_percentiles_min_median_and_population_std(self) -> None:
        stats = summarize([1, 2, 3, 4, 5])
        self.assertEqual(stats["min"], 1)
        self.assertEqual(stats["max"], 5)
        self.assertEqual(stats["median"], 3)
        self.assertEqual(stats["p50"], stats["median"])
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["p95"], 4.8)
        self.assertAlmostEqual(stats["std"], statistics.pstdev([1, 2, 3, 4, 5]))

    def test_single_sample_std_is_zero(self) -> None:
        stats = summarize([4.0])
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["min"], stats["max"])


class TestCompareAndGates(unittest.TestCase):
    def test_twenty_percent_p95_is_a_regression_and_ten_percent_is_not(self) -> None:
        comparison = compare_results(
            _latency(120), _latency(100), baseline_ref="baseline.json"
        )
        row = next(
            item for item in comparison["metrics"] if item["name"] == "latency.p95"
        )
        self.assertEqual(row["delta"], 20)
        self.assertEqual(row["pct_change"], 20)
        self.assertTrue(row["regression"])
        document = {"latency": _latency(120)["latency"], "comparison": comparison}
        over = evaluate_gates(document, ["p95<=200ms"], ["p95=10%"])
        self.assertFalse(over["passed"])
        self.assertTrue(over["checks"][0]["passed"])
        self.assertFalse(over["checks"][1]["passed"])
        exact = evaluate_gates(document, [], ["p95=20%"])
        self.assertTrue(exact["passed"])
        under = evaluate_gates(document, ["p95<=100ms"], [])
        self.assertFalse(under["checks"][0]["passed"])

    def test_seconds_baseline_is_compared_in_milliseconds(self) -> None:
        comparison = compare_results(
            _latency(120), _latency(100, unit="s"), baseline_ref="baseline.json"
        )
        row = next(
            item for item in comparison["metrics"] if item["name"] == "latency.p95"
        )
        self.assertEqual(row["unit"], "ms")
        self.assertEqual(row["baseline"], 100)
        self.assertEqual(row["current"], 120)
        self.assertEqual(row["pct_change"], 20)

    def test_bad_slo_and_regression_without_baseline(self) -> None:
        with self.assertRaises(BenchHarnessError) as bad:
            evaluate_gates({"latency": {"p95": 1, "unit": "ms"}}, ["faster"], [])
        self.assertIn("p95<=200ms", str(bad.exception))
        with self.assertRaises(BenchHarnessError) as missing:
            evaluate_gates({"latency": {"p95": 1, "unit": "ms"}}, [], ["p95=10%"])
        self.assertIn("--baseline", str(missing.exception))


class TestHarness(unittest.TestCase):
    def _scripted(self, tmp: Path, *, batches: tuple[int, ...] = (1,)) -> dict:
        artifact = tmp / "model.onnx"
        artifact.write_bytes(b"scripted")
        clock = VirtualClock()
        return execute(
            BenchConfig(
                artifact=str(artifact),
                target="x86-cpu",
                warmup=1,
                iters=4,
                batches=batches,
            ),
            runner=ScaledRunner(clock),
            clock=clock,
        )

    def test_fake_runner_stages_sum_within_five_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"busy")
            document = execute(
                BenchConfig(
                    artifact=str(artifact),
                    target="x86-cpu",
                    warmup=1,
                    iters=5,
                    batches=(1,),
                ),
                runner=BusyRunner(),
                clock=PerfClock(),
            )
        self.assertLessEqual(_stage_gap(document), 0.05)
        names = [stage["name"] for stage in document["latency"]["stages"]]
        self.assertEqual(
            names, ["preprocess", "h2d", "inference", "d2h", "postprocess"]
        )
        self.assertEqual(document["environment"]["timing_source"], "perf_counter")

    def test_scripted_stats_include_min_median_and_null_or_numeric_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = self._scripted(Path(tmp), batches=(4, 1))
        validate(document)
        latency = document["latency"]
        for key in ("mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std"):
            self.assertIn(key, latency)
            self.assertGreaterEqual(latency[key], 0)
        self.assertLessEqual(latency["min"], latency["median"])
        self.assertLessEqual(latency["median"], latency["max"])
        self.assertAlmostEqual(latency["p95"], 12.2, places=4)
        self.assertGreater(latency["cold_start_ms"], 0)
        self.assertGreater(latency["achievable_hz"], 0)
        self.assertEqual(document["throughput"]["unit"], "Hz")
        self.assertEqual(document["throughput"]["batch_size"], 1)
        self.assertEqual([item["batch_size"] for item in document["batches"]], [4, 1])
        self.assertTrue(document["batches"][1]["primary"])
        self.assertLessEqual(_stage_gap(document), 0.05)
        telemetry = document["telemetry"]
        self.assertIn("gpu_util_percent", telemetry)
        self.assertIn("gpu_power_w", telemetry)
        for key in ("gpu_util_percent", "gpu_power_w"):
            value = telemetry[key]
            self.assertTrue(value is None or isinstance(value, (int, float)))
        if "memory_mb_max" in telemetry:
            self.assertGreaterEqual(telemetry["memory_mb_max"], 0)
        else:
            self.assertIsNone(telemetry["peak_memory_mb"])
        self.assertEqual(document["environment"]["target_profile"], "x86-cpu")
        self.assertGreaterEqual(len(document["environment"]["git_sha"]), 7)
        self.assertEqual(document["run"]["target"], "inference")

    def test_onnx_or_fake_runner_produces_schema_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if _onnx_ready():
                artifact = root / "tiny.onnx"
                _write_tiny_onnx(artifact)
                document = execute(
                    BenchConfig(
                        artifact=str(artifact),
                        target="x86-cpu",
                        warmup=1,
                        iters=3,
                        batches=(1,),
                    )
                )
                self.assertEqual(
                    document["environment"]["execution_provider"],
                    "CPUExecutionProvider",
                )
            else:
                artifact = root / "model.onnx"
                artifact.write_bytes(b"fake")
                document = execute(
                    BenchConfig(
                        artifact=str(artifact),
                        target="x86-cpu",
                        warmup=0,
                        iters=3,
                        batches=(1,),
                    ),
                    runner=BusyRunner(),
                    clock=PerfClock(),
                )
            path = root / "out.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            validate(document)
            validate_path(path)
        self.assertIn("min", document["latency"])
        self.assertIn("median", document["latency"])
        self.assertLessEqual(_stage_gap(document), 0.05)


class TestCli(unittest.TestCase):
    def _parse(self, argv: list[str]):
        return build_parser().parse_args(argv)

    def test_dataset_inputs_and_registry_ids_are_not_yet_supported(self) -> None:
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
                        "--inputs",
                        "coco",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("not yet supported", stderr.getvalue())
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "bench",
                    "--artifact",
                    "my-model",
                    "--target",
                    "x86-cpu",
                    "--inputs",
                    "random",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("not yet supported", stderr.getvalue())
        self.assertIn("registry id", stderr.getvalue())

    def test_slo_pass_and_fail_and_twenty_percent_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"scripted")
            out = root / "out.json"
            baseline = root / "baseline.json"
            markdown = root / "summary.md"

            def run(extra: list[str]) -> int:
                clock = VirtualClock()
                args = self._parse(
                    [
                        "bench",
                        "--artifact",
                        str(artifact),
                        "--target",
                        "x86-cpu",
                        "--warmup",
                        "0",
                        "--iters",
                        "3",
                        "--batch",
                        "1,4",
                        "--inputs",
                        "random",
                        "--json",
                        str(out),
                        "--markdown",
                        str(markdown),
                        *extra,
                    ]
                )
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    return args.func(args, runner=ScaledRunner(clock), clock=clock)

            self.assertEqual(run(["--slo", "p95<=20ms"]), 0)
            document = json.loads(out.read_text(encoding="utf-8"))
            validate(document)
            self.assertGreater(document["latency"]["p95"], 10)
            self.assertLess(document["latency"]["p95"], 20)
            self.assertTrue(document["gates"]["passed"])
            summary = markdown.read_text(encoding="utf-8")
            self.assertTrue(summary.startswith("## Ohho bench\n"))
            for token in (
                "| mean |",
                "| median |",
                "| p95 |",
                "| min |",
                "| max |",
                "| preprocess |",
                "| h2d |",
                "| inference |",
                "| d2h |",
                "| postprocess |",
                "### Gates",
                "### Stages",
            ):
                self.assertIn(token, summary)
            self.assertNotIn("<html", summary.lower())

            self.assertNotEqual(run(["--slo", "p95<=10ms"]), 0)
            failed = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(failed["gates"]["passed"])

            current = json.loads(out.read_text(encoding="utf-8"))
            slowed = copy.deepcopy(current)
            slowed["latency"]["p95"] = current["latency"]["p95"] / 1.2
            baseline.write_text(json.dumps(slowed), encoding="utf-8")
            code = run(["--baseline", str(baseline), "--max-regression", "p95=10%"])
            self.assertNotEqual(code, 0)
            compared = json.loads(out.read_text(encoding="utf-8"))
            row = next(
                item
                for item in compared["comparison"]["metrics"]
                if item["name"] == "latency.p95"
            )
            self.assertAlmostEqual(row["pct_change"], 20.0, places=4)
            self.assertTrue(row["regression"])
            self.assertFalse(compared["gates"]["passed"])
            self.assertEqual(
                run(["--baseline", str(baseline), "--max-regression", "p95=25%"]),
                0,
            )

    def test_missing_baseline_flag_and_unknown_target(self) -> None:
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
                        "--max-regression",
                        "p95=10%",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("--baseline", stderr.getvalue())
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "bench",
                    "--artifact",
                    str(artifact),
                    "--target",
                    "not-a-profile",
                    "--iters",
                    "1",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("Unknown target profile", stderr.getvalue())

    def test_parity_is_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"scripted")
            parity = root / "parity.json"
            parity.write_text(json.dumps({"match": True, "score": 1}), encoding="utf-8")
            out = root / "out.json"
            clock = VirtualClock()
            args = self._parse(
                [
                    "bench",
                    "--artifact",
                    str(artifact),
                    "--target",
                    "x86-cpu",
                    "--warmup",
                    "0",
                    "--iters",
                    "2",
                    "--json",
                    str(out),
                    "--parity",
                    str(parity),
                ]
            )
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = run_bench_command(args, runner=ScaledRunner(clock), clock=clock)
            self.assertEqual(code, 0)
            document = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(document["parity"], {"match": True, "score": 1})
            validate(document)


class TestReporters(unittest.TestCase):
    def test_prometheus_text_and_push_use_the_opener(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = TestHarness()._scripted(Path(tmp))
        text = render_prometheus(document)
        self.assertIn("ohho_bench_latency_ms", text)
        self.assertIn('stat="p95"', text)
        self.assertIn("ohho_bench_achievable_hz", text)
        seen: list[object] = []

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                return b""

        def opener(request: object, timeout: float = 0) -> _Response:
            seen.append(request)
            return _Response()

        push_prometheus("http://127.0.0.1:9/metrics/job/ohho", document, opener=opener)
        self.assertEqual(len(seen), 1)

    def test_wandb_records_latency_and_a_missing_install_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = TestHarness()._scripted(Path(tmp))

        class FakeWandb:
            def __init__(self) -> None:
                self.payload: dict = {}
                self.finished = False

            def init(self, **kwargs: object) -> FakeWandb:
                return self

            def log(self, payload: dict) -> None:
                self.payload.update(payload)

            def finish(self) -> None:
                self.finished = True

        fake = FakeWandb()
        log_wandb(document, module=fake)
        self.assertIn("latency/p95", fake.payload)
        self.assertIn("latency/min", fake.payload)
        self.assertIn("latency/median", fake.payload)
        self.assertTrue(fake.finished)

        def boom() -> object:
            raise ImportError("no wandb")

        with self.assertRaises(BenchHarnessError) as caught:
            log_wandb(document, loader=boom)
        self.assertIn("wandb", str(caught.exception))

    def test_markdown_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = TestHarness()._scripted(Path(tmp))
        text = render_markdown(document)
        self.assertTrue(text.startswith("## Ohho bench\n"))
        self.assertIn("| median |", text)
        self.assertIn("| min |", text)
        self.assertIn("GPU util", text)


class TestCudaClock(unittest.TestCase):
    def test_cpu_provider_uses_perf_counter(self) -> None:
        clock = clock_for_providers(("CPUExecutionProvider",))
        self.assertIsInstance(clock, PerfClock)

    def test_cuda_events_record_device_stages(self) -> None:
        class _Event:
            def __init__(self, cuda: _Cuda) -> None:
                self.cuda = cuda
                self.t = 0.0

            def record(self) -> None:
                self.t = self.cuda.t

            def synchronize(self) -> None:
                return None

            def elapsed_time(self, other: _Event) -> float:
                return (other.t - self.t) * 1000.0

        class _Cuda:
            def __init__(self) -> None:
                self.t = 0.0

            def Event(self, enable_timing: bool = True) -> _Event:
                return _Event(self)

        class _GpuRunner:
            def __init__(self, cuda: _Cuda) -> None:
                self.cuda = cuda

            def preprocess(self, batch: int) -> int:
                return batch

            def h2d(self, prepared: int) -> int:
                self.cuda.t += 0.002
                return prepared

            def inference(self, device_input: int) -> int:
                self.cuda.t += 0.005
                return device_input

            def d2h(self, device_output: int) -> int:
                self.cuda.t += 0.001
                return device_output

            def postprocess(self, host_output: int) -> int:
                return host_output

        cuda = _Cuda()
        sample = CudaEventClock(cuda).measure(_GpuRunner(cuda), 1)
        self.assertAlmostEqual(sample["h2d"], 0.002)
        self.assertAlmostEqual(sample["inference"], 0.005)
        self.assertAlmostEqual(sample["d2h"], 0.001)
        self.assertEqual(CudaEventClock.kind, "cuda_event")

    def test_cuda_provider_without_a_device_stays_on_perf_counter(self) -> None:
        try:
            import torch
        except ImportError:
            clock = clock_for_providers(("CUDAExecutionProvider",))
            self.assertIsInstance(clock, PerfClock)
            return
        clock = clock_for_providers(("CUDAExecutionProvider",))
        if torch.cuda.is_available():
            self.assertIsInstance(clock, CudaEventClock)
        else:
            self.assertIsInstance(clock, PerfClock)


class TestSchemaStillValidatesExamples(unittest.TestCase):
    def test_checked_in_examples_still_validate(self) -> None:
        for name in ("vision", "vla", "llm"):
            validate_path(_ROOT / "ohho" / "bench" / "examples" / f"{name}.json")


if __name__ == "__main__":
    unittest.main()
