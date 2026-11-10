"""OHH-472 sensor sampler: recorded lines, a mocked device, sustained drift.

The suite does not open a management library, a board monitor, or a firmware
command. Sources are fed recorded text or stand-ins. A run with every source
unavailable still validates.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from ohho.bench import BenchConfig, execute, validate
from ohho.bench.clock import VirtualClock
from ohho.bench.command import run_bench_command
from ohho.bench.environment import GpuProbe
from ohho.bench.sensors.base import (
    DOMAIN_BOARD,
    DOMAIN_DEVICE,
    SensorSample,
    SensorSource,
    SourceStatus,
)
from ohho.bench.sensors.nvml import NvmlSource
from ohho.bench.sensors.rapl import RaplSource
from ohho.bench.sensors.sampler import SAMPLE_HZ, SensorSampler, WindowSummary
from ohho.bench.sensors.tegrastats import (
    TegraStatsSource,
    parse_tegrastats_line,
    sample_from_jtop,
)
from ohho.bench.sensors.vcgencmd import VcgencmdSource, parse_vcgencmd
from ohho.bench.sustained import drift_report
from ohho.cli import build_parser, main

_RECORDED = (
    Path(__file__).resolve().parent / "fixtures" / "bench" / "tegrastats_throttled.txt"
)
_STEADY = (
    "RAM 1543/3964MB CPU [23%@102,12%@204] EMC_FREQ 0% GR3D_FREQ 0%@114 "
    "GPU@43.5C POM_5V_IN 2500/2400 POM_5V_GPU 567/500"
)


class _CountingSource(SensorSource):
    source_id = "counter"

    def __init__(self) -> None:
        self.threads: list[str] = []

    def probe(self) -> SourceStatus:
        return SourceStatus(
            self.source_id,
            True,
            power_domain=DOMAIN_DEVICE,
            observes_throttle=True,
        )

    def read(self) -> SensorSample:
        self.threads.append(threading.current_thread().name)
        return SensorSample(self.source_id, power_w=1.0)


class _ConstantDevice:
    def __init__(self) -> None:
        self.threads: list[str] = []

    def power_w(self) -> float:
        self.threads.append(threading.current_thread().name)
        return 20.0

    def memory_used_mb(self) -> float:
        return 512.0

    def temperature_c(self) -> float:
        return 61.0

    def clocks_mhz(self) -> dict[str, float]:
        return {"graphics": 1500.0, "memory": 7000.0}

    def throttled(self) -> tuple[bool, None]:
        return False, None

    def close(self) -> None:
        return None


class _ThrottleSource(SensorSource):
    source_id = "mock"

    def probe(self) -> SourceStatus:
        return SourceStatus(
            self.source_id,
            True,
            power_domain=DOMAIN_DEVICE,
            observes_throttle=True,
        )

    def read(self) -> SensorSample:
        return SensorSample(
            self.source_id,
            power_w=10.0,
            throttled=True,
            throttle_detail="clocks reduced",
        )


class _DriftRunner:
    """Fast for the first 80 measured iterations, then four times slower."""

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.stages = 0

    def _advance(self, base: float) -> None:
        self.stages += 1
        measured = self.stages - 5
        iteration = 0 if measured <= 0 else (measured - 1) // 5
        scale = 1.0 if iteration < 80 else 4.0
        self.clock.advance(base * scale)

    def preprocess(self, batch: int) -> int:
        self._advance(0.001)
        return batch

    def h2d(self, prepared: int) -> int:
        self._advance(0.0001)
        return prepared

    def inference(self, device_input: int) -> int:
        self._advance(0.01)
        return device_input

    def d2h(self, device_output: int) -> int:
        self._advance(0.0001)
        return device_output

    def postprocess(self, host_output: int) -> int:
        self._advance(0.001)
        return host_output


class _Scaled:
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock

    def preprocess(self, batch: int) -> int:
        self.clock.advance(0.001)
        return batch

    def h2d(self, prepared: int) -> int:
        self.clock.advance(0.0001)
        return prepared

    def inference(self, device_input: int) -> int:
        self.clock.advance(0.01)
        return device_input

    def d2h(self, device_output: int) -> int:
        self.clock.advance(0.0001)
        return device_output

    def postprocess(self, host_output: int) -> int:
        self.clock.advance(0.001)
        return host_output


def _unavailable() -> list[SensorSource]:
    def _missing() -> object:
        raise OSError("NVML library not found")

    def _no_jtop() -> object:
        raise ImportError("jtop is not installed")

    return [
        NvmlSource(loader=_missing),
        TegraStatsSource(which=lambda _name: None, jtop_loader=_no_jtop),
        RaplSource(root=Path("no-such-powercap")),
        VcgencmdSource(which=lambda _name: None),
    ]


class TestTegrastatsParse(unittest.TestCase):
    def test_recorded_line_yields_power_and_throttling(self) -> None:
        text = _RECORDED.read_text(encoding="utf-8")
        sample = parse_tegrastats_line(text)
        self.assertAlmostEqual(sample.power_w, 8.456)
        self.assertTrue(sample.throttled)
        self.assertIn("throttled", sample.throttle_detail or "")
        self.assertEqual(sample.host_memory_mb, 2145)
        self.assertAlmostEqual(sample.temperature_c, 87.0)
        self.assertEqual(sample.clocks_mhz["gr3d"], 306)
        self.assertEqual(sample.clocks_mhz["emc"], 2133)
        self.assertEqual(sample.clocks_mhz["cpu0"], 729)
        self.assertEqual(sample.clocks_mhz["cpu1"], 1510)

    def test_steady_line_is_not_throttled_and_other_rails_are_not_summed(self) -> None:
        sample = parse_tegrastats_line(_STEADY)
        self.assertAlmostEqual(sample.power_w, 2.5)
        self.assertFalse(sample.throttled)
        self.assertEqual(sample.host_memory_mb, 1543)
        self.assertEqual(sample.clocks_mhz["gr3d"], 114)
        self.assertAlmostEqual(sample.temperature_c, 43.5)
        partial = parse_tegrastats_line("VDD_SOC 1500mW/1500mW")
        self.assertIsNone(partial.power_w)

    def test_clock_cap_below_max_is_throttling(self) -> None:
        sample = parse_tegrastats_line("VDD_IN 1000mW/1000mW GPU 300MHz/900MHz")
        self.assertAlmostEqual(sample.power_w, 1.0)
        self.assertTrue(sample.throttled)
        self.assertIn("300", sample.throttle_detail or "")
        self.assertEqual(sample.clocks_mhz["gpu"], 300)


class TestOtherSources(unittest.TestCase):
    def test_vcgencmd_bits_clocks_and_a_missing_binary(self) -> None:
        active = parse_vcgencmd(
            {
                "measure_temp": "temp=62.4'C",
                "get_throttled": "throttled=0x4",
                "arm": "frequency(48)=600000000",
                "core": "frequency(1)=250000000",
            }
        )
        self.assertIsNone(active.power_w)
        self.assertAlmostEqual(active.temperature_c, 62.4)
        self.assertTrue(active.throttled)
        self.assertAlmostEqual(active.clocks_mhz["arm"], 600.0)
        self.assertAlmostEqual(active.clocks_mhz["core"], 250.0)
        idle = parse_vcgencmd({"get_throttled": "throttled=0x0"})
        self.assertFalse(idle.throttled)
        history = parse_vcgencmd({"get_throttled": "throttled=0x50000"})
        self.assertFalse(history.throttled)
        missing = VcgencmdSource(which=lambda _name: None).probe()
        self.assertFalse(missing.available)
        self.assertIn("vcgencmd", missing.reason or "")

    def test_rapl_delta_and_wrap_ignore_a_core_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zone = root / "zone0"
            zone.mkdir()
            (zone / "name").write_text("package-0\n", encoding="utf-8")
            (zone / "energy_uj").write_text("1000000\n", encoding="utf-8")
            (zone / "max_energy_range_uj").write_text("100000000\n", encoding="utf-8")
            core = root / "zone1"
            core.mkdir()
            (core / "name").write_text("core\n", encoding="utf-8")
            (core / "energy_uj").write_text("999999999\n", encoding="utf-8")
            clock = {"t": 0.0}
            source = RaplSource(root=root, clock=lambda: clock["t"])
            status = source.probe()
            self.assertTrue(status.available)
            self.assertEqual(status.power_domain, "package")
            self.assertIsNone(source.read().power_w)
            (zone / "energy_uj").write_text("3000000\n", encoding="utf-8")
            clock["t"] = 1.0
            self.assertAlmostEqual(source.read().power_w, 2.0)
            (zone / "energy_uj").write_text("1000000\n", encoding="utf-8")
            clock["t"] = 2.0
            # Counter wrapped: (100_000_000 - 3_000_000) + 1_000_000 = 98 J in 1 s.
            self.assertAlmostEqual(source.read().power_w, 98.0)
        absent = RaplSource(root=Path("no-such-powercap")).probe()
        self.assertFalse(absent.available)
        self.assertIn("RAPL", absent.reason or "")

    def test_rapl_deduplicates_mmio_package_zones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            msr0 = root / "intel-rapl-0"
            msr0.mkdir()
            (msr0 / "name").write_text("package-0\n", encoding="utf-8")
            (msr0 / "energy_uj").write_text("1000000\n", encoding="utf-8")

            mmio0 = root / "intel-rapl-mmio-0"
            mmio0.mkdir()
            (mmio0 / "name").write_text("package-0\n", encoding="utf-8")
            (mmio0 / "energy_uj").write_text("1000000\n", encoding="utf-8")

            msr1 = root / "intel-rapl-1"
            msr1.mkdir()
            (msr1 / "name").write_text("package-1\n", encoding="utf-8")
            (msr1 / "energy_uj").write_text("2000000\n", encoding="utf-8")

            mmio1 = root / "intel-rapl-mmio-1"
            mmio1.mkdir()
            (mmio1 / "name").write_text("package-1\n", encoding="utf-8")
            (mmio1 / "energy_uj").write_text("2000000\n", encoding="utf-8")

            clock = {"t": 0.0}
            source = RaplSource(root=root, clock=lambda: clock["t"])
            status = source.probe()
            self.assertTrue(status.available)
            # Exactly two package zones, preferring msr over mmio
            self.assertEqual(source._zones, [msr0, msr1])

            self.assertIsNone(source.read().power_w)
            (msr0 / "energy_uj").write_text("3000000\n", encoding="utf-8")
            (msr1 / "energy_uj").write_text("3000000\n", encoding="utf-8")
            clock["t"] = 1.0
            # 2 J + 1 J = 3 J over 1 s -> 3.0 W (not 6.0 W from double-counting mmio)
            self.assertAlmostEqual(source.read().power_w, 3.0)

    def test_rapl_mmio_only_is_accepted_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mmio = root / "intel-rapl-mmio-0"
            mmio.mkdir()
            (mmio / "name").write_text("package-0\n", encoding="utf-8")
            (mmio / "energy_uj").write_text("1000000\n", encoding="utf-8")
            clock = {"t": 0.0}
            source = RaplSource(root=root, clock=lambda: clock["t"])
            status = source.probe()
            self.assertTrue(status.available)
            self.assertEqual(source._zones, [mmio])

    def test_rapl_multi_socket_independent_counter_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg0 = root / "intel-rapl-0"
            pkg0.mkdir()
            (pkg0 / "name").write_text("package-0\n", encoding="utf-8")
            (pkg0 / "energy_uj").write_text("90000000\n", encoding="utf-8")
            (pkg0 / "max_energy_range_uj").write_text("100000000\n", encoding="utf-8")

            pkg1 = root / "intel-rapl-1"
            pkg1.mkdir()
            (pkg1 / "name").write_text("package-1\n", encoding="utf-8")
            (pkg1 / "energy_uj").write_text("10000000\n", encoding="utf-8")
            (pkg1 / "max_energy_range_uj").write_text("100000000\n", encoding="utf-8")

            clock = {"t": 0.0}
            source = RaplSource(root=root, clock=lambda: clock["t"])
            source.probe()
            self.assertIsNone(source.read().power_w)

            # pkg0 wraps: (100M - 90M) + 5M = 15M uJ = 15 J
            # pkg1 advances normally: 15M - 10M = 5M uJ = 5 J
            (pkg0 / "energy_uj").write_text("5000000\n", encoding="utf-8")
            (pkg1 / "energy_uj").write_text("15000000\n", encoding="utf-8")
            clock["t"] = 1.0

            # Combined watts should be 15 + 5 = 20 W, not 120 W from summed counters
            self.assertAlmostEqual(source.read().power_w, 20.0)

    def test_jtop_board_object_uses_milliwatts(self) -> None:
        class _Board:
            power = {"tot": {"power": 4000, "avg": 3900}}
            stats = {"RAM": 1500}
            temperature = {"CPU": {"temp": 55.0}, "GPU": {"temp": 60.5}}
            throttled = False

        sample = sample_from_jtop(_Board())
        self.assertAlmostEqual(sample.power_w, 4.0)
        self.assertEqual(sample.host_memory_mb, 1500)
        self.assertAlmostEqual(sample.temperature_c, 60.5)
        self.assertFalse(sample.throttled)

        class _Hot:
            power = {"rail": {"VDD_IN": {"power": 1000}}}
            stats = {}
            temperature = {}
            throttled = True

        hot = sample_from_jtop(_Hot())
        self.assertAlmostEqual(hot.power_w, 1.0)
        self.assertTrue(hot.throttled)


class TestSamplerMath(unittest.TestCase):
    def test_sample_interval_is_ten_hertz_on_the_sampler_thread(self) -> None:
        seen: list[float] = []

        def sleeper(seconds: float) -> bool:
            seen.append(seconds)
            return True

        source = _CountingSource()
        summary = SensorSampler([source], sleeper=sleeper).open().close()
        self.assertEqual(SAMPLE_HZ, 10.0)
        self.assertEqual(summary.sample_hz, 10.0)
        self.assertAlmostEqual(seen[0], 0.1)
        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(source.threads, ["ohho-bench-sensors", "ohho-bench-sensors"])

    def test_mean_peak_and_energy_do_not_add_power_domains(self) -> None:
        summary = WindowSummary(
            sample_hz=10.0,
            sample_count=2,
            samples=[
                (
                    0.0,
                    SensorSample(
                        "board",
                        power_w=10.0,
                        host_memory_mb=100.0,
                        temperature_c=40.0,
                        clocks_mhz={"core": 1000.0},
                    ),
                ),
                (
                    0.1,
                    SensorSample(
                        "board",
                        power_w=30.0,
                        host_memory_mb=110.0,
                        temperature_c=50.0,
                    ),
                ),
                (0.0, SensorSample("accel", power_w=80.0, device_memory_mb=500.0)),
            ],
            statuses=[
                SourceStatus(
                    "board",
                    True,
                    power_domain=DOMAIN_BOARD,
                    observes_throttle=True,
                ),
                SourceStatus(
                    "accel",
                    True,
                    power_domain=DOMAIN_DEVICE,
                    observes_throttle=True,
                ),
            ],
        )
        block = summary.to_block(mean_latency_ms=12.5)
        self.assertEqual(block["power_w_mean"], 20.0)
        self.assertEqual(block["power_w_peak"], 30.0)
        self.assertEqual(block["energy_per_inference_mj"], 250.0)
        self.assertEqual(block["device_memory_mb_peak"], 500.0)
        self.assertEqual(block["host_memory_mb_peak"], 110.0)
        self.assertEqual(block["host_memory_source"], "board")
        self.assertIsNone(block["reason"])

    def test_negative_power_is_ignored(self) -> None:
        summary = WindowSummary(
            sample_hz=10.0,
            sample_count=2,
            samples=[
                (0.0, SensorSample("accel", power_w=-5.0)),
                (0.1, SensorSample("accel", power_w=10.0)),
            ],
            statuses=[
                SourceStatus(
                    "accel", True, power_domain=DOMAIN_DEVICE, observes_throttle=True
                )
            ],
        )
        block = summary.to_block(
            mean_latency_ms=1.0,
            host_fallback_mb=8.0,
            host_fallback_source="process_peak_rss",
        )
        self.assertEqual(block["power_w_mean"], 10.0)
        self.assertEqual(block["power_w_peak"], 10.0)

    def test_drift_report_lists_p95_per_window(self) -> None:
        report = drift_report(
            [(0.1, 10.0), (0.5, 10.0), (1.2, 30.0), (1.8, 30.0)],
            sustained_s=2.0,
            throttling_flagged=True,
        )
        self.assertEqual(len(report["windows"]), 2)
        self.assertAlmostEqual(report["windows"][0]["p95_ms"], 10.0)
        self.assertAlmostEqual(report["windows"][1]["p95_ms"], 30.0)
        self.assertAlmostEqual(report["p95_drift_ms"], 20.0)
        self.assertTrue(report["throttling_flagged"])


class TestHarnessSensors(unittest.TestCase):
    def test_mocked_device_computes_energy_per_inference(self) -> None:
        backend = _ConstantDevice()
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"scripted")
            document = execute(
                BenchConfig(
                    artifact=str(artifact),
                    target="x86-cpu",
                    warmup=0,
                    iters=4,
                ),
                runner=_Scaled(clock),
                clock=clock,
                sources=[NvmlSource(backend=backend)],
                gpu=GpuProbe(),
            )
        validate(document)
        sensors = document["telemetry"]["sensors"]
        self.assertEqual(sensors["power_w_mean"], 20.0)
        self.assertEqual(sensors["power_w_peak"], 20.0)
        self.assertEqual(sensors["device_memory_mb_peak"], 512.0)
        self.assertEqual(sensors["sample_hz"], 10.0)
        self.assertAlmostEqual(
            sensors["energy_per_inference_mj"],
            sensors["power_w_mean"] * document["latency"]["mean"],
        )
        self.assertEqual(document["telemetry"]["power_w_mean"], 20.0)
        self.assertEqual(document["telemetry"]["accelerator_memory_mb_max"], 512.0)
        self.assertEqual(sensors["clocks_mhz"]["nvml.graphics"], 1500.0)
        self.assertIn("ohho-bench-sensors", backend.threads)
        self.assertNotIn("sustained", document)
        reason = sensors["reason"] or ""
        self.assertNotIn("reported power", reason)

    def test_sustained_mode_shows_p95_drift_and_flags_throttling(self) -> None:
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"scripted")
            document = execute(
                BenchConfig(
                    artifact=str(artifact),
                    target="x86-cpu",
                    warmup=0,
                    iters=1,
                    sustained=2,
                ),
                runner=_DriftRunner(clock),
                clock=clock,
                sources=[_ThrottleSource()],
                gpu=GpuProbe(),
            )
        validate(document)
        sustained = document["sustained"]
        self.assertGreaterEqual(len(sustained["windows"]), 2)
        self.assertGreater(
            sustained["windows"][-1]["p95_ms"], sustained["windows"][0]["p95_ms"]
        )
        self.assertGreater(sustained["p95_drift_ms"], 0)
        self.assertTrue(sustained["throttling_flagged"])
        self.assertTrue(document["telemetry"]["sensors"]["throttling_flagged"])
        self.assertGreaterEqual(
            len(document["telemetry"]["sensors"]["throttling_events"]), 1
        )
        self.assertEqual(
            document["telemetry"]["sensors"]["throttling_events"][0]["detail"],
            "clocks reduced",
        )
        self.assertGreater(document["throughput"]["samples"], 80)

    def test_cpu_only_run_without_sensors_reports_null_power_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"scripted")
            out = root / "out.json"
            clock = VirtualClock()
            args = build_parser().parse_args(
                [
                    "bench",
                    "--artifact",
                    str(artifact),
                    "--target",
                    "x86-cpu",
                    "--warmup",
                    "0",
                    "--iters",
                    "4",
                    "--json",
                    str(out),
                ]
            )
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = run_bench_command(
                    args,
                    runner=_Scaled(clock),
                    clock=clock,
                    sources=_unavailable(),
                    gpu=GpuProbe(),
                )
            self.assertEqual(code, 0)
            document = json.loads(out.read_text(encoding="utf-8"))
        validate(document)
        sensors = document["telemetry"]["sensors"]
        self.assertIsNone(sensors["power_w_mean"])
        self.assertIsNone(sensors["power_w_peak"])
        self.assertIsNone(sensors["energy_per_inference_mj"])
        self.assertIsNone(sensors["device_memory_mb_peak"])
        self.assertIsInstance(sensors["reason"], str)
        self.assertIn("power", sensors["reason"])
        reasons = {row["id"]: row["reason"] for row in sensors["sources"]}
        self.assertEqual(set(reasons), {"nvml", "tegrastats", "rapl", "vcgencmd"})
        for reason in reasons.values():
            self.assertTrue(reason)
        self.assertNotIn("power_w_mean", document["telemetry"])
        self.assertIsNone(document["telemetry"]["gpu_power_w"])
        self.assertNotIn("sustained", document)

    def test_negative_sustained_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"x")
            with redirect_stderr(StringIO()):
                code = main(
                    [
                        "bench",
                        "--artifact",
                        str(artifact),
                        "--target",
                        "x86-cpu",
                        "--sustained",
                        "-1",
                    ]
                )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
