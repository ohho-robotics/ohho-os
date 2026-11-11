"""OHH-477 cost estimates from measured throughput and an editable price."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout

from ohho.bench.clock import VirtualClock
from ohho.bench.cost import (
    CostOverrides,
    format_estimate,
    price_catalog,
    rank_by_cost,
    recompute_cost,
)
from ohho.bench.errors import BenchHarnessError
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.harness import BenchConfig, execute
from ohho.bench.report import render_markdown, render_prometheus
from ohho.bench.schema import validate
from ohho.bench.sweep import SweepConfig, execute_sweep
from ohho.cli import main
from ohho.serve.loadgen import LoadResult, TokenSample, to_bench_result
from ohho.targets.schema import BUILTIN_IDS

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "ohho" / "bench" / "examples"


class _FixedRunner:
    """Each stage advances the virtual clock by the same amount."""

    def __init__(self, clock: VirtualClock, seconds: float = 0.01) -> None:
        self.clock = clock
        self.seconds = seconds

    def preprocess(self, batch: int) -> int:
        self.clock.advance(self.seconds)
        return batch

    def h2d(self, value: int) -> int:
        self.clock.advance(self.seconds)
        return value

    def inference(self, value: int) -> int:
        self.clock.advance(self.seconds)
        return value

    def d2h(self, value: int) -> int:
        self.clock.advance(self.seconds)
        return value

    def postprocess(self, value: int) -> int:
        self.clock.advance(self.seconds)
        return None


def _bare(
    unit: str,
    value: float,
    profile: str = "x86-cpu",
    *,
    power: float | None = None,
    workload: str = "inference",
) -> dict:
    document: dict = {
        "run": {"target": workload, "id": profile},
        "throughput": {"value": value, "unit": unit},
        "latency": {"p95": 10.0, "unit": "ms"},
        "telemetry": {},
        "environment": {"target_profile": profile},
    }
    if power is not None:
        document["telemetry"]["power_w_mean"] = power
    return document


def _generate(workload: str, target: str = "cloud-l4") -> dict:
    images: tuple[int, ...] = () if workload == "llm" else (16,)
    return execute_generate(
        GenerateConfig(
            target=target,
            workload=workload,
            precisions=("fp16",),
            concurrencies=(1,),
            prompt_tokens=(8,),
            image_sizes=images,
            max_tokens=2,
            requests=1,
            warmup=0,
        )
    )


class TestPriceTable(unittest.TestCase):
    def test_builtin_table_is_an_estimate_for_every_profile(self) -> None:
        catalog = price_catalog()
        self.assertEqual(catalog["schema"], "ohho.bench.cost_prices/v1")
        self.assertTrue(catalog["estimate"])
        self.assertEqual(catalog["label"], "estimate")
        self.assertIn("estimate", catalog["note"].lower())
        self.assertEqual(catalog["amortisation_hours"], 26280)
        self.assertEqual(catalog["electricity_usd_per_kwh"], 0.15)
        self.assertEqual(catalog["control_rate_hz"], 50)
        self.assertEqual(set(catalog["profiles"]), set(BUILTIN_IDS))
        self.assertEqual(catalog["profiles"]["cloud-l4"]["gpu_hour_usd"], 0.8)
        self.assertEqual(catalog["profiles"]["cloud-h100"]["gpu_hour_usd"], 4.0)
        self.assertEqual(catalog["profiles"]["x86-cpu"]["hardware_usd"], 800)
        self.assertEqual(catalog["profiles"]["x86-cpu"]["typical_watts"], 65)
        for profile_id, row in catalog["profiles"].items():
            self.assertGreater(row["typical_watts"], 0)
            if profile_id.startswith("cloud-"):
                self.assertEqual(row["pricing"], "cloud_gpu_hour")
                self.assertGreater(row["gpu_hour_usd"], 0)
            else:
                self.assertEqual(row["pricing"], "edge_amortised")
                self.assertGreater(row["hardware_usd"], 0)


class TestCostMath(unittest.TestCase):
    def test_edge_action_cost_uses_measured_throughput_and_power(self) -> None:
        document = recompute_cost(_bare("Hz", 20, "x86-cpu", power=100))
        cost = document["cost"]
        inputs = cost["inputs"]
        self.assertTrue(cost["estimate"])
        self.assertEqual(cost["label"], "estimate")
        self.assertEqual(cost["schema"], "ohho.bench.cost/v1")
        self.assertEqual(inputs["pricing"], "edge_amortised")
        self.assertEqual(inputs["watts_source"], "measured")
        self.assertEqual(inputs["watts"], 100)
        self.assertTrue(inputs["electricity_applied"])
        self.assertEqual(inputs["price_source"], "built-in")
        self.assertEqual(inputs["profile_source"], "run")
        hourly = inputs["hardware_usd"] / inputs["amortisation_hours"]
        hourly += (inputs["watts"] / 1000.0) * inputs["electricity_usd_per_kwh"]
        self.assertAlmostEqual(cost["usd_per_hour"], hourly)
        self.assertAlmostEqual(
            cost["usd_per_1k_actions"], hourly * 1000.0 / (20 * 3600)
        )
        self.assertAlmostEqual(cost["actions_per_watt"], 0.2)
        self.assertFalse(cost["meets_control_rate"])
        self.assertNotIn("usd_per_1m_output_tokens", cost)
        self.assertIn("estimate:", format_estimate(cost))

    def test_gpu_price_change_recomputes_without_touching_throughput(self) -> None:
        original = _bare("tokens/s", 45, "cloud-l4", workload="llm")
        before = copy.deepcopy(original)
        priced = recompute_cost(original)
        doubled = recompute_cost(original, CostOverrides(gpu_hour_usd=1.6))
        self.assertEqual(original, before)
        self.assertEqual(priced["throughput"], doubled["throughput"])
        self.assertEqual(priced["latency"], doubled["latency"])
        self.assertEqual(priced["cost"]["inputs"]["tokens_per_s"], 45)
        self.assertEqual(doubled["cost"]["inputs"]["price_source"], "flag")
        self.assertEqual(doubled["cost"]["inputs"]["pricing"], "cloud_gpu_hour")
        self.assertFalse(doubled["cost"]["inputs"]["electricity_applied"])
        self.assertAlmostEqual(priced["cost"]["usd_per_hour"], 0.8)
        self.assertAlmostEqual(
            doubled["cost"]["usd_per_1m_output_tokens"],
            2 * priced["cost"]["usd_per_1m_output_tokens"],
        )
        self.assertNotIn("usd_per_1k_actions", priced["cost"])

    def test_control_rate_changes_robots_and_not_cost_per_action(self) -> None:
        document = _bare("Hz", 20, "x86-cpu")
        base = recompute_cost(document)
        slower = recompute_cost(document, CostOverrides(control_rate_hz=25))
        met = recompute_cost(document, CostOverrides(control_rate_hz=10))
        self.assertAlmostEqual(
            base["cost"]["usd_per_1k_actions"], slower["cost"]["usd_per_1k_actions"]
        )
        self.assertAlmostEqual(
            slower["cost"]["robots_per_device"], 2 * base["cost"]["robots_per_device"]
        )
        self.assertAlmostEqual(
            slower["cost"]["usd_per_robot_hour"] * 2, base["cost"]["usd_per_robot_hour"]
        )
        self.assertFalse(base["cost"]["meets_control_rate"])
        self.assertTrue(met["cost"]["meets_control_rate"])

    def test_user_watts_override_measured_power(self) -> None:
        document = _bare("Hz", 20, "x86-cpu", power=100)
        measured = recompute_cost(document)["cost"]
        user = recompute_cost(document, CostOverrides(watts=10))["cost"]
        self.assertEqual(measured["inputs"]["watts_source"], "measured")
        self.assertEqual(user["inputs"]["watts_source"], "user")
        self.assertEqual(user["inputs"]["price_source"], "flag")
        self.assertAlmostEqual(user["actions_per_watt"], 2.0)
        self.assertGreater(measured["usd_per_hour"], user["usd_per_hour"])

    def test_gpu_power_sample_counts_as_measured(self) -> None:
        document = _bare("Hz", 10, "cloud-l4")
        document["telemetry"] = {"gpu_power_w": 40}
        cost = recompute_cost(document)["cost"]
        self.assertEqual(cost["inputs"]["watts_source"], "measured")
        self.assertEqual(cost["inputs"]["watts"], 40)
        self.assertFalse(cost["inputs"]["electricity_applied"])
        self.assertAlmostEqual(cost["actions_per_watt"], 0.25)

    def test_profile_default_watts_when_power_is_absent(self) -> None:
        cost = recompute_cost(_bare("Hz", 10, "x86-cpu"))["cost"]
        self.assertEqual(cost["inputs"]["watts_source"], "profile_default")
        self.assertEqual(cost["inputs"]["watts"], 65)

    def test_missing_profile_needs_a_price(self) -> None:
        document = _bare("Hz", 10, "x86-cpu")
        with self.assertRaises(BenchHarnessError) as caught:
            recompute_cost(document, CostOverrides(profile="not-a-board"))
        self.assertIn("not-a-board", str(caught.exception))
        priced = recompute_cost(
            document, CostOverrides(profile="custom-board", gpu_hour_usd=3)
        )
        self.assertEqual(priced["cost"]["usd_per_hour"], 3)
        self.assertEqual(priced["cost"]["inputs"]["profile_source"], "flag")
        self.assertEqual(priced["cost"]["inputs"]["watts_source"], "unmeasured")
        with self.assertRaises(BenchHarnessError):
            CostOverrides(gpu_hour_usd=1, hardware_usd=1)
        with self.assertRaises(BenchHarnessError):
            CostOverrides(gpu_hour_usd=-1)
        with self.assertRaises(BenchHarnessError):
            CostOverrides(profile="Not A Profile")

    def test_price_file_replaces_one_rate(self) -> None:
        document = _bare("tokens/s", 10, "cloud-l4", workload="llm")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "ohho.bench.cost_prices/v1",
                        "estimate": True,
                        "profiles": {"cloud-l4": {"gpu_hour_usd": 2.5}},
                    }
                ),
                encoding="utf-8",
            )
            priced = recompute_cost(document, CostOverrides(prices_path=str(path)))
            flagged = recompute_cost(
                document,
                CostOverrides(prices_path=str(path), gpu_hour_usd=9),
            )
            broken = Path(tmp) / "broken.json"
            broken.write_text(
                json.dumps({"schema": "ohho.bench.cost_prices/v1", "estimate": False}),
                encoding="utf-8",
            )
            with self.assertRaises(BenchHarnessError):
                recompute_cost(document, CostOverrides(prices_path=str(broken)))
        self.assertEqual(priced["cost"]["inputs"]["price_source"], "file")
        self.assertEqual(priced["cost"]["usd_per_hour"], 2.5)
        self.assertEqual(priced["cost"]["inputs"]["watts"], 72)
        self.assertEqual(flagged["cost"]["inputs"]["price_source"], "flag")
        self.assertEqual(flagged["cost"]["usd_per_hour"], 9)

    def test_result_without_a_profile_uses_the_cpu_row(self) -> None:
        document = _bare("Hz", 10, "x86-cpu")
        del document["environment"]["target_profile"]
        cost = recompute_cost(document)["cost"]
        self.assertEqual(cost["inputs"]["target_profile"], "x86-cpu")
        self.assertEqual(cost["inputs"]["profile_source"], "default")
        self.assertIn("x86-cpu", cost["inputs"]["note"])
        self.assertTrue(cost["estimate"])

    def test_rank_by_cost_keeps_the_slo_and_sorts_estimates(self) -> None:
        cheap = recompute_cost(_bare("tokens/s", 1000, "cloud-h100", workload="llm"))
        pricey = recompute_cost(_bare("tokens/s", 100, "cloud-l4", workload="llm"))
        cheap["latency"]["p95"] = 10
        pricey["latency"]["p95"] = 30
        ranked = rank_by_cost([pricey, cheap], metric="usd_per_1m_output_tokens")
        self.assertEqual(
            [row["target_profile"] for row in ranked], ["cloud-h100", "cloud-l4"]
        )
        self.assertTrue(
            all(row["estimate"] and row["label"] == "estimate" for row in ranked)
        )
        self.assertIn("gpu_hour_usd", ranked[0]["inputs"])
        under = rank_by_cost(
            [pricey, cheap],
            metric="usd_per_1m_output_tokens",
            max_p95_ms=10,
        )
        self.assertEqual([row["target_profile"] for row in under], ["cloud-h100"])
        self.assertEqual(
            len(
                rank_by_cost(
                    [pricey],
                    metric="usd_per_1m_output_tokens",
                    max_p95_ms=30,
                )
            ),
            1,
        )


class TestRuns(unittest.TestCase):
    def test_policy_run_prices_each_batch_from_hz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.onnx"
            artifact.write_bytes(b"not-onnx")
            clock = VirtualClock()
            document = execute(
                BenchConfig(
                    artifact=str(artifact),
                    target="x86-cpu",
                    warmup=0,
                    iters=1,
                    batches=(1,),
                    workload="vision",
                ),
                runner=_FixedRunner(clock),
                clock=clock,
            )
        validate(document)
        self.assertEqual(document["throughput"]["unit"], "Hz")
        self.assertAlmostEqual(document["throughput"]["value"], 20.0)
        cost = document["cost"]
        self.assertTrue(cost["estimate"])
        self.assertEqual(cost["label"], "estimate")
        self.assertAlmostEqual(cost["inputs"]["actions_per_s"], 20.0)
        self.assertNotIn("usd_per_1m_output_tokens", cost)
        self.assertIn("usd_per_1k_actions", cost)
        self.assertEqual(
            document["batches"][0]["cost"]["usd_per_1k_actions"],
            cost["usd_per_1k_actions"],
        )
        self.assertIn("### Cost (estimate)", render_markdown(document))
        self.assertIn("estimate | true", render_markdown(document))
        self.assertIn('estimate="true"', render_prometheus(document))
        again = recompute_cost(document, CostOverrides(gpu_hour_usd=2))
        self.assertEqual(again["throughput"], document["throughput"])
        self.assertEqual(again["latency"]["mean"], document["latency"]["mean"])
        self.assertNotEqual(
            again["cost"]["usd_per_1k_actions"], document["cost"]["usd_per_1k_actions"]
        )

    def test_generate_llm_vlm_and_vla(self) -> None:
        llm = _generate("llm")
        vlm = _generate("vlm")
        vla = _generate("vla")
        for document in (llm, vlm, vla):
            validate(document)
            self.assertTrue(document["cost"]["estimate"])
            self.assertEqual(document["cost"]["label"], "estimate")
            self.assertEqual(document["cost"]["inputs"]["pricing"], "cloud_gpu_hour")
            self.assertIn("usd_per_1m_output_tokens", document["cost"])
            self.assertGreater(document["cost"]["inputs"]["tokens_per_s"], 0)
            point = document["llm"]["points"][0]
            self.assertIn("cost", point)
            self.assertAlmostEqual(
                point["cost"]["inputs"]["tokens_per_s"], point["tokens_per_s"]
            )
        self.assertNotIn("usd_per_1k_actions", llm["cost"])
        self.assertNotIn("usd_per_1k_actions", vlm["cost"])
        self.assertIn("usd_per_1k_actions", vla["cost"])
        self.assertAlmostEqual(
            vla["cost"]["inputs"]["actions_per_s"], vla["llm"]["requests_per_s"]
        )
        tokens = llm["llm"]["tokens_per_s"]
        repriced = recompute_cost(llm, CostOverrides(gpu_hour_usd=1.6))
        self.assertEqual(repriced["llm"]["tokens_per_s"], tokens)
        self.assertAlmostEqual(
            repriced["cost"]["usd_per_1m_output_tokens"],
            2 * llm["cost"]["usd_per_1m_output_tokens"],
        )

    def test_sweep_and_loadgen_price_token_throughput(self) -> None:
        document = execute_sweep(
            SweepConfig(
                backend="stand-in",
                concurrencies=(1,),
                max_batch_tokens=(512,),
                prefix_cache=(False,),
                chunked_prefill=(False,),
                speculative=(False,),
                workloads=("short",),
                hardware_target="cloud-l4",
                requests=1,
                max_tokens=2,
            )
        )
        validate(document)
        self.assertEqual(document["environment"]["target_profile"], "cloud-l4")
        self.assertTrue(document["cost"]["estimate"])
        self.assertAlmostEqual(document["cost"]["usd_per_hour"], 0.8)
        self.assertIn("usd_per_1m_output_tokens", document["cost"])
        self.assertNotIn("usd_per_1k_actions", document["cost"])
        point = document["llm"]["points"][0]
        self.assertAlmostEqual(
            point["cost"]["inputs"]["tokens_per_s"], point["tokens_per_s"]
        )
        sample = TokenSample(
            ok=True,
            ttft_s=0.01,
            inter_token_s=(0.002,),
            e2e_s=0.02,
            output_tokens=4,
            text="abcd",
        )
        loaded = to_bench_result(
            LoadResult(
                (sample,),
                0.05,
                1,
                "http://127.0.0.1:9/v1",
                "model",
                "ping",
            ),
            backend="example",
            backend_version="0",
            precision="fp16",
            target_profile="cloud-h100",
        )
        validate(loaded)
        self.assertAlmostEqual(loaded["llm"]["tokens_per_s"], 80.0)
        self.assertEqual(loaded["cost"]["inputs"]["gpu_hour_usd"], 4.0)
        self.assertTrue(loaded["cost"]["estimate"])
        self.assertIn("inputs", loaded["cost"])

    def test_checked_in_examples_reprice_in_place_of_a_rerun(self) -> None:
        expected = {
            "vision.json": "usd_per_1k_actions",
            "vla.json": "usd_per_1k_actions",
            "llm.json": "usd_per_1m_output_tokens",
        }
        for name, field in expected.items():
            raw = json.loads((_EXAMPLES / name).read_text(encoding="utf-8"))
            throughput = copy.deepcopy(raw["throughput"])
            priced = recompute_cost(raw, CostOverrides(profile="cloud-l4"))
            doubled = recompute_cost(
                raw, CostOverrides(profile="cloud-l4", gpu_hour_usd=1.6)
            )
            self.assertEqual(priced["throughput"], throughput)
            self.assertEqual(doubled["throughput"], throughput)
            validate(priced)
            validate(doubled)
            self.assertTrue(priced["cost"]["estimate"])
            self.assertEqual(priced["cost"]["label"], "estimate")
            self.assertIn("target_profile", priced["cost"]["inputs"])
            self.assertIn(field, priced["cost"])
            self.assertAlmostEqual(doubled["cost"][field], 2 * priced["cost"][field])


class TestCostCli(unittest.TestCase):
    def test_prices_and_cost_commands_mark_estimates(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["bench", "prices"])
        self.assertEqual(code, 0, stderr.getvalue())
        text = stdout.getvalue()
        self.assertIn("estimate:", text)
        self.assertIn("cloud-l4", text)
        self.assertIn("x86-cpu", text)
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["bench", "prices", "--json"])
        self.assertEqual(code, 0, stderr.getvalue())
        catalog = json.loads(stdout.getvalue())
        self.assertTrue(catalog["estimate"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "cost",
                        str(_EXAMPLES / "llm.json"),
                        "--target",
                        "cloud-h100",
                        "--gpu-hour-usd",
                        "2",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("wrote", stdout.getvalue())
            self.assertIn("estimate:", stderr.getvalue())
            document = json.loads(output.read_text(encoding="utf-8"))
            validate(document)
            original = json.loads((_EXAMPLES / "llm.json").read_text(encoding="utf-8"))
            self.assertEqual(document["throughput"], original["throughput"])
            self.assertEqual(document["cost"]["inputs"]["gpu_hour_usd"], 2)
            self.assertTrue(document["cost"]["estimate"])
            self.assertEqual(document["cost"]["label"], "estimate")
            self.assertIn("tokens_per_s", document["cost"]["inputs"])

    def test_generate_cli_applies_a_gpu_hour_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "--mode",
                        "generate",
                        "--target",
                        "x86-cpu",
                        "--workload",
                        "llm",
                        "--concurrency",
                        "1",
                        "--prompt-tokens",
                        "8",
                        "--precision",
                        "fp16",
                        "--requests",
                        "1",
                        "--max-tokens",
                        "2",
                        "--warmup",
                        "0",
                        "--gpu-hour-usd",
                        "2.5",
                        "--json",
                        str(path),
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("estimate:", stderr.getvalue())
            document = json.loads(path.read_text(encoding="utf-8"))
            validate(document)
            self.assertEqual(document["cost"]["inputs"]["gpu_hour_usd"], 2.5)
            self.assertEqual(document["cost"]["inputs"]["price_source"], "flag")
            self.assertEqual(document["cost"]["inputs"]["pricing"], "cloud_gpu_hour")
            self.assertTrue(document["cost"]["estimate"])
