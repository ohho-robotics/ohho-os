"""OHH-436 generate-mode bench: TTFT, TPOT, concurrency, and precision."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ohho.bench.agreement import local_top1_agreement, top1_action_agreement
from ohho.bench.clock import VirtualClock
from ohho.bench.errors import BenchHarnessError
from ohho.bench.generate import (
    GenerateConfig,
    MeasureRequest,
    PointMeasurement,
    RequestSample,
    _point_row,
    execute_generate,
    image_token_count,
    run_decode,
    samples_from_load,
    supported_backends,
)
from ohho.bench.report import render_markdown
from ohho.bench.schema import validate
from ohho.cli import main
from ohho.serve.loadgen import LoadResult, TokenSample
from ohho.targets.schema import get_profile


def _config(**overrides: object) -> GenerateConfig:
    values: dict[str, object] = {
        "target": "x86-cpu",
        "workload": "llm",
        "backend": "stand-in",
        "model": "stand-in",
        "precisions": ("fp16", "bnb-int4"),
        "concurrencies": (1, 4, 16),
        "prompt_tokens": (16, 64),
        "image_sizes": (0,),
        "max_tokens": 8,
        "requests": 4,
        "warmup": 0,
    }
    values.update(overrides)
    return GenerateConfig(**values)  # type: ignore[arg-type]


def _cell(document: dict, **match: object) -> dict:
    for point in document["llm"]["points"]:
        if all(point[key] == value for key, value in match.items()):
            return point
    raise AssertionError(f"no point matching {match}")


class TestGenerateMetrics(unittest.TestCase):
    def test_stand_in_is_schema_valid_and_reports_ttft_tpot(self) -> None:
        document = execute_generate(_config())
        validate(document)
        self.assertEqual(document["schema"], "ohho.bench/v1")
        self.assertEqual(document["llm"]["schema"], "ohho.bench.llm/v1")
        self.assertEqual(document["run"]["target"], "llm")
        self.assertEqual(document["environment"]["timing_source"], "virtual")
        for block in (
            document["llm"]["ttft"],
            document["llm"]["tpot"],
            document["latency"],
        ):
            for name in ("mean", "median", "p95"):
                self.assertIn(name, block)
                self.assertGreater(block[name], 0)
        self.assertEqual(document["throughput"]["unit"], "tokens/s")
        self.assertGreater(document["throughput"]["value"], 0)
        self.assertEqual(document["accuracy"]["primary"]["name"], "top1_action_token")
        stage_names = [stage["name"] for stage in document["latency"]["stages"]]
        self.assertEqual(stage_names, ["ttft", "tpot"])
        summary = render_markdown(document)
        self.assertIn("### Generate", summary)
        self.assertIn("tokens/s", summary)

    def test_concurrency_raises_throughput_and_latency(self) -> None:
        document = execute_generate(_config(precisions=("fp16",), prompt_tokens=(16,)))
        low = _cell(document, concurrency=1, precision="fp16", prompt_tokens=16)
        mid = _cell(document, concurrency=4, precision="fp16", prompt_tokens=16)
        high = _cell(document, concurrency=16, precision="fp16", prompt_tokens=16)
        self.assertGreater(mid["tokens_per_s"], low["tokens_per_s"])
        self.assertGreater(high["tokens_per_s"], mid["tokens_per_s"])
        self.assertGreater(high["ttft"]["p95"], low["ttft"]["p95"])
        self.assertGreater(high["tpot"]["median"], low["tpot"]["median"])
        self.assertGreater(high["e2e"]["mean"], low["e2e"]["mean"])
        self.assertGreater(high["tokens_per_request_s"], 0)
        self.assertLess(high["tokens_per_request_s"], low["tokens_per_request_s"])

    def test_prompt_and_image_sweeps_increase_ttft(self) -> None:
        prompts = execute_generate(
            _config(precisions=("fp16",), concurrencies=(1,), prompt_tokens=(16, 64))
        )
        short = _cell(prompts, prompt_tokens=16, concurrency=1)
        long = _cell(prompts, prompt_tokens=64, concurrency=1)
        self.assertGreater(long["ttft"]["mean"], short["ttft"]["mean"])

        images = execute_generate(
            _config(
                workload="vlm",
                precisions=("fp16",),
                concurrencies=(1,),
                prompt_tokens=(16,),
                image_sizes=(224, 448),
            )
        )
        self.assertEqual(image_token_count(224), 196)
        self.assertEqual(image_token_count(448), 784)
        small = _cell(images, image_size=224, concurrency=1)
        large = _cell(images, image_size=448, concurrency=1)
        self.assertGreater(large["image_tokens"], small["image_tokens"])
        self.assertGreater(large["ttft"]["p95"], small["ttft"]["p95"])
        validate(images)

    def test_precision_reports_latency_vram_and_agreement(self) -> None:
        document = execute_generate(
            _config(
                precisions=("fp16", "bf16", "bnb-int4", "awq", "gptq"),
                concurrencies=(1,),
                prompt_tokens=(16,),
                workload="vla",
                image_sizes=(224,),
            )
        )
        rows = {row["precision"]: row for row in document["llm"]["precisions"]}
        self.assertEqual(set(rows), {"fp16", "bf16", "bnb-int4", "awq", "gptq"})
        for name in ("ttft", "tpot", "e2e"):
            self.assertIn("mean", rows["fp16"][name])
            self.assertIn("median", rows["bnb-int4"][name])
            self.assertIn("p95", rows["awq"][name])
        self.assertEqual(rows["fp16"]["agreement"]["value"], 1.0)
        self.assertEqual(rows["bf16"]["agreement"]["value"], 1.0)
        for name in ("bnb-int4", "awq", "gptq"):
            self.assertLess(rows[name]["peak_vram_mb"], rows["fp16"]["peak_vram_mb"])
            self.assertGreater(rows[name]["agreement"]["value"], 0.0)
            self.assertLess(rows[name]["agreement"]["value"], 1.0)
            self.assertEqual(rows[name]["vram_source"], "stand-in-weights")
        self.assertEqual(rows["bf16"]["peak_vram_mb"], rows["fp16"]["peak_vram_mb"])
        self.assertEqual(document["run"]["target"], "vla")

    def test_slo_gates_pass_and_fail(self) -> None:
        config = _config(precisions=("fp16",), concurrencies=(1,), prompt_tokens=(16,))
        document = execute_generate(config)
        ttft = document["llm"]["ttft"]["p95"]
        tpot = document["llm"]["tpot"]["p95"]
        passed = execute_generate(
            config,
            slos=(f"ttft_p95<={ttft}ms", f"tpot_p95<={tpot}ms"),
        )
        self.assertTrue(passed["gates"]["passed"])
        failed = execute_generate(config, slos=(f"ttft_p95<{ttft}ms", "tpot_p95<=10s"))
        self.assertFalse(failed["gates"]["passed"])
        self.assertEqual(failed["gates"]["checks"][0]["metric"], "llm.ttft.p95")
        validate(failed)

    def test_llm_rejects_an_image_and_unknown_precision(self) -> None:
        with self.assertRaises(BenchHarnessError):
            _config(image_sizes=(224,))
        with self.assertRaises(BenchHarnessError):
            _config(precisions=("int3",))


class TestGenerateCli(unittest.TestCase):
    def test_cli_writes_a_valid_result_and_enforces_slos(self) -> None:
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
                        "1,4",
                        "--prompt-tokens",
                        "8",
                        "--precision",
                        "fp16,bnb-int4",
                        "--requests",
                        "2",
                        "--max-tokens",
                        "4",
                        "--warmup",
                        "0",
                        "--json",
                        str(path),
                        "--slo",
                        "ttft_p95<=10s,tpot_p95<=10s",
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            document = json.loads(path.read_text(encoding="utf-8"))
            validate(document)
            self.assertTrue(document["gates"]["passed"])
            self.assertEqual(len(document["gates"]["checks"]), 2)

            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "--mode",
                        "generate",
                        "--target",
                        "x86-cpu",
                        "--concurrency",
                        "1",
                        "--prompt-tokens",
                        "8",
                        "--precision",
                        "fp16",
                        "--requests",
                        "2",
                        "--warmup",
                        "0",
                        "--slo",
                        "ttft_p95<=0.0001ms,tpot_p95<=10s",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("gate failed", stderr.getvalue())
            self.assertIn("ttft_p95", stderr.getvalue())

    def test_gpu_backend_is_rejected_on_a_cpu_target(self) -> None:
        import ohho.serve.backends as backends

        called: list[str] = []
        original = backends.open_backend

        def _open(name: str) -> object:
            called.append(name)
            raise AssertionError(name)

        backends.open_backend = _open
        try:
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "--mode",
                        "generate",
                        "--target",
                        "x86-cpu",
                        "--backend",
                        "vllm",
                        "--model",
                        "org/model",
                        "--warmup",
                        "0",
                    ]
                )
        finally:
            backends.open_backend = original
        self.assertEqual(code, 2)
        self.assertIn("not supported", stderr.getvalue())
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "bench",
                    "--mode",
                    "generate",
                    "--target",
                    "x86-cpu",
                    "--baseline",
                    "prev.json",
                    "--warmup",
                    "0",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("--baseline", stderr.getvalue())
        self.assertEqual(called, [])
        cpu = supported_backends(get_profile("x86-cpu"))
        self.assertEqual(cpu, ("stand-in", "transformers"))
        for profile_id in ("x86-cuda", "jetson-orin-nano"):
            names = supported_backends(get_profile(profile_id))
            for backend in ("vllm", "sglang", "tensorrt-llm"):
                self.assertIn(backend, names)

    def test_import_does_not_load_optional_runtimes(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import ohho.bench.generate, sys\n"
                    "loaded = [name for name in ('torch', 'transformers', 'onnxruntime') "
                    "if name in sys.modules]\n"
                    "raise SystemExit(1 if loaded else 0)\n"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestAgreementAndDecode(unittest.TestCase):
    def test_local_agreement_and_parity_hook(self) -> None:
        self.assertEqual(local_top1_agreement((1, 2, 3, 4), (1, 2, 3, 4)), 1.0)
        self.assertEqual(local_top1_agreement((1, 2, 3, 4), (1, 9, 3, 9)), 0.5)
        module = types.ModuleType("ohho.parity")

        def hooked(reference: list[int], candidate: list[int]) -> float:
            del reference, candidate
            return 0.25

        module.top1_action_agreement = hooked
        previous = sys.modules.get("ohho.parity")
        sys.modules["ohho.parity"] = module
        try:
            self.assertEqual(top1_action_agreement([1], [1]), 0.25)
            document = execute_generate(
                _config(precisions=("fp16",), concurrencies=(1,), prompt_tokens=(8,))
            )
        finally:
            if previous is None:
                sys.modules.pop("ohho.parity", None)
            else:
                sys.modules["ohho.parity"] = previous
        self.assertEqual(document["llm"]["agreement"]["value"], 0.25)

    def test_decode_clock_and_load_mapping(self) -> None:
        clock = VirtualClock()
        state = {"n": 0}

        def step() -> int:
            clock.advance(0.01 if state["n"] == 0 else 0.002)
            state["n"] += 1
            return state["n"]

        sample = run_decode(step, 4, clock)
        self.assertAlmostEqual(sample.ttft_s, 0.01)
        self.assertEqual(sample.tpot_s, (0.002, 0.002, 0.002))
        self.assertEqual(sample.tokens, (1, 2, 3, 4))
        result = LoadResult(
            samples=(
                TokenSample(
                    ok=True,
                    ttft_s=0.1,
                    inter_token_s=(0.02, 0.03),
                    e2e_s=0.15,
                    output_tokens=3,
                    text="alpha beta gamma",
                ),
            ),
            wall_s=0.15,
            concurrency=1,
            endpoint_url="http://127.0.0.1/v1",
            model="m",
            prompt="p",
        )
        mapped = samples_from_load(result)
        self.assertEqual(mapped[0].tpot_s, (0.02, 0.03))
        self.assertEqual(len(mapped[0].tokens), 3)
        self.assertIsInstance(mapped[0], RequestSample)

    def test_four_bit_without_cuda_is_refused(self) -> None:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            self.skipTest("torch or transformers is not installed")
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA device is present")
        from ohho.bench.generate import MeasureRequest, TransformersGenerator

        generator = TransformersGenerator("org/model")
        try:
            with self.assertRaises(BenchHarnessError) as caught:
                generator.measure(
                    MeasureRequest(
                        precision="bnb-int4",
                        prompt_tokens=4,
                        image_tokens=0,
                        image_size=0,
                        concurrency=1,
                        requests=1,
                        max_tokens=2,
                    )
                )
        finally:
            generator.close()
        self.assertIn("CUDA", str(caught.exception))

    def test_transformers_backend_skips_without_its_packages(self) -> None:
        torch_spec = importlib.util.find_spec("torch")
        transformers_spec = importlib.util.find_spec("transformers")
        if torch_spec is not None and transformers_spec is not None:
            self.skipTest("torch and transformers are installed")
        with self.assertRaises(BenchHarnessError) as caught:
            execute_generate(
                _config(
                    backend="transformers",
                    model="org/model",
                    precisions=("fp32",),
                    concurrencies=(1,),
                    prompt_tokens=(4,),
                    requests=1,
                    max_tokens=2,
                )
            )
        self.assertIn("not installed", str(caught.exception))


class TestBugbotFindingsRegression(unittest.TestCase):
    def test_serving_rejects_and_skips_4bit_precisions(self) -> None:
        from ohho.bench.generate import ServingGenerator

        gen = ServingGenerator("vllm", "test/model")
        for p in ("bnb-int4", "awq", "gptq"):
            self.assertFalse(gen.supports_precision(p))
        self.assertTrue(gen.supports_precision("fp16"))

        with self.assertRaises(BenchHarnessError) as caught:
            execute_generate(
                _config(
                    target="x86-cuda",
                    backend="vllm",
                    model="test/model",
                    precisions=("bnb-int4",),
                    concurrencies=(1,),
                    prompt_tokens=(8,),
                ),
                generator=gen,
            )
        self.assertIn("cannot run any requested precision", str(caught.exception))

        def _mock_measure(req: MeasureRequest) -> PointMeasurement:
            sample = RequestSample(0.01, (0.002,), 0.02, (1, 2), output_tokens=2)
            return PointMeasurement((sample,), 0.02, None, "unavailable", requests_total=1)

        gen.measure = _mock_measure  # type: ignore[assignment]
        doc = execute_generate(
            _config(
                target="x86-cuda",
                backend="vllm",
                model="test/model",
                precisions=("fp16", "bnb-int4"),
                concurrencies=(1,),
                prompt_tokens=(8,),
            ),
            generator=gen,
        )
        validate(doc)
        self.assertEqual(doc["llm"]["precision"], "fp16")
        self.assertIn("skipped_precisions", doc["llm"])
        skipped = {s["precision"]: s["reason"] for s in doc["llm"]["skipped_precisions"]}
        self.assertIn("bnb-int4", skipped)
        self.assertIn("does not support", skipped["bnb-int4"])
        self.assertIn("bnb-int4", doc["telemetry"]["skipped_precisions"])

    def test_agreement_uses_model_reference_tokens(self) -> None:
        class _ModelGenerator:
            name = "mock-model"

            def measure(self, request: MeasureRequest) -> PointMeasurement:
                if request.precision == "fp16":
                    tokens = (5001, 5002)
                else:
                    tokens = (5001, 9999)
                sample = RequestSample(0.01, (0.002,), 0.02, tokens, output_tokens=2)
                return PointMeasurement((sample,), 0.02, None, "unavailable", requests_total=1)

        doc = execute_generate(
            _config(
                precisions=("fp16", "bnb-int4"),
                concurrencies=(1,),
                prompt_tokens=(8,),
            ),
            generator=_ModelGenerator(),
        )
        validate(doc)
        rows = {r["precision"]: r for r in doc["llm"]["precisions"]}
        self.assertEqual(rows["fp16"]["agreement"]["value"], 1.0)
        self.assertEqual(rows["bnb-int4"]["agreement"]["value"], 0.5)

    def test_samples_from_load_chunk_counts_and_partial_failures(self) -> None:
        chunks = ("h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d")
        sample_ok = TokenSample(
            ok=True,
            ttft_s=0.05,
            inter_token_s=tuple([0.01] * 10),
            e2e_s=0.15,
            output_tokens=11,
            text="hello world",
            chunks=chunks,
        )
        sample_fail = TokenSample(
            ok=False,
            ttft_s=0.0,
            inter_token_s=(),
            e2e_s=0.02,
            output_tokens=0,
            text="",
            error="connection refused",
        )
        result = LoadResult(
            samples=(sample_ok, sample_fail),
            wall_s=0.2,
            concurrency=2,
            endpoint_url="http://127.0.0.1:8000/v1",
            model="m",
            prompt="p",
        )
        mapped = samples_from_load(result)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].output_tokens, 11)
        self.assertEqual(len(mapped[0].tokens), 11)

        point = PointMeasurement(
            samples=mapped,
            wall_s=0.2,
            peak_vram_mb=None,
            vram_source="unavailable",
            requests_total=len(result.samples),
        )
        req = MeasureRequest("fp16", 8, 0, 0, 2, 2, 11)
        cfg = _config(precisions=("fp16",), concurrencies=(2,), prompt_tokens=(8,))
        row = _point_row(cfg, req, point)
        self.assertEqual(row["requests"], 2)
        self.assertEqual(row["requests_ok"], 1)
        self.assertEqual(row["output_tokens"], 11)
        self.assertAlmostEqual(row["tokens_per_s"], 11 / 0.2)

    def test_transformers_awq_and_gptq_require_packages_and_do_not_fall_back(self) -> None:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            self.skipTest("torch or transformers is not installed")
        from ohho.bench.generate import TransformersGenerator

        gen = TransformersGenerator("org/model")
        try:
            with self.assertRaises(BenchHarnessError) as caught_awq:
                gen._load("awq")
            self.assertTrue(
                "CUDA" in str(caught_awq.exception)
                or "AWQ" in str(caught_awq.exception)
                or "could not load" in str(caught_awq.exception)
            )

            with self.assertRaises(BenchHarnessError) as caught_gptq:
                gen._load("gptq")
            self.assertTrue(
                "CUDA" in str(caught_gptq.exception)
                or "GPTQ" in str(caught_gptq.exception)
                or "could not load" in str(caught_gptq.exception)
            )
        finally:
            gen.close()

    def test_transformers_cache_evicts_previous_precision(self) -> None:
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            self.skipTest("torch or transformers is not installed")
        from ohho.bench.generate import TransformersGenerator

        gen = TransformersGenerator("org/model")
        fake_tok = object()
        fake_m1 = object()
        fake_m2 = object()
        gen._tokenizer = fake_tok
        gen._loaded_model = fake_m1
        gen._loaded_precision = "fp16"
        self.assertEqual(len(gen._cache), 1)
        self.assertIn("fp16", gen._cache)

        gen._loaded_model = fake_m2
        gen._loaded_precision = "fp32"
        self.assertEqual(len(gen._cache), 1)
        self.assertNotIn("fp16", gen._cache)
        self.assertIn("fp32", gen._cache)

        gen.close()
        self.assertEqual(len(gen._cache), 0)

    def test_transformers_batched_concurrency(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        import torch
        from ohho.bench.generate import _transformers_batch_sample

        class _MockModel:
            def __call__(
                self,
                input_ids: Any,
                use_cache: bool = True,
                past_key_values: Any = None,
            ) -> Any:
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                vocab_size = 32
                logits = torch.ones((batch_size, seq_len, vocab_size))
                for b in range(batch_size):
                    logits[b, -1, b % vocab_size] = 10.0
                return type(
                    "Out", (), {"logits": logits, "past_key_values": object()}
                )()

        prompt = torch.tensor([[1, 2, 3]])
        samples = _transformers_batch_sample(
            _MockModel(), torch, prompt, max_tokens=3, device="cpu", batch_size=4
        )
        self.assertEqual(len(samples), 4)
        for i, sample in enumerate(samples):
            self.assertEqual(len(sample.tokens), 3)
            self.assertEqual(sample.tokens[0], i % 32)


if __name__ == "__main__":
    unittest.main()
