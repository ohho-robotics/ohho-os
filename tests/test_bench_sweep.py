"""OHH-476 LLM sweep: concurrency, prefix cache, Pareto, and skipped options."""

from __future__ import annotations

import json
import threading
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from ohho.bench.errors import BenchHarnessError
from ohho.bench.schema import validate
from ohho.bench.sweep import (
    SHARED_SYSTEM_PROMPT,
    SHORT_COMMANDS,
    TINY_IMAGE_URL,
    SweepConfig,
    execute_sweep,
    render_pareto_svg,
)
from ohho.cli import main
from ohho.serve.backends.errors import ServingBackendError
from ohho.serve.backends.llama_cpp import LlamaCppBackend
from ohho.serve.backends.sglang import SglangBackend
from ohho.serve.backends.sweep_support import support_for
from ohho.serve.backends.tensorrt_llm import TensorRtLlmBackend
from ohho.serve.backends.vllm import VllmBackend

_CONCURRENCY = (1, 2, 4, 8)


class TestStandInSweep(unittest.TestCase):
    def test_four_concurrencies_times_prefix_cache_and_a_recommendation(self) -> None:
        document = execute_sweep(SweepConfig(target_p95_ttft_ms=40))
        validate(document)
        llm = document["llm"]
        points = llm["points"]
        self.assertEqual(document["schema"], "ohho.bench/v1")
        self.assertEqual(llm["schema"], "ohho.bench.llm/v1")
        self.assertEqual(llm["sweep"]["cells"], 96)
        self.assertFalse(llm["sweep"]["engine_launched"])
        self.assertEqual(llm["sweep"]["timing_source"], "virtual")
        self.assertIn("does not launch", llm["sweep"]["note"])
        self.assertEqual(document["environment"]["timing_source"], "virtual")
        self.assertEqual(document["run"]["host"]["accelerator"], "none")
        pairs = {(point["concurrency"], point["prefix_cache"]) for point in points}
        for concurrency in _CONCURRENCY:
            self.assertIn((concurrency, False), pairs)
            self.assertIn((concurrency, True), pairs)
        workloads = {point["workload"] for point in points}
        self.assertEqual(workloads, {"short", "long-context", "shared-prefix"})
        long = next(point for point in points if point["workload"] == "long-context")
        short = next(point for point in points if point["workload"] == "short")
        self.assertEqual(long["image_tokens"], 256)
        self.assertGreater(long["prompt_tokens"], short["prompt_tokens"])
        self.assertEqual(short["image_tokens"], 0)

        shared = _cells(
            points,
            workload="shared-prefix",
            concurrency=1,
            max_batch_tokens=2048,
            chunked_prefill=False,
        )
        cached = next(point for point in shared if point["prefix_cache"])
        plain = next(point for point in shared if not point["prefix_cache"])
        self.assertLess(cached["ttft"]["p95"], plain["ttft"]["p95"])
        self.assertGreater(cached["tokens_per_s"], plain["tokens_per_s"])
        shorts = _cells(
            points,
            workload="short",
            concurrency=2,
            max_batch_tokens=512,
            chunked_prefill=False,
        )
        short_on = next(point for point in shorts if point["prefix_cache"])
        short_off = next(point for point in shorts if not point["prefix_cache"])
        self.assertEqual(short_on["ttft"]["p95"], short_off["ttft"]["p95"])
        self.assertEqual(short_on["tokens_per_s"], short_off["tokens_per_s"])

        recommendation = llm["recommendation"]
        self.assertTrue(recommendation["met"])
        self.assertEqual(recommendation["workload"], "shared-prefix")
        self.assertTrue(recommendation["prefix_cache"])
        self.assertLessEqual(recommendation["ttft_p95_ms"], 40)
        meeting = [
            point
            for point in points
            if point["workload"] == "shared-prefix" and point["ttft"]["p95"] <= 40
        ]
        best = max(meeting, key=lambda point: point["tokens_per_s"])
        self.assertGreaterEqual(recommendation["tokens_per_s"], best["tokens_per_s"])
        self.assertEqual(llm["ttft"]["p95"], recommendation["ttft_p95_ms"])
        self.assertEqual(llm["tokens_per_s"], recommendation["tokens_per_s"])
        self.assertEqual(llm["concurrency"], recommendation["concurrency"])
        self.assertEqual(len(llm["recommendations"]), 3)

        reasons = {row["option"]: row["reason"] for row in llm["skipped"]}
        self.assertEqual(set(reasons), {"speculative_decoding"})
        self.assertIn("draft", reasons["speculative_decoding"])
        self._assert_pareto(points, llm["pareto"])

        tight = execute_sweep(SweepConfig(target_p95_ttft_ms=0.01))
        self.assertFalse(tight["llm"]["recommendation"]["met"])
        self.assertGreater(tight["llm"]["sweep"]["cells"], 0)
        self.assertNotIn("concurrency", tight["llm"]["recommendation"])
        validate(tight)

    def test_chart_marks_pareto_cells(self) -> None:
        document = execute_sweep(SweepConfig(target_p95_ttft_ms=40, chart="pareto.svg"))
        svg = render_pareto_svg(document)
        self.assertIn("<svg", svg)
        self.assertIn("tokens/s", svg)
        self.assertIn("p95 TTFT", svg)
        self.assertIn("pareto", svg)
        self.assertIn("stand-in", svg)
        self.assertEqual(document["llm"]["sweep"]["chart"], "pareto.svg")

    def test_real_engine_is_not_launched_without_an_endpoint(self) -> None:
        with self.assertRaises(BenchHarnessError) as caught:
            execute_sweep(SweepConfig(backend="vllm", model="org/model"))
        message = str(caught.exception)
        self.assertIn("stand-in", message)
        self.assertIn("one-GPU", message)
        self.assertEqual(caught.exception.exit_code, 2)
        with self.assertRaises(BenchHarnessError) as alias:
            execute_sweep(SweepConfig(backend="trtllm", model="org/model"))
        self.assertIn("tensorrt-llm", str(alias.exception))
        with self.assertRaises(BenchHarnessError):
            execute_sweep(SweepConfig(backend="not-an-engine", model="x"))

    def _assert_pareto(self, points: list[dict], pareto: list[dict]) -> None:
        self.assertGreater(len(pareto), 0)
        for row in pareto:
            group = [other for other in pareto if other["workload"] == row["workload"]]
            for other in group:
                if other is row:
                    continue
                dominated = (
                    other["ttft_p95_ms"] <= row["ttft_p95_ms"]
                    and other["tokens_per_s"] >= row["tokens_per_s"]
                    and (
                        other["ttft_p95_ms"] < row["ttft_p95_ms"]
                        or other["tokens_per_s"] > row["tokens_per_s"]
                    )
                )
                self.assertFalse(dominated)
            match = next(
                point
                for point in points
                if point["workload"] == row["workload"]
                and point["concurrency"] == row["concurrency"]
                and point.get("prefix_cache") == row.get("prefix_cache")
                and point.get("max_batch_tokens") == row.get("max_batch_tokens")
                and point.get("chunked_prefill") == row.get("chunked_prefill")
                and point["ttft"]["p95"] == row["ttft_p95_ms"]
            )
            self.assertEqual(match["tokens_per_s"], row["tokens_per_s"])
            self.assertTrue(match["pareto"])


class TestLiveServerSkips(unittest.TestCase):
    def test_unsupported_options_are_skipped_and_the_sweep_completes(self) -> None:
        bodies: list[bytes] = []
        with _streaming_server(bodies) as endpoint:
            document = execute_sweep(
                SweepConfig(
                    backend="llama-cpp",
                    model="weights.gguf",
                    endpoint=endpoint,
                    concurrencies=_CONCURRENCY,
                    requests=1,
                    max_tokens=1,
                    timeout_s=5,
                    target_p95_ttft_ms=5000,
                )
            )
            joined = b"\n".join(bodies).decode("utf-8")
            bodies.clear()
            execute_sweep(
                SweepConfig(
                    backend="llama-cpp",
                    model="weights.gguf",
                    endpoint=endpoint,
                    concurrencies=(1,),
                    workloads=("short",),
                    requests=4,
                    max_tokens=1,
                    timeout_s=5,
                )
            )
            rotated = b"\n".join(bodies).decode("utf-8")
        validate(document)
        llm = document["llm"]
        self.assertEqual(len(llm["points"]), 12)
        self.assertEqual(
            {point["concurrency"] for point in llm["points"]}, set(_CONCURRENCY)
        )
        self.assertNotIn("prefix_cache", llm["points"][0])
        self.assertFalse(llm["sweep"]["engine_launched"])
        self.assertEqual(document["environment"]["timing_source"], "perf_counter")
        reasons = {row["option"]: row["reason"] for row in llm["skipped"]}
        self.assertEqual(
            set(reasons),
            {
                "max_batch_tokens",
                "prefix_cache",
                "chunked_prefill",
                "speculative_decoding",
            },
        )
        self.assertIn("does not relaunch", reasons["prefix_cache"])
        self.assertIn("does not relaunch", reasons["max_batch_tokens"])
        self.assertIn("chunked-prefill", reasons["chunked_prefill"])
        self.assertIn("draft", reasons["speculative_decoding"])
        self.assertTrue(llm["recommendation"]["met"])
        self.assertLessEqual(llm["recommendation"]["ttft_p95_ms"], 5000)
        self.assertIn(SHARED_SYSTEM_PROMPT[:40], joined)
        self.assertIn("Object 0 is at station 0.", joined)
        self.assertIn(TINY_IMAGE_URL, joined)
        self.assertIn('"content": "stop"', joined)
        for command in SHORT_COMMANDS:
            self.assertIn(command, rotated)
        support = support_for("llama-cpp")
        self.assertTrue(support.axis("prefix_cache").supported)
        self.assertFalse(support.axis("chunked_prefill").supported)
        triton = support_for("tensorrt-llm", launcher="triton")
        self.assertFalse(triton.axis("max_batch_tokens").supported)
        self.assertFalse(triton.axis("prefix_cache").supported)
        self.assertIn("config file", triton.axis("prefix_cache").reason)


class TestSweepLaunchFlags(unittest.TestCase):
    def test_adapters_emit_flags_only_where_they_apply(self) -> None:
        vllm = VllmBackend().plan(
            "org/model",
            "fp16",
            {
                "port": 8001,
                "dry_run": True,
                "max_batch_tokens": 2048,
                "prefix_cache": False,
                "chunked_prefill": True,
                "speculative_model": "org/draft",
                "speculative_tokens": 3,
            },
        )
        command = vllm.commands[0]
        self.assertIn("--max-num-batched-tokens", command)
        self.assertIn("2048", command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        blob = command[command.index("--speculative-config") + 1]
        self.assertIn("org/draft", blob)
        self.assertIn("3", blob)

        sglang = SglangBackend().plan(
            "org/model",
            "fp16",
            {
                "port": 30000,
                "dry_run": True,
                "max_batch_tokens": 1024,
                "prefix_cache": False,
                "chunked_prefill": False,
            },
        )
        sglang_command = sglang.commands[0]
        self.assertIn("--max-prefill-tokens", sglang_command)
        self.assertIn("--disable-radix-cache", sglang_command)
        self.assertIn("--chunked-prefill-size", sglang_command)
        self.assertIn("-1", sglang_command)

        tensor = TensorRtLlmBackend().plan(
            "org/model",
            "fp16",
            {"dry_run": True, "port": 8000, "max_batch_tokens": 4096},
        )
        self.assertIn("--max_num_tokens", tensor.commands[0])
        with self.assertRaises(ServingBackendError) as caught:
            TensorRtLlmBackend().plan(
                "org/model",
                "fp16",
                {"dry_run": True, "port": 8000, "prefix_cache": True},
            )
        self.assertIn("config file", str(caught.exception))

        llama = LlamaCppBackend().plan(
            "model.gguf",
            "fp16",
            {
                "dry_run": True,
                "port": 8080,
                "prefix_cache": True,
                "max_batch_tokens": 1024,
                "concurrency": 4,
            },
        )
        llama_command = llama.commands[0]
        self.assertEqual(llama_command[llama_command.index("--cache-reuse") + 1], "256")
        self.assertIn("--batch-size", llama_command)
        self.assertEqual(llama_command[llama_command.index("--parallel") + 1], "4")
        with self.assertRaises(ServingBackendError) as chunked:
            LlamaCppBackend().plan(
                "model.gguf",
                "auto",
                {"dry_run": True, "port": 8080, "chunked_prefill": True},
            )
        self.assertIn("chunked-prefill", str(chunked.exception))


class TestSweepCli(unittest.TestCase):
    def test_cli_writes_the_result_and_chart(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "result.json"
            chart = root / "pareto.svg"
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "llm",
                        "--sweep",
                        "--backend",
                        "stand-in",
                        "--target-p95-ttft",
                        "40ms",
                        "--concurrency",
                        "1,2,4,8",
                        "--output",
                        str(output),
                        "--chart",
                        str(chart),
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            text = stdout.getvalue()
            self.assertIn("wrote", text)
            self.assertIn("recommend", text)
            self.assertIn("skipped speculative_decoding", text)
            document = json.loads(output.read_text(encoding="utf-8"))
            validate(document)
            self.assertTrue(document["llm"]["recommendation"]["met"])
            self.assertTrue(document["llm"]["recommendation"]["prefix_cache"])
            svg = chart.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("tokens/s", svg)

    def test_engine_sweep_without_endpoint_does_not_fail_closed_as_a_skip(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "llm",
                        "--sweep",
                        "--backend",
                        "vllm",
                        "--model",
                        "org/model",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("one-GPU", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_single_measurement_still_requires_an_endpoint(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "bench",
                        "llm",
                        "--model",
                        "org/model",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("--endpoint", stderr.getvalue())


def _cells(points: list[dict], **wanted: object) -> list[dict]:
    return [
        point
        for point in points
        if all(point.get(key) == value for key, value in wanted.items())
    ]


@contextmanager
def _streaming_server(bodies: list[bytes]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            bodies.append(self.rfile.read(length) if length else b"")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            payload = json.dumps({"choices": [{"delta": {"content": "ok"}}]})
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
