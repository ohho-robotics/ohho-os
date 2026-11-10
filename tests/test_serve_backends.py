"""LLM serving backends, process teardown, and the bench load generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Iterator

from ohho.cli import main
from ohho.serve.backends import ServingBackendError, backend_names, open_backend
from ohho.serve.backends.llama_cpp import LlamaCppBackend
from ohho.serve.backends.sglang import SglangBackend
from ohho.serve.backends.tensorrt_llm import TensorRtLlmBackend
from ohho.serve.backends.tree import descendants, kill_pid, pid_alive
from ohho.serve.backends.vllm import VllmBackend
from ohho.serve.loadgen import (
    LLM_BLOCK_ID,
    LoadGenError,
    LoadResult,
    LoadSpec,
    TokenSample,
    latency_stats,
    run_load,
    to_bench_result,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests" / "fixtures" / "fake_openai_server.py"


def _prefix() -> list[str]:
    return [sys.executable, str(_FIXTURE)]


def _load_fake_server() -> object:
    name = "_ohho_fake_openai_server"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _FIXTURE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {_FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_pid(path: Path, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return int(path.read_text(encoding="utf-8").strip())
        time.sleep(0.02)
    raise AssertionError(f"pid file missing: {path}")


def _wait_dead(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


@contextmanager
def _env(**values: str | None) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestServingCommands(unittest.TestCase):
    def test_backend_names_and_aliases(self) -> None:
        self.assertEqual(
            backend_names(),
            ("vllm", "sglang", "tensorrt-llm", "llama-cpp"),
        )
        self.assertIsInstance(open_backend("trtllm"), TensorRtLlmBackend)
        self.assertIsInstance(open_backend("llama.cpp"), LlamaCppBackend)
        with self.assertRaises(ServingBackendError) as caught:
            open_backend("nope")
        self.assertIn("unknown serving backend", str(caught.exception))

    def test_vllm_command_precision_and_urls(self) -> None:
        plan = VllmBackend().plan(
            "org/model",
            "FP16",
            {
                "host": "127.0.0.1",
                "port": 8001,
                "dry_run": True,
                "tensor_parallel_size": 2,
            },
        )
        self.assertEqual(
            plan.commands[0],
            (
                "vllm",
                "serve",
                "org/model",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
                "--dtype",
                "float16",
                "--tensor-parallel-size",
                "2",
            ),
        )
        self.assertEqual(plan.endpoint_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(plan.health_url, "http://127.0.0.1:8001/health")
        self.assertEqual(plan.version_argv, ("vllm", "--version"))
        fp8 = VllmBackend().plan("org/model", "fp8", {"port": 1, "dry_run": True})
        self.assertIn("--quantization", fp8.commands[0])
        self.assertIn("fp8", fp8.commands[0])

    def test_sglang_command(self) -> None:
        plan = SglangBackend().plan(
            "org/model",
            "bf16",
            {"dry_run": True, "tp_size": 2},
        )
        self.assertEqual(
            plan.commands[0],
            (
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                "org/model",
                "--host",
                "127.0.0.1",
                "--port",
                "30000",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "2",
            ),
        )
        self.assertEqual(plan.health_url, "http://127.0.0.1:30000/health")
        self.assertEqual(plan.endpoint_url, "http://127.0.0.1:30000/v1")

    def test_tensorrt_llm_commands(self) -> None:
        fp16 = TensorRtLlmBackend().plan(
            "org/model",
            "fp16",
            {"dry_run": True, "port": 8000},
        )
        self.assertEqual(
            fp16.commands[0],
            (
                "trtllm-serve",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "org/model",
            ),
        )
        self.assertEqual(fp16.endpoint_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(fp16.health_url, "http://127.0.0.1:8000/health")
        fp8 = TensorRtLlmBackend().plan(
            "org/model",
            "fp8",
            {"dry_run": True, "port": 8000, "tp_size": 2},
        )
        self.assertIn("--kv_cache_dtype", fp8.commands[0])
        self.assertIn("fp8", fp8.commands[0])
        self.assertIn("--tp_size", fp8.commands[0])
        triton = TensorRtLlmBackend().plan(
            "unused",
            "auto",
            {
                "launcher": "triton",
                "dry_run": True,
                "port": 8010,
                "model_repository": "models/repo",
                "openai_port": 9000,
            },
        )
        command = triton.commands[0]
        self.assertEqual(command[0], "tritonserver")
        self.assertIn("--model-repository", command)
        self.assertIn("models/repo", command)
        self.assertIn("--http-port", command)
        self.assertIn("8010", command)
        self.assertEqual(triton.health_url, "http://127.0.0.1:8010/v2/health/ready")
        self.assertEqual(triton.endpoint_url, "http://127.0.0.1:9000/v1")
        with self.assertRaises(ServingBackendError) as missing:
            TensorRtLlmBackend().plan(
                "missing-repo",
                "auto",
                {"launcher": "triton", "port": 8010},
            )
        self.assertIn("does not exist", str(missing.exception))

    def test_llama_cpp_command_cpu_and_jetson(self) -> None:
        cpu = LlamaCppBackend().plan(
            "model.gguf",
            "fp16",
            {"dry_run": True, "port": 8080},
        )
        command = cpu.commands[0]
        self.assertEqual(command[0], "llama-server")
        self.assertIn("-m", command)
        self.assertIn("model.gguf", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "f16")
        self.assertEqual(cpu.endpoint_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(cpu.health_url, "http://127.0.0.1:8080/health")
        jetson = LlamaCppBackend().plan(
            "model.gguf",
            "auto",
            {"dry_run": True, "n_gpu_layers": "auto", "ctx_size": 4096},
        )
        self.assertEqual(
            jetson.commands[0][jetson.commands[0].index("--n-gpu-layers") + 1],
            "auto",
        )
        self.assertNotIn("--cache-type-k", jetson.commands[0])
        self.assertIn("--ctx-size", jetson.commands[0])
        with self.assertRaises(ServingBackendError) as missing:
            LlamaCppBackend().plan("missing.gguf", "auto", {"port": 9})
        self.assertIn("does not exist", str(missing.exception))

    def test_unknown_precision_does_not_start_a_process(self) -> None:
        backend = VllmBackend()
        with self.assertRaises(ServingBackendError):
            backend.launch("org/model", "int4", {"startup_timeout_s": 1})
        self.assertFalse(backend.running)
        self.assertFalse(backend.health())
        with self.assertRaises(ServingBackendError):
            backend.endpoint_url()

    def test_extra_args_reject_a_shell_string(self) -> None:
        with self.assertRaises(ServingBackendError):
            VllmBackend().plan("m", "auto", {"dry_run": True, "extra_args": "echo hi"})

    def test_version_parsers(self) -> None:
        self.assertEqual(VllmBackend().parse_version("0.6.3.post1\n"), "0.6.3.post1")
        self.assertEqual(SglangBackend().parse_version("0.4.6.post1"), "0.4.6.post1")
        self.assertEqual(
            TensorRtLlmBackend().parse_version("TensorRT-LLM version: 0.18.2\n"),
            "0.18.2",
        )
        self.assertEqual(
            TensorRtLlmBackend().parse_version(
                "Triton Inference Server version 2.51.0\n"
            ),
            "2.51.0",
        )
        self.assertEqual(
            LlamaCppBackend().parse_version(
                "version: 4012 (8f0c0c0)\nbuilt with gcc 11.4.0 for x86_64\n"
            ),
            "4012 (8f0c0c0)",
        )
        with self.assertRaises(ServingBackendError):
            VllmBackend().parse_version("no version here")

    def test_missing_engine_and_fake_version(self) -> None:
        missing = str(_FIXTURE.parent / "no-such-llama-server")
        with self.assertRaises(ServingBackendError) as caught:
            LlamaCppBackend().version({"executable": missing})
        self.assertIn("not installed", str(caught.exception))
        self.assertIn("not a dependency of ohho-os", str(caught.exception))
        version = VllmBackend().version({"argv_prefix": _prefix()})
        self.assertEqual(version, "9.9.9")
        argv = [sys.executable, "-c", "import ohho_missing_engine_mod"]
        with self.assertRaises(ServingBackendError) as imported:
            TensorRtLlmBackend().version({"version_argv": argv})
        self.assertIn("not installed", str(imported.exception))


class TestProcessTree(unittest.TestCase):
    def test_descendants_include_a_child(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 2
            seen = False
            while time.monotonic() < deadline:
                if proc.pid in descendants(os.getpid()):
                    seen = True
                    break
                time.sleep(0.05)
            self.assertTrue(seen, f"child {proc.pid} was not listed")
        finally:
            kill_pid(proc.pid)
            proc.wait(timeout=5)

    def test_exit_and_hang_leave_no_child(self) -> None:
        for no_job in ("0", "1"):
            for mode, needle, timeout in (
                # Spawn plus a child pid file can take several seconds on a
                # busy Windows runner. The exit path must still be observed
                # before this deadline; the hang path never serves HTTP.
                ("exit", "exited before it was healthy", 20.0),
                ("hang", "did not become healthy", 10.0),
            ):
                with self.subTest(no_job=no_job, mode=mode):
                    self._failed_launch(no_job, mode, needle, timeout)

    def _failed_launch(
        self, no_job: str, mode: str, needle: str, timeout: float
    ) -> None:
        pid = 0
        backend = VllmBackend()
        try:
            with tempfile_dir() as tmp:
                pid_path = tmp / "child.pid"
                with _env(OHHO_SERVE_NO_JOB=no_job):
                    with self.assertRaises(ServingBackendError) as caught:
                        backend.launch(
                            "tiny",
                            "fp16",
                            {
                                "argv_prefix": _prefix(),
                                "startup_timeout_s": timeout,
                                "env": {
                                    "OHHO_FAKE_MODE": mode,
                                    "OHHO_FAKE_CHILD_PID": str(pid_path),
                                },
                            },
                        )
                message = str(caught.exception)
                self.assertIn(needle, message)
                if mode == "exit":
                    self.assertIn("fake-exit", message)
                pid = _read_pid(pid_path)
                self.assertTrue(_wait_dead(pid), f"child {pid} still running")
                self.assertFalse(backend.running)
        finally:
            backend.teardown()
            if pid and pid_alive(pid):
                kill_pid(pid)

    def test_fake_server_streams_and_teardown_kills_the_child(self) -> None:
        pid = 0
        backend = VllmBackend()
        try:
            with tempfile_dir() as tmp:
                pid_path = tmp / "child.pid"
                backend.launch(
                    "tiny",
                    "fp16",
                    {
                        "argv_prefix": _prefix(),
                        "startup_timeout_s": 30,
                        "env": {
                            "OHHO_FAKE_MODE": "serve",
                            "OHHO_FAKE_CHILD_PID": str(pid_path),
                            "OHHO_FAKE_TOKENS": "ping,pong",
                        },
                    },
                )
                self.assertTrue(backend.health())
                self.assertTrue(backend.endpoint_url().endswith("/v1"))
                pid = _read_pid(pid_path)
                self.assertTrue(pid_alive(pid))
                result = run_load(
                    LoadSpec(
                        endpoint_url=backend.endpoint_url(),
                        model="tiny",
                        requests=2,
                        concurrency=1,
                        max_tokens=4,
                        timeout_s=30,
                    )
                )
                self.assertTrue(all(sample.ok for sample in result.samples))
                self.assertEqual(result.samples[0].text, "pingpong")
                self.assertEqual(result.samples[0].output_tokens, 2)
                self.assertEqual(len(result.samples[0].inter_token_s), 1)
                document = to_bench_result(
                    result,
                    backend="vllm",
                    backend_version="9.9.9",
                    precision="fp16",
                )
                self.assertEqual(document["schema"], "ohho.bench/v1")
                self.assertEqual(document["llm"]["schema"], LLM_BLOCK_ID)
                self.assertEqual(document["llm"]["output_tokens"], 4)
                self.assertEqual(document["llm"]["concurrency"], 1)
                self.assertEqual(document["throughput"]["unit"], "tokens/s")
                self.assertGreater(document["llm"]["tokens_per_s"], 0)
                self.assertGreater(document["llm"]["requests_per_s"], 0)
                self.assertEqual(document["run"]["host"]["accelerator"], "none")
                self.assertEqual(document["run"]["target"], "llm")
                self.assertEqual(
                    [stage["name"] for stage in document["latency"]["stages"]],
                    ["ttft", "inter_token"],
                )
                backend.teardown()
                backend.teardown()
                self.assertTrue(_wait_dead(pid), f"child {pid} still running")
                self.assertFalse(backend.health())
        finally:
            backend.teardown()
            if pid and pid_alive(pid):
                kill_pid(pid)

    def test_fake_server_listens_without_reverse_dns(self) -> None:
        module = _load_fake_server()

        def stall(host: str = "") -> str:
            time.sleep(3)
            return host

        with patch("socket.getfqdn", stall):
            started = time.monotonic()
            server = module._ListeningServer(
                ("127.0.0.1", 0), module._handler(["a"], 0.0)
            )
            elapsed = time.monotonic() - started
            try:
                self.assertLess(elapsed, 1.0)
                port = server.server_address[1]
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
            finally:
                server.server_close()


@contextmanager
def tempfile_dir() -> Iterator[Path]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestLoadGenerator(unittest.TestCase):
    def test_latency_stats(self) -> None:
        stats = latency_stats([10, 20, 30, 40], "ms")
        self.assertEqual(stats["p50"], 25)
        self.assertEqual(stats["median"], 25)
        self.assertAlmostEqual(stats["p90"], 37)
        self.assertAlmostEqual(stats["mean"], 25)
        self.assertAlmostEqual(stats["std"], math.sqrt(125))
        self.assertEqual(stats["samples"], 4)
        empty = latency_stats([], "ms")
        self.assertEqual(empty["samples"], 0)
        self.assertEqual(empty["max"], 0)

    def test_identifier_and_file_digests(self) -> None:
        sample = TokenSample(True, 0.01, (0.002,), 0.02, 2, "ab")
        result = LoadResult(
            samples=(sample,),
            wall_s=0.05,
            concurrency=1,
            endpoint_url="http://127.0.0.1:9/v1",
            model="org/model",
            prompt="ping",
        )
        document = to_bench_result(
            result,
            backend="sglang",
            backend_version="0.0.0",
            precision="bf16",
        )
        digest = hashlib.sha256(b"org/model").hexdigest()
        self.assertEqual(document["run"]["model_sha"], f"sha256:{digest}")
        self.assertEqual(document["llm"]["model_sha_of"], "identifier")
        self.assertEqual(document["run"]["artifact_sha_of"], "serving-setup")
        self.assertEqual(
            document["accuracy"]["primary"]["name"], "request_completion_rate"
        )
        with tempfile_dir() as tmp:
            weights = tmp / "weights.bin"
            weights.write_bytes(b"abc")
            filed = LoadResult(
                samples=(sample,),
                wall_s=0.05,
                concurrency=2,
                endpoint_url="http://127.0.0.1:9/v1",
                model=str(weights),
                prompt="ping",
            )
            hashed = to_bench_result(
                filed,
                backend="llama-cpp",
                backend_version="1",
                precision="fp16",
            )
        self.assertEqual(hashed["llm"]["model_sha_of"], "file")
        self.assertEqual(
            hashed["run"]["model_sha"],
            "sha256:" + hashlib.sha256(b"abc").hexdigest(),
        )
        self.assertEqual(hashed["throughput"]["batch_size"], 2)

    def test_concurrency_and_failed_endpoint(self) -> None:
        peak = {"value": 0, "inflight": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                with lock:
                    peak["inflight"] += 1
                    peak["value"] = max(peak["value"], peak["inflight"])
                try:
                    time.sleep(0.35)
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        self.rfile.read(length)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    payload = json.dumps({"choices": [{"delta": {"content": "x"}}]})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with lock:
                        peak["inflight"] -= 1

            def log_message(self, fmt: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_load(
                LoadSpec(
                    endpoint_url=f"http://127.0.0.1:{port}/v1",
                    model="fake",
                    concurrency=2,
                    requests=4,
                    timeout_s=10,
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(peak["value"], 2)
        self.assertEqual(len(result.samples), 4)
        self.assertTrue(all(sample.text == "x" for sample in result.samples))
        self.assertEqual(result.samples[0].inter_token_s, ())
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            closed = sock.getsockname()[1]
        with self.assertRaises(LoadGenError) as caught:
            run_load(
                LoadSpec(
                    endpoint_url=f"http://127.0.0.1:{closed}/v1",
                    model="fake",
                    requests=1,
                    timeout_s=2,
                )
            )
        self.assertIn("every request failed", str(caught.exception))


class TestBenchCli(unittest.TestCase):
    def test_dry_run_prints_the_vllm_command(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "bench",
                    "launch",
                    "--backend",
                    "vllm",
                    "--model",
                    "org/model",
                    "--precision",
                    "fp16",
                    "--port",
                    "8001",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["backend"], "vllm")
        self.assertEqual(payload["endpoint_url"], "http://127.0.0.1:8001/v1")
        self.assertIn("float16", payload["commands"][0])
        self.assertNotIn("error", stderr.getvalue().lower())

    def test_version_of_a_missing_executable(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        missing = str(_FIXTURE.parent / "no-such-llama-server")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "bench",
                    "version",
                    "--backend",
                    "llama-cpp",
                    "--executable",
                    missing,
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("not installed", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_llm_writes_a_valid_result(self) -> None:
        tokens = {"value": "ok,go"}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for token in tokens["value"].split(","):
                    payload = json.dumps({"choices": [{"delta": {"content": token}}]})
                    self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, fmt: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile_dir() as tmp:
                output = tmp / "result.json"
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(
                        [
                            "bench",
                            "llm",
                            "--endpoint",
                            f"http://127.0.0.1:{port}/v1",
                            "--model",
                            "fake",
                            "--concurrency",
                            "1",
                            "--requests",
                            "1",
                            "--output",
                            str(output),
                            "--backend-name",
                            "vllm",
                            "--backend-version",
                            "9.9.9",
                            "--precision",
                            "fp16",
                        ]
                    )
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertIn("wrote", stdout.getvalue())
                document = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(document["llm"]["schema"], LLM_BLOCK_ID)
                self.assertEqual(document["llm"]["output_tokens"], 2)
                validate = __import__("ohho.bench", fromlist=["validate"]).validate
                validate(document)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_help_lists_every_backend(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["bench", "launch", "--help"])
        self.assertEqual(caught.exception.code, 0)
        text = stdout.getvalue()
        for name in backend_names():
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
