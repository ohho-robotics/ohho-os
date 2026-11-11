"""OHH-498 optional exporters for bench and optimize results.

W&B is mocked. A live import is skipped when the package is absent. Prometheus
is exercised with a local fake HTTP server. No API key is sent and no external
service is contacted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ohho.bench import export_result, validate
from ohho.bench.clock import VirtualClock
from ohho.bench.command import run_bench_command
from ohho.bench.exports import ExportContext, parse_spec
from ohho.bench.exports.prometheus_export import (
    encode_write_request,
    snappy_compress,
    snappy_decompress,
)
from ohho.bench.exports.registry import discover_exporters
from ohho.bench.report import render_markdown
from ohho.cli import build_parser

_ROOT = Path(__file__).resolve().parents[1]
_KEY = "unit-test-not-a-key"


class _Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        self._record("POST", body)
        self._reply(200)

    def do_DELETE(self) -> None:  # noqa: N802
        self._record("DELETE", b"")
        code = 500 if getattr(self.server, "fail_delete", False) else 200
        self._reply(code)

    def _record(self, method: str, body: bytes) -> None:
        headers = {key.lower(): value for key, value in self.headers.items()}
        events = getattr(self.server, "events")
        events.append((method, self.path, body, headers))

    def _reply(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


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


class _Explode:
    def preprocess(self, batch: int) -> int:
        raise AssertionError("runner called")

    def h2d(self, prepared: int) -> int:
        raise AssertionError("runner called")

    def inference(self, device_input: int) -> int:
        raise AssertionError("runner called")

    def d2h(self, device_output: int) -> int:
        raise AssertionError("runner called")

    def postprocess(self, host_output: int) -> int:
        raise AssertionError("runner called")


class _Artifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.refs: list[str] = []

    def add_reference(self, uri: str) -> None:
        self.refs.append(uri)


class _Run:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.url = "https://wandb.ai/acme/ohho/runs/abc123"
        self.id = "abc123"
        self.entity = ""
        self.project = ""
        self.finished = False
        self.artifacts: list[_Artifact] = []

    def log(self, payload: dict[str, Any]) -> None:
        return None

    def log_artifact(self, artifact: _Artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True

    def get_url(self) -> str:
        return self.url


class _Wandb:
    Artifact = _Artifact

    def __init__(self) -> None:
        self.run = _Run()
        self.inits: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> _Run:
        self.inits.append(dict(kwargs))
        entity = kwargs.get("entity")
        if isinstance(entity, str):
            self.run.entity = entity
        project = kwargs.get("project")
        if isinstance(project, str):
            self.run.project = project
        return self.run


def _rich_document() -> dict[str, Any]:
    digest = "sha256:" + "ab" * 32
    return {
        "run": {
            "id": "opt-1",
            "model_sha": digest,
            "artifact_sha": digest,
            "target": "vision",
            "manifest_ref": "model.onnx",
        },
        "latency": {"mean": 10.0, "median": 9.0, "p95": 20.0, "unit": "ms"},
        "accuracy": {"primary": {"name": "success", "value": 0.9}},
        "telemetry": {"sensors": {"energy_per_inference_mj": 3.5, "power_w_mean": 8.0}},
        "environment": {"target_profile": "x86-cpu"},
    }


@contextmanager
def _gateway(fail_delete: bool = False) -> Iterator[_Gateway]:
    httpd = _Gateway(("127.0.0.1", 0), _Handler)
    httpd.events = []  # type: ignore[attr-defined]
    httpd.fail_delete = fail_delete  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=httpd.serve_forever,
        name="ohho-export-test",
        daemon=True,
    )
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _run(
    extra: list[str],
    *,
    export_context: ExportContext | None = None,
    runner: Any = None,
) -> tuple[int, dict[str, Any] | None, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "model.onnx"
        artifact.write_bytes(b"scripted")
        out = root / "out.json"
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
                "2",
                "--json",
                str(out),
                *extra,
            ]
        )
        clock = VirtualClock()
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run_bench_command(
                args,
                runner=runner or _Scaled(clock),
                clock=clock,
                sources=(),
                export_context=export_context,
            )
        document = None
        if out.is_file():
            document = json.loads(out.read_text(encoding="utf-8"))
        return code, document, stdout.getvalue(), stderr.getvalue()


class TestParse(unittest.TestCase):
    def test_specs_and_default_ttl(self) -> None:
        self.assertEqual(parse_spec("wandb"), ("wandb", ""))
        self.assertEqual(parse_spec("wandb:acme/ohho"), ("wandb", "acme/ohho"))
        self.assertEqual(
            parse_spec("prometheus=http://127.0.0.1:9091"),
            ("prometheus", "http://127.0.0.1:9091"),
        )
        self.assertEqual(
            parse_spec("prometheus=remote-write:http://127.0.0.1:9090/api/v1/write"),
            ("prometheus", "remote-write:http://127.0.0.1:9090/api/v1/write"),
        )
        args = build_parser().parse_args(
            [
                "bench",
                "--artifact",
                "m.onnx",
                "--target",
                "x86-cpu",
                "--export",
                "wandb:acme/ohho",
                "--export",
                "prometheus=http://127.0.0.1:9091",
            ]
        )
        self.assertEqual(
            args.export,
            ["wandb:acme/ohho", "prometheus=http://127.0.0.1:9091"],
        )
        self.assertEqual(args.export_ttl, 5.0)

    def test_discovery_names_the_adapters(self) -> None:
        schemes = {item.scheme for item in discover_exporters()}
        self.assertEqual(schemes, {"prometheus", "wandb"})


class TestWandbExport(unittest.TestCase):
    def test_bench_run_links_the_mocked_run(self) -> None:
        fake = _Wandb()
        old = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = _KEY
        try:
            code, document, _stdout, stderr = _run(
                ["--export", "wandb:acme/ohho"],
                export_context=ExportContext(module=fake),
            )
        finally:
            if old is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = old
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("warning:", stderr)
        self.assertIsNotNone(document)
        assert document is not None
        validate(document)
        self.assertEqual(len(fake.inits), 1)
        kwargs = fake.inits[0]
        self.assertEqual(kwargs["entity"], "acme")
        self.assertEqual(kwargs["project"], "ohho")
        self.assertEqual(kwargs["job_type"], "bench")
        self.assertNotIn("api_key", kwargs)
        config = kwargs["config"]
        self.assertEqual(config["kind"], "bench")
        self.assertEqual(config["spec"]["target"], "x86-cpu")
        self.assertEqual(config["spec"]["command"], "bench")
        self.assertTrue(str(config["manifest"]["manifest_ref"]).endswith("model.onnx"))
        self.assertTrue(
            any(ref.endswith("model.onnx") for ref in config["artifact_refs"])
        )
        self.assertTrue(any(ref.startswith("file:") for ref in config["artifact_refs"]))
        self.assertEqual(fake.run.artifacts[0].refs, config["artifact_refs"])
        self.assertIn("latency/p95", fake.run.summary)
        self.assertGreater(fake.run.summary["latency/p95"], 0)
        self.assertEqual(fake.run.summary["accuracy"], 0.0)
        self.assertEqual(fake.run.summary["accuracy_name"], "not_measured")
        self.assertNotIn("energy/mj", fake.run.summary)
        self.assertTrue(fake.run.finished)
        self.assertEqual(document["exports"]["wandb"]["url"], fake.run.url)
        self.assertEqual(document["exports"]["wandb"]["entity"], "acme")
        self.assertEqual(document["exports"]["wandb"]["id"], "abc123")
        rendered = json.dumps(document)
        self.assertIn(fake.run.url, rendered)
        self.assertNotIn(_KEY, rendered)
        self.assertNotIn(_KEY, json.dumps(fake.inits))
        self.assertNotIn(_KEY, stderr)
        self.assertIn(fake.run.url, render_markdown(document))

    def test_optimize_run_summary_omits_secrets(self) -> None:
        document = _rich_document()
        fake = _Wandb()
        warnings = export_result(
            document,
            ["wandb:acme/ohho"],
            kind="optimize",
            spec={
                "command": "optimize",
                "target": "x86-cpu",
                "api_key": _KEY,
                "token": "nope",
            },
            manifest={
                "manifest_ref": "model.onnx",
                "model_sha": document["run"]["model_sha"],
            },
            context=ExportContext(module=fake),
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(fake.inits), 1)
        self.assertEqual(fake.inits[0]["job_type"], "optimize")
        self.assertEqual(fake.run.summary["energy/mj"], 3.5)
        self.assertEqual(fake.run.summary["power/w"], 8.0)
        self.assertEqual(fake.run.summary["accuracy"], 0.9)
        self.assertEqual(fake.run.summary["latency/p95"], 20.0)
        spec = fake.inits[0]["config"]["spec"]
        self.assertEqual(spec["target"], "x86-cpu")
        self.assertNotIn("api_key", spec)
        self.assertNotIn("token", spec)
        self.assertIn("model.onnx", fake.inits[0]["config"]["artifact_refs"])
        self.assertEqual(document["exports"]["wandb"]["url"], fake.run.url)
        self.assertTrue(fake.run.finished)
        self.assertNotIn(_KEY, json.dumps(fake.inits))

        saved = {
            name: os.environ.get(name)
            for name in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY")
        }
        os.environ.pop("WANDB_PROJECT", None)
        os.environ.pop("WANDB_ENTITY", None)
        os.environ["WANDB_API_KEY"] = _KEY
        try:
            fresh_doc = _rich_document()
            fresh = _Wandb()
            export_result(
                fresh_doc,
                ["wandb"],
                kind="optimize",
                spec={"target": "x86-cpu"},
                context=ExportContext(module=fresh),
            )
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.assertEqual(fresh.inits[0]["project"], "ohho")
        self.assertNotIn("entity", fresh.inits[0])
        self.assertNotIn(_KEY, json.dumps(fresh.inits))

    def test_bad_ttl_does_not_open_a_run(self) -> None:
        fake = _Wandb()
        warnings = export_result(
            _rich_document(),
            ["wandb"],
            kind="bench",
            spec={},
            ttl_minutes=-1,
            context=ExportContext(module=fake),
        )
        self.assertTrue(warnings)
        self.assertEqual(fake.inits, [])


class TestPrometheusExport(unittest.TestCase):
    def test_pushgateway_deletes_the_group_after_ttl(self) -> None:
        slept: list[float] = []
        during: list[list[str]] = []
        with _gateway() as httpd:
            port = httpd.server_address[1]

            def sleeper(seconds: float) -> None:
                slept.append(seconds)
                during.append([item[0] for item in httpd.events])

            code, document, _stdout, stderr = _run(
                [
                    "--export",
                    f"prometheus=http://127.0.0.1:{port}",
                    "--export-ttl",
                    "2",
                ],
                export_context=ExportContext(sleeper=sleeper),
            )
            events = list(httpd.events)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("warning:", stderr)
        self.assertEqual(slept, [120.0])
        self.assertEqual(during, [["POST"]])
        self.assertEqual([item[0] for item in events], ["POST", "DELETE"])
        self.assertEqual(events[0][1], events[1][1])
        assert document is not None
        validate(document)
        link = document["exports"]["prometheus"]
        self.assertEqual(link["mode"], "pushgateway")
        self.assertTrue(link["deleted"])
        self.assertEqual(link["ttl_minutes"], 2.0)
        self.assertEqual(urlsplit(link["url"]).path, events[0][1])
        path = events[0][1]
        run_id = document["run"]["id"]
        model = document["run"]["model_sha"]
        self.assertIn("/job/ohho-bench/", path)
        self.assertIn("/target/x86-cpu/", path)
        self.assertIn(f"/run_id/{quote(run_id, safe='')}", path)
        self.assertIn(f"/model/{quote(model, safe='')}", path)
        text = events[0][2].decode("utf-8")
        self.assertIn("# TYPE ohho_latency_ms gauge", text)
        self.assertIn('job="ohho-bench"', text)
        self.assertIn("ohho_accuracy", text)
        headers = events[0][3]
        self.assertIn("text/plain", headers.get("content-type", ""))

    def test_optimize_push_includes_energy_and_a_failed_sibling(self) -> None:
        document = _rich_document()

        def boom() -> object:
            raise ImportError("no wandb")

        with _gateway() as httpd:
            port = httpd.server_address[1]
            warnings = export_result(
                document,
                [
                    "wandb:acme/ohho",
                    f"prometheus=pushgateway:http://127.0.0.1:{port}",
                ],
                kind="optimize",
                spec={"target": "x86-cpu", "model": "policy", "api_key": _KEY},
                ttl_minutes=0,
                context=ExportContext(loader=boom),
            )
            events = list(httpd.events)
        self.assertTrue(any("not installed" in item for item in warnings))
        self.assertNotIn("wandb", document["exports"])
        link = document["exports"]["prometheus"]
        self.assertEqual(link["mode"], "pushgateway")
        self.assertTrue(link["deleted"])
        self.assertEqual(link["ttl_minutes"], 0)
        self.assertEqual([item[0] for item in events], ["POST", "DELETE"])
        path = events[0][1]
        self.assertIn("/job/ohho-optimize/", path)
        self.assertIn("/model/policy/", path)
        self.assertIn("/target/x86-cpu/", path)
        text = events[0][2].decode("utf-8")
        self.assertIn("ohho_energy_mj{", text)
        self.assertTrue(
            any(
                line.startswith("ohho_energy_mj{") and line.endswith(" 3.5")
                for line in text.splitlines()
            )
        )
        self.assertIn("ohho_power_w{", text)
        self.assertIn('name="success"', text)
        self.assertNotIn(_KEY, text)
        self.assertNotIn(_KEY, json.dumps(document))

    def test_delete_failure_is_a_warning_after_the_ttl(self) -> None:
        document = _rich_document()
        slept: list[float] = []
        during: list[list[str]] = []
        with _gateway(fail_delete=True) as httpd:
            port = httpd.server_address[1]

            def sleeper(seconds: float) -> None:
                slept.append(seconds)
                during.append([item[0] for item in httpd.events])

            warnings = export_result(
                document,
                [f"prometheus=http://127.0.0.1:{port}"],
                kind="bench",
                spec={"target": "x86-cpu"},
                ttl_minutes=1,
                context=ExportContext(sleeper=sleeper),
            )
            events = list(httpd.events)
        self.assertTrue(any("HTTP 500" in item for item in warnings))
        self.assertEqual(slept, [60.0])
        self.assertEqual(during, [["POST"]])
        self.assertEqual([item[0] for item in events], ["POST", "DELETE"])
        self.assertFalse(document["exports"]["prometheus"]["deleted"])
        self.assertIn("delete_error", document["exports"]["prometheus"])

    def test_remote_write_posts_protobuf_and_does_not_delete(self) -> None:
        slept: list[float] = []
        with _gateway() as httpd:
            port = httpd.server_address[1]
            code, document, _stdout, stderr = _run(
                [
                    "--export",
                    f"prometheus=remote-write:http://127.0.0.1:{port}/api/v1/write",
                    "--export-ttl",
                    "9",
                ],
                export_context=ExportContext(sleeper=slept.append),
            )
            events = list(httpd.events)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("warning:", stderr)
        self.assertEqual(slept, [])
        self.assertEqual([item[0] for item in events], ["POST"])
        self.assertEqual(events[0][1], "/api/v1/write")
        headers = events[0][3]
        self.assertEqual(headers.get("content-type"), "application/x-protobuf")
        self.assertEqual(headers.get("content-encoding"), "snappy")
        self.assertEqual(headers.get("x-prometheus-remote-write-version"), "0.1.0")
        raw = snappy_decompress(events[0][2])
        self.assertEqual(raw[0], 0x0A)
        self.assertIn(b"ohho_latency_ms", raw)
        self.assertIn(b"__name__", raw)
        self.assertIn(b"ohho-bench", raw)
        assert document is not None
        self.assertEqual(document["exports"]["prometheus"]["mode"], "remote-write")
        self.assertNotIn("deleted", document["exports"]["prometheus"])
        validate(document)

    def test_snappy_roundtrip(self) -> None:
        for size in (0, 1, 60, 61, 200):
            payload = (bytes(range(256)) * 3)[:size]
            self.assertEqual(snappy_decompress(snappy_compress(payload)), payload)
        samples = [("ohho_latency_ms", {"job": "ohho-bench", "stat": "p95"}, 1.5)]
        raw = encode_write_request(samples, 1_000_000)
        self.assertEqual(snappy_decompress(snappy_compress(raw)), raw)
        self.assertIn(b"ohho_latency_ms", raw)


class TestExportWarnings(unittest.TestCase):
    def test_missing_package_does_not_fail_the_bench(self) -> None:
        def boom() -> object:
            raise ImportError("no wandb")

        code, document, _stdout, stderr = _run(
            ["--export", "wandb"],
            export_context=ExportContext(loader=boom),
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("warning:", stderr)
        self.assertIn("not installed", stderr)
        assert document is not None
        self.assertNotIn("exports", document)
        validate(document)

    def test_push_error_does_not_fail_the_bench(self) -> None:
        def opener(request: object, timeout: float = 5) -> object:
            raise OSError("connection refused")

        code, document, _stdout, stderr = _run(
            ["--export", "prometheus=http://127.0.0.1:9", "--export-ttl", "0"],
            export_context=ExportContext(opener=opener),
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("warning:", stderr)
        self.assertIn("connection refused", stderr)
        assert document is not None
        self.assertNotIn("exports", document)
        validate(document)

    def test_unknown_and_empty_specs_are_warnings(self) -> None:
        code, document, _stdout, stderr = _run(["--export", "not-an-exporter"])
        self.assertEqual(code, 0, stderr)
        self.assertIn("unknown exporter", stderr)
        assert document is not None
        validate(document)
        code, document, _stdout, stderr = _run(["--export", "   "])
        self.assertEqual(code, 0, stderr)
        self.assertIn("empty", stderr)

    def test_negative_ttl_exits_before_the_run(self) -> None:
        code, document, _stdout, stderr = _run(
            ["--export", "wandb", "--export-ttl", "-1"],
            runner=_Explode(),
        )
        self.assertEqual(code, 2)
        self.assertIn("export-ttl", stderr)
        self.assertIsNone(document)


class TestOptionalPackage(unittest.TestCase):
    def test_adapters_do_not_import_optional_clients(self) -> None:
        script = (
            "import ohho.bench\n"
            "import ohho.bench.exports.prometheus_export\n"
            "import ohho.bench.exports.wandb_export\n"
            "import sys\n"
            "assert 'wandb' not in sys.modules\n"
            "assert 'prometheus_client' not in sys.modules\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_real_wandb_import_skips_when_absent(self) -> None:
        try:
            import wandb
        except ImportError:
            self.skipTest("wandb is not installed")
        except Exception as exc:
            self.skipTest(f"wandb is not usable ({type(exc).__name__})")
        self.assertTrue(callable(getattr(wandb, "init", None)))


if __name__ == "__main__":
    unittest.main()
