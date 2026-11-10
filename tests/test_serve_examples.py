"""OHH-491: CPU Serve examples, placeholders, and the release gate.

Scripts that need torch, onnxruntime, git, a GPU, or a vendor sensor are not
imported here. Pending commands stay placeholders and print no measurements.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "examples" / "serve"
_METRICS = re.compile(
    r"p50|p90|p95|p99|tokens/s|frames/s|latency_ms|throughput",
    re.IGNORECASE,
)
_TOKEN_KEYS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_REPO",
    "GITHUB_REPOSITORY",
    "GITHUB_OUTPUT",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_block(text: str) -> str:
    start = text.index("## Expected output")
    fence = text.index("```text", start)
    body_start = text.index("\n", fence) + 1
    body_end = text.index("\n```", body_start)
    return text[body_start:body_end].replace("\r\n", "\n").strip("\n")


def _run(
    script: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    if env is None:
        env = os.environ.copy()
        root = str(ROOT)
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not previous else os.pathsep.join((root, previous))
    return subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        env=env,
        timeout=120,
    )


class TestServeExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.support = _load(SERVE / "support.py", "ohho_serve_example_support_test")
        cls.probe = _load(SERVE / "gpu_runner_probe.py", "ohho_serve_gpu_probe_test")

    def test_readme_contract_matches_script(self) -> None:
        for example in self.support.load_manifest():
            readme = (SERVE / example["readme"]).read_text(encoding="utf-8")
            for heading in ("## Hardware", "## Runtime", "## Expected output"):
                self.assertIn(heading, readme, example["id"])
            expected = _expected_block(readme)
            completed = _run(SERVE / example["script"])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = completed.stdout.replace("\r\n", "\n").strip("\n")
            self.assertEqual(actual, expected, example["id"])
            self.assertNotRegex(actual, _METRICS)
            notebook = json.loads(
                (SERVE / example["notebook"]).read_text(encoding="utf-8")
            )
            markdown = []
            for cell in notebook["cells"]:
                if cell.get("cell_type") != "markdown":
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                markdown.append(source)
            joined = "".join(markdown)
            self.assertIn("## Hardware", joined, example["id"])
            self.assertIn("## Runtime", joined, example["id"])
            self.assertIn(expected, joined, example["id"])

    def test_pending_features_are_absent_and_block_when_present(self) -> None:
        support = self.support
        self.assertTrue(support.feature_ready("cli:optimize"))
        self.assertTrue(support.feature_ready("ohho.optimize"))
        self.assertTrue(support.feature_ready("bench:llm"))
        self.assertTrue(support.feature_ready("bench:launch"))
        self.assertTrue(support.feature_ready("export:metrics"))
        self.assertTrue(support.feature_ready("cli:bench"))
        self.assertEqual(
            support.bench_commands(),
            {"validate", "llm", "launch", "version"},
        )
        buffer = StringIO()
        with redirect_stdout(buffer):
            code = support.emit_placeholder(
                example_id="sample",
                hardware="cpu",
                pending="sample feature",
                checks=("cli:bench",),
            )
        text = buffer.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("status: blocked", text)
        self.assertIn("result: none", text)
        self.assertNotRegex(text, _METRICS)

    def test_cpu_and_gpu_runners(self) -> None:
        cpu = _run(SERVE / "run_cpu.py")
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        self.assertIn("serve-examples: ok", cpu.stdout)
        self.assertIn("--- ci-gate notebook", cpu.stdout)
        gpu = _run(SERVE / "run_gpu.py")
        self.assertEqual(gpu.returncode, 0, gpu.stderr)
        self.assertIn("serve-examples-gpu: ok", gpu.stdout)
        self.assertIn("--- act-jetson-fp16", gpu.stdout)
        self.assertNotIn("--- ci-gate", gpu.stdout)

    def test_runner_fails_when_a_script_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.py"
            broken.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
            healthy = Path(tmp) / "healthy.py"
            healthy.write_text("print('ok')\n", encoding="utf-8")
            self.assertEqual(self.support.run_scripts([healthy]), 0)
            self.assertEqual(self.support.run_scripts([broken]), 1)
            self.assertEqual(self.support.run_scripts([]), 1)

    def test_release_gate_text(self) -> None:
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        gpu = (ROOT / ".github" / "workflows" / "serve-examples-gpu.yml").read_text(
            encoding="utf-8"
        )
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        snippet = (SERVE / "ci_gate" / "release-job.yml").read_text(encoding="utf-8")
        self.assertEqual(self.support.release_text_problems(release), [])
        self.assertEqual(self.support.gpu_workflow_problems(gpu), [])
        self.assertIn("python examples/serve/run_cpu.py", ci)
        self.assertIn("python examples/serve/run_cpu.py", snippet)
        self.assertIn("needs: [serve-examples]", snippet)
        missing = "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        problems = self.support.release_text_problems(missing)
        self.assertTrue(problems)

    def test_runner_probe_is_local_logic(self) -> None:
        probe = self.probe
        self.assertFalse(probe.runner_available({}))
        self.assertFalse(
            probe.runner_available(
                {
                    "runners": [
                        {
                            "status": "offline",
                            "labels": [{"name": "linux"}, {"name": "gpu"}],
                        }
                    ]
                }
            )
        )
        self.assertFalse(
            probe.runner_available(
                {"runners": [{"status": "online", "labels": [{"name": "gpu"}]}]}
            )
        )
        self.assertTrue(
            probe.runner_available(
                {
                    "runners": [
                        {
                            "status": "online",
                            "labels": ["self-hosted", "Linux", "GPU"],
                        }
                    ]
                }
            )
        )
        env = os.environ.copy()
        for key in _TOKEN_KEYS:
            env.pop(key, None)
        completed = _run(SERVE / "gpu_runner_probe.py", env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("available=false", completed.stdout)
        self.assertNotIn("http", completed.stdout.lower())

    def test_example_scripts_stay_free_of_heavy_imports(self) -> None:
        banned = (
            "import torch",
            "onnxruntime",
            "import wandb",
            "urllib",
            "socket",
            "subprocess",
        )
        for example in self.support.load_manifest():
            text = (SERVE / example["script"]).read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, example["id"])
