"""Smoke-test every CLI command documented in ``docs/cli.md`` (sim / mock only)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ohho.cli import main

_SIM = ("--transport", "sim://", "--runtime", "native")


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCliSmokeCore(unittest.TestCase):
    """Commands that need no extras beyond the base install."""

    def test_doctor(self):
        code, out, _ = _run_cli(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("OhhO OS", out)

    def test_list(self):
        code, out, _ = _run_cli(["list"])
        self.assertEqual(code, 0)
        self.assertIn("omnibot", out)

    def test_version(self):
        code, out, _ = _run_cli(["version"])
        self.assertEqual(code, 0)
        self.assertIn("OhhO OS", out)

    def test_version_flag(self):
        code, out, _ = _run_cli(["-V"])
        self.assertEqual(code, 0)
        self.assertIn("OhhO OS", out)

    def test_connect_sim(self):
        code, out, _ = _run_cli(["connect", "omnibot", *_SIM])
        self.assertEqual(code, 0)
        self.assertIn("connected:", out)

    def test_sim(self):
        code, out, _ = _run_cli(["sim", "--robot", "omnibot", "--seconds", "1"])
        self.assertEqual(code, 0)
        self.assertIn("simulating", out)

    def test_drive_sim(self):
        code, out, _ = _run_cli(
            ["drive", "omnibot", *_SIM, "--seconds", "0.5", "--vx", "0.1"]
        )
        self.assertEqual(code, 0)
        self.assertIn("driving", out)

    def test_agent_sim(self):
        code, out, _ = _run_cli(["agent", "omnibot", *_SIM, "stop and report status"])
        self.assertEqual(code, 0)
        combined = out
        self.assertTrue(combined.strip(), "agent should print log lines")

    def test_skills_list(self):
        code, out, _ = _run_cli(["skills", "list"])
        self.assertEqual(code, 0)
        self.assertIn("patrol", out)

    def test_market_list_alias(self):
        code, out, _ = _run_cli(["market", "list"])
        self.assertEqual(code, 0)
        self.assertIn("patrol", out)

    def test_skills_run_sim(self):
        code, out, _ = _run_cli(["skills", "run", "omnibot", *_SIM, "status"])
        self.assertEqual(code, 0)
        self.assertIn("skill 'status'", out)

    def test_profile_list(self):
        code, out, _ = _run_cli(["profile", "list"])
        self.assertEqual(code, 0)
        self.assertIn("workstation_single", out)

    def test_profile_show(self):
        code, out, _ = _run_cli(["profile", "show", "workstation_single"])
        self.assertEqual(code, 0)
        self.assertIn("profile:", out)

    def test_profile_detect(self):
        code, out, _ = _run_cli(["profile", "detect"])
        self.assertEqual(code, 0)
        self.assertIn("detected:", out)

    def test_nav_map_sim(self):
        code, out, _ = _run_cli(["nav", "map", "omnibot", *_SIM])
        self.assertEqual(code, 0)
        self.assertIn("cells:", out)

    def test_nav_explore_sim(self):
        code, _, _ = _run_cli(
            [
                "nav",
                "explore",
                "omnibot",
                *_SIM,
                "--frontiers",
                "1",
                "--world",
                "demo",
            ]
        )
        self.assertEqual(code, 0)

    def test_nav_goto_sim(self):
        code, _, _ = _run_cli(
            [
                "nav",
                "goto",
                "omnibot",
                *_SIM,
                "--x",
                "0.5",
                "--y",
                "0.0",
                "--timeout",
                "8",
                "--world",
                "demo",
            ]
        )
        self.assertIn(code, (0, 3), "goto may time out without reaching goal")

    def test_look_and_memory_sim(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)

            def run_in_home(argv: list[str]) -> tuple[int, str, str]:
                prev_home = os.environ.get("HOME")
                prev_prof = os.environ.get("USERPROFILE")
                os.environ["HOME"] = str(home)
                os.environ["USERPROFILE"] = str(home)
                try:
                    return _run_cli(argv)
                finally:
                    if prev_home is not None:
                        os.environ["HOME"] = prev_home
                    else:
                        os.environ.pop("HOME", None)
                    if prev_prof is not None:
                        os.environ["USERPROFILE"] = prev_prof
                    else:
                        os.environ.pop("USERPROFILE", None)

            code, out, _ = run_in_home(["look", "omnibot", *_SIM, "--world", "demo"])
            self.assertEqual(code, 0, out)
            mem_file = home / ".ohho" / "memory" / "omnibot.json"
            self.assertTrue(mem_file.exists(), "look should persist memory")

            code, out, _ = run_in_home(["memory", "show", "omnibot"])
            self.assertEqual(code, 0, out)

            code, out, _ = run_in_home(
                ["memory", "near", "omnibot", "--x", "0.0", "--y", "0.0"]
            )
            self.assertEqual(code, 0, out)

            code, _, _ = run_in_home(["memory", "clear", "omnibot"])
            self.assertEqual(code, 0)

    def test_sim_demo_sim_fast(self):
        # The harness binds its log stream at import, so redirect_stdout
        # does not capture the demo transcript. Exit code is the smoke check.
        code, _out, err = _run_cli(["sim-demo", "--sim", "--fast"])
        self.assertEqual(code, 0, err)

    def test_selftest_sim(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            code, out, _ = _run_cli(["selftest", "--sim", "--out", str(report)])
            self.assertEqual(code, 0, out)
            self.assertTrue(report.is_file())
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertIn("overall_status", data)

    def test_sim_serve_not_built(self):
        for backend in ("isaac", "mujoco"):
            code, out, err = _run_cli(["sim-serve", "--backend", backend])
            self.assertNotEqual(code, 0)
            self.assertIn("not built", out + err)


@unittest.skipUnless(
    _has_fastapi(), "fastapi not installed — pip install 'ohho-os[serve]'"
)
class TestCliSmokeServe(unittest.TestCase):
    """``ohho serve --mock`` and HTTP /health, /load_model, /predict."""

    def test_serve_mock_endpoints(self):
        port = _free_port()
        host = "127.0.0.1"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ohho.cli",
                "serve",
                "--mock",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = f"http://{host}:{port}"
        deadline = time.monotonic() + 15.0
        last_err: Exception | None = None
        try:
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(f"{base}/health", timeout=1) as resp:
                        if resp.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError) as e:
                    last_err = e
                    time.sleep(0.2)
            else:
                self.fail(f"serve did not become ready: {last_err}")

            with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode())
                self.assertEqual(body.get("status"), "ok")

            load_req = urllib.request.Request(
                f"{base}/load_model?model_path=mock-checkpoint",
                method="POST",
            )
            with urllib.request.urlopen(load_req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)

            predict_body = json.dumps(
                {"instruction": "smoke test", "image_base64": ""}
            ).encode()
            predict_req = urllib.request.Request(
                f"{base}/predict",
                data=predict_body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(predict_req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                pred = json.loads(resp.read().decode())
                self.assertIn("vector", pred.get("action", {}))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestCliSmokeSimLoop(unittest.TestCase):
    """``ohho sim-loop --mock`` with minimal work (documented in README)."""

    def test_sim_loop_mock_minimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = _run_cli(
                [
                    "sim-loop",
                    "--mock",
                    "--robot",
                    "omnibot",
                    "--episodes",
                    "1",
                    "--steps-per-episode",
                    "3",
                    "--train-steps",
                    "5",
                    "--eval-steps",
                    "3",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 0, err + out)
            self.assertTrue((Path(tmp) / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
