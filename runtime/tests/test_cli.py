"""Operator CLI against a supervisor of in-process sim services.

The control channel is loopback TCP. These tests do not sleep on a
hardware clock beyond a short poll while the foreground command starts,
and they do not open a non-loopback socket.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ohho.robot_spec import RobotClass

from ohho_runtime.cli import main
from ohho_runtime.control import (
    ControlError,
    matching_log_lines,
    require_loopback_host,
    supervisor_report,
)
from ohho_runtime.log import LOG_FORMAT_ENV, LOG_LEVEL_ENV, RUN_ID_ENV
from ohho_runtime.manifest import (
    ManifestError,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    load_manifest,
    make_service,
)
from ohho_runtime.manifest_file import load_manifest_document
from ohho_runtime.metrics import (
    METRICS_ENABLED_ENV,
    METRICS_HOST_ENV,
    METRICS_PORT_ENV,
    Registry,
)
from ohho_runtime.service import LifecycleState, ManualClock
from ohho_runtime.supervisor import Supervisor, SupervisorError, SupervisorState

_ENV_KEYS = (
    LOG_LEVEL_ENV,
    LOG_FORMAT_ENV,
    RUN_ID_ENV,
    METRICS_ENABLED_ENV,
    METRICS_HOST_ENV,
    METRICS_PORT_ENV,
)


def _noop() -> None:
    return None


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(argv, out=stdout, err=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _svc(
    role: str,
    entry: str,
    *,
    depends: tuple[str, ...] = (),
    criticality: str = "critical",
    restart: str = "never",
    commands_motion: bool = False,
    only_disarmed: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "role": role,
        "entry": entry,
        "depends_on": list(depends),
        "criticality": criticality,
        "readiness_timeout_s": 30,
        "liveness_timeout_s": 30,
        "loop_hz": 1,
        "commands_motion": commands_motion,
    }
    if restart == "never":
        body["restart"] = "never"
    else:
        body["restart"] = {
            "kind": "on-failure",
            "max_restarts": 3,
            "window_s": 60,
            "backoff_s": 0.5,
            "only_while_disarmed": only_disarmed,
        }
    return body


def _document(
    services: list[dict[str, object]],
    robot_class: str = "mobile_base",
) -> dict[str, object]:
    return {
        "robot_class": robot_class,
        "log_level": "INFO",
        "log_format": "json",
        "services": services,
    }


def _core(
    telemetry: str = "sim.ready",
    extra: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    services = [
        _svc("safety_watchdog", "sim.ready"),
        _svc(
            "transport",
            "sim.ready",
            depends=("safety_watchdog",),
            commands_motion=True,
            restart="on-failure",
            only_disarmed=True,
        ),
        _svc("telemetry", telemetry, depends=("transport",)),
        *extra,
    ]
    return _document(services)


def _records(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AssertionError("log line is not an object")
        rows.append(value)
    return rows


class _Up:
    """Foreground ``up`` on a background thread, stopped on exit."""

    def __init__(self, test: unittest.TestCase, argv: list[str]) -> None:
        self.test = test
        self.path = test.tmp / "control.json"  # type: ignore[attr-defined]
        self.argv = [
            "up",
            "--state",
            str(self.path),
            "--poll-interval",
            "0.02",
            *argv,
        ]
        self.box: dict[str, object] = {}
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.down_code: int | None = None
        self.down_out = ""
        self.down_err = ""
        self.thread = threading.Thread(target=self._run, name="ohho-runtime-up")
        self.thread.daemon = True

    def _run(self) -> None:
        try:
            self.box["code"] = main(self.argv, out=self.out, err=self.err)
        except BaseException as exc:
            self.box["code"] = 1
            self.box["exc"] = repr(exc)

    def __enter__(self) -> _Up:
        self.thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.path.is_file():
                code, _out, _err = _invoke(
                    ["status", "--state", str(self.path), "--json"]
                )
                if code == 0:
                    return self
            if not self.thread.is_alive():
                break
            time.sleep(0.02)
        self.test.fail(
            "supervisor did not come up: "
            f"{self.box} stdout={self.out.getvalue()} stderr={self.err.getvalue()}"
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self.thread.is_alive() and self.path.is_file():
            self.down_code, self.down_out, self.down_err = _invoke(
                ["down", "--state", str(self.path)]
            )
        self.thread.join(5)
        if exc_type is not None:
            return False
        if self.thread.is_alive():
            self.test.fail("up command did not exit")
        if self.box.get("code") != 0:
            self.test.fail(
                f"up exited {self.box.get('code')}: "
                f"{self.err.getvalue()} {self.out.getvalue()} {self.box.get('exc')}"
            )
        if self.down_code not in (0, None):
            self.test.fail(f"down exited {self.down_code}: {self.down_err}")
        return False


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._saved = {key: os.environ.get(key) for key in _ENV_KEYS}
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LogFilterTest(unittest.TestCase):
    def test_service_and_level_filter(self) -> None:
        records = [
            (
                1,
                json.dumps(
                    {
                        "level": "INFO",
                        "service": "transport",
                        "event": "service_start",
                    }
                ),
            ),
            (
                2,
                json.dumps(
                    {
                        "level": "ERROR",
                        "service": "transport",
                        "event": "service_failed",
                    }
                ),
            ),
            (
                3,
                json.dumps(
                    {
                        "level": "INFO",
                        "service": "telemetry",
                        "event": "service_start",
                    }
                ),
            ),
        ]
        lines, cursor = matching_log_lines(
            records, service="transport", level="ERROR", tail=10, after=0
        )
        self.assertEqual(cursor, 3)
        self.assertEqual(len(lines), 1)
        self.assertIn("service_failed", lines[0])
        info, _cursor = matching_log_lines(
            records, service="", level="INFO", tail=1, after=0
        )
        self.assertEqual(len(info), 1)
        self.assertIn("telemetry", info[0])
        fresh, _cursor = matching_log_lines(
            records, service="", level="", tail=10, after=2
        )
        self.assertEqual(len(fresh), 1)
        self.assertIn("telemetry", fresh[0])
        empty, _cursor = matching_log_lines(
            records, service="transport", level="INFO", tail=0, after=0
        )
        self.assertEqual(empty, [])

    def test_text_line_filter(self) -> None:
        line = (
            "2026-01-01T00:00:00.000000Z INFO "
            "service=transport event=service_start "
            "robot_class=mobile_base run_id=abc"
        )
        lines, _cursor = matching_log_lines(
            [(1, line)], service="transport", level="WARNING", tail=10, after=0
        )
        self.assertEqual(lines, [])
        lines, _cursor = matching_log_lines(
            [(1, line)], service="transport", level="INFO", tail=10, after=0
        )
        self.assertEqual(lines, [line])


class RestartApiTest(CliTestCase):
    def test_restart_respects_never_policy_and_disarm(self) -> None:
        watchdog = make_service(
            "safety_watchdog",
            PythonCallable(_noop),
            restart=RestartPolicy(RestartKind.NEVER),
            loop_hz=1.0,
            readiness_timeout_s=30.0,
            liveness_timeout_s=30.0,
            qos=("state",),
        )
        transport = make_service(
            "transport",
            PythonCallable(_noop),
            depends_on=("safety_watchdog",),
            commands_motion=True,
            restart=RestartPolicy(
                RestartKind.ON_FAILURE,
                max_restarts=3,
                window_s=60.0,
                backoff_s=0.5,
                only_while_disarmed=True,
            ),
            loop_hz=1.0,
            readiness_timeout_s=30.0,
            liveness_timeout_s=30.0,
            qos=("command", "state"),
        )
        telemetry = make_service(
            "telemetry",
            PythonCallable(_noop),
            depends_on=("transport",),
            restart=RestartPolicy(RestartKind.NEVER),
            loop_hz=1.0,
            readiness_timeout_s=30.0,
            liveness_timeout_s=30.0,
            qos=("telemetry",),
        )
        manifest = load_manifest((watchdog, transport, telemetry))
        clock = ManualClock()
        supervisor = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            metrics=Registry(),
            metrics_enabled=False,
        )
        try:
            supervisor.start()
            with self.assertRaises(SupervisorError) as never:
                supervisor.restart("safety_watchdog")
            self.assertIn("never", str(never.exception))
            self.assertEqual(supervisor.restart_count("safety_watchdog"), 0)
            supervisor.set_disarmed(False)
            with self.assertRaises(SupervisorError) as armed:
                supervisor.restart("transport")
            self.assertIn("disarmed", str(armed.exception))
            self.assertEqual(supervisor.restart_count("transport"), 0)
            self.assertFalse(supervisor.disarmed)
            supervisor.set_disarmed(True)
            supervisor.restart("transport")
            supervisor.restart("transport")
            self.assertEqual(supervisor.restart_count("transport"), 2)
            self.assertEqual(
                supervisor.service("transport").state, LifecycleState.READY
            )
            self.assertTrue(supervisor.disarmed)
            self.assertFalse(supervisor.do_not_operate)
            self.assertEqual(supervisor.events.of("human_clear"), ())
            report = supervisor_report(supervisor, manifest)
            self.assertEqual(report["healthy"], True)
            rows = report["services"]
            self.assertIsInstance(rows, list)
            transport_row = next(
                row
                for row in rows
                if isinstance(row, dict) and row["service"] == "transport"
            )
            self.assertEqual(transport_row["restarts"], 2)
            self.assertEqual(transport_row["state"], "ready")
            self.assertIsInstance(transport_row["heartbeat_age_s"], float)
        finally:
            supervisor.shutdown()
            self.assertEqual(supervisor.state, SupervisorState.STOPPED)


class ManifestFileTest(unittest.TestCase):
    def test_readme_example_loads(self) -> None:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        text = readme.read_text(encoding="utf-8")
        match = re.search(r"```json\n(\{.*?\n\})\n```", text, re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        manifest, robot_class = load_manifest_document(json.loads(match.group(1)))
        self.assertEqual(robot_class, RobotClass.MOBILE_BASE)
        self.assertEqual(
            set(manifest.roles()),
            {"safety_watchdog", "transport", "telemetry"},
        )

    def test_unknown_entry_is_refused(self) -> None:
        with self.assertRaises(ManifestError) as caught:
            load_manifest_document(
                {
                    "robot_class": "mobile_base",
                    "services": [
                        {
                            "role": "safety_watchdog",
                            "entry": "os:system",
                            "loop_hz": 1,
                            "restart": "never",
                        }
                    ],
                }
            )
        self.assertIn("os:system", str(caught.exception))

    def test_console_script_is_declared(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('ohho-runtime = "ohho_runtime.cli:main"', text)


class ControlChannelTest(CliTestCase):
    def test_host_must_be_loopback(self) -> None:
        self.assertEqual(require_loopback_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(require_loopback_host("LocalHost"), "127.0.0.1")
        self.assertEqual(require_loopback_host("::1"), "::1")
        for host in ("0.0.0.0", "8.8.8.8", "example.com"):
            with self.assertRaises(ControlError) as caught:
                require_loopback_host(host)
            self.assertIn("loopback", str(caught.exception))

    def test_up_refuses_non_loopback_host(self) -> None:
        code, _out, err = _invoke(
            [
                "up",
                "--class",
                "mobile_base",
                "--host",
                "0.0.0.0",
                "--state",
                str(self.tmp / "control.json"),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("loopback", err)
        self.assertFalse((self.tmp / "control.json").exists())

    def test_status_refuses_non_loopback_state_file(self) -> None:
        path = self.tmp / "control.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "host": "8.8.8.8",
                    "port": 9,
                    "pid": 1,
                    "run_id": "abc",
                    "robot_class": "mobile_base",
                }
            ),
            encoding="utf-8",
        )
        code, _out, err = _invoke(["status", "--state", str(path), "--json"])
        self.assertEqual(code, 1)
        self.assertIn("loopback", err)


class RuntimeCliTest(CliTestCase):
    def _manifest(self, document: dict[str, object]) -> Path:
        path = self.tmp / "manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_up_status_health_logs_restart_down(self) -> None:
        manifest = self._manifest(_core())
        with _Up(self, ["--manifest", str(manifest)]) as up:
            code, out, err = _invoke(["status", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            status = json.loads(out)
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["robot_class"], "mobile_base")
            self.assertEqual(status["do_not_operate"], False)
            self.assertEqual(status["disarmed"], True)
            self.assertEqual(status["healthy"], True)
            by_role = {row["service"]: row for row in status["services"]}
            self.assertEqual(
                set(by_role),
                {"safety_watchdog", "transport", "telemetry"},
            )
            for row in by_role.values():
                self.assertEqual(row["state"], "ready")
                self.assertEqual(row["restarts"], 0)
                self.assertEqual(row["health"], "ok")
                self.assertGreaterEqual(row["heartbeat_age_s"], 0)
            state = json.loads(up.path.read_text(encoding="utf-8"))
            self.assertEqual(state["host"], "127.0.0.1")
            self.assertEqual(state["version"], 1)
            self.assertIn("127.0.0.1", up.out.getvalue())

            code, out, err = _invoke(["health", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["healthy"], True)
            code, out, _err = _invoke(["health", "--state", str(up.path)])
            self.assertEqual(code, 0)
            self.assertEqual(out.strip(), "healthy")

            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "transport",
                    "--level",
                    "info",
                ]
            )
            self.assertEqual(code, 0, err)
            transport_logs = _records(out)
            self.assertTrue(transport_logs)
            self.assertTrue(
                all(row["service"] == "transport" for row in transport_logs)
            )
            self.assertTrue(
                any(row["event"] == "service_start" for row in transport_logs)
            )
            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "transport",
                    "--level",
                    "error",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(out.strip(), "")
            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "telemetry",
                    "--level",
                    "info",
                ]
            )
            self.assertEqual(code, 0, err)
            telemetry_logs = _records(out)
            self.assertTrue(
                all(row["service"] == "telemetry" for row in telemetry_logs)
            )
            self.assertTrue(telemetry_logs)

            code, _out, err = _invoke(
                ["restart", "safety_watchdog", "--state", str(up.path)]
            )
            self.assertEqual(code, 1)
            self.assertIn("never", err)

            code, out, err = _invoke(["restart", "transport", "--state", str(up.path)])
            self.assertEqual(code, 0, err)
            self.assertIn("restarts=1", out)
            code, out, err = _invoke(["status", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            status = json.loads(out)
            transport = next(
                row for row in status["services"] if row["service"] == "transport"
            )
            self.assertEqual(transport["restarts"], 1)
            self.assertEqual(transport["state"], "ready")
            self.assertEqual(status["disarmed"], True)
            self.assertEqual(status["do_not_operate"], False)
            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "transport",
                    "--level",
                    "info",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertTrue(any(row["event"] == "restart" for row in _records(out)))

            code, out, err = _invoke(["status", "--state", str(up.path)])
            self.assertEqual(code, 0, err)
            self.assertIn("transport", out)
            self.assertIn("ready", out)

        self.assertEqual(up.down_code, 0)
        self.assertIn("stopped", up.down_out)
        self.assertFalse(up.path.exists())
        code, _out, err = _invoke(["status", "--state", str(up.path), "--json"])
        self.assertEqual(code, 1)
        self.assertIn("not running", err)
        self.assertEqual(up.box.get("code"), 0)

    def test_health_fails_when_a_critical_service_fails(self) -> None:
        manifest = self._manifest(_core(telemetry="sim.fail"))
        with _Up(self, ["--manifest", str(manifest)]) as up:
            code, out, err = _invoke(["status", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            status = json.loads(out)
            telemetry = next(
                row for row in status["services"] if row["service"] == "telemetry"
            )
            self.assertEqual(telemetry["state"], "failed")
            self.assertEqual(telemetry["health"], "failed")
            watchdog = next(
                row for row in status["services"] if row["service"] == "safety_watchdog"
            )
            self.assertEqual(watchdog["state"], "ready")
            code, out, _err = _invoke(["health", "--state", str(up.path), "--json"])
            self.assertEqual(code, 1)
            health = json.loads(out)
            self.assertEqual(health["healthy"], False)
            self.assertIn("telemetry", health["unhealthy"])
            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "telemetry",
                    "--level",
                    "error",
                ]
            )
            self.assertEqual(code, 0, err)
            errors = _records(out)
            self.assertTrue(errors)
            self.assertTrue(all(row["service"] == "telemetry" for row in errors))
            self.assertTrue(all(row["level"] == "ERROR" for row in errors))
            code, out, err = _invoke(
                [
                    "logs",
                    "--state",
                    str(up.path),
                    "--service",
                    "safety_watchdog",
                    "--level",
                    "error",
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(out.strip(), "")

    def test_degradable_failure_keeps_health_ok(self) -> None:
        extra = (
            _svc(
                "perception",
                "sim.fail",
                depends=("telemetry",),
                criticality="degradable",
            ),
        )
        manifest = self._manifest(_core(extra=extra))
        with _Up(self, ["--manifest", str(manifest)]) as up:
            code, out, err = _invoke(["health", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            health = json.loads(out)
            self.assertEqual(health["healthy"], True)
            self.assertNotIn("perception", health["unhealthy"])
            perception = next(
                row for row in health["services"] if row["service"] == "perception"
            )
            self.assertEqual(perception["health"], "failed")
            self.assertEqual(perception["criticality"], "degradable")

    def test_second_up_is_refused(self) -> None:
        manifest = self._manifest(_core())
        with _Up(self, ["--manifest", str(manifest)]) as up:
            code, _out, err = _invoke(
                [
                    "up",
                    "--manifest",
                    str(manifest),
                    "--state",
                    str(up.path),
                    "--poll-interval",
                    "0.05",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("already running", err)
            code, out, err = _invoke(["health", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["healthy"], True)

    def test_class_default_sim_services(self) -> None:
        with _Up(self, ["--class", "mobile_base"]) as up:
            code, out, err = _invoke(["status", "--state", str(up.path), "--json"])
            self.assertEqual(code, 0, err)
            status = json.loads(out)
            self.assertEqual(status["state"], "running")
            roles = {row["service"] for row in status["services"]}
            self.assertEqual(
                roles,
                {
                    "safety_watchdog",
                    "transport",
                    "telemetry",
                    "teleop_gateway",
                },
            )
            self.assertTrue(all(row["state"] == "ready" for row in status["services"]))
            code, _out, err = _invoke(["health", "--state", str(up.path)])
            self.assertEqual(code, 0, err)
            code, out, err = _invoke(
                ["restart", "teleop_gateway", "--state", str(up.path)]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("restarts=1", out)

    def test_bad_manifest_does_not_listen(self) -> None:
        path = self.tmp / "manifest.json"
        path.write_text("{", encoding="utf-8")
        code, _out, err = _invoke(
            [
                "up",
                "--manifest",
                str(path),
                "--state",
                str(self.tmp / "control.json"),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("JSON", err)
        self.assertFalse((self.tmp / "control.json").exists())

    def test_unknown_log_level_is_an_error(self) -> None:
        manifest = self._manifest(_core())
        with _Up(self, ["--manifest", str(manifest)]) as up:
            code, _out, err = _invoke(
                ["logs", "--state", str(up.path), "--level", "nope"]
            )
            self.assertEqual(code, 1)
            self.assertIn("level", err.lower())


if __name__ == "__main__":
    unittest.main()
