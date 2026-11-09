"""Structured log shape, level config, and exception notes."""

from __future__ import annotations

import io
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ohho.robot_spec import RobotClass

from ohho_runtime.log import (
    LOG_FIELDS,
    LOG_FORMAT_ENV,
    LOG_LEVEL_ENV,
    RuntimeLog,
    normalize_level,
    note_exception,
    resolve_log_format,
    resolve_log_level,
    resolve_run_id,
)
from ohho_runtime.manifest import (
    ManifestError,
    PythonCallable,
    RestartPolicy,
    load_manifest,
    make_service,
    with_services,
)
from ohho_runtime.metrics import Registry
from ohho_runtime.service import ManualClock
from ohho_runtime.supervisor import Supervisor

_README = Path(__file__).resolve().parents[1] / "README.md"
_WHEN = datetime(2026, 10, 21, 3, 21, tzinfo=timezone.utc)


def _noop() -> None:
    return None


def _service(role: str = "safety_watchdog"):
    return make_service(
        role,
        PythonCallable(_noop),
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy(),
        loop_hz=1.0,
    )


def _logger(
    buf: io.StringIO, *, level: str = "INFO", log_format: str = "json"
) -> RuntimeLog:
    return RuntimeLog(
        service="supervisor",
        robot_class="mobile_base",
        run_id="run-1",
        level=level,
        log_format=log_format,
        stream=buf,
        clock=lambda: _WHEN,
    )


class LogRecordTest(unittest.TestCase):
    def test_json_line_has_stable_fields(self) -> None:
        buf = io.StringIO()
        record = _logger(buf).log("INFO", event="service_start", message="up")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["timestamp"], "2026-10-21T03:21:00.000000Z")
        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["service"], "supervisor")
        self.assertEqual(record["event"], "service_start")
        self.assertEqual(record["robot_class"], "mobile_base")
        self.assertEqual(record["run_id"], "run-1")
        parsed = json.loads(buf.getvalue())
        self.assertTrue(set(LOG_FIELDS).issubset(parsed))
        self.assertEqual(parsed["message"], "up")

    def test_text_mode_is_human_readable(self) -> None:
        buf = io.StringIO()
        _logger(buf, log_format="text").log(
            "WARNING",
            event="heartbeat_miss",
            service="telemetry",
            message="late",
        )
        line = buf.getvalue().strip()
        self.assertFalse(line.startswith("{"))
        self.assertIn("2026-10-21T03:21:00.000000Z", line)
        self.assertIn("WARNING", line)
        self.assertIn("service=telemetry", line)
        self.assertIn("event=heartbeat_miss", line)
        self.assertIn("robot_class=mobile_base", line)
        self.assertIn("run_id=run-1", line)
        self.assertIn("message=late", line)

    def test_level_filters_lower_records(self) -> None:
        buf = io.StringIO()
        log = _logger(buf, level="WARNING")
        self.assertIsNone(log.log("DEBUG", event="hidden_debug"))
        self.assertIsNone(log.log("INFO", event="hidden_info"))
        shown = log.log("ERROR", event="shown")
        self.assertIsNotNone(shown)
        text = buf.getvalue()
        self.assertNotIn("hidden_debug", text)
        self.assertNotIn("hidden_info", text)
        self.assertIn("shown", text)

    def test_supervisor_records_use_the_same_shape(self) -> None:
        buf = io.StringIO()
        logger = _logger(buf)
        sup = Supervisor(
            load_manifest((_service(),)),
            robot_class=RobotClass.MOBILE_BASE,
            clock=ManualClock(),
            logger=logger,
            run_id="run-1",
            metrics_enabled=False,
        )
        sup.start()
        sup.shutdown()
        lines = [line for line in buf.getvalue().splitlines() if line]
        self.assertGreaterEqual(len(lines), 2)
        services = set()
        events = set()
        for line in lines:
            record = json.loads(line)
            self.assertTrue(set(LOG_FIELDS).issubset(record))
            self.assertEqual(record["robot_class"], "mobile_base")
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["timestamp"], "2026-10-21T03:21:00.000000Z")
            services.add(record["service"])
            events.add(record["event"])
        self.assertIn("supervisor", services)
        self.assertIn("safety_watchdog", services)
        self.assertIn("service_start", events)
        self.assertIn("supervisor_start", events)


class LogLevelConfigTest(unittest.TestCase):
    def test_resolve_prefers_env_then_manifest_then_default(self) -> None:
        self.assertEqual(resolve_log_level(None, {}), "INFO")
        self.assertEqual(resolve_log_level("debug", {}), "DEBUG")
        self.assertEqual(
            resolve_log_level("DEBUG", {LOG_LEVEL_ENV: "error"}),
            "ERROR",
        )
        self.assertEqual(normalize_level("warn"), "WARNING")
        with self.assertRaises(ValueError):
            normalize_level("loud")

    def test_resolve_format(self) -> None:
        self.assertEqual(resolve_log_format(None, {}), "json")
        self.assertEqual(resolve_log_format("human", {}), "text")
        self.assertEqual(
            resolve_log_format("json", {LOG_FORMAT_ENV: "text"}),
            "text",
        )

    def test_run_id_uses_explicit_then_env(self) -> None:
        self.assertEqual(resolve_run_id(" given ", {}), "given")
        self.assertEqual(
            resolve_run_id(None, {"OHHO_RUNTIME_RUN_ID": "from-env"}),
            "from-env",
        )
        generated = resolve_run_id(None, {})
        self.assertTrue(generated)
        self.assertNotEqual(generated, resolve_run_id(None, {}))

    def test_manifest_stores_canonical_level_and_format(self) -> None:
        manifest = load_manifest(
            (_service(),),
            log_level="warning",
            log_format="human",
        )
        self.assertEqual(manifest.log_level, "WARNING")
        self.assertEqual(manifest.log_format, "text")
        copied = with_services(manifest, _service("recorder"))
        self.assertEqual(copied.log_level, "WARNING")
        self.assertEqual(copied.log_format, "text")
        with self.assertRaises(ManifestError):
            load_manifest((_service(),), log_level="loud")

    def test_env_overrides_manifest_on_the_supervisor(self) -> None:
        key = LOG_LEVEL_ENV
        previous = os.environ.get(key)
        os.environ[key] = "debug"
        try:
            sup = Supervisor(
                load_manifest((_service(),), log_level="ERROR"),
                robot_class=RobotClass.MOBILE_BASE,
                metrics_enabled=False,
            )
            self.assertEqual(sup.logger.level, "DEBUG")
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def test_manifest_level_and_format_apply_when_env_is_unset(self) -> None:
        level_key = LOG_LEVEL_ENV
        format_key = LOG_FORMAT_ENV
        saved = {key: os.environ.get(key) for key in (level_key, format_key)}
        os.environ.pop(level_key, None)
        os.environ.pop(format_key, None)
        try:
            sup = Supervisor(
                load_manifest((_service(),), log_level="ERROR", log_format="text"),
                robot_class=RobotClass.ARM,
                metrics_enabled=False,
            )
            self.assertEqual(sup.logger.level, "ERROR")
            self.assertEqual(sup.logger.log_format, "text")
            self.assertEqual(sup.logger.robot_class, "arm")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_readme_documents_log_fields(self) -> None:
        text = _README.read_text(encoding="utf-8")
        for field in LOG_FIELDS:
            self.assertIn(f"`{field}`", text)
        self.assertIn("OHHO_RUNTIME_LOG_LEVEL", text)
        self.assertIn("OHHO_RUNTIME_LOG_FORMAT", text)
        self.assertIn("OHHO_RUNTIME_RUN_ID", text)


class NoteExceptionTest(unittest.TestCase):
    def test_note_logs_context_and_counts(self) -> None:
        buf = io.StringIO()
        registry = Registry()
        logger = _logger(buf)
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            note_exception(
                exc,
                service="telemetry",
                event="worker_failed",
                logger=logger,
                metrics=registry,
            )
        record = json.loads(buf.getvalue())
        self.assertEqual(record["service"], "telemetry")
        self.assertEqual(record["event"], "worker_failed")
        self.assertEqual(record["exc_type"], "RuntimeError")
        self.assertIn("boom", str(record["message"]))
        self.assertEqual(record["robot_class"], "mobile_base")
        self.assertEqual(record["run_id"], "run-1")
        text = registry.render()
        self.assertIn("ohho_runtime_exceptions_total", text)
        self.assertIn('event="worker_failed"', text)
        self.assertIn('exc_type="RuntimeError"', text)
        self.assertIn('service="telemetry"', text)

    def test_filtered_note_is_still_counted(self) -> None:
        buf = io.StringIO()
        registry = Registry()
        try:
            raise OSError("missing")
        except OSError as exc:
            note_exception(
                exc,
                service="heartbeat",
                event="socket_probe_failed",
                level="DEBUG",
                logger=_logger(buf, level="INFO"),
                metrics=registry,
            )
        self.assertEqual(buf.getvalue(), "")
        self.assertIn('event="socket_probe_failed"', registry.render())


if __name__ == "__main__":
    unittest.main()
