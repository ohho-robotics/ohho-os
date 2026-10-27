"""Subprocess heartbeat, graceful SIGTERM, and SIGKILL after grace."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from ohho_runtime.heartbeat import (
    IGNORE_TERM_ENV,
    NEVER_READY_ENV,
    STARTED_ENV,
    split_lines,
)
from ohho_runtime.service import (
    LifecycleState,
    SubprocessService,
    UnixSocketUnavailable,
    child_env,
    stop_child,
    unix_sockets_available,
)

_ROOT = Path(__file__).resolve().parents[2]
_WORKER = [sys.executable, "-m", "ohho_runtime.worker"]
_POSIX_SOCKETS = os.name == "posix" and unix_sockets_available()


class FakeChild:
    """Process double. ``ignore`` means the graceful stop does not exit."""

    def __init__(self, *, ignore: bool) -> None:
        self.ignore = ignore
        self.returncode: int | None = None
        self.signals: list[object] = []
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout or 0)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if not self.ignore:
            self.returncode = 0

    def terminate(self) -> None:
        self.signals.append("terminate")
        if not self.ignore:
            self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -int(getattr(signal, "SIGKILL", 9))


class StopChildTest(unittest.TestCase):
    def test_ignored_graceful_stop_is_killed_and_recorded(self) -> None:
        proc = FakeChild(ignore=True)
        record = stop_child(proc, grace_s=0.25)
        self.assertEqual(record.action, "sigkill")
        self.assertTrue(record.forced)
        self.assertEqual(record.grace_s, 0.25)
        self.assertEqual(record.signal, int(getattr(signal, "SIGKILL", 9)))
        self.assertIn("SIGKILL", record.detail)
        self.assertTrue(proc.killed)
        self.assertTrue(proc.signals)

    def test_child_that_exits_is_not_killed(self) -> None:
        proc = FakeChild(ignore=False)
        record = stop_child(proc, grace_s=0.25)
        self.assertEqual(record.action, "sigterm")
        self.assertFalse(record.forced)
        self.assertFalse(proc.killed)
        self.assertEqual(record.signal, int(signal.SIGTERM))

    def test_already_exited_child_is_recorded(self) -> None:
        proc = FakeChild(ignore=True)
        proc.returncode = 0
        record = stop_child(proc, grace_s=0.25)
        self.assertEqual(record.action, "exited")
        self.assertFalse(record.forced)
        self.assertFalse(proc.killed)


class HeartbeatFramingTest(unittest.TestCase):
    def test_split_lines_keeps_a_partial_frame(self) -> None:
        pending, lines = split_lines(b"", b"ready\nbe")
        self.assertEqual(lines, (b"ready",))
        pending, lines = split_lines(pending, b"at\n")
        self.assertEqual(pending, b"")
        self.assertEqual(lines, (b"beat",))


class ConsoleScriptTest(unittest.TestCase):
    def test_distribution_declares_the_worker_script(self) -> None:
        text = (_ROOT / "runtime" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "ohho-runtime"', text)
        self.assertIn(
            'ohho-runtime-service = "ohho_runtime.worker:main"',
            text,
        )

    def test_worker_process_exits_within_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            started = directory / "started"
            err_path = directory / "err.txt"
            env = child_env({STARTED_ENV: str(started)})
            with err_path.open("wb") as err_fh:
                proc = subprocess.Popen(
                    _WORKER,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=err_fh,
                )
            try:
                deadline = time.monotonic() + 10
                while not started.is_file():
                    if proc.poll() is not None:
                        detail = err_path.read_text(encoding="utf-8", errors="replace")
                        self.fail(f"worker exited {proc.returncode}: {detail}")
                    if time.monotonic() > deadline:
                        self.fail("worker did not start")
                    time.sleep(0.02)
                record = stop_child(proc, grace_s=2.0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=2)
            self.assertEqual(
                record.action, "sigterm", err_path.read_text("utf-8", "replace")
            )
            self.assertFalse(record.forced)
            self.assertIsNotNone(proc.poll())


@unittest.skipUnless(not unix_sockets_available(), "this platform has AF_UNIX")
class NoUnixSocketTest(unittest.TestCase):
    def test_start_fails_cleanly_without_a_socket(self) -> None:
        service = SubprocessService(_WORKER, grace_s=0.5)
        service.configure({})
        with self.assertRaises(UnixSocketUnavailable):
            service.start()
        self.assertIs(service.state, LifecycleState.FAILED)
        self.assertIn("Unix domain sockets", service.failure_reason)
        self.assertEqual(service.shutdown_records, ())


@unittest.skipUnless(_POSIX_SOCKETS, "POSIX AF_UNIX is required")
class SubprocessSocketTest(unittest.TestCase):
    def _service(self, **env: str) -> SubprocessService:
        return SubprocessService(
            _WORKER,
            extra_env=env,
            grace_s=0.4,
            readiness_timeout_s=15,
            liveness_timeout_s=15,
        )

    def _wait_until(self, service: SubprocessService, predicate) -> None:  # type: ignore[no-untyped-def]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            service.poll()
            if predicate():
                return
            if service.state is LifecycleState.FAILED:
                self.fail(
                    f"failed early: {service.failure_reason} {service.stderr_text()}"
                )
            time.sleep(0.02)
        self.fail(
            f"timed out state={service.state.value} stderr={service.stderr_text()}"
        )

    def test_heartbeat_readiness_and_graceful_stop(self) -> None:
        service = self._service()
        try:
            service.configure({})
            service.start()
            self._wait_until(
                service,
                lambda: service.state is LifecycleState.READY and service.live(),
            )
            self.assertTrue(service.ready())
            self.assertGreater(service.beats, 0)
            health = service.health()
            self.assertTrue(health.ready)
            self.assertTrue(health.live)
            service.stop()
        finally:
            service.close()
        self.assertIs(service.state, LifecycleState.STOPPED)
        self.assertTrue(
            any(
                r.action == "sigterm" and not r.forced for r in service.shutdown_records
            )
        )
        self.assertFalse(any(r.action == "sigkill" for r in service.shutdown_records))

    def test_never_ready_stays_live(self) -> None:
        service = self._service(**{NEVER_READY_ENV: "1"})
        try:
            service.configure({})
            service.start()
            self._wait_until(service, lambda: service.live() and service.beats > 0)
            self.assertFalse(service.ready())
            self.assertIs(service.state, LifecycleState.STARTING)
            health = service.health()
            self.assertFalse(health.ready)
            self.assertTrue(health.live)
        finally:
            service.close()

    def test_subprocess_that_ignores_sigterm_is_sigkilled_and_recorded(self) -> None:
        grace = 0.4
        service = SubprocessService(
            _WORKER,
            extra_env={IGNORE_TERM_ENV: "1"},
            grace_s=grace,
            readiness_timeout_s=15,
            liveness_timeout_s=15,
        )
        try:
            service.configure({})
            service.start()
            self._wait_until(service, lambda: service.state is LifecycleState.READY)
            started = time.monotonic()
            service.stop()
            elapsed = time.monotonic() - started
        finally:
            service.close()
        self.assertGreaterEqual(elapsed, grace - 0.05)
        self.assertLess(elapsed, grace + 5)
        kills = [r for r in service.shutdown_records if r.action == "sigkill"]
        self.assertEqual(len(kills), 1, service.shutdown_records)
        self.assertTrue(kills[0].forced)
        self.assertEqual(kills[0].grace_s, grace)
        self.assertEqual(kills[0].signal, int(signal.SIGKILL))
        self.assertIn("SIGKILL", kills[0].detail)


if __name__ == "__main__":
    unittest.main()
