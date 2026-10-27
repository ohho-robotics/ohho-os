"""SIGTERM calls stop within the grace period and records an overrun."""

from __future__ import annotations

import signal
import unittest

from ohho_runtime.service import (
    LifecycleState,
    ManualClock,
    ServiceBase,
    SigtermShutdown,
)


class ScriptedService(ServiceBase):
    def __init__(self, clock: ManualClock, **kwargs: float) -> None:
        super().__init__(clock=clock, **kwargs)
        self.manual = clock
        self.advance_on_stop = 0.0
        self.stops = 0

    def on_stop(self) -> None:
        self.stops += 1
        if self.advance_on_stop:
            self.manual.advance(self.advance_on_stop)


class SigtermShutdownTest(unittest.TestCase):
    def _started(self, clock: ManualClock, **kwargs: float) -> ScriptedService:
        service = ScriptedService(clock, **kwargs)
        service.configure({})
        service.mark_ready()
        service.start()
        self.assertIs(service.state, LifecycleState.READY)
        return service

    def test_sigterm_calls_stop_within_grace(self) -> None:
        service = self._started(ManualClock(), grace_s=1.0)
        handler = SigtermShutdown(service)
        handler.handle(signal.SIGTERM)
        self.assertEqual(service.stops, 1)
        self.assertIs(service.state, LifecycleState.STOPPED)
        self.assertEqual(len(service.shutdown_records), 1)
        record = service.shutdown_records[0]
        self.assertEqual(record.action, "sigterm")
        self.assertFalse(record.forced)
        self.assertEqual(record.signal, int(signal.SIGTERM))

    def test_stop_past_grace_is_recorded(self) -> None:
        clock = ManualClock()
        service = self._started(clock, grace_s=1.0)
        service.advance_on_stop = 2.0
        SigtermShutdown(service).handle(signal.SIGTERM)
        self.assertIs(service.state, LifecycleState.STOPPED)
        record = service.shutdown_records[0]
        self.assertEqual(record.action, "grace_exceeded")
        self.assertTrue(record.forced)
        self.assertEqual(record.grace_s, 1.0)
        self.assertIn("grace", record.detail)

    def test_second_sigterm_does_not_raise(self) -> None:
        service = self._started(ManualClock())
        handler = SigtermShutdown(service)
        handler.handle(signal.SIGTERM)
        handler.handle(signal.SIGTERM)
        self.assertEqual(service.stops, 1)
        self.assertEqual(len(service.shutdown_records), 2)
        self.assertIn("illegal", service.shutdown_records[1].detail)

    def test_other_signals_are_ignored(self) -> None:
        service = self._started(ManualClock())
        SigtermShutdown(service).handle(signal.SIGINT)
        self.assertIs(service.state, LifecycleState.READY)
        self.assertEqual(service.shutdown_records, ())

    def test_install_restores_the_previous_handler(self) -> None:
        service = self._started(ManualClock())
        handler = SigtermShutdown(service)
        previous = signal.getsignal(signal.SIGTERM)
        restore = handler.install()
        try:
            self.assertEqual(signal.getsignal(signal.SIGTERM), handler.handle)
        finally:
            restore()
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)
        self.assertIs(service.state, LifecycleState.READY)


if __name__ == "__main__":
    unittest.main()
