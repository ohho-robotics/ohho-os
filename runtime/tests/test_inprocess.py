"""In-process thread service: ready, heartbeat, and grace recording."""

from __future__ import annotations

import time
import unittest

from ohho_runtime.service import InProcessService, LifecycleState, Service


def _work(service: InProcessService) -> None:
    service.heartbeat()
    while not service.stopping():
        service.heartbeat()
        if service.stop_event.wait(0.01):
            return


def _hang(service: InProcessService) -> None:
    time.sleep(1.0)


class InProcessServiceTest(unittest.TestCase):
    def test_thread_becomes_ready_and_stops(self) -> None:
        service = InProcessService(
            _work, grace_s=1.0, liveness_timeout_s=2.0, readiness_timeout_s=2.0
        )
        self.assertIsInstance(service, Service)
        service.configure({})
        service.start()
        self.assertIs(service.state, LifecycleState.READY)
        self.assertTrue(service.ready())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not service.live():
            time.sleep(0.01)
        self.assertTrue(service.live())
        health = service.health()
        self.assertTrue(health.ready)
        self.assertTrue(health.live)
        service.stop()
        self.assertIs(service.state, LifecycleState.STOPPED)
        self.assertFalse(any(record.forced for record in service.shutdown_records))

    def test_thread_that_misses_grace_is_recorded(self) -> None:
        service = InProcessService(
            _hang, grace_s=0.2, liveness_timeout_s=5.0, readiness_timeout_s=5.0
        )
        service.configure({})
        service.start()
        self.assertIs(service.state, LifecycleState.READY)
        service.stop()
        self.assertIs(service.state, LifecycleState.STOPPED)
        forced = [
            record
            for record in service.shutdown_records
            if record.action == "grace_exceeded"
        ]
        self.assertEqual(len(forced), 1)
        self.assertTrue(forced[0].forced)
        self.assertEqual(forced[0].grace_s, 0.2)


if __name__ == "__main__":
    unittest.main()
