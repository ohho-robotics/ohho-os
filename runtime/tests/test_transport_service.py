"""TransportService: connect is start; SafetyGate state freshness is liveness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ohho.safety import AuditLog, SafetyGate, SafetyProfile
from ohho.schema import ConnectionState, Telemetry, TransportStatus
from ohho.transport import BaseTransport

from ohho_runtime.service import (
    HealthStatus,
    LifecycleState,
    ManualClock,
    TransportService,
)


class FakeLink(BaseTransport):
    protocol = "simulated"

    def __init__(self) -> None:
        super().__init__()
        self.connected = False
        self.disconnects = 0
        self.stops = 0
        self._state = ConnectionState.IDLE

    def connect(self) -> TransportStatus:
        self.connected = True
        self._state = ConnectionState.CONNECTED
        status = self.status()
        self._emit_status(status)
        return status

    def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = False
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self._state, label="fake")

    def read(self) -> Telemetry:
        return Telemetry()

    def send_velocity(self, vel: object) -> None:
        return None

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def emergency_stop(self) -> None:
        self.stops += 1

    def release_stop(self) -> None:
        return None

    def push(self, sample: Telemetry) -> None:
        self._emit_telemetry(sample)


class IdleLink(FakeLink):
    def connect(self) -> TransportStatus:
        self.connected = True
        self._state = ConnectionState.IDLE
        status = self.status()
        self._emit_status(status)
        return status


class TransportServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.services: list[TransportService] = []

    def tearDown(self) -> None:
        for service in self.services:
            if (service.state, LifecycleState.STOPPING) in _STOPPABLE:
                try:
                    service.stop()
                except Exception:
                    pass

    def _track(self, service: TransportService) -> TransportService:
        self.services.append(service)
        return service

    def _gate(
        self, clock: ManualClock, profile: SafetyProfile
    ) -> tuple[FakeLink, SafetyGate]:
        inner = FakeLink()
        gate = SafetyGate(
            inner,
            profile=profile,
            audit=AuditLog(Path(self._tmp.name)),
            clock=clock,
            auto_tick=False,
            environ={},
        )
        return inner, gate

    def test_plain_transport_connects_on_start_and_uses_telemetry(self) -> None:
        clock = ManualClock()
        inner = FakeLink()
        service = self._track(
            TransportService(
                inner,
                clock=clock,
                readiness_timeout_s=5,
                liveness_timeout_s=0.5,
            )
        )
        service.configure({})
        service.start()
        self.assertTrue(inner.connected)
        self.assertIs(service.state, LifecycleState.READY)
        self.assertTrue(service.ready())
        self.assertFalse(service.live())
        inner.push(Telemetry())
        self.assertTrue(service.live())
        self.assertTrue(service.ready())
        clock.advance(0.5)
        self.assertFalse(service.live())
        self.assertTrue(service.ready())
        service.stop()
        self.assertEqual(inner.disconnects, 1)
        self.assertIs(service.state, LifecycleState.STOPPED)

    def test_link_that_never_connects_fails_readiness_while_live(self) -> None:
        clock = ManualClock()
        inner = IdleLink()
        service = self._track(
            TransportService(
                inner,
                clock=clock,
                readiness_timeout_s=5,
                liveness_timeout_s=100,
            )
        )
        service.configure({})
        service.start()
        self.assertTrue(inner.connected)
        self.assertIs(service.state, LifecycleState.STARTING)
        self.assertFalse(service.ready())
        service.heartbeat()
        clock.advance(5)
        service.poll()
        self.assertIs(service.state, LifecycleState.FAILED)
        self.assertEqual(service.failure_reason, "readiness timeout")
        health = service.health()
        self.assertFalse(health.ready)
        self.assertTrue(health.live)
        self.assertEqual(health.reason, "readiness timeout")

    def test_gate_liveness_matches_state_freshness(self) -> None:
        clock = ManualClock()
        profile = SafetyProfile(state_timeout_s=0.25)
        inner, gate = self._gate(clock, profile)
        window = gate.profile.state_timeout_s
        service = self._track(
            TransportService(
                gate,
                clock=clock,
                readiness_timeout_s=5,
                liveness_timeout_s=5,
            )
        )
        service.configure({})
        service.start()
        self.assertTrue(inner.connected)
        self.assertEqual(service.liveness_timeout_s, window)
        self.assertIs(service.state, LifecycleState.READY)
        self.assertTrue(service.ready())
        self.assertFalse(service.live())
        self.assertFalse(gate.armed)
        self.assertFalse(gate.latched)
        self._agree(service, gate, clock)

        inner.push(Telemetry())
        self.assertTrue(service.live())
        self.assertTrue(service.ready())
        self.assertIs(service.health().status, HealthStatus.OK)
        self._agree(service, gate, clock)

        clock.advance(0.125)
        self.assertTrue(service.live())
        self._agree(service, gate, clock)
        clock.advance(0.125)
        self.assertFalse(service.live())
        self.assertTrue(service.ready())
        self._agree(service, gate, clock)
        self.assertFalse(gate.armed)
        self.assertFalse(gate.latched)
        self.assertEqual(gate.profile.state_timeout_s, window)
        self.assertEqual(inner.stops, 0)

        service.poll()
        self.assertIs(service.state, LifecycleState.FAILED)
        self.assertEqual(service.failure_reason, "liveness timeout")
        health = service.health()
        self.assertIs(health.status, HealthStatus.FAILED)
        self.assertEqual(health.reason, "liveness timeout")
        self.assertTrue(health.ready)
        self.assertFalse(health.live)
        self._agree(service, gate, clock)

    def test_missing_orientation_does_not_refresh_the_gate(self) -> None:
        clock = ManualClock()
        profile = SafetyProfile(state_timeout_s=0.25, require_orientation=True)
        inner, gate = self._gate(clock, profile)
        service = self._track(TransportService(gate, clock=clock, liveness_timeout_s=5))
        service.configure({})
        service.start()
        inner.push(Telemetry())
        self.assertFalse(service.live())
        self.assertFalse(gate.latched)
        self.assertEqual(inner.stops, 0)
        self._agree(service, gate, clock)
        inner.push(Telemetry(custom={"roll": 0.0, "pitch": 0.0}))
        self.assertTrue(service.live())
        self.assertFalse(gate.latched)
        self._agree(service, gate, clock)

    def test_faulted_sample_is_not_fresh_and_the_wrapper_does_not_command(self) -> None:
        clock = ManualClock()
        profile = SafetyProfile(state_timeout_s=0.25)
        inner, gate = self._gate(clock, profile)
        service = self._track(TransportService(gate, clock=clock, liveness_timeout_s=5))
        service.configure({})
        service.start()
        # The gate itself latches on a faulted sample. The wrapper only observes.
        inner.push(Telemetry(custom={"error_code": 1}))
        self.assertTrue(gate.latched)
        self.assertFalse(gate.armed)
        self.assertFalse(service.live())
        self._agree(service, gate, clock)
        self.assertGreaterEqual(inner.stops, 1)

    def _agree(
        self, service: TransportService, gate: SafetyGate, clock: ManualClock
    ) -> None:
        # The test may read the gate's freshness helper. The runtime must not.
        self.assertEqual(service.live(), not gate._state_stale(clock()))


_STOPPABLE = {
    (LifecycleState.CONFIGURED, LifecycleState.STOPPING),
    (LifecycleState.STARTING, LifecycleState.STOPPING),
    (LifecycleState.READY, LifecycleState.STOPPING),
    (LifecycleState.FAILED, LifecycleState.STOPPING),
}


if __name__ == "__main__":
    unittest.main()
