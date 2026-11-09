"""Every legal and illegal lifecycle edge, and separate readiness/liveness."""

from __future__ import annotations

import unittest

from ohho_runtime.service import (
    LEGAL_TRANSITIONS,
    HealthStatus,
    IllegalTransition,
    LifecycleState,
    ManualClock,
    Service,
    ServiceBase,
    ServiceError,
    move,
)


class ScriptedService(ServiceBase):
    """Test double whose hooks succeed or fail on flags. No threads."""

    def __init__(self, clock: ManualClock, **kwargs: float) -> None:
        super().__init__(clock=clock, **kwargs)
        self.manual = clock
        self.fail_configure = False
        self.fail_stop = False
        self.advance_on_stop = 0.0
        self.configured = 0

    def on_configure(self, config: object) -> None:
        self.configured += 1
        if self.fail_configure:
            raise RuntimeError("configure boom")

    def on_stop(self) -> None:
        if self.advance_on_stop:
            self.manual.advance(self.advance_on_stop)
        if self.fail_stop:
            raise RuntimeError("stop boom")

    def force(self, state: LifecycleState) -> None:
        """Place the machine in ``state`` without checking the edge."""
        with self._lock:
            self._state = state


def _edges(
    history: tuple[LifecycleState, ...],
) -> set[tuple[LifecycleState, LifecycleState]]:
    return set(zip(history, history[1:]))


class TransitionMatrixTest(unittest.TestCase):
    def test_every_pair_is_accepted_or_rejected(self) -> None:
        states = list(LifecycleState)
        self.assertEqual(len(states), 7)
        covered = 0
        for src in states:
            for dst in states:
                covered += 1
                service = ScriptedService(ManualClock())
                service.force(src)
                if (src, dst) in LEGAL_TRANSITIONS:
                    service._transition(dst)
                    self.assertIs(service.state, dst)
                else:
                    with self.assertRaises(IllegalTransition) as caught:
                        service._transition(dst)
                    self.assertIs(caught.exception.src, src)
                    self.assertIs(caught.exception.dst, dst)
                    self.assertIs(service.state, src)
        self.assertEqual(covered, len(states) * len(states))

    def test_move_matches_the_table(self) -> None:
        for src in LifecycleState:
            for dst in LifecycleState:
                if (src, dst) in LEGAL_TRANSITIONS:
                    self.assertIs(move(src, dst), dst)
                else:
                    with self.assertRaises(IllegalTransition):
                        move(src, dst)

    def test_public_paths_cover_every_legal_transition(self) -> None:
        seen: set[tuple[LifecycleState, LifecycleState]] = set()
        for history in _public_histories():
            seen.update(_edges(history))
        self.assertEqual(seen, LEGAL_TRANSITIONS)

    def test_public_methods_reject_illegal_calls(self) -> None:
        service = ScriptedService(ManualClock())
        with self.assertRaises(IllegalTransition):
            service.start()
        self.assertIs(service.state, LifecycleState.CREATED)
        with self.assertRaises(IllegalTransition):
            service.stop()
        self.assertIs(service.state, LifecycleState.CREATED)

        service.configure({})
        service.mark_ready()
        service.start()
        self.assertIs(service.state, LifecycleState.READY)
        with self.assertRaises(IllegalTransition):
            service.configure({})
        with self.assertRaises(IllegalTransition):
            service.start()
        self.assertIs(service.state, LifecycleState.READY)
        service.stop()
        with self.assertRaises(IllegalTransition):
            service.stop()
        self.assertIs(service.state, LifecycleState.STOPPED)

    def test_base_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(ScriptedService(ManualClock()), Service)


class ProbeSeparationTest(unittest.TestCase):
    def test_starting_service_can_be_live_and_not_ready(self) -> None:
        clock = ManualClock()
        service = ScriptedService(clock, liveness_timeout_s=10, readiness_timeout_s=10)
        service.configure({})
        service.start()
        service.heartbeat()
        self.assertIs(service.state, LifecycleState.STARTING)
        self.assertFalse(service.ready())
        self.assertTrue(service.live())
        health = service.health()
        self.assertIs(health.status, HealthStatus.DEGRADED)
        self.assertEqual(health.reason, "not ready")
        self.assertFalse(health.ready)
        self.assertTrue(health.live)

    def test_readiness_timeout_fails_with_a_reason_while_still_live(self) -> None:
        clock = ManualClock()
        service = ScriptedService(clock, readiness_timeout_s=5, liveness_timeout_s=100)
        service.configure({})
        service.start()
        service.heartbeat()
        clock.advance(5)
        self.assertTrue(service.live())
        self.assertFalse(service.ready())
        service.poll()
        self.assertIs(service.state, LifecycleState.FAILED)
        self.assertEqual(service.failure_reason, "readiness timeout")
        health = service.health()
        self.assertIs(health.status, HealthStatus.FAILED)
        self.assertEqual(health.reason, "readiness timeout")
        self.assertFalse(health.ready)
        self.assertTrue(health.live)

    def test_liveness_timeout_fails_while_still_ready(self) -> None:
        clock = ManualClock()
        service = ScriptedService(clock, readiness_timeout_s=100, liveness_timeout_s=5)
        service.configure({})
        service.mark_ready()
        service.start()
        service.heartbeat()
        self.assertIs(service.state, LifecycleState.READY)
        clock.advance(5)
        self.assertTrue(service.ready())
        self.assertFalse(service.live())
        service.poll()
        self.assertIs(service.state, LifecycleState.FAILED)
        self.assertEqual(service.failure_reason, "liveness timeout")
        health = service.health()
        self.assertIs(health.status, HealthStatus.FAILED)
        self.assertEqual(health.reason, "liveness timeout")
        self.assertTrue(health.ready)
        self.assertFalse(health.live)

    def test_degraded_keeps_ready_and_live(self) -> None:
        clock = ManualClock()
        service = ScriptedService(clock)
        service.configure({})
        service.mark_ready()
        service.heartbeat()
        service.start()
        self.assertIs(service.health().status, HealthStatus.OK)
        service.degrade("sensor gap")
        health = service.health()
        self.assertIs(service.state, LifecycleState.READY)
        self.assertIs(health.status, HealthStatus.DEGRADED)
        self.assertEqual(health.reason, "sensor gap")
        self.assertTrue(health.ready)
        self.assertTrue(health.live)
        service.clear_degraded()
        self.assertIs(service.health().status, HealthStatus.OK)

    def test_degrade_before_ready_is_rejected(self) -> None:
        service = ScriptedService(ManualClock())
        service.configure({})
        with self.assertRaises(ServiceError):
            service.degrade("too soon")
        with self.assertRaises(ValueError):
            service.degrade("  ")


def _configured(clock: ManualClock | None = None, **kwargs: float) -> ScriptedService:
    service = ScriptedService(clock or ManualClock(), **kwargs)
    service.configure({})
    return service


def _public_histories() -> list[tuple[LifecycleState, ...]]:
    histories: list[tuple[LifecycleState, ...]] = []

    created = ScriptedService(ManualClock())
    created.fail_configure = True
    try:
        created.configure({})
    except RuntimeError:
        pass
    histories.append(created.history)

    reconfigured = _configured()
    reconfigured.configure({"again": True})
    histories.append(reconfigured.history)

    started = _configured()
    started.start()
    histories.append(started.history)

    stopped_early = _configured()
    stopped_early.stop()
    histories.append(stopped_early.history)

    configure_failed = _configured()
    configure_failed.fail_configure = True
    try:
        configure_failed.configure({})
    except RuntimeError:
        pass
    histories.append(configure_failed.history)

    ready = _configured()
    ready.mark_ready()
    ready.start()
    histories.append(ready.history)

    clock = ManualClock()
    timed_out = _configured(clock, readiness_timeout_s=5, liveness_timeout_s=100)
    timed_out.start()
    timed_out.heartbeat()
    clock.advance(5)
    timed_out.poll()
    histories.append(timed_out.history)

    stopped_starting = _configured()
    stopped_starting.start()
    stopped_starting.stop()
    histories.append(stopped_starting.history)

    clock = ManualClock()
    heartbeat_dead = _configured(clock, readiness_timeout_s=100, liveness_timeout_s=5)
    heartbeat_dead.mark_ready()
    heartbeat_dead.start()
    heartbeat_dead.heartbeat()
    clock.advance(5)
    heartbeat_dead.poll()
    histories.append(heartbeat_dead.history)

    stopped_ready = _configured()
    stopped_ready.mark_ready()
    stopped_ready.start()
    stopped_ready.stop()
    histories.append(stopped_ready.history)

    stop_failed = _configured()
    stop_failed.mark_ready()
    stop_failed.start()
    stop_failed.fail_stop = True
    try:
        stop_failed.stop()
    except RuntimeError:
        pass
    histories.append(stop_failed.history)

    clock = ManualClock()
    failed_then_stopped = _configured(
        clock, readiness_timeout_s=5, liveness_timeout_s=100
    )
    failed_then_stopped.start()
    failed_then_stopped.heartbeat()
    clock.advance(5)
    failed_then_stopped.poll()
    failed_then_stopped.stop()
    histories.append(failed_then_stopped.history)

    restarted = _configured()
    restarted.stop()
    restarted.configure({})
    histories.append(restarted.history)

    return histories


if __name__ == "__main__":
    unittest.main()
