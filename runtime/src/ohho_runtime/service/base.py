"""Service protocol and the base that enforces the lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ohho_runtime.service.lifecycle import (
    LEGAL_TRANSITIONS,
    Clock,
    Health,
    HealthStatus,
    IllegalTransition,
    LifecycleState,
    ServiceError,
    ShutdownRecord,
    move,
)


@runtime_checkable
class Service(Protocol):
    """Lifecycle surface every supervised service exposes."""

    @property
    def state(self) -> LifecycleState: ...

    def configure(self, config: Mapping[str, Any] | None = None) -> None: ...

    def start(self) -> None: ...

    def ready(self) -> bool: ...

    def stop(self) -> None: ...

    def health(self) -> Health: ...

    def heartbeat(self) -> None: ...


class ServiceBase:
    """State machine for ``configure / start / ready / stop``.

    Subclasses implement :meth:`on_configure`, :meth:`on_start`, and
    :meth:`on_stop`. :meth:`poll` is what commits a missed readiness timeout
    or a missed heartbeat into :attr:`LifecycleState.FAILED`. Readiness
    (:meth:`ready`) and liveness (:meth:`live`) do not share a flag.

    The host calls :meth:`poll`. A supervisor is a later layer; this class
    does not arm a robot and does not start a hidden timer thread.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        readiness_timeout_s: float = 5.0,
        liveness_timeout_s: float = 1.0,
        grace_s: float = 1.0,
    ) -> None:
        for name, value in (
            ("readiness_timeout_s", readiness_timeout_s),
            ("liveness_timeout_s", liveness_timeout_s),
            ("grace_s", grace_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number > 0")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        self._clock: Clock = time.monotonic if clock is None else clock
        self._readiness_timeout_s = float(readiness_timeout_s)
        self._liveness_timeout_s = float(liveness_timeout_s)
        self._grace_s = float(grace_s)
        self._lock = threading.RLock()
        self._state = LifecycleState.CREATED
        self._history: list[LifecycleState] = [LifecycleState.CREATED]
        self._config: dict[str, Any] = {}
        self._explicit_ready = False
        self._degraded_reason = ""
        self._failure_reason = ""
        self._starting_at: float | None = None
        self._last_heartbeat_at: float | None = None
        self._shutdown: list[ShutdownRecord] = []

    @property
    def state(self) -> LifecycleState:
        with self._lock:
            return self._state

    @property
    def history(self) -> tuple[LifecycleState, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def config(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._config)

    @property
    def failure_reason(self) -> str:
        with self._lock:
            return self._failure_reason

    @property
    def grace_s(self) -> float:
        return self._grace_s

    @property
    def pid(self) -> int | None:
        """OS pid when this service is its own process.

        In-process services return None. The supervisor does not schedule
        the host process in their place.
        """
        return None

    def request_memory_lock(self) -> bool:
        """Ask this service to ``mlockall`` before it runs.

        The default does not lock and returns False. A service that locks
        its own process returns True.
        """
        return False

    def memory_lock_report(self) -> str | None:
        """``ok``, a refusal string, or None while the lock result is unknown."""
        return None

    @property
    def readiness_timeout_s(self) -> float:
        return self._readiness_timeout_s

    @property
    def liveness_timeout_s(self) -> float:
        return self._liveness_timeout_s

    @property
    def shutdown_records(self) -> tuple[ShutdownRecord, ...]:
        with self._lock:
            return tuple(self._shutdown)

    def clock(self) -> float:
        return self._clock()

    def note_shutdown(self, record: ShutdownRecord) -> None:
        """Append a shutdown record. Does not change the lifecycle by itself."""
        with self._lock:
            self._shutdown.append(record)

    def _transition(self, dst: LifecycleState) -> None:
        with self._lock:
            self._state = move(self._state, dst)
            self._history.append(self._state)

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._failure_reason = reason
            self._state = move(self._state, LifecycleState.FAILED)
            self._history.append(self._state)

    def on_configure(self, config: Mapping[str, Any]) -> None:
        """Hook. Raise to fail configuration."""

    def on_start(self) -> None:
        """Hook. Raise to fail start. Call :meth:`mark_ready` once up."""

    def on_stop(self) -> None:
        """Hook. Raise to fail stop. Honour :attr:`grace_s` when waiting."""

    def configure(self, config: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            src = self._state
            if (src, LifecycleState.CONFIGURED) not in LEGAL_TRANSITIONS:
                raise IllegalTransition(src, LifecycleState.CONFIGURED)
        cfg = dict(config or {})
        try:
            self.on_configure(cfg)
        except Exception as exc:
            self._fail(f"configure failed: {exc}")
            raise
        with self._lock:
            self._config = cfg
            self._failure_reason = ""
            self._explicit_ready = False
            self._degraded_reason = ""
            self._last_heartbeat_at = None
            self._starting_at = None
        self._transition(LifecycleState.CONFIGURED)

    def start(self) -> None:
        with self._lock:
            src = self._state
            if (src, LifecycleState.STARTING) not in LEGAL_TRANSITIONS:
                raise IllegalTransition(src, LifecycleState.STARTING)
        self._transition(LifecycleState.STARTING)
        with self._lock:
            self._starting_at = self._clock()
        try:
            self.on_start()
        except Exception as exc:
            self._fail(f"start failed: {exc}")
            raise
        self.poll()

    def mark_ready(self) -> None:
        """Record that the subclass's own readiness condition is met."""
        with self._lock:
            self._explicit_ready = True

    def ready(self) -> bool:
        """Readiness probe. Does not look at the heartbeat."""
        with self._lock:
            return self._explicit_ready

    def heartbeat(self) -> None:
        """Refresh liveness. Does not change readiness or the lifecycle."""
        with self._lock:
            self._last_heartbeat_at = self._clock()

    def live(self) -> bool:
        """Liveness probe. Does not look at readiness.

        False until the first heartbeat, and false once the last heartbeat
        is at least ``liveness_timeout_s`` old.
        """
        with self._lock:
            last = self._last_heartbeat_at
            if last is None:
                return False
            return (self._clock() - last) < self._liveness_timeout_s

    def _heartbeat_expired(self) -> bool:
        """True when the liveness deadline has elapsed.

        Before the first heartbeat the deadline is measured from start, so a
        service is not failed in the same instant it starts.
        """
        now = self._clock()
        last = self._last_heartbeat_at
        if last is None:
            if self._starting_at is None:
                return False
            return (now - self._starting_at) >= self._liveness_timeout_s
        return (now - last) >= self._liveness_timeout_s

    def poll(self) -> None:
        """Commit readiness and liveness deadlines.

        Readiness is checked first. A service that is not ready and has missed
        its readiness timeout fails with ``readiness timeout`` even if the
        heartbeat is also late. A ready service that misses its heartbeat
        fails with ``liveness timeout``.
        """
        is_ready = self.ready()
        with self._lock:
            if self._state is LifecycleState.STARTING:
                if is_ready:
                    self._state = move(self._state, LifecycleState.READY)
                    self._history.append(self._state)
                elif (
                    self._starting_at is not None
                    and (self._clock() - self._starting_at) >= self._readiness_timeout_s
                ):
                    self._failure_reason = "readiness timeout"
                    self._state = move(self._state, LifecycleState.FAILED)
                    self._history.append(self._state)
                    return
            if self._state in (LifecycleState.STARTING, LifecycleState.READY):
                if self._heartbeat_expired():
                    self._failure_reason = "liveness timeout"
                    self._state = move(self._state, LifecycleState.FAILED)
                    self._history.append(self._state)

    def degrade(self, reason: str) -> None:
        """Mark the running service degraded without leaving ``ready``."""
        if not reason or not reason.strip():
            raise ValueError("degraded health needs a reason")
        with self._lock:
            if self._state is not LifecycleState.READY:
                raise ServiceError(
                    f"degrade is only valid in ready, not {self._state.value}"
                )
            self._degraded_reason = reason

    def clear_degraded(self) -> None:
        with self._lock:
            self._degraded_reason = ""

    def health(self) -> Health:
        """Report status, reason, readiness, and liveness separately."""
        is_ready = self.ready()
        is_live = self.live()
        with self._lock:
            if self._state is LifecycleState.FAILED:
                return Health(
                    HealthStatus.FAILED, self._failure_reason, is_ready, is_live
                )
            degraded = self._degraded_reason
            if (
                self._state is LifecycleState.READY
                and is_ready
                and is_live
                and not degraded
            ):
                return Health(HealthStatus.OK, "", True, True)
            if degraded:
                return Health(HealthStatus.DEGRADED, degraded, is_ready, is_live)
            if not is_ready:
                reason = "not ready"
            elif not is_live:
                reason = (
                    "no heartbeat"
                    if self._last_heartbeat_at is None
                    else "liveness timeout"
                )
            else:
                reason = ""
            return Health(HealthStatus.DEGRADED, reason, is_ready, is_live)

    def adopt_liveness_window(self, seconds: float) -> None:
        """Tighten the liveness window. Never lengthens it."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("liveness window must be a number > 0")
        if seconds <= 0:
            raise ValueError("liveness window must be > 0")
        with self._lock:
            self._liveness_timeout_s = min(self._liveness_timeout_s, float(seconds))

    def stop(self) -> None:
        with self._lock:
            src = self._state
            if (src, LifecycleState.STOPPING) not in LEGAL_TRANSITIONS:
                raise IllegalTransition(src, LifecycleState.STOPPING)
        self._transition(LifecycleState.STOPPING)
        try:
            self.on_stop()
        except Exception as exc:
            self._fail(f"stop failed: {exc}")
            raise
        with self._lock:
            self._explicit_ready = False
        self._transition(LifecycleState.STOPPED)


# Re-export the transition error used by callers that catch configure/start.
__all__ = [
    "IllegalTransition",
    "Service",
    "ServiceBase",
    "ServiceError",
]
