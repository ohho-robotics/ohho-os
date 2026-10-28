"""Lifecycle states, the legal transition table, and health reports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class LifecycleState(Enum):
    """Explicit service lifecycle. Health is reported separately."""

    CREATED = "created"
    CONFIGURED = "configured"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class HealthStatus(Enum):
    """``health()`` status. Readiness and liveness are separate booleans."""

    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


# Every legal edge. Anything absent is illegal, including self-edges other
# than reconfigure (configured → configured).
LEGAL_TRANSITIONS: frozenset[tuple[LifecycleState, LifecycleState]] = frozenset(
    {
        (LifecycleState.CREATED, LifecycleState.CONFIGURED),
        (LifecycleState.CREATED, LifecycleState.FAILED),
        (LifecycleState.CONFIGURED, LifecycleState.CONFIGURED),
        (LifecycleState.CONFIGURED, LifecycleState.STARTING),
        (LifecycleState.CONFIGURED, LifecycleState.STOPPING),
        (LifecycleState.CONFIGURED, LifecycleState.FAILED),
        (LifecycleState.STARTING, LifecycleState.READY),
        (LifecycleState.STARTING, LifecycleState.FAILED),
        (LifecycleState.STARTING, LifecycleState.STOPPING),
        (LifecycleState.READY, LifecycleState.FAILED),
        (LifecycleState.READY, LifecycleState.STOPPING),
        (LifecycleState.STOPPING, LifecycleState.STOPPED),
        (LifecycleState.STOPPING, LifecycleState.FAILED),
        (LifecycleState.FAILED, LifecycleState.STOPPING),
        (LifecycleState.STOPPED, LifecycleState.CONFIGURED),
    }
)


class IllegalTransition(Exception):
    """``src → dst`` is not in :data:`LEGAL_TRANSITIONS`."""

    def __init__(self, src: LifecycleState, dst: LifecycleState) -> None:
        self.src = src
        self.dst = dst
        super().__init__(f"illegal lifecycle transition {src.value} -> {dst.value}")


class ServiceError(Exception):
    """The service could not perform an operation in its current state."""


class UnixSocketUnavailable(ServiceError):
    """This platform cannot host a subprocess heartbeat socket."""


def move(src: LifecycleState, dst: LifecycleState) -> LifecycleState:
    """Return ``dst`` when the edge is legal.

    Raises :class:`IllegalTransition` otherwise. This is the only gate the
    service base uses to change state.
    """
    if (src, dst) not in LEGAL_TRANSITIONS:
        raise IllegalTransition(src, dst)
    return dst


@dataclass(frozen=True)
class Health:
    """One health report. ``ready`` and ``live`` are independent."""

    status: HealthStatus
    reason: str
    ready: bool
    live: bool


@dataclass(frozen=True)
class ShutdownRecord:
    """One recorded shutdown action. A forced kill sets ``forced``."""

    action: str
    forced: bool
    grace_s: float
    detail: str = ""
    signal: int | None = None


class ManualClock:
    """Deterministic clock for tests. ``advance`` is the only way it moves."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("clock cannot go backwards")
        self.now += dt


Clock = Callable[[], float]
