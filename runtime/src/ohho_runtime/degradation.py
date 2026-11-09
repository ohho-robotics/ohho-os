"""Safety-preserving degradation for a supervised robot.

A :class:`Degradation` watches one :class:`~ohho.safety.SafetyGate` and the
services the supervisor is running. When a critical service fails or its
output goes stale, this layer calls the class :class:`~ohho.safety_policy.SafetyPolicy`
through that gate:

* transport loss calls ``on_estop`` (the gate latches)
* every other critical failure calls ``on_watchdog``

What that means for a class stays in the policy. This module does not choose
a velocity, a flight mode, or a posture, and it does not arm the gate.

Consumers of a failed producer read an :class:`InvalidOutput` instead of the
last value. The age of that output is on the marker. The policy runner's
applied command becomes a :class:`HoldCommand` (all zeros) in the same step
as the failure, which is inside one policy period, and the previous command
is not replayed.

Perception failure disables the agent and leaves teleop in place while the
teleop gateway, transport, telemetry, and the watchdog are still ready.
A recorder failure only invalidates its own output.

A critical service that will not restart, or that fails again, latches
do-not-operate on the supervisor. A service restart clears arming and never
restores it. Motion after that restart needs a new arm from the operator.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ohho.registry import RobotSpec
from ohho.safety import SafetyGate
from ohho.safety_policy import SafetyPolicy, load_policy
from ohho.schema import Telemetry

from ohho_runtime.manifest import (
    ROLE_PERCEPTION,
    ROLE_POLICY,
    ROLE_RECORDER,
    ROLE_TRANSPORT,
    Criticality,
    ServiceSpec,
)
from ohho_runtime.service.lifecycle import LifecycleState

# Roles whose failure must not command the robot through the policy hooks.
_NO_MOTION_HOOK = frozenset({ROLE_PERCEPTION, ROLE_RECORDER})

# Teleop may continue only while every one of these is ready.
_TELEOP_REQUIRES = (
    "teleop_gateway",
    "transport",
    "telemetry",
    "safety_watchdog",
)


class DegradationError(ValueError):
    """A degradation input was not usable."""


@dataclass(frozen=True)
class InvalidOutput:
    """Marker returned instead of a failed producer's last value.

    ``age_s`` is how old that output was when it was read. It is not a
    command and it must not be applied.
    """

    role: str
    reason: str
    age_s: float


@dataclass(frozen=True)
class HoldCommand:
    """Zero command the policy runner applies when it falls back to hold."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    mode: str = "hold"


@dataclass(frozen=True)
class DegradationEvent:
    """One degradation decision. ``at`` is the degradation clock."""

    at: float
    kind: str
    role: str
    detail: str = ""


class OutputSlot:
    """Latest sample from one producer, or an explicit invalid marker.

    ``read`` returns the published value only while the slot is valid.
    After :meth:`invalidate`, it returns :class:`InvalidOutput` and the
    previous value is dropped.
    """

    def __init__(self, role: str) -> None:
        if not isinstance(role, str) or not role:
            raise DegradationError("output role must be a non-empty string")
        self.role = role
        self._value: object = None
        self._valid = False
        self._reason = ""
        self._produced_at: float | None = None
        self._invalid_at: float | None = None

    @property
    def valid(self) -> bool:
        """True when :meth:`read` returns the published value."""
        return self._valid

    def publish(self, value: object, at: float) -> None:
        """Store ``value`` as a fresh sample produced at ``at``."""
        self._check_time(at)
        self._value = value
        self._valid = True
        self._reason = ""
        self._produced_at = float(at)
        self._invalid_at = None

    def invalidate(self, at: float, reason: str) -> None:
        """Drop the last value and remember why, at time ``at``."""
        self._check_time(at)
        self._value = None
        self._valid = False
        self._reason = reason or "invalid"
        if self._produced_at is None:
            self._produced_at = float(at)
        self._invalid_at = float(at)

    def age_s(self, at: float) -> float:
        """Seconds since the sample was produced.

        An invalidated sample still has an age. A slot that was never
        published and never invalidated reports infinite age.
        """
        self._check_time(at)
        if self._produced_at is None:
            return math.inf
        return max(0.0, float(at) - self._produced_at)

    def read(self, at: float) -> object:
        """Return the fresh value, or an :class:`InvalidOutput`."""
        age = self.age_s(at)
        if not self._valid:
            reason = self._reason or "invalid"
            return InvalidOutput(self.role, reason, age)
        return self._value

    @staticmethod
    def _check_time(at: object) -> None:
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise DegradationError("output time must be a number")
        if not math.isfinite(float(at)):
            raise DegradationError("output time must be finite")


class PolicyRunner:
    """Motion output of the policy service.

    ``period_s`` is one policy period. :meth:`fail` invalidates the output
    and applies :class:`HoldCommand` immediately, so the zero lands inside
    that period. The previous command is not kept and is not replayed.
    A later :meth:`publish` is accepted only after :meth:`set_healthy`
    says the deadlines are healthy again.
    """

    def __init__(self, slot: OutputSlot, period_s: float) -> None:
        if not isinstance(slot, OutputSlot):
            raise DegradationError("policy runner slot must be an OutputSlot")
        self._slot = slot
        self.period_s = _positive_period(period_s)
        self.mode = "run"
        self.applied: object = HoldCommand()
        self.zeroed_after_s: float | None = None
        self._failed_at: float | None = None
        self._healthy = True

    @property
    def slot(self) -> OutputSlot:
        return self._slot

    def set_period(self, period_s: float) -> None:
        """Set the policy period. The zero on failure stays inside it."""
        self.period_s = _positive_period(period_s)

    def set_healthy(self, healthy: bool) -> None:
        """Record whether policy deadlines are healthy.

        This does not arm and does not by itself leave hold. A fresh
        :meth:`publish` after this is true is what resumes a command.
        """
        if not isinstance(healthy, bool):
            raise DegradationError("healthy must be a bool")
        self._healthy = healthy

    def publish(self, value: object, at: float) -> None:
        """Publish one command, unless hold is still waiting on deadlines."""
        if self.mode == "hold" and not self._healthy:
            return
        self._slot.publish(value, at)
        if (
            self.mode == "hold"
            and self._failed_at is not None
            and at <= self._failed_at
        ):
            return
        self.applied = value
        self.mode = "run"
        self._failed_at = None

    def fail(self, at: float, reason: str) -> None:
        """Invalidate the output and fall back to hold within one period."""
        if self.mode == "hold" and self._failed_at is not None:
            self._slot.invalidate(at, reason or "failed")
            self._healthy = False
            return
        self._slot.invalidate(at, reason or "failed")
        self._failed_at = float(at)
        self._healthy = False
        self.applied = HoldCommand()
        self.mode = "hold"
        # Applied in this step, so the delay is zero, which is within the period.
        self.zeroed_after_s = 0.0

    def read(self, at: float) -> object:
        """What a consumer of the policy stream sees."""
        return self._slot.read(at)


class Degradation:
    """Apply the class safety policy when supervised services fail.

    ``gate`` is the gate the policy hooks are called with. ``policy``
    defaults to :func:`ohho.safety_policy.load_policy` for ``spec``, bound
    to the gate's profile. The supervisor calls :meth:`on_service_failed`,
    :meth:`on_service_restart`, and :meth:`observe`.
    """

    def __init__(
        self,
        spec: RobotSpec,
        gate: SafetyGate,
        *,
        clock: Callable[[], float] | None = None,
        policy: SafetyPolicy | None = None,
    ) -> None:
        if not isinstance(spec, RobotSpec):
            raise DegradationError(
                f"spec must be a RobotSpec, not {type(spec).__name__}"
            )
        if not isinstance(gate, SafetyGate):
            raise DegradationError(
                f"gate must be a SafetyGate, not {type(gate).__name__}"
            )
        chosen = load_policy(spec) if policy is None else policy
        if not isinstance(chosen, SafetyPolicy):
            raise DegradationError(
                f"policy must be a SafetyPolicy, not {type(chosen).__name__}"
            )
        self._spec = spec
        self._gate = gate
        self._policy = chosen.bind(gate.profile)
        self._clock: Callable[[], float] = time.monotonic if clock is None else clock
        self._sample: Telemetry | None = None
        self._outputs: dict[str, OutputSlot] = {}
        self._limits: dict[str, float] = {}
        self._critical: dict[str, bool] = {}
        self._counts: dict[str, int] = {}
        self._states: dict[str, LifecycleState] = {}
        self._stale_latched: set[str] = set()
        self._events: list[DegradationEvent] = []
        self._agent_enabled = True
        self._restarts = 0
        self._dno = False
        self._supervisor: Any = None
        runner_slot = self._slot(ROLE_POLICY)
        self._runner = PolicyRunner(runner_slot, period_s=1.0)
        self._unsub = gate.on_telemetry(self._remember)

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def gate(self) -> SafetyGate:
        return self._gate

    @property
    def policy(self) -> SafetyPolicy:
        """Class policy bound to the gate profile."""
        return self._policy

    @property
    def policy_runner(self) -> PolicyRunner:
        return self._runner

    @property
    def agent_enabled(self) -> bool:
        """False after perception fails. Restart does not turn this back on."""
        return self._agent_enabled

    @property
    def arm_restored(self) -> bool:
        """Always false. A restart never restores arming."""
        return False

    @property
    def restarts(self) -> int:
        """How many service restarts this layer has observed."""
        return self._restarts

    @property
    def events(self) -> tuple[DegradationEvent, ...]:
        return tuple(self._events)

    def output(self, role: str) -> OutputSlot:
        """The slot consumers read for ``role``."""
        return self._slot(role)

    def failures(self, role: str) -> int:
        """How many times ``role`` has failed, not counting stale outputs."""
        return self._counts.get(role, 0)

    def do_not_operate(self) -> bool:
        """True while the supervisor latch, or a local escalate, is set."""
        supervisor = self._supervisor
        if supervisor is not None:
            flag = getattr(supervisor, "do_not_operate", None)
            if isinstance(flag, bool):
                return flag
        return self._dno

    def teleop_enabled(self) -> bool:
        """True while the operator path's required services are ready.

        Perception is not one of those services. A latched gate or a
        do-not-operate latch is not an available teleop path.
        """
        if self.do_not_operate() or self._gate.latched:
            return False
        for role in _TELEOP_REQUIRES:
            state = self._states.get(role)
            if state is not None and state is not LifecycleState.READY:
                return False
        return True

    def accepts_teleop(self) -> bool:
        """True when a live operator command is still allowed."""
        return self.teleop_enabled() and self._gate.armed and not self._gate.latched

    def accepts_agent(self) -> bool:
        """True when the agent may still command motion."""
        return (
            self._agent_enabled and self._runner.mode == "run" and self.accepts_teleop()
        )

    def bind(self, supervisor: object) -> None:
        """Remember the supervisor that should receive do-not-operate."""
        self._supervisor = supervisor

    def watch(self, spec: ServiceSpec) -> None:
        """Track one manifest service for staleness, criticality, and period."""
        if not isinstance(spec, ServiceSpec):
            raise DegradationError("watch expects a ServiceSpec")
        self._limits[spec.role] = spec.liveness_timeout_s
        self._critical[spec.role] = spec.criticality is Criticality.CRITICAL
        if spec.role == ROLE_POLICY:
            self._runner.set_period(1.0 / spec.loop_hz)

    def publish(self, role: str, value: object, at: float) -> None:
        """Publish one sample. A fresh sample clears that role's stale latch."""
        self._stale_latched.discard(role)
        if role == ROLE_POLICY:
            self._runner.publish(value, at)
            return
        self._slot(role).publish(value, at)

    def note_telemetry(self, sample: Telemetry) -> None:
        """Remember a telemetry sample for the next policy hook."""
        if not isinstance(sample, Telemetry):
            raise DegradationError("telemetry must be a Telemetry")
        self._sample = sample

    def on_service_failed(
        self,
        role: str,
        reason: str,
        *,
        criticality: Criticality,
        terminal: bool,
        states: Mapping[str, LifecycleState] | None = None,
        at: float | None = None,
    ) -> None:
        """Invalidate ``role`` and apply the class hook for this failure."""
        if states is not None:
            self._states = dict(states)
        now = self._now() if at is None else float(at)
        why = reason or "failed"
        self._counts[role] = self._counts.get(role, 0) + 1
        self._invalidate(role, why, now)
        if role == ROLE_PERCEPTION:
            self._agent_enabled = False
            self._record("agent_disabled", role, why, now)
        elif role == ROLE_POLICY:
            self._runner.fail(now, why)
            self._record("hold", role, self._runner.mode, now)
            self._apply_hook(role, why, critical=True, at=now)
        elif role != ROLE_RECORDER:
            critical = criticality is Criticality.CRITICAL
            self._apply_hook(role, why, critical=critical, at=now)
        # Perception and the recorder degrade in place. They do not latch
        # do-not-operate. A critical service that cannot restart, or that
        # fails again, does.
        if role in (ROLE_PERCEPTION, ROLE_RECORDER):
            return
        if criticality is Criticality.CRITICAL and (
            terminal or self._counts[role] >= 2
        ):
            self._escalate(role, why, now)

    def on_service_restart(self, role: str, at: float | None = None) -> None:
        """A service came back. Arming is cleared and not restored."""
        now = self._now() if at is None else float(at)
        self._restarts += 1
        self._clear_arming(role, now)
        if role == ROLE_POLICY:
            self._runner.set_healthy(False)

    def observe(
        self,
        at: float,
        states: Mapping[str, LifecycleState] | None = None,
    ) -> None:
        """Apply stale-output deadlines and refresh the teleop view.

        Does not sleep. A sample whose age has reached its liveness window
        is invalidated once. Transport staleness uses the estop hook.
        """
        if states is not None:
            self._states = dict(states)
        now = float(at)
        self._expire(now)
        policy_state = self._states.get(ROLE_POLICY)
        self._runner.set_healthy(policy_state is LifecycleState.READY)

    def _remember(self, sample: Telemetry) -> None:
        self._sample = sample

    def _now(self) -> float:
        return float(self._clock())

    def _slot(self, role: str) -> OutputSlot:
        slot = self._outputs.get(role)
        if slot is None:
            slot = OutputSlot(role)
            self._outputs[role] = slot
        return slot

    def _invalidate(self, role: str, reason: str, at: float) -> None:
        self._slot(role).invalidate(at, reason)
        self._record("invalidate", role, reason, at)

    def _telemetry(self) -> Telemetry | None:
        if self._sample is not None:
            return self._sample
        current = self._gate.read()
        if isinstance(current, Telemetry):
            return current
        return None

    def _apply_hook(self, role: str, reason: str, *, critical: bool, at: float) -> None:
        if role in _NO_MOTION_HOOK:
            return
        if role != ROLE_TRANSPORT and role != ROLE_POLICY and not critical:
            return
        sample = self._telemetry()
        if role == ROLE_TRANSPORT:
            self._policy.on_estop(self._gate, sample)
            self._record("on_estop", role, reason, at)
            return
        self._policy.on_watchdog(self._gate, sample)
        self._record("on_watchdog", role, reason, at)

    def _expire(self, at: float) -> None:
        for role, limit in self._limits.items():
            if role in self._stale_latched:
                continue
            slot = self._outputs.get(role)
            if slot is None or not slot.valid:
                continue
            if slot.age_s(at) < limit:
                continue
            self._stale_latched.add(role)
            self._invalidate(role, "stale", at)
            if role == ROLE_PERCEPTION:
                self._agent_enabled = False
                self._record("agent_disabled", role, "stale", at)
                continue
            if role == ROLE_POLICY:
                self._runner.fail(at, "stale")
                self._record("hold", role, self._runner.mode, at)
            critical = self._critical.get(role, False) or role == ROLE_POLICY
            self._apply_hook(role, "stale", critical=critical, at=at)

    def _escalate(self, role: str, reason: str, at: float) -> None:
        self._dno = True
        # Do-not-operate leaves the gate disarmed. It does not arm, and it
        # does not clear an e-stop the policy already latched.
        self._clear_arming(role, at)
        self._record("escalate", role, reason, at)
        supervisor = self._supervisor
        if supervisor is None:
            return
        escalate = getattr(supervisor, "escalate_do_not_operate", None)
        if callable(escalate):
            escalate(role, reason)

    def _clear_arming(self, role: str, at: float) -> None:
        """Drop arming after a restart. A latch is left latched."""
        gate = self._gate
        if not gate.armed:
            self._record("arming_cleared", role, "already disarmed", at)
            return
        clear_arming = gate.disarm
        clear_arming("service_restart")
        self._record("arming_cleared", role, "service_restart", at)

    def _record(self, kind: str, role: str, detail: str, at: float) -> None:
        self._events.append(DegradationEvent(at, kind, role, detail))


def _positive_period(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DegradationError("policy period must be a number > 0")
    period = float(value)
    if not math.isfinite(period) or period <= 0:
        raise DegradationError("policy period must be a number > 0")
    return period


__all__ = [
    "Degradation",
    "DegradationError",
    "DegradationEvent",
    "HoldCommand",
    "InvalidOutput",
    "OutputSlot",
    "PolicyRunner",
]
