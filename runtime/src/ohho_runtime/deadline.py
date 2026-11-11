"""Monotonic deadline and the ``/ohho/heartbeat`` wire contract (OHH-359).

:class:`Deadline` and :class:`Heartbeat` share one clock. A command window
and a heartbeat miss use :attr:`ohho.safety.SafetyProfile.cmd_timeout_s`.
State freshness uses :attr:`ohho.safety.SafetyProfile.state_timeout_s`.
Those fields are read from the profile the caller passes. This module does
not embed them as literals.

:mod:`ohho_runtime.heartbeat` is the parent/worker socket protocol. It stays
free of an ``ohho`` import. This module is the wire deadline. Expiry does
not latch an e-stop, arm a robot, or send a command.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ohho.registry import list_specs
from ohho.robot_spec import RobotClass, parse_robot_class
from ohho.safety import SafetyProfile, profile_for

Clock = Callable[[], float]

# Wire names shared with the teleop heartbeat contract. Timeouts are not
# in this list: they are copied from a SafetyProfile.
HEARTBEAT_TOPIC = "/ohho/heartbeat"
HEARTBEAT_TYPE = "std_msgs/Empty"
HEARTBEAT_HZ = 10.0
SAFETY_STATE_TOPIC = "/ohho/safety/state"
SAFETY_STATE_TYPE = "std_msgs/String"
SAFETY_STATE_FIELDS = ("armed", "latched", "reason")
ESTOP_RESET_SERVICE = "/ohho/estop_reset"
ESTOP_RESET_TYPE = "std_srvs/Trigger"
# The teleop gate ticks at this period. A dead client is zeroed on the
# next tick after the command timeout, which stays inside STOP_BUDGET_S.
GATE_TICK_S = 0.01
# Published stop budget at the class-maximum command timeout. Do not raise.
STOP_BUDGET_S = 0.35

_TIMEOUT_FIELDS = ("cmd_timeout_s", "state_timeout_s")


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number > 0")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0")
    return number


def _stamp(clock: Clock, now: float | None) -> float:
    value: object = clock() if now is None else now
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("timestamp must be a finite number")
    return number


def _index_after(origin: float, period: float, now: float) -> int:
    """Smallest ``k >= 1`` such that ``origin + k * period`` is after ``now``.

    A sample that lands on a grid point schedules the following point.
    """
    if now < origin:
        return 0
    ratio = (now - origin) / period
    k = math.floor(ratio) + 1
    if k < 1:
        k = 1
    if origin + k * period <= now:
        k += 1
    return k


def _stop_budget(timeout_s: float) -> float:
    """Stop budget for a command timeout.

    At the class maximum the budget is :data:`STOP_BUDGET_S`. A shorter
    timeout keeps the same slack and stays under that budget. The slack is
    at least one gate tick.
    """
    maximum = SafetyProfile().cmd_timeout_s
    if timeout_s == maximum:
        return STOP_BUDGET_S
    slack = STOP_BUDGET_S - maximum
    if slack < GATE_TICK_S:
        slack = GATE_TICK_S
    budget = timeout_s + slack
    if budget > STOP_BUDGET_S and timeout_s <= maximum:
        return STOP_BUDGET_S
    return budget


def _require_profile(profile: SafetyProfile) -> SafetyProfile:
    if not isinstance(profile, SafetyProfile):
        raise TypeError("profile must be a SafetyProfile")
    return profile


@dataclass(frozen=True, slots=True)
class HeartbeatContract:
    """Topics, rate, and timeouts for one heartbeat publisher.

    ``timeout_s`` is the profile command timeout. ``state_timeout_s`` is the
    profile state timeout. ``stop_budget_s`` keeps the slack between the
    class-maximum command timeout and :data:`STOP_BUDGET_S`, so the default
    profile reports that published budget and a shorter profile reports a
    shorter one.
    """

    topic: str
    message_type: str
    rate_hz: float
    period_s: float
    timeout_s: float
    stop_budget_s: float
    gate_tick_s: float
    safety_state_topic: str
    safety_state_type: str
    safety_state_fields: tuple[str, ...]
    state_timeout_s: float
    estop_reset_service: str
    estop_reset_type: str


def wire_contract(profile: SafetyProfile | None = None) -> HeartbeatContract:
    """Build the wire contract from ``profile``, or from a default profile.

    The rate and topic names are the shared teleop contract. The timeouts
    are the profile's ``cmd_timeout_s`` and ``state_timeout_s``.
    """
    chosen = SafetyProfile() if profile is None else _require_profile(profile)
    rate_hz = _positive("rate_hz", HEARTBEAT_HZ)
    timeout_s = chosen.cmd_timeout_s
    return HeartbeatContract(
        topic=HEARTBEAT_TOPIC,
        message_type=HEARTBEAT_TYPE,
        rate_hz=rate_hz,
        period_s=1.0 / rate_hz,
        timeout_s=timeout_s,
        stop_budget_s=_stop_budget(timeout_s),
        gate_tick_s=GATE_TICK_S,
        safety_state_topic=SAFETY_STATE_TOPIC,
        safety_state_type=SAFETY_STATE_TYPE,
        safety_state_fields=SAFETY_STATE_FIELDS,
        state_timeout_s=chosen.state_timeout_s,
        estop_reset_service=ESTOP_RESET_SERVICE,
        estop_reset_type=ESTOP_RESET_TYPE,
    )


def safety_state_json(*, armed: bool, latched: bool, reason: str) -> str:
    """JSON object for ``/ohho/safety/state``.

    Keys are ``armed``, ``latched``, and ``reason``, sorted. This does not
    publish the message and does not change a gate.
    """
    if not isinstance(armed, bool) or not isinstance(latched, bool):
        raise TypeError("armed and latched must be bools")
    if not isinstance(reason, str):
        raise TypeError("reason must be a string")
    payload = {"armed": armed, "latched": latched, "reason": reason}
    return json.dumps(payload, sort_keys=True)


class Deadline:
    """One monotonic window.

    ``expired`` is true before the first :meth:`refresh`, when the clock
    moves backwards, and when the age is at least ``timeout_s``. A smaller
    age is fresh. That is the same edge the command and state watchdogs
    use: fresh while ``age < timeout``, expired at ``age >= timeout``.

    The clock defaults to :func:`time.monotonic`. Tests pass their own.
    ``timeout_s`` has no default. Use :meth:`command` or :meth:`state` so
    the number comes from a :class:`~ohho.safety.SafetyProfile`.
    """

    def __init__(self, timeout_s: float, *, clock: Clock | None = None) -> None:
        self._timeout_s = _positive("timeout_s", timeout_s)
        self._clock: Clock = time.monotonic if clock is None else clock
        self._last: float | None = None
        self._lock = threading.Lock()

    @classmethod
    def command(cls, profile: SafetyProfile, *, clock: Clock | None = None) -> Deadline:
        """Command window: ``profile.cmd_timeout_s``."""
        chosen = _require_profile(profile)
        return cls(chosen.cmd_timeout_s, clock=clock)

    @classmethod
    def state(cls, profile: SafetyProfile, *, clock: Clock | None = None) -> Deadline:
        """State window: ``profile.state_timeout_s``."""
        chosen = _require_profile(profile)
        return cls(chosen.state_timeout_s, clock=clock)

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def refresh(self, now: float | None = None) -> None:
        """Mark a fresh sample at ``now`` or at the clock."""
        stamp = _stamp(self._clock, now)
        with self._lock:
            self._last = stamp

    def age_s(self, now: float | None = None) -> float | None:
        """Seconds since :meth:`refresh`, or ``None`` before the first one.

        A negative age means the clock moved backwards. :meth:`expired`
        treats that as a miss.
        """
        stamp = _stamp(self._clock, now)
        with self._lock:
            last = self._last
        if last is None:
            return None
        return stamp - last

    def expired(self, now: float | None = None) -> bool:
        """True when the window has not been refreshed inside ``timeout_s``."""
        age = self.age_s(now)
        if age is None or age < 0.0:
            return True
        return age >= self._timeout_s

    def fresh(self, now: float | None = None) -> bool:
        """True when a sample is strictly inside the window."""
        return not self.expired(now)

    def remaining_s(self, now: float | None = None) -> float:
        """Seconds left in the window, or ``0`` when it is already expired."""
        if self.expired(now):
            return 0.0
        age = self.age_s(now)
        if age is None:
            return 0.0
        left = self._timeout_s - age
        if left < 0.0:
            return 0.0
        return left


class Heartbeat:
    """Publish schedule at the wire rate, missed after the command timeout.

    The miss window is ``profile.cmd_timeout_s``. The period is ``1 / rate_hz``
    and defaults to :data:`HEARTBEAT_HZ`. :meth:`start` makes the first beat
    due immediately. :meth:`beat` refreshes the deadline and schedules the
    next grid point after that sample. A late beat does not leave a backlog.

    The deadline stays expired until the first beat. A miss does not latch
    and does not command the robot.
    """

    def __init__(
        self,
        profile: SafetyProfile | None = None,
        *,
        rate_hz: float = HEARTBEAT_HZ,
        clock: Clock | None = None,
    ) -> None:
        chosen = SafetyProfile() if profile is None else _require_profile(profile)
        self._clock: Clock = time.monotonic if clock is None else clock
        self._deadline = Deadline.command(chosen, clock=self._clock)
        self._rate_hz = _positive("rate_hz", rate_hz)
        self._period_s = 1.0 / self._rate_hz
        self._origin: float | None = None
        self._next_index = 0
        self._lock = threading.Lock()

    @property
    def timeout_s(self) -> float:
        return self._deadline.timeout_s

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    @property
    def period_s(self) -> float:
        return self._period_s

    def start(self, now: float | None = None) -> None:
        """Arm the publish grid. Does not count as a beat."""
        stamp = _stamp(self._clock, now)
        with self._lock:
            self._origin = stamp
            self._next_index = 0

    def beat(self, now: float | None = None) -> None:
        """Record one heartbeat and schedule the next publish point."""
        stamp = _stamp(self._clock, now)
        self._deadline.refresh(stamp)
        with self._lock:
            if self._origin is None:
                self._origin = stamp
            self._next_index = _index_after(self._origin, self._period_s, stamp)

    def due(self, now: float | None = None) -> bool:
        """True when :meth:`start` has run and the next grid point has arrived."""
        stamp = _stamp(self._clock, now)
        with self._lock:
            origin = self._origin
            index = self._next_index
            period = self._period_s
        if origin is None:
            return False
        return stamp >= origin + index * period

    def expired(self, now: float | None = None) -> bool:
        """True when the command-timeout window has no fresh beat."""
        return self._deadline.expired(now)

    def fresh(self, now: float | None = None) -> bool:
        return not self.expired(now)

    def age_s(self, now: float | None = None) -> float | None:
        return self._deadline.age_s(now)


@dataclass(frozen=True, slots=True)
class TimeoutCeiling:
    """The longest command and state timeouts a class may declare.

    Both numbers are copied from a default :class:`~ohho.safety.SafetyProfile`.
    Every robot class uses that public profile. A declared timeout may be
    shorter. It may not be longer.
    """

    robot_class: str
    cmd_timeout_s: float
    state_timeout_s: float


class TimeoutConformanceError(ValueError):
    """A declared timeout is above the class maximum, or is not usable."""

    def __init__(self, robot_class: str, reasons: Sequence[str]) -> None:
        self.robot_class = robot_class
        self.reasons = tuple(reasons)
        detail = "; ".join(self.reasons)
        super().__init__(f"timeout conformance failed for {robot_class}: {detail}")


def class_timeout_ceiling(robot_class: RobotClass | str) -> TimeoutCeiling:
    """Public command and state timeouts for ``robot_class``.

    The values are ``SafetyProfile().cmd_timeout_s`` and
    ``SafetyProfile().state_timeout_s``. An unknown class raises
    ``ValueError``.
    """
    kind = parse_robot_class(robot_class)
    profile = SafetyProfile()
    return TimeoutCeiling(
        robot_class=kind.value,
        cmd_timeout_s=profile.cmd_timeout_s,
        state_timeout_s=profile.state_timeout_s,
    )


def _raw_timeouts(declared: object) -> tuple[tuple[str, ...], object, object]:
    if isinstance(declared, (str, bytes)) or declared is None:
        return (("timeouts must name cmd_timeout_s and state_timeout_s",), None, None)
    if isinstance(declared, Mapping):
        missing = [name for name in _TIMEOUT_FIELDS if name not in declared]
        if missing:
            return (tuple(f"{name} is missing" for name in missing), None, None)
        return ((), declared["cmd_timeout_s"], declared["state_timeout_s"])
    missing = [name for name in _TIMEOUT_FIELDS if not hasattr(declared, name)]
    if missing:
        return (tuple(f"{name} is missing" for name in missing), None, None)
    return (
        (),
        getattr(declared, "cmd_timeout_s"),
        getattr(declared, "state_timeout_s"),
    )


def _checked_timeout(name: str, value: object, limit: float) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} must be a finite number > 0"
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return f"{name} must be a finite number > 0"
    if number > limit:
        return f"{name} {number} exceeds class maximum {limit}"
    return None


def check_timeout_conformance(
    declared: object,
    robot_class: RobotClass | str,
    *,
    ceiling: SafetyProfile | None = None,
) -> None:
    """Fail when ``declared`` timeouts exceed the class maximum.

    The maximum is :func:`class_timeout_ceiling`. ``ceiling``, when given,
    is a :class:`~ohho.safety.SafetyProfile` whose timeouts can only make
    that maximum stricter. A longer ``cmd_timeout_s`` or ``state_timeout_s``
    raises :class:`TimeoutConformanceError`. An equal or shorter timeout
    passes. This does not construct a gate and does not arm a robot.
    """
    cap = class_timeout_ceiling(robot_class)
    cmd_limit = cap.cmd_timeout_s
    state_limit = cap.state_timeout_s
    if ceiling is not None:
        if not isinstance(ceiling, SafetyProfile):
            raise TypeError("ceiling must be a SafetyProfile")
        cmd_limit = min(cmd_limit, ceiling.cmd_timeout_s)
        state_limit = min(state_limit, ceiling.state_timeout_s)
    errors, cmd_value, state_value = _raw_timeouts(declared)
    if errors:
        raise TimeoutConformanceError(cap.robot_class, errors)
    reasons: list[str] = []
    for name, value, limit in (
        ("cmd_timeout_s", cmd_value, cmd_limit),
        ("state_timeout_s", state_value, state_limit),
    ):
        reason = _checked_timeout(name, value, limit)
        if reason is not None:
            reasons.append(reason)
    if reasons:
        raise TimeoutConformanceError(cap.robot_class, reasons)


def check_registered_timeouts() -> int:
    """Check every registered robot's public profile. Return how many.

    The declared timeouts are :func:`ohho.safety.profile_for`. The maximum
    is the robot's class ceiling. A profile that raises either timeout
    fails the check.
    """
    specs = list_specs()
    for spec in specs:
        check_timeout_conformance(profile_for(spec.id, spec), spec.robot_class)
    return len(specs)


__all__ = [
    "ESTOP_RESET_SERVICE",
    "ESTOP_RESET_TYPE",
    "GATE_TICK_S",
    "HEARTBEAT_HZ",
    "HEARTBEAT_TOPIC",
    "HEARTBEAT_TYPE",
    "SAFETY_STATE_FIELDS",
    "SAFETY_STATE_TOPIC",
    "SAFETY_STATE_TYPE",
    "STOP_BUDGET_S",
    "Deadline",
    "Heartbeat",
    "HeartbeatContract",
    "TimeoutCeiling",
    "TimeoutConformanceError",
    "check_registered_timeouts",
    "check_timeout_conformance",
    "class_timeout_ceiling",
    "safety_state_json",
    "wire_contract",
]
