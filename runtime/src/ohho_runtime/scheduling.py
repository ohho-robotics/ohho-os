"""Apply a service's scheduling config, or report that it was not applied.

Defaults leave the process alone: policy ``other``, no niceness, no
realtime priority, no affinity, and no ``mlockall``. On Linux, outside
in-process dev mode, the supervisor applies the declared values at
service start. It probes ``CAP_SYS_NICE``, ``RLIMIT_RTPRIO``, and
``RLIMIT_MEMLOCK``. A refusal does not stop the service and does not
substitute another priority. The refusal is a warning and a counter.

On macOS, Windows, and in dev mode the config is ignored. One warning
covers every service.

Recommended order, highest first: safety watchdog and transport, then
control loops, then perception and the agent. Do not raise a policy or
a priority until :func:`ohho_runtime.timing.loop_report` for that
service shows the misses the change is for. This is configuration, not
a hard real-time promise.

OS calls are injected. The default on Linux is :class:`LinuxOsCalls`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ohho_runtime.manifest import SchedulePolicy, SchedulingHint, ServiceSpec
from ohho_runtime.os_sched import LinuxOsCalls

SCHEDULING_REFUSED = "scheduling_refused_total"
SCHEDULING_IGNORED = "scheduling_ignored_total"
SCHEDULING_APPLIED = "scheduling_applied_total"

_LOG = logging.getLogger("ohho_runtime.scheduling")
_METRIC_NAMES = (SCHEDULING_REFUSED, SCHEDULING_IGNORED, SCHEDULING_APPLIED)


class OsCalls(Protocol):
    """Scheduler operations. Tests supply a fake. Linux uses :class:`LinuxOsCalls`."""

    def cap_sys_nice(self) -> bool:
        """Effective ``CAP_SYS_NICE``."""

    def rlimit_rtprio(self) -> tuple[int, int]:
        """``RLIMIT_RTPRIO`` ``(soft, hard)``."""

    def rlimit_memlock(self) -> tuple[int, int]:
        """``RLIMIT_MEMLOCK`` ``(soft, hard)``."""

    def rlimit_infinity(self) -> int:
        """Unlimited rlimit sentinel."""

    def get_nice(self, pid: int) -> int:
        """Absolute niceness of ``pid``."""

    def set_nice(self, pid: int, value: int) -> None:
        """Set absolute niceness of ``pid`` to ``value``."""

    def set_scheduler(self, pid: int, policy: str, priority: int) -> None:
        """Set ``policy`` (``fifo`` or ``rr``) and exactly ``priority``."""

    def set_affinity(self, pid: int, cpus: tuple[int, ...]) -> None:
        """Pin ``pid`` to exactly ``cpus``."""

    def mlockall(self) -> None:
        """Lock the calling process."""


class SchedulingMetrics:
    """Counters for applied, ignored, and refused scheduling."""

    def __init__(self) -> None:
        self._counts = {name: 0 for name in _METRIC_NAMES}

    def inc(self, name: str) -> None:
        """Add one to ``name``."""
        if name not in self._counts:
            raise KeyError(name)
        self._counts[name] += 1

    def get(self, name: str) -> int:
        """Current value of ``name``."""
        if name not in self._counts:
            raise KeyError(name)
        return self._counts[name]

    def snapshot(self) -> dict[str, int]:
        """Copy of the counters."""
        return dict(self._counts)


@dataclass(frozen=True)
class FieldRefusal:
    """One setting the applier did not install."""

    field: str
    declared: str
    reason: str


@dataclass(frozen=True)
class SchedulingRefusal:
    """A refusal tied to a service role, for the supervisor event log."""

    role: str
    field: str
    declared: str
    reason: str


@dataclass(frozen=True)
class ApplyOutcome:
    """What :func:`apply_scheduling` changed and what it refused."""

    applied: tuple[str, ...] = ()
    refusals: tuple[FieldRefusal, ...] = ()


def describe_scheduling(hint: SchedulingHint) -> str:
    """Short text of the OS settings in ``hint``. Period is not included."""
    parts: list[str] = []
    if hint.nice is not None:
        parts.append(f"nice={hint.nice}")
    if hint.policy is not SchedulePolicy.OTHER or hint.priority is not None:
        parts.append(f"policy={hint.policy.value}")
    if hint.priority is not None:
        parts.append(f"priority={hint.priority}")
    if hint.cpu_affinity:
        cpus = ",".join(str(cpu) for cpu in hint.cpu_affinity)
        parts.append(f"cpu_affinity={cpus}")
    if hint.mlockall:
        parts.append("mlockall=true")
    return " ".join(parts)


def _limit_allows(soft: int, need: int, infinity: int) -> bool:
    if soft == infinity:
        return True
    return soft >= need


def _memlock_allows(soft: int, infinity: int) -> bool:
    if soft == infinity:
        return True
    return soft > 0


def _probe(os_calls: OsCalls) -> tuple[bool, tuple[int, int], tuple[int, int], int]:
    cap = os_calls.cap_sys_nice()
    rtprio = os_calls.rlimit_rtprio()
    memlock = os_calls.rlimit_memlock()
    infinity = os_calls.rlimit_infinity()
    return cap, rtprio, memlock, infinity


def apply_scheduling(
    hint: SchedulingHint,
    *,
    pid: int | None,
    os_calls: OsCalls,
    same_process: bool,
    mlock_delegated: bool,
) -> ApplyOutcome:
    """Apply ``hint`` to ``pid``, or record each setting that was not applied.

    A realtime priority is never clamped to the rlimit. Affinity is the
    declared CPU set or it is left alone. ``mlockall`` runs only in the
    service process (``same_process``). A delegated lock is the child's
    job; this function does not also lock the caller.
    """
    if not hint.requests_os_change():
        return ApplyOutcome()
    applied: list[str] = []
    refusals: list[FieldRefusal] = []
    if hint.nice is not None:
        _apply_nice(hint, pid, os_calls, applied, refusals)
    if hint.policy is not SchedulePolicy.OTHER:
        _apply_policy(hint, pid, os_calls, applied, refusals)
    if hint.cpu_affinity:
        _apply_affinity(hint, pid, os_calls, applied, refusals)
    if hint.mlockall and not mlock_delegated:
        _apply_mlock(pid, os_calls, same_process, applied, refusals)
    return ApplyOutcome(tuple(applied), tuple(refusals))


def _apply_nice(
    hint: SchedulingHint,
    pid: int | None,
    os_calls: OsCalls,
    applied: list[str],
    refusals: list[FieldRefusal],
) -> None:
    declared = f"nice {hint.nice}"
    if pid is None:
        refusals.append(FieldRefusal("nice", declared, "service has no process id"))
        return
    try:
        cap = os_calls.cap_sys_nice()
        current = os_calls.get_nice(pid)
    except OSError as exc:
        refusals.append(
            FieldRefusal("nice", declared, f"nice could not be read ({exc})")
        )
        return
    requested = hint.nice
    if requested is None:
        return
    if requested < current and not cap:
        refusals.append(
            FieldRefusal(
                "nice",
                declared,
                f"CAP_SYS_NICE not held; current nice is {current}",
            )
        )
        return
    try:
        os_calls.set_nice(pid, requested)
    except OSError as exc:
        refusals.append(FieldRefusal("nice", declared, str(exc)))
        return
    applied.append(f"nice={requested}")


def _apply_policy(
    hint: SchedulingHint,
    pid: int | None,
    os_calls: OsCalls,
    applied: list[str],
    refusals: list[FieldRefusal],
) -> None:
    priority = hint.priority
    declared = f"policy {hint.policy.value} priority {priority}"
    if pid is None:
        refusals.append(FieldRefusal("policy", declared, "service has no process id"))
        return
    if isinstance(priority, bool) or not isinstance(priority, int):
        refusals.append(
            FieldRefusal("policy", declared, "fifo and rr require an integer priority")
        )
        return
    try:
        cap, rtprio, _memlock, infinity = _probe(os_calls)
    except OSError as exc:
        refusals.append(
            FieldRefusal("policy", declared, f"capability probe failed ({exc})")
        )
        return
    soft = rtprio[0]
    if not cap and not _limit_allows(soft, priority, infinity):
        refusals.append(
            FieldRefusal(
                "policy",
                declared,
                f"CAP_SYS_NICE not held, RLIMIT_RTPRIO soft {soft}",
            )
        )
        return
    try:
        os_calls.set_scheduler(pid, hint.policy.value, priority)
    except OSError as exc:
        refusals.append(FieldRefusal("policy", declared, str(exc)))
        return
    applied.append(f"policy={hint.policy.value} priority={priority}")


def _apply_affinity(
    hint: SchedulingHint,
    pid: int | None,
    os_calls: OsCalls,
    applied: list[str],
    refusals: list[FieldRefusal],
) -> None:
    cpus = ",".join(str(cpu) for cpu in hint.cpu_affinity)
    declared = f"cpu_affinity {cpus}"
    if pid is None:
        refusals.append(
            FieldRefusal("cpu_affinity", declared, "service has no process id")
        )
        return
    try:
        os_calls.set_affinity(pid, hint.cpu_affinity)
    except OSError as exc:
        refusals.append(FieldRefusal("cpu_affinity", declared, str(exc)))
        return
    applied.append(f"cpu_affinity={cpus}")


def _apply_mlock(
    pid: int | None,
    os_calls: OsCalls,
    same_process: bool,
    applied: list[str],
    refusals: list[FieldRefusal],
) -> None:
    declared = "mlockall"
    try:
        _cap, _rtprio, memlock, infinity = _probe(os_calls)
    except OSError as exc:
        refusals.append(
            FieldRefusal(
                "mlockall", declared, f"RLIMIT_MEMLOCK could not be read ({exc})"
            )
        )
        return
    soft = memlock[0]
    if not _memlock_allows(soft, infinity):
        refusals.append(
            FieldRefusal(
                "mlockall",
                declared,
                f"RLIMIT_MEMLOCK soft {soft}",
            )
        )
        return
    if pid is None:
        refusals.append(FieldRefusal("mlockall", declared, "service has no process id"))
        return
    if not same_process:
        refusals.append(
            FieldRefusal(
                "mlockall",
                declared,
                f"mlockall locks the calling process, not pid {pid}",
            )
        )
        return
    try:
        os_calls.mlockall()
    except OSError as exc:
        refusals.append(FieldRefusal("mlockall", declared, str(exc)))
        return
    applied.append("mlockall")


def _service_pid(service: object) -> int | None:
    """Positive pid for a service process. Zero and other values are ignored.

    A pid of 0 would mean the calling process for some kernel calls. The
    supervisor does not use that to schedule itself.
    """
    pid = getattr(service, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return pid


def _ignored_message(platform: str, mode: str) -> str:
    return (
        f"scheduling config ignored on {platform} in {mode} mode; "
        "declared policy and priority were not applied"
    )


class SchedulingController:
    """Supervisor-side scheduling: ignore once, or apply and record refusals."""

    def __init__(
        self,
        *,
        mode: str,
        platform: str,
        os_calls: OsCalls | None,
        record: Callable[[str, str, str], None],
    ) -> None:
        self.mode = mode
        self.platform = platform
        self.metrics = SchedulingMetrics()
        self.refusals: list[SchedulingRefusal] = []
        self._os = os_calls
        self._record = record
        self._warned = False
        self._ignored: set[str] = set()
        self._delegated: set[str] = set()
        self._waiting: set[str] = set()

    def prepare(self, spec: ServiceSpec, service: object) -> None:
        """Before start. Dev mode and non-Linux warn once and apply nothing.

        On Linux, a permitted ``mlockall`` is handed to a service that
        locks itself, before that process is created.
        """
        hint = spec.scheduling
        if hint is None or not hint.requests_os_change():
            return
        role = spec.role
        if self.mode == "dev" or self.platform != "linux":
            self._ignore(role, hint)
            return
        self._delegated.discard(role)
        if not hint.mlockall:
            return
        if not self._memlock_permitted():
            return
        request = getattr(service, "request_memory_lock", None)
        if callable(request) and request() is True:
            self._delegated.add(role)
            self._waiting.add(role)

    def finish(self, spec: ServiceSpec, service: object) -> None:
        """After start. Apply niceness, policy, affinity, and an in-process lock."""
        hint = spec.scheduling
        if hint is None or not hint.requests_os_change():
            return
        role = spec.role
        if role in self._ignored:
            return
        pid = _service_pid(service)
        outcome = apply_scheduling(
            hint,
            pid=pid,
            os_calls=self._require_os(),
            same_process=pid is not None and pid == os.getpid(),
            mlock_delegated=role in self._delegated,
        )
        for detail in outcome.applied:
            self._note_applied(role, detail)
        for refusal in outcome.refusals:
            self._note_refused(role, refusal)

    def collect(
        self, spec: ServiceSpec, service: object, *, final: bool = False
    ) -> None:
        """Record a delegated ``mlockall`` result once the service reports it."""
        role = spec.role
        if role not in self._waiting:
            return
        report = getattr(service, "memory_lock_report", None)
        text = ""
        if callable(report):
            got = report()
            if got is not None:
                text = str(got).strip()
        if text:
            self._waiting.discard(role)
            if text == "ok":
                self._note_applied(role, "mlockall")
                return
            self._note_refused(role, FieldRefusal("mlockall", "mlockall", text))
            return
        state = getattr(getattr(service, "state", None), "value", "")
        if final or state in ("failed", "stopped"):
            self._waiting.discard(role)
            self._note_refused(
                role,
                FieldRefusal(
                    "mlockall", "mlockall", "mlockall result was not reported"
                ),
            )

    def _ignore(self, role: str, hint: SchedulingHint) -> None:
        if role in self._ignored:
            return
        self._ignored.add(role)
        if not self._warned:
            self._warned = True
            self.metrics.inc(SCHEDULING_IGNORED)
            _LOG.warning("%s", _ignored_message(self.platform, self.mode))
        self._record("scheduling_ignored", role, describe_scheduling(hint))

    def _memlock_permitted(self) -> bool:
        try:
            soft, _hard = self._require_os().rlimit_memlock()
            infinity = self._require_os().rlimit_infinity()
        except OSError:
            return False
        return _memlock_allows(soft, infinity)

    def _require_os(self) -> OsCalls:
        if self._os is None:
            self._os = LinuxOsCalls()
        return self._os

    def _note_applied(self, role: str, detail: str) -> None:
        self.metrics.inc(SCHEDULING_APPLIED)
        self._record("scheduling_applied", role, detail)

    def _note_refused(self, role: str, refusal: FieldRefusal) -> None:
        recorded = SchedulingRefusal(
            role=role,
            field=refusal.field,
            declared=refusal.declared,
            reason=refusal.reason,
        )
        self.refusals.append(recorded)
        self.metrics.inc(SCHEDULING_REFUSED)
        _LOG.warning(
            "%s",
            "scheduling refused for "
            f"{role}: {refusal.declared} was not applied "
            f"({refusal.reason}); left unchanged",
        )
        self._record(
            "scheduling_refused",
            role,
            f"{refusal.declared}: {refusal.reason}",
        )


def scheduling_metrics_map(metrics: SchedulingMetrics) -> Mapping[str, int]:
    """Snapshot helper for callers that only need the mapping."""
    return metrics.snapshot()


__all__ = [
    "SCHEDULING_APPLIED",
    "SCHEDULING_IGNORED",
    "SCHEDULING_REFUSED",
    "ApplyOutcome",
    "FieldRefusal",
    "OsCalls",
    "SchedulingController",
    "SchedulingMetrics",
    "SchedulingRefusal",
    "apply_scheduling",
    "describe_scheduling",
]
