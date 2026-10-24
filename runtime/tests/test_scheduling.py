"""Per-service scheduling. OS calls are fakes, so these tests do not need Linux.

The privileged test runs when this process is root or holds ``CAP_SYS_NICE``.
When the same probe the scheduler uses allows the declared realtime priority,
that test reads ``/proc/<pid>/sched`` and ``taskset``. When ``CAP_SYS_NICE``
is absent and ``RLIMIT_RTPRIO`` does not allow the priority, it expects that
refusal and checks that ``SCHED_FIFO`` was not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from ohho.robot_spec import RobotClass

from ohho_runtime.manifest import (
    Criticality,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    SchedulePolicy,
    SchedulingHint,
    ServiceSpec,
    class_default_manifest,
    load_manifest,
    make_service,
)
from ohho_runtime.os_sched import (
    MLOCK_ENV,
    MLOCK_STATUS_ENV,
    LinuxOsCalls,
    cap_sys_nice_from_status,
    lock_memory_if_requested,
    privileged_linux,
)
from ohho_runtime.scheduling import (
    SCHEDULING_APPLIED,
    SCHEDULING_IGNORED,
    SCHEDULING_REFUSED,
    FieldRefusal,
    OsCalls,
    SchedulingRefusal,
    _limit_allows,
    _probe,
    apply_scheduling,
)
from ohho_runtime.service import LifecycleState, ManualClock, ServiceBase
from ohho_runtime.service.subprocess import SubprocessService
from ohho_runtime.supervisor import Supervisor, SupervisorState

# Kernel prio for a realtime task is MAX_RT_PRIO - 1 - sched_priority.
_MAX_RT_PRIO = 100


def _noop() -> None:
    return None


def _entry() -> PythonCallable:
    return PythonCallable(_noop)


def parse_proc_sched(text: str) -> dict[str, int]:
    """Pull ``policy`` and ``prio`` out of a ``/proc/<pid>/sched`` body."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        name = key.strip()
        if name not in ("policy", "prio"):
            continue
        found[name] = int(value.strip())
    return found


def parse_taskset_mask(text: str) -> int:
    """Parse the hex mask from ``taskset -p`` output."""
    line = ""
    for candidate in text.splitlines():
        if candidate.strip():
            line = candidate.strip()
    if ":" not in line:
        raise ValueError(f"taskset output has no mask: {text!r}")
    raw = line.rsplit(":", 1)[1].strip().lower()
    if raw.startswith("0x"):
        raw = raw[2:]
    return int(raw, 16)


class FakeOs:
    """Records scheduling calls. Refusals raise ``OSError`` and store nothing."""

    def __init__(
        self,
        *,
        cap_sys_nice: bool = False,
        rtprio: tuple[int, int] = (0, 0),
        memlock: tuple[int, int] = (0, 0),
        infinity: int = -1,
        nice: int = 0,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self.cap = cap_sys_nice
        self._rtprio = rtprio
        self._memlock = memlock
        self._infinity = infinity
        self._nice = nice
        self.fail = set(fail)
        self.calls: list[tuple[object, ...]] = []
        self.nice_values: dict[int, int] = {}
        self.scheduler: dict[int, tuple[str, int]] = {}
        self.affinity: dict[int, tuple[int, ...]] = {}
        self.locked = False

    def cap_sys_nice(self) -> bool:
        self.calls.append(("cap_sys_nice",))
        return self.cap

    def rlimit_rtprio(self) -> tuple[int, int]:
        self.calls.append(("rlimit_rtprio",))
        return self._rtprio

    def rlimit_memlock(self) -> tuple[int, int]:
        self.calls.append(("rlimit_memlock",))
        return self._memlock

    def rlimit_infinity(self) -> int:
        return self._infinity

    def get_nice(self, pid: int) -> int:
        self.calls.append(("get_nice", pid))
        return self.nice_values.get(pid, self._nice)

    def set_nice(self, pid: int, value: int) -> None:
        self.calls.append(("set_nice", pid, value))
        if "nice" in self.fail:
            raise PermissionError(1, "Operation not permitted")
        self.nice_values[pid] = value

    def set_scheduler(self, pid: int, policy: str, priority: int) -> None:
        self.calls.append(("set_scheduler", pid, policy, priority))
        if "scheduler" in self.fail:
            raise PermissionError(1, "Operation not permitted")
        self.scheduler[pid] = (policy, priority)

    def set_affinity(self, pid: int, cpus: tuple[int, ...]) -> None:
        self.calls.append(("set_affinity", pid, tuple(cpus)))
        if "affinity" in self.fail:
            raise PermissionError(1, "Operation not permitted")
        self.affinity[pid] = tuple(cpus)

    def mlockall(self) -> None:
        self.calls.append(("mlockall",))
        if "mlockall" in self.fail:
            raise PermissionError(1, "Cannot allocate memory")
        self.locked = True

    def of(self, name: str) -> list[tuple[object, ...]]:
        return [call for call in self.calls if call[0] == name]


class _PidService(ServiceBase):
    """In-process lifecycle with a caller-supplied pid. No thread and no sleep."""

    def __init__(
        self,
        pid: int,
        *,
        delegate_lock: bool = False,
        report: str | None = None,
        clock: Callable[[], float] | None = None,
        readiness_timeout_s: float = 5.0,
        liveness_timeout_s: float = 1.0,
        grace_s: float = 1.0,
    ) -> None:
        super().__init__(
            clock=clock,
            readiness_timeout_s=readiness_timeout_s,
            liveness_timeout_s=liveness_timeout_s,
            grace_s=grace_s,
        )
        self._os_pid = pid
        self._delegate = delegate_lock
        self._report = report
        self.lock_requested = False
        self.starts = 0

    @property
    def pid(self) -> int:
        return self._os_pid

    def request_memory_lock(self) -> bool:
        self.lock_requested = True
        return self._delegate

    def memory_lock_report(self) -> str | None:
        return self._report

    def on_start(self) -> None:
        self.starts += 1
        self.mark_ready()
        self.heartbeat()

    def on_stop(self) -> None:
        return None

    def crash(self) -> None:
        self._fail("crashed")


class _PidFactory:
    def __init__(
        self,
        pid: int,
        *,
        delegate_lock: bool = False,
        report: str | None = None,
    ) -> None:
        self._pid = pid
        self._delegate = delegate_lock
        self._report = report
        self.service: _PidService | None = None

    def __call__(
        self,
        spec: ServiceSpec,
        *,
        clock: Callable[[], float],
        grace_s: float,
    ) -> _PidService:
        service = _PidService(
            self._pid,
            delegate_lock=self._delegate,
            report=self._report,
            clock=clock,
            readiness_timeout_s=spec.readiness_timeout_s,
            liveness_timeout_s=spec.liveness_timeout_s,
            grace_s=grace_s,
        )
        self.service = service
        return service


def _service(
    hint: SchedulingHint | None,
    *,
    role: str = "perception",
    criticality: Criticality = Criticality.DEGRADABLE,
    restart: RestartPolicy | None = None,
    depends_on: tuple[str, ...] = (),
) -> ServiceSpec:
    return make_service(
        role,
        _entry(),
        depends_on=depends_on,
        criticality=criticality,
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy() if restart is None else restart,
        loop_hz=1.0,
        scheduling=hint,
    )


def _full_hint() -> SchedulingHint:
    return SchedulingHint(
        policy=SchedulePolicy.FIFO,
        priority=20,
        nice=-5,
        cpu_affinity=(0, 1),
        mlockall=True,
    )


class ParseTest(unittest.TestCase):
    def test_proc_sched_policy_and_prio(self) -> None:
        text = (
            "python (42, #threads: 1)\n"
            "-------------------------------------------------\n"
            "se.exec_start                                :         0.000000\n"
            "policy                                       :                    1\n"
            "prio                                         :                   98\n"
        )
        fields = parse_proc_sched(text)
        self.assertEqual(fields["policy"], 1)
        self.assertEqual(fields["prio"], 98)

    def test_taskset_mask(self) -> None:
        self.assertEqual(parse_taskset_mask("pid 42's current affinity mask: 1\n"), 1)
        self.assertEqual(
            parse_taskset_mask("pid 42's current affinity mask: 0xf\n"), 15
        )

    def test_cap_sys_nice_bit(self) -> None:
        mask = 1 << 23
        self.assertTrue(cap_sys_nice_from_status(f"CapEff:\t{mask:016x}\n"))
        self.assertFalse(cap_sys_nice_from_status("CapEff:\t0\n"))
        self.assertFalse(cap_sys_nice_from_status("Name:\tpython\n"))


@unittest.skipUnless(sys.platform != "linux", "a Linux host may be privileged")
class PrivilegeProbeTest(unittest.TestCase):
    def test_probe_is_false_off_linux(self) -> None:
        self.assertFalse(privileged_linux())


class ApplyTest(unittest.TestCase):
    def test_cap_sets_the_declared_priority_without_clamping(self) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(5, 5))
        hint = SchedulingHint(policy="fifo", priority=20)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.scheduler[7], ("fifo", 20))
        self.assertEqual(outcome.refusals, ())
        self.assertEqual(fake.of("set_scheduler"), [("set_scheduler", 7, "fifo", 20)])

    def test_rlimit_allows_the_declared_priority_without_the_cap(self) -> None:
        fake = FakeOs(cap_sys_nice=False, rtprio=(20, 20))
        hint = SchedulingHint(policy="rr", priority=20)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.scheduler[7], ("rr", 20))
        self.assertEqual(outcome.refusals, ())

    def test_unlimited_rlimit_allows_the_declared_priority(self) -> None:
        fake = FakeOs(cap_sys_nice=False, rtprio=(-1, -1), infinity=-1)
        hint = SchedulingHint(policy="fifo", priority=40)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.scheduler[7], ("fifo", 40))
        self.assertEqual(outcome.refusals, ())

    def test_does_not_clamp_priority_to_the_rlimit(self) -> None:
        fake = FakeOs(cap_sys_nice=False, rtprio=(5, 5))
        hint = SchedulingHint(policy="fifo", priority=20)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.of("set_scheduler"), [])
        self.assertEqual(fake.scheduler, {})
        self.assertEqual(len(outcome.refusals), 1)
        refusal = outcome.refusals[0]
        self.assertEqual(refusal.field, "policy")
        self.assertIn("priority 20", refusal.declared)
        self.assertIn("CAP_SYS_NICE not held", refusal.reason)
        self.assertIn("RLIMIT_RTPRIO soft 5", refusal.reason)

    def test_soft_zero_rlimit_refuses_fifo_priority_one(self) -> None:
        fake = FakeOs(cap_sys_nice=False, rtprio=(0, 0))
        priority = 1
        self.assertFalse(_realtime_allowed(fake, priority))
        hint = SchedulingHint(policy="fifo", priority=priority)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.of("set_scheduler"), [])
        self.assertEqual(fake.scheduler, {})
        self.assertEqual(
            outcome.refusals,
            (
                FieldRefusal(
                    "policy",
                    "policy fifo priority 1",
                    "CAP_SYS_NICE not held, RLIMIT_RTPRIO soft 0",
                ),
            ),
        )

    def test_realtime_gate_follows_cap_or_rlimit(self) -> None:
        self.assertTrue(_realtime_allowed(FakeOs(cap_sys_nice=True, rtprio=(0, 0)), 1))
        self.assertTrue(_realtime_allowed(FakeOs(cap_sys_nice=False, rtprio=(1, 1)), 1))
        self.assertTrue(
            _realtime_allowed(
                FakeOs(cap_sys_nice=False, rtprio=(-1, -1), infinity=-1), 1
            )
        )
        self.assertFalse(
            _realtime_allowed(FakeOs(cap_sys_nice=False, rtprio=(0, 99)), 1)
        )

    def test_scheduler_error_does_not_store_a_different_priority(self) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(99, 99), fail=frozenset({"scheduler"}))
        hint = SchedulingHint(policy="fifo", priority=20)
        outcome = apply_scheduling(
            hint,
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.of("set_scheduler"), [("set_scheduler", 7, "fifo", 20)])
        self.assertEqual(fake.scheduler, {})
        self.assertEqual(outcome.applied, ())
        self.assertEqual(len(outcome.refusals), 1)

    def test_raising_priority_via_nice_needs_cap_sys_nice(self) -> None:
        fake = FakeOs(cap_sys_nice=False, nice=0)
        outcome = apply_scheduling(
            SchedulingHint(nice=-5),
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.of("set_nice"), [])
        self.assertIn("CAP_SYS_NICE not held", outcome.refusals[0].reason)
        self.assertIn("nice -5", outcome.refusals[0].declared)

    def test_lowering_priority_via_nice_does_not_need_the_cap(self) -> None:
        fake = FakeOs(cap_sys_nice=False, nice=0)
        outcome = apply_scheduling(
            SchedulingHint(nice=5),
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.nice_values[7], 5)
        self.assertEqual(outcome.refusals, ())

    def test_affinity_is_the_declared_set(self) -> None:
        fake = FakeOs()
        outcome = apply_scheduling(
            SchedulingHint(cpu_affinity=(1, 3)),
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.affinity[7], (1, 3))
        self.assertEqual(outcome.applied, ("cpu_affinity=1,3",))

    def test_affinity_failure_leaves_the_mask_unchanged(self) -> None:
        fake = FakeOs(fail=frozenset({"affinity"}))
        outcome = apply_scheduling(
            SchedulingHint(cpu_affinity=(0, 1)),
            pid=7,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertNotIn(7, fake.affinity)
        self.assertEqual(outcome.refusals[0].field, "cpu_affinity")
        self.assertEqual(outcome.applied, ())

    def test_mlockall_runs_only_in_the_service_process(self) -> None:
        fake = FakeOs(memlock=(4096, 4096))
        pid = os.getpid()
        outcome = apply_scheduling(
            SchedulingHint(mlockall=True),
            pid=pid,
            os_calls=fake,
            same_process=True,
            mlock_delegated=False,
        )
        self.assertTrue(fake.locked)
        self.assertEqual(outcome.applied, ("mlockall",))
        other = FakeOs(memlock=(4096, 4096))
        refused = apply_scheduling(
            SchedulingHint(mlockall=True),
            pid=7,
            os_calls=other,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertFalse(other.locked)
        self.assertIn("not pid 7", refused.refusals[0].reason)

    def test_mlockall_rlimit_zero_does_not_call(self) -> None:
        fake = FakeOs(memlock=(0, 0))
        outcome = apply_scheduling(
            SchedulingHint(mlockall=True),
            pid=os.getpid(),
            os_calls=fake,
            same_process=True,
            mlock_delegated=False,
        )
        self.assertFalse(fake.locked)
        self.assertIn("RLIMIT_MEMLOCK soft 0", outcome.refusals[0].reason)

    def test_missing_pid_refuses_every_declared_field(self) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(99, 99), memlock=(4096, 4096))
        hint = SchedulingHint(
            policy="rr",
            priority=3,
            nice=1,
            cpu_affinity=(0,),
            mlockall=True,
        )
        outcome = apply_scheduling(
            hint,
            pid=None,
            os_calls=fake,
            same_process=False,
            mlock_delegated=False,
        )
        self.assertEqual(fake.of("set_scheduler"), [])
        self.assertEqual(fake.of("set_nice"), [])
        self.assertEqual(fake.of("set_affinity"), [])
        self.assertEqual(fake.of("mlockall"), [])
        self.assertEqual(
            {item.field for item in outcome.refusals},
            {"nice", "policy", "cpu_affinity", "mlockall"},
        )
        for refusal in outcome.refusals:
            self.assertIn("service has no process id", refusal.reason)


class MemoryLockReportTest(unittest.TestCase):
    def test_unset_request_does_not_lock(self) -> None:
        called: list[int] = []
        result = lock_memory_if_requested({}, lambda: called.append(1))
        self.assertIsNone(result)
        self.assertEqual(called, [])

    def test_refusal_is_logged_and_written(self) -> None:
        def boom() -> None:
            raise OSError(12, "Cannot allocate memory")

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "mlock.txt")
            env = {MLOCK_ENV: "1", MLOCK_STATUS_ENV: path}
            with self.assertLogs("ohho_runtime.os_sched", level="WARNING") as logs:
                result = lock_memory_if_requested(env, boom)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.startswith("refused:"))
            self.assertIn("refused:", Path(path).read_text(encoding="utf-8"))
            self.assertTrue(any("mlockall refused" in line for line in logs.output))

    def test_subprocess_service_requests_a_lock_without_starting(self) -> None:
        service = SubprocessService(
            [sys.executable, "-c", "pass"],
            clock=ManualClock(),
        )
        self.assertIsNone(service.pid)
        self.assertFalse(service.memory_lock_requested)
        self.assertTrue(service.request_memory_lock())
        self.assertTrue(service.memory_lock_requested)
        self.assertIsNone(service.memory_lock_report())


class SupervisorSchedulingTest(unittest.TestCase):
    def _ignored(self, *, mode: str, platform: str) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(99, 99), memlock=(4096, 4096))
        clock = ManualClock()
        manifest = load_manifest(
            [
                _service(_full_hint(), role="perception"),
                _service(
                    _full_hint(),
                    role="recorder",
                    criticality=Criticality.OPTIONAL,
                    depends_on=("perception",),
                ),
            ]
        )
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            mode=mode,
            platform=platform,
            os_calls=fake,
        )
        with self.assertLogs("ohho_runtime.scheduling", level="WARNING") as logs:
            sup.start()
        ignored = [line for line in logs.output if "scheduling config ignored" in line]
        self.assertEqual(len(ignored), 1)
        self.assertIn(platform, ignored[0])
        self.assertIn(mode, ignored[0])
        self.assertNotIn("scheduling refused", "\n".join(logs.output))
        self.assertEqual(fake.of("set_scheduler"), [])
        self.assertEqual(fake.of("set_nice"), [])
        self.assertEqual(fake.of("set_affinity"), [])
        self.assertEqual(fake.of("mlockall"), [])
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_IGNORED], 1)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 0)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_APPLIED], 0)
        self.assertEqual(len(sup.events.of("scheduling_ignored")), 2)
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertEqual(clock.now, 0.0)
        self.assertIs(sup.service("perception").state, LifecycleState.READY)
        self.assertIs(sup.service("recorder").state, LifecycleState.READY)

    def test_macos_and_windows_ignore_scheduling_once(self) -> None:
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform):
                self._ignored(mode="process", platform=platform)

    def test_dev_mode_ignores_scheduling_once_on_linux(self) -> None:
        self._ignored(mode="dev", platform="linux")

    def test_defaults_do_not_change_scheduling(self) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(99, 99), memlock=(4096, 4096))
        clock = ManualClock()
        sup = Supervisor(
            class_default_manifest(RobotClass.MOBILE_BASE),
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            mode="process",
            platform="linux",
            os_calls=fake,
        )
        with self.assertNoLogs("ohho_runtime.scheduling", level="WARNING"):
            sup.start()
        self.assertEqual(fake.calls, [])
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertEqual(clock.now, 0.0)

    def test_unprivileged_refusal_still_starts_the_service(self) -> None:
        fake = FakeOs(
            cap_sys_nice=False,
            rtprio=(0, 0),
            memlock=(0, 0),
            fail=frozenset({"affinity"}),
        )
        clock = ManualClock()
        factory = _PidFactory(4242)
        sup = Supervisor(
            load_manifest([_service(_full_hint())]),
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            mode="process",
            platform="linux",
            os_calls=fake,
            service_factory=factory,
        )
        with self.assertLogs("ohho_runtime.scheduling", level="WARNING") as logs:
            sup.start()
        refused = [line for line in logs.output if "scheduling refused" in line]
        self.assertGreaterEqual(len(refused), 1)
        joined = "\n".join(refused)
        self.assertIn("priority 20", joined)
        self.assertIn("left unchanged", joined)
        self.assertIn("CAP_SYS_NICE", joined)
        self.assertIn("RLIMIT_RTPRIO", joined)
        self.assertIn("RLIMIT_MEMLOCK", joined)
        self.assertNotIn("scheduling config ignored", joined)
        self.assertEqual(fake.of("set_scheduler"), [])
        self.assertEqual(fake.of("set_nice"), [])
        self.assertEqual(fake.of("mlockall"), [])
        self.assertEqual(fake.scheduler, {})
        self.assertEqual(fake.nice_values, {})
        self.assertFalse(fake.locked)
        self.assertNotIn(4242, fake.affinity)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 4)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_APPLIED], 0)
        self.assertEqual(sup.scheduling_refusals()[0].role, "perception")
        self.assertIs(sup.service("perception").state, LifecycleState.READY)
        self.assertIs(sup.state, SupervisorState.RUNNING)
        self.assertEqual(clock.now, 0.0)
        service = factory.service
        self.assertIsNotNone(service)
        assert service is not None
        self.assertFalse(service.lock_requested)

    def test_linux_applies_the_declared_values(self) -> None:
        pid = os.getpid()
        fake = FakeOs(
            cap_sys_nice=True,
            rtprio=(99, 99),
            memlock=(4096, 4096),
            nice=0,
        )
        factory = _PidFactory(pid)
        hint = SchedulingHint(
            policy=SchedulePolicy.FIFO,
            priority=12,
            nice=-2,
            cpu_affinity=(0, 2),
            mlockall=True,
        )
        sup = Supervisor(
            load_manifest([_service(hint)]),
            robot_class=RobotClass.MOBILE_BASE,
            clock=ManualClock(),
            mode="process",
            platform="linux",
            os_calls=fake,
            service_factory=factory,
        )
        sup.start()
        self.assertEqual(fake.scheduler[pid], ("fifo", 12))
        self.assertEqual(fake.nice_values[pid], -2)
        self.assertEqual(fake.affinity[pid], (0, 2))
        self.assertTrue(fake.locked)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 0)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_APPLIED], 4)
        self.assertIs(sup.service("perception").state, LifecycleState.READY)
        details = [event.detail for event in sup.events.of("scheduling_applied")]
        self.assertIn("policy=fifo priority=12", details)

    def test_delegated_mlock_reports_ok_without_locking_the_caller(self) -> None:
        fake = FakeOs(memlock=(4096, 4096))
        factory = _PidFactory(4242, delegate_lock=True, report="ok")
        sup = Supervisor(
            load_manifest([_service(SchedulingHint(mlockall=True))]),
            robot_class=RobotClass.MOBILE_BASE,
            clock=ManualClock(),
            mode="process",
            platform="linux",
            os_calls=fake,
            service_factory=factory,
        )
        sup.start()
        self.assertFalse(fake.locked)
        self.assertEqual(fake.of("mlockall"), [])
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_APPLIED], 1)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 0)
        self.assertIn("mlockall", sup.events.of("scheduling_applied")[0].detail)
        service = factory.service
        assert service is not None
        self.assertTrue(service.lock_requested)

    def test_missing_mlock_report_is_a_refusal_at_shutdown(self) -> None:
        fake = FakeOs(memlock=(4096, 4096))
        sup = Supervisor(
            load_manifest([_service(SchedulingHint(mlockall=True))]),
            robot_class=RobotClass.MOBILE_BASE,
            clock=ManualClock(),
            mode="process",
            platform="linux",
            os_calls=fake,
            service_factory=_PidFactory(4242, delegate_lock=True, report=None),
        )
        sup.start()
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 0)
        self.assertIs(sup.service("perception").state, LifecycleState.READY)
        with self.assertLogs("ohho_runtime.scheduling", level="WARNING") as logs:
            sup.shutdown()
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 1)
        self.assertTrue(any("not reported" in line for line in logs.output))
        self.assertIn("left unchanged", logs.output[0])

    def test_restart_applies_the_same_priority_again(self) -> None:
        fake = FakeOs(cap_sys_nice=True, rtprio=(99, 99))
        factory = _PidFactory(4242)
        clock = ManualClock()
        sup = Supervisor(
            load_manifest(
                [
                    _service(
                        SchedulingHint(policy="fifo", priority=10),
                        restart=RestartPolicy(
                            RestartKind.ON_FAILURE,
                            max_restarts=3,
                            window_s=60.0,
                            backoff_s=0.5,
                        ),
                    )
                ]
            ),
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            mode="process",
            platform="linux",
            os_calls=fake,
            service_factory=factory,
        )
        sup.start()
        service = factory.service
        assert service is not None
        service.crash()
        sup.poll()
        self.assertEqual(
            fake.of("set_scheduler"), [("set_scheduler", 4242, "fifo", 10)]
        )
        clock.advance(0.5)
        sup.poll()
        self.assertEqual(
            fake.of("set_scheduler"),
            [
                ("set_scheduler", 4242, "fifo", 10),
                ("set_scheduler", 4242, "fifo", 10),
            ],
        )
        self.assertEqual(service.starts, 2)
        self.assertEqual(sup.scheduling_metrics[SCHEDULING_REFUSED], 0)
        self.assertIs(sup.service("perception").state, LifecycleState.READY)


def _realtime_allowed(os_calls: OsCalls, priority: int) -> bool:
    """Same gate as policy application: capability, or an rlimit that allows it.

    Root without ``CAP_SYS_NICE`` and with ``RLIMIT_RTPRIO`` soft 0 cannot
    set ``SCHED_FIFO``. The probe is :func:`_probe` and :func:`_limit_allows`.
    """
    try:
        cap, rtprio, _memlock, infinity = _probe(os_calls)
    except OSError:
        return False
    return bool(cap) or _limit_allows(rtprio[0], priority, infinity)


def _child_env() -> dict[str, str]:
    keys = ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "SYSTEMROOT")
    return {key: os.environ[key] for key in keys if key in os.environ}


@unittest.skipUnless(
    privileged_linux(),
    "needs root or CAP_SYS_NICE on Linux",
)
class PrivilegedLinuxTest(unittest.TestCase):
    def test_policy_priority_and_affinity_are_visible(self) -> None:
        priority = 1
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertIsNone(proc.poll())
            assert proc.pid is not None
            cpus = sorted(os.sched_getaffinity(proc.pid))
            self.assertTrue(cpus)
            cpu = cpus[0]
            hint = SchedulingHint(
                policy=SchedulePolicy.FIFO,
                priority=priority,
                cpu_affinity=(cpu,),
            )
            calls = LinuxOsCalls()
            sup = Supervisor(
                load_manifest([_service(hint)]),
                robot_class=RobotClass.MOBILE_BASE,
                clock=ManualClock(),
                mode="process",
                platform="linux",
                os_calls=calls,
                service_factory=_PidFactory(proc.pid),
            )
            sup.start()
            if _realtime_allowed(calls, priority):
                self.assertEqual(
                    sup.scheduling_refusals(),
                    (),
                    sup.scheduling_refusals(),
                )
                sched_text = Path(f"/proc/{proc.pid}/sched").read_text(encoding="utf-8")
                fields = parse_proc_sched(sched_text)
                self.assertEqual(fields["policy"], os.SCHED_FIFO, sched_text)
                self.assertEqual(
                    fields["prio"], _MAX_RT_PRIO - 1 - priority, sched_text
                )
                taskset = subprocess.run(
                    ["taskset", "-p", str(proc.pid)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(taskset.returncode, 0, taskset.stderr)
                self.assertEqual(parse_taskset_mask(taskset.stdout), 1 << cpu)
            else:
                cap, rtprio, _memlock, infinity = _probe(calls)
                soft = rtprio[0]
                self.assertFalse(bool(cap) or _limit_allows(soft, priority, infinity))
                policy_refusals = [
                    item for item in sup.scheduling_refusals() if item.field == "policy"
                ]
                self.assertEqual(
                    policy_refusals,
                    [
                        SchedulingRefusal(
                            role="perception",
                            field="policy",
                            declared=f"policy fifo priority {priority}",
                            reason=(
                                f"CAP_SYS_NICE not held, RLIMIT_RTPRIO soft {soft}"
                            ),
                        )
                    ],
                    sup.scheduling_refusals(),
                )
                sched_text = Path(f"/proc/{proc.pid}/sched").read_text(encoding="utf-8")
                fields = parse_proc_sched(sched_text)
                self.assertIn("policy", fields, sched_text)
                self.assertNotEqual(fields["policy"], os.SCHED_FIFO, sched_text)
            self.assertIs(sup.service("perception").state, LifecycleState.READY)
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    unittest.main()
