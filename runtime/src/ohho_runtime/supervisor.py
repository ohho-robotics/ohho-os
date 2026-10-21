"""Process supervisor: dependency order, restarts, and shutdown.

Dev mode runs every service in-process on the caller's thread. It does
not spawn processes, open sockets, or install signal handlers, so the
same tests run on Linux, macOS, and Windows. The host calls
:meth:`Supervisor.poll`. Nothing here sleeps.

A service starts only once every dependency is ``ready``. Restarts use
:class:`~ohho_runtime.backoff.ExponentialBackoff` and the manifest's
``max_restarts`` / ``window_s``. The failure that reaches ``max_restarts``
inside the window is not restarted. With ``max_restarts=3`` and
``window_s=60``, the third failure inside 60 seconds stops restarts.

A critical service that hits that cap latches do-not-operate. The
supervisor holds the class safe action and refuses motion until
:meth:`Supervisor.clear_do_not_operate`. The hook is the extension point
for a later policy. This module does not command the robot and does not
change the safety gate.

A ``never`` restart policy is not restarted and does not latch
do-not-operate. Dependents of a terminal failure are not started.

A non-zero exit, an unhandled exception, a heartbeat timeout that
schedules a restart, or a restart limit writes one crash bundle.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass

from ohho_runtime.backoff import ExponentialBackoff
from ohho_runtime.crash import (
    DEFAULT_LOG_LIMIT,
    LogHistory,
    classify_crash,
    read_exit_code,
    read_failure_exception,
    write_crash_bundle,
)
from ohho_runtime.log import (
    RuntimeLog,
    note_exception,
    resolve_log_format,
    resolve_log_level,
    resolve_run_id,
)
from ohho_runtime.manifest import (
    Criticality,
    PythonCallable,
    RestartKind,
    RuntimeManifest,
    ServiceSpec,
    resolve_manifest,
)
from ohho_runtime.metrics import MetricsEndpoint, Registry, get_registry, publish_loop
from ohho_runtime.service import LifecycleState, ServiceBase
from ohho_runtime.timing import LoopTimer, loop_report


class SupervisorError(Exception):
    """The supervisor could not apply an operation."""


class SupervisorState(Enum):
    """Supervisor lifecycle. Do-not-operate is a latch, not a health probe."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    DO_NOT_OPERATE = "do_not_operate"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ClassSafeAction(Enum):
    """Class hold applied while do-not-operate is latched.

    The supervisor records the hold. It does not send a command.
    """

    STOP = "stop"
    STOP_BASE_HOLD_ARM = "stop_base_hold_arm"
    HOLD = "hold"
    DO_NOT_OPERATE = "do_not_operate"


_EVENT_LEVEL = {
    "stop_failed": "ERROR",
    "service_failed": "ERROR",
    "restart_limit": "ERROR",
    "do_not_operate": "ERROR",
    "grace_exceeded": "WARNING",
    "restart_refused": "WARNING",
    "blocked": "WARNING",
    "safe_hold": "WARNING",
}


_CLASS_SAFE: dict[RobotClass, ClassSafeAction] = {
    RobotClass.MOBILE_BASE: ClassSafeAction.STOP,
    RobotClass.MOBILE_MANIPULATOR: ClassSafeAction.STOP_BASE_HOLD_ARM,
    RobotClass.ARM: ClassSafeAction.HOLD,
    RobotClass.QUADRUPED: ClassSafeAction.STOP,
    RobotClass.AERIAL: ClassSafeAction.DO_NOT_OPERATE,
    RobotClass.HUMANOID: ClassSafeAction.STOP,
}


def class_safe_action(robot_class: RobotClass) -> ClassSafeAction:
    """Return the safe hold for ``robot_class``.

    Quadruped stop does not sit. Humanoid stop does not stand. Aerial
    do-not-operate sends no setpoints; landing stays with a later policy.
    """
    if not isinstance(robot_class, RobotClass):
        raise SupervisorError(
            f"robot_class must be a RobotClass, not {type(robot_class).__name__}"
        )
    try:
        return _CLASS_SAFE[robot_class]
    except KeyError:
        raise SupervisorError(f"no class safe action for {robot_class!r}") from None


@dataclass(frozen=True)
class DoNotOperateNotice:
    """Payload for the do-not-operate hook.

    ``allows_motion`` is always false. The hook does not arm the robot.
    """

    robot_class: RobotClass
    safe_action: ClassSafeAction
    role: str
    reason: str
    allows_motion: bool = False


DoNotOperateHook = Callable[[DoNotOperateNotice], None]


@dataclass(frozen=True)
class SupervisorEvent:
    """One supervisor event. ``at`` is the supervisor clock."""

    at: float
    kind: str
    role: str = ""
    detail: str = ""


class EventLog:
    """In-memory runtime event log for supervisor transitions."""

    def __init__(self) -> None:
        self._events: list[SupervisorEvent] = []

    def record(
        self,
        at: float,
        kind: str,
        role: str = "",
        detail: str = "",
    ) -> None:
        """Append one event."""
        self._events.append(SupervisorEvent(float(at), kind, role, detail))

    @property
    def events(self) -> tuple[SupervisorEvent, ...]:
        return tuple(self._events)

    def of(self, kind: str) -> tuple[SupervisorEvent, ...]:
        """Events whose ``kind`` matches."""
        return tuple(event for event in self._events if event.kind == kind)

    def __iter__(self) -> Iterator[SupervisorEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


@dataclass(frozen=True)
class DependencyGraph:
    """Start and stop order for one manifest.

    ``edges`` are ``(dependency, dependent)`` pairs. Start order is the
    manifest's topological order. Stop order is the reverse.
    """

    order: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def start_order(self) -> tuple[str, ...]:
        """Roles to start, dependencies first."""
        return self.order

    def stop_order(self) -> tuple[str, ...]:
        """Roles to stop, the reverse of :meth:`start_order`."""
        return tuple(reversed(self.order))

    def dependencies(self, role: str) -> tuple[str, ...]:
        """Roles ``role`` waits on."""
        self._known(role)
        return tuple(dep for dep, dependent in self.edges if dependent == role)

    def dependents(self, role: str) -> tuple[str, ...]:
        """Roles that wait on ``role``."""
        self._known(role)
        return tuple(dependent for dep, dependent in self.edges if dep == role)

    def _known(self, role: str) -> None:
        if role not in self.order:
            raise SupervisorError(f"unknown service role {role!r}")


def build_dependency_graph(manifest: RuntimeManifest) -> DependencyGraph:
    """Build the start graph from a loaded manifest.

    The manifest has already rejected cycles. An order that places a
    service before one of its dependencies is a supervisor error.
    """
    if not isinstance(manifest, RuntimeManifest):
        raise SupervisorError("manifest must be a RuntimeManifest")
    order = manifest.roles()
    position = {role: index for index, role in enumerate(order)}
    edges: list[tuple[str, str]] = []
    for service in manifest.services:
        for dep in service.depends_on:
            if position[dep] >= position[service.role]:
                raise SupervisorError(
                    f"{service.role} is ordered before its dependency {dep}"
                )
            edges.append((dep, service.role))
    return DependencyGraph(order=order, edges=tuple(edges))


@runtime_checkable
class ServiceBuilder(Protocol):
    """Build the in-process service for one manifest entry."""

    def __call__(
        self,
        spec: ServiceSpec,
        *,
        clock: Callable[[], float],
        grace_s: float,
    ) -> ServiceBase: ...


def _call_target(
    target: Callable[..., object],
    service: ServiceBase,
    *,
    logger: RuntimeLog | None = None,
    metrics: Registry | None = None,
) -> None:
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError) as exc:
        note_exception(
            exc,
            service="supervisor",
            event="target_signature",
            level="DEBUG",
            logger=logger,
            metrics=metrics,
        )
        target()
        return
    positional = [
        item
        for item in parameters.values()
        if item.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    if not positional:
        target()
        return
    target(service)


def _callable_target(spec: ServiceSpec) -> Callable[..., object]:
    entry = spec.entry
    if isinstance(entry, PythonCallable):
        return entry.target
    raise SupervisorError(
        "dev mode runs in-process callables only; "
        f"service {spec.role} has entry {type(entry).__name__}"
    )


class DevService(ServiceBase):
    """Synchronous in-process service.

    No thread, socket, or signal. ``start`` calls the target on the
    supervisor thread. A target that returns without failing is ready.
    """

    def __init__(
        self,
        target: Callable[..., object],
        *,
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
        self._target = target

    def on_start(self) -> None:
        _call_target(self._target, self, logger=self._logger, metrics=self._metrics)
        if self.state is LifecycleState.STARTING:
            self.mark_ready()
            self.heartbeat()

    def on_stop(self) -> None:
        return None


class DevServiceFactory:
    """Default dev-mode builder. In-process callables only."""

    def __call__(
        self,
        spec: ServiceSpec,
        *,
        clock: Callable[[], float],
        grace_s: float,
    ) -> DevService:
        return DevService(
            _callable_target(spec),
            clock=clock,
            readiness_timeout_s=spec.readiness_timeout_s,
            liveness_timeout_s=spec.liveness_timeout_s,
            grace_s=grace_s,
        )


@dataclass
class _Run:
    spec: ServiceSpec
    service: ServiceBase
    backoff: ExponentialBackoff | None
    launched: bool = False
    terminal: bool = False
    failure_latched: bool = False
    ready_open: bool = False
    restarts: int = 0
    restart_at: float | None = None
    failure_times: list[float] = field(default_factory=list)


def _grace(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("grace_s must be a number > 0")
    number = float(value)
    if number <= 0:
        raise ValueError("grace_s must be > 0")
    return number


class Supervisor:
    """Start a manifest in dependency order and stop it in reverse.

    ``mode`` is ``dev``: in-process, no OS-specific process control.
    Pass a clock, usually :class:`~ohho_runtime.service.ManualClock` in
    tests, and call :meth:`poll` after it moves. :meth:`start` does not
    block and does not sleep.
    """

    def __init__(
        self,
        manifest: RuntimeManifest,
        *,
        robot_class: RobotClass,
        clock: Callable[[], float] | None = None,
        grace_s: float = 1.0,
        mode: str = "dev",
        service_factory: ServiceBuilder | None = None,
        on_do_not_operate: DoNotOperateHook | None = None,
        events: EventLog | None = None,
        logger: RuntimeLog | None = None,
        metrics: Registry | None = None,
        run_id: str | None = None,
        metrics_enabled: bool | None = None,
        metrics_host: str | None = None,
        metrics_port: int | None = None,
        crash_dir: str | None = None,
    ) -> None:
        if mode != "dev":
            raise SupervisorError(
                f"supervisor mode {mode!r} is not available; dev is the in-process mode"
            )
        if not isinstance(robot_class, RobotClass):
            raise SupervisorError(
                f"robot_class must be a RobotClass, not {type(robot_class).__name__}"
            )
        self._manifest = manifest
        self._robot_class = robot_class
        self._mode = mode
        self._clock: Callable[[], float] = time.monotonic if clock is None else clock
        self._grace_s = _grace(grace_s)
        self._graph = build_dependency_graph(manifest)
        self._events = EventLog() if events is None else events
        self._on_do_not_operate = (
            _noop_hook if on_do_not_operate is None else on_do_not_operate
        )
        if logger is not None and run_id is None and logger.run_id:
            self._run_id = logger.run_id
        else:
            self._run_id = resolve_run_id(run_id)
        self._metrics = get_registry() if metrics is None else metrics
        if logger is None:
            logger = RuntimeLog(
                service="supervisor",
                robot_class=robot_class.value,
                run_id=self._run_id,
                level=resolve_log_level(manifest.log_level),
                log_format=resolve_log_format(manifest.log_format),
            )
        self._logger = logger
        self._crash_dir = None if crash_dir is None else str(crash_dir)
        attached = self._logger.history
        if attached is None:
            self._log_history: object = LogHistory()
            self._logger.attach_history(self._log_history)
        else:
            self._log_history = attached
        self._endpoint = MetricsEndpoint(
            self._metrics,
            enabled=metrics_enabled,
            host=metrics_host,
            port=metrics_port,
            collect=self.collect_metrics,
        )
        self._loops: dict[str, LoopTimer] = {}
        self._state = SupervisorState.IDLE
        self._dno = False
        self._holding_safe = False
        self._disarmed = True
        self._initial_up = False
        self._started: list[str] = []
        self._stopped: list[str] = []
        self._shut_down = False
        self._shutdown_elapsed_s = 0.0
        factory = DevServiceFactory() if service_factory is None else service_factory
        self._runs: dict[str, _Run] = {}
        for spec in manifest.services:
            backoff = None
            if spec.restart.kind is RestartKind.ON_FAILURE:
                backoff = ExponentialBackoff(spec.restart.backoff_s)
            service = factory(spec, clock=self._clock, grace_s=self._grace_s)
            bind = getattr(service, "bind_log", None)
            if callable(bind):
                bind(
                    self._logger.bind(service=spec.role),
                    service=spec.role,
                    metrics=self._metrics,
                )
            self._runs[spec.role] = _Run(spec=spec, service=service, backoff=backoff)

    @classmethod
    def for_spec(
        cls,
        spec: RobotSpec,
        **kwargs: object,
    ) -> Supervisor:
        """Supervise ``resolve_manifest(spec)`` for that spec's class."""
        if not isinstance(spec, RobotSpec):
            raise SupervisorError(
                f"spec must be a RobotSpec, not {type(spec).__name__}"
            )
        kind = spec.robot_class
        if not isinstance(kind, RobotClass):
            raise SupervisorError("spec.robot_class is not set")
        if "robot_class" in kwargs:
            raise SupervisorError("for_spec takes robot_class from the spec")
        return cls(resolve_manifest(spec), robot_class=kind, **kwargs)  # type: ignore[arg-type]

    @property
    def state(self) -> SupervisorState:
        return self._state

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def graph(self) -> DependencyGraph:
        return self._graph

    @property
    def events(self) -> EventLog:
        return self._events

    @property
    def logger(self) -> RuntimeLog:
        """Structured logger for this supervisor and its services."""
        return self._logger

    @property
    def run_id(self) -> str:
        """Run id stamped on every log record and metric label."""
        return self._run_id

    @property
    def metrics_endpoint(self) -> MetricsEndpoint:
        """``/metrics`` endpoint. Disabled endpoints do not bind."""
        return self._endpoint

    @property
    def grace_s(self) -> float:
        return self._grace_s

    @property
    def robot_class(self) -> RobotClass:
        return self._robot_class

    @property
    def safe_action(self) -> ClassSafeAction:
        """Class hold. Active only while :attr:`holding_safe_state` is set."""
        return class_safe_action(self._robot_class)

    @property
    def holding_safe_state(self) -> bool:
        """True while the class safe hold is latched."""
        return self._holding_safe

    @property
    def do_not_operate(self) -> bool:
        """True until a human clears the do-not-operate latch."""
        return self._dno

    @property
    def disarmed(self) -> bool:
        """External disarm flag. The supervisor never arms."""
        return self._disarmed

    @property
    def started_order(self) -> tuple[str, ...]:
        """Roles that have been launched, dependencies first."""
        return tuple(self._started)

    @property
    def stopped_order(self) -> tuple[str, ...]:
        """Roles stopped by :meth:`shutdown`, reverse of start."""
        return tuple(self._stopped)

    @property
    def shutdown_elapsed_s(self) -> float:
        """Clock time :meth:`shutdown` took. Zero before shutdown."""
        return self._shutdown_elapsed_s

    @property
    def shutdown_within_grace(self) -> bool:
        """True after a shutdown whose elapsed time is within ``grace_s``."""
        return self._shut_down and self._shutdown_elapsed_s <= self._grace_s

    def set_disarmed(self, disarmed: bool) -> None:
        """Record an external disarm flag.

        The supervisor does not arm and does not clear a safety latch.
        A service with ``only_while_disarmed`` restarts only while this
        flag is true.
        """
        if not isinstance(disarmed, bool):
            raise SupervisorError("disarmed must be a bool")
        self._disarmed = disarmed

    def service(self, role: str) -> ServiceBase:
        """Return the live service for ``role``."""
        return self._run(role).service

    def restart_count(self, role: str) -> int:
        """How many times ``role`` has been restarted."""
        return self._run(role).restarts

    def restart_exhausted(self, role: str) -> bool:
        """True when ``role`` will not be restarted again."""
        return self._run(role).terminal

    def allows_motion(self) -> bool:
        """False while do-not-operate, starting, or a critical service is down.

        A manifest entry that commands motion and is marked do-not-operate
        also refuses motion. This does not send a command.
        """
        if self._dno or self._holding_safe:
            return False
        if self._state is not SupervisorState.RUNNING:
            return False
        for run in self._runs.values():
            spec = run.spec
            if spec.commands_motion and spec.do_not_operate:
                return False
            if spec.criticality is not Criticality.CRITICAL:
                continue
            if run.terminal or run.service.state is not LifecycleState.READY:
                return False
        return True

    def start(self) -> None:
        """Begin startup. Call :meth:`poll` to move a clock forward."""
        if self._state is not SupervisorState.IDLE:
            raise SupervisorError(f"cannot start from {self._state.value}")
        self._endpoint.start()
        self._state = SupervisorState.STARTING
        self._note("supervisor_start")
        self.poll()

    def poll(self) -> None:
        """Advance one step: deadlines, due restarts, then ready dependents.

        Does not sleep. A service that is not ready blocks its dependents
        until a later poll sees ``ready``.
        """
        if self._state in (
            SupervisorState.IDLE,
            SupervisorState.STOPPING,
            SupervisorState.STOPPED,
        ):
            return
        self._observe()
        self._restart_due()
        self._launch_ready()
        self._refresh_state()

    def shutdown(self) -> None:
        """Stop launched services in reverse start order.

        Elapsed time is measured on the supervisor clock. A stop that
        pushes the clock past ``grace_s`` is logged and the remaining
        services still stop, so the order completes.
        """
        if self._state is SupervisorState.STOPPED:
            return
        self._state = SupervisorState.STOPPING
        begin = self._now()
        self._note("shutdown_start")
        for role in reversed(self._started):
            run = self._runs[role]
            run.restart_at = None
            service = run.service
            if service.state in (LifecycleState.STOPPED, LifecycleState.CREATED):
                self._stopped.append(role)
                self._note("service_stop", role, "already stopped")
                continue
            try:
                service.stop()
            except Exception as exc:
                note_exception(
                    exc,
                    service=role,
                    event="stop_failed",
                    logger=self._logger,
                    metrics=self._metrics,
                )
                self._events.record(self._now(), "stop_failed", role, str(exc))
            self._stopped.append(role)
            self._note("service_stop", role)
            if self._now() - begin > self._grace_s:
                self._note(
                    "grace_exceeded",
                    role,
                    "shutdown exceeded the grace period",
                )
        self._shutdown_elapsed_s = self._now() - begin
        self._shut_down = True
        detail = "within_grace" if self.shutdown_within_grace else "grace_exceeded"
        self._note("shutdown_done", "", detail)
        self._state = SupervisorState.STOPPED
        try:
            self._endpoint.stop()
        except Exception as exc:
            note_exception(
                exc,
                service="supervisor",
                event="metrics_stop_failed",
                logger=self._logger,
                metrics=self._metrics,
            )
            raise

    def clear_do_not_operate(self) -> None:
        """Human clear of the do-not-operate latch.

        Does not restart a failed service, does not arm, and does not
        command motion. Motion stays refused while a critical service
        is failed or still marked do-not-operate on the manifest.
        """
        if not self._dno:
            raise SupervisorError("supervisor is not do-not-operate")
        self._dno = False
        self._holding_safe = False
        self._note("human_clear", "", "human clear")
        if self._state is SupervisorState.DO_NOT_OPERATE:
            self._state = (
                SupervisorState.RUNNING
                if self._initial_up
                else SupervisorState.STARTING
            )

    def _now(self) -> float:
        return float(self._clock())

    def _note(self, kind: str, role: str = "", detail: str = "") -> None:
        """Append the in-memory event and write one structured log record."""
        self._events.record(self._now(), kind, role, detail)
        self._logger.log(
            _EVENT_LEVEL.get(kind, "INFO"),
            event=kind,
            service=role or "supervisor",
            message=detail,
        )

    def observe_loop(self, role: str, timer: LoopTimer) -> None:
        """Attach the loop owned by ``role``.

        Scrapes read :meth:`LoopTimer.stats`. This does not start the loop
        and does not command motion.
        """
        if not isinstance(timer, LoopTimer):
            raise SupervisorError("timer must be a LoopTimer")
        self._run(role)
        self._loops[role] = timer

    def collect_metrics(self) -> None:
        """Copy supervisor, service, and loop state into the registry."""
        base = {"robot_class": self._robot_class.value, "run_id": self._run_id}
        self._metrics.set(
            "ohho_runtime_do_not_operate",
            1.0 if self._dno else 0.0,
            base,
        )
        self._metrics.set(
            "ohho_runtime_allows_motion",
            1.0 if self.allows_motion() else 0.0,
            base,
        )
        self._metrics.set_state(
            "ohho_runtime_supervisor_state",
            self._state.value,
            base,
        )
        for role, run in self._runs.items():
            labels = {"service": role, **base}
            service = run.service
            state = service.state.value
            self._metrics.set_state("ohho_runtime_service_state", state, labels)
            self._metrics.set(
                "ohho_runtime_service_restarts_total",
                float(run.restarts),
                labels,
            )
            self._set_optional(
                "ohho_runtime_heartbeat_age_seconds",
                getattr(service, "heartbeat_age_s", None),
                labels,
            )
            misses = getattr(service, "heartbeat_misses", 0)
            if isinstance(misses, bool):
                misses = 0
            self._metrics.set(
                "ohho_runtime_heartbeat_misses_total",
                float(misses) if isinstance(misses, (int, float)) else 0.0,
                labels,
            )
            self._set_optional(
                "ohho_runtime_start_duration_seconds",
                getattr(service, "start_duration_s", None),
                labels,
            )
            self._set_optional(
                "ohho_runtime_stop_duration_seconds",
                getattr(service, "stop_duration_s", None),
                labels,
            )
            beats = getattr(service, "beats", None)
            if isinstance(beats, int) and not isinstance(beats, bool):
                self._metrics.set(
                    "ohho_runtime_heartbeat_beats_total",
                    float(beats),
                    labels,
                )
            timer = self._loops.get(role)
            if timer is not None:
                publish_loop(self._metrics, labels, loop_report(timer.stats()))

    def metrics_text(self) -> str:
        """Collect current samples and return Prometheus text."""
        self.collect_metrics()
        return self._metrics.render()

    def _set_optional(self, name: str, value: object, labels: dict[str, str]) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        number = float(value)
        if not math.isfinite(number):
            return
        self._metrics.set(name, number, labels)

    def _run(self, role: str) -> _Run:
        try:
            return self._runs[role]
        except KeyError:
            raise SupervisorError(f"unknown service role {role!r}") from None

    def _observe(self) -> None:
        for run in self._runs.values():
            if run.terminal:
                continue
            if run.service.state in (LifecycleState.STARTING, LifecycleState.READY):
                run.service.poll()
            if run.service.state is LifecycleState.READY:
                self._log_ready(run)
            if (
                run.service.state is LifecycleState.FAILED
                and not run.terminal
                and not run.failure_latched
            ):
                self._handle_failure(
                    run.spec.role,
                    run.service.failure_reason,
                    exc=read_failure_exception(run.service),
                )

    def _restart_due(self) -> None:
        now = self._now()
        due = [
            role
            for role, run in self._runs.items()
            if run.restart_at is not None and now >= run.restart_at and not run.terminal
        ]
        for role in due:
            run = self._runs[role]
            run.restart_at = None
            run.failure_latched = False
            self._boot(run, restart=True)

    def _launch_ready(self) -> None:
        guard = len(self._runs) + 1
        for _ in range(guard):
            launched = False
            for role in self._graph.start_order():
                if self._try_launch(role):
                    launched = True
            if not launched:
                return
        raise SupervisorError("dependency launch did not settle")

    def _try_launch(self, role: str) -> bool:
        run = self._runs[role]
        if run.launched or run.terminal:
            return False
        if self._dno and run.spec.commands_motion:
            run.terminal = True
            self._note(
                "blocked",
                role,
                "do-not-operate refuses a motion service",
            )
            return False
        status = self._dependency_status(role)
        if status == "blocked":
            run.terminal = True
            self._note("blocked", role, "dependency failed")
            return False
        if status != "ready":
            return False
        run.launched = True
        self._boot(run, restart=False)
        return True

    def _dependency_status(self, role: str) -> str:
        blocked = False
        waiting = False
        for dep in self._graph.dependencies(role):
            run = self._runs[dep]
            if run.terminal:
                blocked = True
            elif run.service.state is not LifecycleState.READY:
                waiting = True
        if blocked:
            return "blocked"
        if waiting:
            return "waiting"
        return "ready"

    def _boot(self, run: _Run, *, restart: bool) -> None:
        service = run.service
        role = run.spec.role
        if restart:
            try:
                if service.state is not LifecycleState.STOPPED:
                    service.stop()
            except Exception as exc:
                run.failure_latched = True
                run.terminal = True
                note_exception(
                    exc,
                    service=role,
                    event="stop_failed",
                    logger=self._logger,
                    metrics=self._metrics,
                )
                self._events.record(self._now(), "stop_failed", role, str(exc))
                if run.spec.criticality is Criticality.CRITICAL:
                    self._enter_dno(role, "restart stop failed")
                return
            run.restarts += 1
            self._note("restart", role, str(run.restarts))
        else:
            self._started.append(role)
            self._note("service_start", role)
        run.failure_latched = False
        run.restart_at = None
        try:
            service.configure({"role": role, "loop_hz": run.spec.loop_hz})
            service.start()
        except Exception as exc:
            note_exception(
                exc,
                service=role,
                event="service_failed",
                logger=self._logger,
                metrics=self._metrics,
            )
            self._handle_failure(role, str(exc), exc=exc)
            return
        if service.state is LifecycleState.READY:
            self._log_ready(run)
        elif service.state is LifecycleState.FAILED:
            self._handle_failure(role, service.failure_reason)

    def _log_ready(self, run: _Run) -> None:
        if run.ready_open:
            return
        run.ready_open = True
        self._note("service_ready", run.spec.role)

    def _handle_failure(
        self,
        role: str,
        detail: str,
        *,
        exc: BaseException | None = None,
    ) -> None:
        run = self._run(role)
        if run.terminal or run.failure_latched:
            return
        run.failure_latched = True
        run.ready_open = False
        reason = detail or run.service.failure_reason or "failed"
        self._note("service_failed", role, reason)
        outcome = self._apply_restart_policy(run, role)
        self._write_crash_bundle(run, reason, exc=exc, outcome=outcome)

    def _apply_restart_policy(self, run: _Run, role: str) -> str:
        """Apply the manifest restart policy. Returns the outcome token."""
        policy = run.spec.restart
        backoff = run.backoff
        if policy.kind is not RestartKind.ON_FAILURE or backoff is None:
            run.terminal = True
            self._note("restart_refused", role, "restart policy is never")
            return "restart_refused"
        if policy.only_while_disarmed and not self._disarmed:
            run.terminal = True
            self._note(
                "restart_refused",
                role,
                "restart requires the robot to be disarmed",
            )
            if run.spec.criticality is Criticality.CRITICAL:
                self._enter_dno(
                    role,
                    "critical service cannot restart while the robot is not disarmed",
                )
            return "restart_refused"
        now = self._now()
        prior = [stamp for stamp in run.failure_times if stamp >= now - policy.window_s]
        if not prior:
            backoff.reset()
        prior.append(now)
        run.failure_times = prior
        if len(prior) >= policy.max_restarts:
            run.terminal = True
            run.restart_at = None
            self._note(
                "restart_limit",
                role,
                f"{len(prior)} failures in {policy.window_s}s",
            )
            if run.spec.criticality is Criticality.CRITICAL:
                self._enter_dno(role, "critical service exceeded its restart limit")
            return "restart_limit"
        delay = backoff.next_delay()
        run.restart_at = now + delay
        self._note("restart_scheduled", role, f"{delay}")
        return "restart_scheduled"

    def _write_crash_bundle(
        self,
        run: _Run,
        detail: str,
        *,
        exc: BaseException | None,
        outcome: str,
    ) -> None:
        """Write one bundle when this failure is a crash. Failures here are logged."""
        try:
            if exc is None:
                exc = read_failure_exception(run.service)
            exit_code = read_exit_code(run.service)
            reason = classify_crash(
                detail=detail,
                exc=exc,
                exit_code=exit_code,
                outcome=outcome,
            )
            if reason is None:
                return
            role = run.spec.role
            service_logs = _history_for(self._log_history, role, DEFAULT_LOG_LIMIT)
            supervisor_logs = _history_for(
                self._log_history,
                "supervisor",
                DEFAULT_LOG_LIMIT,
            )
            timer = self._loops.get(role)
            loop = None if timer is None else loop_report(timer.stats())
            try:
                metrics_text = self.metrics_text()
            except Exception as metrics_exc:
                note_exception(
                    metrics_exc,
                    service="supervisor",
                    event="crash_bundle_metrics_failed",
                    logger=self._logger,
                    metrics=self._metrics,
                )
                metrics_text = ""
            states = []
            for item in getattr(run.service, "history", ()):
                states.append(str(getattr(item, "value", item)))
            current = getattr(run.service, "state", "")
            write_crash_bundle(
                service=role,
                run_id=self._run_id,
                reason=reason,
                state_dir=self._crash_dir,
                manifest=self._manifest,
                service_logs=service_logs,
                supervisor_logs=supervisor_logs,
                history=states,
                state=str(getattr(current, "value", current)),
                restarts=run.restarts,
                restart_exhausted=run.terminal,
                failure_reason=detail,
                metrics_text=metrics_text,
                exit_code=exit_code,
                exc=exc,
                detail=detail,
                loop=loop,
                robot_class=self._robot_class.value,
                log_limit=DEFAULT_LOG_LIMIT,
                logger=self._logger,
                metrics=self._metrics,
            )
        except Exception as bundle_exc:
            note_exception(
                bundle_exc,
                service="supervisor",
                event="crash_bundle_failed",
                logger=self._logger,
                metrics=self._metrics,
            )

    def _enter_dno(self, role: str, reason: str) -> None:
        if self._dno:
            return
        self._dno = True
        self._holding_safe = True
        self._state = SupervisorState.DO_NOT_OPERATE
        action = class_safe_action(self._robot_class)
        notice = DoNotOperateNotice(
            robot_class=self._robot_class,
            safe_action=action,
            role=role,
            reason=reason,
            allows_motion=False,
        )
        self._note("do_not_operate", role, reason)
        self._note("safe_hold", role, action.value)
        self._on_do_not_operate(notice)

    def _refresh_state(self) -> None:
        if self._state in (SupervisorState.STOPPING, SupervisorState.STOPPED):
            return
        if self._dno:
            self._state = SupervisorState.DO_NOT_OPERATE
            return
        if not self._initial_up:
            if self._runs and all(
                run.service.state is LifecycleState.READY for run in self._runs.values()
            ):
                self._initial_up = True
                self._state = SupervisorState.RUNNING
            else:
                self._state = SupervisorState.STARTING
            return
        self._state = SupervisorState.RUNNING


def _history_for(history: object, service: str, limit: int) -> list[dict[str, object]]:
    method = getattr(history, "for_service", None)
    if not callable(method):
        return []
    found = method(service, limit)
    rows: list[dict[str, object]] = []
    for item in found:
        if isinstance(item, Mapping):
            rows.append({str(key): value for key, value in item.items()})
    return rows


def _noop_hook(notice: DoNotOperateNotice) -> None:
    del notice


__all__ = [
    "ClassSafeAction",
    "DependencyGraph",
    "DevService",
    "DevServiceFactory",
    "DoNotOperateHook",
    "DoNotOperateNotice",
    "EventLog",
    "ServiceBuilder",
    "Supervisor",
    "SupervisorError",
    "SupervisorEvent",
    "SupervisorState",
    "build_dependency_graph",
    "class_safe_action",
]
