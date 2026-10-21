"""Runtime manifest: the services that run on a robot.

A :class:`RuntimeManifest` is a set of services. Each service has a role,
an entry (Python callable, console script, or ROS 2 launch/node),
dependencies, criticality, readiness and liveness timeouts, a restart
policy, a periodic loop rate, named QoS profiles, and an optional
scheduling hint.

Loading a manifest rejects an unknown QoS profile name and a dependency
cycle. The named QoS set is small on purpose.

:func:`class_default_manifest` is keyed by ``robot_class``. Every class
includes transport, the safety watchdog, telemetry, and the teleop
gateway. Aerial adds an autopilot heartbeat. Humanoid adds a mode monitor.

:func:`resolve_manifest` is what a caller uses for one spec. A plugin may
implement ``services(spec)``. The method is optional and is read with
``getattr``, so a plugin that does not define it still loads. ``None``
selects the class default. A returned manifest replaces the class default
and must still contain transport, the safety watchdog, and telemetry.
Motion services depend on the watchdog so it starts first.

This module does not start those services, connect to a robot, or arm.
An aerial spec that is experimental, or not flight-ready, is marked
do-not-operate on services that command motion. That flag does not arm
and does not change ``SafetyGate``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass, parse_robot_class

from ohho_runtime import entries
from ohho_runtime.qos import QOS_PROFILES

ROLE_TRANSPORT = "transport"
ROLE_SAFETY_WATCHDOG = "safety_watchdog"
ROLE_TELEMETRY = "telemetry"
ROLE_TELEOP_GATEWAY = "teleop_gateway"
ROLE_POLICY = "policy"
ROLE_PERCEPTION = "perception"
ROLE_RECORDER = "recorder"
ROLE_AUTOPILOT_HEARTBEAT = "autopilot_heartbeat"
ROLE_MODE_MONITOR = "mode_monitor"

# Present on every class default. A plugin manifest may not omit these.
REQUIRED_ROLES: tuple[str, ...] = (
    ROLE_TRANSPORT,
    ROLE_SAFETY_WATCHDOG,
    ROLE_TELEMETRY,
)

# Services that can command motion. The watchdog has to be up first.
_MOTION_ROLES = frozenset({ROLE_TRANSPORT, ROLE_TELEOP_GATEWAY, ROLE_POLICY})

_REQUIRED_CRITICAL = frozenset(REQUIRED_ROLES)
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]*\Z")

_WATCHDOG_HZ = 20.0
_TRANSPORT_HZ = 50.0
_TELEMETRY_HZ = 20.0
_TELEOP_HZ = 20.0
_HEARTBEAT_HZ = 10.0
_MODE_HZ = 10.0


class ManifestError(ValueError):
    """A runtime manifest failed to load or resolve."""


class Criticality(str, Enum):
    """How a service failure is treated. ``degradable`` is not a latch."""

    CRITICAL = "critical"
    DEGRADABLE = "degradable"
    OPTIONAL = "optional"


class RestartKind(str, Enum):
    """Whether a failed service may be restarted."""

    NEVER = "never"
    ON_FAILURE = "on-failure"


def _positive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number > 0


def _enum_value(kind: type[Enum], value: object, label: str) -> Enum:
    if isinstance(value, kind):
        return value
    try:
        return kind(value)
    except (ValueError, TypeError):
        allowed = ", ".join(repr(item.value) for item in kind)
        raise ManifestError(f"unknown {label} {value!r}; expected {allowed}") from None


def _role_name(value: object) -> str:
    if isinstance(value, str) and _ROLE_RE.fullmatch(value):
        return value
    raise ManifestError(f"service role {value!r} must be a lowercase identifier")


def _string_tuple(value: object, what: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{what} must be a sequence of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ManifestError(f"{what} must be a sequence of strings")
        items.append(item)
    return tuple(items)


def _token(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ch.isspace() for ch in value)
    ):
        raise ManifestError(f"{what} must be a non-empty string without whitespace")
    return value


@dataclass(frozen=True)
class RestartPolicy:
    """``never``, or ``on-failure`` with a cap, a window, and backoff.

    ``only_while_disarmed`` records that a later supervisor may restart the
    service only while the robot is disarmed. This object does not restart
    anything itself.
    """

    kind: RestartKind = RestartKind.NEVER
    max_restarts: int = 0
    window_s: float = 0.0
    backoff_s: float = 0.0
    only_while_disarmed: bool = False

    def __post_init__(self) -> None:
        kind = _enum_value(RestartKind, self.kind, "restart policy")
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.only_while_disarmed, bool):
            raise ManifestError("only_while_disarmed must be a bool")
        if kind is RestartKind.NEVER:
            if (
                self.max_restarts
                or self.window_s
                or self.backoff_s
                or self.only_while_disarmed
            ):
                raise ManifestError(
                    "restart policy 'never' does not take max_restarts, "
                    "window_s, backoff, or a disarm condition"
                )
            return
        restarts = self.max_restarts
        if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
            raise ManifestError("on-failure restart requires max_restarts >= 1")
        if not _positive(self.window_s):
            raise ManifestError("on-failure restart requires window_s > 0")
        if not _positive(self.backoff_s):
            raise ManifestError("on-failure restart requires backoff_s > 0")
        object.__setattr__(self, "window_s", float(self.window_s))
        object.__setattr__(self, "backoff_s", float(self.backoff_s))


@dataclass(frozen=True)
class SchedulingHint:
    """Optional period, priority, and CPU affinity.

    Configuration only. Not a hard real-time promise on a stock kernel.
    """

    period_s: float | None = None
    priority: int | None = None
    cpu_affinity: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.period_s is not None:
            if not _positive(self.period_s):
                raise ManifestError("scheduling period_s must be a number > 0")
            object.__setattr__(self, "period_s", float(self.period_s))
        if self.priority is not None and (
            isinstance(self.priority, bool) or not isinstance(self.priority, int)
        ):
            raise ManifestError("scheduling priority must be an int")
        affinity = self.cpu_affinity
        if isinstance(affinity, list):
            affinity = tuple(affinity)
        if isinstance(affinity, (str, bytes)) or not isinstance(affinity, tuple):
            raise ManifestError("cpu_affinity must be a tuple of cpu indexes")
        cleaned: list[int] = []
        for cpu in affinity:
            if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0:
                raise ManifestError("cpu_affinity entries must be ints >= 0")
            cleaned.append(cpu)
        object.__setattr__(self, "cpu_affinity", tuple(cleaned))


@dataclass(frozen=True)
class PythonCallable:
    """In-process entry. ``target`` is not called while the manifest loads."""

    target: Any

    def __post_init__(self) -> None:
        if not callable(self.target):
            raise ManifestError("python entry target must be callable")


@dataclass(frozen=True)
class ConsoleScript:
    """A console script name, for example ``ohho-runtime-service``."""

    name: str

    def __post_init__(self) -> None:
        _token(self.name, "console script")


@dataclass(frozen=True)
class Ros2Launch:
    """A ROS 2 launch file. Loading the manifest does not import ROS 2."""

    package: str
    launch_file: str

    def __post_init__(self) -> None:
        _token(self.package, "ROS 2 package")
        _token(self.launch_file, "ROS 2 launch file")


@dataclass(frozen=True)
class Ros2Node:
    """A ROS 2 node. Loading the manifest does not import ROS 2."""

    package: str
    node: str

    def __post_init__(self) -> None:
        _token(self.package, "ROS 2 package")
        _token(self.node, "ROS 2 node")


ServiceEntry = PythonCallable | ConsoleScript | Ros2Launch | Ros2Node


def _check_entry(entry: object) -> ServiceEntry:
    if isinstance(entry, (PythonCallable, ConsoleScript, Ros2Launch, Ros2Node)):
        return entry
    raise ManifestError(
        "entry must be a PythonCallable, ConsoleScript, Ros2Launch, or Ros2Node"
    )


@dataclass(frozen=True)
class ServiceSpec:
    """One supervised service in a :class:`RuntimeManifest`."""

    role: str
    entry: ServiceEntry
    depends_on: tuple[str, ...] = ()
    criticality: Criticality = Criticality.CRITICAL
    readiness_timeout_s: float = 5.0
    liveness_timeout_s: float = 1.0
    restart: RestartPolicy = RestartPolicy()
    loop_hz: float = 1.0
    qos: tuple[str, ...] = ()
    scheduling: SchedulingHint | None = None
    commands_motion: bool = False
    do_not_operate: bool = False

    def __post_init__(self) -> None:
        role = _role_name(self.role)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "entry", _check_entry(self.entry))
        criticality = _enum_value(Criticality, self.criticality, "criticality")
        object.__setattr__(self, "criticality", criticality)
        if role in _REQUIRED_CRITICAL and criticality is not Criticality.CRITICAL:
            raise ManifestError(f"{role} criticality must be 'critical'")
        for name, value in (
            ("readiness_timeout_s", self.readiness_timeout_s),
            ("liveness_timeout_s", self.liveness_timeout_s),
        ):
            if not _positive(value):
                raise ManifestError(f"{role} {name} must be a number > 0")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.restart, RestartPolicy):
            raise ManifestError(f"{role} restart must be a RestartPolicy")
        if role == ROLE_SAFETY_WATCHDOG and self.restart.kind is not RestartKind.NEVER:
            raise ManifestError(
                "safety_watchdog restart policy must be 'never' "
                "(it is not restarted while motion is permitted)"
            )
        if (
            role == ROLE_TRANSPORT
            and self.restart.kind is RestartKind.ON_FAILURE
            and not self.restart.only_while_disarmed
        ):
            raise ManifestError(
                "transport on-failure restart must set only_while_disarmed"
            )
        if not _positive(self.loop_hz):
            raise ManifestError(f"service {role!r} loop_hz must be a number > 0")
        object.__setattr__(self, "loop_hz", float(self.loop_hz))
        deps = _string_tuple(self.depends_on, f"service {role!r} depends_on")
        for dep in deps:
            _role_name(dep)
        if len(deps) != len(set(deps)):
            raise ManifestError(f"service {role!r} lists a dependency more than once")
        if role == ROLE_SAFETY_WATCHDOG and deps:
            raise ManifestError("safety_watchdog cannot depend on another service")
        object.__setattr__(self, "depends_on", deps)
        profiles = _string_tuple(self.qos, f"service {role!r} qos")
        object.__setattr__(self, "qos", profiles)
        if self.scheduling is not None and not isinstance(
            self.scheduling, SchedulingHint
        ):
            raise ManifestError(f"{role} scheduling must be a SchedulingHint or None")
        if not isinstance(self.commands_motion, bool) or not isinstance(
            self.do_not_operate, bool
        ):
            raise ManifestError(
                f"{role} commands_motion and do_not_operate must be bools"
            )


def make_service(
    role: str,
    entry: ServiceEntry,
    *,
    depends_on: Sequence[str] = (),
    criticality: Criticality | str = Criticality.CRITICAL,
    readiness_timeout_s: float = 5.0,
    liveness_timeout_s: float = 1.0,
    restart: RestartPolicy | None = None,
    loop_hz: float,
    qos: Sequence[str] = (),
    scheduling: SchedulingHint | None = None,
    commands_motion: bool = False,
    do_not_operate: bool = False,
) -> ServiceSpec:
    """Build one :class:`ServiceSpec`. ``loop_hz`` is required."""
    return ServiceSpec(
        role=role,
        entry=entry,
        depends_on=tuple(depends_on),
        criticality=criticality,  # type: ignore[arg-type]
        readiness_timeout_s=readiness_timeout_s,
        liveness_timeout_s=liveness_timeout_s,
        restart=RestartPolicy() if restart is None else restart,
        loop_hz=loop_hz,
        qos=tuple(qos),
        scheduling=scheduling,
        commands_motion=commands_motion,
        do_not_operate=do_not_operate,
    )


@dataclass(frozen=True)
class RuntimeManifest:
    """Validated service set. Constructing it loads and checks the set."""

    services: tuple[ServiceSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", _validate_and_order(self.services))

    def roles(self) -> tuple[str, ...]:
        """Service roles in start order. The watchdog is first when present."""
        return tuple(service.role for service in self.services)

    def service(self, role: str) -> ServiceSpec:
        """Return the service for ``role``."""
        for service in self.services:
            if service.role == role:
                return service
        raise ManifestError(f"runtime manifest has no service {role!r}")


def load_manifest(services: Sequence[ServiceSpec]) -> RuntimeManifest:
    """Validate ``services`` and return a manifest.

    Unknown QoS profile names and dependency cycles raise
    :class:`ManifestError`.
    """
    return RuntimeManifest(services=tuple(services))


def with_services(manifest: RuntimeManifest, *services: ServiceSpec) -> RuntimeManifest:
    """Copy ``manifest`` with extra services appended.

    A plugin hook can start from the class default and add a service
    without dropping transport, the safety watchdog, or telemetry.
    """
    if not isinstance(manifest, RuntimeManifest):
        raise TypeError("manifest must be a RuntimeManifest")
    return load_manifest((*manifest.services, *services))


def _validate_and_order(services: object) -> tuple[ServiceSpec, ...]:
    if isinstance(services, (str, bytes)) or not isinstance(services, Sequence):
        raise ManifestError("a runtime manifest needs a sequence of services")
    specs: list[ServiceSpec] = []
    for service in services:
        if not isinstance(service, ServiceSpec):
            raise ManifestError(
                "a runtime manifest needs ServiceSpec objects, "
                f"not {type(service).__name__}"
            )
        specs.append(service)
    by_role: dict[str, ServiceSpec] = {}
    duplicates: list[str] = []
    for service in specs:
        if service.role in by_role:
            duplicates.append(service.role)
        by_role[service.role] = service
    problems: list[str] = []
    if duplicates:
        names = ", ".join(repr(role) for role in duplicates)
        problems.append(f"duplicate service role {names}")
    for service in specs:
        seen: set[str] = set()
        for name in service.qos:
            if name in seen:
                problems.append(
                    f"service {service.role!r} lists QoS profile {name!r} more than once"
                )
            seen.add(name)
            if name not in QOS_PROFILES:
                problems.append(
                    f"unknown QoS profile {name!r} on service {service.role!r}"
                )
        for dep in service.depends_on:
            if dep not in by_role:
                problems.append(
                    f"service {service.role!r} depends on unknown role {dep!r}"
                )
    if problems:
        raise ManifestError("runtime manifest rejected: " + "; ".join(problems))

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []
    stack: list[str] = []

    def visit(role: str) -> None:
        if role in visited:
            return
        if role in visiting:
            start = stack.index(role)
            cycle = " -> ".join((*stack[start:], role))
            raise ManifestError(f"dependency cycle in runtime manifest: {cycle}")
        visiting.add(role)
        stack.append(role)
        for dep in by_role[role].depends_on:
            visit(dep)
        stack.pop()
        visiting.remove(role)
        visited.add(role)
        order.append(role)

    for role in by_role:
        visit(role)
    return tuple(by_role[role] for role in order)


def _require_core(manifest: RuntimeManifest, *, source: str) -> None:
    present = {service.role for service in manifest.services}
    missing = [role for role in REQUIRED_ROLES if role not in present]
    if missing:
        names = ", ".join(repr(role) for role in missing)
        raise ManifestError(
            f"{source} removed required {names}; "
            "transport, safety_watchdog, and telemetry cannot be removed"
        )
    for service in manifest.services:
        if (
            service.role in _MOTION_ROLES
            and ROLE_SAFETY_WATCHDOG not in service.depends_on
        ):
            raise ManifestError(
                f"{service.role} must depend on safety_watchdog so the watchdog starts first"
            )


def _apply_aerial_gate(manifest: RuntimeManifest, spec: RobotSpec) -> RuntimeManifest:
    """Mark motion services do-not-operate when an aerial spec is not flyable.

    ``flight_ready`` is false for every non-aerial spec, so the class check
    comes first. A true flag is never cleared: a plugin may be stricter.
    """
    if spec.robot_class is not RobotClass.AERIAL:
        return manifest
    if not spec.experimental and spec.flight_ready:
        return manifest
    services: list[ServiceSpec] = []
    changed = False
    for service in manifest.services:
        blocks = service.role in _MOTION_ROLES or service.commands_motion
        if blocks and not service.do_not_operate:
            services.append(replace(service, do_not_operate=True))
            changed = True
        else:
            services.append(service)
    if not changed:
        return manifest
    return RuntimeManifest(services=tuple(services))


def _call(target: Any) -> PythonCallable:
    return PythonCallable(target)


def _never() -> RestartPolicy:
    return RestartPolicy(RestartKind.NEVER)


def _restart_disarmed() -> RestartPolicy:
    return RestartPolicy(
        RestartKind.ON_FAILURE,
        max_restarts=3,
        window_s=60.0,
        backoff_s=0.5,
        only_while_disarmed=True,
    )


def _restart_backoff(max_restarts: int = 5) -> RestartPolicy:
    return RestartPolicy(
        RestartKind.ON_FAILURE,
        max_restarts=max_restarts,
        window_s=60.0,
        backoff_s=0.5,
    )


def _ground_services() -> tuple[ServiceSpec, ...]:
    watchdog = make_service(
        ROLE_SAFETY_WATCHDOG,
        _call(entries.safety_watchdog),
        readiness_timeout_s=2.0,
        liveness_timeout_s=0.5,
        restart=_never(),
        loop_hz=_WATCHDOG_HZ,
        qos=("state",),
        scheduling=SchedulingHint(period_s=1.0 / _WATCHDOG_HZ),
    )
    transport = make_service(
        ROLE_TRANSPORT,
        _call(entries.transport),
        depends_on=(ROLE_SAFETY_WATCHDOG,),
        restart=_restart_disarmed(),
        loop_hz=_TRANSPORT_HZ,
        qos=("command", "state"),
        commands_motion=True,
    )
    telemetry = make_service(
        ROLE_TELEMETRY,
        _call(entries.telemetry),
        depends_on=(ROLE_TRANSPORT,),
        restart=_never(),
        loop_hz=_TELEMETRY_HZ,
        qos=("telemetry", "state"),
    )
    teleop = make_service(
        ROLE_TELEOP_GATEWAY,
        _call(entries.teleop_gateway),
        depends_on=(ROLE_SAFETY_WATCHDOG, ROLE_TRANSPORT, ROLE_TELEMETRY),
        restart=_restart_backoff(),
        loop_hz=_TELEOP_HZ,
        qos=("command",),
        commands_motion=True,
    )
    return (watchdog, transport, telemetry, teleop)


def _class_services(robot_class: RobotClass) -> tuple[ServiceSpec, ...]:
    services = list(_ground_services())
    if robot_class is RobotClass.AERIAL:
        services.append(
            make_service(
                ROLE_AUTOPILOT_HEARTBEAT,
                _call(entries.autopilot_heartbeat),
                depends_on=(ROLE_SAFETY_WATCHDOG, ROLE_TELEMETRY),
                readiness_timeout_s=2.0,
                liveness_timeout_s=0.5,
                restart=_restart_backoff(3),
                loop_hz=_HEARTBEAT_HZ,
                qos=("state",),
            )
        )
    elif robot_class is RobotClass.HUMANOID:
        services.append(
            make_service(
                ROLE_MODE_MONITOR,
                _call(entries.mode_monitor),
                depends_on=(ROLE_TELEMETRY,),
                restart=_restart_backoff(3),
                loop_hz=_MODE_HZ,
                qos=("state",),
            )
        )
    return tuple(services)


def _build_class_defaults() -> Mapping[RobotClass, RuntimeManifest]:
    built: dict[RobotClass, RuntimeManifest] = {}
    for robot_class in RobotClass:
        manifest = load_manifest(_class_services(robot_class))
        _require_core(manifest, source=f"class default {robot_class.value}")
        built[robot_class] = manifest
    return MappingProxyType(built)


CLASS_DEFAULTS: Mapping[RobotClass, RuntimeManifest] = _build_class_defaults()


def class_default_manifest(robot_class: RobotClass | str) -> RuntimeManifest:
    """Return the default manifest for one robot class.

    Every class includes transport, the safety watchdog, telemetry, and
    the teleop gateway. Aerial adds ``autopilot_heartbeat``. Humanoid adds
    ``mode_monitor``. This does not read a spec and does not apply the
    aerial do-not-operate gate; use :func:`resolve_manifest` for a spec.
    """
    if isinstance(robot_class, RobotClass):
        kind = robot_class
    else:
        kind = parse_robot_class(robot_class)
    try:
        return CLASS_DEFAULTS[kind]
    except KeyError:
        raise ManifestError(f"no class default manifest for {kind!r}") from None


@runtime_checkable
class ServicesProvider(Protocol):
    """Optional ``services(spec)`` hook.

    ``RobotPlugin`` does not declare this method. Callers discover it with
    ``getattr``, so a plugin that does not define it still loads. ``None``
    keeps the class default. A returned manifest is the full service set
    for that spec: it may add services, and it may not omit transport, the
    safety watchdog, or telemetry.
    """

    def services(self, spec: RobotSpec) -> RuntimeManifest | None:
        """Return the manifest for ``spec``, or ``None`` for the class default."""


_PLUGIN_CACHE: tuple[object, ...] | None = None


def _ordered_entries() -> list[Any]:
    import importlib.metadata

    from ohho.plugins import BUILTIN_ENTRY_VALUE, ENTRY_GROUP

    found = importlib.metadata.entry_points(group=ENTRY_GROUP)
    if isinstance(found, Mapping):
        found = found.get(ENTRY_GROUP, ())
    entries_found = list(found)
    if not any(
        getattr(entry, "value", None) == BUILTIN_ENTRY_VALUE for entry in entries_found
    ):
        entries_found.append(
            importlib.metadata.EntryPoint(
                name="builtin",
                value=BUILTIN_ENTRY_VALUE,
                group=ENTRY_GROUP,
            )
        )
    builtins: list[Any] = []
    rest: list[Any] = []
    seen_builtin = False
    for entry in entries_found:
        if getattr(entry, "value", None) == BUILTIN_ENTRY_VALUE:
            if seen_builtin:
                continue
            seen_builtin = True
            builtins.append(entry)
        else:
            rest.append(entry)
    rest.sort(key=lambda entry: str(getattr(entry, "name", "")))
    return builtins + rest


def _materialize(loaded: object) -> object:
    """Accept a plugin instance, a plugin class, or a zero-arg factory."""
    if isinstance(loaded, type):
        return loaded()
    if callable(loaded) and not isinstance(getattr(loaded, "name", None), str):
        return loaded()
    return loaded


def _load_plugin_objects() -> tuple[object, ...]:
    from ohho.plugins import BUILTIN_ENTRY_VALUE

    loaded_objects: list[object] = []
    for entry in _ordered_entries():
        is_builtin = getattr(entry, "value", None) == BUILTIN_ENTRY_VALUE
        try:
            loaded_objects.append(_materialize(entry.load()))
        except Exception:
            if is_builtin:
                raise
            continue
    return tuple(loaded_objects)


def _plugin_objects() -> tuple[object, ...]:
    global _PLUGIN_CACHE
    if _PLUGIN_CACHE is None:
        _PLUGIN_CACHE = _load_plugin_objects()
    return _PLUGIN_CACHE


def plugin_for_spec(spec: RobotSpec) -> object | None:
    """Return the plugin object that registered ``spec``, if discovery finds one.

    Objects come from the public ``ohho.robots`` entry-point group. The
    ``services`` attribute is not required.
    """
    if not isinstance(spec, RobotSpec):
        raise TypeError(f"spec must be a RobotSpec, not {type(spec).__name__}")
    from ohho.plugins import loaded_plugins

    owner: str | None = None
    for info in loaded_plugins():
        if info.error is not None:
            continue
        if spec.id in info.spec_ids:
            owner = info.name
            break
    if owner is None:
        return None
    for plugin in _plugin_objects():
        if getattr(plugin, "name", None) == owner:
            return plugin
    return None


def _services_from(plugin: object, spec: RobotSpec) -> RuntimeManifest | None:
    hook = getattr(plugin, "services", None)
    if hook is None:
        return None
    if not callable(hook):
        raise ManifestError("services hook is not callable")
    result = hook(spec)
    if result is None:
        return None
    if not isinstance(result, RuntimeManifest):
        raise ManifestError(
            "services() must return a RuntimeManifest or None, "
            f"not {type(result).__name__}"
        )
    return result


def resolve_manifest(
    spec: RobotSpec,
    plugin: object | None = None,
) -> RuntimeManifest:
    """Resolve the manifest for ``spec``.

    When ``plugin`` is omitted, the plugin that registered ``spec`` is
    loaded through the public ``ohho.robots`` entry-point group. If that
    object has a callable ``services`` and it returns a manifest, that
    manifest is the resolution. ``None``, or a missing method, keeps the
    class default. The hook is not called at import time.
    """
    if not isinstance(spec, RobotSpec):
        raise TypeError(f"spec must be a RobotSpec, not {type(spec).__name__}")
    kind = spec.robot_class
    if not isinstance(kind, RobotClass):
        raise ManifestError("spec.robot_class is not set")
    chosen = plugin if plugin is not None else plugin_for_spec(spec)
    provided = None if chosen is None else _services_from(chosen, spec)
    if provided is None:
        manifest = class_default_manifest(kind)
        source = "class default"
    else:
        manifest = provided
        source = "plugin manifest"
    _require_core(manifest, source=source)
    return _apply_aerial_gate(manifest, spec)


__all__ = [
    "CLASS_DEFAULTS",
    "REQUIRED_ROLES",
    "ROLE_AUTOPILOT_HEARTBEAT",
    "ROLE_MODE_MONITOR",
    "ROLE_PERCEPTION",
    "ROLE_POLICY",
    "ROLE_RECORDER",
    "ROLE_SAFETY_WATCHDOG",
    "ROLE_TELEMETRY",
    "ROLE_TELEOP_GATEWAY",
    "ROLE_TRANSPORT",
    "ConsoleScript",
    "Criticality",
    "ManifestError",
    "PythonCallable",
    "RestartKind",
    "RestartPolicy",
    "Ros2Launch",
    "Ros2Node",
    "RuntimeManifest",
    "SchedulingHint",
    "ServiceEntry",
    "ServiceSpec",
    "ServicesProvider",
    "class_default_manifest",
    "load_manifest",
    "make_service",
    "plugin_for_spec",
    "resolve_manifest",
    "with_services",
]
