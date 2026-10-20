"""Robot plugin protocol and ``ohho.robots`` entry-point discovery.

A plugin contributes specs, transport schemes and sim backends. ``safety_policy``
returns ``None`` to keep the class default. ``SafetyGate`` calls it when the
robot connects (OHH-313) and refuses a policy that loosens class caps.
Discovery does not call it, so listing plugins cannot command a robot.
``descriptor`` returns ``None`` until C-04 / OHH-214. ``doctor`` is the
read-only hook R-06 will dispatch. Discovery does not call those two either.

Built-ins use this path. ``pyproject.toml`` registers
``ohho.robots.builtin:plugin``. A source checkout that has not been reinstalled
has no entry point; discovery then loads that same module directly.

One plugin failing to import is recorded and skipped. A duplicate robot id or
transport scheme is a load error naming both plugins; the earlier plugin keeps
its robots. The built-in plugin is applied first so a third-party collision
does not drop sim, OmniBot or Go2.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .registry import RobotSpec
from .transport import Transport

ENTRY_GROUP = "ohho.robots"
BUILTIN_ENTRY_VALUE = "ohho.robots.builtin:plugin"

# ``factory(spec, address, runtime)`` — the runtime is whatever ``get_runtime``
# returned. Factories must not import a vendor SDK until they themselves run.
TransportFactory = Callable[[RobotSpec, str, Any], Transport]
SimFactory = Callable[[RobotSpec, str, Any], Any]

_PLUGIN_METHODS = (
    "specs",
    "transports",
    "sim_backends",
    "safety_policy",
    "descriptor",
    "doctor",
)


class RobotPlugin(Protocol):
    """What an ``ohho.robots`` entry point returns."""

    name: str

    def specs(self) -> list[RobotSpec]:
        """Robot specs this plugin registers."""

    def transports(self) -> Mapping[str, TransportFactory]:
        """URI scheme to ``factory(spec, address, runtime)``."""

    def sim_backends(self) -> Mapping[str, SimFactory]:
        """Sim backend name to ``factory(spec, address, runtime)``."""

    def safety_policy(self, spec: RobotSpec) -> object | None:
        """A stricter class policy, or ``None`` for the class default.

        ``SafetyGate`` loads this when the robot connects. Discovery does not
        call it. A policy that loosens class caps is refused at that load.
        """

    def descriptor(self, spec: RobotSpec) -> Mapping[str, Any] | None:
        """RobotDescriptor v1, or ``None`` until C-04 / OHH-214."""

    def doctor(
        self,
        spec: RobotSpec,
        uri: str | None = None,
        read_only: bool = True,
    ) -> DoctorReport | None:
        """Read-only doctor hook. ``None`` means the plugin has nothing to add."""


@dataclass(frozen=True)
class DoctorReport:
    """Result of a plugin doctor hook. It must not command the robot.

    R-06 (``ohho doctor --robot``) will print this. Discovery never calls
    ``doctor``.
    """

    ok: bool
    summary: str = ""
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginInfo:
    """One discovered plugin, including a plugin that failed to load."""

    name: str
    entry_name: str
    spec_ids: tuple[str, ...]
    specs: tuple[RobotSpec, ...]
    schemes: tuple[str, ...]
    sim_backends: tuple[str, ...]
    error: str | None = None
    fallback: bool = False
    distribution: str = ""
    # The instance, so a later policy load can call ``safety_policy`` without
    # discovery calling it. Not part of equality or the list report.
    plugin: Any = field(default=None, compare=False, hash=False)

    @property
    def ok(self) -> bool:
        return self.error is None


_CACHE: tuple[PluginInfo, ...] | None = None
_in_load = False


def _read_entry_points() -> list[Any]:
    """Return installed entry points in :data:`ENTRY_GROUP`.

    ``entry_points(group=...)`` is the selector C-02 specifies. It works on
    Python 3.10 through 3.13 (3.10 returns the selected group when parameters
    are passed).
    """
    found = importlib.metadata.entry_points(group=ENTRY_GROUP)
    if isinstance(found, Mapping):
        found = found.get(ENTRY_GROUP, ())
    return list(found)


def _points_at_builtin(entry: Any) -> bool:
    return getattr(entry, "value", None) == BUILTIN_ENTRY_VALUE


def _fallback_entry() -> Any:
    return importlib.metadata.EntryPoint(
        name="builtin",
        value=BUILTIN_ENTRY_VALUE,
        group=ENTRY_GROUP,
    )


def _distribution_name(entry: Any) -> str:
    dist = getattr(entry, "dist", None)
    if dist is None:
        return ""
    name = getattr(dist, "name", None)
    if isinstance(name, str) and name:
        return name
    metadata = getattr(dist, "metadata", None)
    if metadata is not None:
        got = metadata.get("Name") if hasattr(metadata, "get") else None
        if isinstance(got, str):
            return got
    return ""


def _discover_entries() -> list[tuple[Any, bool]]:
    """Entry points in load order, as ``(entry, fallback)``.

    The built-in plugin is first. A missing built-in entry point (source
    checkout, or an install from before this group existed) appends one
    fallback that imports the same module.
    """
    found = _read_entry_points()
    pairs = [(entry, False) for entry in found]
    if not any(_points_at_builtin(entry) for entry in found):
        pairs.append((_fallback_entry(), True))

    builtins: list[tuple[Any, bool]] = []
    rest: list[tuple[Any, bool]] = []
    seen_builtin = False
    for entry, fallback in pairs:
        if _points_at_builtin(entry):
            if seen_builtin:
                continue
            seen_builtin = True
            builtins.append((entry, fallback))
        else:
            rest.append((entry, fallback))
    rest.sort(key=lambda item: str(getattr(item[0], "name", "")))
    return builtins + rest


def _missing_methods(obj: object) -> list[str]:
    return [
        method for method in _PLUGIN_METHODS if not callable(getattr(obj, method, None))
    ]


def _is_instance_plugin(obj: object) -> bool:
    if isinstance(obj, type):
        return False
    name = getattr(obj, "name", None)
    return isinstance(name, str) and bool(name) and not _missing_methods(obj)


def _materialize(loaded: object) -> object:
    """Accept a plugin instance, a plugin class, or a zero-arg factory."""
    if isinstance(loaded, type):
        loaded = loaded()
    elif callable(loaded) and not _is_instance_plugin(loaded):
        loaded = loaded()
    if isinstance(loaded, type):
        raise TypeError(
            "ohho.robots entry point returned a class; call it so discovery "
            "receives a RobotPlugin instance"
        )
    missing = _missing_methods(loaded)
    name = getattr(loaded, "name", None)
    if not isinstance(name, str) or not name:
        missing = ["name", *missing]
    if missing:
        raise TypeError(f"RobotPlugin is missing {', '.join(missing)}")
    return loaded


def _plugin_name(plugin: object) -> str:
    name = getattr(plugin, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("RobotPlugin.name must be a non-empty string")
    return name


def _coerce_specs(plugin: object, name: str) -> tuple[RobotSpec, ...]:
    raw = plugin.specs()  # type: ignore[attr-defined]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"plugin {name!r} specs() must return a sequence of RobotSpec")
    specs: list[RobotSpec] = []
    for spec in raw:
        if not isinstance(spec, RobotSpec):
            raise TypeError(
                f"plugin {name!r} specs() must return RobotSpec objects, "
                f"not {type(spec).__name__}"
            )
        if not spec.id:
            raise ValueError(f"plugin {name!r} returned a spec with an empty id")
        if not spec.source:
            spec = dataclasses.replace(spec, source=name)
        specs.append(spec)
    return tuple(specs)


def _coerce_mapping(raw: object, what: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError(f"{what}() must return a mapping of name to factory")
    out: dict[str, Any] = {}
    for key, factory in raw.items():
        scheme = str(key).strip().lower()
        if not scheme:
            raise ValueError(f"{what}() has an empty name")
        if not callable(factory):
            raise TypeError(f"{what}() name {scheme!r} is not a factory")
        if scheme in out:
            raise ValueError(f"duplicate {what} name {scheme!r}")
        out[scheme] = factory
    return out


def _load_one(entry: Any, *, fallback: bool) -> PluginInfo:
    entry_name = str(getattr(entry, "name", "") or "")
    name = entry_name or "<unknown>"
    specs: tuple[RobotSpec, ...] = ()
    plugin: object | None = None
    try:
        plugin = _materialize(entry.load())
        name = _plugin_name(plugin)
        specs = _coerce_specs(plugin, name)
        schemes = _coerce_mapping(plugin.transports(), "transports")  # type: ignore[attr-defined]
        backends = _coerce_mapping(plugin.sim_backends(), "sim_backends")  # type: ignore[attr-defined]
    except Exception as exc:
        return PluginInfo(
            name=name,
            entry_name=entry_name,
            spec_ids=tuple(spec.id for spec in specs),
            specs=specs,
            schemes=(),
            sim_backends=(),
            error=f"{type(exc).__name__}: {exc}",
            fallback=fallback,
            distribution=_distribution_name(entry),
            plugin=plugin,
        )
    return PluginInfo(
        name=name,
        entry_name=entry_name,
        spec_ids=tuple(spec.id for spec in specs),
        specs=specs,
        schemes=tuple(schemes),
        sim_backends=tuple(backends),
        error=None,
        fallback=fallback,
        distribution=_distribution_name(entry),
        plugin=plugin,
    )


def _collision_messages(
    info: PluginInfo, seen_ids: Mapping[str, str], seen_schemes: Mapping[str, str]
) -> list[str]:
    messages: list[str] = []
    seen_in_plugin: set[str] = set()
    for robot_id in info.spec_ids:
        if robot_id in seen_in_plugin:
            messages.append(
                f"duplicate robot id {robot_id!r} declared twice by plugin {info.name!r}"
            )
        seen_in_plugin.add(robot_id)
    if messages:
        return messages
    for robot_id in info.spec_ids:
        owner = seen_ids.get(robot_id)
        if owner is not None:
            messages.append(
                f"duplicate robot id {robot_id!r} from plugins {owner!r} and {info.name!r}"
            )
    for scheme in info.schemes:
        owner = seen_schemes.get(scheme)
        if owner is not None:
            messages.append(
                f"duplicate transport scheme {scheme!r} from plugins "
                f"{owner!r} and {info.name!r}"
            )
    return messages


def _reject_collisions(infos: Sequence[PluginInfo]) -> tuple[PluginInfo, ...]:
    seen_ids: dict[str, str] = {}
    seen_schemes: dict[str, str] = {}
    accepted: list[PluginInfo] = []
    for info in infos:
        if info.error:
            accepted.append(info)
            continue
        messages = _collision_messages(info, seen_ids, seen_schemes)
        if messages:
            accepted.append(dataclasses.replace(info, error="; ".join(messages)))
            continue
        for robot_id in info.spec_ids:
            seen_ids[robot_id] = info.name
        for scheme in info.schemes:
            seen_schemes[scheme] = info.name
        accepted.append(info)
    return tuple(accepted)


def _load_all() -> tuple[PluginInfo, ...]:
    loaded = [
        _load_one(entry, fallback=fallback) for entry, fallback in _discover_entries()
    ]
    return _reject_collisions(loaded)


def plugin_named(name: str) -> object | None:
    """The loaded plugin instance for ``name``, or ``None``.

    Discovery stores the instance and does not call ``safety_policy``.
    The gate calls that hook when it loads a policy.
    """
    if not name:
        return None
    for info in loaded_plugins():
        if info.error is None and info.name == name and info.plugin is not None:
            return info.plugin
    return None


def loaded_plugins() -> tuple[PluginInfo, ...]:
    """Discover plugins once per process. Failures are records, not raises."""
    global _CACHE, _in_load
    if _CACHE is not None:
        return _CACHE
    if _in_load:
        raise RuntimeError("plugin discovery re-entered")
    _in_load = True
    try:
        _CACHE = _load_all()
    except Exception:
        _CACHE = None
        raise
    finally:
        _in_load = False
    return _CACHE


def register_loaded_specs() -> None:
    """Register every spec from plugins that loaded cleanly.

    Called from :func:`ohho.registry._ensure_plugins`. A plugin with a load
    error contributes nothing, so the other robots stay available.
    """
    from .registry import register

    for info in loaded_plugins():
        if info.error is not None:
            continue
        for spec in info.specs:
            register(spec)


def reload_plugins() -> tuple[PluginInfo, ...]:
    """Drop the cache and fill the registry from entry points again.

    Specs previously added with ``register`` or ``load_manifest`` are removed.
    """
    global _CACHE, _in_load
    _CACHE = None
    _in_load = False
    from . import registry as registry_mod

    registry_mod._REGISTRY.clear()
    registry_mod._plugins_loaded = False
    registry_mod._plugins_loading = False
    registry_mod._ensure_plugins()
    return loaded_plugins()


def format_plugin_report(infos: Sequence[PluginInfo] | None = None) -> str:
    """Text for ``ohho list --plugins``: each plugin, its specs, any load error."""
    if infos is None:
        infos = loaded_plugins()
    lines: list[str] = []
    for info in infos:
        specs = ", ".join(info.spec_ids) if info.spec_ids else "—"
        label = f"{info.name} (source fallback)" if info.fallback else info.name
        lines.append(f"{label}: {specs}")
        if info.error:
            lines.append(f"  error: {info.error}")
    return "\n".join(lines)


__all__ = [
    "BUILTIN_ENTRY_VALUE",
    "ENTRY_GROUP",
    "DoctorReport",
    "PluginInfo",
    "RobotPlugin",
    "SimFactory",
    "TransportFactory",
    "format_plugin_report",
    "loaded_plugins",
    "plugin_named",
    "register_loaded_specs",
    "reload_plugins",
]
