"""Load a runtime manifest from a JSON document.

The file names an entry the runtime already ships. It does not import an
arbitrary callable, and loading it does not start a service or arm.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ohho.robot_spec import RobotClass, parse_robot_class

from ohho_runtime import entries, sim_services
from ohho_runtime.manifest import (
    ManifestError,
    PythonCallable,
    RestartPolicy,
    RuntimeManifest,
    ServiceSpec,
    class_default_manifest,
    load_manifest,
    make_service,
)

_DOC_KEYS = frozenset({"robot_class", "services", "log_level", "log_format"})
_SERVICE_KEYS = frozenset(
    {
        "role",
        "entry",
        "depends_on",
        "criticality",
        "readiness_timeout_s",
        "liveness_timeout_s",
        "restart",
        "loop_hz",
        "qos",
        "commands_motion",
        "do_not_operate",
    }
)
_RESTART_KEYS = frozenset(
    {"kind", "max_restarts", "window_s", "backoff_s", "only_while_disarmed"}
)

_TARGETS = {
    "sim.ready": sim_services.ready,
    "sim.fail": sim_services.fail,
    "entries.safety_watchdog": entries.safety_watchdog,
    "entries.transport": entries.transport,
    "entries.telemetry": entries.telemetry,
    "entries.teleop_gateway": entries.teleop_gateway,
    "entries.autopilot_heartbeat": entries.autopilot_heartbeat,
    "entries.mode_monitor": entries.mode_monitor,
}


def class_manifest(name: str) -> tuple[RuntimeManifest, RobotClass]:
    """Return the class default manifest and its robot class."""
    try:
        kind = parse_robot_class(name)
    except ValueError as exc:
        raise ManifestError(str(exc)) from None
    return class_default_manifest(kind), kind


def load_manifest_path(path: Path) -> tuple[RuntimeManifest, RobotClass]:
    """Read a JSON manifest from ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    return load_manifest_document(raw)


def load_manifest_document(
    document: Mapping[str, object],
) -> tuple[RuntimeManifest, RobotClass]:
    """Validate a JSON object and return the manifest and robot class."""
    unknown = set(document) - _DOC_KEYS
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise ManifestError(f"manifest has unknown fields: {names}")
    robot_class = document.get("robot_class")
    if not isinstance(robot_class, str) or not robot_class.strip():
        raise ManifestError("manifest robot_class must be a string")
    try:
        kind = parse_robot_class(robot_class)
    except ValueError as exc:
        raise ManifestError(str(exc)) from None
    services = document.get("services")
    if not isinstance(services, list):
        raise ManifestError("manifest services must be a list")
    specs = tuple(_service(item) for item in services)
    log_level = _optional_text(document.get("log_level", ""), "log_level")
    log_format = _optional_text(document.get("log_format", ""), "log_format")
    return (
        load_manifest(specs, log_level=log_level, log_format=log_format),
        kind,
    )


def _optional_text(value: object, what: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ManifestError(f"manifest {what} must be a string")
    return value


def _service(value: object) -> ServiceSpec:
    if not isinstance(value, Mapping):
        raise ManifestError("each service must be an object")
    unknown = set(value) - _SERVICE_KEYS
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise ManifestError(f"service has unknown fields: {names}")
    role = value.get("role")
    if not isinstance(role, str):
        raise ManifestError("service role must be a string")
    if "loop_hz" not in value:
        raise ManifestError(f"service {role} needs loop_hz")
    loop_hz = _number(value.get("loop_hz"), f"service {role} loop_hz")
    kwargs: dict[str, object] = {
        "depends_on": _strings(
            value.get("depends_on", []), f"service {role} depends_on"
        ),
        "restart": _restart(value.get("restart", "never"), role),
        "loop_hz": loop_hz,
        "qos": _strings(value.get("qos", []), f"service {role} qos"),
        "commands_motion": _bool(
            value.get("commands_motion", False), f"service {role} commands_motion"
        ),
        "do_not_operate": _bool(
            value.get("do_not_operate", False), f"service {role} do_not_operate"
        ),
    }
    if "criticality" in value:
        criticality = value.get("criticality")
        if not isinstance(criticality, str):
            raise ManifestError(f"service {role} criticality must be a string")
        kwargs["criticality"] = criticality
    if "readiness_timeout_s" in value:
        kwargs["readiness_timeout_s"] = _number(
            value.get("readiness_timeout_s"),
            f"service {role} readiness_timeout_s",
        )
    if "liveness_timeout_s" in value:
        kwargs["liveness_timeout_s"] = _number(
            value.get("liveness_timeout_s"),
            f"service {role} liveness_timeout_s",
        )
    return make_service(role, _entry(value.get("entry"), role), **kwargs)  # type: ignore[arg-type]


def _entry(name: object, role: str) -> PythonCallable:
    if not isinstance(name, str) or name not in _TARGETS:
        allowed = ", ".join(sorted(_TARGETS))
        raise ManifestError(
            f"service {role} entry {name!r} is not a runtime entry; "
            f"expected one of {allowed}"
        )
    return PythonCallable(_TARGETS[name])


def _restart(value: object, role: str) -> RestartPolicy:
    if value is None or value == "never":
        return RestartPolicy()
    if not isinstance(value, Mapping):
        raise ManifestError(f"service {role} restart must be 'never' or an object")
    unknown = set(value) - _RESTART_KEYS
    if unknown:
        names = ", ".join(repr(name) for name in sorted(unknown))
        raise ManifestError(f"service {role} restart has unknown fields: {names}")
    kind = value.get("kind", "on-failure")
    only_while_disarmed = value.get("only_while_disarmed", False)
    if not isinstance(only_while_disarmed, bool):
        raise ManifestError(f"service {role} only_while_disarmed must be a bool")
    max_restarts = value.get("max_restarts", 0)
    if isinstance(max_restarts, bool) or not isinstance(max_restarts, int):
        raise ManifestError(f"service {role} max_restarts must be an int")
    return RestartPolicy(
        kind=kind,  # type: ignore[arg-type]
        max_restarts=max_restarts,
        window_s=_number(value.get("window_s", 0), f"service {role} window_s"),
        backoff_s=_number(value.get("backoff_s", 0), f"service {role} backoff_s"),
        only_while_disarmed=only_while_disarmed,
    )


def _strings(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{what} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ManifestError(f"{what} must be a list of strings")
        items.append(item)
    return tuple(items)


def _bool(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{what} must be a bool")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{what} must be a number")
    return float(value)


__all__ = [
    "class_manifest",
    "load_manifest_document",
    "load_manifest_path",
]
