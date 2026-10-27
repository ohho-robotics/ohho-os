"""Crash diagnostics bundles for a supervised service.

A bundle is a directory under the runtime state directory. The state
directory comes from an explicit argument, then ``OHHO_RUNTIME_STATE_DIR``,
then the manifest ``state_dir``, then ``~/.ohho/runtime``. Each bundle is
named with the service, a UTC timestamp, and the run id.

The directory holds the last structured log records for that service and
for the supervisor, the service state history and restart count, a metrics
snapshot, the resolved manifest with secret-looking values redacted, the
Python and platform versions, the exit code or traceback, and LoopTimer
stats when a loop is attached.

Retention is capped by count and by total size. The oldest bundles are
removed. Writing a bundle emits the ``crash_bundle`` log event and
increments ``ohho_runtime_crash_bundles_total``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ohho
import ohho_runtime
from ohho_runtime.log import RuntimeLog, get_logger
from ohho_runtime.manifest import (
    ConsoleScript,
    PythonCallable,
    Ros2Launch,
    Ros2Node,
    RuntimeManifest,
    SchedulingHint,
    ServiceSpec,
)
from ohho_runtime.metrics import CRASH_BUNDLES_METRIC, Registry, get_registry

STATE_DIR_ENV = "OHHO_RUNTIME_STATE_DIR"
CRASH_DIR_NAME = "crashes"
DEFAULT_LOG_LIMIT = 100
DEFAULT_MAX_BUNDLES = 8
DEFAULT_MAX_BYTES = 1024 * 1024
_HISTORY_CAPACITY = 500
_REDACTED = "***"

BUNDLE_FILES = (
    "bundle.json",
    "failure.json",
    "logs-service.jsonl",
    "logs-supervisor.jsonl",
    "loop.json",
    "manifest.json",
    "metrics.txt",
    "platform.json",
    "state.json",
)

_TIMESTAMP = re.compile(r"\d{8}T\d{12}Z\Z")
_SECRET_KEY = re.compile(
    r"(?i)(password|passwd|secret|token|credential|api[-_]?key|"
    r"authorization|private[-_]?key|access[-_]?key|passphrase|cookie|arming)"
)
_SECRET_VALUE = re.compile(
    r"(?i)bearer\s+\S+|sk-[A-Za-z0-9_\-]{8,}|i am physically present[^\n]*"
)
_EXCEPTION_PREFIXES = (
    "start failed:",
    "configure failed:",
    "stop failed:",
    "service error:",
)

_CREATED_LOCK = threading.Lock()
_LAST_CREATED = 0


class CrashBundleError(ValueError):
    """A crash bundle could not be described or written."""


@dataclass(frozen=True)
class CrashBundle:
    """One written diagnostics directory."""

    path: Path
    service: str
    run_id: str
    reason: str
    timestamp: str


class LogHistory:
    """Bounded store of structured log records. Oldest records are dropped."""

    def __init__(self, capacity: int = _HISTORY_CAPACITY) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise CrashBundleError("log history capacity must be an int >= 1")
        self._items: deque[dict[str, object]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, object]) -> None:
        """Store a shallow copy of ``record``."""
        copied = {str(key): value for key, value in record.items()}
        with self._lock:
            self._items.append(copied)

    def for_service(self, service: str, limit: int) -> list[dict[str, object]]:
        """Return the last ``limit`` records whose ``service`` matches."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise CrashBundleError("log limit must be an int >= 0")
        with self._lock:
            matched = [
                dict(item) for item in self._items if item.get("service") == service
            ]
        if limit == 0:
            return []
        return matched[-limit:]


def default_state_dir() -> Path:
    """Default runtime state directory, ``~/.ohho/runtime``."""
    return Path.home() / ".ohho" / "runtime"


def resolve_state_dir(
    *,
    explicit: str | None = None,
    manifest_dir: str = "",
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the directory that contains crash bundles.

    An explicit argument wins. Otherwise ``OHHO_RUNTIME_STATE_DIR`` overrides
    the manifest ``state_dir``. The default is :func:`default_state_dir`.
    """
    env = os.environ if environ is None else environ
    chosen = ""
    if explicit is not None and str(explicit).strip():
        chosen = str(explicit).strip()
    else:
        raw = env.get(STATE_DIR_ENV, "")
        if raw.strip():
            chosen = raw.strip()
        elif manifest_dir.strip():
            chosen = manifest_dir.strip()
    if not chosen:
        return default_state_dir()
    if "\x00" in chosen:
        raise CrashBundleError("state dir must not contain a NUL")
    return Path(chosen).expanduser()


def classify_crash(
    *,
    detail: str,
    exc: BaseException | None,
    exit_code: int | None,
    outcome: str,
) -> str | None:
    """Return the bundle reason, or ``None`` when this failure is not a crash.

    A crash is a non-zero exit, an unhandled exception, a heartbeat timeout
    that schedules a restart, or a restart limit. One failure produces one
    reason. The restart limit wins when that failure is the one that hits it.
    """
    if outcome == "restart_limit":
        return "restart_limit"
    if (
        isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code != 0
    ):
        return "nonzero_exit"
    if exc is not None or _exception_detail(detail):
        return "unhandled_exception"
    if outcome == "restart_scheduled" and _heartbeat_detail(detail):
        return "heartbeat_timeout"
    return None


def read_exit_code(service: object) -> int | None:
    """Return a service exit status, or ``None`` when it has not exited."""
    for name in ("exit_code", "returncode"):
        value = getattr(service, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None


def read_failure_exception(service: object) -> BaseException | None:
    """Return an exception the service stored, if it stored one."""
    exc = getattr(service, "failure_exception", None)
    if isinstance(exc, BaseException):
        return exc
    return None


def redact(value: object) -> object:
    """Replace secret-looking keys and values with ``***``.

    Credentials, tokens, and the arming phrase are removed. Other strings,
    numbers, and lists are kept.
    """
    return _redact(_jsonable(value))


def write_crash_bundle(
    *,
    service: str,
    run_id: str,
    reason: str = "requested",
    state_dir: str | None = None,
    manifest: RuntimeManifest | Mapping[str, object] | None = None,
    service_logs: Sequence[Mapping[str, object]] | None = None,
    supervisor_logs: Sequence[Mapping[str, object]] | None = None,
    history: Sequence[str] | None = None,
    state: str = "",
    restarts: int = 0,
    restart_exhausted: bool = False,
    failure_reason: str = "",
    metrics_text: str = "",
    exit_code: int | None = None,
    exc: BaseException | None = None,
    traceback_text: str = "",
    exc_type: str = "",
    detail: str = "",
    loop: Mapping[str, object] | None = None,
    robot_class: str = "",
    timestamp: str | None = None,
    log_limit: int = DEFAULT_LOG_LIMIT,
    max_bundles: int = DEFAULT_MAX_BUNDLES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    logger: RuntimeLog | None = None,
    metrics: Registry | None = None,
    environ: Mapping[str, str] | None = None,
) -> CrashBundle:
    """Write one crash bundle and return its directory.

    This is the on-demand API. The supervisor calls it when a service
    crashes. Secret-looking manifest, log, and failure text is redacted.
    """
    role = _token(service, "service")
    run = _token(run_id, "run_id")
    why = _token(reason, "reason")
    _bounds(log_limit, max_bundles, max_bytes)
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise CrashBundleError("exit_code must be an int or None")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 0:
        raise CrashBundleError("restarts must be an int >= 0")
    stamp = _timestamp(timestamp)
    manifest_dir = manifest.state_dir if isinstance(manifest, RuntimeManifest) else ""
    root = resolve_state_dir(
        explicit=state_dir,
        manifest_dir=manifest_dir,
        environ=environ,
    )
    crashes = root / CRASH_DIR_NAME
    directory = _bundle_dir(crashes, role, stamp, run)
    tb, kind = _failure_text(exc, traceback_text, exc_type)
    service_records = _tail(service_logs, log_limit)
    supervisor_records = _tail(supervisor_logs, log_limit)
    document = _manifest_document(manifest)
    complete = False
    try:
        crashes.mkdir(parents=True, exist_ok=True)
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(
            directory / "manifest.json",
            redact(document),
        )
        _write_json(directory / "platform.json", _platform_info())
        _write_json(
            directory / "failure.json",
            redact(
                {
                    "reason": why,
                    "detail": detail,
                    "exit_code": exit_code,
                    "exc_type": kind,
                    "traceback": tb,
                }
            ),
        )
        _write_json(
            directory / "state.json",
            redact(
                {
                    "service": role,
                    "robot_class": robot_class,
                    "state": state,
                    "history": list(history or ()),
                    "restarts": restarts,
                    "restart_exhausted": bool(restart_exhausted),
                    "failure_reason": failure_reason,
                }
            ),
        )
        _write_jsonl(directory / "logs-service.jsonl", redact(service_records))
        _write_jsonl(directory / "logs-supervisor.jsonl", redact(supervisor_records))
        if loop is None:
            loop_document: dict[str, object] = {"attached": False}
        else:
            loop_document = {"attached": True, "stats": redact(dict(loop))}
        _write_json(directory / "loop.json", loop_document)
        rendered = (
            metrics_text
            if metrics_text.endswith("\n") or not metrics_text
            else (metrics_text + "\n")
        )
        (directory / "metrics.txt").write_text(
            _SECRET_VALUE.sub(_REDACTED, rendered),
            encoding="utf-8",
        )
        created = _created_ns()
        _write_json(
            directory / "bundle.json",
            {
                "kind": "ohho_runtime_crash_bundle",
                "service": role,
                "robot_class": robot_class,
                "timestamp": stamp,
                "run_id": run,
                "reason": why,
                "created_ns": created,
            },
        )
        complete = True
        _prune(crashes, keep=directory, max_bundles=max_bundles, max_bytes=max_bytes)
    except Exception:
        if not complete and directory.exists():
            shutil.rmtree(directory, ignore_errors=False)
        raise
    bundle = CrashBundle(
        path=directory,
        service=role,
        run_id=run,
        reason=why,
        timestamp=stamp,
    )
    _record_written(
        bundle,
        robot_class=robot_class,
        logger=logger,
        metrics=metrics,
    )
    return bundle


def _record_written(
    bundle: CrashBundle,
    *,
    robot_class: str,
    logger: RuntimeLog | None,
    metrics: Registry | None,
) -> None:
    registry = metrics if isinstance(metrics, Registry) else get_registry()
    registry.inc(
        CRASH_BUNDLES_METRIC,
        labels={
            "service": bundle.service,
            "reason": bundle.reason,
            "robot_class": robot_class,
            "run_id": bundle.run_id,
        },
    )
    log = get_logger() if logger is None else logger
    log.log(
        "ERROR",
        event="crash_bundle",
        service=bundle.service,
        robot_class=robot_class,
        run_id=bundle.run_id,
        message=bundle.reason,
        bundle=bundle.path.name,
        crash_reason=bundle.reason,
    )


def _token(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ch.isspace() for ch in value)
    ):
        raise CrashBundleError(f"{what} must be a non-empty token")
    return value.strip()


def _bounds(log_limit: int, max_bundles: int, max_bytes: int) -> None:
    for name, value in (
        ("log_limit", log_limit),
        ("max_bundles", max_bundles),
        ("max_bytes", max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CrashBundleError(f"{name} must be an int >= 1")


def _timestamp(value: str | None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
        return moment.strftime("%Y%m%dT%H%M%S") + f"{moment.microsecond:06d}Z"
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise CrashBundleError("timestamp must be YYYYMMDDTHHMMSSFFFFFFZ")
    return value


def _safe(value: str, fallback: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in "-_" else "_" for ch in value]
    text = "".join(cleaned).strip("._")[:64]
    return text or fallback


def _bundle_dir(crashes: Path, service: str, stamp: str, run_id: str) -> Path:
    base = f"{_safe(service, 'service')}-{stamp}-{_safe(run_id, 'run')}"
    candidate = crashes / base
    suffix = 2
    while candidate.exists():
        candidate = crashes / f"{base}-{suffix}"
        suffix += 1
    return candidate


def _created_ns() -> int:
    global _LAST_CREATED
    with _CREATED_LOCK:
        now = time.time_ns()
        if now <= _LAST_CREATED:
            now = _LAST_CREATED + 1
        _LAST_CREATED = now
        return now


def _exception_detail(detail: str) -> bool:
    text = detail.strip().lower()
    return any(text.startswith(prefix) for prefix in _EXCEPTION_PREFIXES)


def _heartbeat_detail(detail: str) -> bool:
    return detail.strip().lower() == "liveness timeout"


def _failure_text(
    exc: BaseException | None,
    traceback_text: str,
    exc_type: str,
) -> tuple[str, str]:
    kind = exc_type
    text = traceback_text
    if exc is not None:
        if not kind:
            kind = type(exc).__name__
        if not text:
            text = "".join(traceback.format_exception(exc))
    return text, kind


def _tail(
    records: Sequence[Mapping[str, object]] | None,
    limit: int,
) -> list[dict[str, object]]:
    if records is None:
        return []
    copied = [{str(key): value for key, value in record.items()} for record in records]
    return copied[-limit:]


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _redact(value: object) -> object:
    if isinstance(value, str):
        return _SECRET_VALUE.sub(_REDACTED, value)
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_KEY.search(name):
                redacted[name] = _REDACTED
            else:
                redacted[name] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _manifest_document(
    manifest: RuntimeManifest | Mapping[str, object] | None,
) -> object:
    if manifest is None:
        return {}
    if isinstance(manifest, RuntimeManifest):
        return {
            "log_level": manifest.log_level,
            "log_format": manifest.log_format,
            "state_dir": manifest.state_dir,
            "services": [_service_document(service) for service in manifest.services],
        }
    if isinstance(manifest, Mapping):
        return {str(key): value for key, value in manifest.items()}
    raise CrashBundleError("manifest must be a RuntimeManifest or a mapping")


def _service_document(service: ServiceSpec) -> dict[str, object]:
    scheduling = service.scheduling
    return {
        "role": service.role,
        "entry": _entry_document(service.entry),
        "depends_on": list(service.depends_on),
        "criticality": service.criticality.value,
        "readiness_timeout_s": service.readiness_timeout_s,
        "liveness_timeout_s": service.liveness_timeout_s,
        "restart": {
            "kind": service.restart.kind.value,
            "max_restarts": service.restart.max_restarts,
            "window_s": service.restart.window_s,
            "backoff_s": service.restart.backoff_s,
            "only_while_disarmed": service.restart.only_while_disarmed,
        },
        "loop_hz": service.loop_hz,
        "qos": list(service.qos),
        "scheduling": None if scheduling is None else _scheduling_document(scheduling),
        "commands_motion": service.commands_motion,
        "do_not_operate": service.do_not_operate,
    }


def _scheduling_document(hint: SchedulingHint) -> dict[str, object]:
    return {
        "period_s": hint.period_s,
        "priority": hint.priority,
        "cpu_affinity": list(hint.cpu_affinity),
    }


def _entry_document(entry: object) -> dict[str, object]:
    if isinstance(entry, PythonCallable):
        target = entry.target
        module = getattr(target, "__module__", "")
        qualname = getattr(target, "__qualname__", "")
        if (
            isinstance(module, str)
            and isinstance(qualname, str)
            and module
            and qualname
        ):
            name = f"{module}.{qualname}"
        elif isinstance(qualname, str) and qualname:
            name = qualname
        else:
            name = type(target).__name__
        return {"kind": "python", "target": name}
    if isinstance(entry, ConsoleScript):
        return {"kind": "console", "name": entry.name}
    if isinstance(entry, Ros2Launch):
        return {
            "kind": "ros2_launch",
            "package": entry.package,
            "launch_file": entry.launch_file,
        }
    if isinstance(entry, Ros2Node):
        return {"kind": "ros2_node", "package": entry.package, "node": entry.node}
    return {"kind": type(entry).__name__}


def _platform_info() -> dict[str, str]:
    python = sys.version.splitlines()[0]
    return {
        "python_version": platform.python_version(),
        "python": python,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ohho_version": ohho.__version__,
        "ohho_runtime_version": ohho_runtime.__version__,
    }


def _write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: object) -> None:
    lines: list[str] = []
    if isinstance(records, list):
        for record in records:
            lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _created_of(path: Path) -> int:
    marker = path / "bundle.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return 0
    value = payload.get("created_ns", 0) if isinstance(payload, dict) else 0
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _bundles(crashes: Path) -> list[Path]:
    if not crashes.is_dir():
        return []
    found: list[Path] = []
    for child in crashes.iterdir():
        if child.is_dir() and (child / "bundle.json").is_file():
            found.append(child)
    found.sort(key=lambda item: (_created_of(item), item.name))
    return found


def _prune(crashes: Path, *, keep: Path, max_bundles: int, max_bytes: int) -> None:
    bundles = _bundles(crashes)

    def total() -> int:
        return sum(_dir_size(item) for item in bundles)

    while True:
        over_count = len(bundles) > max_bundles
        over_size = len(bundles) > 1 and total() > max_bytes
        if not over_count and not over_size:
            return
        victim = next((item for item in bundles if item != keep), None)
        if victim is None:
            return
        shutil.rmtree(victim)
        bundles.remove(victim)


__all__ = [
    "BUNDLE_FILES",
    "CRASH_DIR_NAME",
    "DEFAULT_LOG_LIMIT",
    "DEFAULT_MAX_BUNDLES",
    "DEFAULT_MAX_BYTES",
    "STATE_DIR_ENV",
    "CrashBundle",
    "CrashBundleError",
    "LogHistory",
    "classify_crash",
    "default_state_dir",
    "read_exit_code",
    "read_failure_exception",
    "redact",
    "resolve_state_dir",
    "write_crash_bundle",
]
