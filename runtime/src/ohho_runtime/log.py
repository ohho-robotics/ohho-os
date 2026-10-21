"""Structured logging for the onboard runtime.

Every record carries the same fields: ``timestamp``, ``level``, ``service``,
``event``, ``robot_class``, and ``run_id``. The default format is one JSON
object per line. ``text`` format is the human-readable development mode.

The level comes from the manifest when that field is set, and
``OHHO_RUNTIME_LOG_LEVEL`` overrides it. The format comes from the manifest
``log_format`` field, and ``OHHO_RUNTIME_LOG_FORMAT`` overrides it.
``OHHO_RUNTIME_RUN_ID`` pins the run id; otherwise the process generates one.

This module does not import ``ohho``. A worker that only imports
:mod:`ohho_runtime.heartbeat` can log without loading the rest of the stack.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import TextIO

LOG_LEVEL_ENV = "OHHO_RUNTIME_LOG_LEVEL"
LOG_FORMAT_ENV = "OHHO_RUNTIME_LOG_FORMAT"
RUN_ID_ENV = "OHHO_RUNTIME_RUN_ID"

LOG_FIELDS = ("timestamp", "level", "service", "event", "robot_class", "run_id")

_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_PROCESS_RUN_ID = ""
_PROCESS_RUN_LOCK = threading.Lock()


class _Sink:
    """One stream shared by a logger and the children it binds."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.lock = threading.Lock()


def normalize_level(value: str) -> str:
    """Return the canonical level name for ``value``.

    ``WARN`` is accepted as ``WARNING``. Anything else raises ``ValueError``.
    """
    token = value.strip().upper()
    if token == "WARN":
        token = "WARNING"
    if token not in _LEVELS:
        allowed = "DEBUG, INFO, WARNING, ERROR, CRITICAL"
        raise ValueError(f"unknown log level {value!r}; expected {allowed}")
    return token


def normalize_format(value: str, *, allow_empty: bool = False) -> str:
    """Return ``json`` or ``text``.

    ``human`` and ``dev`` are the text mode. An empty string stays empty when
    ``allow_empty`` is set, so a manifest can mean "not set".
    """
    token = value.strip().lower()
    if token == "":
        if allow_empty:
            return ""
        return "json"
    if token in {"json", "jsonl"}:
        return "json"
    if token in {"text", "human", "dev"}:
        return "text"
    raise ValueError(f"unknown log format {value!r}; expected json or text")


def resolve_log_level(
    manifest_level: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the active level.

    ``OHHO_RUNTIME_LOG_LEVEL`` wins when it is set and non-empty. Otherwise
    the manifest level is used. The default is ``INFO``.
    """
    env = os.environ if environ is None else environ
    raw = env.get(LOG_LEVEL_ENV)
    if raw is not None and raw.strip():
        return normalize_level(raw)
    if manifest_level is not None and str(manifest_level).strip():
        return normalize_level(str(manifest_level))
    return "INFO"


def resolve_log_format(
    manifest_format: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve ``json`` or ``text``.

    ``OHHO_RUNTIME_LOG_FORMAT`` wins when it is set and non-empty. Otherwise
    the manifest format is used. The default is ``json``.
    """
    env = os.environ if environ is None else environ
    raw = env.get(LOG_FORMAT_ENV)
    if raw is not None and raw.strip():
        return normalize_format(raw)
    if manifest_format is not None and str(manifest_format).strip():
        return normalize_format(str(manifest_format))
    return "json"


def resolve_run_id(
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return ``explicit``, else ``OHHO_RUNTIME_RUN_ID``, else a new id."""
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env = os.environ if environ is None else environ
    raw = env.get(RUN_ID_ENV, "")
    if raw.strip():
        return raw.strip()
    return uuid.uuid4().hex


def process_run_id() -> str:
    """Run id for logs that are not bound to a supervisor."""
    global _PROCESS_RUN_ID
    with _PROCESS_RUN_LOCK:
        if not _PROCESS_RUN_ID:
            _PROCESS_RUN_ID = resolve_run_id(None)
        return _PROCESS_RUN_ID


def _timestamp(clock: Callable[[], datetime | float] | None) -> str:
    if clock is None:
        moment = datetime.now(timezone.utc)
    else:
        value = clock()
        if isinstance(value, datetime):
            moment = value
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            else:
                moment = moment.astimezone(timezone.utc)
        else:
            moment = datetime.fromtimestamp(float(value), tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond:06d}Z"


def _format_text(record: Mapping[str, object]) -> str:
    base = (
        f"{record['timestamp']} {record['level']} "
        f"service={record['service']} event={record['event']} "
        f"robot_class={record['robot_class']} run_id={record['run_id']}"
    )
    extras: list[str] = []
    message = record.get("message", "")
    if message:
        extras.append(f"message={message}")
    reserved = set(LOG_FIELDS) | {"message"}
    for key in sorted(record):
        if key in reserved:
            continue
        extras.append(f"{key}={record[key]}")
    if not extras:
        return base
    return base + " " + " ".join(extras)


def format_record(record: Mapping[str, object], log_format: str) -> str:
    """Render one record as a JSON line or a human line."""
    if log_format == "text":
        return _format_text(record)
    payload = {key: record[key] for key in record}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class RuntimeLog:
    """Logger that writes one structured record per call.

    ``bind`` shares the stream and keeps ``robot_class`` and ``run_id``.
    A record below :attr:`level` is dropped and the call returns ``None``.
    """

    def __init__(
        self,
        *,
        service: str = "runtime",
        robot_class: str = "",
        run_id: str = "",
        level: str = "INFO",
        log_format: str = "json",
        stream: TextIO | None = None,
        clock: Callable[[], datetime | float] | None = None,
        sink: _Sink | None = None,
    ) -> None:
        self.service = service
        self.robot_class = robot_class
        self.run_id = run_id
        self.level = normalize_level(level)
        self.log_format = normalize_format(log_format)
        self._clock = clock
        self._sink = (
            sink
            if sink is not None
            else _Sink(sys.stderr if stream is None else stream)
        )

    def bind(self, *, service: str) -> RuntimeLog:
        """Return a logger for ``service`` on the same stream."""
        return RuntimeLog(
            service=service,
            robot_class=self.robot_class,
            run_id=self.run_id,
            level=self.level,
            log_format=self.log_format,
            clock=self._clock,
            sink=self._sink,
        )

    def log(
        self,
        level: str,
        *,
        event: str,
        service: str | None = None,
        robot_class: str | None = None,
        run_id: str | None = None,
        message: str = "",
        **extra: object,
    ) -> dict[str, object] | None:
        """Write one record when ``level`` is enabled.

        The returned mapping is the record that was written. ``None`` means
        the level was filtered out.
        """
        canonical = normalize_level(level)
        if _LEVELS[canonical] < _LEVELS[self.level]:
            return None
        if not isinstance(event, str) or not event or any(ch.isspace() for ch in event):
            raise ValueError("event must be a non-empty token")
        record: dict[str, object] = {
            "timestamp": _timestamp(self._clock),
            "level": canonical,
            "service": self.service if service is None else service,
            "event": event,
            "robot_class": self.robot_class if robot_class is None else robot_class,
            "run_id": self.run_id if run_id is None else run_id,
            "message": message,
        }
        for key, value in extra.items():
            if key in record:
                raise ValueError(f"reserved log field {key}")
            record[key] = value
        line = format_record(record, self.log_format)
        with self._sink.lock:
            self._sink.stream.write(line + "\n")
            flush = getattr(self._sink.stream, "flush", None)
            if callable(flush):
                flush()
        return record


def get_logger() -> RuntimeLog:
    """Process logger using the current level and format environment."""
    return RuntimeLog(
        service="runtime",
        robot_class="",
        run_id=process_run_id(),
        level=resolve_log_level(None),
        log_format=resolve_log_format(None),
    )


def note_exception(
    exc: BaseException,
    *,
    service: str,
    event: str,
    robot_class: str = "",
    run_id: str = "",
    message: str = "",
    level: str = "ERROR",
    logger: RuntimeLog | None = None,
    metrics: object | None = None,
) -> None:
    """Log ``exc`` with context and count it.

    The metric is ``ohho_runtime_exceptions_total``. Counting happens even
    when the log line is below the active level, so a filtered line is still
    not a silent swallow. This function does not re-raise.
    """
    log = get_logger() if logger is None else logger
    resolved_robot = robot_class or log.robot_class
    resolved_run = run_id or log.run_id
    text = message or f"{type(exc).__name__}: {exc}"
    log.log(
        level,
        service=service,
        event=event,
        robot_class=resolved_robot,
        run_id=resolved_run,
        message=text,
        exc_type=type(exc).__name__,
    )
    from ohho_runtime.metrics import record_exception

    record_exception(
        service=service,
        event=event,
        exc_type=type(exc).__name__,
        robot_class=resolved_robot,
        run_id=resolved_run,
        registry=metrics,
    )


__all__ = [
    "LOG_FIELDS",
    "LOG_FORMAT_ENV",
    "LOG_LEVEL_ENV",
    "RUN_ID_ENV",
    "RuntimeLog",
    "format_record",
    "get_logger",
    "normalize_format",
    "normalize_level",
    "note_exception",
    "process_run_id",
    "resolve_log_format",
    "resolve_log_level",
    "resolve_run_id",
]
