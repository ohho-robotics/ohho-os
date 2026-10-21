"""Prometheus text metrics for the onboard runtime.

The registry and the renderer are pure: tests read :meth:`Registry.render`
without opening a socket. :class:`MetricsEndpoint` serves ``GET /metrics``
on loopback (``127.0.0.1``) unless a host is set. The port defaults to
9470. ``OHHO_RUNTIME_METRICS=1`` turns the endpoint on; ``0`` turns it off.
``OHHO_RUNTIME_METRICS_HOST`` and ``OHHO_RUNTIME_METRICS_PORT`` override the
bind. An explicit constructor argument wins over the environment.

No third-party metrics library is required. This module does not import
``ohho``.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_METRICS_HOST = "127.0.0.1"
DEFAULT_METRICS_PORT = 9470
METRICS_ENABLED_ENV = "OHHO_RUNTIME_METRICS"
METRICS_HOST_ENV = "OHHO_RUNTIME_METRICS_HOST"
METRICS_PORT_ENV = "OHHO_RUNTIME_METRICS_PORT"
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# name, type, help. Order is the order of the exposition.
CATALOG: tuple[tuple[str, str, str], ...] = (
    (
        "ohho_runtime_service_state",
        "gauge",
        "Current lifecycle state of a supervised service (1 = current).",
    ),
    (
        "ohho_runtime_service_restarts_total",
        "counter",
        "Times a supervised service has been restarted.",
    ),
    (
        "ohho_runtime_heartbeat_age_seconds",
        "gauge",
        "Seconds since the service last heartbeat.",
    ),
    (
        "ohho_runtime_heartbeat_misses_total",
        "counter",
        "Heartbeat deadlines the service has missed.",
    ),
    (
        "ohho_runtime_heartbeat_beats_total",
        "counter",
        "Heartbeat lines a subprocess service has accepted.",
    ),
    (
        "ohho_runtime_start_duration_seconds",
        "gauge",
        "Seconds the last start took.",
    ),
    (
        "ohho_runtime_stop_duration_seconds",
        "gauge",
        "Seconds the last stop took.",
    ),
    (
        "ohho_runtime_loop_cycle_seconds",
        "gauge",
        "LoopTimer cycle time in seconds at the labeled quantile.",
    ),
    (
        "ohho_runtime_loop_jitter_seconds",
        "gauge",
        "LoopTimer inter-start jitter in seconds.",
    ),
    (
        "ohho_runtime_loop_deadline_misses_total",
        "counter",
        "LoopTimer deadline misses for the service that owns the loop.",
    ),
    (
        "ohho_runtime_loop_samples",
        "gauge",
        "LoopTimer samples reported for the service that owns the loop.",
    ),
    (
        "ohho_runtime_loop_lateness_seconds",
        "gauge",
        "Latest LoopTimer lateness in seconds.",
    ),
    (
        "ohho_runtime_exceptions_total",
        "counter",
        "Caught runtime exceptions that were logged.",
    ),
    (
        "ohho_runtime_do_not_operate",
        "gauge",
        "1 when the supervisor do-not-operate latch is set.",
    ),
    (
        "ohho_runtime_allows_motion",
        "gauge",
        "1 when the supervisor allows motion.",
    ),
    (
        "ohho_runtime_supervisor_state",
        "gauge",
        "Current supervisor state (1 = current).",
    ),
)

METRIC_NAMES = tuple(item[0] for item in CATALOG)

_LOOP_QUANTILES = (("0.5", "p50_s"), ("0.95", "p95_s"), ("0.99", "p99_s"))

_SampleKey = tuple[str, tuple[tuple[str, str], ...]]

_REGISTRY: Registry | None = None
_REGISTRY_LOCK = threading.Lock()


def _label_key(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for key, value in labels.items():
        if not isinstance(key, str) or not key or any(ch.isspace() for ch in key):
            raise ValueError("metric label name must be a non-empty token")
        items.append((key, str(value)))
    return tuple(sorted(items))


def _without_state(
    labels: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(item for item in labels if item[0] != "state")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: float) -> str:
    if value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return format(value, ".12g")


def _parse_bool(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{METRICS_ENABLED_ENV} must be 0 or 1, not {value!r}")


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metrics port must be an int, not {value!r}") from exc
    if isinstance(value, bool) or port < 0 or port > 65535:
        raise ValueError("metrics port must be an int from 0 to 65535")
    return port


class MetricsBind:
    """Resolved endpoint settings. ``enabled`` false does not bind."""

    __slots__ = ("enabled", "host", "port")

    def __init__(self, *, enabled: bool, host: str, port: int) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port


def resolve_metrics_bind(
    *,
    enabled: bool | None = None,
    host: str | None = None,
    port: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> MetricsBind:
    """Resolve enablement, host, and port.

    An argument that is not ``None`` wins. Otherwise the environment is
    read. The default is disabled, ``127.0.0.1``, port 9470.
    """
    env = os.environ if environ is None else environ
    if enabled is None:
        raw = env.get(METRICS_ENABLED_ENV, "")
        enabled = _parse_bool(raw) if raw.strip() else False
    if host is None:
        host = env.get(METRICS_HOST_ENV, "").strip() or DEFAULT_METRICS_HOST
    if port is None:
        raw_port = env.get(METRICS_PORT_ENV, "").strip()
        port = _parse_port(raw_port) if raw_port else DEFAULT_METRICS_PORT
    else:
        port = _parse_port(str(port))
    if (
        not isinstance(host, str)
        or not host.strip()
        or any(ch.isspace() for ch in host)
    ):
        raise ValueError("metrics host must be a non-empty address")
    return MetricsBind(enabled=bool(enabled), host=host.strip(), port=port)


class Registry:
    """In-process metric samples and a Prometheus text renderer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, str] = {}
        self._type: dict[str, str] = {}
        self._samples: dict[_SampleKey, float] = {}
        for name, kind, help_text in CATALOG:
            self.define(name, kind, help_text)

    def define(self, name: str, kind: str, help_text: str) -> None:
        """Register a metric. ``kind`` is ``counter`` or ``gauge``."""
        if kind not in {"counter", "gauge"}:
            raise ValueError("metric type must be counter or gauge")
        if not help_text or "\n" in help_text:
            raise ValueError("metric help must be one non-empty line")
        self._help[name] = help_text
        self._type[name] = kind

    def _check_value(self, name: str, value: object) -> float | None:
        if name not in self._help:
            raise KeyError(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("metric value must be a number")
        number = float(value)
        if not math.isfinite(number):
            return None
        return number

    def set(
        self,
        name: str,
        value: object,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Set one sample to ``value``. Non-finite values are skipped."""
        number = self._check_value(name, value)
        if number is None:
            return
        key = (name, _label_key({} if labels is None else labels))
        with self._lock:
            self._samples[key] = number

    def inc(
        self,
        name: str,
        value: object = 1.0,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Add ``value`` to one sample."""
        number = self._check_value(name, value)
        if number is None:
            return
        key = (name, _label_key({} if labels is None else labels))
        with self._lock:
            self._samples[key] = self._samples.get(key, 0.0) + number

    def set_state(self, name: str, state: str, labels: Mapping[str, str]) -> None:
        """Publish 1 for ``state`` and drop other states with the same labels."""
        if name not in self._help:
            raise KeyError(name)
        base = {key: value for key, value in labels.items() if key != "state"}
        base_key = _label_key(base)
        full = _label_key({**base, "state": state})
        with self._lock:
            drop = [
                key
                for key in self._samples
                if key[0] == name and _without_state(key[1]) == base_key
            ]
            for key in drop:
                del self._samples[key]
            self._samples[(name, full)] = 1.0

    def render(self) -> str:
        """Return the exposition, including help and type lines with no samples."""
        with self._lock:
            help_items = list(self._help.items())
            types = dict(self._type)
            samples = list(self._samples.items())
        grouped: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = {}
        for (name, labels), value in samples:
            grouped.setdefault(name, []).append((labels, value))
        lines: list[str] = []
        for name, help_text in help_items:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {types[name]}")
            family = sorted(grouped.get(name, ()), key=lambda item: item[0])
            for labels, value in family:
                rendered = _format_number(value)
                if labels:
                    body = ",".join(
                        f'{key}="{_escape_label(val)}"' for key, val in labels
                    )
                    lines.append(f"{name}{{{body}}} {rendered}")
                else:
                    lines.append(f"{name} {rendered}")
            lines.append("")
        text = "\n".join(lines).rstrip()
        return text + "\n"


def get_registry() -> Registry:
    """Process registry used when a caller does not pass one."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = Registry()
        return _REGISTRY


def reset_registry() -> Registry:
    """Replace the process registry. Tests use this for isolation."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = Registry()
        return _REGISTRY


def record_exception(
    *,
    service: str,
    event: str,
    exc_type: str,
    robot_class: str = "",
    run_id: str = "",
    registry: object | None = None,
) -> None:
    """Increment ``ohho_runtime_exceptions_total``."""
    target = registry if isinstance(registry, Registry) else get_registry()
    target.inc(
        "ohho_runtime_exceptions_total",
        labels={
            "service": service,
            "event": event,
            "exc_type": exc_type,
            "robot_class": robot_class,
            "run_id": run_id,
        },
    )


def publish_loop(
    registry: Registry,
    labels: Mapping[str, str],
    report: Mapping[str, float | int],
) -> None:
    """Copy a :func:`ohho_runtime.timing.loop_report` into loop metrics.

    Cycle time is three gauges distinguished by ``quantile`` (``0.5``,
    ``0.95``, ``0.99``). Jitter, deadline misses, sample count, and the
    latest lateness come from the same report.
    """
    registry.set(
        "ohho_runtime_loop_jitter_seconds",
        float(report["jitter_s"]),
        labels,
    )
    registry.set(
        "ohho_runtime_loop_deadline_misses_total",
        float(report["deadline_misses"]),
        labels,
    )
    registry.set("ohho_runtime_loop_samples", float(report["samples"]), labels)
    registry.set(
        "ohho_runtime_loop_lateness_seconds",
        float(report["lateness_s"]),
        labels,
    )
    for quantile, key in _LOOP_QUANTILES:
        labeled = dict(labels)
        labeled["quantile"] = quantile
        registry.set("ohho_runtime_loop_cycle_seconds", float(report[key]), labeled)


class MetricsEndpoint:
    """HTTP ``/metrics`` endpoint. Disabled endpoints do not bind a socket."""

    def __init__(
        self,
        registry: Registry,
        *,
        enabled: bool | None = None,
        host: str | None = None,
        port: int | None = None,
        collect: Callable[[], None] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        bind = resolve_metrics_bind(
            enabled=enabled,
            host=host,
            port=port,
            environ=environ,
        )
        self._registry = registry
        self._collect = collect
        self.enabled = bind.enabled
        self.host = bind.host
        self.port = bind.port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def listening(self) -> bool:
        """True after a successful :meth:`start` until :meth:`stop`."""
        return self._server is not None

    @property
    def address(self) -> tuple[str, int] | None:
        """Bound ``(host, port)``, or ``None`` when the endpoint is not listening."""
        server = self._server
        if server is None:
            return None
        bound = server.server_address
        return str(bound[0]), int(bound[1])

    def handle_path(self, path: str) -> tuple[int, str, bytes]:
        """Serve one path. ``/metrics`` runs ``collect`` then renders text.

        A collector failure is logged, counted, and returned as HTTP 500.
        It is not discarded.
        """
        route = path.split("?", 1)[0]
        if route != "/metrics":
            return 404, "text/plain; charset=utf-8", b"not found\n"
        try:
            if self._collect is not None:
                self._collect()
            body = self._registry.render().encode("utf-8")
        except Exception as exc:
            from ohho_runtime.log import note_exception

            note_exception(
                exc,
                service="metrics",
                event="scrape_failed",
                metrics=self._registry,
            )
            return 500, "text/plain; charset=utf-8", b"scrape failed\n"
        return 200, PROMETHEUS_CONTENT_TYPE, body

    def start(self) -> None:
        """Bind and serve when enabled. A disabled endpoint does not bind."""
        if not self.enabled or self._server is not None:
            return
        handler = _handler_class(self)
        server = _MetricsServer((self.host, self.port), handler)
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05),
            name="ohho-runtime-metrics",
            daemon=True,
        )
        self._server = server
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the server. Safe to call when the endpoint never started."""
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)


class _MetricsServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_class(endpoint: MetricsEndpoint) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = endpoint.handle_path(self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            del fmt, args

    return Handler


__all__ = [
    "CATALOG",
    "DEFAULT_METRICS_HOST",
    "DEFAULT_METRICS_PORT",
    "METRIC_NAMES",
    "METRICS_ENABLED_ENV",
    "METRICS_HOST_ENV",
    "METRICS_PORT_ENV",
    "PROMETHEUS_CONTENT_TYPE",
    "MetricsBind",
    "MetricsEndpoint",
    "Registry",
    "get_registry",
    "publish_loop",
    "record_exception",
    "reset_registry",
    "resolve_metrics_bind",
]
