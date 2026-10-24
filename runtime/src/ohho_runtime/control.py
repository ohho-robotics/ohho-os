"""Loopback control channel for a running supervisor.

``up`` binds ``127.0.0.1`` (or ``::1`` when asked) and writes a state file.
The other commands send one JSON object per connection. Nothing here
listens on a non-loopback address, and nothing here signals a process by
pid. Shutdown is a request the supervisor handles itself.

Dev mode has no per-service thread. The foreground loop refreshes liveness
for services that are already ``ready`` so a one-shot target is not failed
for lack of a loop of its own. A service that failed to start is not
refreshed and stays failed.
"""

from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ohho_runtime.log import normalize_level, note_exception
from ohho_runtime.manifest import Criticality, RuntimeManifest
from ohho_runtime.metrics import Registry
from ohho_runtime.service import HealthStatus, LifecycleState
from ohho_runtime.supervisor import Supervisor, SupervisorError, SupervisorState

STATE_VERSION = 1
LOOPBACK_IPV4 = "127.0.0.1"
LOOPBACK_IPV6 = "::1"
_MAX_MESSAGE = 1_048_576
_MAX_LOG_LINES = 2000
_READ_TIMEOUT_S = 0.2

_LEVEL_RANK = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_TEXT_LEVEL = re.compile(r"\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s")
_TEXT_SERVICE = re.compile(r"\bservice=(\S+)")


class ControlError(Exception):
    """The control channel could not complete a request."""


@dataclass(frozen=True)
class ControlState:
    """Where a running supervisor is listening. ``pid`` is informational."""

    host: str
    port: int
    pid: int
    run_id: str
    robot_class: str

    def to_json(self) -> dict[str, object]:
        """JSON object written to the state file."""
        return {
            "version": STATE_VERSION,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
            "run_id": self.run_id,
            "robot_class": self.robot_class,
        }


class LogBuffer(io.TextIOBase):
    """Text stream that keeps the most recent structured log lines."""

    def __init__(self, *, limit: int = _MAX_LOG_LINES) -> None:
        super().__init__()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("log buffer limit must be an int >= 1")
        self._limit = limit
        self._lock = threading.Lock()
        self._partial = ""
        self._seq = 0
        self._records: list[tuple[int, str]] = []

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            raise TypeError("log buffer expects text")
        with self._lock:
            self._partial += data
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                if not line:
                    continue
                self._seq += 1
                self._records.append((self._seq, line))
            extra = len(self._records) - self._limit
            if extra > 0:
                del self._records[:extra]
        return len(data)

    def flush(self) -> None:
        return None

    def snapshot(self) -> list[tuple[int, str]]:
        """``(sequence, line)`` pairs in order. Sequence numbers only grow."""
        with self._lock:
            return list(self._records)


def default_state_path() -> Path:
    """State file used when ``--state`` is omitted."""
    return Path.home() / ".ohho" / "runtime" / "control.json"


def require_loopback_host(host: str) -> str:
    """Return ``127.0.0.1`` or ``::1``.

    ``localhost`` is the IPv4 loopback. Any other name or address is
    refused before a socket is created.
    """
    if not isinstance(host, str):
        raise ControlError("control host must be loopback")
    token = host.strip().lower()
    if token == "localhost":
        return LOOPBACK_IPV4
    try:
        address = ipaddress.ip_address(token)
    except ValueError as exc:
        raise ControlError("control host must be loopback") from exc
    if address.version == 4 and address == ipaddress.ip_address(LOOPBACK_IPV4):
        return LOOPBACK_IPV4
    if address.version == 6 and address == ipaddress.ip_address(LOOPBACK_IPV6):
        return LOOPBACK_IPV6
    raise ControlError("control host must be loopback")


def read_state(path: Path) -> ControlState:
    """Load a state file. A non-loopback host is refused."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlError(f"state file {path} is not JSON") from exc
    except OSError as exc:
        raise ControlError(f"state file {path} could not be read") from exc
    if not isinstance(raw, dict):
        raise ControlError("state file must be a JSON object")
    if raw.get("version") != STATE_VERSION:
        raise ControlError("state file version is not supported")
    host = raw.get("host")
    if not isinstance(host, str):
        raise ControlError("state file host is missing")
    host = require_loopback_host(host)
    port = raw.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ControlError("state file port is invalid")
    pid = raw.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
        raise ControlError("state file pid is invalid")
    run_id = raw.get("run_id")
    robot_class = raw.get("robot_class")
    if not isinstance(run_id, str) or not run_id:
        raise ControlError("state file run_id is missing")
    if not isinstance(robot_class, str):
        raise ControlError("state file robot_class is missing")
    return ControlState(
        host=host,
        port=port,
        pid=pid,
        run_id=run_id,
        robot_class=robot_class,
    )


def write_state(path: Path, state: ControlState) -> None:
    """Atomically replace ``path`` with ``state``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_json(), sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def transact(
    host: str,
    port: int,
    payload: Mapping[str, object],
    *,
    timeout: float = 5.0,
) -> dict[str, object]:
    """Send one JSON request to a loopback supervisor and return the response."""
    address = require_loopback_host(host)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ControlError("control port is invalid")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ControlError("control timeout must be a number > 0")
    try:
        sock = socket.create_connection((address, port), timeout=float(timeout))
    except OSError as exc:
        raise ControlError(f"cannot reach supervisor at {address}:{port}") from exc
    try:
        sock.settimeout(float(timeout))
        raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        sock.sendall(raw.encode("utf-8") + b"\n")
        text = _read_line(sock)
    finally:
        sock.close()
    try:
        response = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ControlError("control response is not JSON") from exc
    if not isinstance(response, dict):
        raise ControlError("control response must be a JSON object")
    return response


def ping_state(path: Path, *, timeout: float = 0.5) -> bool:
    """True when ``path`` names a supervisor that answers ``ping``."""
    if not path.is_file():
        return False
    try:
        state = read_state(path)
        response = transact(state.host, state.port, {"op": "ping"}, timeout=timeout)
    except (
        OSError,
        ControlError,
        json.JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
    ):
        return False
    return response.get("ok") is True


def parse_log_line(line: str) -> dict[str, object] | None:
    """Return the fields needed to filter one structured line.

    JSON lines keep every field. Text lines yield ``level`` and ``service``.
    A line that is neither is ignored.
    """
    text = line.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        return value
    level = _TEXT_LEVEL.search(text)
    service = _TEXT_SERVICE.search(text)
    if level is None or service is None:
        return None
    return {"level": level.group(1), "service": service.group(1)}


def matching_log_lines(
    records: list[tuple[int, str]],
    *,
    service: str,
    level: str,
    tail: int,
    after: int,
) -> tuple[list[str], int]:
    """Filter ``records`` by service and minimum level.

    ``after`` is a sequence cursor. The first request (``after`` 0) returns
    at most ``tail`` matching lines. A later request returns the matching
    lines newer than that cursor. The returned cursor is the newest sequence
    in ``records``, whether or not that line matched.
    """
    cursor = records[-1][0] if records else after
    minimum = _LEVEL_RANK[level] if level else 0
    matched: list[str] = []
    for seq, line in records:
        if seq <= after:
            continue
        parsed = parse_log_line(line)
        if parsed is None:
            continue
        if service and parsed.get("service") != service:
            continue
        try:
            canonical = normalize_level(str(parsed.get("level", "")))
        except ValueError:
            continue
        if _LEVEL_RANK[canonical] < minimum:
            continue
        matched.append(line)
    if after <= 0:
        matched = matched[-tail:] if tail else []
    return matched, cursor


def supervisor_report(
    supervisor: Supervisor,
    manifest: RuntimeManifest,
) -> dict[str, object]:
    """Status and health for one supervisor.

    ``healthy`` is false when do-not-operate is latched or any critical
    service is not ``ok``. Degradable and optional services are reported
    and do not, on their own, clear ``healthy``.
    """
    specs = {spec.role: spec for spec in manifest.services}
    rows: list[dict[str, object]] = []
    unhealthy: list[str] = []
    for role in supervisor.graph.start_order():
        spec = specs[role]
        service = supervisor.service(role)
        report = service.health()
        age = service.heartbeat_age_s
        row: dict[str, object] = {
            "service": role,
            "state": service.state.value,
            "restarts": supervisor.restart_count(role),
            "heartbeat_age_s": None if age is None else float(age),
            "criticality": spec.criticality.value,
            "health": report.status.value,
            "reason": report.reason,
        }
        rows.append(row)
        if (
            spec.criticality is Criticality.CRITICAL
            and report.status is not HealthStatus.OK
        ):
            unhealthy.append(role)
    healthy = not unhealthy and not supervisor.do_not_operate
    return {
        "state": supervisor.state.value,
        "run_id": supervisor.run_id,
        "robot_class": supervisor.robot_class.value,
        "do_not_operate": supervisor.do_not_operate,
        "allows_motion": supervisor.allows_motion(),
        "disarmed": supervisor.disarmed,
        "services": rows,
        "healthy": healthy,
        "unhealthy": unhealthy,
    }


class ControlSession:
    """Foreground host: poll the supervisor and serve one control socket."""

    def __init__(
        self,
        supervisor: Supervisor,
        manifest: RuntimeManifest,
        logs: LogBuffer,
        *,
        state_path: Path,
        host: str,
        port: int,
        poll_interval: float,
        metrics: Registry,
    ) -> None:
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ControlError("control port must be from 0 to 65535")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or poll_interval <= 0
        ):
            raise ControlError("poll interval must be a number > 0")
        self._supervisor = supervisor
        self._manifest = manifest
        self._logs = logs
        self._state_path = state_path
        self._host = require_loopback_host(host)
        self._port = port
        self._poll_interval = float(poll_interval)
        self._metrics = metrics
        self._sock: socket.socket | None = None
        self._stop = False
        self._closed = False

    def serve(self, *, ready: Callable[[str, int], None] | None = None) -> None:
        """Bind, start the supervisor, then serve until ``down`` or interrupt."""
        self._sock = _bind(self._host, self._port)
        try:
            self._sock.settimeout(self._poll_interval)
            self._supervisor.start()
            bound = self._sock.getsockname()
            port = int(bound[1])
            write_state(
                self._state_path,
                ControlState(
                    host=self._host,
                    port=port,
                    pid=os.getpid(),
                    run_id=self._supervisor.run_id,
                    robot_class=self._supervisor.robot_class.value,
                ),
            )
            if ready is not None:
                ready(self._host, port)
            while not self._stop:
                self._tick()
                self._accept_one()
        finally:
            self._close()

    def _tick(self) -> None:
        for role in self._supervisor.graph.start_order():
            service = self._supervisor.service(role)
            if service.state is LifecycleState.READY:
                service.heartbeat()
        self._supervisor.poll()

    def _accept_one(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            conn, addr = sock.accept()
        except TimeoutError:
            return
        except InterruptedError:
            return
        with conn:
            conn.settimeout(_READ_TIMEOUT_S)
            peer = addr[0]
            if not _is_loopback_peer(peer):
                self._reply(
                    conn,
                    {"ok": False, "error": "control channel is loopback only"},
                )
                return
            self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        try:
            text = _read_line(conn)
        except ControlError as exc:
            self._reply(conn, {"ok": False, "error": str(exc)})
            return
        except (ConnectionError, TimeoutError):
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            message = f"control message is not JSON: {exc}"
            self._reply(conn, {"ok": False, "error": message})
            return
        if not isinstance(payload, dict):
            self._reply(
                conn,
                {"ok": False, "error": "control message must be a JSON object"},
            )
            return
        try:
            response = self._dispatch(payload)
        except (ControlError, SupervisorError, ValueError) as exc:
            response = {"ok": False, "error": str(exc)}
        except Exception as exc:
            note_exception(
                exc,
                service="supervisor",
                event="control_failed",
                logger=self._supervisor.logger,
                metrics=self._metrics,
            )
            response = {"ok": False, "error": str(exc)}
        self._reply(conn, response)

    def _reply(self, conn: socket.socket, payload: Mapping[str, object]) -> None:
        try:
            _send(conn, payload)
        except (ConnectionError, TimeoutError):
            return

    def _dispatch(self, payload: Mapping[str, object]) -> dict[str, object]:
        op = payload.get("op")
        if not isinstance(op, str) or not op:
            raise ControlError("control message needs an op")
        if op == "ping":
            return {"ok": True, "result": {"run_id": self._supervisor.run_id}}
        if op in {"status", "health"}:
            return {
                "ok": True,
                "result": supervisor_report(self._supervisor, self._manifest),
            }
        if op == "logs":
            return {"ok": True, "result": self._logs_result(payload)}
        if op == "restart":
            return {"ok": True, "result": self._restart(payload)}
        if op == "down":
            self._stop = True
            return {"ok": True, "result": {"state": "stopped"}}
        raise ControlError(f"unknown control op {op!r}")

    def _restart(self, payload: Mapping[str, object]) -> dict[str, object]:
        service = payload.get("service")
        if (
            not isinstance(service, str)
            or not service
            or any(ch.isspace() for ch in service)
        ):
            raise ControlError("restart needs a service")
        self._supervisor.restart(service)
        report = supervisor_report(self._supervisor, self._manifest)
        rows = report["services"]
        if not isinstance(rows, list):
            raise ControlError("status report has no services")
        for row in rows:
            if isinstance(row, dict) and row.get("service") == service:
                return {
                    "service": service,
                    "restarts": row.get("restarts", 0),
                    "state": row.get("state", ""),
                }
        raise ControlError(f"unknown service role {service!r}")

    def _logs_result(self, payload: Mapping[str, object]) -> dict[str, object]:
        service = payload.get("service", "")
        level = payload.get("level", "")
        if not isinstance(service, str) or not isinstance(level, str):
            raise ControlError("logs filters must be strings")
        tail = payload.get("tail", 200)
        after = payload.get("after", 0)
        if isinstance(tail, bool) or not isinstance(tail, int) or tail < 0:
            raise ControlError("tail must be an int >= 0")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ControlError("after must be an int >= 0")
        canonical = ""
        if level:
            try:
                canonical = normalize_level(level)
            except ValueError as exc:
                raise ControlError(str(exc)) from None
        lines, cursor = matching_log_lines(
            self._logs.snapshot(),
            service=service,
            level=canonical,
            tail=tail,
            after=after,
        )
        return {"lines": lines, "cursor": cursor}

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._supervisor.state is not SupervisorState.STOPPED:
                    self._supervisor.shutdown()
            finally:
                sock = self._sock
                self._sock = None
                if sock is not None:
                    sock.close()
        finally:
            self._state_path.unlink(missing_ok=True)


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if host == LOOPBACK_IPV6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if family is socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(8)
    except OSError:
        sock.close()
        raise
    return sock


def _is_loopback_peer(host: str) -> bool:
    try:
        return bool(ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


def _read_line(conn: socket.socket) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        try:
            chunk = conn.recv(4096)
        except TimeoutError as exc:
            raise ControlError("control response timed out") from exc
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_MESSAGE:
            raise ControlError("control message is too large")
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    raw = b"".join(chunks)
    if not raw:
        raise ControlError("empty control message")
    line = raw.split(b"\n", 1)[0]
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ControlError("control message is not utf-8") from exc


def _send(conn: socket.socket, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    conn.sendall(text.encode("utf-8") + b"\n")


__all__ = [
    "LOOPBACK_IPV4",
    "LOOPBACK_IPV6",
    "STATE_VERSION",
    "ControlError",
    "ControlSession",
    "ControlState",
    "LogBuffer",
    "default_state_path",
    "matching_log_lines",
    "parse_log_line",
    "ping_state",
    "read_state",
    "require_loopback_host",
    "supervisor_report",
    "transact",
    "write_state",
]
