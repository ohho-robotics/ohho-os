"""Subprocess service: a console script with a Unix-socket heartbeat.

SIGTERM is the graceful stop. If the child is still alive after ``grace_s``,
it is killed and the kill is recorded. Heartbeats are ``ready`` / ``beat``
lines on an ``AF_UNIX`` stream socket. Platforms that cannot bind one raise
:class:`UnixSocketUnavailable` instead of pretending the socket exists.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Protocol

from ohho_runtime.heartbeat import (
    BEAT_LINE,
    INTERVAL_ENV,
    READY_LINE,
    SOCK_ENV,
    split_lines,
    unix_sockets_available,
)
from ohho_runtime.log import note_exception
from ohho_runtime.service.base import ServiceBase
from ohho_runtime.service.lifecycle import (
    LEGAL_TRANSITIONS,
    LifecycleState,
    ShutdownRecord,
    UnixSocketUnavailable,
)

_ENV_ALLOW = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
)


def package_src() -> str:
    """``runtime/src``, so ``python -m ohho_runtime.worker`` can import."""
    return str(Path(__file__).resolve().parents[2])


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the worker. Only runtime keys may be added.

    The parent environment is not copied wholesale. The arming switch is not
    a runtime key, so it is not forwarded.
    """
    env = {key: os.environ[key] for key in _ENV_ALLOW if key in os.environ}
    src = package_src()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not prior else src + os.pathsep + prior
    for key, value in (extra or {}).items():
        if not key.startswith("OHHO_RUNTIME_"):
            raise ValueError("subprocess extra env must be an OHHO_RUNTIME_ key")
        env[key] = value
    return env


class ChildProcess(Protocol):
    """The process operations ``stop_child`` needs. ``Popen`` satisfies this."""

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, sig: int) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _request_stop(proc: ChildProcess) -> None:
    """Ask the child to exit. POSIX delivers SIGTERM; other platforms terminate."""
    if os.name == "posix":
        proc.send_signal(signal.SIGTERM)
        return
    proc.terminate()


def stop_child(proc: ChildProcess, *, grace_s: float) -> ShutdownRecord:
    """SIGTERM, then SIGKILL if the child ignores it past ``grace_s``.

    The kill is the returned record (``action="sigkill"``, ``forced=True``).
    A child that exits in time is ``action="sigterm"`` and not forced.
    """
    if (
        isinstance(grace_s, bool)
        or not isinstance(grace_s, (int, float))
        or grace_s <= 0
    ):
        raise ValueError("grace_s must be > 0")
    if proc.poll() is not None:
        return ShutdownRecord(
            action="exited",
            forced=False,
            grace_s=float(grace_s),
            detail="child had already exited",
        )
    _request_stop(proc)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired as exc:
        note_exception(
            exc,
            service="subprocess",
            event="stop_grace_exceeded",
            level="WARNING",
        )
        proc.kill()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired as hung:
            note_exception(
                hung,
                service="subprocess",
                event="stop_kill_timeout",
            )
        return ShutdownRecord(
            action="sigkill",
            forced=True,
            grace_s=float(grace_s),
            signal=int(getattr(signal, "SIGKILL", 9)),
            detail="child ignored SIGTERM; SIGKILL after grace",
        )
    return ShutdownRecord(
        action="sigterm",
        forced=False,
        grace_s=float(grace_s),
        signal=int(getattr(signal, "SIGTERM", 15)),
        detail="child exited within grace",
    )


class HeartbeatServer:
    """Accept one worker connection and dispatch ``ready`` / ``beat`` lines."""

    def __init__(
        self,
        directory: Path,
        *,
        on_ready: Callable[[], None],
        on_beat: Callable[[], None],
    ) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise UnixSocketUnavailable("AF_UNIX is not available")
        self.path = str(directory / "hb.sock")
        self._on_ready = on_ready
        self._on_beat = on_beat
        self.beats = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        if os.path.exists(self.path):
            os.unlink(self.path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(1)
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(
            target=self._run,
            name="ohho-runtime-heartbeat",
            daemon=True,
        )
        self._thread.start()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _run(self) -> None:
        conn: socket.socket | None = None
        pending = b""
        while not self._stop.is_set():
            if conn is None:
                try:
                    conn, _addr = self._sock.accept()
                except socket.timeout as exc:
                    note_exception(
                        exc,
                        service="subprocess",
                        event="heartbeat_accept_wait",
                        level="DEBUG",
                    )
                    continue
                except OSError as exc:
                    note_exception(
                        exc,
                        service="subprocess",
                        event="heartbeat_accept_failed",
                        level="WARNING",
                    )
                    break
                conn.settimeout(0.2)
            try:
                chunk = conn.recv(256)
            except socket.timeout as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="heartbeat_recv_wait",
                    level="DEBUG",
                )
                continue
            except OSError as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="heartbeat_recv_failed",
                    level="WARNING",
                )
                break
            if not chunk:
                conn.close()
                conn = None
                pending = b""
                continue
            pending, lines = split_lines(pending, chunk)
            for line in lines:
                self._dispatch(line)
        if conn is not None:
            try:
                conn.close()
            except OSError as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="heartbeat_conn_close_failed",
                    level="DEBUG",
                )

    def _dispatch(self, line: bytes) -> None:
        if line == READY_LINE:
            self._ready.set()
            self._on_ready()
        elif line == BEAT_LINE:
            self.beats += 1
            self._on_beat()

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError as exc:
            note_exception(
                exc,
                service="subprocess",
                event="heartbeat_socket_close_failed",
                level="DEBUG",
            )
        self._thread.join(timeout=1.0)
        try:
            os.unlink(self.path)
        except OSError as exc:
            note_exception(
                exc,
                service="subprocess",
                event="heartbeat_socket_unlink_failed",
                level="DEBUG",
            )


class SubprocessService(ServiceBase):
    """Supervise ``ohho-runtime-service`` (``python -m ohho_runtime.worker``).

    ``argv`` is the console script, usually
    ``[sys.executable, "-m", "ohho_runtime.worker"]``. Heartbeats on the
    socket call :meth:`heartbeat`. A ``ready`` line calls :meth:`mark_ready`.
    ``stop`` sends SIGTERM and records SIGKILL when the grace period expires.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        extra_env: Mapping[str, str] | None = None,
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
        if not argv:
            raise ValueError("argv must not be empty")
        self._argv = tuple(argv)
        self._extra = dict(extra_env or {})
        self._dir: Path | None = None
        self._server: HeartbeatServer | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_path: Path | None = None
        self._stderr_fh: BinaryIO | None = None
        self._stderr_cache = ""

    @property
    def beats(self) -> int:
        server = self._server
        return 0 if server is None else server.beats

    @property
    def exit_code(self) -> int | None:
        """Child exit status, or ``None`` while the child is running.

        Reading this polls the child. It does not signal the child.
        """
        proc = self._proc
        if proc is None:
            return None
        code = proc.poll()
        if isinstance(code, bool) or not isinstance(code, int):
            return None
        return code

    def stderr_text(self) -> str:
        self._capture_stderr()
        return self._stderr_cache

    def _capture_stderr(self) -> None:
        handle = self._stderr_fh
        if handle is not None:
            try:
                handle.flush()
            except OSError as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="stderr_flush_failed",
                    level="DEBUG",
                )
        path = self._stderr_path
        if path is None or not path.is_file():
            return
        try:
            self._stderr_cache = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            note_exception(
                exc,
                service="subprocess",
                event="stderr_read_failed",
                level="WARNING",
            )
            return

    def on_start(self) -> None:
        if not unix_sockets_available():
            raise UnixSocketUnavailable(
                "Unix domain sockets are not available; "
                "subprocess heartbeat cannot be hosted"
            )
        try:
            self._dir = Path(tempfile.mkdtemp(prefix="ohho-rt-"))
            self._server = HeartbeatServer(
                self._dir,
                on_ready=self.mark_ready,
                on_beat=self.heartbeat,
            )
            env = child_env(self._extra)
            env[SOCK_ENV] = self._server.path
            env.setdefault(INTERVAL_ENV, "0.05")
            self._stderr_path = self._dir / "stderr.txt"
            self._stderr_fh = self._stderr_path.open("wb", buffering=0)
            if os.name == "posix":
                self._proc = subprocess.Popen(
                    list(self._argv),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=self._stderr_fh,
                    start_new_session=True,
                )
            else:
                self._proc = subprocess.Popen(
                    list(self._argv),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=self._stderr_fh,
                )
        except Exception:
            self._kill_quiet()
            raise

    def on_stop(self) -> None:
        # Stop the child before closing the socket. Closing first would make
        # a child that ignores SIGTERM exit on a socket error, and the kill
        # would not be recorded.
        proc = self._proc
        if proc is not None:
            self.note_shutdown(stop_child(proc, grace_s=self.grace_s))
            if proc.poll() is not None:
                self._proc = None
        self._kill_quiet()

    def close(self) -> None:
        """Stop when the lifecycle allows it, and always reap the child."""
        if (self.state, LifecycleState.STOPPING) in LEGAL_TRANSITIONS:
            try:
                self.stop()
                return
            except Exception as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="close_stop_failed",
                )
        self._kill_quiet()

    def _kill_quiet(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="child_kill_failed",
                )
        server = self._server
        self._server = None
        if server is not None:
            server.close()
        handle = self._stderr_fh
        self._stderr_fh = None
        if handle is not None:
            try:
                handle.close()
            except OSError as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="stderr_close_failed",
                    level="DEBUG",
                )
        self._capture_stderr()
        directory = self._dir
        self._dir = None
        if directory is not None:
            try:
                for child in directory.iterdir():
                    try:
                        child.unlink()
                    except OSError as exc:
                        note_exception(
                            exc,
                            service="subprocess",
                            event="runtime_file_unlink_failed",
                            level="DEBUG",
                        )
                directory.rmdir()
            except OSError as exc:
                note_exception(
                    exc,
                    service="subprocess",
                    event="runtime_dir_cleanup_failed",
                    level="DEBUG",
                )
