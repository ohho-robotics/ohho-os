"""Console-script body for a supervised subprocess.

Entry point: ``ohho-runtime-service`` and ``python -m ohho_runtime.worker``.

The process installs a SIGTERM handler, optionally writes a started file, then
sends ``ready`` and ``beat`` lines on the Unix socket from
``OHHO_RUNTIME_HEARTBEAT_SOCK``. SIGTERM sets the stop flag and the process
exits. ``OHHO_RUNTIME_IGNORE_SIGTERM=1`` ignores SIGTERM so a parent can test
grace-period escalation. ``OHHO_RUNTIME_MLOCKALL=1`` locks this process first;
a refusal is logged and the process still starts. This module does not import
``ohho`` and does not command a robot.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from pathlib import Path

from ohho_runtime.heartbeat import (
    BEAT_LINE,
    INTERVAL_ENV,
    IGNORE_TERM_ENV,
    NEVER_READY_ENV,
    READY_LINE,
    SOCK_ENV,
    STARTED_ENV,
)
from ohho_runtime.os_sched import lock_memory_if_requested


def _connect(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 5.0
    while True:
        try:
            sock.connect(path)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                sock.close()
                raise
            time.sleep(0.02)


def _send(sock: socket.socket, line: bytes) -> None:
    sock.sendall(line + b"\n")


def main() -> None:
    """Run until SIGTERM, unless the ignore flag is set."""
    lock_memory_if_requested()
    stop = threading.Event()

    def _on_term(_signum: int, _frame: object = None) -> None:
        stop.set()

    if os.environ.get(IGNORE_TERM_ENV) == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, _on_term)

    started = os.environ.get(STARTED_ENV, "")
    if started:
        Path(started).write_text("started\n", encoding="utf-8")

    sock_path = os.environ.get(SOCK_ENV, "")
    interval = float(os.environ.get(INTERVAL_ENV, "0.05"))
    if interval <= 0:
        interval = 0.05
    never_ready = os.environ.get(NEVER_READY_ENV) == "1"

    sock: socket.socket | None = None
    if sock_path:
        sock = _connect(sock_path)
        if not never_ready:
            _send(sock, READY_LINE)

    try:
        while not stop.is_set():
            if sock is not None:
                try:
                    _send(sock, BEAT_LINE)
                except OSError:
                    break
            if stop.wait(interval):
                break
    finally:
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    main()
