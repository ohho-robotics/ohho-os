"""Heartbeat socket constants shared by the parent and the worker.

No third-party imports and no ``ohho`` import: the worker process loads this
module before it opens a socket.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

from ohho_runtime.log import note_exception

SOCK_ENV = "OHHO_RUNTIME_HEARTBEAT_SOCK"
INTERVAL_ENV = "OHHO_RUNTIME_HEARTBEAT_INTERVAL"
IGNORE_TERM_ENV = "OHHO_RUNTIME_IGNORE_SIGTERM"
NEVER_READY_ENV = "OHHO_RUNTIME_NEVER_READY"
STARTED_ENV = "OHHO_RUNTIME_STARTED_FILE"

READY_LINE = b"ready"
BEAT_LINE = b"beat"


def split_lines(pending: bytes, chunk: bytes) -> tuple[bytes, tuple[bytes, ...]]:
    """Append ``chunk`` and return ``(remainder, complete stripped lines)``."""
    pending += chunk
    lines: list[bytes] = []
    while b"\n" in pending:
        line, pending = pending.split(b"\n", 1)
        lines.append(line.strip())
    return pending, tuple(lines)


def unix_sockets_available() -> bool:
    """True when this process can bind an ``AF_UNIX`` stream socket."""
    if not hasattr(socket, "AF_UNIX"):
        return False
    directory = tempfile.mkdtemp(prefix="ohho-rt-")
    path = os.path.join(directory, "s")
    ok = False
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.bind(path)
            ok = True
        finally:
            probe.close()
    except OSError as exc:
        note_exception(
            exc,
            service="heartbeat",
            event="socket_probe_failed",
            level="DEBUG",
        )
        ok = False
    try:
        os.unlink(path)
    except OSError as exc:
        note_exception(
            exc,
            service="heartbeat",
            event="socket_probe_unlink_failed",
            level="DEBUG",
        )
    try:
        Path(directory).rmdir()
    except OSError as exc:
        note_exception(
            exc,
            service="heartbeat",
            event="socket_probe_rmdir_failed",
            level="DEBUG",
        )
    return ok
