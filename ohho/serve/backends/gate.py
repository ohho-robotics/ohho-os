"""Supervisor for one serving process tree.

Started as ``python -m ohho.serve.backends.gate``. It reads one JSON command
from stdin, spawns those argv lists, and if they exit it kills any child
they left behind. The parent stops the whole tree by killing this process
group (POSIX) or its Windows job object.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

from ohho.serve.backends.tree import descendants, kill_pid, pid_alive


def main() -> int:
    """Run until every spawned command exits, then stop their descendants."""
    try:
        return _run()
    except Exception as exc:
        print(f"error: serving gate failed: {exc}", file=sys.stderr)
        return 1


def _run() -> int:
    line = sys.stdin.buffer.readline().decode("utf-8")
    if not line.strip():
        print("error: serving gate received an empty command", file=sys.stderr)
        return 2
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"error: serving gate received invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("error: serving gate command must be a JSON object", file=sys.stderr)
        return 2
    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        print("error: serving gate requires at least one command", file=sys.stderr)
        return 2
    procs = _spawn(commands, payload)
    if isinstance(procs, int):
        return procs
    remembered: set[int] = set()
    while True:
        alive = False
        for proc in procs:
            remembered.add(proc.pid)
            try:
                remembered |= descendants(proc.pid)
            except Exception:
                pass
            if proc.poll() is None:
                alive = True
        if not alive:
            code = 0
            for proc in procs:
                if proc.returncode:
                    code = int(proc.returncode)
                    break
            _stop(remembered)
            return code
        time.sleep(0.02)


def _spawn(
    commands: list[Any], payload: dict[str, Any]
) -> list[subprocess.Popen[bytes]] | int:
    cwd = payload.get("cwd") or None
    overrides = payload.get("env")
    env = None
    if isinstance(overrides, dict):
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in overrides.items()})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    procs: list[subprocess.Popen[bytes]] = []
    for argv in commands:
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(part, str) and part for part in argv)
        ):
            print("error: serving command must be a list of strings", file=sys.stderr)
            _stop({proc.pid for proc in procs})
            return 2
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                creationflags=flags,
            )
        except FileNotFoundError:
            print(f"error: executable not found: {argv[0]}", file=sys.stderr)
            _stop({proc.pid for proc in procs})
            return 127
        except OSError as exc:
            print(f"error: failed to start serving command: {exc}", file=sys.stderr)
            _stop({proc.pid for proc in procs})
            return 1
        procs.append(proc)
    return procs


def _stop(pids: set[int]) -> None:
    """Stop ``pids`` and processes they created. Each pid is signalled once.

    A second ``taskkill`` for a pid that has already exited can block on
    Windows when the id is reused or a console host is still closing.
    """
    me = os.getpid()
    sent: set[int] = {me, 0}
    pending = {pid for pid in pids if pid > 0 and pid != me}
    for _ in range(3):
        extra: set[int] = set()
        for pid in list(pending):
            if pid in sent or not pid_alive(pid):
                sent.add(pid)
                continue
            try:
                extra |= descendants(pid)
            except Exception:
                continue
        pending |= {pid for pid in extra if pid > 0 and pid != me}
        for pid in list(pending):
            if pid in sent or not pid_alive(pid):
                sent.add(pid)
                continue
            sent.add(pid)
            kill_pid(pid, tree=False)
        if pending <= sent:
            return
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
