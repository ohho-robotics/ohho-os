"""Tiny local stand-in for an OpenAI-compatible streaming server.

Tests point a serving backend at this file. It is not an engine.

``OHHO_FAKE_MODE`` selects the behavior:

- ``serve`` — ``GET /health`` and streaming ``POST /v1/chat/completions``
- ``exit`` — start a sleeper child, then exit 1
- ``hang`` — start a sleeper child and stay up without serving HTTP

``OHHO_FAKE_CHILD_PID`` is a file the sleeper writes its pid into.
``OHHO_FAKE_TOKENS`` is a comma-separated list of stream chunks.
``--version`` prints ``OHHO_FAKE_VERSION`` (default ``9.9.9``) and exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> int:
    """Run the mode selected by the environment."""
    if "--version" in sys.argv:
        print(os.environ.get("OHHO_FAKE_VERSION", "9.9.9"))
        return 0
    mode = os.environ.get("OHHO_FAKE_MODE", "serve")
    if mode == "exit":
        return _exit_after_child()
    if mode == "hang":
        return _hang()
    if mode == "serve":
        return _serve()
    print(f"error: unknown OHHO_FAKE_MODE {mode}", file=sys.stderr)
    return 2


def _flag(name: str, default: str | None = None) -> str | None:
    argv = sys.argv[1:]
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def _spawn_sleeper(pid_path: str) -> subprocess.Popen[bytes]:
    code = (
        "import os, sys, time\n"
        "path, seconds = sys.argv[1], float(sys.argv[2])\n"
        "with open(path, 'w', encoding='utf-8') as handle:\n"
        "    handle.write(str(os.getpid()))\n"
        "time.sleep(seconds)\n"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-c", code, pid_path, "120"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


def _wait_for_pid(path: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.02)
    print("error: child pid file was not written", file=sys.stderr)
    raise SystemExit(1)


def _child_path() -> str:
    path = os.environ.get("OHHO_FAKE_CHILD_PID", "")
    if not path:
        print("error: OHHO_FAKE_CHILD_PID is required", file=sys.stderr)
        raise SystemExit(2)
    return path


def _exit_after_child() -> int:
    path = _child_path()
    _spawn_sleeper(path)
    _wait_for_pid(path)
    time.sleep(0.35)
    print("fake-exit", file=sys.stderr, flush=True)
    # Skip interpreter shutdown. A live child plus inherited stdio can
    # keep python.exe running for seconds, and the supervisor would not
    # observe the exit in time to kill that child.
    os._exit(1)


def _hang() -> int:
    path = _child_path()
    _spawn_sleeper(path)
    _wait_for_pid(path)
    time.sleep(3600)
    return 0


def _serve() -> int:
    host = _flag("--host", "127.0.0.1") or "127.0.0.1"
    port = int(_flag("--port", "0") or "0")
    tokens = [
        part for part in os.environ.get("OHHO_FAKE_TOKENS", "Hel,lo").split(",") if part
    ]
    delay = float(os.environ.get("OHHO_FAKE_DELAY_S", "0") or "0")
    child = os.environ.get("OHHO_FAKE_CHILD_PID")
    if child:
        _spawn_sleeper(child)
    handler = _handler(tokens, delay)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        print(f"error: could not bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    server.serve_forever()
    return 0


def _handler(tokens: list[str], delay: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in {"/health", "/v1/health"}:
                self.send_error(404)
                return
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                self.rfile.read(length)
            if path != "/v1/chat/completions":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in tokens:
                if delay:
                    time.sleep(delay)
                payload = json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": token}}]}
                )
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


if __name__ == "__main__":
    raise SystemExit(main())
