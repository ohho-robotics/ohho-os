"""MCP JSON-RPC server for on-robot tools (stdio transport, stdlib only).

Handlers live in :mod:`ohho.mcp_tools`. That includes the OHH-173 on-robot
catalog and the OHH-47 Link v0 tools (``connect``, ``drive``, ``telemetry``,
``nav_goto``, ``look``) plus ``emergency_stop``. Transports are ``sim://``
unless the human started the process with ``--allow-hardware``. The server
never arms a robot and never releases an e-stop (OHH-278).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from . import __version__
from .mcp_tools import RobotSession, build_tools, tools_by_name

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "ohho-os"


class McpServer:
    def __init__(self, session: Optional[RobotSession] = None) -> None:
        self.session = session or RobotSession()
        self._tools = tools_by_name(self.session)

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                return self._ok(
                    req_id,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": SERVER_NAME, "version": __version__},
                    },
                )
            if method == "ping":
                return self._ok(req_id, {})
            if method == "tools/list":
                return self._ok(
                    req_id,
                    {
                        "tools": [
                            {
                                "name": t.name,
                                "description": t.description,
                                "inputSchema": t.input_schema,
                            }
                            for t in build_tools(self.session)
                        ]
                    },
                )
            if method == "tools/call":
                name = params.get("name")
                if not name:
                    return self._err(req_id, -32602, "Missing 'name' parameter.")
                tool = self._tools.get(str(name))
                if tool is None:
                    return self._err(req_id, -32601, f"Unknown tool: {name}")
                args = params.get("arguments") or {}
                if not isinstance(args, dict):
                    return self._err(req_id, -32602, "'arguments' must be an object.")
                result = tool.handler(args)
                return self._ok(req_id, result)
            return self._err(req_id, -32601, f"Unknown method: {method}")
        except Exception as exc:  # noqa: BLE001 — surface to MCP client
            return self._err(req_id, -32603, f"Internal error: {exc}")

    @staticmethod
    def _ok(req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


def run_stdio(session: Optional[RobotSession] = None) -> int:
    """Read MCP messages from stdin and write responses to stdout."""
    server = McpServer(session=session)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        headers: Dict[str, str] = {}
        line = stdin.readline()
        if not line:
            break
        while line.strip():
            if b":" in line:
                key, val = line.decode("ascii", errors="replace").split(":", 1)
                headers[key.strip().lower()] = val.strip()
            line = stdin.readline()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            continue
        body = stdin.read(length)
        if not body:
            break
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            responses = [server.handle(item) for item in payload]
            out = json.dumps(responses).encode("utf-8")
        else:
            out = json.dumps(server.handle(payload)).encode("utf-8")
        stdout.write(f"Content-Length: {len(out)}\r\n\r\n".encode("ascii"))
        stdout.write(out)
        stdout.flush()
    if session and session.robot is not None:
        session.disconnect()
    return 0


def handle_request(
    request: Dict[str, Any], session: Optional[RobotSession] = None
) -> Dict[str, Any]:
    """Test helper — handle a single JSON-RPC request dict."""
    return McpServer(session=session).handle(request)
