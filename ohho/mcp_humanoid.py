"""MCP tools for a humanoid (OHH-335).

Tools are chosen from capabilities. A robot id never selects them.
``humanoid.locomotion`` enables ``humanoid_walk`` and ``humanoid_stop``.
``humanoid.locomotion`` or ``humanoid.posture`` enables ``emergency_stop``
and ``humanoid_telemetry``.

``arm``, ``damp``, ``move_joints``, dances and trajectory tools are never
registered, even when a config names them. This module does not call
:meth:`ohho.robot.Robot.arm`, :meth:`ohho.robot.Robot.damp`,
:meth:`ohho.robot.Robot.move_joints`, or :meth:`ohho.robot.Robot.release_stop`.

Sim-only unless the session was constructed with ``allow_hardware``. A tool
argument cannot turn that flag on. ``humanoid_walk`` is clamped with
:class:`~ohho.humanoid_policy.HumanoidPolicy` before the backend sees it.
Every tool payload includes ``backend``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import capabilities as caps
from .adapters.errors import AdapterUnavailable
from .humanoid_policy import HumanoidPolicy, humanoid_denied, humanoid_mcp_allowed
from .registry import RobotSpec, UnknownRobot, get_spec
from .robot import Robot
from .safety import SafetyGate, normalize_action
from .schema import Telemetry
from .session import RobotSession

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

HUMANOID_TOOL_NAMES: tuple[str, ...] = (
    "humanoid_walk",
    "humanoid_stop",
    "emergency_stop",
    "humanoid_telemetry",
)

_LOCOMOTION_TOOLS = frozenset({"humanoid_walk", "humanoid_stop"})
_EITHER_TOOLS = frozenset({"emergency_stop", "humanoid_telemetry"})

# Whole-token and folded names that must never be advertised.
_DENIED_ACTIONS = frozenset({"arm", "damp", "move_joints", "release_stop"})
_DENIED_FOLDED = frozenset({"arm", "damp", "movejoints", "releasestop"})
_DENIED_TOKENS = frozenset({"arm", "damp"})
_DENIED_FRAGMENTS = (
    "dance",
    "trajectory",
    "traj",
    "zerotorque",
    "handshake",
    "kick",
    "shoot",
    "liondance",
)
_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_CONFIG_KEYS = ("tools", "mcp_tools", "allowlist")


class HardwareRefused(ValueError):
    """A non-sim URI was requested without ``allow_hardware``."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        super().__init__(
            f"transport {uri!r} is not allowed — humanoid MCP tools only "
            "support sim:// unless the session was opened with allow_hardware. "
            "No tool argument or environment variable can enable hardware."
        )


@dataclass(frozen=True)
class McpTool:
    """One MCP tool. ``handler`` returns a content payload."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    read_only: bool = True


class HumanoidMcpSession:
    """Humanoid tools bound to one :class:`~ohho.session.RobotSession`.

    The robot id comes from that session (flag, then ``OHHO_ROBOT``, then
    config, then ``sim``). ``allow_hardware`` is not a tool argument.
    ``config`` may name extra tools; the hard denylist still drops them.
    ``robot`` is a test seam for an already-open simulated link.
    """

    def __init__(
        self,
        robot_id: str | None = None,
        *,
        transport: str | None = None,
        runtime: str = "native",
        allow_hardware: bool = False,
        config: Mapping[str, Any] | None = None,
        robot: Robot | None = None,
        home: Path | str | None = None,
        cwd: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.session = RobotSession(
            robot_id,
            transport=transport,
            runtime=runtime,
            home=home,
            cwd=cwd,
            environ=environ,
            config_path=config_path,
        )
        self.allow_hardware = bool(allow_hardware)
        self.config = dict(config) if config else {}
        self._robot = robot

    @property
    def robot(self) -> Robot | None:
        return self._robot

    @property
    def robot_id(self) -> str:
        if self._robot is not None:
            return self._robot.spec.id
        return self.session.robot_id

    def transport_uri(self) -> str:
        raw = self.session.transport
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return "sim://"

    def connect(self) -> Robot:
        """Open the resolved robot. Does not arm, damp, or command joints."""
        if self._robot is not None:
            return self._robot
        uri = normalize_mcp_transport(
            self.transport_uri(), allow_hardware=self.allow_hardware
        )
        profile = self.session.profile
        extra: dict[str, Any] = {}
        if profile.sim_backend:
            extra["sim_backend"] = profile.sim_backend
        if profile.asset_path:
            extra["asset_path"] = profile.asset_path
        if profile.safety_overrides:
            extra["safety_overrides"] = dict(profile.safety_overrides)
        self._robot = Robot.connect(
            self.session.robot_id,
            transport=uri,
            runtime=self.session.runtime,
            **extra,
        )
        return self._robot

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None


def transport_scheme(uri: str) -> str:
    text = uri.strip()
    if "://" in text:
        return text.split("://", 1)[0].lower()
    if ":" in text:
        return text.split(":", 1)[0].lower()
    return text.lower()


def normalize_mcp_transport(transport: str | None, *, allow_hardware: bool) -> str:
    """Return a URI this session may open.

    ``allow_hardware`` is the session flag only. The environment is not read.
    """
    uri = (transport or "sim://").strip() or "sim://"
    scheme = transport_scheme(uri)
    if scheme == "sim":
        return uri if "://" in uri else "sim://"
    if not allow_hardware:
        raise HardwareRefused(uri)
    if "://" not in uri:
        raise ValueError(f"transport {uri!r} is not a URI")
    return uri


def _tokens(name: str) -> set[str]:
    tokens: set[str] = set()
    for piece in re.split(r"[._\-\s]+", name):
        bits = _TOKEN_RE.findall(piece)
        if bits:
            tokens.update(bit.lower() for bit in bits)
        elif piece:
            tokens.add(piece.lower())
    return tokens


def _folded(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_action(name))


def is_hard_denied_tool(name: object) -> bool:
    """True for arm, damp, joints, dances, trajectories, and the shared denylist.

    Blank names are denied so a config cannot advertise an empty tool.
    """
    if not isinstance(name, str) or not name.strip():
        return True
    action = normalize_action(name)
    folded = _folded(name)
    if humanoid_denied(action) or action in _DENIED_ACTIONS:
        return True
    if folded in _DENIED_FOLDED or "movejoints" in folded:
        return True
    if _tokens(name) & _DENIED_TOKENS:
        return True
    if "move" in _tokens(name) and "joints" in _tokens(name):
        return True
    return any(fragment in folded for fragment in _DENIED_FRAGMENTS)


def configured_tool_names(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Tool names a config asked for. Denied names are kept here and dropped later."""
    if not config:
        return ()
    names: list[str] = []
    for key in _CONFIG_KEYS:
        raw = config.get(key)
        if isinstance(raw, str):
            names.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, str)):
            names.extend(str(item) for item in raw)
    return tuple(names)


def _capable(spec: RobotSpec, name: str) -> bool:
    locomotion = spec.has(caps.HUMANOID_LOCOMOTION)
    posture = spec.has(caps.HUMANOID_POSTURE)
    if name in _LOCOMOTION_TOOLS:
        return locomotion
    if name in _EITHER_TOOLS:
        return locomotion or posture
    return False


def exposed_tool_names(
    spec: RobotSpec | None, config: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Capability tools, plus any config name that is both allowed and capable.

    The hard denylist is applied last. A config cannot add ``arm``, ``damp``,
    ``move_joints``, a dance, or a trajectory tool.
    """
    for name in HUMANOID_TOOL_NAMES:
        if is_hard_denied_tool(name):
            raise RuntimeError(f"built-in MCP tool {name!r} is on the hard denylist")
    if spec is None:
        return ()
    ordered: list[str] = []
    requested = list(HUMANOID_TOOL_NAMES) + list(configured_tool_names(config))
    for name in requested:
        if name in ordered or is_hard_denied_tool(name):
            continue
        if name not in HUMANOID_TOOL_NAMES or not _capable(spec, name):
            continue
        ordered.append(name)
    return tuple(ordered)


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _text(payload: dict[str, Any], *, error: bool) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": error,
    }


def _backend_name(session: HumanoidMcpSession) -> str:
    bot = session.robot
    if bot is not None:
        return "sim" if bot.simulated else "hardware"
    scheme = transport_scheme(session.transport_uri())
    return "sim" if scheme == "sim" else "hardware"


def _with_backend(
    session: HumanoidMcpSession, payload: dict[str, Any], *, error: bool
) -> dict[str, Any]:
    body = dict(payload)
    body["backend"] = _backend_name(session)
    return _text(body, error=error)


def _ok(session: HumanoidMcpSession, payload: dict[str, Any]) -> dict[str, Any]:
    return _with_backend(session, payload, error=False)


def _fail(session: HumanoidMcpSession, payload: dict[str, Any]) -> dict[str, Any]:
    return _with_backend(session, payload, error=True)


def _spec_of(session: HumanoidMcpSession) -> RobotSpec | None:
    if session.robot is not None:
        return session.robot.spec
    try:
        return get_spec(session.session.robot_id)
    except UnknownRobot:
        return None


def _open(session: HumanoidMcpSession) -> Robot | dict[str, Any]:
    try:
        return session.connect()
    except HardwareRefused as exc:
        return _fail(
            session,
            {
                "error": "hardware_refused",
                "reason": "sim_only",
                "transport": exc.uri,
                "message": str(exc),
            },
        )
    except (UnknownRobot, AdapterUnavailable, ValueError) as exc:
        return _fail(session, {"error": "connect_failed", "message": str(exc)})


def _require(
    session: HumanoidMcpSession, tool: str
) -> tuple[Robot, RobotSpec] | dict[str, Any]:
    opened = _open(session)
    if isinstance(opened, dict):
        return opened
    spec = opened.spec
    if tool not in exposed_tool_names(spec, session.config):
        return _fail(
            session,
            {
                "error": "not_available",
                "tool": tool,
                "message": f"{tool} is not enabled for this robot's capabilities",
            },
        )
    return opened, spec


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _is_latched(bot: Robot) -> bool:
    transport = bot.transport
    if isinstance(transport, SafetyGate):
        return bool(transport.latched)
    if getattr(transport, "_estopped", False):
        return True
    return bool(getattr(transport, "_estop_latched", False))


def _policy(bot: Robot, uri: str) -> HumanoidPolicy:
    policy = HumanoidPolicy.for_spec(bot.spec)
    policy.arming(uri, {})
    return policy


def _clamp_walk(
    bot: Robot, uri: str, vx: float, vy: float, wz: float
) -> tuple[tuple[float, float, float], tuple[str, ...]] | str:
    policy = _policy(bot, uri)
    if not humanoid_mcp_allowed("move"):
        return "mcp_refused"
    decision = policy.clamp(
        "velocity",
        ((vx, vy, wz), (0.0, 0.0, 0.0), float(policy.profile.cmd_timeout_s)),
        bot.telemetry(),
    )
    if decision.reject or len(decision.applied) != 3:
        return decision.reject or "rejected"
    applied = tuple(float(item) for item in decision.applied)
    return (applied[0], applied[1], applied[2]), decision.reasons


def _humanoid_block(sample: Telemetry) -> dict[str, Any] | None:
    block = sample.custom.get("humanoid")
    if isinstance(block, Mapping):
        return dict(block)
    return None


def _telemetry_body(bot: Robot) -> dict[str, Any]:
    sample = bot.transport.read()
    payload: dict[str, Any] = {
        "timestamp": sample.timestamp,
        "battery": sample.battery,
        "humanoid": _humanoid_block(sample),
    }
    if sample.odom is not None:
        odom = sample.odom
        payload["odom"] = {
            "x": odom.x,
            "y": odom.y,
            "theta": odom.theta,
            "vx": odom.vx,
            "vy": odom.vy,
            "omega": odom.omega,
        }
    if sample.joints:
        payload["joints"] = [
            {"name": joint.name, "position": joint.position, "velocity": joint.velocity}
            for joint in sample.joints
        ]
    return payload


def build_humanoid_tools(session: HumanoidMcpSession) -> list[McpTool]:
    """Tools for ``session``. Denied config names are omitted, not registered."""
    spec = _spec_of(session)
    names = exposed_tool_names(spec, session.config)

    def humanoid_walk(params: dict[str, Any]) -> dict[str, Any]:
        # Tool arguments cannot set allow_hardware, and this handler does
        # not arm, damp, or send joint targets.
        opened = _require(session, "humanoid_walk")
        if isinstance(opened, dict):
            return opened
        bot, _spec = opened
        vx = _finite(params.get("vx", 0.0))
        vy = _finite(params.get("vy", 0.0))
        raw_wz = params.get("wz", params.get("w", 0.0))
        wz = _finite(raw_wz)
        if vx is None or vy is None or wz is None:
            return _fail(
                session, {"error": "bad_arguments", "message": "non-finite velocity"}
            )
        if _is_latched(bot):
            bot.stop()
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": "estop_latched",
                    "sent": False,
                    "message": (
                        "Emergency stop is latched, so the walk was not sent. "
                        "This MCP server cannot call release_stop."
                    ),
                },
            )
        if not bot.simulated:
            gate = bot.transport if isinstance(bot.transport, SafetyGate) else None
            if gate is None or not gate.armed:
                bot.stop()
                return _fail(
                    session,
                    {
                        "error": "motion_refused",
                        "reason": "not_armed",
                        "sent": False,
                        "message": (
                            "Hardware link is disarmed, so the walk was not sent. "
                            "This MCP process cannot arm."
                        ),
                    },
                )
        clamped = _clamp_walk(bot, session.transport_uri(), vx, vy, wz)
        requested = {"vx": vx, "vy": vy, "wz": wz}
        if isinstance(clamped, str):
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": clamped,
                    "sent": False,
                    "requested": requested,
                    "message": "walk was not forwarded",
                },
            )
        applied, reasons = clamped
        bot.drive(vx=applied[0], vy=applied[1], w=applied[2])
        sent = {"vx": applied[0], "vy": applied[1], "wz": applied[2]}
        return _ok(
            session,
            {
                "ok": True,
                "sent": True,
                "action": "move",
                "requested": requested,
                "applied": sent,
                "clamped": requested != sent,
                "reasons": list(reasons),
            },
        )

    def humanoid_stop(_: dict[str, Any]) -> dict[str, Any]:
        opened = _require(session, "humanoid_stop")
        if isinstance(opened, dict):
            return opened
        bot, _spec = opened
        if not humanoid_mcp_allowed("stop_move"):
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": "mcp_refused",
                    "sent": False,
                    "action": "stop_move",
                },
            )
        bot.stop()
        return _ok(session, {"ok": True, "sent": True, "action": "stop_move"})

    def emergency_stop(_: dict[str, Any]) -> dict[str, Any]:
        opened = _require(session, "emergency_stop")
        if isinstance(opened, dict):
            return opened
        bot, _spec = opened
        bot.emergency_stop()
        return _ok(
            session,
            {
                "ok": True,
                "sent": True,
                "action": "emergency_stop",
                "latched": _is_latched(bot),
            },
        )

    def humanoid_telemetry(_: dict[str, Any]) -> dict[str, Any]:
        opened = _require(session, "humanoid_telemetry")
        if isinstance(opened, dict):
            return opened
        bot, _spec = opened
        return _ok(session, _telemetry_body(bot))

    catalog = {
        "humanoid_walk": McpTool(
            "humanoid_walk",
            "Walk at a body velocity. Clamped by the humanoid safety policy "
            "before the backend. Does not arm, damp, or send joint targets.",
            _obj(
                {
                    "vx": {
                        "type": "number",
                        "description": "Forward m/s. Positive is forward.",
                    },
                    "vy": {"type": "number", "description": "Lateral m/s."},
                    "wz": {"type": "number", "description": "Yaw rate rad/s."},
                }
            ),
            humanoid_walk,
            read_only=False,
        ),
        "humanoid_stop": McpTool(
            "humanoid_stop",
            "Send zero body velocity. Does not damp and does not change mode.",
            _obj({}),
            humanoid_stop,
            read_only=False,
        ),
        "emergency_stop": McpTool(
            "emergency_stop",
            "Latch emergency stop. Does not clear the latch and does not arm.",
            _obj({}),
            emergency_stop,
            read_only=False,
        ),
        "humanoid_telemetry": McpTool(
            "humanoid_telemetry",
            "Read humanoid telemetry. Does not send a command.",
            _obj({}),
            humanoid_telemetry,
            read_only=True,
        ),
    }
    tools = [catalog[name] for name in names]
    listed = tuple(tool.name for tool in tools)
    if listed != names:
        raise RuntimeError(f"humanoid tool order drifted: {listed}")
    for tool in tools:
        if is_hard_denied_tool(tool.name):
            raise RuntimeError(f"refusing to register {tool.name}")
    return tools


def tools_by_name(session: HumanoidMcpSession) -> dict[str, McpTool]:
    return {tool.name: tool for tool in build_humanoid_tools(session)}


def handle_request(
    request: dict[str, Any], session: HumanoidMcpSession | None = None
) -> dict[str, Any]:
    """Handle one MCP JSON-RPC request for the humanoid tools."""
    bound = session or HumanoidMcpSession()
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params") or {}
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ohho-os"},
            },
        }
    if method == "tools/list":
        tools = build_humanoid_tools(bound)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    }
                    for tool in tools
                ]
            },
        }
    if method == "tools/call":
        name = params.get("name")
        if not name:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Missing 'name' parameter."},
            }
        tool = tools_by_name(bound).get(str(name))
        if tool is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "'arguments' must be an object."},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": tool.handler(args)}
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def payload_of(result: Mapping[str, Any]) -> dict[str, Any]:
    """JSON object stored in an MCP tool result."""
    content = result.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("tool result has no content")
    text = content[0].get("text") if isinstance(content[0], Mapping) else None
    if not isinstance(text, str):
        raise ValueError("tool result content is not text")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("tool payload is not an object")
    return parsed


__all__ = [
    "HUMANOID_TOOL_NAMES",
    "HardwareRefused",
    "HumanoidMcpSession",
    "McpTool",
    "build_humanoid_tools",
    "configured_tool_names",
    "exposed_tool_names",
    "handle_request",
    "is_hard_denied_tool",
    "normalize_mcp_transport",
    "payload_of",
    "tools_by_name",
    "transport_scheme",
]
