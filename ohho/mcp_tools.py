"""On-robot MCP tool handlers for OhhO OS (stdlib only).

OHH-278: tools open ``sim://`` unless the human started ``ohho mcp serve``
with ``--allow-hardware``. That flag is not an environment variable and not
a tool argument. Hardware links use :class:`~ohho.safety.SafetyGate` via
:meth:`ohho.robot.Robot.connect` and stay disarmed — this process never arms
and never releases an e-stop. ``emergency_stop`` is always allowed and latches.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapters import available_adapters
from .adapters.errors import AdapterUnavailable
from .capabilities import BASE_HOLONOMIC
from .mcp_split import ON_ROBOT_MCP_PRODUCTS
from .registry import UnknownRobot, list_specs
from .robot import Robot
from .safety import SafetyGate
from .schema import clamp

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class RobotSession:
    """Optional live robot connection shared across tool calls."""

    robot_id: str = "omnibot"
    transport: str = "sim://"
    runtime: str = "native"
    memory_path: Optional[str] = None
    # Set only by ``ohho mcp serve --allow-hardware``. Tools must not assign it.
    allow_hardware: bool = False
    _robot: Optional[Robot] = field(default=None, repr=False)
    _world: str = field(default="", repr=False)

    def connect(self) -> Robot:
        """Open the session link. Never arms.

        An MCP stdio server's stdin is the JSON-RPC pipe, not an interactive
        terminal, so a hardware link stays disarmed until a human arms it
        from their own terminal.
        """
        if self._robot is not None:
            return self._robot
        self.transport = normalize_mcp_transport(
            self.transport, allow_hardware=self.allow_hardware
        )
        # Robot.connect wraps every non-sim transport in SafetyGate. Do not
        # call arm() or release_stop() here.
        self._robot = Robot.connect(
            self.robot_id, transport=self.transport, runtime=self.runtime
        )
        return self._robot

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
        self._world = ""

    @property
    def robot(self) -> Optional[Robot]:
        return self._robot


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler
    product: str
    read_only: bool = True


def _obj(props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _text(payload: Any) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": False,
    }


def _err(message: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _structured_error(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "isError": True,
    }


# Link v0 — real SDK calls. sim:// unless the human passed --allow-hardware.
SIM_TOOL_NAMES: tuple[str, ...] = ("connect", "drive", "telemetry", "nav_goto", "look")

# Names that must never be registered. Checked in ``_finalize_tools``.
_DENIED_TOKENS = frozenset({"arm", "disarm", "release", "limit", "limits", "profile"})
_DENIED_FOLDED = ("releasestop", "disarmoverride", "safetyprofile")
_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

_ASK_HUMAN = (
    "Ask the human to arm the robot in their own terminal with "
    "OHHO_ARM_HARDWARE=1 and `ohho connect <robot> --transport <uri> --arm`, "
    "then type the confirmation. This MCP process cannot arm: stdin is the "
    "JSON-RPC pipe, not an interactive terminal. Do not retry motion."
)


class HardwareRefused(ValueError):
    """A non-sim URI was requested without ``--allow-hardware``."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        super().__init__(
            f"transport {uri!r} is not allowed — Link v0 tools only support sim:// "
            "unless a human started the server with --allow-hardware. "
            "No tool argument or environment variable can enable hardware."
        )


def transport_scheme(uri: str) -> str:
    text = uri.strip()
    if "://" in text:
        return text.split("://", 1)[0].lower()
    if ":" in text:
        return text.split(":", 1)[0].lower()
    return text.lower()


def normalize_mcp_transport(transport: Optional[str], *, allow_hardware: bool) -> str:
    """Return a URI this MCP server may open.

    ``allow_hardware`` is the CLI flag only. This function does not read the
    environment, so ``OHHO_ALLOW_HARDWARE`` / ``OHHO_ARM_HARDWARE`` cannot
    turn a hardware link on.
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


def _tool_name_tokens(name: str) -> set[str]:
    tokens: set[str] = set()
    for piece in re.split(r"[._\-\s]+", name):
        bits = _TOKEN_RE.findall(piece)
        if bits:
            tokens.update(bit.lower() for bit in bits)
        elif piece:
            tokens.add(piece.lower())
    return tokens


def assert_mcp_tool_name_allowed(name: str) -> None:
    """Raise if ``name`` would arm, release an e-stop, or change limits.

    Called when tools are registered so a future tool with such a name fails
    before it is advertised.
    """
    tokens = _tool_name_tokens(name)
    hit = sorted(_DENIED_TOKENS.intersection(tokens))
    folded = re.sub(r"[^a-z0-9]", "", name.lower())
    folded_hit = [needle for needle in _DENIED_FOLDED if needle in folded]
    if hit or folded_hit:
        why = ", ".join(hit + folded_hit)
        raise ValueError(
            f"MCP tool {name!r} is forbidden (OHH-278: {why}). "
            "Agents cannot arm, release an e-stop, or change safety "
            "profiles or limits. emergency_stop remains allowed."
        )


def _backend(bot: Robot) -> str:
    return "sim" if bot.simulated else "hardware"


def _is_latched(bot: Robot) -> bool:
    transport = bot.transport
    if isinstance(transport, SafetyGate):
        return bool(transport.latched)
    return bool(getattr(transport, "_estopped", False))


def _not_connected(*, catalog: bool) -> Dict[str, Any]:
    if catalog:
        return _err("Not connected — call connect.connect first.")
    return _err("not connected — call connect first")


def _connect_failure(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, HardwareRefused):
        return _structured_error(
            {
                "error": "hardware_refused",
                "reason": "sim_only",
                "transport": exc.uri,
                "message": str(exc),
            }
        )
    return _err(str(exc))


def _hardware_session_error(bot: Robot) -> Dict[str, Any]:
    return _structured_error(
        {
            "error": "hardware_refused",
            "reason": "sim_only",
            "backend": _backend(bot),
            "message": (
                "Link v0 tools only support sim:// unless a human started "
                "`ohho mcp serve --allow-hardware`. No tool argument or "
                "environment variable can enable hardware."
            ),
        }
    )


def _require_link(
    session: RobotSession, *, catalog: bool = False
) -> Optional[Dict[str, Any]]:
    """Refuse a missing session, or hardware when the CLI flag is off."""
    if session.robot is None:
        return _not_connected(catalog=catalog)
    bot = session.robot
    if not bot.simulated and not session.allow_hardware:
        return _hardware_session_error(bot)
    return None


def _velocity_report(bot: Robot, vx: float, vy: float, w: float) -> Dict[str, Any]:
    """Clamp a request to the spec cap and, on a gate, the safety profile."""
    cap_vx = float(bot.spec.max_lin)
    cap_w = float(bot.spec.max_ang)
    holonomic = bot.has(BASE_HOLONOMIC)
    cap_vy = float(bot.spec.max_lin) if holonomic else 0.0
    gate = bot.transport
    if isinstance(gate, SafetyGate):
        profile = gate.profile
        cap_vx = min(cap_vx, float(profile.max_vx))
        cap_vy = min(cap_vy, float(profile.max_vy))
        cap_w = min(cap_w, float(profile.max_wz))
    requested = {"vx": float(vx), "vy": float(vy), "w": float(w)}
    applied_vy = clamp(requested["vy"], -cap_vy, cap_vy) if holonomic else 0.0
    applied = {
        "vx": clamp(requested["vx"], -cap_vx, cap_vx),
        "vy": applied_vy,
        "w": clamp(requested["w"], -cap_w, cap_w),
    }
    return {
        "requested": requested,
        "applied": applied,
        "clamped": any(requested[key] != applied[key] for key in ("vx", "vy", "w")),
        "cap": {"vx": cap_vx, "vy": cap_vy, "w": cap_w},
    }


def _motion_error(
    session: RobotSession,
    *,
    catalog: bool = False,
    report: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Block motion on a disarmed or latched link. ``emergency_stop`` does not use this."""
    refused = _require_link(session, catalog=catalog)
    if refused is not None:
        return refused
    bot = session.robot
    assert bot is not None
    if _is_latched(bot):
        bot.stop()
        payload: Dict[str, Any] = {
            "error": "motion_refused",
            "reason": "estop_latched",
            "backend": _backend(bot),
            "sent": False,
            "message": (
                "Emergency stop is latched, so motion is refused and zero "
                "velocity is sent. This MCP server cannot call release_stop. "
                "Ask the human to release the e-stop in their own terminal."
            ),
        }
        if report:
            payload.update(report)
        return _structured_error(payload)
    if not bot.simulated:
        gate = bot.transport if isinstance(bot.transport, SafetyGate) else None
        if gate is None or not gate.armed:
            bot.stop()
            payload = {
                "error": "motion_refused",
                "reason": "not_armed",
                "backend": "hardware",
                "sent": False,
                "message": (
                    "Hardware link is disarmed, so motion is refused and zero "
                    "velocity is sent. " + _ASK_HUMAN
                ),
            }
            if report:
                payload.update(report)
            return _structured_error(payload)
    return None


def _parse_velocity(
    params: Dict[str, Any], defaults: Dict[str, float]
) -> Optional[tuple]:
    try:
        return tuple(
            float(params.get(name, default)) for name, default in defaults.items()
        )
    except (TypeError, ValueError):
        return None


def _finalize_tools(tools: List[McpTool]) -> List[McpTool]:
    """Registration-time checks. A forbidden name fails here, not at call time."""
    seen: set[str] = set()
    for tool in tools:
        assert_mcp_tool_name_allowed(tool.name)
        if tool.name in seen:
            raise ValueError(f"duplicate MCP tool {tool.name}")
        seen.add(tool.name)
        if tool.product not in ON_ROBOT_MCP_PRODUCTS:
            raise ValueError(
                f"tool {tool.name} has non on-robot product {tool.product}"
            )
    return tools


def _maybe_demo_world(bot: Robot, world: str) -> None:
    if world != "demo":
        return
    add = getattr(bot.transport, "add_object", None)
    if not callable(add):
        return
    from .adapters.sim import demo_world

    for ob in demo_world():
        add(ob.x, ob.y, ob.radius, ob.label)


def _telemetry_payload(bot: Robot) -> Dict[str, Any]:
    t = bot.telemetry()
    out: Dict[str, Any] = {"timestamp": t.timestamp}
    if t.odom is not None:
        o = t.odom
        out["odom"] = {
            "x": o.x,
            "y": o.y,
            "theta": o.theta,
            "vx": o.vx,
            "vy": o.vy,
            "omega": o.omega,
        }
    if t.battery is not None:
        out["battery"] = t.battery
    if t.joints:
        out["joints"] = [
            {"name": j.name, "position": j.position, "velocity": j.velocity}
            for j in t.joints
        ]
    if t.custom:
        out["custom"] = dict(t.custom)
    return out


def _manual_step(bot: Robot) -> Optional[Callable[[float], Any]]:
    """Use ``step(dt)`` only when the sim thread is not already advancing time."""
    step_fn = getattr(bot.transport, "step", None)
    if not callable(step_fn):
        return None
    thread = getattr(bot.transport, "_thread", None)
    if thread is not None and getattr(thread, "is_alive", lambda: False)():
        return None
    return step_fn


def _hold_velocity(bot: Robot, vx: float, vy: float, w: float, seconds: float) -> None:
    seconds = max(0.0, min(float(seconds), 60.0))
    step_fn = _manual_step(bot)
    if step_fn is not None:
        dt = 0.05
        ticks = max(1, int(round(seconds / dt)))
        for _ in range(ticks):
            bot.drive(vx=vx, vy=vy, w=w)
            step_fn(dt)
        bot.stop()
        step_fn(dt)
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bot.drive(vx=vx, vy=vy, w=w)
        time.sleep(0.05)
    bot.stop()
    time.sleep(0.05)


def _apply_demo_world(session: RobotSession, world: str) -> None:
    bot = session.robot
    if bot is None or world != "demo" or session._world == "demo":
        return
    _maybe_demo_world(bot, world)
    session._world = "demo"


def build_tools(session: RobotSession) -> List[McpTool]:
    """Register on-robot tools backed by OhhO OS where possible."""

    def connect_list_robots(_: Dict[str, Any]) -> Dict[str, Any]:
        rows = [
            {
                "id": s.id,
                "name": s.name,
                "category": s.category,
                "capabilities": list(s.capabilities),
            }
            for s in list_specs()
        ]
        return _text(rows)

    def connect_get_status(_: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _text(
                {
                    "connected": False,
                    "robotId": session.robot_id,
                    "transport": session.transport,
                    "note": "Call connect.connect to attach (sim:// by default).",
                }
            )
        bot = session.robot
        if not bot.simulated and not session.allow_hardware:
            return _hardware_session_error(bot)
        st = bot.status()
        return _text(
            {
                "connected": True,
                "robotId": bot.spec.id,
                "protocol": bot.transport.protocol,
                "runtime": bot.runtime.name,
                "state": st.state.value,
                "label": st.label,
                "simulated": bot.simulated,
                "backend": _backend(bot),
            }
        )

    def connect_get_telemetry(_: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_link(session, catalog=True)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        t = bot.telemetry()
        odom = None
        if t.odom is not None:
            odom = {"x": t.odom.x, "y": t.odom.y, "theta": t.odom.theta}
        joints = None
        if t.joints:
            joints = [
                {"name": j.name, "position": j.position, "velocity": j.velocity}
                for j in t.joints
            ]
        return _text(
            {
                "battery": t.battery,
                "odom": odom,
                "joints": joints,
                "backend": _backend(bot),
            }
        )

    def _open(params: Dict[str, Any], *, robot_key: str) -> Dict[str, Any]:
        # Tool arguments cannot set allow_hardware. Only the CLI flag can.
        robot_id = str(params.get(robot_key) or session.robot_id)
        runtime = str(params.get("runtime") or session.runtime)
        try:
            if "transport" not in params or params.get("transport") is None:
                raw: Optional[str] = session.transport
            else:
                raw = str(params.get("transport"))
            transport = normalize_mcp_transport(
                raw, allow_hardware=session.allow_hardware
            )
        except ValueError as exc:
            return _connect_failure(exc)
        session.disconnect()
        session.robot_id = robot_id
        session.transport = transport
        session.runtime = runtime
        try:
            bot = session.connect()
        except (UnknownRobot, AdapterUnavailable, ValueError) as exc:
            return _connect_failure(exc)
        return {
            "bot": bot,
            "transport": transport,
        }

    def connect_connect(params: Dict[str, Any]) -> Dict[str, Any]:
        opened = _open(params, robot_key="robotId")
        if "bot" not in opened:
            return opened
        bot = opened["bot"]
        return _text(
            {
                "connected": True,
                "robot": repr(bot),
                "transport": opened["transport"],
                "backend": _backend(bot),
            }
        )

    def connect_send_velocity(params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = _parse_velocity(params, {"vx": 0.0, "vy": 0.0, "w": 0.0})
        if parsed is None:
            return _err("velocity arguments must be numbers")
        vx, vy, w = parsed
        if session.robot is None:
            return _not_connected(catalog=True)
        report = _velocity_report(session.robot, vx, vy, w)
        refused = _motion_error(session, catalog=True, report=report)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        applied = report["applied"]
        bot.drive(vx=applied["vx"], vy=applied["vy"], w=applied["w"])
        return _text({"ok": True, "backend": _backend(bot), **report})

    def latch_emergency_stop(_: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _not_connected(catalog=False)
        bot = session.robot
        bot.emergency_stop()
        return _text(
            {
                "estop": True,
                "latched": _is_latched(bot),
                "backend": _backend(bot),
            }
        )

    def bridge_list_adapters(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text(available_adapters())

    def pilot_apply_joystick(params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = _parse_velocity(params, {"vx": 0.0, "w": 0.0})
        if parsed is None:
            return _err("velocity arguments must be numbers")
        vx, w = parsed
        if session.robot is None:
            return _not_connected(catalog=True)
        report = _velocity_report(session.robot, vx, 0.0, w)
        refused = _motion_error(session, catalog=True, report=report)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        applied = report["applied"]
        bot.drive(vx=applied["vx"], vy=applied["vy"], w=applied["w"])
        return _text({"ok": True, "backend": _backend(bot), **report})

    def autonomy_navigate_to(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _motion_error(session, catalog=True)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        try:
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
        except (TypeError, ValueError):
            return _err("x and y must be numbers")
        # Lightweight stub: drive toward the target using a bearing heuristic.
        t = bot.telemetry()
        if t.odom is None:
            return _structured_error(
                {
                    "error": "no_odometry",
                    "backend": _backend(bot),
                    "message": "No odometry available for navigation.",
                }
            )
        dx, dy = x - t.odom.x, y - t.odom.y
        import math

        bearing = math.atan2(dy, dx)
        err = bearing - t.odom.theta
        report = _velocity_report(
            bot, min(0.15, (dx * dx + dy * dy) ** 0.5), 0.0, max(-0.6, min(0.6, err))
        )
        applied = report["applied"]
        bot.drive(vx=applied["vx"], vy=applied["vy"], w=applied["w"])
        return _text(
            {
                "navigating": True,
                "target": {"x": x, "y": y},
                "backend": _backend(bot),
                **report,
            }
        )

    def frame_list_processes(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text(
            [
                {"name": "ohho-runtime", "status": "running"},
                {
                    "name": "robot-transport",
                    "status": "running" if session.robot else "stopped",
                },
            ]
        )

    def bench_run_selftest(params: Dict[str, Any]) -> Dict[str, Any]:
        robot_id = str(params.get("robotId", session.robot_id))
        return _text(
            {
                "robotId": robot_id,
                "mode": "sim",
                "hint": "Run `ohho selftest --sim` for a full JSON report on this machine.",
            }
        )

    def serve_list_models(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text(
            [{"id": "mock-policy", "status": "available", "backend": "stdlib-mock"}]
        )

    def data_list_episodes(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text([])

    def sim_connect(params: Dict[str, Any]) -> Dict[str, Any]:
        opened = _open(params, robot_key="robot")
        if "bot" not in opened:
            return opened
        bot = opened["bot"]
        world = str(params.get("world") or "")
        if bot.simulated:
            _maybe_demo_world(bot, world)
            if world == "demo":
                session._world = "demo"
        status = bot.status()
        return _text(
            {
                "connected": True,
                "robot": bot.spec.id,
                "transport": opened["transport"],
                "protocol": status.protocol,
                "label": status.label,
                "backend": _backend(bot),
            }
        )

    def sim_drive(params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = _parse_velocity(
            params, {"vx": 0.1, "vy": 0.0, "w": 0.0, "seconds": 1.0}
        )
        if parsed is None:
            return _err("velocity arguments must be numbers")
        vx, vy, w, seconds = parsed
        if session.robot is None:
            return _not_connected(catalog=False)
        report = _velocity_report(session.robot, vx, vy, w)
        refused = _motion_error(session, report=report)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        applied = report["applied"]
        _hold_velocity(bot, applied["vx"], applied["vy"], applied["w"], seconds)
        return _text(
            {
                "drove": True,
                "backend": _backend(bot),
                **report,
                "telemetry": _telemetry_payload(bot),
            }
        )

    def sim_telemetry(_: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_link(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        payload = _telemetry_payload(bot)
        payload["backend"] = _backend(bot)
        return _text(payload)

    def sim_nav_goto(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _motion_error(session)
        if refused is not None:
            return refused
        if "x" not in params or "y" not in params:
            return _err("nav_goto requires x and y")
        from .nav import Navigator

        bot = session.robot
        assert bot is not None
        if not bot.simulated:
            return _structured_error(
                {
                    "error": "sim_only",
                    "reason": "nav_is_sim",
                    "backend": "hardware",
                    "message": (
                        "nav_goto uses the simulator map and only supports sim://. "
                        + _ASK_HUMAN
                    ),
                }
            )
        _apply_demo_world(session, str(params.get("world") or ""))
        result = Navigator(bot).navigate_to(
            float(params["x"]),
            float(params["y"]),
            timeout=float(params.get("timeout", 30.0)),
            step_fn=_manual_step(bot),
        )
        return _text(
            {
                "reached": result.reached,
                "reason": result.reason,
                "x": result.x,
                "y": result.y,
                "distance_remaining": result.distance_remaining,
                "backend": _backend(bot),
                "telemetry": _telemetry_payload(bot),
            }
        )

    def sim_look(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_link(session)
        if refused is not None:
            return refused
        from .memory import SpatialMemory, default_memory_path
        from .perception import SimPerceptor, remember_detections

        bot = session.robot
        assert bot is not None
        if not bot.simulated:
            return _structured_error(
                {
                    "error": "sim_only",
                    "reason": "look_is_sim_perception",
                    "backend": "hardware",
                    "message": "look reads the simulator's labeled world and only supports sim://.",
                }
            )
        _apply_demo_world(session, str(params.get("world") or ""))
        detections = SimPerceptor(bot).look()
        path = session.memory_path or default_memory_path(bot.spec.id)
        memory = SpatialMemory(path=path)
        remembered = remember_detections(memory, detections)
        memory.save()
        return _text(
            {
                "detections": [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "x": d.x,
                        "y": d.y,
                        "bearing": d.bearing,
                        "range": d.range,
                    }
                    for d in detections
                ],
                "remembered": remembered,
                "memory_path": str(path),
                "backend": _backend(bot),
            }
        )

    tools = [
        McpTool(
            "connect.listRobots",
            "List robot specs known to OhhO OS on this device.",
            _obj({}),
            connect_list_robots,
            "connect",
            True,
        ),
        McpTool(
            "connect.getStatus",
            "Connection status for the active robot session (if any).",
            _obj({}),
            connect_get_status,
            "connect",
            True,
        ),
        McpTool(
            "connect.getTelemetry",
            "Latest telemetry from the connected robot.",
            _obj({}),
            connect_get_telemetry,
            "connect",
            True,
        ),
        McpTool(
            "connect.connect",
            "Connect to a robot (defaults to simulation).",
            _obj(
                {
                    "robotId": {"type": "string", "description": "Robot spec id"},
                    "transport": {
                        "type": "string",
                        "description": "Transport URI, e.g. sim://",
                    },
                    "runtime": {
                        "type": "string",
                        "description": "native | ros2 | auto",
                    },
                }
            ),
            connect_connect,
            "connect",
            False,
        ),
        McpTool(
            "connect.sendVelocity",
            "Send a velocity command to the connected robot base.",
            _obj(
                {
                    "vx": {"type": "number"},
                    "vy": {"type": "number"},
                    "w": {"type": "number"},
                }
            ),
            connect_send_velocity,
            "connect",
            False,
        ),
        McpTool(
            "connect.emergencyStop",
            "Latch emergency stop on the connected robot. Always allowed; does not release the latch or arm.",
            _obj({}),
            latch_emergency_stop,
            "connect",
            False,
        ),
        McpTool(
            "bridge.listAdapters",
            "List transport adapters available in this OhhO OS install.",
            _obj({}),
            bridge_list_adapters,
            "bridge",
            True,
        ),
        McpTool(
            "pilot.applyJoystick",
            "Apply a teleop joystick command (vx, w) to the connected robot.",
            _obj({"vx": {"type": "number"}, "w": {"type": "number"}}),
            pilot_apply_joystick,
            "pilot",
            False,
        ),
        McpTool(
            "autonomy.navigateTo",
            "Navigate toward a planar goal using on-robot control (sim-safe).",
            _obj({"x": {"type": "number"}, "y": {"type": "number"}}, ["x", "y"]),
            autonomy_navigate_to,
            "autonomy",
            False,
        ),
        McpTool(
            "frame.listProcesses",
            "List representative on-robot processes managed by OhhO OS.",
            _obj({}),
            frame_list_processes,
            "frame",
            True,
        ),
        McpTool(
            "bench.runSelfTest",
            "Start or describe a bench self-test for the robot.",
            _obj({"robotId": {"type": "string"}}),
            bench_run_selftest,
            "bench",
            False,
        ),
        McpTool(
            "serve.listModels",
            "List policy models available to the on-robot serve stack.",
            _obj({}),
            serve_list_models,
            "serve",
            True,
        ),
        McpTool(
            "data.listEpisodes",
            "List recorded episodes on this robot (empty when none are stored).",
            _obj({}),
            data_list_episodes,
            "data",
            True,
        ),
        McpTool(
            "connect",
            "Connect to a robot. sim:// only, unless the human started the server with --allow-hardware. Replaces any existing session. Never arms.",
            _obj(
                {
                    "robot": {
                        "type": "string",
                        "description": "Robot id from `ohho list` (default: omnibot).",
                    },
                    "transport": {
                        "type": "string",
                        "description": "sim:// unless the server was started with --allow-hardware. A tool argument cannot enable hardware.",
                    },
                    "world": {
                        "type": "string",
                        "description": "Optional sim world preset; use 'demo' for labeled obstacles.",
                    },
                }
            ),
            sim_connect,
            "connect",
            False,
        ),
        McpTool(
            "drive",
            "Hold a velocity command, then stop. Clamped to the sim or safety-profile cap. Refused when a hardware link is disarmed.",
            _obj(
                {
                    "vx": {"type": "number", "description": "Forward m/s"},
                    "vy": {
                        "type": "number",
                        "description": "Lateral m/s (holonomic only)",
                    },
                    "w": {"type": "number", "description": "Yaw rate rad/s"},
                    "seconds": {
                        "type": "number",
                        "description": "How long to hold the command",
                    },
                }
            ),
            sim_drive,
            "pilot",
            False,
        ),
        McpTool(
            "telemetry",
            "Read odometry, battery, and joint state from the connected sim robot.",
            _obj({}),
            sim_telemetry,
            "connect",
            True,
        ),
        McpTool(
            "nav_goto",
            "Navigate to a world-frame (x, y) using native A* (sim only).",
            _obj(
                {
                    "x": {"type": "number", "description": "Goal x (m)"},
                    "y": {"type": "number", "description": "Goal y (m)"},
                    "timeout": {"type": "number", "description": "Max seconds"},
                    "world": {
                        "type": "string",
                        "description": "Optional 'demo' obstacles (sim); only applied if not yet set.",
                    },
                },
                ["x", "y"],
            ),
            sim_nav_goto,
            "autonomy",
            False,
        ),
        McpTool(
            "look",
            "Sim perception: detect labeled obstacles in view and update spatio-temporal memory.",
            _obj(
                {
                    "world": {
                        "type": "string",
                        "description": "Optional 'demo' obstacles (sim).",
                    },
                }
            ),
            sim_look,
            "connect",
            False,
        ),
        McpTool(
            "emergency_stop",
            "Latch emergency stop on the connected robot (simulation or hardware). Always allowed. Does not clear the latch and does not arm.",
            _obj({}),
            latch_emergency_stop,
            "connect",
            False,
        ),
    ]

    return _finalize_tools(tools)


def tools_by_name(session: RobotSession) -> Dict[str, McpTool]:
    return {t.name: t for t in build_tools(session)}
