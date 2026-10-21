"""On-robot MCP tool handlers for OhhO OS (stdlib only)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapters import available_adapters
from .mcp_split import ON_ROBOT_MCP_PRODUCTS
from .registry import UnknownRobot, list_specs
from .robot import Robot

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class RobotSession:
    """Optional live robot connection shared across tool calls."""

    robot_id: str = "omnibot"
    transport: str = "sim://"
    runtime: str = "native"
    memory_path: Optional[str] = None
    _robot: Optional[Robot] = field(default=None, repr=False)
    _world: str = field(default="", repr=False)

    def connect(self) -> Robot:
        if self._robot is not None:
            return self._robot
        self._robot = Robot.connect(
            self.robot_id, transport=self.transport, runtime=self.runtime
        )
        if self._robot.gated:
            self._robot.arm()
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


# Link v0 — real SDK calls, simulation only. Registered beside the OHH-173 catalog.
SIM_TOOL_NAMES: tuple[str, ...] = ("connect", "drive", "telemetry", "nav_goto", "look")


def _normalize_sim_transport(transport: Optional[str]) -> str:
    uri = (transport or "sim://").strip() or "sim://"
    scheme = uri.split(":", 1)[0].lower()
    if scheme != "sim":
        raise ValueError(
            f"transport '{uri}' is not allowed — Link v0 tools only support sim://"
        )
    return uri if "://" in uri else "sim://"


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


def _require_sim(session: RobotSession) -> Optional[Dict[str, Any]]:
    if session.robot is None:
        return _err("not connected — call connect first")
    if not session.robot.simulated:
        return _err("Link v0 tools only support sim://")
    return None


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
            }
        )

    def connect_get_telemetry(_: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _err("Not connected — call connect.connect first.")
        t = session.robot.telemetry()
        odom = None
        if t.odom is not None:
            odom = {"x": t.odom.x, "y": t.odom.y, "theta": t.odom.theta}
        joints = None
        if t.joints:
            joints = [
                {"name": j.name, "position": j.position, "velocity": j.velocity}
                for j in t.joints
            ]
        return _text({"battery": t.battery, "odom": odom, "joints": joints})

    def connect_connect(params: Dict[str, Any]) -> Dict[str, Any]:
        robot_id = str(params.get("robotId", session.robot_id))
        transport = str(params.get("transport", session.transport))
        runtime = str(params.get("runtime", session.runtime))
        session.disconnect()
        session.robot_id = robot_id
        session.transport = transport
        session.runtime = runtime
        try:
            bot = session.connect()
        except UnknownRobot as e:
            return _err(str(e))
        return _text({"connected": True, "robot": repr(bot)})

    def connect_send_velocity(params: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _err("Not connected — call connect.connect first.")
        vx = float(params.get("vx", 0.0))
        vy = float(params.get("vy", 0.0))
        w = float(params.get("w", 0.0))
        session.robot.drive(vx=vx, vy=vy, w=w)
        return _text({"ok": True, "vx": vx, "vy": vy, "w": w})

    def connect_emergency_stop(_: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _err("Not connected — call connect.connect first.")
        session.robot.emergency_stop()
        return _text({"estop": True})

    def bridge_list_adapters(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text(available_adapters())

    def pilot_apply_joystick(params: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _err("Not connected — call connect.connect first.")
        vx = float(params.get("vx", 0.0))
        w = float(params.get("w", 0.0))
        session.robot.drive(vx=vx, w=w)
        return _text({"ok": True})

    def autonomy_navigate_to(params: Dict[str, Any]) -> Dict[str, Any]:
        if session.robot is None:
            return _err("Not connected — call connect.connect first.")
        x = float(params.get("x", 0.0))
        y = float(params.get("y", 0.0))
        # Lightweight stub: drive toward the target using a bearing heuristic.
        t = session.robot.telemetry()
        if t.odom is None:
            return _err("No odometry available for navigation.")
        dx, dy = x - t.odom.x, y - t.odom.y
        import math

        bearing = math.atan2(dy, dx)
        err = bearing - t.odom.theta
        session.robot.drive(
            vx=min(0.15, (dx * dx + dy * dy) ** 0.5), w=max(-0.6, min(0.6, err))
        )
        return _text({"navigating": True, "target": {"x": x, "y": y}})

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
        robot_id = str(params.get("robot") or session.robot_id)
        raw_transport = params.get("transport")
        try:
            transport = _normalize_sim_transport(
                None if raw_transport is None else str(raw_transport)
            )
        except ValueError as exc:
            return _err(str(exc))
        world = str(params.get("world") or "")
        runtime = str(params.get("runtime") or session.runtime)
        session.disconnect()
        session.robot_id = robot_id
        session.transport = transport
        session.runtime = runtime
        try:
            bot = session.connect()
        except UnknownRobot as exc:
            return _err(str(exc))
        _maybe_demo_world(bot, world)
        if world == "demo":
            session._world = "demo"
        status = bot.status()
        return _text(
            {
                "connected": True,
                "robot": bot.spec.id,
                "transport": transport,
                "protocol": status.protocol,
                "label": status.label,
            }
        )

    def sim_drive(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_sim(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        _hold_velocity(
            bot,
            float(params.get("vx", 0.1)),
            float(params.get("vy", 0.0)),
            float(params.get("w", 0.0)),
            float(params.get("seconds", 1.0)),
        )
        return _text({"drove": True, "telemetry": _telemetry_payload(bot)})

    def sim_telemetry(_: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_sim(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        return _text(_telemetry_payload(bot))

    def sim_nav_goto(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_sim(session)
        if refused is not None:
            return refused
        if "x" not in params or "y" not in params:
            return _err("nav_goto requires x and y")
        from .nav import Navigator

        bot = session.robot
        assert bot is not None
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
                "telemetry": _telemetry_payload(bot),
            }
        )

    def sim_look(params: Dict[str, Any]) -> Dict[str, Any]:
        refused = _require_sim(session)
        if refused is not None:
            return refused
        from .memory import SpatialMemory, default_memory_path
        from .perception import SimPerceptor, remember_detections

        bot = session.robot
        assert bot is not None
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
            "Latch emergency stop on the connected robot.",
            _obj({}),
            connect_emergency_stop,
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
            "Connect to a robot in simulation (sim:// only). Replaces any existing session.",
            _obj(
                {
                    "robot": {
                        "type": "string",
                        "description": "Robot id from `ohho list` (default: omnibot).",
                    },
                    "transport": {
                        "type": "string",
                        "description": "Must be sim:// for this tool.",
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
            "Send a velocity command on the connected sim robot for a duration, then stop.",
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
    ]

    for t in tools:
        if t.product not in ON_ROBOT_MCP_PRODUCTS:
            raise ValueError(f"tool {t.name} has non on-robot product {t.product}")
    return tools


def tools_by_name(session: RobotSession) -> Dict[str, McpTool]:
    return {t.name: t for t in build_tools(session)}
