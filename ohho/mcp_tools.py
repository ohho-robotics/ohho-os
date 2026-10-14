"""On-robot MCP tool handlers for OhhO OS (stdlib only)."""

from __future__ import annotations

import json
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
    _robot: Optional[Robot] = field(default=None, repr=False)

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
        session.robot.drive(vx=min(0.15, (dx * dx + dy * dy) ** 0.5), w=max(-0.6, min(0.6, err)))
        return _text({"navigating": True, "target": {"x": x, "y": y}})

    def frame_list_processes(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text(
            [
                {"name": "ohho-runtime", "status": "running"},
                {"name": "robot-transport", "status": "running" if session.robot else "stopped"},
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
        return _text([{"id": "mock-policy", "status": "available", "backend": "stdlib-mock"}])

    def data_list_episodes(_: Dict[str, Any]) -> Dict[str, Any]:
        return _text([])

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
                    "transport": {"type": "string", "description": "Transport URI, e.g. sim://"},
                    "runtime": {"type": "string", "description": "native | ros2 | auto"},
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
    ]

    for t in tools:
        if t.product not in ON_ROBOT_MCP_PRODUCTS:
            raise ValueError(f"tool {t.name} has non on-robot product {t.product}")
    return tools


def tools_by_name(session: RobotSession) -> Dict[str, McpTool]:
    return {t.name: t for t in build_tools(session)}
