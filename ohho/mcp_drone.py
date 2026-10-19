"""MCP tools for an aircraft (OHH-293).

Sim-only unless the human started ``ohho mcp serve --allow-hardware``.
That flag is the OHH-278 switch shared with ground robots. A tool argument
cannot turn it on.

This module never calls :meth:`ohho.robot.Robot.arm`,
:meth:`ohho.robot.Robot.arm_flight`, :meth:`ohho.robot.Robot.takeoff`,
:meth:`ohho.robot.Robot.rtl`, or :meth:`ohho.robot.Robot.release_stop`.
``drone_arm``, ``drone_takeoff``, ``drone_rtl``, and ``release_stop`` are
not registered. :func:`ohho.safety.drone_mcp_allowed` is checked again
before any flight call.

Velocity is clamped to the drone safety profile before the backend sees it.
A position that is already outside the geofence, or that the clamped
velocity would leave within the command window, is an error: hold is
requested and the setpoint is not sent. Position is not clipped onto the
fence. The sim ceiling is 3 m. A hardware link uses the gate's geofence
ceiling.

Experimental: nothing has flown. Nothing here sets ``flight_ready`` true.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional

from .adapters.errors import AdapterUnavailable
from .aerial import Frame, convert_vector, convert_yaw, finite_number
from .capabilities import AERIAL
from .mcp_tools import (
    HardwareRefused,
    McpTool,
    RobotSession,
    _backend,
    _is_latched,
    _obj,
    _structured_error,
    _text,
    normalize_mcp_transport,
    transport_scheme,
)
from .registry import UnknownRobot, get_spec
from .robot import Robot
from .safety import (
    DRONE_FENCE_ALT_M,
    SafetyGate,
    SafetyProfile,
    clip_horizontal,
    drone_mcp_allowed,
    fence_violation,
    profile_for,
)
from .schema import Telemetry, clamp

# Bound at registration. ``build_drone_tools`` refuses to start if D-04
# stops allowing one of these actions.
_BOUND_ACTIONS: tuple[tuple[str, str], ...] = (
    ("drone_hold", "hold"),
    ("drone_land", "land"),
    ("drone_velocity", "offboard_velocity"),
)

DRONE_TOOL_NAMES: tuple[str, ...] = (
    "drone_connect",
    "drone_telemetry",
    "drone_velocity",
    "drone_hold",
    "drone_land",
)

_DEFAULT_ROBOT = "px4-sitl"
_FRAMES = ("LOCAL_NED", "LOCAL_ENU", "BODY_FRD")

_NOT_ARMED = (
    "Hardware link is disarmed, so the setpoint was not sent. Ask the human "
    "to arm the safety gate in their own terminal (OHHO_ARM_HARDWARE=1, then "
    "the typed confirmation). This MCP process cannot arm: stdin is the "
    "JSON-RPC pipe, not an interactive terminal. There is no drone_arm or "
    "drone_takeoff tool."
)
_LATCHED = (
    "Emergency stop is latched, so the setpoint was not sent. This MCP "
    "server cannot call release_stop. Ask the human to release the e-stop "
    "in their own terminal."
)
_FENCE = (
    "Setpoint refused: it would leave the geofence. Hold was requested. "
    "Velocity may be clamped; position is not clipped onto the fence. "
    "This tool cannot change the fence, arm, or take off."
)


def _default_robot(session: RobotSession) -> str:
    """Prefer the session robot when it is already an aircraft."""
    try:
        spec = get_spec(session.robot_id)
    except UnknownRobot:
        return _DEFAULT_ROBOT
    if spec.has(AERIAL):
        return spec.id
    return _DEFAULT_ROBOT


def _result_backend(session: RobotSession, uri: Optional[str] = None) -> str:
    bot = session.robot
    if bot is not None:
        return _backend(bot)
    text = uri if uri is not None else session.transport
    scheme = transport_scheme(text or "sim://")
    return "sim" if scheme == "sim" else "hardware"


def _ok(
    session: RobotSession,
    payload: dict[str, Any],
    *,
    uri: Optional[str] = None,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    body = dict(payload)
    body["backend"] = backend or _result_backend(session, uri)
    return _text(body)


def _fail(
    session: RobotSession,
    payload: dict[str, Any],
    *,
    uri: Optional[str] = None,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    body = dict(payload)
    body["backend"] = backend or _result_backend(session, uri)
    return _structured_error(body)


def _gate(bot: Robot) -> Optional[SafetyGate]:
    transport = bot.transport
    if isinstance(transport, SafetyGate):
        return transport
    return None


def _limits(bot: Robot) -> tuple[SafetyProfile, float]:
    """Profile plus the altitude ceiling this tool will enforce.

    Sim uses the drone profile ceiling, and never a number above the 3 m
    sim default. Hardware uses the gate geofence ceiling, which a safety
    config may set independently of that default.
    """
    gate = _gate(bot)
    profile = gate.profile if gate is not None else profile_for(bot.spec.id, bot.spec)
    if bot.simulated:
        ceiling = min(float(profile.fence_max_alt_m), float(DRONE_FENCE_ALT_M))
    else:
        ceiling = float(profile.fence_max_alt_m)
    if ceiling != profile.fence_max_alt_m:
        profile = dataclasses.replace(
            profile,
            fence_max_alt_m=ceiling,
            takeoff_alt_m=min(float(profile.takeoff_alt_m), ceiling),
        )
    return profile, ceiling


def _caps(bot: Robot, profile: SafetyProfile, ceiling: float) -> dict[str, float]:
    horizontal = float(profile.max_horizontal)
    vertical = float(profile.max_vz)
    yaw = float(profile.max_wz)
    if bot.spec.max_lin > 0:
        horizontal = min(horizontal, float(bot.spec.max_lin))
    climb = float(getattr(bot.spec, "max_climb", 0.0) or 0.0)
    if climb > 0:
        vertical = min(vertical, climb)
    if bot.spec.max_ang > 0:
        yaw = min(yaw, float(bot.spec.max_ang))
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "yaw_rate": yaw,
        "altitude_m": ceiling,
        "radius_m": float(profile.fence_radius_m),
    }


def _sample(bot: Robot) -> Telemetry:
    """Current transport snapshot.

    ``Robot.telemetry`` keeps the last pushed sample. Flight commands update
    the backend immediately and only the next ``step`` pushes again, so a
    result that reported the cache would still show the previous mode.
    """
    return bot.transport.read()


def _drone_dict(bot: Robot) -> dict[str, Any]:
    sample = _sample(bot)
    drone = sample.custom.get("drone") if sample.custom else None
    if isinstance(drone, dict):
        return drone
    return {}


def _summary(bot: Robot) -> dict[str, Any]:
    drone = _drone_dict(bot)
    raw_pose = drone.get("pose")
    pose: dict[str, Any] = raw_pose if isinstance(raw_pose, dict) else {}
    frame_name = drone.get("pose_frame")
    altitude: Any = None
    if isinstance(frame_name, str) and "z" in pose:
        try:
            frame = Frame(frame_name)
            z = float(pose["z"])
        except (TypeError, ValueError):
            frame = None
            z = 0.0
        if frame is Frame.LOCAL_NED:
            altitude = -z
        elif frame is Frame.LOCAL_ENU:
            altitude = z
    return {
        "armed": drone.get("armed"),
        "flight_mode": drone.get("flight_mode"),
        "landed_state": drone.get("landed_state"),
        "altitude_m": altitude,
        "pose_frame": frame_name if isinstance(frame_name, str) else None,
        "flight_ready": drone.get("flight_ready", False),
    }


def _pose(
    bot: Robot,
) -> Optional[tuple[float, float, float, float, Frame]]:
    drone = _drone_dict(bot)
    pose = drone.get("pose")
    frame_name = drone.get("pose_frame")
    if not isinstance(pose, dict) or not isinstance(frame_name, str):
        return None
    try:
        frame = Frame(frame_name)
        x = finite_number("x", pose.get("x"))
        y = finite_number("y", pose.get("y"))
        z = finite_number("z", pose.get("z"))
        yaw = finite_number("yaw", pose.get("yaw", 0.0))
    except (TypeError, ValueError):
        return None
    return x, y, z, yaw, frame


def _body_to_local(
    vx: float, vy: float, vz: float, yaw: float, pose_frame: Frame
) -> tuple[float, float, float]:
    """BODY_FRD velocity into the pose frame.

    Yaw matches ``DroneSimTransport``: clockwise from north in ``LOCAL_NED``.
    """
    yaw_ned = yaw
    if pose_frame is Frame.LOCAL_ENU:
        yaw_ned = convert_yaw(yaw, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED)
    elif pose_frame is not Frame.LOCAL_NED:
        raise ValueError("frame")
    c = math.cos(yaw_ned)
    s = math.sin(yaw_ned)
    north = vx * c - vy * s
    east = vx * s + vy * c
    if pose_frame is Frame.LOCAL_NED:
        return north, east, vz
    return east, north, -vz


def _velocity_in_pose_frame(
    vx: float,
    vy: float,
    vz: float,
    source: Frame,
    pose_frame: Frame,
    yaw: float,
) -> tuple[float, float, float]:
    if source is Frame.BODY_FRD:
        return _body_to_local(vx, vy, vz, yaw, pose_frame)
    if pose_frame is Frame.BODY_FRD:
        raise ValueError("frame")
    return convert_vector(vx, vy, vz, source=source, target=pose_frame)


def _projection(
    pose: tuple[float, float, float, float, Frame],
    vx: float,
    vy: float,
    vz: float,
    frame: Frame,
    profile: SafetyProfile,
) -> Optional[tuple[str, bool]]:
    """Why the pose is outside, or why the command window would leave it.

    ``(reason, projected)``. ``projected`` is false when the vehicle is
    already outside. None means the clamped velocity stays inside.
    """
    x, y, z, yaw, pose_frame = pose
    current = fence_violation(x, y, z, pose_frame, profile)
    if current is not None:
        return current, False
    try:
        pvx, pvy, pvz = _velocity_in_pose_frame(vx, vy, vz, frame, pose_frame, yaw)
    except (TypeError, ValueError):
        return "frame", True
    dt = float(profile.cmd_timeout_s)
    future = fence_violation(
        x + pvx * dt, y + pvy * dt, z + pvz * dt, pose_frame, profile
    )
    if future is not None:
        return future, True
    return None


def _clamp_velocity(
    vx: float, vy: float, vz: float, yaw_rate: float, caps: dict[str, float]
) -> dict[str, float]:
    cx, cy = clip_horizontal(vx, vy, caps["horizontal"])
    return {
        "vx": cx,
        "vy": cy,
        "vz": clamp(vz, -caps["vertical"], caps["vertical"]),
        "yaw_rate": clamp(yaw_rate, -caps["yaw_rate"], caps["yaw_rate"]),
    }


def _changed(requested: dict[str, float], applied: dict[str, float]) -> bool:
    return any(abs(requested[key] - applied[key]) > 1e-9 for key in requested)


def _parse_velocity(
    params: dict[str, Any],
) -> tuple[float, float, float, float, Frame]:
    raw = params.get("frame")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "frame is required (LOCAL_NED, LOCAL_ENU, or BODY_FRD). "
            "There is no default frame."
        )
    try:
        frame = Frame(raw.strip())
    except ValueError as exc:
        raise ValueError("frame must be LOCAL_NED, LOCAL_ENU, or BODY_FRD") from exc
    numbers: list[float] = []
    for name in ("vx", "vy", "vz", "yaw_rate"):
        if name not in params or params[name] is None:
            numbers.append(0.0)
            continue
        numbers.append(finite_number(name, params[name]))
    return numbers[0], numbers[1], numbers[2], numbers[3], frame


def _require_aircraft(
    session: RobotSession, *, always: bool = False
) -> Optional[dict[str, Any]]:
    """None when a drone session may proceed.

    ``always`` is land: a hardware link is still accepted when the server
    was started without ``--allow-hardware``. Velocity, hold, and telemetry
    are not.
    """
    bot = session.robot
    if bot is None:
        return _fail(
            session,
            {
                "error": "not_connected",
                "message": "not connected — call drone_connect first",
            },
        )
    if not bot.has(AERIAL):
        return _fail(
            session,
            {
                "error": "not_aerial",
                "robot": bot.spec.id,
                "message": f"{bot.spec.id} has no aerial capability",
            },
        )
    if always or bot.simulated or session.allow_hardware:
        return None
    return _fail(
        session,
        {
            "error": "hardware_refused",
            "reason": "sim_only",
            "message": (
                "Drone tools only open sim:// unless a human started the "
                "server with --allow-hardware. No tool argument or "
                "environment variable can enable hardware."
            ),
        },
    )


def _refusal_message(reason: str) -> str:
    if reason == "not_armed":
        return _NOT_ARMED
    if reason == "estop_latched":
        return _LATCHED
    return f"setpoint refused ({reason})"


def _call_flight(bot: Robot, action: str, fn: Callable[[], object]) -> Optional[str]:
    """Forward one allowlisted flight action. Return a gate refusal, or None.

    A denied action is not forwarded. ``"denied"`` is that refusal.
    """
    if not drone_mcp_allowed(action):
        return "denied"
    gate = _gate(bot)
    if gate is not None:
        gate.last_error = None
    fn()
    if gate is not None and gate.last_error:
        return gate.last_error
    return None


def _robot_arg(params: dict[str, Any], default: str) -> str:
    for key in ("robot", "robotId"):
        value = params.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def build_drone_tools(session: RobotSession) -> list[McpTool]:
    """Drone tools for one :class:`~ohho.mcp_tools.RobotSession`.

    Registration fails if ``hold``, ``land``, or ``offboard_velocity`` is
    no longer allowed by :func:`ohho.safety.drone_mcp_allowed`.
    """
    for tool_name, action in _BOUND_ACTIONS:
        if not drone_mcp_allowed(action):
            raise ValueError(
                f"refusing to register {tool_name}: "
                f"drone_mcp_allowed({action!r}) is false"
            )

    def drone_connect(params: dict[str, Any]) -> dict[str, Any]:
        # Tool arguments cannot set allow_hardware. The session flag is the
        # CLI flag. This handler does not call arm() or arm_flight().
        robot_id = _robot_arg(params, _default_robot(session))
        runtime = str(params.get("runtime") or session.runtime)
        if "transport" not in params or params.get("transport") is None:
            raw = session.transport or "sim://"
        else:
            raw = str(params.get("transport"))
        try:
            transport = normalize_mcp_transport(
                raw, allow_hardware=session.allow_hardware
            )
        except HardwareRefused as exc:
            return _fail(
                session,
                {
                    "error": "hardware_refused",
                    "reason": "sim_only",
                    "transport": exc.uri,
                    "message": str(exc),
                },
                backend="hardware",
            )
        except ValueError as exc:
            scheme = transport_scheme(raw)
            return _fail(
                session,
                {"error": "bad_arguments", "transport": raw, "message": str(exc)},
                backend="sim" if scheme == "sim" else "hardware",
            )
        session.disconnect()
        session.robot_id = robot_id
        session.transport = transport
        session.runtime = runtime
        try:
            bot = session.connect()
        except (UnknownRobot, AdapterUnavailable, ValueError) as exc:
            return _fail(
                session,
                {"error": "connect_failed", "message": str(exc)},
                uri=transport,
            )
        if not bot.has(AERIAL):
            session.disconnect()
            return _fail(
                session,
                {
                    "error": "not_aerial",
                    "robot": robot_id,
                    "message": f"{robot_id} has no aerial capability",
                },
                uri=transport,
            )
        gate = _gate(bot)
        summary = _summary(bot)
        return _ok(
            session,
            {
                "connected": True,
                "robot": bot.spec.id,
                "transport": transport,
                "protocol": bot.transport.protocol,
                "aerial": True,
                "gate_armed": bool(gate.armed) if gate is not None else False,
                "autopilot_armed": summary["armed"] is True,
                "drone": summary,
            },
        )

    def drone_telemetry(_: dict[str, Any]) -> dict[str, Any]:
        refused = _require_aircraft(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        sample = _sample(bot)
        payload: dict[str, Any] = {
            "timestamp": sample.timestamp,
            "battery": sample.battery,
            "drone": _drone_dict(bot) or None,
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
        return _ok(session, payload)

    def drone_velocity(params: dict[str, Any]) -> dict[str, Any]:
        refused = _require_aircraft(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        try:
            vx, vy, vz, yaw_rate, frame = _parse_velocity(params)
        except (TypeError, ValueError) as exc:
            return _fail(session, {"error": "bad_arguments", "message": str(exc)})
        profile, ceiling = _limits(bot)
        caps = _caps(bot, profile, ceiling)
        requested = {"vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate}
        applied = _clamp_velocity(vx, vy, vz, yaw_rate, caps)
        report: dict[str, Any] = {
            "requested": requested,
            "applied": applied,
            "clamped": _changed(requested, applied),
            "cap": caps,
            "frame": frame.value,
            "window_s": profile.cmd_timeout_s,
        }
        if _is_latched(bot):
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": "estop_latched",
                    "sent": False,
                    "message": _LATCHED,
                    **report,
                },
            )
        pose = _pose(bot)
        if pose is not None:
            violation = _projection(
                pose, applied["vx"], applied["vy"], applied["vz"], frame, profile
            )
            if violation is not None:
                reason, projected = violation
                hold_reason = _call_flight(bot, "hold", bot.hold)
                fence_body: dict[str, Any] = {
                    "error": "geofence",
                    "reason": reason,
                    "projected": projected,
                    "sent": False,
                    "hold": True,
                    "hold_sent": hold_reason is None,
                    "message": _FENCE,
                    "drone": _summary(bot),
                    **report,
                }
                if hold_reason is not None:
                    fence_body["hold_reason"] = hold_reason
                return _fail(session, fence_body)
        if not bot.simulated:
            gate = _gate(bot)
            if gate is None or not gate.armed:
                return _fail(
                    session,
                    {
                        "error": "motion_refused",
                        "reason": "not_armed",
                        "sent": False,
                        "message": _NOT_ARMED,
                        **report,
                    },
                )
        refusal = _call_flight(
            bot,
            "offboard_velocity",
            lambda: bot.offboard_velocity(
                applied["vx"],
                applied["vy"],
                applied["vz"],
                applied["yaw_rate"],
                frame,
            ),
        )
        if refusal is not None:
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": refusal,
                    "sent": False,
                    "message": _refusal_message(refusal),
                    "drone": _summary(bot),
                    **report,
                },
            )
        return _ok(
            session,
            {"ok": True, "sent": True, "drone": _summary(bot), **report},
        )

    def drone_hold(_: dict[str, Any]) -> dict[str, Any]:
        refused = _require_aircraft(session)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        if _is_latched(bot):
            return _fail(
                session,
                {
                    "error": "motion_refused",
                    "reason": "estop_latched",
                    "sent": False,
                    "action": "hold",
                    "message": _LATCHED,
                    "drone": _summary(bot),
                },
            )
        refusal = _call_flight(bot, "hold", bot.hold)
        payload: dict[str, Any] = {
            "ok": refusal is None,
            "action": "hold",
            "sent": refusal is None,
            "drone": _summary(bot),
        }
        if refusal is not None:
            payload["error"] = "motion_refused"
            payload["reason"] = refusal
            payload["message"] = _refusal_message(refusal)
            return _fail(session, payload)
        return _ok(session, payload)

    def drone_land(_: dict[str, Any]) -> dict[str, Any]:
        # Always attempted, including when --allow-hardware was not passed.
        # The handler does not arm and does not clear an e-stop.
        refused = _require_aircraft(session, always=True)
        if refused is not None:
            return refused
        bot = session.robot
        assert bot is not None
        refusal = _call_flight(bot, "land", bot.land)
        payload = {
            "ok": refusal is None,
            "action": "land",
            "sent": refusal is None,
            "latched": _is_latched(bot),
            "drone": _summary(bot),
        }
        if refusal is not None:
            return _fail(
                session,
                {
                    **payload,
                    "error": "land_refused",
                    "reason": refusal,
                    "message": (
                        "land was not forwarded. This tool does not arm the "
                        "gate and does not release an e-stop."
                    ),
                },
            )
        return _ok(session, payload)

    def _tool(
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        read_only: bool,
    ) -> McpTool:
        product = "connect" if read_only or name == "drone_connect" else "pilot"
        return McpTool(name, description, schema, handler, product, read_only)

    frame_schema = {
        "type": "string",
        "enum": list(_FRAMES),
        "description": "Required. LOCAL_NED, LOCAL_ENU, or BODY_FRD. No default.",
    }
    tools = [
        _tool(
            "drone_connect",
            "Connect an aircraft. sim:// unless the human started the server "
            "with --allow-hardware. Never arms the gate or the autopilot.",
            _obj(
                {
                    "robot": {
                        "type": "string",
                        "description": "Aerial robot id (default px4-sitl).",
                    },
                    "transport": {
                        "type": "string",
                        "description": "sim:// unless --allow-hardware. A tool argument cannot enable hardware.",
                    },
                    "runtime": {
                        "type": "string",
                        "description": "native | ros2 | auto",
                    },
                }
            ),
            drone_connect,
            read_only=False,
        ),
        _tool(
            "drone_telemetry",
            "Read aircraft telemetry. Does not send a setpoint.",
            _obj({}),
            drone_telemetry,
            read_only=True,
        ),
        _tool(
            "drone_velocity",
            "Velocity setpoint, clamped to the drone caps before the backend. "
            "A setpoint that would leave the geofence is refused and hold is "
            "sent. Hardware motion waits until a human has armed from a TTY. "
            "This tool does not arm.",
            _obj(
                {
                    "vx": {"type": "number"},
                    "vy": {"type": "number"},
                    "vz": {
                        "type": "number",
                        "description": "In frame. Positive is down in LOCAL_NED and BODY_FRD.",
                    },
                    "yaw_rate": {"type": "number", "description": "rad/s"},
                    "frame": frame_schema,
                },
                ["frame"],
            ),
            drone_velocity,
            read_only=False,
        ),
        _tool(
            "drone_hold",
            "Request autopilot hold. Does not arm.",
            _obj({}),
            drone_hold,
            read_only=False,
        ),
        _tool(
            "drone_land",
            "Request land. Always allowed, including when hardware was not "
            "enabled. Does not arm and does not clear an e-stop.",
            _obj({}),
            drone_land,
            read_only=False,
        ),
    ]
    names = tuple(tool.name for tool in tools)
    if names != DRONE_TOOL_NAMES:
        raise RuntimeError(f"drone tool order drifted: {names}")
    return tools
