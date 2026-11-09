"""WebSocket server for simulated and ROS 2 robot telemetry and teleoperation.

Exposes one JSON WebSocket session. The in-process ``sim://`` backend needs no
ROS. The ``robot`` backend connects through :class:`~ohho.robot.Robot` (and
therefore :class:`~ohho.safety.SafetyGate` on a non-sim transport). The
``ros2`` backend bridges that same session to rosbridge topics the Connect
console already uses (``/cmd_vel``, ``/odom``, joint state, front camera,
``/emergency_stop``).

Client JSON:
    - ``velocity``: Twist-like ``{"linear": {"x", "y"}, "angular": {"z"}}``
      or Connect fields ``linearX`` / ``linearY`` / ``angularZ``
    - ``joints``: ``{"<joint_name>": radians, ...}``
    - ``deadman``: boolean. Missing, false, or any other value is not held.
    - ``estop``: boolean. ``true`` latches; only ``false`` clears it.
    - ``mode`` (optional): ``"stand"`` | ``"lie_down"`` | ``"recover"`` — mapped to
      allowlisted high-level actions on a connected :class:`~ohho.robot.Robot`.
    - ``modes`` (optional): ``{"request": "<posture>", "confirm": "<phrase>"}``.
      ``request`` is a ``humanoid.posture`` value the spec declares
      (``prepare``, ``walk``, ``damping``, ``custom``, ``get_up``, ``lie_down``).
      ``walk_mode`` is accepted when ``walk`` is declared, and ``damp`` when
      ``damping`` is declared. The check is the capability and that list.
      ``backend=sim`` does not need ``confirm``. ``backend=robot`` on a URI that
      is not an explicit sim target does: ``confirm`` must be the arming phrase,
      and ``OHHO_ARM_HARDWARE=1`` must already be set. This module never sets
      that variable and never arms the gate.

Server JSON:
    - ``odom``: ``{"x": float, "y": float, "yaw": float}``
    - ``joints``: ``{"<joint_name>": float, ...}``
    - ``protocolVersion``: ``ohho.PROTOCOL_VERSION``
    - ``backend``: ``"sim"``, ``"ros2"``, or ``"robot"``
    - ``posture`` / ``battery`` (optional, when the robot reports them)
    - ``modes``: the last posture decision, when a request was seen
    - ``camera``: base64 JPEG when a front-camera frame has arrived, otherwise
      ``no_camera: true`` (never both)

Safety:
    - Velocity is zero unless the deadman is held, e-stop is clear, at least
      one client is connected, and a client message arrived less than 300 ms
      ago.
    - A missing ``deadman`` field commands zero immediately.
    - A dropped socket commands zero immediately when no client remains, and
      always within 300 ms.
    - ``estop`` latches until a message with ``estop: false`` arrives.
    - Silence, a released deadman, or e-stop zeros velocity and leaves the
      current posture. That path does not damp. A posture request is applied
      only while the deadman is held and e-stop is clear.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from . import capabilities as caps
from .humanoid_policy import explicit_sim_target
from .protocol import PROTOCOL_VERSION
from .registry import RobotSpec, UnknownRobot, get_spec
from .safety import ARM_ENV, ARM_PHRASE
from .schema import Telemetry

# Commanded velocity is zero once this long has passed with no client message.
DEADMAN_TIMEOUT_S = 0.300
# Alias used by posture tests. Silence is strictly longer than this interval.
SILENCE_S = DEADMAN_TIMEOUT_S

# Request token -> (declared posture mode it requires, transport action).
# ``walk_mode`` is the section 6.1 name for the declared mode ``walk``.
_MODE_REQUESTS: dict[str, tuple[str, str]] = {
    "prepare": ("prepare", "prepare"),
    "walk": ("walk", "walk_mode"),
    "walk_mode": ("walk", "walk_mode"),
    "damping": ("damping", "damp"),
    "damp": ("damping", "damp"),
    "custom": ("custom", "custom"),
    "get_up": ("get_up", "get_up"),
    "lie_down": ("lie_down", "lie_down"),
}

NOT_IMPLEMENTED_BACKENDS = ("isaac", "mujoco")

# Topics the Connect console Gazebo row publishes and subscribes.
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
JOINT_STATE_TOPIC = "/joint_states"
ARM_JOINT_STATE_TOPIC = "/arm/joint_states"
ARM_JOINT_COMMAND_TOPIC = "/arm/joint_commands"
ESTOP_TOPIC = "/emergency_stop"
FRONT_CAMERA_TOPIC = "/camera/front/image_raw"

ROS_SUBSCRIPTIONS: tuple[tuple[str, str], ...] = (
    (ODOM_TOPIC, "nav_msgs/Odometry"),
    (JOINT_STATE_TOPIC, "sensor_msgs/JointState"),
    (ARM_JOINT_STATE_TOPIC, "sensor_msgs/JointState"),
    (FRONT_CAMERA_TOPIC, "sensor_msgs/Image"),
    (f"{FRONT_CAMERA_TOPIC}/compressed", "sensor_msgs/CompressedImage"),
)

# Browser / VR teleop posture commands → SafetyGate allowlisted sport actions.
TELEOP_CLIENT_MODES: dict[str, str] = {
    "stand": "stand_up",
    "lie_down": "stand_down",
    "recover": "recovery_stand",
}


@dataclass(frozen=True)
class ModeDecision:
    """Whether a ``modes`` request may be sent, and the action to send."""

    accepted: bool
    action: str = ""
    reason: str = ""


def declared_posture_modes(spec: RobotSpec | None) -> frozenset[str]:
    """Posture modes the spec declares on ``humanoid.posture``.

    An absent capability, or a capability with no ``modes`` list, declares
    nothing. The robot id is not consulted.
    """
    if spec is None or not spec.has(caps.HUMANOID_POSTURE):
        return frozenset()
    params = spec.capability_params
    if not isinstance(params, Mapping):
        return frozenset()
    raw = params.get(caps.HUMANOID_POSTURE)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        return frozenset()
    modes = raw.get("modes")
    if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
        return frozenset()
    return frozenset(mode for mode in modes if isinstance(mode, str) and mode)


def mode_confirmation_required(backend: str, uri: str | None) -> bool:
    """True for a robot backend whose URI is not an explicit sim target.

    ``backend=sim`` never needs the phrase. ``sim://``, ``simulated://`` and a
    loopback humanoid-mujoco URI with ``sim_target=True`` are sim targets.
    """
    if backend != "robot":
        return False
    return not explicit_sim_target(uri)


def decide_mode_request(
    spec: RobotSpec | None,
    request: object,
    *,
    backend: str,
    uri: str | None,
    confirm: object,
    env: Mapping[str, str],
) -> ModeDecision:
    """Accept a posture request or refuse it.

    The request must name a mode the spec declares. On a non-sim robot URI the
    arming phrase and ``OHHO_ARM_HARDWARE=1`` must already be present. This
    function only reads ``env``.
    """
    if spec is None or not spec.has(caps.HUMANOID_POSTURE):
        return ModeDecision(False, reason="capability")
    if backend not in ("sim", "robot"):
        return ModeDecision(False, reason="backend")
    if not isinstance(request, str) or not request.strip():
        return ModeDecision(False, reason="request")
    name = request.strip()
    mapped = _MODE_REQUESTS.get(name)
    declared = declared_posture_modes(spec)
    if mapped is None or (name not in declared and mapped[0] not in declared):
        return ModeDecision(False, reason="undeclared")
    if mode_confirmation_required(backend, uri):
        if confirm != ARM_PHRASE:
            return ModeDecision(False, reason="confirm")
        if env.get(ARM_ENV) != "1":
            return ModeDecision(False, reason="arm_env")
    return ModeDecision(True, action=mapped[1])


class SimServeUnavailable(RuntimeError):
    """Raised when sim-serve requirements are missing or a backend is not implemented."""


def backend_entry(backend: str) -> str:
    """Return ``available`` or ``not-implemented`` for a sim-serve backend name.

    Isaac Sim and MuJoCo have no runtime here. Callers must not import those
    packages; this function only classifies the name. ``robot`` is available
    and connects through :class:`~ohho.robot.Robot`.
    """
    if backend in ("sim", "ros2", "robot"):
        return "available"
    return "not-implemented"


def _coerce_bool(value: Any) -> Optional[bool]:
    """Parse a JSON boolean. Strings ``true``/``false`` are accepted; other values are not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
    return None


# A Foxglove bridge binds 8765 (robot.launch.py, simulation.launch.py).
DEFAULT_PORT = 8790


def parse_velocity(val: Any) -> tuple[float, float, float]:
    """Extract (vx, vy, w) from a velocity object.

    Supports nested ROS Twist style (``{"linear": {"x": ...}, "angular": {"z": ...}}``),
    dotted keys (``{"linear.x": ...}``), flat keys (``{"linear_x": ...}``),
    Connect console fields (``{"linearX": ...}``), or coordinate keys
    (``{"x": ..., "y": ..., "z": ...}``).
    """
    if not isinstance(val, dict):
        return 0.0, 0.0, 0.0

    vx, vy, w = 0.0, 0.0, 0.0

    # Nested ROS-style: {"linear": {"x": ...}, "angular": {"z": ...}}
    if "linear" in val or "angular" in val:
        lin = val.get("linear")
        if isinstance(lin, dict):
            vx = float(lin.get("x", 0.0))
            vy = float(lin.get("y", 0.0))
        elif isinstance(lin, (int, float)) and not isinstance(lin, bool):
            vx = float(lin)

        ang = val.get("angular")
        if isinstance(ang, dict):
            w = float(ang.get("z", 0.0))
        elif isinstance(ang, (int, float)) and not isinstance(ang, bool):
            w = float(ang)

    # Dotted keys
    if "linear.x" in val:
        vx = float(val["linear.x"])
    if "linear.y" in val:
        vy = float(val["linear.y"])
    if "angular.z" in val:
        w = float(val["angular.z"])

    # Flat underscore keys
    if "linear_x" in val:
        vx = float(val["linear_x"])
    if "linear_y" in val:
        vy = float(val["linear_y"])
    if "angular_z" in val:
        w = float(val["angular_z"])

    # Connect console Velocity fields
    if "linearX" in val:
        vx = float(val["linearX"])
    if "linearY" in val:
        vy = float(val["linearY"])
    if "angularZ" in val:
        w = float(val["angularZ"])

    # Short coordinate keys
    if "x" in val and "linear" not in val:
        vx = float(val["x"])
    if "y" in val and "linear" not in val:
        vy = float(val["y"])
    if "z" in val and "angular" not in val:
        w = float(val["z"])

    return vx, vy, w


@dataclass(frozen=True)
class ClientCommand:
    """One client JSON message, normalized.

    ``deadman`` is true only when the field is present and true. A missing
    field is not a held deadman. ``estop`` is None when the field is absent
    so a latch is neither set nor cleared. ``velocity`` / ``joints`` are None
    when those fields are absent.
    """

    deadman: bool
    estop: Optional[bool]
    velocity: Optional[tuple[float, float, float]]
    joints: Optional[dict[str, float]]


def parse_client_command(data: Any) -> ClientCommand:
    """Normalize a teleop session message from the Connect console or a test."""
    if not isinstance(data, dict):
        return ClientCommand(deadman=False, estop=None, velocity=None, joints=None)

    if "deadman" in data:
        deadman = _coerce_bool(data.get("deadman")) is True
    else:
        deadman = False

    estop: Optional[bool] = None
    if "estop" in data:
        estop = _coerce_bool(data.get("estop"))

    velocity: Optional[tuple[float, float, float]] = None
    if "velocity" in data:
        velocity = parse_velocity(data.get("velocity"))

    joints: Optional[dict[str, float]] = None
    raw_joints = data.get("joints")
    if isinstance(raw_joints, dict):
        parsed: dict[str, float] = {}
        for name, position in raw_joints.items():
            if isinstance(position, bool) or not isinstance(position, (int, float)):
                continue
            parsed[str(name)] = float(position)
        joints = parsed

    return ClientCommand(deadman=deadman, estop=estop, velocity=velocity, joints=joints)


def gated_velocity(
    target: tuple[float, float, float],
    *,
    deadman: bool,
    estopped: bool,
    client_count: int,
    command_window_open: bool,
    now: float,
    last_msg_time: float,
    timeout_s: float = DEADMAN_TIMEOUT_S,
) -> tuple[float, float, float]:
    """Return ``target`` only while teleop is allowed; otherwise zero.

    Motion requires a held deadman, a clear e-stop, a connected client, an open
    command window, and a client message newer than ``timeout_s`` (300 ms).
    """
    fresh = last_msg_time > 0.0 and (now - last_msg_time) < timeout_s
    if (
        not deadman
        or estopped
        or client_count <= 0
        or not command_window_open
        or not fresh
    ):
        return (0.0, 0.0, 0.0)
    return (float(target[0]), float(target[1]), float(target[2]))


def build_telemetry_message(
    *,
    odom: dict[str, float],
    joints: dict[str, float],
    backend: str,
    camera: Optional[str],
) -> dict[str, Any]:
    """Server telemetry. ``camera`` and ``no_camera`` are mutually exclusive."""
    msg: dict[str, Any] = {
        "protocolVersion": PROTOCOL_VERSION,
        "odom": {
            "x": float(odom.get("x", 0.0)),
            "y": float(odom.get("y", 0.0)),
            "yaw": float(odom.get("yaw", 0.0)),
        },
        "joints": {str(name): float(position) for name, position in joints.items()},
        "backend": backend,
    }
    if camera:
        msg["camera"] = camera
    else:
        msg["no_camera"] = True
    return msg


def yaw_from_quaternion(qw: float, qz: float) -> float:
    """Planar yaw from a quaternion that is mostly a rotation about z."""
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * (qz * qz))


def odom_from_ros(msg: Any) -> Optional[dict[str, float]]:
    """Read ``x``, ``y``, ``yaw`` from a ``nav_msgs/Odometry`` rosbridge payload."""
    if not isinstance(msg, dict):
        return None
    pose_wrap = msg.get("pose")
    pose = pose_wrap.get("pose") if isinstance(pose_wrap, dict) else None
    if not isinstance(pose, dict):
        pose = {}
    pos = pose.get("position") if isinstance(pose.get("position"), dict) else {}
    ori = pose.get("orientation") if isinstance(pose.get("orientation"), dict) else {}
    try:
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        qw = float(ori.get("w", 1.0))
        qz = float(ori.get("z", 0.0))
    except (TypeError, ValueError):
        return None
    return {"x": x, "y": y, "yaw": yaw_from_quaternion(qw, qz)}


def joints_from_ros(msg: Any) -> dict[str, float]:
    """Read a name→position map from a ``sensor_msgs/JointState`` payload."""
    if not isinstance(msg, dict):
        return {}
    names = msg.get("name") or []
    positions = msg.get("position") or []
    if not isinstance(names, list) or not isinstance(positions, list):
        return {}
    parsed: dict[str, float] = {}
    for name, position in zip(names, positions):
        if isinstance(position, bool) or not isinstance(position, (int, float)):
            continue
        parsed[str(name)] = float(position)
    return parsed


def is_front_camera_topic(topic: str) -> bool:
    """True for ``/camera/front/image_raw`` and encodings under that prefix."""
    return topic == FRONT_CAMERA_TOPIC or topic.startswith(f"{FRONT_CAMERA_TOPIC}/")


def camera_jpeg_from_ros(topic: str, msg: Any) -> Optional[str]:
    """Return base64 image bytes from a front-camera rosbridge message.

    Empty payloads and any other topic return None so the session reports
    ``no_camera`` instead of inventing a frame. Raw ``rgb8``/``bgr8`` arrays
    are encoded as base64; they are not converted into a JPEG container.
    """
    if not isinstance(topic, str) or not is_front_camera_topic(topic):
        return None
    if not isinstance(msg, dict):
        return None
    data = msg.get("data")
    if isinstance(data, str):
        payload = data.strip()
        return payload or None
    if (
        isinstance(data, list)
        and data
        and all(isinstance(item, int) and not isinstance(item, bool) for item in data)
    ):
        raw = bytes(int(item) & 0xFF for item in data)
        if not raw:
            return None
        return base64.b64encode(raw).decode("ascii")
    return None


def cmd_vel_rosbridge(vx: float, vy: float, w: float) -> dict[str, Any]:
    """rosbridge publish for ``/cmd_vel``, matching the Connect console Twist."""
    return {
        "op": "publish",
        "topic": CMD_VEL_TOPIC,
        "type": "geometry_msgs/Twist",
        "msg": {
            "linear": {"x": float(vx), "y": float(vy), "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(w)},
        },
    }


def joint_command_rosbridge(joints: dict[str, float]) -> dict[str, Any]:
    """rosbridge publish for ``/arm/joint_commands``."""
    return {
        "op": "publish",
        "topic": ARM_JOINT_COMMAND_TOPIC,
        "type": "sensor_msgs/JointState",
        "msg": {
            "name": list(joints.keys()),
            "position": [float(position) for position in joints.values()],
            "velocity": [],
            "effort": [],
        },
    }


def estop_rosbridge(estopped: bool) -> dict[str, Any]:
    """rosbridge publish for ``/emergency_stop``."""
    return {
        "op": "publish",
        "topic": ESTOP_TOPIC,
        "type": "std_msgs/Bool",
        "msg": {"data": bool(estopped)},
    }


def teleop_mode_to_action(mode: Any) -> Optional[str]:
    """Map a client ``mode`` string to a sport action name, or ``None`` if unknown."""
    if not isinstance(mode, str) or not mode.strip():
        return None
    return TELEOP_CLIENT_MODES.get(mode.strip().lower())


def posture_from_telemetry(t: Telemetry) -> Optional[str]:
    """Best-effort human-readable posture for teleop UI (robot-agnostic)."""
    custom = t.custom or {}
    if "posture" in custom:
        return str(custom["posture"])
    body_h = custom.get("body_height")
    if isinstance(body_h, (int, float)):
        if body_h < 0.12:
            return "lying"
        if body_h > 0.2:
            return "standing"
    mode = custom.get("mode")
    if mode is not None:
        return str(mode)
    return None


def telemetry_extras(t: Telemetry) -> dict[str, Any]:
    """Optional fields merged into WebSocket telemetry broadcasts."""
    out: dict[str, Any] = {}
    if t.battery is not None:
        out["battery"] = float(t.battery)
    posture = posture_from_telemetry(t)
    if posture is not None:
        out["posture"] = posture
    return out


def is_ros2_available() -> bool:
    """True if rclpy or ROS environment or OHHO_ROSBRIDGE is set."""
    import importlib.util

    return (
        importlib.util.find_spec("rclpy") is not None
        or bool(os.environ.get("ROS_DISTRO"))
        or bool(os.environ.get("OHHO_ROSBRIDGE"))
    )


class SimServe:
    """Manages the WebSocket server and the simulation, robot, or rosbridge backend."""

    def __init__(
        self,
        backend: str = "sim",
        robot: str | None = None,
        transport: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        rosbridge: str = "ws://localhost:9090",
        rate_hz: float = 20.0,
        ssl_cert: Optional[str] = None,
        ssl_key: Optional[str] = None,
    ) -> None:
        if backend == "robot" and not transport:
            raise ValueError("backend 'robot' requires a transport URI (--transport)")
        if backend_entry(backend) != "available":
            raise SimServeUnavailable(f"backend '{backend}' is not-implemented")

        self.backend = backend
        if not robot:
            from .plugins import preferred_robot_id

            robot = preferred_robot_id()
        self.robot_id = robot
        self._transport_uri = transport
        self.host = host
        self.port = port
        self.rosbridge_url = rosbridge
        self.rate_hz = rate_hz
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key

        # Session safety state
        self._deadman: bool = False
        self._last_msg_time: float = 0.0
        self._command_window_open: bool = False
        self._estopped: bool = False
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_w: float = 0.0
        self._target_joints: dict[str, float] = {}
        self._applied_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._posture: str = ""
        self._mode_result: dict[str, Any] | None = None
        self._timeout_handle: Optional[asyncio.TimerHandle] = None

        # Telemetry state
        self._latest_odom: dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._latest_joints: dict[str, float] = {}
        self._camera_frame: Optional[str] = None

        # Server infrastructure
        self._server: Any = None
        self._clients: set[Any] = set()
        self._running: bool = False
        self._bot: Any = None
        self._rosbridge_ws: Any = None
        self._rosbridge_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Start backend connections and the WebSocket server."""
        if self.backend == "ros2" and not is_ros2_available():
            raise SimServeUnavailable(
                "backend 'ros2' depends on the [ros2] extra "
                "(source your ROS 2 environment or set OHHO_ROSBRIDGE)."
            )

        try:
            import websockets
        except ImportError:
            raise SimServeUnavailable(
                "sim-serve requires websockets: "
                "pip install 'ohho-os[serve]' or 'ohho-os[sim-serve]'"
            )

        self._loop = asyncio.get_running_loop()
        self._running = True

        if self.backend in ("sim", "robot"):
            from .robot import Robot

            uri = "sim://" if self.backend == "sim" else self._transport_uri
            assert uri is not None
            self._bot = Robot.connect(self.robot_id, transport=uri, runtime="native")
            await self._arm_sim_link()
        elif self.backend == "ros2":
            self._rosbridge_ws = await websockets.connect(self.rosbridge_url)
            for topic, msg_type in ROS_SUBSCRIPTIONS:
                await self._rosbridge_ws.send(
                    json.dumps({"op": "subscribe", "topic": topic, "type": msg_type})
                )
            self._rosbridge_task = asyncio.create_task(self._rosbridge_loop())

        ssl_ctx = None
        if self.ssl_cert and self.ssl_key:
            import ssl
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=self.ssl_cert, keyfile=self.ssl_key)

        self._server = await websockets.serve(self._handle_client, self.host, self.port, ssl=ssl_ctx)
        if self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def _arm_sim_link(self) -> None:
        """Arm a simulator or loopback link. A hardware link stays disarmed."""
        bot = self._bot
        if bot is None or not getattr(bot, "gated", False):
            return
        gate = getattr(bot, "transport", None)
        if getattr(gate, "requires_hardware_arming", True):
            return
        arm = getattr(bot, "arm", None)
        if not callable(arm):
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                if arm():
                    return
            except Exception:
                return
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        """Shut down the server and disconnect backends."""
        self._running = False
        self._command_window_open = False
        self._deadman = False
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        if self._rosbridge_task is not None:
            self._rosbridge_task.cancel()
            try:
                await self._rosbridge_task
            except asyncio.CancelledError:
                pass
            self._rosbridge_task = None

        # Last command is zero, after the broadcast loop can no longer overwrite it.
        await self._send_velocity()

        if self._rosbridge_ws is not None:
            try:
                await self._rosbridge_ws.close()
            except Exception:
                pass
            self._rosbridge_ws = None

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._clients.clear()

        if self._bot is not None:
            try:
                self._bot.disconnect()
            except Exception:
                pass
            self._bot = None

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run the server synchronously until stop_event is set or interrupted."""

        async def _main() -> None:
            await self.start()
            try:
                while self._running:
                    if stop_event is not None and stop_event.is_set():
                        break
                    await asyncio.sleep(0.05)
            finally:
                await self.stop()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            pass

    # ── Client handling ───────────────────────────────────────────────────────

    async def _handle_client(self, websocket: Any, *args: Any) -> None:
        self._clients.add(websocket)
        try:
            await websocket.send(json.dumps(self._telemetry_message()))

            async for raw in websocket:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, dict):
                    self._on_client_message(data)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            if not self._clients:
                self._deadman = False
                self._command_window_open = False
                if self._timeout_handle is not None:
                    self._timeout_handle.cancel()
                    self._timeout_handle = None
                await self._send_velocity()

    def _on_client_message(self, data: dict[str, Any]) -> None:
        command = parse_client_command(data)
        now = time.monotonic()
        self._last_msg_time = now
        # Any well-formed client object refreshes the 300 ms window. The
        # deadman and e-stop gates still decide whether that window can move.
        self._command_window_open = True

        if command.estop is True:
            self._estopped = True
            self._on_estop_changed(True)
        elif command.estop is False:
            self._estopped = False
            self._on_estop_changed(False)

        # Missing deadman is not held, so velocity goes to zero on this message.
        self._deadman = command.deadman

        if command.velocity is not None:
            self._target_vx, self._target_vy, self._target_w = command.velocity

        if command.joints is not None and not self._estopped:
            self._target_joints = command.joints
            self._apply_joints(command.joints)

        if "mode" in data:
            self._apply_teleop_mode(data["mode"])

        if "modes" in data:
            self._on_modes(data["modes"])

        self._arm_timeout()
        vx, vy, w = self.commanded_velocity(now)
        if self.backend in ("sim", "robot") and (
            self._loop is None or not self._loop.is_running()
        ):
            self._drive_sync(vx, vy, w)
        self._schedule_gated_velocity()

    def _arm_timeout(self) -> None:
        """Fire once, 300 ms after the latest client message."""
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        self._timeout_handle = loop.call_later(DEADMAN_TIMEOUT_S, self._on_timeout)

    def _on_timeout(self) -> None:
        self._timeout_handle = None
        self._command_window_open = False
        if self._running and self._loop is not None:
            self._loop.create_task(self._send_velocity())

    def _schedule_gated_velocity(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.create_task(self._send_velocity())

    def commanded_velocity(self, now: float) -> tuple[float, float, float]:
        """Velocity to send now. Silence, a released deadman, or e-stop is zero.

        Silence is strictly longer than :data:`SILENCE_S`. This does not change
        posture and does not consult the client count. The live socket path
        still uses :func:`gated_velocity`.
        """
        silent = (now - self._last_msg_time) > SILENCE_S
        if silent or not self._deadman or self._estopped:
            return 0.0, 0.0, 0.0
        return self._target_vx, self._target_vy, self._target_w

    def _current_spec(self) -> RobotSpec | None:
        bot = self._bot
        spec = getattr(bot, "spec", None) if bot is not None else None
        if isinstance(spec, RobotSpec):
            return spec
        try:
            return get_spec(self.robot_id)
        except UnknownRobot:
            return None

    def _on_modes(self, value: object) -> None:
        """Apply a posture request, or record why it was refused.

        A released deadman or a latched e-stop leaves the current posture
        unchanged and does not damp. The arming environment is only read.
        """
        if self._estopped or not self._deadman:
            reason = "estop" if self._estopped else "deadman"
            self._mode_result = {"accepted": False, "reason": reason}
            return
        if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
            self._mode_result = {"accepted": False, "reason": "modes"}
            return
        decision = decide_mode_request(
            self._current_spec(),
            value.get("request"),
            backend=self.backend,
            uri=self._transport_uri if self.backend == "robot" else "sim://",
            confirm=value.get("confirm"),
            env=os.environ,
        )
        if not decision.accepted:
            self._mode_result = {"accepted": False, "reason": decision.reason}
            return
        bot = self._bot
        if bot is None:
            self._mode_result = {"accepted": False, "reason": "no_robot"}
            return
        from .humanoid import apply_posture

        try:
            apply_posture(bot, decision.action)
        except Exception:
            self._mode_result = {"accepted": False, "reason": "transport"}
            return
        self._posture = decision.action
        self._mode_result = {"accepted": True, "action": decision.action}

    def _current_velocity(self) -> tuple[float, float, float]:
        """Gate read at send time so a queued task cannot revive an old command."""
        if not self._running:
            return (0.0, 0.0, 0.0)
        return gated_velocity(
            (self._target_vx, self._target_vy, self._target_w),
            deadman=self._deadman,
            estopped=self._estopped,
            client_count=len(self._clients),
            command_window_open=self._command_window_open,
            now=time.monotonic(),
            last_msg_time=self._last_msg_time,
        )

    def _telemetry_message(self) -> dict[str, Any]:
        odom, joints = self._get_telemetry()
        msg = build_telemetry_message(
            odom=odom,
            joints=joints,
            backend=self.backend,
            camera=self._camera_frame,
        )
        if self._bot is not None:
            try:
                msg.update(telemetry_extras(self._bot.telemetry()))
            except Exception:
                pass
        if self._mode_result is not None:
            msg["modes"] = dict(self._mode_result)
        return msg

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / max(1.0, self.rate_hz)
        while self._running:
            await self._send_velocity()
            payload = json.dumps(self._telemetry_message())
            if self._clients:
                aws = [ws.send(payload) for ws in list(self._clients)]
                await asyncio.gather(*aws, return_exceptions=True)
            await asyncio.sleep(interval)

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _apply_teleop_mode(self, mode: Any) -> None:
        if self._estopped:
            return
        action = teleop_mode_to_action(mode)
        if action is None or self._bot is None:
            return
        transport = self._bot.transport
        command = getattr(transport, "command", None)
        if callable(command):
            command(action)

    def _drive_sync(self, vx: float, vy: float, w: float) -> None:
        """Send a body velocity on the sim or robot backend. Does not change posture."""
        self._applied_velocity = (float(vx), float(vy), float(w))
        if self._bot is not None:
            self._bot.drive(vx=vx, vy=vy, w=w)

    async def _apply_velocity(self, vx: float, vy: float, w: float) -> None:
        """Drive an explicit velocity. Used when a caller already gated it."""
        if self.backend in ("sim", "robot"):
            self._drive_sync(vx, vy, w)
        elif self.backend == "ros2" and self._rosbridge_ws is not None:
            try:
                await self._rosbridge_ws.send(
                    json.dumps(cmd_vel_rosbridge(vx, vy, w))
                )
            except Exception:
                pass

    async def _send_velocity(self) -> None:
        vx, vy, w = self._current_velocity()
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                self._bot.drive(vx=vx, vy=vy, w=w)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None:
                try:
                    await self._rosbridge_ws.send(
                        json.dumps(cmd_vel_rosbridge(vx, vy, w))
                    )
                except Exception:
                    pass

    def _apply_joints(self, joints: dict[str, float]) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                for name, pos in joints.items():
                    self._bot.transport.send_joint_command(name, pos)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None and self._loop is not None:
                payload = json.dumps(joint_command_rosbridge(joints))
                self._loop.create_task(self._send_ros(payload))

    def _on_estop_changed(self, estopped: bool) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                if estopped:
                    self._bot.emergency_stop()
                else:
                    self._bot.release_stop()
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None and self._loop is not None:
                payload = json.dumps(estop_rosbridge(estopped))
                self._loop.create_task(self._send_ros(payload))

    async def _send_ros(self, payload: str) -> None:
        if self._rosbridge_ws is None:
            return
        try:
            await self._rosbridge_ws.send(payload)
        except Exception:
            pass

    def _get_telemetry(self) -> tuple[dict[str, float], dict[str, float]]:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                t = (
                    self._bot.transport.read()
                    if hasattr(self._bot.transport, "read")
                    else self._bot.telemetry()
                )
                if t and t.odom:
                    odom = {
                        "x": float(t.odom.x),
                        "y": float(t.odom.y),
                        "yaw": float(t.odom.theta),
                    }
                else:
                    odom = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                joints = {
                    j.name: float(j.position)
                    for j in (t.joints if t and t.joints else [])
                }
                return odom, joints
            return {"x": 0.0, "y": 0.0, "yaw": 0.0}, {}
        if self.backend == "ros2":
            return dict(self._latest_odom), dict(self._latest_joints)
        return {"x": 0.0, "y": 0.0, "yaw": 0.0}, {}

    async def _rosbridge_loop(self) -> None:
        if self._rosbridge_ws is None:
            return
        try:
            async for raw in self._rosbridge_ws:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                topic = data.get("topic")
                msg = data.get("msg", {})
                if not isinstance(topic, str):
                    continue
                if topic == ODOM_TOPIC:
                    odom = odom_from_ros(msg)
                    if odom is not None:
                        self._latest_odom = odom
                elif topic in (JOINT_STATE_TOPIC, ARM_JOINT_STATE_TOPIC):
                    parsed = joints_from_ros(msg)
                    if parsed:
                        merged = dict(self._latest_joints)
                        merged.update(parsed)
                        self._latest_joints = merged
                elif is_front_camera_topic(topic):
                    frame = camera_jpeg_from_ros(topic, msg)
                    if frame:
                        self._camera_frame = frame
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


def serve_sim(
    backend: str = "sim",
    robot: str | None = None,
    transport: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    rosbridge: str = "ws://localhost:9090",
    rate_hz: float = 20.0,
    stop_event: Optional[threading.Event] = None,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
) -> None:
    """Serve a robot simulation, live robot, or rosbridge gateway over WebSocket.

    Parameters:
        backend: ``sim``, ``ros2``, or ``robot``. ``isaac`` and ``mujoco`` raise
            ``SimServeUnavailable`` with the text ``not-implemented``.
        robot: robot spec id from ``ohho list`` (e.g. ``legged-base``).
        transport: transport URI for the ``robot`` backend (e.g. ``dds://lo?domain=1``).
        host: WebSocket server host.
        port: WebSocket server port. Defaults to 8790 so it does not clash
            with a Foxglove bridge on 8765.
        rosbridge: rosbridge WebSocket URL (for "ros2" backend).
        rate_hz: telemetry broadcast rate in Hz (default 20.0).
        stop_event: optional threading.Event to request server shutdown.
        ssl_cert: optional SSL certificate PEM file.
        ssl_key: optional SSL private key PEM file.
    """
    if backend == "robot" and not transport:
        raise ValueError("backend 'robot' requires a transport URI (--transport)")
    if backend_entry(backend) != "available":
        raise SimServeUnavailable(f"backend '{backend}' is not-implemented")
    server = SimServe(
        backend=backend,
        robot=robot,
        transport=transport,
        host=host,
        port=port,
        rosbridge=rosbridge,
        rate_hz=rate_hz,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
    )
    server.run(stop_event=stop_event)
