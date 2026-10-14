"""WebSocket server for simulated and ROS 2 robot telemetry and teleoperation.

Exposes one JSON WebSocket session. The in-process ``sim://`` backend needs no
ROS. The ``ros2`` backend bridges that same session to rosbridge topics the
Connect console already uses (``/cmd_vel``, ``/odom``, joint state, front
camera, ``/emergency_stop``).

Client JSON:
    - ``velocity``: Twist-like ``{"linear": {"x", "y"}, "angular": {"z"}}``
      or Connect fields ``linearX`` / ``linearY`` / ``angularZ``
    - ``joints``: ``{"<joint_name>": radians, ...}``
    - ``deadman``: boolean. Missing, false, or any other value is not held.
    - ``estop``: boolean. ``true`` latches; only ``false`` clears it.

Server JSON:
    - ``odom``: ``{"x": float, "y": float, "yaw": float}``
    - ``joints``: ``{"<joint_name>": float, ...}``
    - ``backend``: ``"sim"`` or ``"ros2"``
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
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

# Commanded velocity is zero once this long has passed with no client message.
DEADMAN_TIMEOUT_S = 0.300

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


class SimServeUnavailable(RuntimeError):
    """Raised when sim-serve requirements are missing or a backend is not implemented."""


def backend_entry(backend: str) -> str:
    """Return ``available`` or ``not-implemented`` for a sim-serve backend name.

    Isaac Sim and MuJoCo have no runtime here. Callers must not import those
    packages; this function only classifies the name.
    """
    if backend in ("sim", "ros2"):
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


def is_ros2_available() -> bool:
    """True if rclpy or ROS environment or OHHO_ROSBRIDGE is set."""
    import importlib.util

    return (
        importlib.util.find_spec("rclpy") is not None
        or bool(os.environ.get("ROS_DISTRO"))
        or bool(os.environ.get("OHHO_ROSBRIDGE"))
    )


class SimServe:
    """Manages the WebSocket server and the simulation or rosbridge backend."""

    def __init__(
        self,
        backend: str = "sim",
        robot: str = "omnibot",
        host: str = "0.0.0.0",
        port: int = 8765,
        rosbridge: str = "ws://localhost:9090",
        rate_hz: float = 20.0,
    ) -> None:
        if backend_entry(backend) != "available":
            raise SimServeUnavailable(f"backend '{backend}' is not-implemented")

        self.backend = backend
        self.robot_id = robot
        self.host = host
        self.port = port
        self.rosbridge_url = rosbridge
        self.rate_hz = rate_hz

        # Session safety state
        self._deadman: bool = False
        self._last_msg_time: float = 0.0
        self._command_window_open: bool = False
        self._estopped: bool = False
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_w: float = 0.0
        self._target_joints: dict[str, float] = {}
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

        if self.backend == "sim":
            from .robot import Robot

            self._bot = Robot.connect(
                self.robot_id, transport="sim://", runtime="native"
            )
        elif self.backend == "ros2":
            self._rosbridge_ws = await websockets.connect(self.rosbridge_url)
            for topic, msg_type in ROS_SUBSCRIPTIONS:
                await self._rosbridge_ws.send(
                    json.dumps({"op": "subscribe", "topic": topic, "type": msg_type})
                )
            self._rosbridge_task = asyncio.create_task(self._rosbridge_loop())

        self._server = await websockets.serve(self._handle_client, self.host, self.port)
        if self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

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

        self._arm_timeout()
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
        return build_telemetry_message(
            odom=odom,
            joints=joints,
            backend=self.backend,
            camera=self._camera_frame,
        )

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

    async def _send_velocity(self) -> None:
        vx, vy, w = self._current_velocity()
        if self.backend == "sim":
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
        if self.backend == "sim":
            if self._bot is not None:
                for name, pos in joints.items():
                    self._bot.transport.send_joint_command(name, pos)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None and self._loop is not None:
                payload = json.dumps(joint_command_rosbridge(joints))
                self._loop.create_task(self._send_ros(payload))

    def _on_estop_changed(self, estopped: bool) -> None:
        if self.backend == "sim":
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
        if self.backend == "sim":
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
    robot: str = "omnibot",
    host: str = "0.0.0.0",
    port: int = 8765,
    rosbridge: str = "ws://localhost:9090",
    rate_hz: float = 20.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Serve a robot simulation or rosbridge gateway over WebSocket.

    Parameters:
        backend: ``sim`` or ``ros2``. ``isaac`` and ``mujoco`` raise
            ``SimServeUnavailable`` with the text ``not-implemented``.
        robot: robot spec id (e.g. "omnibot").
        host: WebSocket server host.
        port: WebSocket server port.
        rosbridge: rosbridge WebSocket URL (for "ros2" backend).
        rate_hz: telemetry broadcast rate in Hz (default 20.0).
        stop_event: optional threading.Event to request server shutdown.
    """
    if backend_entry(backend) != "available":
        raise SimServeUnavailable(f"backend '{backend}' is not-implemented")
    server = SimServe(
        backend=backend,
        robot=robot,
        host=host,
        port=port,
        rosbridge=rosbridge,
        rate_hz=rate_hz,
    )
    server.run(stop_event=stop_event)
