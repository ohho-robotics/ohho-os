"""WebSocket server for simulated and ROS 2 robot telemetry and teleoperation.

Exposes a JSON WebSocket protocol over a single port, serving either the
in-process ``sim://`` simulator, a ``robot`` transport URI, or a bridge to
ROS 2 topics via rosbridge.

Protocol:
    Client sends:
        - ``velocity``: ``{"linear": {"x": ..., "y": ...}, "angular": {"z": ...}}``
        - ``joints``: ``{"<joint_name>": <radians>, ...}``
        - ``deadman``: boolean
        - ``estop``: boolean
        - ``modes``: ``{"request": "<posture>", "confirm": "<phrase>"}``

    ``modes.request`` is a ``humanoid.posture`` value the spec declares
    (``prepare``, ``walk``, ``damping``, ``custom``, ``get_up``, ``lie_down``).
    The locomotion request may also be named ``walk_mode`` when ``walk`` is
    declared, and ``damp`` when ``damping`` is declared. Any other request is
    refused. The check is the capability and that declared list, never a robot id.

    ``backend=sim`` does not need ``confirm``. ``backend=robot`` with a URI that
    is not an explicit sim target does: ``confirm`` must be the arming phrase,
    and ``OHHO_ARM_HARDWARE=1`` must already be in the server environment. This
    module never sets that variable and never arms the gate.

    Server sends:
        - ``odom``: ``{"x": float, "y": float, "yaw": float}``
        - ``joints``: ``{"joint_name": float, ...}``
        - ``backend``: ``"sim"``, ``"ros2"``, or ``"robot"``
        - ``modes``: the last posture decision, when a request was seen
        - either ``camera``: base64-encoded JPEG or ``no_camera: true``

Safety invariants:
    - If ``deadman`` is false, or no client message arrives for 300 ms, the
      commanded velocity is zero and the posture is left as it is. That path
      does not damp.
    - A posture request is applied only while the deadman is held and e-stop
      is clear.
    - ``estop`` latches until a message with ``estop: false`` arrives.
"""

from __future__ import annotations

import asyncio
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
from .registry import RobotSpec, UnknownRobot, get_spec
from .safety import ARM_ENV, ARM_PHRASE


class SimServeUnavailable(RuntimeError):
    """Raised when sim-serve requirements are missing."""


# No fresh client message for this long: zero velocity, leave the mode.
SILENCE_S = 0.300

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


def parse_velocity(val: Any) -> tuple[float, float, float]:
    """Extract (vx, vy, w) from various velocity dictionary representations.

    Supports nested ROS Twist style (``{"linear": {"x": ...}, "angular": {"z": ...}}``),
    dotted keys (``{"linear.x": ...}``), flat keys (``{"linear_x": ...}``),
    or coordinate keys (``{"x": ..., "y": ..., "z": ...}``).
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
        elif isinstance(lin, (int, float)):
            vx = float(lin)

        ang = val.get("angular")
        if isinstance(ang, dict):
            w = float(ang.get("z", 0.0))
        elif isinstance(ang, (int, float)):
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

    # Short coordinate keys
    if "x" in val and "linear" not in val:
        vx = float(val["x"])
    if "y" in val and "linear" not in val:
        vy = float(val["y"])
    if "z" in val and "angular" not in val:
        w = float(val["z"])

    return vx, vy, w


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
        robot: str = "omnibot",
        host: str = "0.0.0.0",
        port: int = 8765,
        rosbridge: str = "ws://localhost:9090",
        rate_hz: float = 20.0,
        transport: str | None = None,
    ) -> None:
        if backend not in ("sim", "ros2", "robot"):
            raise ValueError(
                f"unsupported backend '{backend}' (choose: sim, ros2, robot)"
            )
        uri = "" if transport is None else str(transport).strip()
        if backend == "robot" and not uri:
            raise ValueError("backend 'robot' requires a transport URI")

        self.backend = backend
        self.robot_id = robot
        self.host = host
        self.port = port
        self.rosbridge_url = rosbridge
        self.rate_hz = rate_hz
        if backend == "sim":
            self.transport_uri = "sim://"
        elif backend == "robot":
            self.transport_uri = uri
        else:
            self.transport_uri = ""

        # Session safety state
        self._deadman: bool = False
        self._last_msg_time: float = 0.0
        self._estopped: bool = False
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_w: float = 0.0
        self._target_joints: dict[str, float] = {}
        self._applied_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._posture: str = ""
        self._mode_result: dict[str, Any] | None = None

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
                "backend 'ros2' depends on the [ros2] extra (source your ROS 2 environment or set OHHO_ROSBRIDGE)."
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

            self._bot = Robot.connect(
                self.robot_id, transport=self.transport_uri, runtime="native"
            )
        elif self.backend == "ros2":
            self._rosbridge_ws = await websockets.connect(self.rosbridge_url)
            await self._rosbridge_ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "topic": "/odom",
                        "type": "nav_msgs/Odometry",
                    }
                )
            )
            await self._rosbridge_ws.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "topic": "/joint_states",
                        "type": "sensor_msgs/JointState",
                    }
                )
            )
            self._rosbridge_task = asyncio.create_task(self._rosbridge_loop())

        self._server = await websockets.serve(self._handle_client, self.host, self.port)
        if self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Shut down the server and disconnect backends."""
        self._running = False
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
            # Send initial telemetry immediately upon connection
            odom, joints = self._get_telemetry()
            initial_msg: dict[str, Any] = {
                "odom": odom,
                "joints": joints,
                "backend": self.backend,
            }
            if self._camera_frame is not None:
                initial_msg["camera"] = self._camera_frame
            else:
                initial_msg["no_camera"] = True
            await websocket.send(json.dumps(initial_msg))

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
                await self._apply_velocity(0.0, 0.0, 0.0)

    def commanded_velocity(self, now: float) -> tuple[float, float, float]:
        """Velocity to send now. Silence, a released deadman, or e-stop is zero.

        This does not change posture.
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
            uri=self.transport_uri,
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

    def _on_client_message(self, data: dict[str, Any]) -> None:
        now = time.monotonic()
        self._last_msg_time = now

        # Estop latch: latches until a message with estop: false arrives
        if "estop" in data:
            estop_val = data["estop"]
            if bool(estop_val):
                self._estopped = True
                self._on_estop_changed(True)
            elif estop_val is False or estop_val == 0:
                self._estopped = False
                self._on_estop_changed(False)

        # Deadman switch
        self._deadman = bool(data.get("deadman", False))

        # Velocity command
        if "velocity" in data:
            self._target_vx, self._target_vy, self._target_w = parse_velocity(
                data["velocity"]
            )

        # Joint commands
        if "joints" in data and isinstance(data["joints"], dict):
            self._target_joints = {
                str(k): float(v)
                for k, v in data["joints"].items()
                if isinstance(v, (int, float))
            }
            if not self._estopped:
                self._apply_joints(self._target_joints)

        if "modes" in data:
            self._on_modes(data["modes"])

        # A fresh message is not silence. Deadman and e-stop still zero velocity.
        vx, vy, w = self.commanded_velocity(now)
        if self.backend in ("sim", "robot"):
            self._drive_sync(vx, vy, w)
        elif self._loop is not None and self._loop.is_running():
            self._loop.create_task(self._apply_velocity(vx, vy, w))

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / max(1.0, self.rate_hz)
        while self._running:
            now = time.monotonic()

            # Silence, a released deadman, or e-stop: zero velocity, keep posture.
            vx, vy, w = self.commanded_velocity(now)
            await self._apply_velocity(vx, vy, w)

            odom, joints = self._get_telemetry()
            msg: dict[str, Any] = {
                "odom": odom,
                "joints": joints,
                "backend": self.backend,
            }
            if self._mode_result is not None:
                msg["modes"] = dict(self._mode_result)
            if self._camera_frame is not None:
                msg["camera"] = self._camera_frame
            else:
                msg["no_camera"] = True

            payload = json.dumps(msg)
            if self._clients:
                aws = [ws.send(payload) for ws in list(self._clients)]
                await asyncio.gather(*aws, return_exceptions=True)

            await asyncio.sleep(interval)

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _drive_sync(self, vx: float, vy: float, w: float) -> None:
        """Send a body velocity on the sim or robot backend. Does not change posture."""
        self._applied_velocity = (float(vx), float(vy), float(w))
        if self._bot is not None:
            self._bot.drive(vx=vx, vy=vy, w=w)

    async def _apply_velocity(self, vx: float, vy: float, w: float) -> None:
        if self.backend in ("sim", "robot"):
            self._drive_sync(vx, vy, w)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None:
                twist_msg = {
                    "op": "publish",
                    "topic": "/cmd_vel",
                    "type": "geometry_msgs/Twist",
                    "msg": {
                        "linear": {"x": float(vx), "y": float(vy), "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": float(w)},
                    },
                }
                try:
                    await self._rosbridge_ws.send(json.dumps(twist_msg))
                except Exception:
                    pass

    def _apply_joints(self, joints: dict[str, float]) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                for name, pos in joints.items():
                    self._bot.transport.send_joint_command(name, pos)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None and self._loop is not None:
                joint_cmd = {
                    "op": "publish",
                    "topic": "/arm/joint_commands",
                    "type": "sensor_msgs/JointState",
                    "msg": {
                        "name": list(joints.keys()),
                        "position": [float(v) for v in joints.values()],
                        "velocity": [],
                        "effort": [],
                    },
                }
                self._loop.create_task(self._rosbridge_ws.send(json.dumps(joint_cmd)))

    def _on_estop_changed(self, estopped: bool) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                if estopped:
                    self._bot.emergency_stop()
                else:
                    self._bot.release_stop()
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None and self._loop is not None:
                estop_msg = {
                    "op": "publish",
                    "topic": "/emergency_stop",
                    "type": "std_msgs/Bool",
                    "msg": {"data": bool(estopped)},
                }
                self._loop.create_task(self._rosbridge_ws.send(json.dumps(estop_msg)))

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
        elif self.backend == "ros2":
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
                if topic == "/odom":
                    pose = msg.get("pose", {}).get("pose", {})
                    pos = pose.get("position", {})
                    ori = pose.get("orientation", {})
                    x = float(pos.get("x", 0.0))
                    y = float(pos.get("y", 0.0))
                    qw = float(ori.get("w", 1.0))
                    qz = float(ori.get("z", 0.0))
                    yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * (qz * qz))
                    self._latest_odom = {"x": x, "y": y, "yaw": yaw}
                elif topic in ("/joint_states", "/arm/joint_states"):
                    names = msg.get("name", [])
                    positions = msg.get("position", [])
                    self._latest_joints = {
                        str(n): float(p) for n, p in zip(names, positions)
                    }
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
    transport: str | None = None,
) -> None:
    """Serve a robot simulation, a robot URI, or a rosbridge gateway over WebSocket.

    Parameters:
        backend: "sim", "ros2", or "robot".
        robot: robot spec id.
        host: WebSocket server host.
        port: WebSocket server port.
        rosbridge: rosbridge WebSocket URL (for "ros2" backend).
        rate_hz: telemetry broadcast rate in Hz (default 20.0).
        stop_event: optional threading.Event to request server shutdown.
        transport: transport URI for the "robot" backend.
    """
    server = SimServe(
        backend=backend,
        robot=robot,
        host=host,
        port=port,
        rosbridge=rosbridge,
        rate_hz=rate_hz,
        transport=transport,
    )
    server.run(stop_event=stop_event)
