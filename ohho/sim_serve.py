"""WebSocket server for simulated and ROS 2 robot telemetry and teleoperation.

Exposes a JSON WebSocket protocol over a single port, serving either the
in-process ``sim://`` simulator or bridging to ROS 2 topics via rosbridge.

Protocol:
    Client sends:
        - ``velocity``: ``{"linear": {"x": ..., "y": ...}, "angular": {"z": ...}}``
        - ``joints``: ``{"<joint_name>": <radians>, ...}``
        - ``deadman``: boolean
        - ``estop``: boolean
        - ``mode`` (optional): ``"stand"`` | ``"lie_down"`` | ``"recover"`` — mapped to
          allowlisted high-level actions on a connected :class:`~ohho.robot.Robot`.

    Server sends:
        - ``odom``: ``{"x": float, "y": float, "yaw": float}``
        - ``joints``: ``{"<joint_name>": float, ...}``
        - ``backend``: ``"sim"``, ``"ros2"``, or ``"robot"``
        - ``posture`` / ``battery`` (optional, when the robot reports them)
        - either ``camera``: base64-encoded JPEG or ``no_camera: true``

Safety invariants:
    - If ``deadman`` is false, or no client message arrives for 300 ms, the
      commanded velocity is zero.
    - ``estop`` latches until a message with ``estop: false`` arrives.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from .config import RobotProfile, load_robot_profile, resolve_robot_id
from .schema import Telemetry

# Browser / VR teleop posture commands → SafetyGate allowlisted sport actions.
TELEOP_CLIENT_MODES: dict[str, str] = {
    "stand": "stand_up",
    "lie_down": "stand_down",
    "recover": "recovery_stand",
}


class SimServeUnavailable(RuntimeError):
    """Raised when sim-serve requirements are missing."""


# OmniBot's Foxglove bridge binds 8765 (robot.launch.py, simulation.launch.py).
DEFAULT_PORT = 8790


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
    """Manages the WebSocket server and the simulation or rosbridge backend."""

    def __init__(
        self,
        backend: str = "sim",
        robot: str | None = None,
        transport: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        rosbridge: str = "ws://localhost:9090",
        rate_hz: float = 20.0,
        *,
        home: Path | str | None = None,
        cwd: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        if backend not in ("sim", "ros2", "robot"):
            raise ValueError(
                f"unsupported backend '{backend}' (choose: sim, ros2, robot)"
            )
        if backend == "robot" and not transport:
            raise ValueError("backend 'robot' requires a transport URI (--transport)")

        explicit = isinstance(robot, str) and bool(robot.strip())
        self.resolved = resolve_robot_id(
            robot if explicit else None,
            home=home,
            environ=environ,
            config_path=config_path,
        )
        if explicit and home is None and cwd is None and config_path is None:
            self.profile = RobotProfile(self.resolved.robot_id)
        else:
            self.profile = load_robot_profile(
                self.resolved.robot_id, home=home, cwd=cwd
            )
        self.backend = backend
        self.robot_id = self.resolved.robot_id
        self._transport_uri = transport
        self.host = host
        self.port = port
        self.rosbridge_url = rosbridge
        self.rate_hz = rate_hz

        # Session safety state
        self._deadman: bool = False
        self._last_msg_time: float = 0.0
        self._estopped: bool = False
        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_w: float = 0.0
        self._target_joints: dict[str, float] = {}

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

            extra: dict[str, Any] = {}
            if self.profile.sim_backend:
                extra["sim_backend"] = self.profile.sim_backend
            if self.profile.asset_path:
                extra["asset_path"] = self.profile.asset_path
            if self.profile.safety_overrides:
                extra["safety_overrides"] = dict(self.profile.safety_overrides)
            uri = "sim://" if self.backend == "sim" else self._transport_uri
            if not uri:
                raise SimServeUnavailable(
                    "backend 'robot' requires a transport URI (--transport)"
                )
            self._bot = Robot.connect(
                self.robot_id,
                transport=uri,
                runtime="native",
                **extra,
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
            initial_msg = self._telemetry_message(odom, joints)
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

        if "mode" in data:
            self._apply_teleop_mode(data["mode"])

        # Joint commands
        if "joints" in data and isinstance(data["joints"], dict):
            self._target_joints = {
                str(k): float(v)
                for k, v in data["joints"].items()
                if isinstance(v, (int, float))
            }
            if not self._estopped:
                self._apply_joints(self._target_joints)

        # Update velocity command immediately
        if self._deadman and not self._estopped:
            vx, vy, w = self._target_vx, self._target_vy, self._target_w
        else:
            vx, vy, w = 0.0, 0.0, 0.0

        if self._loop is not None and self._loop.is_running():
            self._loop.create_task(self._apply_velocity(vx, vy, w))

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / max(1.0, self.rate_hz)
        while self._running:
            now = time.monotonic()

            # Timeout check: if no message arrives for 300 ms, velocity is zero
            if (
                (now - self._last_msg_time > 0.300)
                or not self._deadman
                or self._estopped
            ):
                await self._apply_velocity(0.0, 0.0, 0.0)
            else:
                await self._apply_velocity(
                    self._target_vx, self._target_vy, self._target_w
                )

            odom, joints = self._get_telemetry()
            msg = self._telemetry_message(odom, joints)

            payload = json.dumps(msg)
            if self._clients:
                aws = [ws.send(payload) for ws in list(self._clients)]
                await asyncio.gather(*aws, return_exceptions=True)

            await asyncio.sleep(interval)

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _telemetry_message(
        self, odom: dict[str, float], joints: dict[str, float]
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "odom": odom,
            "joints": joints,
            "backend": self.backend,
        }
        if self._camera_frame is not None:
            msg["camera"] = self._camera_frame
        else:
            msg["no_camera"] = True
        if self._bot is not None:
            try:
                msg.update(telemetry_extras(self._bot.telemetry()))
            except Exception:
                pass
        return msg

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

    async def _apply_velocity(self, vx: float, vy: float, w: float) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                self._bot.drive(vx=vx, vy=vy, w=w)
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
    robot: str | None = None,
    transport: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    rosbridge: str = "ws://localhost:9090",
    rate_hz: float = 20.0,
    stop_event: Optional[threading.Event] = None,
    *,
    home: Path | str | None = None,
    cwd: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: Path | str | None = None,
) -> None:
    """Serve a robot simulation or rosbridge gateway over WebSocket.

    Parameters:
        backend: "sim", "ros2", or "robot".
        robot: robot spec id (e.g. "omnibot", "unitree-go2"). Omit it to use
            the shared resolver.
        transport: transport URI for the "robot" backend (e.g. ``dds://lo?domain=1``).
        host: WebSocket server host.
        port: WebSocket server port. Defaults to 8790 so it does not clash
            with OmniBot's Foxglove bridge on 8765.
        rosbridge: rosbridge WebSocket URL (for "ros2" backend).
        rate_hz: telemetry broadcast rate in Hz (default 20.0).
        stop_event: optional threading.Event to request server shutdown.
    """
    server = SimServe(
        backend=backend,
        robot=robot,
        transport=transport,
        host=host,
        port=port,
        rosbridge=rosbridge,
        rate_hz=rate_hz,
        home=home,
        cwd=cwd,
        environ=environ,
        config_path=config_path,
    )
    server.run(stop_event=stop_event)
