"""WebSocket server for simulated and ROS 2 robot telemetry and teleoperation.

Exposes a JSON WebSocket protocol over a single port, serving either the
in-process ``sim://`` simulator or bridging to ROS 2 topics via rosbridge.
``backend`` is ``sim``, ``ros2``, or ``robot``.

Protocol:
    Client sends:
        - ``velocity``: ``{"linear": {"x", "y", "z"}, "angular": {"z"}}``.
          Ground robots use ``x``, ``y``, and yaw. Aircraft also use ``linear.z``
          in ``frame``.
        - ``frame``: ``BODY_FRD`` for velocity teleop (also ``LOCAL_NED`` or
          ``LOCAL_ENU``). Required before an aircraft setpoint is streamed.
          Positive ``linear.z`` in ``BODY_FRD`` is down; this server does not
          invert stick-up into climb.
        - ``joints``: ``{"<joint_name>": <radians>, ...}``
        - ``deadman``: boolean. Omitted means false.
        - ``estop``: boolean
        - ``mode`` (optional): ``"stand"`` | ``"lie_down"`` | ``"recover"`` — mapped to
          allowlisted high-level actions on a connected :class:`~ohho.robot.Robot`.
        - ``flight`` (aircraft): ``{"request": "takeoff"|"land"|"rtl"|"hold"|"arm"|"disarm",
          "alt_m": metres, "confirm": "<arm phrase>"}``. ``arm`` asks the autopilot
          to arm motors. It does not arm the SafetyGate.

    Server sends:
        - ``odom``: ``{"x": float, "y": float, "yaw": float}`` (planar shadow)
        - ``joints``: ``{"<joint_name>": float, ...}``
        - ``protocolVersion``: ``ohho.PROTOCOL_VERSION``
        - ``backend``: ``"sim"``, ``"ros2"``, or ``"robot"``
        - ``posture`` / ``battery`` (optional, when the robot reports them)
        - ``drone``: the ``custom["drone"]`` object when the robot publishes one
        - ``streaming`` / ``frame`` / ``flight_status`` for an aircraft session
        - either ``camera``: base64-encoded JPEG or ``no_camera: true``

Safety invariants:
    - If ``deadman`` is false, or no client message arrives for 300 ms, the
      commanded velocity is zero. On an aircraft the server publishes one zero
      offboard velocity and then stops the setpoint stream. It does not land.
      The last non-zero setpoint is not repeated. On PX4 that silence trips
      the offboard-loss failsafe.
    - ``estop`` latches until a message with ``estop: false`` arrives, and calls
      ``emergency_stop`` (the in-process drone sim lands). No further setpoint
      is published while the latch is set.
    - ``takeoff`` and ``arm`` on ``backend=robot`` with a non-sim URI require
      the SafetyGate arm phrase and ``OHHO_ARM_HARDWARE=1`` already set in the
      process environment. The WebSocket payload cannot set that variable and
      cannot arm the gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any, Mapping, Optional

from .aerial import Frame
from .capabilities import AERIAL
from .protocol import PROTOCOL_VERSION
from .safety import ARM_ENV, ARM_PHRASE, DRONE_TAKEOFF_ALT_M
from .schema import Telemetry

log = logging.getLogger("ohho.sim_serve")

# Socket silence. Matches the long-standing sim-serve deadman window.
SETPOINT_SILENCE_S = 0.300

_FLIGHT_REQUESTS = frozenset({"takeoff", "land", "rtl", "hold", "arm", "disarm"})
# Human phrase + process env. land/rtl/hold/disarm stay available without it.
_CONFIRM_REQUESTS = frozenset({"takeoff", "arm"})
_SIM_SCHEMES = frozenset({"sim", "simulated"})

# Browser / VR teleop posture commands → SafetyGate allowlisted sport actions.
TELEOP_CLIENT_MODES: dict[str, str] = {
    "stand": "stand_up",
    "lie_down": "stand_down",
    "recover": "recovery_stand",
}


class SimServeUnavailable(RuntimeError):
    """Raised when sim-serve requirements are missing."""


def _vertical(value: Any) -> float:
    """``linear.z``. Unusable values are 0 so a null climb does not drop the socket."""
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


# OmniBot's Foxglove bridge binds 8765 (robot.launch.py, simulation.launch.py).
DEFAULT_PORT = 8790


def parse_twist(val: Any) -> tuple[float, float, float, float]:
    """Extract ``(vx, vy, vz, yaw_rate)`` from a velocity object.

    ``vx``, ``vy``, and yaw use the same keys as :func:`parse_velocity`.
    ``vz`` is ``linear.z``, ``linear.z``, or ``linear_z`` only. A bare ``z``
    stays yaw, which is what ground teleop already sends.

    The numbers are in the frame named by the client message. This function
    does not negate ``vz`` and does not negate yaw. In ``BODY_FRD``, positive
    ``vz`` is down and positive yaw rate is nose-right.
    """
    if not isinstance(val, dict):
        return 0.0, 0.0, 0.0, 0.0

    vx, vy, vz, w = 0.0, 0.0, 0.0, 0.0

    # Nested ROS-style: {"linear": {"x": ...}, "angular": {"z": ...}}
    if "linear" in val or "angular" in val:
        lin = val.get("linear")
        if isinstance(lin, dict):
            vx = float(lin.get("x", 0.0))
            vy = float(lin.get("y", 0.0))
            vz = _vertical(lin.get("z", 0.0))
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
    if "linear.z" in val:
        vz = _vertical(val["linear.z"])
    if "angular.z" in val:
        w = float(val["angular.z"])

    # Flat underscore keys
    if "linear_x" in val:
        vx = float(val["linear_x"])
    if "linear_y" in val:
        vy = float(val["linear_y"])
    if "linear_z" in val:
        vz = _vertical(val["linear_z"])
    if "angular_z" in val:
        w = float(val["angular_z"])

    # Short coordinate keys. Bare ``z`` is yaw, not climb.
    if "x" in val and "linear" not in val:
        vx = float(val["x"])
    if "y" in val and "linear" not in val:
        vy = float(val["y"])
    if "z" in val and "angular" not in val:
        w = float(val["z"])

    return vx, vy, vz, w


def parse_velocity(val: Any) -> tuple[float, float, float]:
    """Extract ``(vx, vy, w)`` from various velocity dictionary representations.

    Supports nested ROS Twist style (``{"linear": {"x": ...}, "angular": {"z": ...}}``),
    dotted keys (``{"linear.x": ...}``), flat keys (``{"linear_x": ...}``),
    or coordinate keys (``{"x": ..., "y": ..., "z": ...}``). ``linear.z`` is
    not part of this tuple; aerial teleop reads it from :func:`parse_twist`.
    """
    vx, vy, _vz, w = parse_twist(val)
    return vx, vy, w


def parse_frame(value: Any) -> Optional[Frame]:
    """A client frame string, or ``None`` when it is not a :class:`Frame`.

    There is no default. ``BODY_FRD``, ``LOCAL_NED``, and ``LOCAL_ENU`` are
    the only names. Matching is case-insensitive.
    """
    if isinstance(value, Frame):
        return value
    if not isinstance(value, str):
        return None
    try:
        return Frame(value.strip().upper())
    except ValueError:
        return None


def is_sim_transport_uri(uri: Optional[str]) -> bool:
    """True when ``uri`` uses the ``sim`` or ``simulated`` scheme."""
    if not uri:
        return False
    scheme = uri.split("://", 1)[0].strip().lower()
    return scheme in _SIM_SCHEMES


def requires_flight_confirmation(backend: str, transport: Optional[str]) -> bool:
    """True when takeoff/arm need the arm phrase and ``OHHO_ARM_HARDWARE``.

    ``backend=sim`` does not. ``backend=robot`` with ``sim://`` does not.
    Any other robot-backend URI does.
    """
    return backend == "robot" and not is_sim_transport_uri(transport)


def normalize_phrase(value: Any) -> str:
    """Fold a confirmation phrase the way SafetyGate folds ``ARM_PHRASE``."""
    return " ".join(str(value).split()).casefold().rstrip(".")


def privileged_flight_block(confirm: Any, environ: Mapping[str, str]) -> str:
    """``''`` when a takeoff or arm request may be forwarded.

    ``environ`` must be the process environment. This does not write it, and
    it does not look at a WebSocket payload for variable names. A missing
    ``OHHO_ARM_HARDWARE=1`` or a phrase other than ``ARM_PHRASE`` is refused.
    """
    if environ.get(ARM_ENV) != "1":
        return "env"
    if normalize_phrase(confirm) != normalize_phrase(ARM_PHRASE):
        return "not_confirmed"
    return ""


def parse_takeoff_altitude(alt_m: Any) -> tuple[Optional[float], str]:
    """Teleop takeoff altitude in metres, positive up.

    Omitted ``alt_m`` uses the 3 m teleop ceiling. A higher request is
    refused. The reason is ``''`` when ``alt_m`` is usable.
    """
    if alt_m is None:
        return DRONE_TAKEOFF_ALT_M, ""
    if isinstance(alt_m, bool) or not isinstance(alt_m, (int, float)):
        return None, "bad_altitude"
    if not math.isfinite(alt_m) or alt_m <= 0.0:
        return None, "bad_altitude"
    altitude = float(alt_m)
    if altitude > DRONE_TAKEOFF_ALT_M:
        return None, "altitude"
    return altitude, ""


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
    drone = (t.custom or {}).get("drone")
    if isinstance(drone, dict):
        out["drone"] = drone
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
        robot: str = "omnibot",
        transport: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        rosbridge: str = "ws://localhost:9090",
        rate_hz: float = 20.0,
    ) -> None:
        if backend not in ("sim", "ros2", "robot"):
            raise ValueError(
                f"unsupported backend '{backend}' (choose: sim, ros2, robot)"
            )
        if backend == "robot" and not transport:
            raise ValueError("backend 'robot' requires a transport URI (--transport)")

        self.backend = backend
        self.robot_id = robot
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
        self._target_vz: float = 0.0
        self._target_w: float = 0.0
        self._frame: Optional[Frame] = None
        self._target_joints: dict[str, float] = {}
        self._aerial: bool = False
        self._aerial_stream_open: bool = False
        self._flight_status: Optional[dict[str, Any]] = None
        self._command_error: Optional[str] = None
        self._last_drone_step: Optional[float] = None

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

            uri = "sim://" if self.backend == "sim" else self._transport_uri
            assert uri is not None
            self._bot = Robot.connect(self.robot_id, transport=uri, runtime="native")
            await self._arm_sim_link()
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

        self._aerial = self._detect_aerial()
        self._server = await websockets.serve(self._handle_client, self.host, self.port)
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
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        if self._aerial:
            self._close_aerial_stream()

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
                await self._dispatch_motion()

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

        # Deadman switch. A message that omits it clears the switch.
        self._deadman = bool(data.get("deadman", False))

        # Aircraft frame. Ground sessions ignore it. An unknown name does not
        # become BODY_FRD, and it does not consume the velocity in this message.
        frame_rejected = False
        if self._aerial and "frame" in data:
            parsed = parse_frame(data["frame"])
            if parsed is None:
                self._command_error = "frame"
                frame_rejected = True
            else:
                self._frame = parsed
                if self._command_error == "frame":
                    self._command_error = None

        if "velocity" in data and not frame_rejected:
            vx, vy, vz, w = parse_twist(data["velocity"])
            self._target_vx = vx
            self._target_vy = vy
            self._target_vz = vz
            self._target_w = w

        if "flight" in data:
            self._on_flight(data["flight"])

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

        self._schedule_motion()

    def _schedule_motion(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.create_task(self._dispatch_motion())

    def _motion_live(self, now: float) -> bool:
        """True while the deadman is held and the socket is inside 300 ms."""
        if self._estopped or not self._deadman or self._last_msg_time <= 0.0:
            return False
        return (now - self._last_msg_time) <= SETPOINT_SILENCE_S

    async def _dispatch_motion(self) -> None:
        """Publish the live setpoint, or zero and (for aircraft) stop the stream."""
        if self._aerial:
            if self._motion_live(time.monotonic()):
                self._open_aerial_stream()
            else:
                self._close_aerial_stream()
            return
        if self._motion_live(time.monotonic()):
            await self._apply_velocity(
                self._target_vx, self._target_vy, self._target_w, self._target_vz
            )
        else:
            await self._apply_velocity(0.0, 0.0, 0.0, 0.0)

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / max(1.0, self.rate_hz)
        while self._running:
            now = time.monotonic()
            await self._dispatch_motion()
            prev = self._last_drone_step
            self._last_drone_step = now
            if prev is not None:
                self._step_drone_sim(min(max(now - prev, 0.0), 0.1))

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
            "protocolVersion": PROTOCOL_VERSION,
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
        if self._flight_status is not None:
            msg["flight_status"] = dict(self._flight_status)
        if self._aerial:
            msg["streaming"] = self._aerial_stream_open
            if self._frame is not None:
                msg["frame"] = self._frame.value
        if self._command_error:
            msg["command_error"] = self._command_error
        return msg

    def _detect_aerial(self) -> bool:
        bot = self._bot
        if bot is None:
            return False
        has = getattr(bot, "has", None)
        if not callable(has):
            return False
        try:
            return bool(has(AERIAL))
        except Exception:
            return False

    def _set_flight_status(self, request: str, accepted: bool, reason: str) -> None:
        self._flight_status = {
            "request": request,
            "accepted": accepted,
            "reason": reason,
        }

    def _on_flight(self, flight: Any) -> None:
        """Forward one flight request. Never writes ``os.environ`` and never calls ``Robot.arm``."""
        if not isinstance(flight, dict):
            self._set_flight_status("", False, "bad_request")
            return
        raw = flight.get("request")
        request = raw.strip().lower() if isinstance(raw, str) else ""
        if request not in _FLIGHT_REQUESTS:
            self._set_flight_status(request, False, "unknown_request")
            return
        if not self._aerial or self._bot is None:
            self._set_flight_status(request, False, "not_aerial")
            return
        if self._estopped:
            self._set_flight_status(request, False, "estop")
            return
        altitude: Optional[float] = None
        if request == "takeoff":
            altitude, why = parse_takeoff_altitude(flight.get("alt_m"))
            if why or altitude is None:
                self._set_flight_status(request, False, why or "bad_altitude")
                return
        if request in _CONFIRM_REQUESTS and requires_flight_confirmation(
            self.backend, self._transport_uri
        ):
            # Phrase comes from ``confirm``. The env var is the process environment
            # only — payload keys are not consulted and are not copied into os.environ.
            why = privileged_flight_block(flight.get("confirm"), os.environ)
            if why:
                log.warning("sim-serve: %s refused: %s", request, why)
                self._set_flight_status(request, False, why)
                return
        self._invoke_flight(request, altitude)

    def _invoke_flight(self, request: str, altitude: Optional[float]) -> None:
        bot = self._bot
        if bot is None:
            self._set_flight_status(request, False, "not_aerial")
            return
        try:
            if request == "takeoff":
                bot.takeoff(DRONE_TAKEOFF_ALT_M if altitude is None else altitude)
            elif request == "land":
                bot.land()
            elif request == "rtl":
                bot.rtl()
            elif request == "hold":
                bot.hold()
            elif request == "arm":
                bot.arm_flight()
            elif request == "disarm":
                bot.disarm_flight()
            else:
                self._set_flight_status(request, False, "unknown_request")
                return
        except (TypeError, ValueError):
            self._set_flight_status(request, False, "bad_arguments")
            return
        if getattr(bot, "gated", False):
            reason = getattr(getattr(bot, "transport", None), "last_error", None)
            if reason:
                self._set_flight_status(request, False, str(reason))
                return
        self._set_flight_status(request, True, "")

    def _open_aerial_stream(self) -> None:
        frame = self._frame
        if frame is None or self._bot is None:
            self._command_error = "frame"
            return
        if self._command_error == "frame":
            self._command_error = None
        self._aerial_stream_open = True
        self._send_offboard(
            self._target_vx, self._target_vy, self._target_vz, self._target_w, frame
        )

    def _close_aerial_stream(self) -> None:
        """One zero velocity, then silence. Does not land. Drops the retained setpoint.

        An e-stop already owns the vehicle, so this only stops the stream and
        does not publish another setpoint on top of ``emergency_stop``.
        """
        was_open = self._aerial_stream_open
        self._aerial_stream_open = False
        frame = self._frame
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vz = 0.0
        self._target_w = 0.0
        if self._estopped or not was_open or frame is None or self._bot is None:
            return
        self._send_offboard(0.0, 0.0, 0.0, 0.0, frame)

    def _send_offboard(
        self, vx: float, vy: float, vz: float, yaw_rate: float, frame: Frame
    ) -> None:
        if not all(math.isfinite(v) for v in (vx, vy, vz, yaw_rate)):
            self._command_error = "bad_velocity"
            return
        bot = self._bot
        if bot is None:
            return
        try:
            bot.offboard_velocity(vx, vy, vz, yaw_rate, frame)
        except (TypeError, ValueError):
            self._command_error = "bad_velocity"
        except Exception:
            log.error("sim-serve: offboard_velocity failed", exc_info=True)

    def _step_drone_sim(self, dt: float) -> None:
        """Advance the in-process drone stub. Ground sims keep their own thread."""
        if dt <= 0.0 or not math.isfinite(dt) or self._bot is None:
            return
        transport = getattr(self._bot, "transport", None)
        if getattr(transport, "backend", None) != "drone-sim":
            return
        step = getattr(transport, "step", None)
        if not callable(step):
            return
        try:
            step(dt)
        except Exception:
            log.error("sim-serve: drone-sim step failed", exc_info=True)

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

    async def _apply_velocity(
        self, vx: float, vy: float, w: float, vz: float = 0.0
    ) -> None:
        if self.backend in ("sim", "robot"):
            if self._bot is not None:
                # Planar base command. Aircraft setpoints go through offboard_velocity.
                self._bot.drive(vx=vx, vy=vy, w=w)
        elif self.backend == "ros2":
            if self._rosbridge_ws is not None:
                twist_msg = {
                    "op": "publish",
                    "topic": "/cmd_vel",
                    "type": "geometry_msgs/Twist",
                    "msg": {
                        "linear": {"x": float(vx), "y": float(vy), "z": float(vz)},
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
    transport: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    rosbridge: str = "ws://localhost:9090",
    rate_hz: float = 20.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Serve a robot simulation or rosbridge gateway over WebSocket.

    Parameters:
        backend: "sim", "ros2", or "robot".
        robot: robot spec id (e.g. "omnibot", "unitree-go2", "legged-base", "px4-sitl").
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
    )
    server.run(stop_event=stop_event)
