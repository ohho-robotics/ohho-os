"""Crazyflie backend (OHH-303).

Two URIs, one transport:

* ``crazyflie://radio://0/80/2M/E7E7E7E7E7`` — Crazyradio via ``cflib``
  (optional extra ``crazyflie``). The library is imported inside
  :meth:`CflibLink.open`, not at import time.
* ``crazyflie+ros2://cf21`` — one Crazyswarm2 robot. Services and topics
  are ``/<name>/takeoff``, ``land``, ``go_to``, ``cmd_full_state``, and
  ``emergency``. ``rclpy`` and ``crazyflie_interfaces`` are imported inside
  :meth:`Crazyswarm2Link.open`.

``backend_id`` is ``crazyflie``, so :class:`ohho.safety.SafetyGate` treats
e-stop as a motor cut and writes that backend name on the audit line. The
cut is ``supervisor.send_emergency_stop`` on the radio and the Crazyswarm2
``emergency`` service on ROS. That is the documented exception for a
sub-250 g aircraft in a net (a Crazyflie 2.1 is about 30 g). It is not the
land-then-disarm policy used for PX4.

Crazyswarm2 ``backend:=sim`` does not implement every service it advertises.
The matrix in :data:`SIM_GAPS` matches ``crazyflie_sim/crazyflie_server.py``
on upstream ``main`` as fetched 19 October 2026. This module still *calls*
``emergency`` (the Ohho e-stop contract). It does not call ``cmd_vel_legacy``:
that topic is roll, pitch, yaw rate, and thrust, and the sim callback only
logs ``cmd_vel_legacy not yet implemented``. Velocity, when a position is
already known, is ``cmd_full_state``, which the sim does execute.

Nothing here has flown. CI injects a link and does not open a radio.
``flight_ready`` stays false. ``gps_fix`` stays ``none``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Optional

from ..aerial import (
    Frame,
    convert_vector,
    convert_yaw,
    finite_number,
    require_frame,
)
from ..drone.telemetry import (
    DronePose,
    DroneTelemetry,
    DroneVelocity,
    GpsFix,
    LandedState,
)
from ..registry import RobotSpec
from ..schema import ConnectionState, Odometry, Telemetry, TransportStatus, Velocity
from ..transport import BaseTransport
from .errors import AdapterUnavailable

logger = logging.getLogger(__name__)

BACKEND_ID = "crazyflie"
EXAMPLE_RADIO_URI = "radio://0/80/2M/E7E7E7E7E7"
EXAMPLE_CF_NAME = "cf21"

# High-level polynomials are unstable when the duration is much under a second.
# 2 s is the Crazyswarm hello-world takeoff/land duration. A longer move is
# distance / Ohho cap, and never shorter than this floor.
HL_DURATION_FLOOR_S = 2.0
LAND_HEIGHT_M = 0.0

# Crazyflie 2.1 is a 1S pack. Readings outside this window are not converted
# into a fraction (a 4S number must not look full).
CF21_VBAT_EMPTY_V = 3.0
CF21_VBAT_FULL_V = 4.2
CF21_VOLTAGE_MIN_V = 2.5
CF21_VOLTAGE_MAX_V = 4.5

# crazyflie_interfaces/msg/Status.msg supervisor_info bits.
_SUPERVISOR_IS_ARMED = 2
_SUPERVISOR_IS_FLYING = 16
_PM_STATE_LOW_POWER = 3

# GoTo.srv comments the yaw field as degrees. The Crazyswarm2 Python client
# writes radians straight into that field (its goTo docstring, and the
# np.pi/4 examples). CrazyflieSIL.goTo forwards the same number into
# plan_go_to, which is radians. This adapter sends radians.
GO_TO_YAW_UNIT = "rad"

# What crazyflie_sim/crazyflie_server.py on main (19 October 2026) actually does.
# "Executed" calls CrazyflieSIL. "Log only" logs a line and returns. "Absent"
# is not created on the sim node. cmd_vel_legacy is log-only and is not a
# linear-velocity topic.
SIM_EXECUTED = frozenset(
    {
        "takeoff",
        "land",
        "go_to",
        "cmd_full_state",
        "upload_trajectory",
        "start_trajectory",
    }
)
SIM_LOG_ONLY = frozenset(
    {
        "emergency",
        "notify_setpoints_stop",
        "cmd_vel_legacy",
        "cmd_hover",
    }
)
SIM_ABSENT = frozenset(
    {
        "arm",
        "rtl",
        "cmd_position",
        "cmd_velocity_world",
        "pose",
        "status",
        "parameters",
    }
)

SIM_GAPS: dict[str, str] = {
    "emergency": (
        "The /<name>/emergency service exists. The sim callback logs "
        "'emergency not yet implemented' and does not cut the simulated motors. "
        "Ohho still calls it. On the cflib and cpp backends the same service "
        "cuts the motors. That is the sub-250 g netted-aircraft exception."
    ),
    "cmd_vel_legacy": (
        "The sim subscribes and logs 'cmd_vel_legacy not yet implemented'. "
        "The Twist is pitch, roll, yaw rate, and thrust (degrees and raw "
        "thrust), not a body velocity. Ohho flight methods do not publish it."
    ),
    "cmd_hover": "The sim callback logs 'cmd_hover not yet implemented'.",
    "notify_setpoints_stop": (
        "The sim callback logs 'Notify setpoint stop not yet implemented'."
    ),
    "arm": (
        "The sim server does not create an arm service. The overview table "
        "marks arming n/a. Ohho does not call it on crazyflie+ros2://."
    ),
    "rtl": ("Crazyflie has no return-to-launch service. Ohho does not fly a return."),
    "pose": (
        "Sim default logging does not publish /<name>/pose, so there is no "
        "position feedback. A full-state setpoint needs a pose or a prior go_to."
    ),
    "status": (
        "Sim default logging does not publish /<name>/status, so there is no "
        "battery voltage. SafetyGate refuses takeoff while the fraction is missing."
    ),
    "go_to_from_idle": (
        "CrazyflieSIL.goTo raises unless the mode is already high-level, which "
        "takeoff sets. A go_to before takeoff is ignored by the sim server."
    ),
    "cmd_position": "The sim server does not subscribe to cmd_position.",
    "cmd_velocity_world": "The sim server does not subscribe to cmd_velocity_world.",
    "parameters": "The sim backend does not expose the Crazyflie parameter table.",
}

TRANSPORT_GAPS: dict[str, str] = {
    "cmd_full_state_needs_pose": (
        "cmd_full_state needs a position. Publishing the origin would jump the "
        "setpoint. Sim does not publish pose, and takeoff does not report xy."
    ),
    "hold_needs_pose": (
        "Hold while streaming needs a position for a zero cmd_full_state. "
        "notify_setpoints_stop is log-only in the sim, so it is not called."
    ),
    "set_yaw_needs_pose": (
        "set_yaw is a go_to of the current position. Without xy that would be "
        "a jump to the origin, so nothing is sent."
    ),
    "commands_after_emergency": (
        "After emergency, further setpoints are not sent. A real Crazyflie "
        "latches until reboot. release_stop does not clear that."
    ),
    "planar_velocity": (
        "Robot.drive is planar and is not a Crazyflie setpoint. A zero Twist "
        "is not land and not emergency. cmd_vel_legacy is not used."
    ),
    "body_position": (
        "BODY_FRD is not a Crazyswarm go_to. relative=true is a world offset, "
        "not a body offset."
    ),
}


class CrazyflieUnsupported(ValueError):
    """A setpoint this backend will not translate into a Crazyflie command."""


def hl_duration(distance_m: float, speed_mps: float) -> float:
    """Seconds for a high-level polynomial, never under the 2 s floor."""
    distance = abs(finite_number("distance_m", distance_m))
    speed = finite_number("speed_mps", speed_mps)
    if speed <= 0.0:
        return HL_DURATION_FLOOR_S
    return max(distance / speed, HL_DURATION_FLOOR_S)


def battery_fraction_from_voltage(voltage_v: object) -> Optional[float]:
    """1S state of charge from a pack voltage, or None when it is not 1S-shaped.

    3.0 V is empty and 4.2 V is full. Values outside 2.5–4.5 V are not a
    Crazyflie 2.1 reading and do not become a fraction.
    """
    if isinstance(voltage_v, bool) or not isinstance(voltage_v, (int, float)):
        return None
    if not math.isfinite(voltage_v):
        return None
    if voltage_v < CF21_VOLTAGE_MIN_V or voltage_v > CF21_VOLTAGE_MAX_V:
        return None
    span = CF21_VBAT_FULL_V - CF21_VBAT_EMPTY_V
    fraction = (float(voltage_v) - CF21_VBAT_EMPTY_V) / span
    return min(1.0, max(0.0, fraction))


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """``(x, y, z, w)`` for a z-up yaw in radians. Roll and pitch stay zero."""
    yaw = finite_number("yaw", yaw)
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Yaw about z-up from a quaternion ``(x, y, z, w)``."""
    x = finite_number("x", x)
    y = finite_number("y", y)
    z = finite_number("z", z)
    w = finite_number("w", w)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def _duration_fields(seconds: float) -> tuple[int, int]:
    """``(sec, nanosec)`` for a ROS 2 duration."""
    seconds = max(0.0, finite_number("seconds", seconds))
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return sec, nanosec


def cf_name(address: str) -> str:
    """The robot name in ``crazyflie+ros2://<name>``."""
    name = address.strip().strip("/")
    if not name or "/" in name or name in {".", ".."}:
        raise AdapterUnavailable(
            "crazyflie+ros2:// expects one Crazyflie name, for example "
            "crazyflie+ros2://cf21"
        )
    if not all(char.isalnum() or char in "-_" for char in name):
        raise AdapterUnavailable(
            f"crazyflie+ros2:// name {name!r} must be letters, digits, '_' or '-'"
        )
    return name


def to_crazyswarm_position(
    x: float, y: float, z: float, yaw: float, frame: Frame
) -> tuple[float, float, float, float]:
    """Position and yaw in the Crazyswarm world (z up, yaw CCW from +x).

    That world is treated as ``LOCAL_ENU``. It is the lab frame, not a
    geographic east. ``LOCAL_NED`` is converted. ``BODY_FRD`` is refused:
    a relative go_to is a world offset, not a body offset.
    """
    frame = require_frame(frame)
    x = finite_number("x", x)
    y = finite_number("y", y)
    z = finite_number("z", z)
    yaw = finite_number("yaw", yaw)
    if frame is Frame.BODY_FRD:
        raise CrazyflieUnsupported(TRANSPORT_GAPS["body_position"])
    if frame is Frame.LOCAL_ENU:
        return (x, y, z, yaw)
    ex, ey, ez = convert_vector(x, y, z, source=frame, target=Frame.LOCAL_ENU)
    yaw_enu = convert_yaw(yaw, source=frame, target=Frame.LOCAL_ENU)
    return (ex, ey, ez, yaw_enu)


def to_crazyswarm_velocity(
    vx: float,
    vy: float,
    vz: float,
    yaw_rate: float,
    frame: Frame,
    *,
    yaw_enu: float,
) -> tuple[float, float, float, float]:
    """World velocity (z up, m/s) and a body yaw rate (nose-right, rad/s).

    Linear velocity is expressed in the Crazyswarm world. The yaw rate is
    the body z rate ``cmd_full_state`` stores in ``twist.angular``: positive
    is nose-right, matching ``BODY_FRD`` and ``LOCAL_NED``.
    """
    frame = require_frame(frame)
    vx = finite_number("vx", vx)
    vy = finite_number("vy", vy)
    vz = finite_number("vz", vz)
    yaw_rate = finite_number("yaw_rate", yaw_rate)
    yaw_enu = finite_number("yaw_enu", yaw_enu)
    if frame is Frame.LOCAL_ENU:
        return (vx, vy, vz, -yaw_rate)
    if frame is Frame.LOCAL_NED:
        ex, ey, ez = convert_vector(vx, vy, vz, source=frame, target=Frame.LOCAL_ENU)
        return (ex, ey, ez, yaw_rate)
    yaw_ned = convert_yaw(yaw_enu, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED)
    cos_yaw = math.cos(yaw_ned)
    sin_yaw = math.sin(yaw_ned)
    north = vx * cos_yaw - vy * sin_yaw
    east = vx * sin_yaw + vy * cos_yaw
    ex, ey, ez = convert_vector(
        north, east, vz, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU
    )
    # BODY_FRD yaw rate is already nose-right, so it is not sign-flipped.
    return (ex, ey, ez, yaw_rate)


def _fill_duration(duration: Any, seconds: float) -> None:
    sec, nanosec = _duration_fields(seconds)
    duration.sec = sec
    duration.nanosec = nanosec


class CflibLink:
    """Crazyradio commands through ``cflib``. Tests inject the Crazyflie object.

    :meth:`open` without an injected object imports ``cflib`` and calls
    ``SyncCrazyflie.open_link``. That talks to a USB radio. Unit tests do
    not call that path.
    """

    def __init__(self, radio_uri: str, *, crazyflie: Any = None) -> None:
        if not radio_uri.lower().startswith("radio://"):
            raise AdapterUnavailable(
                "crazyflie:// expects a cflib radio URI, for example "
                f"crazyflie://{EXAMPLE_RADIO_URI}"
            )
        self.radio_uri = radio_uri
        self._cf = crazyflie
        self._sync: Any = None
        self.radio_opened = False

    def open(self) -> None:
        if self._cf is not None:
            return
        try:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        except Exception as exc:
            raise AdapterUnavailable(
                "The Crazyradio adapter needs cflib: pip install 'ohho-os[crazyflie]'. "
                "Nothing was opened."
            ) from exc
        cflib.crtp.init_drivers(enable_debug_driver=False)
        sync = SyncCrazyflie(self.radio_uri, cf=Crazyflie(rw_cache=None))
        sync.open_link()
        self._sync = sync
        self._cf = sync.cf
        self.radio_opened = True

    def close(self) -> None:
        sync = self._sync
        self._sync = None
        if sync is not None:
            close = getattr(sync, "close_link", None)
            if callable(close):
                close()
        self.radio_opened = False

    def sample(self) -> dict[str, Any]:
        """No log block is configured. Pose and battery stay unknown."""
        return {}

    def _cf_or_raise(self) -> Any:
        if self._cf is None:
            raise AdapterUnavailable("cflib link is not open")
        return self._cf

    def takeoff(self, *, height_m: float, duration_s: float) -> None:
        # yaw=None keeps the current heading. The default 0 would yaw to zero.
        self._cf_or_raise().high_level_commander.takeoff(height_m, duration_s, yaw=None)

    def land(self, *, height_m: float, duration_s: float) -> None:
        self._cf_or_raise().high_level_commander.land(height_m, duration_s, yaw=None)

    def go_to(
        self,
        *,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration_s: float,
        relative: bool,
    ) -> None:
        self._cf_or_raise().high_level_commander.go_to(
            x, y, z, yaw, duration_s, relative=relative
        )

    def cmd_full_state(
        self,
        *,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        acc: tuple[float, float, float],
        yaw: float,
        yaw_rate: float,
    ) -> None:
        # cflib rates are degrees/s. Ohho stays in rad/s until this edge.
        qx, qy, qz, qw = yaw_quaternion(yaw)
        self._cf_or_raise().commander.send_full_state_setpoint(
            list(pos),
            list(vel),
            list(acc),
            [qx, qy, qz, qw],
            0.0,
            0.0,
            math.degrees(yaw_rate),
        )

    def velocity_world(
        self, *, vx: float, vy: float, vz: float, yaw_rate: float
    ) -> None:
        self._cf_or_raise().commander.send_velocity_world_setpoint(
            vx, vy, vz, math.degrees(yaw_rate)
        )

    def cmd_vel_legacy(
        self, *, roll_deg: float, pitch_deg: float, yaw_rate_deg_s: float, thrust: int
    ) -> None:
        """RPYT setpoint. Not a velocity, and not used by Ohho flight methods."""
        self._cf_or_raise().commander.send_setpoint(
            roll_deg, pitch_deg, yaw_rate_deg_s, thrust
        )

    def emergency(self) -> None:
        cf = self._cf_or_raise()
        supervisor = getattr(cf, "supervisor", None)
        send = getattr(supervisor, "send_emergency_stop", None)
        if callable(send):
            send()
            return
        legacy = getattr(getattr(cf, "loc", None), "send_emergency_stop", None)
        if callable(legacy):
            legacy()
            return
        raise AdapterUnavailable("cflib Crazyflie has no emergency stop method")

    def arm(self, *, armed: bool) -> None:
        send = getattr(
            getattr(self._cf_or_raise(), "supervisor", None),
            "send_arming_request",
            None,
        )
        if not callable(send):
            raise AdapterUnavailable("cflib Crazyflie has no arming request")
        send(bool(armed))

    def disarm(self) -> None:
        self.arm(armed=False)


def load_crazyswarm_types() -> Any:
    """Import Crazyswarm2 message classes. Failure means the link cannot open."""
    try:
        from crazyflie_interfaces.msg import FullState, Status
        from crazyflie_interfaces.srv import GoTo, Land, Takeoff
        from geometry_msgs.msg import PoseStamped, Twist
        from std_srvs.srv import Empty
    except Exception as exc:
        raise AdapterUnavailable(
            "crazyflie+ros2:// needs rclpy and the Crazyswarm2 "
            "crazyflie_interfaces package on PYTHONPATH. Source ROS 2, "
            "then Robot.connect(..., runtime='ros2'). Nothing was opened."
        ) from exc

    class _Types:
        Empty = Empty
        Takeoff = Takeoff
        Land = Land
        GoTo = GoTo
        FullState = FullState
        Twist = Twist
        PoseStamped = PoseStamped
        Status = Status

    return _Types()


class Crazyswarm2Link:
    """One robot's Crazyswarm2 services. ``node`` is an rclpy node or a fake.

    ``types`` is injected in tests. The real classes load only when it is
    omitted. This object does not start a radio.
    """

    def __init__(self, name: str, node: Any, *, types: Any = None) -> None:
        self.name = cf_name(name)
        self._node = node
        self._types = types
        self._clients: dict[str, Any] = {}
        self._pubs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._pose: Optional[tuple[float, float, float, float]] = None
        self._voltage_v: Optional[float] = None
        self._armed: Optional[bool] = None
        self._flying: Optional[bool] = None
        self._failsafe: list[str] = []
        self.published: list[str] = []

    def open(self) -> None:
        if self._node is None:
            raise AdapterUnavailable(
                "crazyflie+ros2:// needs a Ros2Runtime node. "
                "Connect with runtime='ros2'."
            )
        types = self._types if self._types is not None else load_crazyswarm_types()
        self._types = types
        prefix = f"/{self.name}"
        self._clients["emergency"] = self._node.create_client(
            types.Empty, prefix + "/emergency"
        )
        self._clients["takeoff"] = self._node.create_client(
            types.Takeoff, prefix + "/takeoff"
        )
        self._clients["land"] = self._node.create_client(types.Land, prefix + "/land")
        self._clients["go_to"] = self._node.create_client(types.GoTo, prefix + "/go_to")
        self._pubs["cmd_full_state"] = self._node.create_publisher(
            types.FullState, prefix + "/cmd_full_state", 1
        )
        # Created so the topic exists. Flight methods do not publish it:
        # backend:=sim only logs, and the payload is not a velocity.
        self._pubs["cmd_vel_legacy"] = self._node.create_publisher(
            types.Twist, prefix + "/cmd_vel_legacy", 10
        )
        self._node.create_subscription(
            types.PoseStamped, prefix + "/pose", self._on_pose, 10
        )
        self._node.create_subscription(
            types.Status, prefix + "/status", self._on_status, 10
        )

    def close(self) -> None:
        return None

    def sample(self) -> dict[str, Any]:
        with self._lock:
            sample: dict[str, Any] = {"failsafe": list(self._failsafe)}
            if self._pose is not None:
                sample["pose_enu"] = self._pose
            if self._voltage_v is not None:
                sample["voltage_v"] = self._voltage_v
            if self._armed is not None:
                sample["armed"] = self._armed
            if self._flying is not None:
                sample["flying"] = self._flying
            return sample

    def _on_pose(self, msg: Any) -> None:
        position = msg.pose.position
        orientation = msg.pose.orientation
        yaw = yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        pose = (
            float(position.x),
            float(position.y),
            float(position.z),
            yaw,
        )
        with self._lock:
            self._pose = pose

    def _on_status(self, msg: Any) -> None:
        voltage = getattr(msg, "battery_voltage", None)
        info = int(getattr(msg, "supervisor_info", 0) or 0)
        pm_state = getattr(msg, "pm_state", None)
        failsafe: list[str] = []
        if pm_state == _PM_STATE_LOW_POWER:
            failsafe.append("battery")
        with self._lock:
            if isinstance(voltage, (int, float)) and not isinstance(voltage, bool):
                self._voltage_v = float(voltage)
            self._armed = bool(info & _SUPERVISOR_IS_ARMED)
            self._flying = bool(info & _SUPERVISOR_IS_FLYING)
            self._failsafe = failsafe

    def _call(self, service: str, request: Any) -> None:
        client = self._clients.get(service)
        if client is None:
            raise AdapterUnavailable(
                f"Crazyswarm2 client /{self.name}/{service} is not open"
            )
        client.call_async(request)

    def takeoff(self, *, height_m: float, duration_s: float) -> None:
        request = self._types.Takeoff.Request()
        request.group_mask = 0
        request.height = float(height_m)
        _fill_duration(request.duration, duration_s)
        self._call("takeoff", request)

    def land(self, *, height_m: float, duration_s: float) -> None:
        request = self._types.Land.Request()
        request.group_mask = 0
        request.height = float(height_m)
        _fill_duration(request.duration, duration_s)
        self._call("land", request)

    def go_to(
        self,
        *,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration_s: float,
        relative: bool,
    ) -> None:
        request = self._types.GoTo.Request()
        request.group_mask = 0
        request.relative = bool(relative)
        request.goal.x = float(x)
        request.goal.y = float(y)
        request.goal.z = float(z)
        # Radians. See GO_TO_YAW_UNIT.
        request.yaw = float(yaw)
        _fill_duration(request.duration, duration_s)
        self._call("go_to", request)

    def cmd_full_state(
        self,
        *,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        acc: tuple[float, float, float],
        yaw: float,
        yaw_rate: float,
    ) -> None:
        msg = self._types.FullState()
        header = msg.header
        header.frame_id = "/world"
        self._stamp(header)
        qx, qy, qz, qw = yaw_quaternion(yaw)
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(yaw_rate)
        msg.acc.x = float(acc[0])
        msg.acc.y = float(acc[1])
        msg.acc.z = float(acc[2])
        self._pubs["cmd_full_state"].publish(msg)
        self.published.append("cmd_full_state")

    def cmd_vel_legacy(
        self, *, roll_deg: float, pitch_deg: float, yaw_rate_deg_s: float, thrust: float
    ) -> None:
        """Publish RPYT. Flight methods do not call this. Sim does not execute it."""
        msg = self._types.Twist()
        # Crazyswarm1 packing, kept by the cmd_vel_legacy topic: linear.x is
        # pitch (degrees), linear.y is roll (degrees), linear.z is thrust,
        # angular.z is yaw rate (degrees/s).
        msg.linear.x = float(pitch_deg)
        msg.linear.y = float(roll_deg)
        msg.linear.z = float(thrust)
        msg.angular.z = float(yaw_rate_deg_s)
        self._pubs["cmd_vel_legacy"].publish(msg)
        self.published.append("cmd_vel_legacy")

    def emergency(self) -> None:
        request = self._types.Empty.Request()
        self._call("emergency", request)

    def _stamp(self, header: Any) -> None:
        get_clock = getattr(self._node, "get_clock", None)
        if not callable(get_clock):
            return
        try:
            header.stamp = get_clock().now().to_msg()
        except Exception:
            return


class CrazyflieTransport(BaseTransport):
    """Crazyflie 2.1 over Crazyradio or Crazyswarm2.

    ``link`` is the test seam. Without it, :meth:`connect` builds a
    :class:`CflibLink` or :class:`Crazyswarm2Link` and, for the radio, opens
    it. Pass ``heartbeat=False`` in tests so no thread starts.
    """

    protocol = "crazyflie"
    backend_id = BACKEND_ID
    backend = BACKEND_ID

    def __init__(
        self,
        spec: RobotSpec,
        address: str,
        *,
        kind: str,
        runtime: Any = None,
        link: Any = None,
        heartbeat: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if kind not in ("cflib", "crazyswarm2"):
            raise ValueError(f"unknown crazyflie kind {kind!r}")
        if spec is None:
            raise AdapterUnavailable("crazyflie transport needs a robot spec")
        self.spec = spec
        self.kind = kind
        self._runtime = runtime
        if kind == "cflib":
            if not str(address).lower().startswith("radio://"):
                raise AdapterUnavailable(
                    "crazyflie:// expects a cflib radio URI, for example "
                    f"crazyflie://{EXAMPLE_RADIO_URI}"
                )
            self.address = str(address)
        else:
            self.address = cf_name(address)
        self._link = link
        self._heartbeat = link is None if heartbeat is None else heartbeat
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = ConnectionState.IDLE
        self._connected_since: Optional[float] = None
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.gaps: list[str] = []
        self._gap_seen: set[str] = set()
        self._cut = False
        self._streaming = False
        self._flight_mode = "unknown"
        self._target_height: Optional[float] = None
        self._cmd_x: Optional[float] = None
        self._cmd_y: Optional[float] = None
        self._cmd_z: Optional[float] = None
        self._yaw_enu = 0.0
        climb = spec.max_climb if spec.max_climb > 0.0 else 1.0
        horizontal = spec.max_lin if spec.max_lin > 0.0 else 2.0
        self._climb = climb
        self._horizontal = horizontal

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        self._ensure_link()
        self._link.open()
        with self._lock:
            self._state = ConnectionState.CONNECTED
            self._connected_since = time.time()
        status = self.status()
        self._emit_status(status)
        self._emit_telemetry(self._snapshot())
        if self._heartbeat and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="ohho-crazyflie-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return status

    def disconnect(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)
        link = self._link
        if link is not None:
            close = getattr(link, "close", None)
            if callable(close):
                close()
        with self._lock:
            self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"crazyflie · {self.kind} · {self.address}",
            connected_since=self._connected_since,
        )

    def read(self) -> Telemetry:
        return self._snapshot()

    def _ensure_link(self) -> None:
        if self._link is not None:
            return
        if self.kind == "cflib":
            self._link = CflibLink(self.address)
            return
        node = None
        if self._runtime is not None:
            node = getattr(self._runtime, "node", None)
        self._link = Crazyswarm2Link(self.address, node)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(0.05):
            if self._state != ConnectionState.CONNECTED:
                return
            try:
                self._emit_telemetry(self._snapshot())
            except Exception:
                logger.exception("crazyflie: heartbeat telemetry failed")

    # ── commands ─────────────────────────────────────────────────────────────
    def send_velocity(self, vel: Velocity) -> None:
        """Planar base command. Not a Crazyflie setpoint.

        The gate sends a zero velocity on refusal. That must not become a
        motor cut, a land, or ``cmd_vel_legacy`` (which the sim ignores and
        which is not a velocity).
        """
        finite_number("linear_x", vel.linear_x)
        finite_number("linear_y", vel.linear_y)
        finite_number("angular_z", vel.angular_z)
        self._note_gap("planar_velocity")

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def send_flight(self, action: str, **fields: object) -> None:
        """One Ohho flight request. Unsupported sim commands are not sent."""
        if self._cut and action != "emergency":
            self._note_gap("commands_after_emergency")
            return
        if action == "arm":
            self._cmd_arm(True)
        elif action == "disarm":
            self._cmd_arm(False)
        elif action == "takeoff":
            self._cmd_takeoff(fields.get("alt_m"))
        elif action == "land":
            self._cmd_land()
        elif action == "hold":
            self._cmd_hold()
        elif action == "rtl":
            self._note_gap("rtl")
        elif action == "offboard_position":
            self._cmd_position(fields)
        elif action == "offboard_velocity":
            self._cmd_velocity(fields)
        elif action == "set_yaw":
            self._cmd_yaw(fields)
        else:
            self._note_gap(action)

    def emergency_stop(self) -> None:
        """Cut the motors. Same path as :meth:`motor_cut`."""
        self.motor_cut()

    def motor_cut(self) -> None:
        """Call the Crazyflie emergency stop once.

        On ROS this is ``/<name>/emergency``. On the radio it is
        ``supervisor.send_emergency_stop`` (legacy ``loc.send_emergency_stop``
        when the supervisor port is absent). The sim server accepts the
        service and does not stop the simulated aircraft; see
        :data:`SIM_GAPS`.
        """
        if self._cut:
            return
        self._cut = True
        self._flight_mode = "locked"
        self._streaming = False
        self._dispatch("emergency")

    def release_stop(self) -> None:
        """Does not clear a Crazyflie emergency latch and does not resume."""
        return None

    def _note_gap(self, name: str) -> None:
        if name in self._gap_seen:
            return
        self._gap_seen.add(name)
        self.gaps.append(name)
        detail = SIM_GAPS.get(name) or TRANSPORT_GAPS.get(name) or name
        logger.warning("crazyflie: %s", detail)

    def _dispatch(self, name: str, **fields: Any) -> None:
        self.sent.append((name, dict(fields)))
        fn = getattr(self._link, name)
        fn(**fields)

    def _cmd_arm(self, armed: bool) -> None:
        # Sim does not create the service. Calling it would time out.
        if self.kind != "cflib":
            self._note_gap("arm")
            return
        if armed:
            self._dispatch("arm", armed=True)
        else:
            self._dispatch("disarm")

    def _cmd_takeoff(self, alt_m: object) -> None:
        altitude = finite_number("alt_m", alt_m)
        if altitude <= 0.0:
            raise ValueError("alt_m must be positive up, in metres")
        duration = hl_duration(altitude, self._climb)
        self._target_height = altitude
        self._flight_mode = "takeoff"
        self._streaming = False
        self._dispatch("takeoff", height_m=altitude, duration_s=duration)

    def _cmd_land(self) -> None:
        height = self._target_height if self._target_height is not None else 0.0
        duration = hl_duration(height, self._climb)
        self._flight_mode = "land"
        self._streaming = False
        self._dispatch("land", height_m=LAND_HEIGHT_M, duration_s=duration)

    def _cmd_hold(self) -> None:
        # Takeoff and go_to already end in a hover. Do not send
        # notify_setpoints_stop: the sim only logs that callback.
        if not self._streaming:
            self._flight_mode = "hover"
            return
        if self.kind == "cflib":
            self._flight_mode = "hover"
            self._dispatch("velocity_world", vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
            return
        pos = self._known_position()
        if pos is None:
            self._note_gap("hold_needs_pose")
            return
        self._flight_mode = "hover"
        self._send_full_state(pos, (0.0, 0.0, 0.0), 0.0)

    def _cmd_position(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        x, y, z, yaw = to_crazyswarm_position(
            finite_number("x", fields.get("x")),
            finite_number("y", fields.get("y")),
            finite_number("z", fields.get("z")),
            finite_number("yaw", fields.get("yaw")),
            frame,
        )
        duration = self._goto_duration(x, y, z)
        self._cmd_x, self._cmd_y, self._cmd_z = x, y, z
        self._yaw_enu = yaw
        self._target_height = z
        self._flight_mode = "goto"
        self._streaming = False
        self._dispatch(
            "go_to",
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            duration_s=duration,
            relative=False,
        )

    def _cmd_velocity(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        world_x, world_y, world_z, body_rate = to_crazyswarm_velocity(
            finite_number("vx", fields.get("vx")),
            finite_number("vy", fields.get("vy")),
            finite_number("vz", fields.get("vz")),
            finite_number("yaw_rate", fields.get("yaw_rate")),
            frame,
            yaw_enu=self._current_yaw(),
        )
        if self.kind == "cflib":
            self._flight_mode = "offboard"
            self._streaming = True
            self._dispatch(
                "velocity_world",
                vx=world_x,
                vy=world_y,
                vz=world_z,
                yaw_rate=body_rate,
            )
            return
        pos = self._known_position()
        if pos is None:
            self._note_gap("cmd_full_state_needs_pose")
            return
        self._flight_mode = "offboard"
        self._streaming = True
        self._send_full_state(pos, (world_x, world_y, world_z), body_rate)

    def _cmd_yaw(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        if frame is Frame.BODY_FRD:
            raise CrazyflieUnsupported(TRANSPORT_GAPS["body_position"])
        yaw_in = finite_number("yaw", fields.get("yaw"))
        yaw = to_crazyswarm_position(0.0, 0.0, 0.0, yaw_in, frame)[3]
        pos = self._known_position()
        if pos is None:
            self._note_gap("set_yaw_needs_pose")
            return
        x, y, z = pos
        self._yaw_enu = yaw
        duration = self._goto_duration(x, y, z)
        self._flight_mode = "goto"
        self._streaming = False
        self._dispatch(
            "go_to",
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            duration_s=duration,
            relative=False,
        )

    def _send_full_state(
        self,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float],
        yaw_rate: float,
    ) -> None:
        self._dispatch(
            "cmd_full_state",
            pos=pos,
            vel=vel,
            acc=(0.0, 0.0, 0.0),
            yaw=self._yaw_enu,
            yaw_rate=yaw_rate,
        )

    def _goto_duration(self, x: float, y: float, z: float) -> float:
        current = self._known_position()
        if current is None:
            return HL_DURATION_FLOOR_S
        dx = x - current[0]
        dy = y - current[1]
        dz = z - current[2]
        horizontal_t = abs(math.hypot(dx, dy)) / self._horizontal
        vertical_t = abs(dz) / self._climb if self._climb > 0.0 else 0.0
        return max(horizontal_t, vertical_t, HL_DURATION_FLOOR_S)

    def _current_yaw(self) -> float:
        sample = self._link_sample()
        pose = sample.get("pose_enu")
        if isinstance(pose, (list, tuple)) and len(pose) >= 4:
            return float(pose[3])
        return self._yaw_enu

    def _known_position(self) -> Optional[tuple[float, float, float]]:
        sample = self._link_sample()
        pose = sample.get("pose_enu")
        if isinstance(pose, (list, tuple)) and len(pose) >= 3:
            return (float(pose[0]), float(pose[1]), float(pose[2]))
        if self._cmd_x is None or self._cmd_y is None or self._cmd_z is None:
            return None
        return (self._cmd_x, self._cmd_y, self._cmd_z)

    def _link_sample(self) -> dict[str, Any]:
        link = self._link
        if link is None:
            return {}
        sample = link.sample()
        if not isinstance(sample, dict):
            return {}
        return sample

    # ── telemetry ────────────────────────────────────────────────────────────
    def _snapshot(self) -> Telemetry:
        sample = self._link_sample()
        pose = sample.get("pose_enu")
        measured = isinstance(pose, (list, tuple)) and len(pose) >= 4
        if measured:
            px, py, pz, pyaw = (
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
                float(pose[3]),
            )
        else:
            px = py = pz = pyaw = 0.0
        voltage = sample.get("voltage_v")
        fraction = sample.get("battery")
        if not _valid_fraction(fraction):
            fraction = battery_fraction_from_voltage(voltage)
        voltage_v = 0.0
        if isinstance(voltage, (int, float)) and not isinstance(voltage, bool):
            if math.isfinite(voltage) and float(voltage) > 0.0:
                voltage_v = float(voltage)
        flying = sample.get("flying")
        landed = LandedState.IN_AIR if flying is True else LandedState.UNKNOWN
        armed_sample = sample.get("armed")
        armed = armed_sample is True
        failsafe = sample.get("failsafe") or []
        if not isinstance(failsafe, (list, tuple)):
            failsafe = []
        flags = [str(flag) for flag in failsafe if isinstance(flag, str) and flag]
        with self._lock:
            mode = self._flight_mode
            connected = self._state == ConnectionState.CONNECTED
        drone = DroneTelemetry(
            pose_frame="LOCAL_ENU",
            velocity_frame="LOCAL_ENU",
            armed=armed,
            flight_mode=mode,
            landed_state=landed,
            pose=DronePose(x=px, y=py, z=pz, yaw=pyaw),
            velocity=DroneVelocity(),
            battery=fraction if _valid_fraction(fraction) else None,
            voltage_v=voltage_v,
            current_a=0.0,
            gps_fix=GpsFix.NONE,
            failsafe=flags,
            link_ok=connected,
            link_quality=1.0 if connected else 0.0,
            odom=Odometry(x=px, y=py, theta=pyaw),
            timestamp=time.time(),
        )
        return drone.to_ohho_telemetry()


def _valid_fraction(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= float(value) <= 1.0


__all__ = [
    "BACKEND_ID",
    "CF21_VBAT_EMPTY_V",
    "CF21_VBAT_FULL_V",
    "EXAMPLE_CF_NAME",
    "EXAMPLE_RADIO_URI",
    "GO_TO_YAW_UNIT",
    "HL_DURATION_FLOOR_S",
    "LAND_HEIGHT_M",
    "SIM_ABSENT",
    "SIM_EXECUTED",
    "SIM_GAPS",
    "SIM_LOG_ONLY",
    "TRANSPORT_GAPS",
    "CflibLink",
    "CrazyflieTransport",
    "CrazyflieUnsupported",
    "Crazyswarm2Link",
    "battery_fraction_from_voltage",
    "cf_name",
    "hl_duration",
    "load_crazyswarm_types",
    "to_crazyswarm_position",
    "to_crazyswarm_velocity",
    "yaw_from_quaternion",
    "yaw_quaternion",
]
