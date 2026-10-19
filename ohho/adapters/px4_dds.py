"""PX4 v1.17 over uXRCE-DDS (``px4+dds://``).

The companion talks to PX4 through a ROS 2 graph that Micro XRCE-DDS Agent
already bridges. This transport creates publishers and subscriptions on an
rclpy node. It does not open a UDP socket to the flight controller, and it
does not start the agent, SITL, or Gazebo.

Pinned firmware is PX4 v1.17.0 with ``px4_msgs`` branch ``release/1.17``.
v1.18 is refused. ROS 2 Jazzy is the distro this was written against; Humble
is the alternative. The agent line is Micro XRCE-DDS Agent 2.4.3.

``dds://`` stays the Unitree CycloneDDS adapter. This scheme is ``px4+dds``.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from ..aerial import (
    Frame,
    convert_vector,
    convert_yaw,
    convert_yaw_rate,
    finite_number,
    require_frame,
)
from ..registry import RobotSpec
from ..schema import (
    ConnectionState,
    Odometry,
    Telemetry,
    TransportStatus,
    Velocity,
)
from ..transport import BaseTransport
from .errors import AdapterUnavailable

# PX4 v1.17.0 / px4_msgs release/1.17. Not a v1.18 release candidate.
PX4_VERSION = "1.17.0"
PX4_MSGS_BRANCH = "release/1.17"
MICRO_XRCE_AGENT_VERSION = "2.4.3"
ROS_DISTRO = "jazzy"
ROS_DISTRO_ALTERNATIVE = "humble"
DEFAULT_AGENT_PORT = 8888
OFFBOARD_PERIOD_S = 0.05  # 20 Hz, which is at least 10 Hz
_STREAM_FRESH_S = 0.2
DISARM_ALTITUDE_M = 0.25
_ARM_ENV = "OHHO_ARM_HARDWARE"
_NS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# OffboardControlMode.msg on release/1.17. `actuator` is the pre-1.17 field.
PINNED_OFFBOARD_FIELDS: dict[str, str] = {
    "timestamp": "uint64",
    "position": "bool",
    "velocity": "bool",
    "acceleration": "bool",
    "attitude": "bool",
    "body_rate": "bool",
    "thrust_and_torque": "bool",
    "direct_actuator": "bool",
}

# Vehicle commands (MAVLink command ids). These are not vehicle_status.nav_state
# values: nav_state 21 is ORBIT and nav_state 22 is AUTO_VTOL_TAKEOFF.
VEHICLE_CMD_NAV_RETURN_TO_LAUNCH = 20
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_NAV_TAKEOFF = 22
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_DO_FLIGHTTERMINATION = 185
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
FORCE_DISARM_PARAM2 = 21196.0

# DO_SET_MODE custom main/sub modes (px4_custom_mode.h), not nav_state numbers.
_MAV_MODE_CUSTOM = 1.0
_CUSTOM_MAIN_AUTO = 4.0
_CUSTOM_MAIN_OFFBOARD = 6.0
_CUSTOM_SUB_AUTO_LOITER = 3.0

# vehicle_status.msg navigation states on release/1.17.
NAVIGATION_STATE_DESCEND = 12
NAVIGATION_STATE_AUTO_TAKEOFF = 17
NAVIGATION_STATE_AUTO_LAND = 18
NAVIGATION_STATE_AUTO_PRECLAND = 20
ARMING_STATE_ARMED = 2

# sensor_gps.msg fix_type on px4_msgs release/1.17. 0 and 1 are both no fix.
_GPS_FIX_BY_TYPE = {
    0: "none",
    1: "none",
    2: "2d",
    3: "3d",
    4: "dgps",
    5: "rtk_float",
    6: "rtk_fixed",
}

_NAV_NAMES = {
    0: "manual",
    1: "altctl",
    2: "posctl",
    3: "auto_mission",
    4: "auto_loiter",
    5: "auto_rtl",
    6: "position_slow",
    8: "altitude_cruise",
    10: "acro",
    12: "descend",
    13: "termination",
    14: "offboard",
    15: "stabilized",
    17: "auto_takeoff",
    18: "auto_land",
    19: "auto_follow_target",
    20: "auto_precland",
    21: "orbit",
    22: "auto_vtol_takeoff",
}

_MSG_NAMES = (
    "OffboardControlMode",
    "TrajectorySetpoint",
    "VehicleCommand",
    "VehicleStatus",
    "VehicleLocalPosition",
    "VehicleAttitude",
    "BatteryStatus",
    "VehicleLandDetected",
    "FailsafeFlags",
)

_OUT_TOPICS = (
    ("vehicle_status", "VehicleStatus", "_on_vehicle_status"),
    ("vehicle_local_position", "VehicleLocalPosition", "_on_local_position"),
    ("vehicle_attitude", "VehicleAttitude", "_on_attitude"),
    ("battery_status", "BatteryStatus", "_on_battery"),
    ("vehicle_land_detected", "VehicleLandDetected", "_on_land"),
    ("failsafe_flags", "FailsafeFlags", "_on_failsafe"),
)

_IN_TOPICS = (
    ("offboard_control_mode", "OffboardControlMode"),
    ("trajectory_setpoint", "TrajectorySetpoint"),
    ("vehicle_command", "VehicleCommand"),
)

# DDS topic suffix for PX4 v1.17.0 / px4_msgs v1.17.0.
# uxrce_dds_client generate_topic_name appends ``_v<N>`` only when
# MESSAGE_VERSION is non-zero. Firmware treats a missing constant as 0
# (dds_topics.h.em MessageVersionHelper), which keeps the bare name.
# The integers below are those pinned message files.
PX4_TOPIC_VERSIONS: dict[str, int] = {
    "offboard_control_mode": 0,  # OffboardControlMode.msg: no MESSAGE_VERSION
    "trajectory_setpoint": 0,  # TrajectorySetpoint.msg: MESSAGE_VERSION = 0
    "vehicle_command": 0,  # VehicleCommand.msg: MESSAGE_VERSION = 0
    "vehicle_status": 1,  # VehicleStatus.msg: MESSAGE_VERSION = 1
    "vehicle_local_position": 1,  # VehicleLocalPosition.msg: MESSAGE_VERSION = 1
    "vehicle_attitude": 0,  # VehicleAttitude.msg: MESSAGE_VERSION = 0
    "battery_status": 1,  # BatteryStatus.msg: MESSAGE_VERSION = 1
    "vehicle_land_detected": 0,  # VehicleLandDetected.msg: MESSAGE_VERSION = 0
    "failsafe_flags": 0,  # FailsafeFlags.msg: no MESSAGE_VERSION
    # SensorGps.msg has no MESSAGE_VERSION on release/1.17, so the topic
    # stays bare. Read-only connect and the doctor subscribe to it.
    "sensor_gps": 0,
}

_TYPE_ALIASES = {"boolean": "bool", "bool": "bool", "uint64": "uint64"}
_NAN = float("nan")


class Px4DdsError(AdapterUnavailable):
    """The PX4 DDS link cannot be used.

    Raised for a bad URI, a non-loopback host without hardware arming, a
    ``px4_msgs`` field mismatch, or a missing ROS 2 / ``px4_msgs`` install.
    """


class Px4CommandRefused(Px4DdsError):
    """A flight command was refused. No arm, disarm, or termination was sent."""


@dataclass(frozen=True)
class Px4Qos:
    """QoS for the PX4 graph.

    Reliability is best-effort and depth is 1 on every endpoint. Publishers
    that PX4 subscribes to are volatile. Subscriptions to PX4 are
    transient-local. This is not the depth-10 reliable profile used by
    :class:`~ohho.adapters.ros2.Ros2Transport`.
    """

    reliability: str
    durability: str
    depth: int

    def as_rclpy(self) -> Any:
        from rclpy.qos import (  # type: ignore
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )

        durability = {
            "volatile": DurabilityPolicy.VOLATILE,
            "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
        }[self.durability]
        if self.reliability != "best_effort":
            raise Px4DdsError("PX4 QoS reliability is best-effort")
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.depth,
        )


PUBLISHER_QOS = Px4Qos(reliability="best_effort", durability="volatile", depth=1)
SUBSCRIPTION_QOS = Px4Qos(
    reliability="best_effort", durability="transient_local", depth=1
)


def qos_for_node(spec: Px4Qos) -> Any:
    """Return an rclpy ``QoSProfile`` when rclpy is installed, else ``spec``."""
    try:
        return spec.as_rclpy()
    except ImportError:
        return spec


def is_sim_target(host: str, port: Optional[int] = None) -> bool:
    """True when ``host`` is a loopback address.

    The decision is the parsed host. ``port`` is the agent port and is not
    part of the decision: port 8888 on a non-loopback host is still hardware.
    The URI scheme is not an argument, so a host cannot become a sim target
    by being named ``sim`` or ``dds``.
    """
    if port is not None and (
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
    ):
        raise Px4DdsError("px4+dds port must be an int in 1..65535")
    return _loopback_host(host)


def parse_px4_dds_uri(uri: str) -> tuple[str, int, str]:
    """Parse ``px4+dds://HOST:PORT?ns=NAME``.

    Returns ``(host, port, namespace)``. The namespace is empty when ``ns``
    is omitted. Credentials are rejected so a secret cannot sit in the URI.
    """
    try:
        parsed = urlparse(uri)
    except ValueError as exc:
        raise Px4DdsError(f"invalid px4+dds URI: {exc}") from exc
    if parsed.scheme != "px4+dds":
        raise Px4DdsError(
            f"URI scheme {parsed.scheme!r} is not px4+dds. "
            "dds:// is the Unitree adapter and is not accepted here."
        )
    if parsed.username or parsed.password:
        raise Px4DdsError("px4+dds URI does not take credentials")
    host = parsed.hostname
    if not host:
        raise Px4DdsError("px4+dds URI needs a host, for example 127.0.0.1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise Px4DdsError(f"invalid px4+dds port: {exc}") from exc
    if port is None:
        port = DEFAULT_AGENT_PORT
    query = parse_qs(parsed.query, keep_blank_values=True)
    unknown = sorted(set(query) - {"ns"})
    if unknown:
        raise Px4DdsError(
            "px4+dds URI query only accepts ns. Unknown keys: " + ", ".join(unknown)
        )
    ns_values = query.get("ns", [])
    if len(ns_values) > 1:
        raise Px4DdsError("px4+dds URI has a repeated ns key")
    namespace = ns_values[0].strip() if ns_values else ""
    if namespace and _NS_RE.fullmatch(namespace) is None:
        raise Px4DdsError(f"px4+dds namespace {namespace!r} must be one ROS name token")
    return host, port, namespace


def offboard_field_mismatch(fields: Mapping[str, str]) -> Optional[str]:
    """Error text when ``fields`` are not the release/1.17 OffboardControlMode.

    ``None`` means the pin matches. ``boolean`` and ``bool`` are the same type.
    """
    got = _normalize_fields(fields)
    expected = _normalize_fields(PINNED_OFFBOARD_FIELDS)
    if got == expected:
        return None
    return (
        "px4_msgs OffboardControlMode does not match "
        f"{PX4_MSGS_BRANCH} (PX4 v{PX4_VERSION}). "
        f"Expected {expected}. Got {got}. "
        "The v1.17 message has thrust_and_torque and direct_actuator and has "
        "no actuator field. Refusing to connect. v1.18 is not supported."
    )


def read_offboard_fields(cls: Any) -> dict[str, str]:
    """Field name to type string for an ``OffboardControlMode`` class."""
    getter = getattr(cls, "get_fields_and_field_types", None)
    raw: Any = None
    if callable(getter):
        raw = getter()
    if not isinstance(raw, dict):
        raw = getattr(cls, "PX4_FIELD_TYPES", None)
    if not isinstance(raw, dict) or not raw:
        raise Px4DdsError(
            "OffboardControlMode does not expose field types "
            "(get_fields_and_field_types or PX4_FIELD_TYPES). "
            f"Refusing to connect: the {PX4_MSGS_BRANCH} pin cannot be checked."
        )
    return {str(key): str(value) for key, value in raw.items()}


def check_vehicle_command(
    command: int, param1: float, param2: float, param3: float
) -> None:
    """Refuse flight termination, force-disarm, and any command we do not send.

    Normal arm is command 400 with param1 1 and param2 0. Normal disarm is
    command 400 with param1 0 and param2 0. param2 21196 is the force-disarm
    magic and is refused for every command.
    """
    if command == VEHICLE_CMD_DO_FLIGHTTERMINATION:
        raise Px4CommandRefused(
            "flight termination (vehicle command 185) is never sent"
        )
    if param2 == FORCE_DISARM_PARAM2:
        raise Px4CommandRefused("force-disarm param2 21196 is never sent")
    if command == VEHICLE_CMD_COMPONENT_ARM_DISARM:
        if param1 not in (0.0, 1.0) or param2 != 0.0 or param3 != 0.0:
            raise Px4CommandRefused(
                "arm/disarm is param1 1 or 0 with param2 0. "
                "force-disarm param2 21196 is never sent"
            )
        return
    if command == VEHICLE_CMD_DO_SET_MODE:
        offboard = (
            param1 == _MAV_MODE_CUSTOM
            and param2 == _CUSTOM_MAIN_OFFBOARD
            and param3 == 0.0
        )
        loiter = (
            param1 == _MAV_MODE_CUSTOM
            and param2 == _CUSTOM_MAIN_AUTO
            and param3 == _CUSTOM_SUB_AUTO_LOITER
        )
        if not offboard and not loiter:
            raise Px4CommandRefused(
                "DO_SET_MODE is only offboard (main 6) or auto-loiter (main 4, sub 3)"
            )
        return
    if command in (
        VEHICLE_CMD_NAV_LAND,
        VEHICLE_CMD_NAV_TAKEOFF,
        VEHICLE_CMD_NAV_RETURN_TO_LAUNCH,
    ):
        return
    raise Px4CommandRefused(
        f"vehicle command {command} is not one this transport sends"
    )


def estop_action(landed_state: str, altitude_m: Optional[float]) -> str:
    """What an e-stop sends: ``disarm``, ``hold_then_land``, or ``land``.

    Disarm only when the vehicle is on the ground and altitude is at most
    0.25 m. Unknown is never treated as on the ground: hold, then land.
    In air, takeoff, and landing send land. Nothing in this table disarms
    in the air or sends force-disarm.
    """
    low = (
        altitude_m is not None
        and math.isfinite(altitude_m)
        and 0.0 <= altitude_m <= DISARM_ALTITUDE_M
    )
    if landed_state == "on_ground" and low:
        return "disarm"
    if landed_state == "unknown":
        return "hold_then_land"
    return "land"


def message_version(msg_cls: Any, leaf: str) -> int:
    """DDS suffix version for one pinned topic. ``0`` means no ``_vN``.

    The message class ``MESSAGE_VERSION`` is used when the definition
    exposes it. Without ``px4_msgs`` that constant is missing, and
    :data:`PX4_TOPIC_VERSIONS` is the version. A constant that disagrees
    with the pin raises :class:`Px4DdsError`. This does not subscribe to
    the bare name instead.
    """
    try:
        pinned = PX4_TOPIC_VERSIONS[leaf]
    except KeyError as exc:
        raise Px4DdsError(
            f"no pinned PX4 v{PX4_VERSION} MESSAGE_VERSION for topic {leaf!r}"
        ) from exc
    raw = getattr(msg_cls, "MESSAGE_VERSION", None)
    if raw is None:
        return pinned
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise Px4DdsError(
            f"{leaf} MESSAGE_VERSION must be a non-negative int, got {raw!r}"
        )
    if raw != pinned:
        name = getattr(msg_cls, "__name__", type(msg_cls).__name__)
        raise Px4DdsError(
            f"{name}.MESSAGE_VERSION is {raw}, but px4_msgs {PX4_MSGS_BRANCH} "
            f"(PX4 v{PX4_VERSION}) pins {leaf} at {pinned}. "
            "Refusing to subscribe to a guessed topic."
        )
    return raw


def fmu_topic(namespace: str, direction: str, name: str, *, version: int) -> str:
    """``/fmu/out/vehicle_status_v1`` or ``/{ns}/fmu/in/vehicle_command``.

    ``version`` 0 is the bare name. A positive version is the ``_vN``
    suffix PX4 v1.17 publishes. The version is required so a caller cannot
    omit it and land on the unversioned topic.
    """
    if direction not in ("in", "out"):
        raise Px4DdsError(f"bad fmu direction {direction!r}")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise Px4DdsError(f"bad MESSAGE_VERSION {version!r} for {name}")
    leaf = name if version == 0 else f"{name}_v{version}"
    if namespace:
        return f"/{namespace}/fmu/{direction}/{leaf}"
    return f"/fmu/{direction}/{leaf}"


def attitude_rpy(q: Any) -> Optional[tuple[float, float, float]]:
    """Roll, pitch, yaw from a Hamilton ``(w, x, y, z)`` body-FRD to NED quaternion."""
    try:
        parts = [float(v) for v in q]
    except (TypeError, ValueError):
        return None
    if len(parts) != 4 or not all(math.isfinite(v) for v in parts):
        return None
    w, x, y, z = parts
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-6:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return (roll, pitch, yaw)


def rotate_body_to_ned(
    q: Any, vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate a body-FRD vector into NED with the vehicle attitude."""
    rpy = attitude_rpy(q)
    if rpy is None:
        raise Px4CommandRefused(
            "BODY_FRD needs a finite vehicle_attitude quaternion before it "
            "can be published as a NED TrajectorySetpoint"
        )
    w, x, y, z = _unit_quat(q)
    vx, vy, vz = vector
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def landed_state_of(land_msg: Any, nav_state: Optional[int]) -> str:
    """``on_ground``, ``in_air``, ``takeoff``, ``landing``, or ``unknown``.

    Missing land detection is ``unknown``, which is never on the ground.
    ``maybe_landed`` without ``landed`` stays in the air.
    """
    if land_msg is None or not hasattr(land_msg, "landed"):
        return "unknown"
    if bool(land_msg.landed):
        return "on_ground"
    if nav_state == NAVIGATION_STATE_AUTO_TAKEOFF:
        return "takeoff"
    if nav_state in (
        NAVIGATION_STATE_AUTO_LAND,
        NAVIGATION_STATE_DESCEND,
        NAVIGATION_STATE_AUTO_PRECLAND,
    ):
        return "landing"
    return "in_air"


def altitude_m_of(local: Any) -> Optional[float]:
    """Positive-up metres used by the disarm gate.

    Valid ``dist_bottom`` is height above the surface. Valid NED ``z``
    contributes ``max(0, -z)``, height above the local origin. When both
    exist, the larger one is used, so a low sensor reading cannot disarm a
    vehicle that is still high. ``None`` means neither reading is valid.
    """
    if local is None:
        return None
    readings: list[float] = []
    if bool(getattr(local, "dist_bottom_valid", False)):
        dist = _finite(getattr(local, "dist_bottom", None))
        if dist is not None and dist >= 0.0:
            readings.append(dist)
    if bool(getattr(local, "z_valid", False)):
        z = _finite(getattr(local, "z", None))
        if z is not None:
            readings.append(max(0.0, -z))
    if not readings:
        return None
    return max(readings)


@dataclass(frozen=True)
class _Trajectory:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    yaw: float
    yawspeed: float
    mode: str


def _idle_trajectory() -> _Trajectory:
    nan3 = (_NAN, _NAN, _NAN)
    return _Trajectory(nan3, nan3, _NAN, _NAN, "position")


def _normalize_fields(fields: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, typ in fields.items():
        raw = str(typ).strip().lower()
        out[str(name).strip()] = _TYPE_ALIASES.get(raw, raw)
    return out


def _loopback_host(host: str) -> bool:
    text = host.strip().lower().strip("[]")
    if text.startswith("::ffff:"):
        text = text[len("::ffff:") :]
    if text in {"localhost", "::1", "0:0:0:0:0:0:0:1"}:
        return True
    parts = text.split(".")
    if len(parts) != 4:
        return False
    nums: list[int] = []
    for part in parts:
        if not part.isdigit():
            return False
        number = int(part)
        if number > 255:
            return False
        nums.append(number)
    return nums[0] == 127


def _unit_quat(q: Any) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _nav_state(status: Any) -> Optional[int]:
    if status is None or not hasattr(status, "nav_state"):
        return None
    try:
        return int(status.nav_state)
    except (TypeError, ValueError):
        return None


def _flight_mode(status: Any) -> str:
    nav = _nav_state(status)
    if nav is None:
        return "unknown"
    return _NAV_NAMES.get(nav, f"nav_state_{nav}")


def _armed(status: Any) -> bool:
    if status is None or not hasattr(status, "arming_state"):
        return False
    try:
        return int(status.arming_state) == ARMING_STATE_ARMED
    except (TypeError, ValueError):
        return False


def gps_fix_name(fix_type: Any) -> Optional[str]:
    """Map a PX4 ``sensor_gps.fix_type`` to the Ohho gps_fix string.

    ``None`` means the value was missing or not a release/1.17 fix type.
    That is not reported as ``none``: ``none`` is fix type 0 or 1, observed.
    """
    if isinstance(fix_type, bool) or not isinstance(fix_type, int):
        return None
    return _GPS_FIX_BY_TYPE.get(fix_type)


def _prearm_passed(status: Any) -> bool:
    return bool(
        status is not None and getattr(status, "pre_flight_checks_pass", False) is True
    )


def _failsafe_names(failsafe: Any, status: Any) -> list[str]:
    found: set[str] = set()
    if failsafe is not None:
        if bool(getattr(failsafe, "offboard_control_signal_lost", False)):
            found.add("offboard_loss")
        if bool(getattr(failsafe, "manual_control_signal_lost", False)):
            found.add("rc_loss")
        if bool(getattr(failsafe, "geofence_breached", False)):
            found.add("geofence")
        warning = getattr(failsafe, "battery_warning", 0)
        try:
            warning_on = int(warning) != 0
        except (TypeError, ValueError):
            warning_on = bool(warning)
        if (
            warning_on
            or bool(getattr(failsafe, "battery_unhealthy", False))
            or bool(getattr(failsafe, "battery_low_remaining_time", False))
        ):
            found.add("battery")
        if bool(getattr(failsafe, "gcs_connection_lost", False)):
            found.add("datalink")
    if status is not None and (
        bool(getattr(status, "gcs_connection_lost", False))
        or bool(getattr(status, "high_latency_data_link_lost", False))
    ):
        found.add("datalink")
    order = ("offboard_loss", "rc_loss", "geofence", "battery", "datalink")
    return [name for name in order if name in found]


def _pose(local: Any, attitude: Any) -> dict[str, Optional[float]]:
    rpy = None
    if attitude is not None and hasattr(attitude, "q"):
        rpy = attitude_rpy(attitude.q)
    x = y = z = None
    if local is not None and bool(getattr(local, "xy_valid", False)):
        x = _finite(getattr(local, "x", None))
        y = _finite(getattr(local, "y", None))
    if local is not None and bool(getattr(local, "z_valid", False)):
        z = _finite(getattr(local, "z", None))
    roll = pitch = yaw = None
    if rpy is not None:
        roll, pitch, yaw = rpy
    elif local is not None:
        yaw = _finite(getattr(local, "heading", None))
    return {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw}


def _velocity(local: Any) -> dict[str, Optional[float]]:
    vx = vy = vz = None
    if local is not None and bool(getattr(local, "v_xy_valid", False)):
        vx = _finite(getattr(local, "vx", None))
        vy = _finite(getattr(local, "vy", None))
    if local is not None and bool(getattr(local, "v_z_valid", False)):
        vz = _finite(getattr(local, "vz", None))
    # vehicle_angular_velocity is not in the D-06 subscription set.
    return {"vx": vx, "vy": vy, "vz": vz, "yaw_rate": None}


def _battery_fraction(battery: Any) -> Optional[float]:
    if battery is None:
        return None
    remaining = _finite(getattr(battery, "remaining", None))
    if remaining is None or remaining < 0.0 or remaining > 1.0:
        return None
    return remaining


def _voltage_v(battery: Any) -> Optional[float]:
    if battery is None:
        return None
    number = _finite(getattr(battery, "voltage_v", None))
    if number is None or number <= 0.0:
        return None
    return number


def _current_a(battery: Any) -> Optional[float]:
    """Battery current in amperes. The v1.17 invalid sentinel is -1."""
    if battery is None:
        return None
    number = _finite(getattr(battery, "current_a", None))
    if number is None or number < 0.0:
        return None
    return number


def _shadow(
    pose: Mapping[str, Optional[float]], vel: Mapping[str, Optional[float]]
) -> Odometry:
    def num(value: Optional[float]) -> float:
        return 0.0 if value is None else float(value)

    return Odometry(
        x=num(pose.get("x")),
        y=num(pose.get("y")),
        theta=num(pose.get("yaw")),
        vx=num(vel.get("vx")),
        vy=num(vel.get("vy")),
        omega=0.0,
    )


def _local_position(
    x: float, y: float, z: float, frame: Frame
) -> tuple[float, float, float]:
    frame = require_frame(frame)
    if frame is Frame.BODY_FRD:
        raise Px4CommandRefused(
            "offboard_position in BODY_FRD is refused: TrajectorySetpoint "
            "position is local NED, and a body position is not rotated into it"
        )
    return convert_vector(x, y, z, source=frame, target=Frame.LOCAL_NED)


def _local_yaw(yaw: float, frame: Frame) -> float:
    frame = require_frame(frame)
    if frame is Frame.BODY_FRD:
        raise Px4CommandRefused(
            "absolute yaw in BODY_FRD is refused: yaw is a local heading"
        )
    return convert_yaw(yaw, source=frame, target=Frame.LOCAL_NED)


class Px4DdsTransport(BaseTransport):
    """PX4 uXRCE-DDS transport.

    ``uri`` is ``px4+dds://HOST:PORT?ns=NAME``. ``HOST:PORT`` is where Micro
    XRCE-DDS Agent is expected to be listening. The ROS 2 graph is the link
    this object actually opens. Loopback hosts are sim targets
    (:func:`is_sim_target`). Any other host refuses :meth:`connect` unless
    ``OHHO_ARM_HARDWARE`` is ``1`` in the supplied environment. That check
    does not arm motors. ``read_only=True`` skips it because connect then
    creates no publishers and sends no vehicle command. The hardware-arming
    flag on the object does not change.

    :attr:`sim_target` is the explicit sim classification. ``SafetyGate``
    reads it from :func:`ohho.safety.requires_hardware_arming` and does not
    parse this URI. ``Robot`` flight methods on a gated connection call
    ``command``, which applies the aerial profile and then
    :meth:`send_flight`. ``backend_id`` is ``px4-dds`` so that e-stop stays
    land / hold-then-land / disarm-on-ground and is not treated as a sim
    descent or a Crazyflie motor cut.
    """

    protocol = "px4-dds"
    backend_id = "px4-dds"

    def __init__(
        self,
        spec: Optional[RobotSpec],
        uri: str,
        *,
        runtime: Any = None,
        node: Any = None,
        msg_factory: Any = None,
        environ: Optional[Mapping[str, str]] = None,
        clock: Optional[Callable[[], float]] = None,
        read_only: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.uri = uri
        self.read_only = read_only
        host, port, namespace = parse_px4_dds_uri(uri)
        self.host = host
        self.port = port
        self.namespace = namespace
        self._runtime = runtime
        self._node = node
        self._msg_factory = msg_factory
        self._environ = environ
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self.publisher_qos = PUBLISHER_QOS
        self.subscription_qos = SUBSCRIPTION_QOS
        self.offboard_period_s = OFFBOARD_PERIOD_S
        self.publisher_topics: list[str] = []
        self.subscription_topics: list[str] = []
        self._classes: dict[str, Any] = {}
        self._pub_mode: Any = None
        self._pub_setpoint: Any = None
        self._pub_command: Any = None
        self._subs: list[Any] = []
        self._timer: Any = None
        self._traj = _idle_trajectory()
        self._status_msg: Any = None
        self._local_msg: Any = None
        self._attitude_msg: Any = None
        self._battery_msg: Any = None
        self._land_msg: Any = None
        self._failsafe_msg: Any = None
        self._offboard_published = 0
        self._last_offboard_t: Optional[float] = None
        self._estopped = False
        # Land, RTL, hold, and disarm pause the heartbeat so a setpoint does
        # not fight the mode command. A later offboard trajectory clears it.
        self._suppress_offboard = False
        self._gps_fix: Optional[str] = None
        self.vehicle_commands_sent = 0
        self._state = ConnectionState.IDLE
        self._connected_since: Optional[float] = None

    @property
    def sim_target(self) -> bool:
        """True only for a loopback agent host. The scheme string is not used."""
        return is_sim_target(self.host, self.port)

    @property
    def requires_hardware_arming(self) -> bool:
        """True for every host that is not a loopback sim target."""
        return not self.sim_target

    def connect(self) -> TransportStatus:
        """Create the PX4 graph endpoints and, unless read-only, the heartbeat.

        Refuses a message definition that is not ``px4_msgs`` ``release/1.17``
        before any endpoint is created. A non-loopback host without hardware
        arming is refused the same way, except a read-only connect: that path
        creates no ``/fmu/in`` publisher, sends no ``VehicleCommand``, and
        does not start the offboard heartbeat. ``requires_hardware_arming``
        stays true for a non-loopback host, so the safety gate still treats
        it as hardware.
        """
        if self._state is ConnectionState.CONNECTED:
            raise Px4DdsError("already connected")
        if (
            not self.read_only
            and self.requires_hardware_arming
            and not self._hardware_arming_granted()
        ):
            raise Px4DdsError(
                f"px4+dds://{self.host}:{self.port} is not a loopback sim target. "
                "Refusing to connect without hardware arming "
                f"({_ARM_ENV}=1). No ROS publishers were created."
            )
        classes = self._load_messages()
        mismatch = offboard_field_mismatch(
            read_offboard_fields(classes["OffboardControlMode"])
        )
        if mismatch:
            raise Px4DdsError(mismatch)
        node = self._resolve_node()
        self._classes = classes
        try:
            self._create_graph(node, classes)
        except Exception:
            self._destroy_graph()
            self._state = ConnectionState.ERROR
            raise
        self._state = ConnectionState.CONNECTED
        self._connected_since = time.time()
        status = self.status()
        self._emit_status(status)
        return status

    def disconnect(self) -> None:
        self._destroy_graph()
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        ns = self.namespace or "/"
        mode = " read-only" if self.read_only else ""
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"PX4 DDS {self.host}:{self.port} ns={ns}{mode}",
            connected_since=self._connected_since,
        )

    def read(self) -> Telemetry:
        with self._lock:
            return self._snapshot_locked()

    def send_velocity(self, vel: Velocity) -> None:
        """Planar velocity in local NED. ``vz`` is 0, so this does not climb.

        A zero velocity is hover. It is not land. Land is
        :meth:`send_flight` ``land`` or :meth:`emergency_stop`.
        """
        if self.read_only:
            self._refuse_readonly_velocity(vel)
            return
        if self._estopped:
            return
        self._require_connected()
        vx = finite_number("linear_x", vel.linear_x)
        vy = finite_number("linear_y", vel.linear_y)
        yaw_rate = finite_number("angular_z", vel.angular_z)
        self._set_velocity(vx, vy, 0.0, yaw_rate)

    def send_joint_command(self, name: str, position: float) -> None:
        """No arm on this adapter. The call is ignored."""
        return None

    def send_flight(self, action: str, **fields: Any) -> None:
        """Run one aerial action. Refusals raise :class:`Px4CommandRefused`.

        ``arm`` is refused until the offboard heartbeat has published
        ``OffboardControlMode`` and PX4 has reported pre-arm checks passing.
        ``disarm`` is refused unless the vehicle is on the ground and low.
        ``land`` is vehicle command 21 (NAV_LAND), never a zero setpoint.
        """
        if self.read_only:
            raise Px4CommandRefused(
                "read-only px4+dds connect does not send VehicleCommand, "
                "does not arm, and does not start an offboard heartbeat"
            )
        if self._estopped and action not in ("land",):
            raise Px4CommandRefused("e-stop is latched")
        self._require_connected()
        if action == "arm":
            self._arm()
        elif action == "disarm":
            self._disarm()
        elif action == "takeoff":
            self._takeoff(fields)
        elif action == "land":
            self._publish_land()
        elif action == "rtl":
            self._publish_simple(VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        elif action == "hold":
            self._publish_hold()
        elif action == "offboard_position":
            self._offboard_position(fields)
        elif action == "offboard_velocity":
            self._offboard_velocity(fields)
        elif action == "set_yaw":
            self._set_yaw(fields)
        else:
            raise Px4CommandRefused(f"unknown flight action {action!r}")

    def emergency_stop(self) -> None:
        """Latch e-stop and ask PX4 to land, or to disarm when already down.

        In air, taking off, landing, or on the ground above 0.25 m: NAV_LAND.
        Unknown land state: loiter, then NAV_LAND. On the ground at or below
        0.25 m: a normal disarm (param2 0). Force-disarm 21196 and flight
        termination 185 are not sent. A zero trajectory setpoint is not sent.
        """
        if self.read_only:
            with self._lock:
                self._estopped = True
            raise Px4CommandRefused(
                "read-only px4+dds connect does not publish land, hold, or disarm. "
                "No VehicleCommand was sent."
            )
        with self._lock:
            self._estopped = True
            action = estop_action(self._landed_locked(), self._altitude_locked())
            ready = self._pub_command is not None
        if not ready:
            raise Px4DdsError(
                "e-stop latched, but vehicle_command is not published; "
                "land was not sent"
            )
        if action == "disarm":
            self._publish_disarm()
        elif action == "hold_then_land":
            self._publish_hold()
            self._publish_land()
        else:
            self._publish_land()

    def release_stop(self) -> None:
        """Clear the local e-stop flag.

        Does not arm the autopilot and does not send an offboard mode command.
        """
        with self._lock:
            self._estopped = False

    # ── graph ────────────────────────────────────────────────────────────────
    def _hardware_arming_granted(self) -> bool:
        env = os.environ if self._environ is None else self._environ
        return str(env.get(_ARM_ENV, "")).strip() == "1"

    def _load_messages(self) -> dict[str, Any]:
        if self._msg_factory is not None:
            classes = {}
            for name in _MSG_NAMES:
                cls = getattr(self._msg_factory, name, None)
                if cls is None:
                    raise Px4DdsError(f"msg_factory is missing {name}")
                classes[name] = cls
            extra = getattr(self._msg_factory, "SensorGps", None)
            if extra is not None:
                classes["SensorGps"] = extra
            return classes
        try:
            import importlib

            module = importlib.import_module("px4_msgs.msg")
        except ImportError as exc:
            raise Px4DdsError(
                "px4_msgs is not importable. This transport is pinned to "
                f"PX4 v{PX4_VERSION} and px4_msgs branch {PX4_MSGS_BRANCH} "
                "(OffboardControlMode fields thrust_and_torque and "
                "direct_actuator; no actuator field). Source ROS 2 "
                f"{ROS_DISTRO} ({ROS_DISTRO_ALTERNATIVE} is the alternative) "
                "and build that branch beside the node. Micro XRCE-DDS Agent "
                f"{MICRO_XRCE_AGENT_VERSION}. The [ros2] extra does not pip-install "
                "rclpy or px4_msgs. v1.18 is not supported."
            ) from exc
        classes = {}
        for name in _MSG_NAMES:
            cls = getattr(module, name, None)
            if cls is None:
                raise Px4DdsError(
                    f"px4_msgs.msg has no {name}. Checkout {PX4_MSGS_BRANCH}."
                )
            classes[name] = cls
        extra = getattr(module, "SensorGps", None)
        if extra is not None:
            classes["SensorGps"] = extra
        return classes

    def _resolve_node(self) -> Any:
        if self._node is not None:
            return self._node
        if self._runtime is not None:
            node = getattr(self._runtime, "node", None)
            if node is not None:
                self._node = node
                return node
        raise Px4DdsError(
            "Px4DdsTransport needs an rclpy node (pass node= or a Ros2Runtime). "
            f"Source ROS 2 {ROS_DISTRO} ({ROS_DISTRO_ALTERNATIVE} is the alternative)."
        )

    def _create_graph(self, node: Any, classes: dict[str, Any]) -> None:
        if self.read_only:
            self._create_readonly_graph(node, classes)
            return
        create_pub = getattr(node, "create_publisher", None)
        create_sub = getattr(node, "create_subscription", None)
        create_timer = getattr(node, "create_timer", None)
        if (
            not callable(create_pub)
            or not callable(create_sub)
            or not callable(create_timer)
        ):
            raise Px4DdsError(
                "the ROS node does not provide create_publisher, "
                "create_subscription, and create_timer"
            )
        pub_qos = qos_for_node(PUBLISHER_QOS)
        sub_qos = qos_for_node(SUBSCRIPTION_QOS)
        # Resolve every name before creating an endpoint. A version mismatch
        # then leaves the graph empty instead of guessing an unversioned topic.
        pub_topics = {
            leaf: fmu_topic(
                self.namespace,
                "in",
                leaf,
                version=message_version(classes[cls_name], leaf),
            )
            for leaf, cls_name in _IN_TOPICS
        }
        sub_plan: list[tuple[str, str, str]] = []
        for leaf, cls_name, method_name in _OUT_TOPICS:
            topic = fmu_topic(
                self.namespace,
                "out",
                leaf,
                version=message_version(classes[cls_name], leaf),
            )
            sub_plan.append((topic, cls_name, method_name))
        self._pub_mode = create_pub(
            classes["OffboardControlMode"],
            pub_topics["offboard_control_mode"],
            pub_qos,
        )
        self._pub_setpoint = create_pub(
            classes["TrajectorySetpoint"],
            pub_topics["trajectory_setpoint"],
            pub_qos,
        )
        self._pub_command = create_pub(
            classes["VehicleCommand"], pub_topics["vehicle_command"], pub_qos
        )
        self.publisher_topics = [pub_topics[leaf] for leaf, _cls_name in _IN_TOPICS]
        for topic, cls_name, method_name in sub_plan:
            callback = getattr(self, method_name)
            sub = create_sub(classes[cls_name], topic, callback, sub_qos)
            self._subs.append(sub)
            self.subscription_topics.append(topic)
        if not 0 < self.offboard_period_s <= 0.1:
            raise Px4DdsError("offboard heartbeat must be at least 10 Hz")
        self._timer = create_timer(self.offboard_period_s, self._on_heartbeat)

    def _create_readonly_graph(self, node: Any, classes: dict[str, Any]) -> None:
        """Subscriptions only. No ``/fmu/in`` publisher and no heartbeat timer."""
        create_sub = getattr(node, "create_subscription", None)
        if not callable(create_sub):
            raise Px4DdsError(
                "read-only px4+dds needs create_subscription on the ROS node. "
                "No publisher was created."
            )
        sub_qos = qos_for_node(SUBSCRIPTION_QOS)
        self.publisher_topics = []
        for leaf, cls_name, method_name in _OUT_TOPICS:
            topic = fmu_topic(
                self.namespace,
                "out",
                leaf,
                version=message_version(classes[cls_name], leaf),
            )
            callback = getattr(self, method_name)
            sub = create_sub(classes[cls_name], topic, callback, sub_qos)
            self._subs.append(sub)
            self.subscription_topics.append(topic)
        gps_cls = classes.get("SensorGps")
        if gps_cls is not None:
            topic = fmu_topic(
                self.namespace,
                "out",
                "sensor_gps",
                version=message_version(gps_cls, "sensor_gps"),
            )
            sub = create_sub(gps_cls, topic, self._on_gps, sub_qos)
            self._subs.append(sub)
            self.subscription_topics.append(topic)
        self._timer = None

    def _refuse_readonly_velocity(self, vel: Velocity) -> None:
        """A zero from SafetyGate is a no-op. Any other velocity is refused."""
        vx = finite_number("linear_x", vel.linear_x)
        vy = finite_number("linear_y", vel.linear_y)
        yaw = finite_number("angular_z", vel.angular_z)
        if vx == 0.0 and vy == 0.0 and yaw == 0.0:
            return
        raise Px4CommandRefused(
            "read-only px4+dds connect does not publish trajectory setpoints "
            "or any /fmu/in topic"
        )

    def _destroy_graph(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            cancel = getattr(timer, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
        node = self._node
        if node is not None:
            destroy_timer = getattr(node, "destroy_timer", None)
            if callable(destroy_timer) and timer is not None:
                try:
                    destroy_timer(timer)
                except Exception:
                    pass
            destroy_pub = getattr(node, "destroy_publisher", None)
            destroy_sub = getattr(node, "destroy_subscription", None)
            for pub in (self._pub_mode, self._pub_setpoint, self._pub_command):
                if pub is not None and callable(destroy_pub):
                    try:
                        destroy_pub(pub)
                    except Exception:
                        pass
            if callable(destroy_sub):
                for sub in self._subs:
                    try:
                        destroy_sub(sub)
                    except Exception:
                        pass
        self._pub_mode = None
        self._pub_setpoint = None
        self._pub_command = None
        self._subs = []

    def _on_heartbeat(self) -> None:
        if self.read_only:
            return
        with self._lock:
            if (
                self._estopped
                or self._suppress_offboard
                or self._pub_mode is None
                or self._pub_setpoint is None
            ):
                return
            mode = self._make_mode_locked()
            setpoint = self._make_setpoint_locked()
            mode_pub = self._pub_mode
            sp_pub = self._pub_setpoint
        self._publish_stream(mode_pub, sp_pub, mode, setpoint)

    # ── incoming ─────────────────────────────────────────────────────────────
    def _on_vehicle_status(self, msg: Any) -> None:
        self._store("_status_msg", msg)

    def _on_local_position(self, msg: Any) -> None:
        self._store("_local_msg", msg)

    def _on_attitude(self, msg: Any) -> None:
        self._store("_attitude_msg", msg)

    def _on_battery(self, msg: Any) -> None:
        self._store("_battery_msg", msg)

    def _on_land(self, msg: Any) -> None:
        self._store("_land_msg", msg)

    def _on_failsafe(self, msg: Any) -> None:
        self._store("_failsafe_msg", msg)

    def _on_gps(self, msg: Any) -> None:
        name = gps_fix_name(getattr(msg, "fix_type", None))
        with self._lock:
            if name is not None:
                self._gps_fix = name
            snap = self._snapshot_locked()
        self._emit_telemetry(snap)

    def _store(self, attr: str, msg: Any) -> None:
        with self._lock:
            setattr(self, attr, msg)
            snap = self._snapshot_locked()
        self._emit_telemetry(snap)

    def _snapshot_locked(self) -> Telemetry:
        status = self._status_msg
        local = self._local_msg
        pose = _pose(local, self._attitude_msg)
        vel = _velocity(local)
        link_ok = status is not None
        drone: dict[str, Any] = {
            "armed": _armed(status),
            "flight_mode": _flight_mode(status),
            "landed_state": landed_state_of(self._land_msg, _nav_state(status)),
            "pose": pose,
            "pose_frame": "LOCAL_NED",
            "velocity": vel,
            "velocity_frame": "LOCAL_NED",
            "voltage_v": _voltage_v(self._battery_msg),
            "current_a": _current_a(self._battery_msg),
            "gps_fix": self._gps_fix if self._gps_fix is not None else "none",
            "failsafe": _failsafe_names(self._failsafe_msg, status),
            "link_ok": link_ok,
            "link_quality": 1.0 if link_ok else 0.0,
            "experimental": True,
            "flight_ready": False,
        }
        return Telemetry(
            odom=_shadow(pose, vel),
            battery=_battery_fraction(self._battery_msg),
            custom={"drone": drone},
            timestamp=time.time(),
        )

    def _landed_locked(self) -> str:
        return landed_state_of(self._land_msg, _nav_state(self._status_msg))

    def _altitude_locked(self) -> Optional[float]:
        return altitude_m_of(self._local_msg)

    # ── commands ─────────────────────────────────────────────────────────────
    def _require_connected(self) -> None:
        if self._state is not ConnectionState.CONNECTED:
            raise Px4DdsError("not connected")

    def _arm(self) -> None:
        with self._lock:
            if self._estopped:
                raise Px4CommandRefused("e-stop is latched")
            if not self._stream_alive_locked():
                raise Px4CommandRefused(
                    "offboard stream is not alive; OffboardControlMode has not "
                    "been published at >= 10 Hz"
                )
            if not _prearm_passed(self._status_msg):
                raise Px4CommandRefused(
                    "PX4 pre-arm checks have not passed (no vehicle_status, "
                    "or pre_flight_checks_pass is false)"
                )
        self._publish_arm_disarm(1.0)

    def _disarm(self) -> None:
        with self._lock:
            if estop_action(self._landed_locked(), self._altitude_locked()) != "disarm":
                raise Px4CommandRefused(
                    "disarm is refused unless landed_state is on_ground and "
                    f"altitude is <= {DISARM_ALTITUDE_M} m. "
                    "force-disarm 21196 is never sent"
                )
        self._publish_disarm()

    def _takeoff(self, fields: Mapping[str, Any]) -> None:
        if "alt_m" not in fields:
            raise Px4CommandRefused("takeoff requires alt_m, positive up, in metres")
        alt = finite_number("alt_m", fields["alt_m"])
        if alt <= 0.0:
            raise ValueError("alt_m must be positive up, in metres")
        # param7 is the positive-up metres the caller asked for. PX4's MAVLink
        # definition of NAV_TAKEOFF param7 is AMSL; SITL with no GPS origin is
        # the case where that number matches a local climb.
        self._publish_command(VEHICLE_CMD_NAV_TAKEOFF, param7=alt)

    def _offboard_position(self, fields: Mapping[str, Any]) -> None:
        frame = require_frame(fields.get("frame"))
        north, east, down = _local_position(
            fields["x"], fields["y"], fields["z"], frame
        )
        yaw = _local_yaw(fields["yaw"], frame)
        traj = _Trajectory(
            (north, east, down),
            (_NAN, _NAN, _NAN),
            yaw,
            _NAN,
            "position",
        )
        self._install_trajectory(traj, switch_mode=True)

    def _offboard_velocity(self, fields: Mapping[str, Any]) -> None:
        frame = require_frame(fields.get("frame"))
        vx = finite_number("vx", fields["vx"])
        vy = finite_number("vy", fields["vy"])
        vz = finite_number("vz", fields["vz"])
        yaw_rate = finite_number("yaw_rate", fields["yaw_rate"])
        if frame is Frame.BODY_FRD:
            with self._lock:
                quat = getattr(self._attitude_msg, "q", None)
            vx, vy, vz = rotate_body_to_ned(quat, (vx, vy, vz))
            # Body yaw rate is a rotation about body z. TrajectorySetpoint.yawspeed
            # is the NED down component of that vector.
            _nx, _ny, yaw_rate = rotate_body_to_ned(quat, (0.0, 0.0, yaw_rate))
        else:
            vx, vy, vz = convert_vector(
                vx, vy, vz, source=frame, target=Frame.LOCAL_NED
            )
            yaw_rate = convert_yaw_rate(yaw_rate, source=frame, target=Frame.LOCAL_NED)
        self._set_velocity(vx, vy, vz, yaw_rate)

    def _set_yaw(self, fields: Mapping[str, Any]) -> None:
        frame = require_frame(fields.get("frame"))
        yaw = _local_yaw(fields["yaw"], frame)
        traj = _Trajectory(
            (_NAN, _NAN, _NAN),
            (_NAN, _NAN, _NAN),
            yaw,
            _NAN,
            "position",
        )
        self._install_trajectory(traj, switch_mode=True)

    def _set_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        traj = _Trajectory(
            (_NAN, _NAN, _NAN),
            (vx, vy, vz),
            _NAN,
            yaw_rate,
            "velocity",
        )
        self._install_trajectory(traj, switch_mode=True)

    def _install_trajectory(self, traj: _Trajectory, *, switch_mode: bool) -> None:
        with self._lock:
            if self._estopped:
                raise Px4CommandRefused("e-stop is latched")
            self._suppress_offboard = False
            self._traj = traj
            mode = self._make_mode_locked()
            setpoint = self._make_setpoint_locked()
            mode_pub = self._pub_mode
            sp_pub = self._pub_setpoint
        if mode_pub is None or sp_pub is None:
            raise Px4DdsError("offboard publishers are not ready")
        self._publish_stream(mode_pub, sp_pub, mode, setpoint)
        if switch_mode:
            self._publish_offboard_mode()

    def _stream_alive_locked(self) -> bool:
        timer = self._timer
        if timer is None or bool(getattr(timer, "cancelled", False)):
            return False
        if not 0 < self.offboard_period_s <= 0.1:
            return False
        if self._offboard_published < 1 or self._last_offboard_t is None:
            return False
        if self._estopped:
            return False
        return (self._clock() - self._last_offboard_t) <= _STREAM_FRESH_S

    def _make_mode_locked(self) -> Any:
        msg = self._classes["OffboardControlMode"]()
        _assign(msg, timestamp=self._timestamp_us())
        flags = (
            "position",
            "velocity",
            "acceleration",
            "attitude",
            "body_rate",
            "thrust_and_torque",
            "direct_actuator",
        )
        selected = "velocity" if self._traj.mode == "velocity" else "position"
        for name in flags:
            _assign(msg, **{name: name == selected})
        return msg

    def _make_setpoint_locked(self) -> Any:
        msg = self._classes["TrajectorySetpoint"]()
        traj = self._traj
        _assign(
            msg, timestamp=self._timestamp_us(), yaw=traj.yaw, yawspeed=traj.yawspeed
        )
        _set_vec(msg, "position", traj.position)
        _set_vec(msg, "velocity", traj.velocity)
        _set_vec(msg, "acceleration", (_NAN, _NAN, _NAN))
        _set_vec(msg, "jerk", (_NAN, _NAN, _NAN))
        return msg

    def _publish_stream(
        self, mode_pub: Any, sp_pub: Any, mode: Any, setpoint: Any
    ) -> None:
        if self.read_only:
            raise Px4CommandRefused(
                "read-only px4+dds connect does not publish OffboardControlMode"
            )
        mode_pub.publish(mode)
        sp_pub.publish(setpoint)
        with self._lock:
            self._offboard_published += 1
            self._last_offboard_t = self._clock()

    def _publish_offboard_mode(self) -> None:
        self._publish_command(
            VEHICLE_CMD_DO_SET_MODE,
            param1=_MAV_MODE_CUSTOM,
            param2=_CUSTOM_MAIN_OFFBOARD,
            param3=0.0,
        )

    def _pause_offboard_stream(self) -> None:
        """Stop the heartbeat until the next explicit trajectory."""
        with self._lock:
            self._suppress_offboard = True

    def _publish_hold(self) -> None:
        self._pause_offboard_stream()
        self._publish_command(
            VEHICLE_CMD_DO_SET_MODE,
            param1=_MAV_MODE_CUSTOM,
            param2=_CUSTOM_MAIN_AUTO,
            param3=_CUSTOM_SUB_AUTO_LOITER,
        )

    def _publish_land(self) -> None:
        self._publish_simple(VEHICLE_CMD_NAV_LAND)

    def _publish_disarm(self) -> None:
        self._pause_offboard_stream()
        self._publish_arm_disarm(0.0)

    def _publish_arm_disarm(self, param1: float) -> None:
        self._publish_command(
            VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=param1, param2=0.0
        )

    def _publish_simple(self, command: int) -> None:
        if command in (
            VEHICLE_CMD_NAV_LAND,
            VEHICLE_CMD_NAV_RETURN_TO_LAUNCH,
        ):
            self._pause_offboard_stream()
        self._publish_command(command)

    def _publish_command(
        self,
        command: int,
        *,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        if self.read_only:
            raise Px4CommandRefused(
                "read-only px4+dds connect does not send VehicleCommand"
            )
        check_vehicle_command(command, param1, param2, param3)
        pub = self._pub_command
        if pub is None:
            raise Px4DdsError("vehicle_command publisher is not ready")
        msg = self._classes["VehicleCommand"]()
        _assign(
            msg,
            timestamp=self._timestamp_us(),
            param1=float(param1),
            param2=float(param2),
            param3=float(param3),
            param4=float(param4),
            param5=float(param5),
            param6=float(param6),
            param7=float(param7),
            command=int(command),
            target_system=1,
            target_component=1,
            source_system=1,
            source_component=1,
            confirmation=0,
            from_external=True,
        )
        pub.publish(msg)
        self.vehicle_commands_sent += 1

    def _timestamp_us(self) -> int:
        node = self._node
        get_clock = getattr(node, "get_clock", None) if node is not None else None
        if callable(get_clock):
            try:
                return int(get_clock().now().nanoseconds / 1000)
            except Exception:
                pass
        return int(self._clock() * 1_000_000)


def _assign(msg: Any, **fields: Any) -> None:
    for key, value in fields.items():
        try:
            setattr(msg, key, value)
        except AttributeError as exc:
            raise Px4DdsError(
                f"{type(msg).__name__} has no field {key}. "
                f"px4_msgs {PX4_MSGS_BRANCH} is required."
            ) from exc


def _set_vec(msg: Any, field: str, values: tuple[float, float, float]) -> None:
    nums = [float(v) for v in values]
    current = getattr(msg, field, None)
    if current is not None and not isinstance(current, (list, tuple, float, int)):
        try:
            for index, value in enumerate(nums):
                current[index] = value
            return
        except Exception:
            pass
    try:
        setattr(msg, field, nums)
    except AttributeError as exc:
        raise Px4DdsError(
            f"{type(msg).__name__}.{field} cannot be set. "
            f"px4_msgs {PX4_MSGS_BRANCH} is required."
        ) from exc


__all__ = [
    "ARMING_STATE_ARMED",
    "DEFAULT_AGENT_PORT",
    "DISARM_ALTITUDE_M",
    "FORCE_DISARM_PARAM2",
    "MICRO_XRCE_AGENT_VERSION",
    "OFFBOARD_PERIOD_S",
    "PINNED_OFFBOARD_FIELDS",
    "PUBLISHER_QOS",
    "PX4_MSGS_BRANCH",
    "PX4_TOPIC_VERSIONS",
    "PX4_VERSION",
    "Px4CommandRefused",
    "Px4DdsError",
    "Px4DdsTransport",
    "Px4Qos",
    "ROS_DISTRO",
    "ROS_DISTRO_ALTERNATIVE",
    "SUBSCRIPTION_QOS",
    "VEHICLE_CMD_COMPONENT_ARM_DISARM",
    "VEHICLE_CMD_DO_FLIGHTTERMINATION",
    "VEHICLE_CMD_DO_SET_MODE",
    "VEHICLE_CMD_NAV_LAND",
    "VEHICLE_CMD_NAV_RETURN_TO_LAUNCH",
    "VEHICLE_CMD_NAV_TAKEOFF",
    "altitude_m_of",
    "attitude_rpy",
    "check_vehicle_command",
    "estop_action",
    "fmu_topic",
    "gps_fix_name",
    "is_sim_target",
    "landed_state_of",
    "message_version",
    "offboard_field_mismatch",
    "parse_px4_dds_uri",
    "qos_for_node",
    "read_offboard_fields",
    "rotate_body_to_ned",
]
