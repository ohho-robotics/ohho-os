"""Aerial telemetry carried in ``Telemetry.custom["drone"]`` (OHH-285).

Experimental and not flight-ready. ``to_ohho_telemetry`` always writes
``experimental`` true and ``flight_ready`` false. Nothing in this module sets
``flight_ready`` true.

Pose frames are the plain strings ``LOCAL_NED`` and ``LOCAL_ENU``. OHH-286
adds the Frame enum in ``ohho/aerial.py``; this module does not import it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

from ohho.schema import Odometry, Telemetry

# Pose and velocity frames are plain strings. OHH-286 adds the Frame enum.
LOCAL_NED = "LOCAL_NED"
LOCAL_ENU = "LOCAL_ENU"
# Velocity teleop may use body axes. Pose stays local NED or ENU.
BODY_FRD = "BODY_FRD"

POSE_FRAMES = frozenset({LOCAL_NED, LOCAL_ENU})
VELOCITY_FRAMES = frozenset({LOCAL_NED, LOCAL_ENU, BODY_FRD})

# MAV_LANDED_STATE on EXTENDED_SYS_STATE. 0 and None are unknown, never
# on_ground.
MAV_LANDED_STATE_UNDEFINED = 0
MAV_LANDED_STATE_ON_GROUND = 1
MAV_LANDED_STATE_IN_AIR = 2
MAV_LANDED_STATE_TAKEOFF = 3
MAV_LANDED_STATE_LANDING = 4

_MAV_LANDED_STATE = {
    MAV_LANDED_STATE_UNDEFINED: "unknown",
    MAV_LANDED_STATE_ON_GROUND: "on_ground",
    MAV_LANDED_STATE_IN_AIR: "in_air",
    MAV_LANDED_STATE_TAKEOFF: "takeoff",
    MAV_LANDED_STATE_LANDING: "landing",
}

# GPS_FIX_TYPE values that have a name in the plan table. Anything else,
# including Crazyflie's lack of a fix, is ``none``.
_MAV_GPS_FIX = {
    2: "2d",
    3: "3d",
    4: "dgps",
    5: "rtk_float",
    6: "rtk_fixed",
}

FAILSAFE_FLAGS = (
    "offboard_loss",
    "rc_loss",
    "geofence",
    "battery",
    "datalink",
)

# Keys of Telemetry.custom["drone"], in plan order. Battery fraction is also
# copied onto Telemetry.battery. experimental / flight_ready are fixed.
DRONE_TELEMETRY_KEYS = (
    "armed",
    "flight_mode",
    "landed_state",
    "pose",
    "pose_frame",
    "velocity",
    "velocity_frame",
    "battery",
    "voltage_v",
    "current_a",
    "gps_fix",
    "failsafe",
    "link_ok",
    "link_quality",
    "experimental",
    "flight_ready",
)


class LandedState(str, Enum):
    """Where the autopilot says the vehicle is.

    ``unknown`` means the vehicle has not reported a landed state, or the
    report was not one of the four known values. It is never treated as
    on the ground.
    """

    ON_GROUND = "on_ground"
    IN_AIR = "in_air"
    TAKEOFF = "takeoff"
    LANDING = "landing"
    UNKNOWN = "unknown"


class GpsFix(str, Enum):
    """GNSS fix class. Crazyflie and a missing receiver use ``none``."""

    NONE = "none"
    FIX_2D = "2d"
    FIX_3D = "3d"
    DGPS = "dgps"
    RTK_FLOAT = "rtk_float"
    RTK_FIXED = "rtk_fixed"


class FlightMode(str, Enum):
    """Mode names a PX4-shaped backend may send.

    ``DroneTelemetry.flight_mode`` also accepts any other non-empty string.
    The string stored in telemetry is the backend's own mode table entry.
    """

    DISARMED = "disarmed"
    MANUAL = "manual"
    STABILIZED = "stabilized"
    ALTITUDE = "altitude"
    POSITION = "position"
    OFFBOARD = "offboard"
    TAKEOFF = "takeoff"
    LAND = "land"
    RTL = "rtl"
    HOLD = "hold"
    UNKNOWN = "unknown"


def _require_frame(value: object, allowed: frozenset[str], what: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    allowed_text = ", ".join(sorted(allowed))
    raise ValueError(
        f"{what} must be one of {allowed_text}; got {value!r}. "
        "OHH-286 adds the Frame enum; until then these are plain strings."
    )


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _optional_fraction(value: object, name: str) -> float | None:
    """A fraction in ``[0, 1]``, or None when the backend has no sample."""
    if value is None:
        return None
    number = _finite_float(value, name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _parse_landed_state(value: object) -> LandedState | None:
    """Return a landed state, or None when ``value`` is not a known report.

    ``None``, ``0`` and ``unknown`` are known reports of "not yet known".
    They return ``LandedState.UNKNOWN``, not None.
    """
    if isinstance(value, LandedState):
        return value
    if value is None:
        return LandedState.UNKNOWN
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        name = _MAV_LANDED_STATE.get(value)
        if name is None:
            return None
        return LandedState(name)
    if isinstance(value, str):
        try:
            return LandedState(value.strip().lower())
        except ValueError:
            return None
    return None


def coerce_landed_state(value: object, *, strict: bool = False) -> LandedState:
    """Map a report onto ``LandedState``.

    Unrecognised values raise when ``strict`` is set (telemetry construction).
    The e-stop path uses ``strict=False`` and gets ``unknown``, which is never
    on the ground.
    """
    parsed = _parse_landed_state(value)
    if parsed is None:
        if strict:
            known = ", ".join(state.value for state in LandedState)
            raise ValueError(f"landed_state must be one of {known}; got {value!r}")
        return LandedState.UNKNOWN
    return parsed


def landed_state_from_mav(value: int | None) -> LandedState:
    """Map ``MAV_LANDED_STATE``. Unrecognised values become ``unknown``."""
    return coerce_landed_state(value, strict=False)


def flight_mode_name(mode: FlightMode | str) -> str:
    """Backend mode string. Enum members contribute their value."""
    if isinstance(mode, FlightMode):
        return mode.value
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    raise ValueError("flight_mode must be a non-empty string")


def _coerce_gps(value: GpsFix | str) -> GpsFix:
    if isinstance(value, GpsFix):
        return value
    if isinstance(value, str):
        try:
            return GpsFix(value.strip().lower())
        except ValueError as exc:
            known = ", ".join(fix.value for fix in GpsFix)
            raise ValueError(f"gps_fix must be one of {known}; got {value!r}") from exc
    raise TypeError("gps_fix must be a GpsFix or one of its strings")


def gps_fix_from_mav(fix_type: int) -> GpsFix:
    """Map MAVLink ``GPS_FIX_TYPE``. Unknown codes, including 0 and 1, are ``none``."""
    if isinstance(fix_type, bool) or not isinstance(fix_type, int):
        raise TypeError("gps fix type must be an int")
    return GpsFix(_MAV_GPS_FIX.get(fix_type, GpsFix.NONE.value))


def positive_up_m(z: float, pose_frame: str) -> float:
    """Local-origin height, positive up.

    ``LOCAL_NED`` stores z down, so the positive-up value is ``-z``.
    ``LOCAL_ENU`` stores z up. This is not a global altitude above ground;
    a backend that has a better AGL figure should pass that to e-stop instead.
    """
    frame = _require_frame(pose_frame, POSE_FRAMES, "pose_frame")
    height = _finite_float(z, "z")
    if frame == LOCAL_NED:
        return -height
    return height


def _copy_flags(flags: object) -> tuple[str, ...]:
    if isinstance(flags, str) or not isinstance(flags, (list, tuple)):
        raise TypeError("failsafe must be a list of strings")
    copied: list[str] = []
    for flag in flags:
        if not isinstance(flag, str) or not flag.strip():
            raise ValueError("failsafe flags must be non-empty strings")
        copied.append(flag.strip())
    return tuple(copied)


def _copy_custom(custom: object) -> dict[str, object]:
    if not isinstance(custom, dict):
        raise TypeError("custom must be a dict")
    return dict(custom)


@dataclass(frozen=True, slots=True)
class DronePose:
    """Position and attitude in ``pose_frame``. Units are metres and radians."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True, slots=True)
class DroneVelocity:
    """Velocity in ``velocity_frame``. ``vz`` follows that frame, not a climb sign."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


def _copy_pose(pose: object) -> DronePose:
    if not isinstance(pose, DronePose):
        raise TypeError("pose must be a DronePose")
    return DronePose(
        x=_finite_float(pose.x, "pose.x"),
        y=_finite_float(pose.y, "pose.y"),
        z=_finite_float(pose.z, "pose.z"),
        roll=_finite_float(pose.roll, "pose.roll"),
        pitch=_finite_float(pose.pitch, "pose.pitch"),
        yaw=_finite_float(pose.yaw, "pose.yaw"),
    )


def _copy_velocity(velocity: object) -> DroneVelocity:
    if not isinstance(velocity, DroneVelocity):
        raise TypeError("velocity must be a DroneVelocity")
    return DroneVelocity(
        vx=_finite_float(velocity.vx, "velocity.vx"),
        vy=_finite_float(velocity.vy, "velocity.vy"),
        vz=_finite_float(velocity.vz, "velocity.vz"),
        yaw_rate=_finite_float(velocity.yaw_rate, "velocity.yaw_rate"),
    )


def _copy_odom(odom: object) -> Odometry | None:
    if odom is None:
        return None
    if not isinstance(odom, Odometry):
        raise TypeError("odom must be an Odometry planar shadow or None")
    return Odometry(
        x=_finite_float(odom.x, "odom.x"),
        y=_finite_float(odom.y, "odom.y"),
        theta=_finite_float(odom.theta, "odom.theta"),
        vx=_finite_float(odom.vx, "odom.vx"),
        vy=_finite_float(odom.vy, "odom.vy"),
        omega=_finite_float(odom.omega, "odom.omega"),
    )


def _assign(obj: object, values: dict[str, object]) -> None:
    for name, value in values.items():
        object.__setattr__(obj, name, value)


@dataclass(frozen=True)
class DroneTelemetry:
    """One aerial snapshot, and the mapping into ``ohho.schema.Telemetry``.

    ``pose_frame`` and ``velocity_frame`` are required. There is no default
    that silently picks NED or ENU. ``flight_ready`` is not a field.
    """

    pose_frame: str
    velocity_frame: str
    armed: bool = False
    flight_mode: FlightMode | str = FlightMode.UNKNOWN
    landed_state: LandedState | str | int | None = LandedState.UNKNOWN
    pose: DronePose = field(default_factory=DronePose)
    velocity: DroneVelocity = field(default_factory=DroneVelocity)
    battery: float | None = None
    voltage_v: float = 0.0
    current_a: float = 0.0
    gps_fix: GpsFix | str = GpsFix.NONE
    failsafe: list[str] | tuple[str, ...] = ()
    link_ok: bool = False
    link_quality: float | None = None
    odom: Odometry | None = None
    custom: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _assign(
            self,
            {
                "pose_frame": _require_frame(
                    self.pose_frame, POSE_FRAMES, "pose_frame"
                ),
                "velocity_frame": _require_frame(
                    self.velocity_frame, VELOCITY_FRAMES, "velocity_frame"
                ),
                "armed": _require_bool(self.armed, "armed"),
                "flight_mode": flight_mode_name(self.flight_mode),
                "landed_state": coerce_landed_state(self.landed_state, strict=True),
                "pose": _copy_pose(self.pose),
                "velocity": _copy_velocity(self.velocity),
                "battery": _optional_fraction(self.battery, "battery"),
                "voltage_v": _finite_float(self.voltage_v, "voltage_v"),
                "current_a": _finite_float(self.current_a, "current_a"),
                "gps_fix": _coerce_gps(self.gps_fix),
                "failsafe": _copy_flags(self.failsafe),
                "link_ok": _require_bool(self.link_ok, "link_ok"),
                "link_quality": _optional_fraction(self.link_quality, "link_quality"),
                "odom": _copy_odom(self.odom),
                "custom": _copy_custom(self.custom),
                "timestamp": _finite_float(self.timestamp, "timestamp"),
            },
        )

    def local_altitude_up_m(self) -> float:
        """Positive-up height from ``pose.z`` and ``pose_frame``. Not global AGL."""
        return positive_up_m(self.pose.z, self.pose_frame)

    def planar_odom(self) -> Odometry:
        """Horizontal shadow for recorders that only read ``Telemetry.odom``.

        The shadow is x, y, yaw and the matching horizontal velocity. It is
        not an altitude. ``pose_frame`` and ``velocity_frame`` must already
        be the same frame; body-frame velocity is not written into a local pose.
        """
        if self.pose_frame != self.velocity_frame:
            raise ValueError(
                "planar odom shadow requires pose_frame == velocity_frame "
                f"({self.pose_frame!r} != {self.velocity_frame!r})"
            )
        return Odometry(
            x=self.pose.x,
            y=self.pose.y,
            theta=self.pose.yaw,
            vx=self.velocity.vx,
            vy=self.velocity.vy,
            omega=self.velocity.yaw_rate,
        )

    def as_custom(self) -> dict[str, object]:
        """The ``custom["drone"]`` object. Keys match the OHH-285 table.

        ``experimental`` is true and ``flight_ready`` is false on every call.
        ``landed_state`` is copied from the snapshot; ``unknown`` is not
        rewritten to ``on_ground``.
        """
        landed = self.landed_state
        gps = self.gps_fix
        mode = self.flight_mode
        if not isinstance(landed, LandedState):
            raise RuntimeError("landed_state was not normalised")
        if not isinstance(gps, GpsFix):
            raise RuntimeError("gps_fix was not normalised")
        if not isinstance(mode, str):
            raise RuntimeError("flight_mode was not normalised")
        payload: dict[str, object] = {
            "armed": self.armed,
            "flight_mode": mode,
            "landed_state": landed.value,
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "z": self.pose.z,
                "roll": self.pose.roll,
                "pitch": self.pose.pitch,
                "yaw": self.pose.yaw,
            },
            "pose_frame": self.pose_frame,
            "velocity": {
                "vx": self.velocity.vx,
                "vy": self.velocity.vy,
                "vz": self.velocity.vz,
                "yaw_rate": self.velocity.yaw_rate,
            },
            "velocity_frame": self.velocity_frame,
            "battery": self.battery,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "gps_fix": gps.value,
            "failsafe": list(self.failsafe),
            "link_ok": self.link_ok,
            "link_quality": self.link_quality,
            "experimental": True,
            "flight_ready": False,
        }
        if tuple(payload) != DRONE_TELEMETRY_KEYS:
            raise RuntimeError("drone custom keys do not match the OHH-285 table")
        return payload

    def to_ohho_telemetry(self, *, include_planar_odom: bool = False) -> Telemetry:
        """Map this snapshot into the shared ``Telemetry`` type.

        Battery fraction is ``Telemetry.battery`` and ``custom["drone"]["battery"]``.
        ``include_planar_odom`` fills ``Telemetry.odom`` from :meth:`planar_odom`
        when the caller did not supply a shadow. An explicit ``odom`` wins.
        """
        odom = self.odom
        if odom is None and include_planar_odom:
            odom = self.planar_odom()
        custom = {key: value for key, value in self.custom.items() if key != "drone"}
        custom["drone"] = self.as_custom()
        return Telemetry(
            odom=odom,
            battery=self.battery,
            custom=custom,
            timestamp=self.timestamp,
        )


__all__ = [
    "BODY_FRD",
    "DRONE_TELEMETRY_KEYS",
    "FAILSAFE_FLAGS",
    "LOCAL_ENU",
    "LOCAL_NED",
    "MAV_LANDED_STATE_IN_AIR",
    "MAV_LANDED_STATE_LANDING",
    "MAV_LANDED_STATE_ON_GROUND",
    "MAV_LANDED_STATE_TAKEOFF",
    "MAV_LANDED_STATE_UNDEFINED",
    "POSE_FRAMES",
    "VELOCITY_FRAMES",
    "DronePose",
    "DroneTelemetry",
    "DroneVelocity",
    "FlightMode",
    "GpsFix",
    "LandedState",
    "coerce_landed_state",
    "flight_mode_name",
    "gps_fix_from_mav",
    "landed_state_from_mav",
    "positive_up_m",
]
