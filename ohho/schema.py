"""Core data types shared across OhhO OS.

Ported from the TypeScript transport layer (website/lib/connect/types.ts) so the
SDK and the web consoles speak the same shapes. Pure standard library — no
third-party dependencies, importable anywhere.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Snapshot id shared with RobotDescriptor telemetry blocks.
TELEMETRY_SCHEMA_ID = "ohho.telemetry.v1"

_TELEMETRY_KEYS = frozenset(
    {"schema_id", "odom", "joints", "battery", "scan", "custom", "timestamp"}
)
_ODOM_KEYS = ("x", "y", "theta", "vx", "vy", "omega")


class ConnectionState(str, Enum):
    """Lifecycle of a transport link."""

    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class Velocity:
    """A base velocity command. linear_y is honoured only on holonomic bases."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class Odometry:
    """Robot pose + body-frame velocity estimate."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


@dataclass
class JointReading:
    name: str
    position: float
    velocity: Optional[float] = None


@dataclass
class Scan:
    """A planar range scan in the robot frame (lidar-like, sim or synthesized).

    Ray ``k`` points at ``angle_min + k * angle_increment`` radians relative to
    the robot's heading; ``ranges[k]`` is the hit distance in metres, or
    ``range_max`` when nothing was hit.
    """

    angle_min: float
    angle_increment: float
    ranges: list[float] = field(default_factory=list)
    range_max: float = 4.0


@dataclass
class Telemetry:
    """One best-effort snapshot of a robot's state. All fields optional.

    ``custom["humanoid"]`` is the generic humanoid block in
    ``ohho.telemetry.v1``: ``mode``, ``vendor_mode``, ``fall_state``,
    ``base_rpy``, ``base_ang_vel``, ``joints``, ``odom``, ``battery`` and
    ``experimental``. :mod:`ohho.humanoid_telemetry` builds and checks it.
    When that block carries a planar odom (``x``, ``y``, ``yaw``), constructing
    or loading a snapshot copies it onto :attr:`odom` (``theta`` is yaw) so
    recorder and console code keep reading pose from :attr:`odom`. Body
    velocities already on :attr:`odom` stay. An unknown or missing fall state
    is not interpreted here and is never treated as upright.
    """

    odom: Optional[Odometry] = None
    joints: list[JointReading] = field(default_factory=list)
    battery: Optional[float] = None  # 0..1 fraction when known
    scan: Optional[Scan] = None  # planar range scan, when the robot has one
    custom: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        shadow_humanoid_odom(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready ``ohho.telemetry.v1`` object."""
        return {
            "schema_id": TELEMETRY_SCHEMA_ID,
            "odom": None if self.odom is None else _odom_to_dict(self.odom),
            "joints": [_joint_to_dict(joint) for joint in self.joints],
            "battery": self.battery,
            "scan": None if self.scan is None else _scan_to_dict(self.scan),
            "custom": _json_ready(self.custom),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Telemetry:
        """Load a snapshot. A valid humanoid planar odom is shadowed onto ``odom``."""
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
            raise TypeError("telemetry must be an object")
        extra = set(raw) - _TELEMETRY_KEYS
        if extra:
            raise ValueError(f"telemetry unknown keys {sorted(extra)}")
        schema_id = raw.get("schema_id", TELEMETRY_SCHEMA_ID)
        if schema_id != TELEMETRY_SCHEMA_ID:
            raise ValueError(f"telemetry schema_id must be {TELEMETRY_SCHEMA_ID}")
        custom = raw.get("custom", {})
        if custom is None:
            custom = {}
        if not isinstance(custom, dict):
            raise TypeError("telemetry.custom must be an object")
        timestamp = raw.get("timestamp", None)
        return cls(
            odom=_odom_from_dict(raw.get("odom")),
            joints=_joints_from_dict(raw.get("joints", [])),
            battery=_optional_float("battery", raw.get("battery")),
            scan=_scan_from_dict(raw.get("scan")),
            custom=_json_ready(custom),
            timestamp=time.time()
            if timestamp is None
            else _required_float("timestamp", timestamp),
        )


@dataclass
class TransportStatus:
    """Live status of a transport link."""

    protocol: str
    state: ConnectionState
    label: str = ""
    latency_ms: float = 0.0
    msg_rate: float = 0.0
    error: Optional[str] = None
    connected_since: Optional[float] = None


def shadow_humanoid_odom(telemetry: Telemetry) -> None:
    """Copy planar ``x``, ``y``, ``yaw`` from ``custom["humanoid"]["odom"]``.

    ``theta`` on :class:`Odometry` receives yaw. Velocities already stored on
    ``telemetry.odom`` are kept. A missing block, or a planar odom that is not
    three finite numbers, leaves the snapshot unchanged. This does not read
    fall state.
    """
    custom = telemetry.custom
    if isinstance(custom, (str, bytes)) or not isinstance(custom, Mapping):
        return
    block = custom.get("humanoid")
    if isinstance(block, (str, bytes)) or not isinstance(block, Mapping):
        return
    raw = block.get("odom")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        return
    x = _maybe_finite(raw.get("x"))
    y = _maybe_finite(raw.get("y"))
    yaw = _maybe_finite(raw.get("yaw"))
    if x is None or y is None or yaw is None:
        return
    prev = telemetry.odom
    if prev is not None and prev.x == x and prev.y == y and prev.theta == yaw:
        return
    telemetry.odom = Odometry(
        x=x,
        y=y,
        theta=yaw,
        vx=0.0 if prev is None else prev.vx,
        vy=0.0 if prev is None else prev.vy,
        omega=0.0 if prev is None else prev.omega,
    )


def _odom_to_dict(odom: Odometry) -> dict[str, float]:
    return {
        "x": odom.x,
        "y": odom.y,
        "theta": odom.theta,
        "vx": odom.vx,
        "vy": odom.vy,
        "omega": odom.omega,
    }


def _joint_to_dict(joint: JointReading) -> dict[str, Any]:
    return {"name": joint.name, "position": joint.position, "velocity": joint.velocity}


def _scan_to_dict(scan: Scan) -> dict[str, Any]:
    return {
        "angle_min": scan.angle_min,
        "angle_increment": scan.angle_increment,
        "ranges": list(scan.ranges),
        "range_max": scan.range_max,
    }


def _odom_from_dict(raw: Any) -> Optional[Odometry]:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError("odom must be an object")
    extra = set(raw) - set(_ODOM_KEYS)
    if extra:
        raise ValueError(f"odom unknown keys {sorted(extra)}")
    values = {
        key: _required_float(f"odom.{key}", raw.get(key, 0.0)) for key in _ODOM_KEYS
    }
    return Odometry(**values)


def _joints_from_dict(raw: Any) -> list[JointReading]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("joints must be a list")
    joints: list[JointReading] = []
    for index, item in enumerate(raw):
        if isinstance(item, (str, bytes)) or not isinstance(item, Mapping):
            raise TypeError(f"joints[{index}] must be an object")
        extra = set(item) - {"name", "position", "velocity"}
        if extra:
            raise ValueError(f"joints[{index}] unknown keys {sorted(extra)}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"joints[{index}].name must be a non-empty string")
        velocity = item.get("velocity")
        joints.append(
            JointReading(
                name=name,
                position=_required_float(
                    f"joints[{index}].position", item.get("position")
                ),
                velocity=None
                if velocity is None
                else _required_float(f"joints[{index}].velocity", velocity),
            )
        )
    return joints


def _scan_from_dict(raw: Any) -> Optional[Scan]:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError("scan must be an object")
    extra = set(raw) - {"angle_min", "angle_increment", "ranges", "range_max"}
    if extra:
        raise ValueError(f"scan unknown keys {sorted(extra)}")
    ranges = raw.get("ranges", [])
    if not isinstance(ranges, list):
        raise TypeError("scan.ranges must be a list")
    return Scan(
        angle_min=_required_float("scan.angle_min", raw.get("angle_min")),
        angle_increment=_required_float(
            "scan.angle_increment", raw.get("angle_increment")
        ),
        ranges=[_required_float("scan.ranges", item) for item in ranges],
        range_max=_required_float("scan.range_max", raw.get("range_max", 4.0)),
    )


def _optional_float(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    return _required_float(name, value)


def _required_float(name: str, value: Any) -> float:
    number = _maybe_finite(value)
    if number is None:
        raise ValueError(f"{name} must be a finite number")
    return number


def _maybe_finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _json_ready(value: Any) -> Any:
    """Copy ``value`` into plain JSON types."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("telemetry value must be finite")
        return value
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_ready(to_dict())
    raise TypeError(f"telemetry value {type(value).__name__} is not JSON-compatible")


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the inclusive range [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
