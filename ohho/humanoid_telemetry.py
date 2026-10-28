"""Generic ``custom["humanoid"]`` block for ``ohho.telemetry.v1``.

Any humanoid backend builds this block and attaches it to :class:`~ohho.schema.Telemetry`.
The planar odom is copied onto ``Telemetry.odom`` so recorder and console code
keep reading pose from the common field. Vendor mode names are stored for logs
and are not interpreted. ``experimental`` stays true until a dated real run.

Joint channels are ``q``, ``dq``, ``tau`` and ``temp``, included only where the
backend reported them. When a spec or ``joint_groups`` mapping is passed, the
group names must match that mapping and each reported channel must have one
value per joint, in that group's order.

An unknown or missing fall state is not upright. Use :func:`ohho.humanoid.is_upright`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from .humanoid import FallState, HumanoidMode, is_upright
from .schema import TELEMETRY_SCHEMA_ID, Telemetry, shadow_humanoid_odom

HUMANOID_BLOCK_KEYS: tuple[str, ...] = (
    "mode",
    "vendor_mode",
    "fall_state",
    "base_rpy",
    "base_ang_vel",
    "joints",
    "odom",
    "battery",
    "experimental",
)

JOINT_CHANNELS: tuple[str, ...] = ("q", "dq", "tau", "temp")

HUMANOID_BLOCK_SCHEMA: dict[str, Any] = {
    "schema_id": TELEMETRY_SCHEMA_ID,
    "path": 'custom["humanoid"]',
    "keys": {
        "mode": "OFF, DAMPING, PREPARE, WALK, CUSTOM, FALLEN, or UNKNOWN",
        "vendor_mode": "vendor mode name, logs only",
        "fall_state": "upright, falling, fallen, recovering, or unknown",
        "base_rpy": "[roll, pitch, yaw] in radians",
        "base_ang_vel": "[roll, pitch, yaw] in radians per second",
        "joints": (
            "per joint group from the spec: q, dq, tau, temp where reported; "
            "index i is joint_groups[group][i]"
        ),
        "odom": "planar x, y, yaw; shadowed onto Telemetry.odom (theta = yaw)",
        "battery": "fraction from 0 to 1, and voltage",
        "experimental": "true until a dated real run",
    },
    "fall_states": [member.value for member in FallState],
    "unknown_fall_state_is_upright": False,
    "odom_shadow": "Telemetry.odom x, y, theta=yaw",
}


@dataclass(frozen=True)
class PlanarOdom:
    """Planar pose. ``yaw`` is radians. Shadowed onto ``Telemetry.odom.theta``."""

    x: float
    y: float
    yaw: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "yaw": float(self.yaw)}


@dataclass(frozen=True)
class HumanoidBattery:
    """Battery fraction (0..1) and voltage."""

    fraction: float
    voltage: float

    def to_dict(self) -> dict[str, float]:
        return {"fraction": float(self.fraction), "voltage": float(self.voltage)}


@dataclass(frozen=True)
class JointGroupSample:
    """Reported samples for one joint group. Absent channels are ``None``."""

    q: Optional[tuple[float, ...]] = None
    dq: Optional[tuple[float, ...]] = None
    tau: Optional[tuple[float, ...]] = None
    temp: Optional[tuple[float, ...]] = None

    def to_dict(self) -> dict[str, list[float]]:
        body: dict[str, list[float]] = {}
        for key in JOINT_CHANNELS:
            value = getattr(self, key)
            if value is not None:
                body[key] = [float(item) for item in value]
        return body


@dataclass(frozen=True)
class HumanoidTelemetry:
    """Validated ``custom["humanoid"]`` block."""

    mode: HumanoidMode
    vendor_mode: str
    fall_state: FallState
    base_rpy: tuple[float, float, float]
    base_ang_vel: tuple[float, float, float]
    joints: Mapping[str, JointGroupSample]
    odom: PlanarOdom
    battery: HumanoidBattery
    experimental: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.joints, MappingProxyType):
            object.__setattr__(self, "joints", MappingProxyType(dict(self.joints)))

    def to_dict(self) -> dict[str, Any]:
        """JSON object with the section 5.2 keys."""
        return {
            "mode": self.mode.value,
            "vendor_mode": self.vendor_mode,
            "fall_state": self.fall_state.value,
            "base_rpy": [float(value) for value in self.base_rpy],
            "base_ang_vel": [float(value) for value in self.base_ang_vel],
            "joints": {name: sample.to_dict() for name, sample in self.joints.items()},
            "odom": self.odom.to_dict(),
            "battery": self.battery.to_dict(),
            "experimental": bool(self.experimental),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        spec: object | None = None,
        joint_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> HumanoidTelemetry:
        """Validate ``raw`` and return the block. Groups are checked when given."""
        return validate_humanoid_telemetry(raw, spec=spec, joint_groups=joint_groups)


def build_humanoid_telemetry(
    *,
    mode: object,
    vendor_mode: str,
    fall_state: object,
    base_rpy: Sequence[float],
    base_ang_vel: Sequence[float],
    joints: Mapping[str, Any],
    odom: Mapping[str, float] | PlanarOdom,
    battery: Mapping[str, float] | HumanoidBattery,
    experimental: bool = True,
    spec: object | None = None,
    joint_groups: Mapping[str, Sequence[str]] | None = None,
) -> HumanoidTelemetry:
    """Build a validated block. ``experimental`` defaults to true.

    Pass ``spec`` or ``joint_groups`` to require the joint groups to match.
    """
    return validate_humanoid_telemetry(
        {
            "mode": mode,
            "vendor_mode": vendor_mode,
            "fall_state": fall_state,
            "base_rpy": base_rpy,
            "base_ang_vel": base_ang_vel,
            "joints": joints,
            "odom": odom,
            "battery": battery,
            "experimental": experimental,
        },
        spec=spec,
        joint_groups=joint_groups,
    )


def validate_humanoid_telemetry(
    raw: Mapping[str, Any] | HumanoidTelemetry,
    *,
    spec: object | None = None,
    joint_groups: Mapping[str, Sequence[str]] | None = None,
) -> HumanoidTelemetry:
    """Check the section 5.2 keys and, when given, the spec's joint groups.

    Raises ``TypeError`` or ``ValueError`` when the block is not the generic
    humanoid shape. Does not read ``vendor_mode`` except to require a string.
    """
    if isinstance(raw, HumanoidTelemetry):
        raw = raw.to_dict()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError("humanoid telemetry must be an object")
    missing = [key for key in HUMANOID_BLOCK_KEYS if key not in raw]
    extra = sorted(set(raw) - set(HUMANOID_BLOCK_KEYS))
    if missing or extra:
        raise _key_error("humanoid telemetry", missing, extra)
    groups = _resolve_groups(spec, joint_groups)
    return HumanoidTelemetry(
        mode=_require_mode(raw["mode"]),
        vendor_mode=_vendor_mode(raw["vendor_mode"]),
        fall_state=_require_fall_state(raw["fall_state"]),
        base_rpy=_vec3(raw["base_rpy"], "base_rpy"),
        base_ang_vel=_vec3(raw["base_ang_vel"], "base_ang_vel"),
        joints=_parse_joints(raw["joints"], groups),
        odom=_planar(raw["odom"]),
        battery=_battery(raw["battery"]),
        experimental=_experimental(raw["experimental"]),
    )


def apply_humanoid_telemetry(
    telemetry: Telemetry, block: HumanoidTelemetry
) -> Telemetry:
    """Store the block and shadow its planar odom onto ``telemetry.odom``."""
    if not isinstance(block, HumanoidTelemetry):
        raise TypeError("block must be HumanoidTelemetry")
    if not isinstance(telemetry.custom, dict):
        raise TypeError("telemetry.custom must be an object")
    telemetry.custom["humanoid"] = block.to_dict()
    shadow_humanoid_odom(telemetry)
    return telemetry


def telemetry_with_humanoid(block: HumanoidTelemetry, **fields: Any) -> Telemetry:
    """A :class:`Telemetry` whose custom block and planar odom match ``block``."""
    if not isinstance(block, HumanoidTelemetry):
        raise TypeError("block must be HumanoidTelemetry")
    custom = fields.pop("custom", None)
    if custom is None:
        custom = {}
    if not isinstance(custom, dict):
        raise TypeError("custom must be an object")
    merged = dict(custom)
    merged["humanoid"] = block.to_dict()
    return Telemetry(custom=merged, **fields)


def dumps_humanoid_telemetry(block: HumanoidTelemetry) -> str:
    """JSON text of a humanoid block."""
    return json.dumps(block.to_dict())


def loads_humanoid_telemetry(
    text: str,
    *,
    spec: object | None = None,
    joint_groups: Mapping[str, Sequence[str]] | None = None,
) -> HumanoidTelemetry:
    """Parse and validate a humanoid block."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("humanoid telemetry must be a JSON object")
    return validate_humanoid_telemetry(data, spec=spec, joint_groups=joint_groups)


def dumps_telemetry(telemetry: Telemetry) -> str:
    """JSON text of an ``ohho.telemetry.v1`` snapshot."""
    return json.dumps(telemetry.to_dict())


def loads_telemetry(
    text: str,
    *,
    spec: object | None = None,
    joint_groups: Mapping[str, Sequence[str]] | None = None,
) -> Telemetry:
    """Load a snapshot. A humanoid block is validated and its odom is shadowed."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("telemetry must be a JSON object")
    telemetry = Telemetry.from_dict(data)
    if isinstance(telemetry.custom, dict) and "humanoid" in telemetry.custom:
        block = telemetry.custom["humanoid"]
        if isinstance(block, (str, bytes)) or not isinstance(block, Mapping):
            raise TypeError("humanoid telemetry must be an object")
        checked = validate_humanoid_telemetry(
            block, spec=spec, joint_groups=joint_groups
        )
        telemetry.custom["humanoid"] = checked.to_dict()
        shadow_humanoid_odom(telemetry)
    return telemetry


def _key_error(where: str, missing: list[str], extra: list[str]) -> ValueError:
    parts: list[str] = []
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"unknown {extra}")
    return ValueError(f"{where} " + ", ".join(parts))


def _resolve_groups(
    spec: object | None,
    joint_groups: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, Sequence[str]] | None:
    if spec is None:
        return joint_groups
    if joint_groups is not None:
        raise TypeError("pass spec or joint_groups, not both")
    groups = getattr(spec, "joint_groups", None)
    if groups is None:
        raise TypeError("spec has no joint_groups")
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Mapping):
        raise TypeError("joint_groups must be a mapping")
    return groups


def _require_mode(value: object) -> HumanoidMode:
    if isinstance(value, HumanoidMode):
        return value
    if isinstance(value, str):
        try:
            return HumanoidMode[value.strip().upper()]
        except KeyError:
            pass
    raise ValueError(
        "mode must be OFF, DAMPING, PREPARE, WALK, CUSTOM, FALLEN, or UNKNOWN"
    )


def _require_fall_state(value: object) -> FallState:
    if isinstance(value, FallState):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        for member in FallState:
            if member.value == key:
                return member
    raise ValueError(
        "fall_state must be upright, falling, fallen, recovering, or unknown"
    )


def _vendor_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("vendor_mode must be a string")
    return value


def _experimental(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("experimental must be a bool")
    return value


def _finite(where: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a real number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _vec3(value: object, where: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where} must be a sequence of 3 numbers")
    if len(value) != 3:
        raise ValueError(f"{where} must have 3 numbers")
    return (
        _finite(where, value[0]),
        _finite(where, value[1]),
        _finite(where, value[2]),
    )


def _planar(value: object) -> PlanarOdom:
    if isinstance(value, PlanarOdom):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
        raise TypeError("odom must be an object")
    extra = sorted(set(value) - {"x", "y", "yaw"})
    missing = [key for key in ("x", "y", "yaw") if key not in value]
    if missing or extra:
        raise _key_error("odom", missing, extra)
    return PlanarOdom(
        x=_finite("odom.x", value["x"]),
        y=_finite("odom.y", value["y"]),
        yaw=_finite("odom.yaw", value["yaw"]),
    )


def _battery(value: object) -> HumanoidBattery:
    if isinstance(value, HumanoidBattery):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
        raise TypeError("battery must be an object")
    extra = sorted(set(value) - {"fraction", "voltage"})
    missing = [key for key in ("fraction", "voltage") if key not in value]
    if missing or extra:
        raise _key_error("battery", missing, extra)
    fraction = _finite("battery.fraction", value["fraction"])
    voltage = _finite("battery.voltage", value["voltage"])
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("battery.fraction must be between 0 and 1")
    return HumanoidBattery(fraction=fraction, voltage=voltage)


def _parse_joints(
    raw: object,
    joint_groups: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, JointGroupSample]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError("joints must be an object")
    if joint_groups is None:
        names = list(raw)
    else:
        if isinstance(joint_groups, (str, bytes)) or not isinstance(
            joint_groups, Mapping
        ):
            raise TypeError("joint_groups must be a mapping")
        names = []
        for name in joint_groups:
            if not isinstance(name, str) or not name:
                raise ValueError("joint group name must be a non-empty string")
            names.append(name)
        extra = sorted(set(raw) - set(names))
        missing = sorted(set(names) - set(raw))
        if missing or extra:
            parts: list[str] = []
            if missing:
                parts.append(f"missing {missing}")
            if extra:
                parts.append(f"unknown {extra}")
            raise ValueError("joints groups do not match the spec: " + ", ".join(parts))
    parsed: dict[str, JointGroupSample] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("joint group name must be a non-empty string")
        expected = (
            None if joint_groups is None else _group_length(joint_groups[name], name)
        )
        parsed[name] = _parse_sample(raw[name], f"joints.{name}", expected)
    return MappingProxyType(parsed)


def _group_length(joints: object, name: str) -> int:
    if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
        raise TypeError(f"joint_groups[{name!r}] must be a sequence of joint names")
    return len(joints)


def _parse_sample(raw: object, where: str, expected: int | None) -> JointGroupSample:
    if isinstance(raw, JointGroupSample):
        raw = raw.to_dict()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise TypeError(f"{where} must be an object")
    extra = sorted(set(raw) - set(JOINT_CHANNELS))
    if extra:
        raise ValueError(f"{where} unknown keys {extra}")
    found: dict[str, tuple[float, ...]] = {}
    length = expected
    for key in JOINT_CHANNELS:
        if key not in raw or raw[key] is None:
            continue
        values = _float_tuple(raw[key], f"{where}.{key}")
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError(
                f"{where}.{key} has {len(values)} values, expected {length}"
            )
        found[key] = values
    return JointGroupSample(
        q=found.get("q"),
        dq=found.get("dq"),
        tau=found.get("tau"),
        temp=found.get("temp"),
    )


def _float_tuple(value: object, where: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where} must be a sequence of numbers")
    return tuple(_finite(where, item) for item in value)


__all__ = [
    "HUMANOID_BLOCK_KEYS",
    "HUMANOID_BLOCK_SCHEMA",
    "JOINT_CHANNELS",
    "HumanoidBattery",
    "HumanoidTelemetry",
    "JointGroupSample",
    "PlanarOdom",
    "apply_humanoid_telemetry",
    "build_humanoid_telemetry",
    "dumps_humanoid_telemetry",
    "dumps_telemetry",
    "is_upright",
    "loads_humanoid_telemetry",
    "loads_telemetry",
    "telemetry_with_humanoid",
    "validate_humanoid_telemetry",
]
