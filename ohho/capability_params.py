"""Typed parameters for the capability vocabulary.

``Robot.has`` and ``RobotSpec.has`` stay string checks. Parameters are a
JSON object on ``RobotSpec.capability_params``, keyed by capability name.
They describe the robot (speeds, poses, frames, groups). They are not safety
caps, and ``SafetyGate`` does not read them.

Perception capabilities (``perception.rgb``, ``perception.depth``,
``perception.scan``) stay strings. A parameter object for one of those, when
present, is an open object. Every other known capability has a schema below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .capabilities import (
    AERIAL,
    BASE_DRIVE,
    BASE_HOLONOMIC,
    END_EFFECTOR_DEX_HAND,
    END_EFFECTOR_GRIPPER,
    HEAD_GAZE,
    HUMANOID_LOCOMOTION,
    HUMANOID_POSTURE,
    LEGGED_POSE,
    MANIPULATION,
    MANIPULATION_DUAL_ARM,
    WHOLE_BODY_JOINT,
    is_known,
)

# Closed sets from the capability table. A declared subset may be smaller.
POSTURE_MODES: tuple[str, ...] = (
    "damping",
    "prepare",
    "walk",
    "custom",
    "get_up",
    "lie_down",
)
CONTROL_MODES: tuple[str, ...] = ("position", "torque", "hybrid")

Parser = Callable[[Any], None]


def _where(capability: str, field: str) -> str:
    return f"capability_params[{capability!r}].{field}"


def _object(raw: Any, capability: str) -> Mapping[str, Any]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise ValueError(f"capability_params[{capability!r}] must be an object")
    return raw


def _exact(raw: Mapping[str, Any], allowed: set[str], capability: str) -> None:
    extra = set(raw) - allowed
    missing = allowed - set(raw)
    if missing:
        raise ValueError(f"capability_params[{capability!r}] missing {sorted(missing)}")
    if extra:
        raise ValueError(
            f"capability_params[{capability!r}] unknown keys {sorted(extra)}"
        )


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _optional_finite(value: Any, where: str) -> Optional[float]:
    if value is None:
        return None
    return _finite(value, where)


def _str_list(
    raw: Any, where: str, *, allowed: Optional[Sequence[str]] = None
) -> list[str]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{where} must be a list of strings")
    found: list[str] = []
    seen: set[str] = set()
    permitted = None if allowed is None else set(allowed)
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{where} must be a list of strings")
        if permitted is not None and item not in permitted:
            raise ValueError(f"{where} value {item!r} is not allowed")
        if item in seen:
            raise ValueError(f"{where} repeats {item!r}")
        seen.add(item)
        found.append(item)
    return found


@dataclass(frozen=True)
class DriveParams:
    """``max_vx`` / ``max_vy`` / ``max_wz`` for ground motion.

    The same object is used for ``base.drive`` and ``base.holonomic_drive``.
    ``max_vy`` is the spec's lateral limit. Strafing is allowed only when the
    robot also has ``base.holonomic_drive``. Not a safety cap.
    """

    max_vx: float
    max_vy: float
    max_wz: float

    def to_dict(self) -> dict[str, float]:
        return {
            "max_vx": float(self.max_vx),
            "max_vy": float(self.max_vy),
            "max_wz": float(self.max_wz),
        }

    @classmethod
    def from_dict(cls, raw: Any, *, capability: str = BASE_DRIVE) -> DriveParams:
        data = _object(raw, capability)
        _exact(data, {"max_vx", "max_vy", "max_wz"}, capability)
        return cls(
            max_vx=_finite(data["max_vx"], _where(capability, "max_vx")),
            max_vy=_finite(data["max_vy"], _where(capability, "max_vy")),
            max_wz=_finite(data["max_wz"], _where(capability, "max_wz")),
        )


@dataclass(frozen=True)
class LeggedPoseParams:
    """Pose names a legged robot can be asked to hold."""

    poses: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"poses": list(self.poses)}

    @classmethod
    def from_dict(cls, raw: Any) -> LeggedPoseParams:
        data = _object(raw, LEGGED_POSE)
        _exact(data, {"poses"}, LEGGED_POSE)
        poses = _str_list(data["poses"], _where(LEGGED_POSE, "poses"))
        return cls(poses=tuple(poses))


@dataclass(frozen=True)
class AerialParams:
    """Frame names and the climb cap in m/s. Not a safety cap."""

    frames: tuple[str, ...]
    max_climb: float

    def to_dict(self) -> dict[str, Any]:
        return {"frames": list(self.frames), "max_climb": float(self.max_climb)}

    @classmethod
    def from_dict(cls, raw: Any) -> AerialParams:
        data = _object(raw, AERIAL)
        _exact(data, {"frames", "max_climb"}, AERIAL)
        frames = _str_list(data["frames"], _where(AERIAL, "frames"))
        return cls(
            frames=tuple(frames),
            max_climb=_finite(data["max_climb"], _where(AERIAL, "max_climb")),
        )


@dataclass(frozen=True)
class HumanoidLocomotionParams:
    """Gait names and the vendor-reported speed range, in m/s.

    ``speed_min`` and ``speed_max`` may be omitted when a document only knows
    one end. Derived parameters set ``speed_min`` to 0 when the spec has no
    separate minimum, and ``speed_max`` from the spec's forward limit.
    """

    gaits: tuple[str, ...]
    speed_min: Optional[float] = None
    speed_max: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"gaits": list(self.gaits)}
        if self.speed_min is not None:
            body["speed_min"] = float(self.speed_min)
        if self.speed_max is not None:
            body["speed_max"] = float(self.speed_max)
        return body

    @classmethod
    def from_dict(cls, raw: Any) -> HumanoidLocomotionParams:
        data = _object(raw, HUMANOID_LOCOMOTION)
        missing = {"gaits"} - set(data)
        if missing:
            raise ValueError(
                f"capability_params[{HUMANOID_LOCOMOTION!r}] missing {sorted(missing)}"
            )
        extra = set(data) - {"gaits", "speed_min", "speed_max"}
        if extra:
            raise ValueError(
                f"capability_params[{HUMANOID_LOCOMOTION!r}] unknown keys {sorted(extra)}"
            )
        gaits = _str_list(data["gaits"], _where(HUMANOID_LOCOMOTION, "gaits"))
        speed_min = (
            None
            if "speed_min" not in data
            else _optional_finite(
                data["speed_min"], _where(HUMANOID_LOCOMOTION, "speed_min")
            )
        )
        speed_max = (
            None
            if "speed_max" not in data
            else _optional_finite(
                data["speed_max"], _where(HUMANOID_LOCOMOTION, "speed_max")
            )
        )
        if speed_min is not None and speed_max is not None and speed_min > speed_max:
            raise ValueError(
                f"capability_params[{HUMANOID_LOCOMOTION!r}] speed_min is above speed_max"
            )
        return cls(gaits=tuple(gaits), speed_min=speed_min, speed_max=speed_max)


@dataclass(frozen=True)
class HumanoidPostureParams:
    """A subset of the humanoid posture modes."""

    modes: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"modes": list(self.modes)}

    @classmethod
    def from_dict(cls, raw: Any) -> HumanoidPostureParams:
        data = _object(raw, HUMANOID_POSTURE)
        _exact(data, {"modes"}, HUMANOID_POSTURE)
        modes = _str_list(
            data["modes"],
            _where(HUMANOID_POSTURE, "modes"),
            allowed=POSTURE_MODES,
        )
        return cls(modes=tuple(modes))


@dataclass(frozen=True)
class WholeBodyJointParams:
    """Joint-group names, control modes, and the command rate in hertz.

    ``rate_hz`` is ``None`` when the spec does not declare a rate. An empty
    ``control_modes`` list means none are declared, not that every mode is on.
    """

    joint_groups: Mapping[str, tuple[str, ...]]
    control_modes: tuple[str, ...]
    rate_hz: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_groups": {
                name: list(joints) for name, joints in self.joint_groups.items()
            },
            "control_modes": list(self.control_modes),
            "rate_hz": None if self.rate_hz is None else float(self.rate_hz),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> WholeBodyJointParams:
        data = _object(raw, WHOLE_BODY_JOINT)
        _exact(
            data,
            {"joint_groups", "control_modes", "rate_hz"},
            WHOLE_BODY_JOINT,
        )
        groups_raw = data["joint_groups"]
        where = _where(WHOLE_BODY_JOINT, "joint_groups")
        if isinstance(groups_raw, (str, bytes)) or not isinstance(groups_raw, Mapping):
            raise ValueError(f"{where} must be an object")
        groups: dict[str, tuple[str, ...]] = {}
        for name, joints in groups_raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{where} has an invalid group name")
            groups[name] = tuple(_str_list(joints, f"{where}.{name}"))
        modes = _str_list(
            data["control_modes"],
            _where(WHOLE_BODY_JOINT, "control_modes"),
            allowed=CONTROL_MODES,
        )
        rate = _optional_finite(data["rate_hz"], _where(WHOLE_BODY_JOINT, "rate_hz"))
        if rate is not None and rate < 0:
            raise ValueError(f"{_where(WHOLE_BODY_JOINT, 'rate_hz')} must be >= 0")
        return cls(joint_groups=groups, control_modes=tuple(modes), rate_hz=rate)


@dataclass(frozen=True)
class ManipulationParams:
    """Arm-group names and end-effector frame names."""

    arm_groups: tuple[str, ...]
    ee_frames: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"arm_groups": list(self.arm_groups), "ee_frames": list(self.ee_frames)}

    @classmethod
    def from_dict(cls, raw: Any) -> ManipulationParams:
        data = _object(raw, MANIPULATION)
        _exact(data, {"arm_groups", "ee_frames"}, MANIPULATION)
        return cls(
            arm_groups=tuple(
                _str_list(data["arm_groups"], _where(MANIPULATION, "arm_groups"))
            ),
            ee_frames=tuple(
                _str_list(data["ee_frames"], _where(MANIPULATION, "ee_frames"))
            ),
        )


@dataclass(frozen=True)
class DualArmParams:
    """Left and right group names, plus the manipulation fields."""

    left: str
    right: str
    arm_groups: tuple[str, ...]
    ee_frames: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "arm_groups": list(self.arm_groups),
            "ee_frames": list(self.ee_frames),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> DualArmParams:
        data = _object(raw, MANIPULATION_DUAL_ARM)
        _exact(
            data,
            {"left", "right", "arm_groups", "ee_frames"},
            MANIPULATION_DUAL_ARM,
        )
        left = data["left"]
        right = data["right"]
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(
                f"capability_params[{MANIPULATION_DUAL_ARM!r}] left and right must be strings"
            )
        return cls(
            left=left,
            right=right,
            arm_groups=tuple(
                _str_list(
                    data["arm_groups"], _where(MANIPULATION_DUAL_ARM, "arm_groups")
                )
            ),
            ee_frames=tuple(
                _str_list(data["ee_frames"], _where(MANIPULATION_DUAL_ARM, "ee_frames"))
            ),
        )


@dataclass(frozen=True)
class EndEffectorParams:
    """Degrees of freedom and the force range in newtons.

    ``None`` means that value is not declared on the spec. Used for both
    ``end_effector.gripper`` and ``end_effector.dex_hand``.
    """

    dof: Optional[int]
    force_min: Optional[float]
    force_max: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dof": self.dof,
            "force_min": None if self.force_min is None else float(self.force_min),
            "force_max": None if self.force_max is None else float(self.force_max),
        }

    @classmethod
    def from_dict(
        cls, raw: Any, *, capability: str = END_EFFECTOR_GRIPPER
    ) -> EndEffectorParams:
        data = _object(raw, capability)
        _exact(data, {"dof", "force_min", "force_max"}, capability)
        dof = data["dof"]
        if dof is not None and (
            isinstance(dof, bool) or not isinstance(dof, int) or dof < 0
        ):
            raise ValueError(
                f"{_where(capability, 'dof')} must be an integer >= 0 or null"
            )
        force_min = _optional_finite(data["force_min"], _where(capability, "force_min"))
        force_max = _optional_finite(data["force_max"], _where(capability, "force_max"))
        if force_min is not None and force_max is not None and force_min > force_max:
            raise ValueError(
                f"capability_params[{capability!r}] force_min is above force_max"
            )
        return cls(dof=dof, force_min=force_min, force_max=force_max)


@dataclass(frozen=True)
class HeadGazeParams:
    """Pitch and yaw limits in radians. ``None`` means that end is not declared."""

    pitch_min: Optional[float] = None
    pitch_max: Optional[float] = None
    yaw_min: Optional[float] = None
    yaw_max: Optional[float] = None

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "pitch_min": None if self.pitch_min is None else float(self.pitch_min),
            "pitch_max": None if self.pitch_max is None else float(self.pitch_max),
            "yaw_min": None if self.yaw_min is None else float(self.yaw_min),
            "yaw_max": None if self.yaw_max is None else float(self.yaw_max),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> HeadGazeParams:
        data = _object(raw, HEAD_GAZE)
        _exact(
            data,
            {"pitch_min", "pitch_max", "yaw_min", "yaw_max"},
            HEAD_GAZE,
        )
        pitch_min = _optional_finite(data["pitch_min"], _where(HEAD_GAZE, "pitch_min"))
        pitch_max = _optional_finite(data["pitch_max"], _where(HEAD_GAZE, "pitch_max"))
        yaw_min = _optional_finite(data["yaw_min"], _where(HEAD_GAZE, "yaw_min"))
        yaw_max = _optional_finite(data["yaw_max"], _where(HEAD_GAZE, "yaw_max"))
        if pitch_min is not None and pitch_max is not None and pitch_min > pitch_max:
            raise ValueError(
                f"capability_params[{HEAD_GAZE!r}] pitch_min is above pitch_max"
            )
        if yaw_min is not None and yaw_max is not None and yaw_min > yaw_max:
            raise ValueError(
                f"capability_params[{HEAD_GAZE!r}] yaw_min is above yaw_max"
            )
        return cls(
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            yaw_min=yaw_min,
            yaw_max=yaw_max,
        )


def _parse_drive(raw: Any) -> None:
    DriveParams.from_dict(raw, capability=BASE_DRIVE)


def _parse_holonomic(raw: Any) -> None:
    DriveParams.from_dict(raw, capability=BASE_HOLONOMIC)


def _parse_gripper(raw: Any) -> None:
    EndEffectorParams.from_dict(raw, capability=END_EFFECTOR_GRIPPER)


def _parse_hand(raw: Any) -> None:
    EndEffectorParams.from_dict(raw, capability=END_EFFECTOR_DEX_HAND)


def _parse_pose(raw: Any) -> None:
    LeggedPoseParams.from_dict(raw)


def _parse_aerial(raw: Any) -> None:
    AerialParams.from_dict(raw)


def _parse_locomotion(raw: Any) -> None:
    HumanoidLocomotionParams.from_dict(raw)


def _parse_posture(raw: Any) -> None:
    HumanoidPostureParams.from_dict(raw)


def _parse_whole_body(raw: Any) -> None:
    WholeBodyJointParams.from_dict(raw)


def _parse_manipulation(raw: Any) -> None:
    ManipulationParams.from_dict(raw)


def _parse_dual_arm(raw: Any) -> None:
    DualArmParams.from_dict(raw)


def _parse_gaze(raw: Any) -> None:
    HeadGazeParams.from_dict(raw)


# Capabilities with a typed parameter object. Perception names are absent on purpose.
PARSERS: dict[str, Parser] = {
    BASE_DRIVE: _parse_drive,
    BASE_HOLONOMIC: _parse_holonomic,
    LEGGED_POSE: _parse_pose,
    AERIAL: _parse_aerial,
    HUMANOID_LOCOMOTION: _parse_locomotion,
    HUMANOID_POSTURE: _parse_posture,
    WHOLE_BODY_JOINT: _parse_whole_body,
    MANIPULATION: _parse_manipulation,
    MANIPULATION_DUAL_ARM: _parse_dual_arm,
    END_EFFECTOR_GRIPPER: _parse_gripper,
    END_EFFECTOR_DEX_HAND: _parse_hand,
    HEAD_GAZE: _parse_gaze,
}

PARAMETERIZED: frozenset[str] = frozenset(PARSERS)


def validate_capability_params(params: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a parameter object does not match its schema."""
    if isinstance(params, (str, bytes)) or not isinstance(params, Mapping):
        raise ValueError("capability_params must be an object")
    for name, value in params.items():
        if not isinstance(name, str) or not is_known(name):
            raise ValueError(f"unknown capability {name!r}")
        parser = PARSERS.get(name)
        if parser is None:
            if not isinstance(value, Mapping):
                raise ValueError(f"capability_params[{name!r}] must be an object")
            continue
        parser(value)


def reject_unknown_capabilities(
    capabilities: Sequence[object], params: Mapping[str, Any] | None
) -> None:
    """Reject capability names that are not in the vocabulary, then check schemas."""
    unknown: list[str] = []
    for name in capabilities:
        if not isinstance(name, str) or not is_known(name):
            unknown.append(repr(name))
    if params:
        if isinstance(params, (str, bytes)) or not isinstance(params, Mapping):
            raise ValueError("capability_params must be an object")
        for name in params:
            if not isinstance(name, str) or not is_known(name):
                label = repr(name)
                if label not in unknown:
                    unknown.append(label)
    if unknown:
        raise ValueError("unknown capabilities: " + ", ".join(unknown))
    if params:
        validate_capability_params(params)


def _arm_groups(joint_groups: Mapping[str, Sequence[str]]) -> list[str]:
    return [str(name) for name in joint_groups if "arm" in str(name).lower()]


def _match_group(joint_groups: Mapping[str, Sequence[str]], *candidates: str) -> str:
    lowered = {str(name).lower(): str(name) for name in joint_groups}
    for candidate in candidates:
        found = lowered.get(candidate)
        if found is not None:
            return found
    return ""


def _group_dof(
    joint_groups: Mapping[str, Sequence[str]], *needles: str
) -> Optional[int]:
    for name, joints in joint_groups.items():
        lowered = str(name).lower()
        if any(needle in lowered for needle in needles):
            return len(tuple(joints))
    return None


def derive_capability_params(
    capabilities: Sequence[str],
    *,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    max_climb: float,
    joint_groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Parameter objects implied by declared capabilities and spec values.

    Undeclared limits stay empty or ``None``. This does not invent poses,
    gaits, frames, or force limits the spec does not carry.
    """
    declared = {name for name in capabilities if isinstance(name, str)}
    groups = joint_groups or {}
    derived: dict[str, Any] = {}
    if BASE_DRIVE in declared:
        derived[BASE_DRIVE] = DriveParams(max_vx, max_vy, max_wz).to_dict()
    if BASE_HOLONOMIC in declared:
        derived[BASE_HOLONOMIC] = DriveParams(max_vx, max_vy, max_wz).to_dict()
    if LEGGED_POSE in declared:
        derived[LEGGED_POSE] = LeggedPoseParams(()).to_dict()
    if AERIAL in declared:
        derived[AERIAL] = AerialParams((), max_climb).to_dict()
    if HUMANOID_LOCOMOTION in declared:
        derived[HUMANOID_LOCOMOTION] = HumanoidLocomotionParams(
            (), speed_min=0.0, speed_max=float(max_vx)
        ).to_dict()
    if HUMANOID_POSTURE in declared:
        derived[HUMANOID_POSTURE] = HumanoidPostureParams(()).to_dict()
    if WHOLE_BODY_JOINT in declared:
        mapping = {
            str(name): tuple(str(joint) for joint in joints)
            for name, joints in groups.items()
        }
        derived[WHOLE_BODY_JOINT] = WholeBodyJointParams(mapping, ()).to_dict()
    if MANIPULATION in declared:
        derived[MANIPULATION] = ManipulationParams(
            tuple(_arm_groups(groups)), ()
        ).to_dict()
    if MANIPULATION_DUAL_ARM in declared:
        left = _match_group(groups, "left", "left_arm", "arm_left")
        right = _match_group(groups, "right", "right_arm", "arm_right")
        arm_groups: tuple[str, ...]
        if left and right:
            arm_groups = (left, right)
        else:
            arm_groups = tuple(_arm_groups(groups))
        derived[MANIPULATION_DUAL_ARM] = DualArmParams(
            left, right, arm_groups, ()
        ).to_dict()
    if END_EFFECTOR_GRIPPER in declared:
        derived[END_EFFECTOR_GRIPPER] = EndEffectorParams(
            _group_dof(groups, "gripper", "grip"), None, None
        ).to_dict()
    if END_EFFECTOR_DEX_HAND in declared:
        derived[END_EFFECTOR_DEX_HAND] = EndEffectorParams(
            _group_dof(groups, "hand", "dex"), None, None
        ).to_dict()
    if HEAD_GAZE in declared:
        derived[HEAD_GAZE] = HeadGazeParams().to_dict()
    return derived


def merge_capability_params(
    capabilities: Sequence[str],
    explicit: Mapping[str, Any] | None,
    *,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    max_climb: float,
    joint_groups: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    """Derived parameters, with an explicit object winning per capability name.

    Keys in ``explicit`` are stored as given, so a document that predates a
    field still round-trips. Declared capabilities the explicit object does
    not mention are filled from the spec.
    """
    derived = derive_capability_params(
        capabilities,
        max_vx=max_vx,
        max_vy=max_vy,
        max_wz=max_wz,
        max_climb=max_climb,
        joint_groups=joint_groups or {},
    )
    if not explicit:
        return derived
    merged = dict(derived)
    for key, value in explicit.items():
        merged[str(key)] = value
    return merged
