"""Robot class, per-class limits, assets and CI-sim metadata for ``RobotSpec`` v2.

Additive types only. ``SafetyGate`` does not read them. Flat ``max_lin`` /
``max_ang`` remain the spec clamps; safety caps stay on ``SafetyProfile``.
The rule is ``docs/architecture/robot-plugins.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Union

from . import capabilities as caps


class RobotClass(str, Enum):
    """Closed set of robot classes. Not a free string."""

    MOBILE_BASE = "mobile_base"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    ARM = "arm"
    QUADRUPED = "quadruped"
    AERIAL = "aerial"
    HUMANOID = "humanoid"


# Legacy ``category`` strings, plus the enum values themselves. Lookup is
# case-insensitive. Anything else is unknown and becomes ``mobile_base``.
CATEGORY_TO_CLASS: Mapping[str, RobotClass] = MappingProxyType(
    {
        "wheeled": RobotClass.MOBILE_BASE,
        "mobile-base": RobotClass.MOBILE_BASE,
        "mobile_base": RobotClass.MOBILE_BASE,
        "mobile-manipulator": RobotClass.MOBILE_MANIPULATOR,
        "mobile_manipulator": RobotClass.MOBILE_MANIPULATOR,
        "arm": RobotClass.ARM,
        "manipulator": RobotClass.ARM,
        "legged": RobotClass.QUADRUPED,
        "quadruped": RobotClass.QUADRUPED,
        "aerial": RobotClass.AERIAL,
        "drone": RobotClass.AERIAL,
        "humanoid": RobotClass.HUMANOID,
        "biped": RobotClass.HUMANOID,
    }
)

# Same capability set as the built-in ``sim`` robot. Applied only when a
# manifest's category is unknown and it does not name capabilities itself.
GENERIC_CAPABILITIES: tuple[str, ...] = (
    caps.BASE_DRIVE,
    caps.BASE_HOLONOMIC,
    caps.PERCEPTION_RGB,
    caps.PERCEPTION_SCAN,
)

BUILTIN_SOURCE = "builtin"

# Scenario ids for ``CiSim.scenario`` when a robot opts into the sim matrix
# (``ohho list --json --ci-sim``, OHH-316). ``enabled`` defaults to false.
# A built-in opts in only when its class scenario has a sim backend.
CI_SCENARIOS: Mapping[RobotClass, str] = MappingProxyType(
    {
        RobotClass.MOBILE_BASE: "drive_square",
        RobotClass.MOBILE_MANIPULATOR: "drive_square",
        RobotClass.ARM: "arm_home",
        RobotClass.QUADRUPED: "quadruped_walk",
        RobotClass.AERIAL: "aerial_square",
        RobotClass.HUMANOID: "humanoid_walk",
    }
)

_EMPTY: Mapping[Any, Any] = MappingProxyType({})


def robot_class_for_category(category: str) -> RobotClass:
    """Map a legacy category string to a robot class.

    Unknown categories, including the historical default ``"unknown"``, map to
    ``mobile_base``.
    """
    return CATEGORY_TO_CLASS.get(category.strip().lower(), RobotClass.MOBILE_BASE)


def category_is_known(category: str) -> bool:
    """True when ``category`` is in the mapping table."""
    return category.strip().lower() in CATEGORY_TO_CLASS


def parse_robot_class(value: object) -> RobotClass:
    """Parse a ``robot_class`` value. Raises ``ValueError`` when it is not one."""
    if isinstance(value, RobotClass):
        return value
    try:
        return RobotClass(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise ValueError(f"unknown robot_class {value!r}") from None


@dataclass(frozen=True)
class GroundLimits:
    """Velocity description for a ground mobile base.

    Mirrors the spec's flat ``max_lin`` / ``max_ang`` clamps. Not a safety cap.
    """

    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 1.0


@dataclass(frozen=True)
class ArmLimits:
    """Arm description. ``None`` means the class limit is not declared.

    Not a safety cap. ``SafetyGate`` keeps its own joint step and effort limits.
    """

    max_joint_step: float | None = None
    max_tau: float | None = None


@dataclass(frozen=True)
class LeggedLimits:
    """Velocity description for a quadruped. Not a safety cap."""

    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 1.0


@dataclass(frozen=True)
class AerialLimits:
    """Aerial fields the drone drafts currently hang off the flat ``RobotSpec``.

    They live here so those drafts can later drop the flat fields and keep
    reading them through read-only properties. A flat manifest cannot set
    ``flight_ready``; see ``limits_from_manifest``.
    """

    max_climb: float = 0.0
    gps_fix: str = ""
    requires_hardware_arming: bool = False
    experimental: bool = False
    flight_ready: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_climb, bool) or not isinstance(
            self.max_climb, (int, float)
        ):
            raise TypeError("max_climb must be a number")
        if not isinstance(self.gps_fix, str):
            raise TypeError("gps_fix must be a string")
        for name in ("requires_hardware_arming", "experimental", "flight_ready"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True)
class HumanoidLimits:
    """Humanoid description from the class table.

    Velocity fields mirror the flat spec clamps when a manifest omits them.
    Joint fields stay ``None`` until a plugin declares them. Not a safety cap.
    """

    max_vx: float = 0.5
    max_vy: float = 0.5
    max_wz: float = 1.0
    max_joint_step: float | None = None
    max_kp: float | None = None
    max_tau: float | None = None


@dataclass(frozen=True)
class MobileManipulatorLimits:
    """Ground velocity plus an arm. The mobile-manipulator class has both."""

    ground: GroundLimits = field(default_factory=GroundLimits)
    arm: ArmLimits = field(default_factory=ArmLimits)


ClassLimits = Union[
    GroundLimits,
    ArmLimits,
    LeggedLimits,
    AerialLimits,
    HumanoidLimits,
    MobileManipulatorLimits,
]

_LIMITS_TYPE: Mapping[RobotClass, type] = MappingProxyType(
    {
        RobotClass.MOBILE_BASE: GroundLimits,
        RobotClass.MOBILE_MANIPULATOR: MobileManipulatorLimits,
        RobotClass.ARM: ArmLimits,
        RobotClass.QUADRUPED: LeggedLimits,
        RobotClass.AERIAL: AerialLimits,
        RobotClass.HUMANOID: HumanoidLimits,
    }
)


def limits_type_for(robot_class: RobotClass) -> type:
    """The limits dataclass for ``robot_class``."""
    return _LIMITS_TYPE[robot_class]


def as_class_limits(value: object) -> ClassLimits:
    """Return ``value`` when it is a per-class limits object."""
    if isinstance(
        value,
        (
            GroundLimits,
            MobileManipulatorLimits,
            ArmLimits,
            LeggedLimits,
            AerialLimits,
            HumanoidLimits,
        ),
    ):
        return value
    raise TypeError(f"limits must be a class limits object, not {type(value).__name__}")


def default_limits(
    robot_class: RobotClass, max_lin: float, max_ang: float
) -> ClassLimits:
    """Class limits filled from the flat spec clamps. Aerial extras stay at defaults."""
    if robot_class is RobotClass.MOBILE_BASE:
        return GroundLimits(max_vx=max_lin, max_vy=max_lin, max_wz=max_ang)
    if robot_class is RobotClass.MOBILE_MANIPULATOR:
        return MobileManipulatorLimits(
            ground=GroundLimits(max_vx=max_lin, max_vy=max_lin, max_wz=max_ang)
        )
    if robot_class is RobotClass.ARM:
        return ArmLimits()
    if robot_class is RobotClass.QUADRUPED:
        return LeggedLimits(max_vx=max_lin, max_vy=max_lin, max_wz=max_ang)
    if robot_class is RobotClass.AERIAL:
        return AerialLimits()
    if robot_class is RobotClass.HUMANOID:
        return HumanoidLimits(max_vx=max_lin, max_vy=max_lin, max_wz=max_ang)
    raise TypeError(f"unknown robot_class {robot_class!r}")


def _num(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _opt_float(raw: Mapping[str, Any], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return float(raw[key])


def _velocity(
    src: Mapping[str, Any], max_lin: float, max_ang: float
) -> tuple[float, float, float]:
    return (
        _num(src.get("max_vx"), max_lin),
        _num(src.get("max_vy"), max_lin),
        _num(src.get("max_wz"), max_ang),
    )


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    if key not in raw or raw[key] is None:
        return None
    section = raw[key]
    if not isinstance(section, Mapping):
        raise ValueError(f"limits.{key} must be an object")
    return section


def _arm_from(src: Mapping[str, Any]) -> ArmLimits:
    return ArmLimits(
        max_joint_step=_opt_float(src, "max_joint_step"),
        max_tau=_opt_float(src, "max_tau"),
    )


def _climb(raw: Mapping[str, Any], manifest: Mapping[str, Any]) -> float:
    if raw.get("max_climb") is not None:
        return float(raw["max_climb"])
    if manifest.get("max_climb") is not None:
        return float(manifest["max_climb"])
    return 0.0


def _text_field(raw: Mapping[str, Any], manifest: Mapping[str, Any], key: str) -> str:
    if key in raw and raw[key] is not None:
        return str(raw[key])
    if key in manifest and manifest[key] is not None:
        return str(manifest[key])
    return ""


def _strict_bool(raw: Mapping[str, Any], manifest: Mapping[str, Any], key: str) -> bool:
    """True only for a real boolean ``True``, matching the drone manifest rule."""
    if key in raw:
        return raw[key] is True
    return manifest.get(key) is True


def _aerial_from(
    raw: Mapping[str, Any], manifest: Mapping[str, Any], *, typed: bool
) -> AerialLimits:
    """Build aerial limits.

    A flat manifest (no ``limits.kind``) cannot set ``flight_ready``. A typed
    document, including ``RobotSpec.to_dict``, may carry the boolean.
    """
    flight_ready = raw.get("flight_ready") is True if typed else False
    return AerialLimits(
        max_climb=_climb(raw, manifest),
        gps_fix=_text_field(raw, manifest, "gps_fix"),
        requires_hardware_arming=_strict_bool(
            raw, manifest, "requires_hardware_arming"
        ),
        experimental=_strict_bool(raw, manifest, "experimental"),
        flight_ready=flight_ready,
    )


def limits_from_manifest(
    raw: Mapping[str, Any],
    robot_class: RobotClass,
    max_lin: float,
    max_ang: float,
    manifest: Mapping[str, Any],
) -> ClassLimits:
    """Build class limits from a manifest ``limits`` object.

    Old manifests only have ``max_lin`` / ``max_ang``. Those stay on the spec;
    the typed object is filled from them. ``limits.kind``, when present, must
    match ``robot_class``.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("limits must be an object")
    kind = raw.get("kind")
    expected = robot_class.value
    if kind is not None and kind != expected:
        raise ValueError(
            f"limits.kind {kind!r} does not match robot_class {expected!r}"
        )
    typed = kind is not None
    if robot_class is RobotClass.MOBILE_BASE:
        vx, vy, wz = _velocity(raw, max_lin, max_ang)
        return GroundLimits(max_vx=vx, max_vy=vy, max_wz=wz)
    if robot_class is RobotClass.MOBILE_MANIPULATOR:
        ground_raw = _section(raw, "ground")
        arm_raw = _section(raw, "arm")
        vx, vy, wz = _velocity(
            ground_raw if ground_raw is not None else raw, max_lin, max_ang
        )
        return MobileManipulatorLimits(
            ground=GroundLimits(max_vx=vx, max_vy=vy, max_wz=wz),
            arm=_arm_from(arm_raw if arm_raw is not None else {}),
        )
    if robot_class is RobotClass.ARM:
        return _arm_from(raw)
    if robot_class is RobotClass.QUADRUPED:
        vx, vy, wz = _velocity(raw, max_lin, max_ang)
        return LeggedLimits(max_vx=vx, max_vy=vy, max_wz=wz)
    if robot_class is RobotClass.AERIAL:
        return _aerial_from(raw, manifest, typed=typed)
    if robot_class is RobotClass.HUMANOID:
        vx, vy, wz = _velocity(raw, max_lin, max_ang)
        return HumanoidLimits(
            max_vx=vx,
            max_vy=vy,
            max_wz=wz,
            max_joint_step=_opt_float(raw, "max_joint_step"),
            max_kp=_opt_float(raw, "max_kp"),
            max_tau=_opt_float(raw, "max_tau"),
        )
    raise TypeError(f"unknown robot_class {robot_class!r}")


def limits_to_dict(
    limits: ClassLimits, max_lin: float, max_ang: float
) -> dict[str, Any]:
    """Serialize limits. ``max_lin`` / ``max_ang`` stay keys so old readers still work."""
    if isinstance(limits, GroundLimits):
        return {
            "kind": RobotClass.MOBILE_BASE.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "max_vx": limits.max_vx,
            "max_vy": limits.max_vy,
            "max_wz": limits.max_wz,
        }
    if isinstance(limits, MobileManipulatorLimits):
        return {
            "kind": RobotClass.MOBILE_MANIPULATOR.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "ground": {
                "max_vx": limits.ground.max_vx,
                "max_vy": limits.ground.max_vy,
                "max_wz": limits.ground.max_wz,
            },
            "arm": {
                "max_joint_step": limits.arm.max_joint_step,
                "max_tau": limits.arm.max_tau,
            },
        }
    if isinstance(limits, ArmLimits):
        return {
            "kind": RobotClass.ARM.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "max_joint_step": limits.max_joint_step,
            "max_tau": limits.max_tau,
        }
    if isinstance(limits, LeggedLimits):
        return {
            "kind": RobotClass.QUADRUPED.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "max_vx": limits.max_vx,
            "max_vy": limits.max_vy,
            "max_wz": limits.max_wz,
        }
    if isinstance(limits, AerialLimits):
        return {
            "kind": RobotClass.AERIAL.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "max_climb": limits.max_climb,
            "gps_fix": limits.gps_fix,
            "requires_hardware_arming": limits.requires_hardware_arming,
            "experimental": limits.experimental,
            "flight_ready": limits.flight_ready,
        }
    if isinstance(limits, HumanoidLimits):
        return {
            "kind": RobotClass.HUMANOID.value,
            "max_lin": max_lin,
            "max_ang": max_ang,
            "max_vx": limits.max_vx,
            "max_vy": limits.max_vy,
            "max_wz": limits.max_wz,
            "max_joint_step": limits.max_joint_step,
            "max_kp": limits.max_kp,
            "max_tau": limits.max_tau,
        }
    raise TypeError(f"unknown limits type {type(limits).__name__}")


@dataclass(frozen=True)
class RobotAssets:
    """Description-file locations. Paths are stored, not opened."""

    urdf: str = ""
    mjcf: str = ""
    meshes: tuple[str, ...] = ()
    licence: str = ""

    def __post_init__(self) -> None:
        for name in ("urdf", "mjcf", "licence"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"assets.{name} must be a string")
        meshes = self.meshes
        if isinstance(meshes, (str, bytes)) or not isinstance(meshes, Sequence):
            raise TypeError("assets.meshes must be a sequence of paths")
        object.__setattr__(self, "meshes", tuple(str(item) for item in meshes))


@dataclass(frozen=True)
class CiSim:
    """Whether a registered robot joins the sim CI matrix, and which scenario."""

    enabled: bool = False
    scenario: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("ci_sim.enabled must be a bool")
        if not isinstance(self.scenario, str):
            raise TypeError("ci_sim.scenario must be a string")


def assets_from_dict(raw: Mapping[str, Any] | None) -> RobotAssets:
    """Parse an ``assets`` object. ``None`` is the empty asset set."""
    if raw is None:
        return RobotAssets()
    if not isinstance(raw, Mapping):
        raise ValueError("assets must be an object")
    meshes = raw.get("meshes", ())
    if meshes is None:
        meshes = ()
    if isinstance(meshes, (str, bytes)) or not isinstance(meshes, Sequence):
        raise ValueError("assets.meshes must be a list of paths")
    return RobotAssets(
        urdf=_as_str(raw.get("urdf", ""), "assets.urdf"),
        mjcf=_as_str(raw.get("mjcf", ""), "assets.mjcf"),
        meshes=tuple(str(item) for item in meshes),
        licence=_as_str(raw.get("licence", ""), "assets.licence"),
    )


def assets_to_dict(assets: RobotAssets) -> dict[str, Any]:
    """JSON-ready assets object."""
    return {
        "urdf": assets.urdf,
        "mjcf": assets.mjcf,
        "meshes": list(assets.meshes),
        "licence": assets.licence,
    }


def ci_sim_from_dict(raw: Mapping[str, Any] | None) -> CiSim:
    """Parse a ``ci_sim`` object. ``None`` is disabled with no scenario."""
    if raw is None:
        return CiSim()
    if not isinstance(raw, Mapping):
        raise ValueError("ci_sim must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("ci_sim.enabled must be a bool")
    scenario = raw.get("scenario", "")
    if not isinstance(scenario, str):
        raise ValueError("ci_sim.scenario must be a string")
    return CiSim(enabled=enabled, scenario=scenario)


def ci_sim_to_dict(ci_sim: CiSim) -> dict[str, Any]:
    """JSON-ready ``ci_sim`` object."""
    return {"enabled": ci_sim.enabled, "scenario": ci_sim.scenario}


def _as_str(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def freeze_joint_groups(
    groups: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, tuple[str, ...]]:
    """Copy joint groups into an immutable name → ordered names mapping."""
    if groups is None:
        return _EMPTY
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Mapping):
        raise TypeError("joint_groups must be a mapping of name to joint names")
    frozen: dict[str, tuple[str, ...]] = {}
    for name, joints in groups.items():
        if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
            raise TypeError(
                f"joint_groups[{name!r}] must be an ordered sequence of joint names"
            )
        frozen[str(name)] = tuple(str(joint) for joint in joints)
    if not frozen:
        return _EMPTY
    return MappingProxyType(frozen)


def freeze_capability_params(params: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Copy capability parameters into an immutable JSON-like mapping."""
    if params is None:
        return _EMPTY
    if isinstance(params, (str, bytes)) or not isinstance(params, Mapping):
        raise TypeError("capability_params must be a mapping")
    frozen = {str(key): _freeze_json(value) for key, value in params.items()}
    if not frozen:
        return _EMPTY
    return MappingProxyType(frozen)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (str, bytes)):
        if isinstance(value, bytes):
            raise TypeError("capability_params values must be JSON-compatible")
        return value
    if isinstance(value, Sequence):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(
        f"capability_params values must be JSON-compatible, not {type(value).__name__}"
    )


def resolve_capability_params(
    capabilities: Sequence[str],
    explicit: Mapping[str, Any] | None,
    limits: ClassLimits,
    joint_groups: Mapping[str, Sequence[str]] | None,
    *,
    max_lin: float,
    max_ang: float,
) -> dict[str, Any]:
    """Capability parameters for a spec.

    Velocity numbers come from the class limits when that object has them, and
    from ``max_lin`` / ``max_ang`` otherwise. Climb comes from ``AerialLimits``.
    An explicit mapping wins per capability name. Not a safety cap.
    """
    # Imported lazily: capability_params imports the capability names only.
    from .capability_params import merge_capability_params

    if isinstance(limits, MobileManipulatorLimits):
        max_vx, max_vy, max_wz = (
            limits.ground.max_vx,
            limits.ground.max_vy,
            limits.ground.max_wz,
        )
    elif isinstance(limits, (GroundLimits, LeggedLimits, HumanoidLimits)):
        max_vx, max_vy, max_wz = limits.max_vx, limits.max_vy, limits.max_wz
    else:
        max_vx, max_vy, max_wz = max_lin, max_lin, max_ang
    max_climb = float(limits.max_climb) if isinstance(limits, AerialLimits) else 0.0
    return merge_capability_params(
        capabilities,
        explicit,
        max_vx=max_vx,
        max_vy=max_vy,
        max_wz=max_wz,
        max_climb=max_climb,
        joint_groups=joint_groups,
    )


def thaw_json(value: Any) -> Any:
    """Turn frozen JSON-like values back into plain dicts and lists."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
