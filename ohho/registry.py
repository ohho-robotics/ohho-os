"""Robot registry and the ``RobotSpec`` capability manifest.

Built-in robots are defined here as data and registered by the ``ohho.robots``
built-in plugin (see :mod:`ohho.plugins`). Users add their own with
``register(...)`` or by loading a manifest file (``load_manifest``). A spec says
what a robot *is* — its class, capabilities, degrees of freedom, limits and
which adapter speaks its native protocol — not how it does any of it.

``category`` is the legacy free string. ``robot_class`` is the closed enum.
The additive fields and the category table live in ``ohho.robot_spec``. The
rule is ``docs/architecture/robot-plugins.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from . import capabilities as caps
from .capability_params import reject_unknown_capabilities
from .robot_spec import (
    BUILTIN_SOURCE,
    GENERIC_CAPABILITIES,
    AerialLimits,
    CiSim,
    ClassLimits,
    RobotAssets,
    RobotClass,
    as_class_limits,
    assets_from_dict,
    assets_to_dict,
    category_is_known,
    ci_sim_from_dict,
    ci_sim_to_dict,
    default_limits,
    freeze_capability_params,
    freeze_joint_groups,
    limits_from_manifest,
    limits_to_dict,
    limits_type_for,
    parse_robot_class,
    resolve_capability_params,
    robot_class_for_category,
    thaw_json,
)


class UnknownRobot(KeyError):
    """Raised when a robot id is not in the registry."""


@dataclass(frozen=True)
class RobotSpec:
    """What a robot is: identity, capabilities, limits and which adapter speaks it.

    ``category`` is the legacy free string. ``robot_class`` is derived from it
    when omitted. ``joint_groups``, ``limits``, ``assets``, ``ci_sim``,
    ``source``, ``capability_params`` and ``sim_backend`` all default, so
    existing constructors and manifests keep working. An empty ``sim_backend``
    means ``sim://`` keeps today's simulator. A name selects a plugin sim
    backend (``kinematic``, and whatever else a plugin registers).

    ``limits`` describes the class. It is not a safety cap. ``SafetyGate`` still
    uses ``SafetyProfile``, and ``Robot.drive`` still clamps with ``max_lin``
    and ``max_ang``.
    """

    id: str
    name: str
    category: str
    capabilities: tuple[str, ...] = ()
    adapter: str = "sim"
    dof_base: int = 3
    dof_arm: int = 0
    max_lin: float = 0.5
    max_ang: float = 1.0
    joint_names: tuple[str, ...] = ()
    robot_class: RobotClass | None = None
    joint_groups: Mapping[str, Sequence[str]] | None = field(default=None, hash=False)
    limits: ClassLimits | None = None
    assets: RobotAssets | None = None
    ci_sim: CiSim | None = None
    source: str = ""
    capability_params: Mapping[str, Any] | None = field(default=None, hash=False)
    sim_backend: str = ""

    def __post_init__(self) -> None:
        robot_class = self.robot_class
        if robot_class is None:
            robot_class = robot_class_for_category(self.category)
        elif not isinstance(robot_class, RobotClass):
            robot_class = parse_robot_class(robot_class)
        object.__setattr__(self, "robot_class", robot_class)

        object.__setattr__(self, "joint_groups", freeze_joint_groups(self.joint_groups))

        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if not isinstance(self.sim_backend, str):
            raise TypeError("sim_backend must be a string")
        object.__setattr__(self, "sim_backend", self.sim_backend.strip())

        assets = self.assets if self.assets is not None else RobotAssets()
        if not isinstance(assets, RobotAssets):
            raise TypeError("assets must be a RobotAssets")
        object.__setattr__(self, "assets", assets)

        ci_sim = self.ci_sim if self.ci_sim is not None else CiSim()
        if not isinstance(ci_sim, CiSim):
            raise TypeError("ci_sim must be a CiSim")
        object.__setattr__(self, "ci_sim", ci_sim)

        expected = limits_type_for(robot_class)
        if self.limits is None:
            limits = default_limits(robot_class, self.max_lin, self.max_ang)
        else:
            limits = self.limits
        if not isinstance(limits, expected):
            raise TypeError(
                f"{robot_class.value} limits must be {expected.__name__}, "
                f"not {type(limits).__name__}"
            )
        object.__setattr__(self, "limits", limits)
        object.__setattr__(
            self,
            "capability_params",
            freeze_capability_params(
                resolve_capability_params(
                    self.capabilities,
                    self.capability_params,
                    limits,
                    self.joint_groups,
                    max_lin=self.max_lin,
                    max_ang=self.max_ang,
                )
            ),
        )

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready manifest. ``spec_from_dict(spec.to_dict())`` round-trips."""
        robot_class = self.robot_class
        assets = self.assets
        ci_sim = self.ci_sim
        groups = self.joint_groups
        params = self.capability_params
        if not isinstance(robot_class, RobotClass):
            raise TypeError("robot_class was not resolved")
        if not isinstance(assets, RobotAssets):
            raise TypeError("assets must be a RobotAssets")
        if not isinstance(ci_sim, CiSim):
            raise TypeError("ci_sim must be a CiSim")
        if groups is None or params is None:
            raise TypeError("joint_groups and capability_params were not resolved")
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "robot_class": robot_class.value,
            "capabilities": list(self.capabilities),
            "capability_params": thaw_json(params),
            "adapter": self.adapter,
            "dof_base": self.dof_base,
            "dof_arm": self.dof_arm,
            "max_lin": self.max_lin,
            "max_ang": self.max_ang,
            "joint_names": list(self.joint_names),
            "joint_groups": thaw_json(groups),
            "limits": limits_to_dict(
                as_class_limits(self.limits), self.max_lin, self.max_ang
            ),
            "assets": assets_to_dict(assets),
            "ci_sim": ci_sim_to_dict(ci_sim),
            "source": self.source,
            "sim_backend": self.sim_backend,
        }

    def _aerial_limits(self) -> AerialLimits | None:
        if isinstance(self.limits, AerialLimits):
            return self.limits
        return None

    @property
    def max_climb(self) -> float:
        """Climb cap in m/s. Reads ``AerialLimits``; ``0.0`` for every other class."""
        limits = self._aerial_limits()
        return 0.0 if limits is None else limits.max_climb

    @property
    def gps_fix(self) -> str:
        """GPS fix the aerial drafts store. Empty when this spec is not aerial."""
        limits = self._aerial_limits()
        return "" if limits is None else limits.gps_fix

    @property
    def requires_hardware_arming(self) -> bool:
        """Aerial manifest flag. This is not the ``SafetyGate`` arming check."""
        limits = self._aerial_limits()
        return False if limits is None else limits.requires_hardware_arming

    @property
    def experimental(self) -> bool:
        """Aerial draft flag. False for every non-aerial spec."""
        limits = self._aerial_limits()
        return False if limits is None else limits.experimental

    @property
    def flight_ready(self) -> bool:
        """Aerial draft flag. A flat manifest cannot set this true."""
        limits = self._aerial_limits()
        return False if limits is None else limits.flight_ready


# Poses the built-in quadruped adapter already forwards
# (``ohho/adapters/unitree.py`` sport methods). ``euler`` and ``stop_move`` are
# commands, not poses, so they are not listed. This is not a safety allowlist.
_QUADRUPED_POSES = (
    "stand_up",
    "stand_down",
    "balance_stand",
    "recovery_stand",
    "sit",
    "rise_sit",
    "damp",
)

# OmniBot's SO-101 arm joints (kept in sync with the ROS arm driver defaults).
_OMNIBOT_ARM = (
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
)

_BUILTINS: dict[str, RobotSpec] = {
    "omnibot": RobotSpec(
        id="omnibot",
        name="OmniBot",
        category="mobile-manipulator",
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.PERCEPTION_RGB,
            caps.PERCEPTION_DEPTH,
            caps.PERCEPTION_SCAN,
            caps.MANIPULATION,
        ),
        adapter="yahboom-serial",
        dof_base=3,
        dof_arm=6,
        max_lin=0.2,
        max_ang=1.0,
        joint_names=_OMNIBOT_ARM,
        joint_groups={"arm": _OMNIBOT_ARM},
        robot_class=RobotClass.MOBILE_MANIPULATOR,
        source=BUILTIN_SOURCE,
    ),
    "unitree-go2": RobotSpec(
        id="unitree-go2",
        name="Unitree Go2",
        category="legged",
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.PERCEPTION_RGB,
            caps.PERCEPTION_DEPTH,
            caps.PERCEPTION_SCAN,
            caps.LEGGED_POSE,
        ),
        adapter="unitree-dds",
        dof_base=3,
        dof_arm=0,
        max_lin=1.5,
        max_ang=2.0,
        robot_class=RobotClass.QUADRUPED,
        source=BUILTIN_SOURCE,
        capability_params={caps.LEGGED_POSE: {"poses": list(_QUADRUPED_POSES)}},
    ),
    "sim": RobotSpec(
        id="sim",
        name="Generic Sim Bot",
        category="wheeled",
        capabilities=GENERIC_CAPABILITIES,
        adapter="sim",
        dof_base=3,
        dof_arm=0,
        max_lin=1.0,
        max_ang=2.0,
        robot_class=RobotClass.MOBILE_BASE,
        source=BUILTIN_SOURCE,
    ),
}

# Filled by plugin discovery on first lookup. Not seeded from ``_BUILTINS``:
# those specs enter through the ``ohho.robots`` entry point (or the source
# fallback in ``ohho.plugins``) so a failed plugin cannot hide behind a second copy.
_REGISTRY: dict[str, RobotSpec] = {}
_plugins_loaded = False
_plugins_loading = False


def builtin_specs() -> list[RobotSpec]:
    """Built-in specs, without consulting entry points.

    :class:`ohho.robots.builtin.BuiltinPlugin` returns these. Live lookups go
    through :func:`list_specs`.
    """
    return list(_BUILTINS.values())


def _ensure_plugins() -> None:
    """Register specs from ``ohho.robots`` once. Re-entrant while loading."""
    global _plugins_loaded, _plugins_loading
    if _plugins_loaded or _plugins_loading:
        return
    _plugins_loading = True
    try:
        from .plugins import register_loaded_specs

        register_loaded_specs()
    finally:
        _plugins_loading = False
    _plugins_loaded = True


def register(spec: RobotSpec) -> RobotSpec:
    """Add (or replace) a robot in the registry.

    Capability names must be in the vocabulary. Parameter objects are checked
    against their schemas. The parameters describe the robot; they do not
    change safety caps. Plugin specs are loaded first so a later ``register``
    replaces one id without hiding the others.
    """
    reject_unknown_capabilities(spec.capabilities, spec.capability_params)
    _ensure_plugins()
    _REGISTRY[spec.id] = spec
    return spec


def get_spec(robot_id: str) -> RobotSpec:
    _ensure_plugins()
    try:
        return _REGISTRY[robot_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRobot(
            f"unknown robot '{robot_id}'. Known robots: {known}"
        ) from None


def list_specs() -> list[RobotSpec]:
    _ensure_plugins()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def spec_from_dict(d: dict) -> RobotSpec:
    """Build a RobotSpec from a manifest dict (supports nested dof/limits).

    Manifests that only set ``category``, ``dof`` and ``limits.max_lin`` /
    ``limits.max_ang`` still load. ``category`` maps onto ``robot_class``. An
    unknown category becomes ``mobile_base``; when that manifest also omits
    ``capabilities``, it receives :data:`GENERIC_CAPABILITIES`.
    """
    if not isinstance(d, dict):
        raise ValueError("robot manifest must be an object")
    dof = d.get("dof", {})
    if dof is None:
        dof = {}
    if not isinstance(dof, dict):
        raise ValueError("dof must be an object")
    raw_limits = d.get("limits", {})
    if raw_limits is None:
        raw_limits = {}
    if not isinstance(raw_limits, dict):
        raise ValueError("limits must be an object")

    category = d.get("category", "unknown")
    if not isinstance(category, str):
        raise ValueError("category must be a string")

    explicit_class = "robot_class" in d and d["robot_class"] not in (None, "")
    if explicit_class:
        robot_class = parse_robot_class(d["robot_class"])
    else:
        robot_class = robot_class_for_category(category)

    if "capabilities" in d:
        capabilities = tuple(d["capabilities"])
    elif not explicit_class and not category_is_known(category):
        capabilities = GENERIC_CAPABILITIES
    else:
        capabilities = ()

    max_lin = float(raw_limits.get("max_lin", d.get("max_lin", 0.5)))
    max_ang = float(raw_limits.get("max_ang", d.get("max_ang", 1.0)))
    source = d.get("source", "")
    if source is None:
        source = ""
    if not isinstance(source, str):
        raise ValueError("source must be a string")
    sim_backend = d.get("sim_backend", "")
    if sim_backend is None:
        sim_backend = ""
    if not isinstance(sim_backend, str):
        raise ValueError("sim_backend must be a string")

    return RobotSpec(
        id=d["id"],
        name=d.get("name", d["id"]),
        category=category,
        robot_class=robot_class,
        capabilities=capabilities,
        capability_params=d.get("capability_params"),
        adapter=d.get("adapter", "sim"),
        dof_base=int(dof.get("base", d.get("dof_base", 3))),
        dof_arm=int(dof.get("arm", d.get("dof_arm", 0))),
        max_lin=max_lin,
        max_ang=max_ang,
        joint_names=tuple(d.get("joint_names", ())),
        joint_groups=d.get("joint_groups"),
        limits=limits_from_manifest(raw_limits, robot_class, max_lin, max_ang, d),
        assets=assets_from_dict(d.get("assets")),
        ci_sim=ci_sim_from_dict(d.get("ci_sim")),
        source=source,
        sim_backend=sim_backend,
    )


def load_manifest(path: Union[str, Path], *, register_it: bool = True) -> RobotSpec:
    """Load a robot manifest from JSON (stdlib) or YAML (needs the ``yaml`` extra)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover - exercised only without pyyaml
            raise RuntimeError(
                "YAML manifests require pyyaml: pip install 'ohho-os[yaml]'"
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    spec = spec_from_dict(data)
    if register_it:
        register(spec)
    return spec
