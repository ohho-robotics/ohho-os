"""Robot registry and the ``RobotSpec`` capability manifest.

Built-in robots are defined here as data; users add their own with
``register(...)`` or by loading a manifest file (``load_manifest``). A spec says
what a robot *is* — its category, capabilities, degrees of freedom, limits and
which adapter speaks its native protocol — not how it does any of it.

``max_lin`` is the horizontal speed cap (m/s) and ``max_ang`` is the yaw-rate
cap (rad/s). Aerial specs also set ``max_climb`` (m/s) and ``gps_fix``.
``requires_hardware_arming`` is true when a non-loopback URI needs a human
present. ``flight_ready`` is false on every built-in spec; nothing here sets
it true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from . import capabilities as caps


class UnknownRobot(KeyError):
    """Raised when a robot id is not in the registry."""


@dataclass(frozen=True)
class RobotSpec:
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
    # Aerial. Ground robots leave these at the defaults.
    max_climb: float = 0.0
    gps_fix: str = ""
    requires_hardware_arming: bool = False
    experimental: bool = False
    flight_ready: bool = False

    def has(self, capability: str) -> bool:
        return capability in self.capabilities


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
    ),
    "sim": RobotSpec(
        id="sim",
        name="Generic Sim Bot",
        category="wheeled",
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.PERCEPTION_RGB,
            caps.PERCEPTION_SCAN,
        ),
        adapter="sim",
        dof_base=3,
        dof_arm=0,
        max_lin=1.0,
        max_ang=2.0,
    ),
    # Ohho caps, not PX4 MPC_XY_VEL_MAX. Default transport is sim:// (drone-sim).
    # px4-x500 is spec only: nothing connects. Its arming flag is for a
    # non-loopback URI (px4+dds://, mavlink+serial://) when that exists.
    "px4-sitl": RobotSpec(
        id="px4-sitl",
        name="PX4 SITL",
        category="aerial",
        capabilities=(caps.AERIAL, caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        adapter="sim",
        dof_base=3,
        dof_arm=0,
        max_lin=2.0,
        max_ang=0.8,
        max_climb=1.0,
        gps_fix="3d",
        requires_hardware_arming=False,
        experimental=True,
        flight_ready=False,
    ),
    "crazyflie-sim": RobotSpec(
        id="crazyflie-sim",
        name="Crazyflie sim",
        category="aerial",
        capabilities=(caps.AERIAL, caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        adapter="sim",
        dof_base=3,
        dof_arm=0,
        max_lin=2.0,
        max_ang=0.8,
        max_climb=1.0,
        gps_fix="none",
        requires_hardware_arming=False,
        experimental=True,
        flight_ready=False,
    ),
    # Same Ohho caps as px4-sitl. GPS is always none: a Crazyflie 2.1 has no
    # fix. crazyflie:// and crazyflie+ros2:// select the Crazyflie backend.
    # sim:// stays the ground sim; the kinematic aircraft is crazyflie-sim.
    # The flag records that a radio URI needs a person present. The gate
    # also keys off protocol != "simulated".
    "crazyflie-cf21": RobotSpec(
        id="crazyflie-cf21",
        name="Crazyflie 2.1",
        category="aerial",
        capabilities=(caps.AERIAL, caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        adapter="crazyflie",
        dof_base=3,
        dof_arm=0,
        max_lin=2.0,
        max_ang=0.8,
        max_climb=1.0,
        gps_fix="none",
        requires_hardware_arming=True,
        experimental=True,
        flight_ready=False,
    ),
    "px4-x500": RobotSpec(
        id="px4-x500",
        name="PX4 X500",
        category="aerial",
        capabilities=(caps.AERIAL, caps.BASE_DRIVE, caps.BASE_HOLONOMIC),
        adapter="px4-dds",
        dof_base=3,
        dof_arm=0,
        max_lin=2.0,
        max_ang=0.8,
        max_climb=1.0,
        gps_fix="3d",
        requires_hardware_arming=True,
        experimental=True,
        flight_ready=False,
    ),
}

_REGISTRY: dict[str, RobotSpec] = dict(_BUILTINS)


def register(spec: RobotSpec) -> RobotSpec:
    """Add (or replace) a robot in the registry."""
    _REGISTRY[spec.id] = spec
    return spec


def get_spec(robot_id: str) -> RobotSpec:
    try:
        return _REGISTRY[robot_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownRobot(
            f"unknown robot '{robot_id}'. Known robots: {known}"
        ) from None


def list_specs() -> list[RobotSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def spec_from_dict(d: dict) -> RobotSpec:
    """Build a RobotSpec from a manifest dict (supports nested dof/limits)."""
    dof = d.get("dof", {})
    limits = d.get("limits", {})
    return RobotSpec(
        id=d["id"],
        name=d.get("name", d["id"]),
        category=d.get("category", "unknown"),
        capabilities=tuple(d.get("capabilities", ())),
        adapter=d.get("adapter", "sim"),
        dof_base=int(dof.get("base", d.get("dof_base", 3))),
        dof_arm=int(dof.get("arm", d.get("dof_arm", 0))),
        max_lin=float(limits.get("max_lin", d.get("max_lin", 0.5))),
        max_ang=float(limits.get("max_ang", d.get("max_ang", 1.0))),
        joint_names=tuple(d.get("joint_names", ())),
        max_climb=float(limits.get("max_climb", d.get("max_climb", 0.0))),
        gps_fix=str(d.get("gps_fix", "") or ""),
        requires_hardware_arming=d.get("requires_hardware_arming") is True,
        experimental=d.get("experimental") is True,
        # A manifest cannot mark a robot ready to fly.
        flight_ready=False,
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
