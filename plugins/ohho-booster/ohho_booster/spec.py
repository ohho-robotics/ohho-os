"""``booster-t2`` spec. Limits stay at the generic humanoid defaults.

Sim-first: velocity numbers are the class defaults (0.5 m/s, 1.0 rad/s).
Joint step, kp and tau stay undeclared. These are descriptions, not safety
caps. ``SafetyGate`` does not read them.
"""

from __future__ import annotations

from ohho import capabilities as caps
from ohho.humanoid_sim import HUMANOID_MUJOCO_BACKEND
from ohho.registry import RobotSpec
from ohho.robot_spec import HumanoidLimits, RobotAssets, RobotClass

from .assets import load_manifest
from .joints import JOINT_GROUPS, T2_JOINT_NAMES

# Generic HumanoidLimits / RobotSpec defaults. Not a vendor brochure speed.
_MAX_LIN = 0.5
_MAX_ANG = 1.0

_POSTURE_MODES = (
    "damping",
    "prepare",
    "walk",
    "custom",
    "get_up",
    "lie_down",
)


def booster_t2_spec() -> RobotSpec:
    """The 31-DoF humanoid spec registered by this plugin."""
    paths = load_manifest()["paths"]
    groups = {name: list(joints) for name, joints in JOINT_GROUPS.items()}
    return RobotSpec(
        id="booster-t2",
        name="Booster T2",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        capabilities=(
            caps.BASE_DRIVE,
            caps.BASE_HOLONOMIC,
            caps.HUMANOID_LOCOMOTION,
            caps.HUMANOID_POSTURE,
            caps.WHOLE_BODY_JOINT,
            caps.MANIPULATION,
            caps.MANIPULATION_DUAL_ARM,
            caps.HEAD_GAZE,
            caps.PERCEPTION_RGB,
            caps.PERCEPTION_DEPTH,
        ),
        capability_params={
            caps.HUMANOID_LOCOMOTION: {
                "gaits": ["walk"],
                "speed_min": 0.0,
                "speed_max": _MAX_LIN,
            },
            caps.HUMANOID_POSTURE: {"modes": list(_POSTURE_MODES)},
            caps.WHOLE_BODY_JOINT: {
                "joint_groups": groups,
                "control_modes": ["position", "torque", "hybrid"],
                "rate_hz": None,
            },
        },
        adapter="sim",
        sim_backend=HUMANOID_MUJOCO_BACKEND,
        dof_base=3,
        dof_arm=14,
        max_lin=_MAX_LIN,
        max_ang=_MAX_ANG,
        joint_names=T2_JOINT_NAMES,
        joint_groups=JOINT_GROUPS,
        limits=HumanoidLimits(max_vx=_MAX_LIN, max_vy=_MAX_LIN, max_wz=_MAX_ANG),
        assets=RobotAssets(
            urdf=str(paths["urdf"]),
            mjcf=str(paths["mjcf"]),
            meshes=(str(paths["meshes"]),),
            licence="BSD-3-Clause",
        ),
        source="booster",
    )
