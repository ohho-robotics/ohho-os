"""Ground + arm safety policy for the mobile manipulator.

The numbers are the caps the gate already applied for this robot. A plugin
policy may only tighten them; these match the class caps, so load accepts them.
"""

from __future__ import annotations

from ohho.safety import SafetyProfile, register_baseline
from ohho.safety_policy import GroundArmPolicy

from .spec import ROBOT_ID

# Mirrors the arm adapter's default joint min/max. Keep the two in sync.
_JOINT_LIMITS = {
    "arm_shoulder_pan": (-3.14, 3.14),
    "arm_shoulder_lift": (-1.57, 1.57),
    "arm_elbow_flex": (-1.57, 1.57),
    "arm_wrist_flex": (-1.57, 1.57),
    "arm_wrist_roll": (-3.14, 3.14),
    "arm_gripper": (-0.1, 0.8),
}

OMNIBOT_PROFILE = SafetyProfile(
    max_vx=0.2,
    max_vy=0.2,
    max_wz=1.0,
    max_accel_x=0.5,
    max_accel_y=0.5,
    max_accel_wz=2.0,
    joint_limits=_JOINT_LIMITS,
)
register_baseline(ROBOT_ID, OMNIBOT_PROFILE)


def policy_for(spec: object) -> GroundArmPolicy | None:
    """``GroundPolicy`` for the base and ``ArmPolicy`` for the joints.

    ``None`` when ``spec`` is not this robot, so another id keeps its class
    default. Discovery does not call this.
    """
    if spec is None or getattr(spec, "id", "") != ROBOT_ID:
        return None
    return GroundArmPolicy(OMNIBOT_PROFILE)
