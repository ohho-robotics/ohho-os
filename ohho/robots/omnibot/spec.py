"""Spec and joint groups for the built-in mobile manipulator."""

from __future__ import annotations

from ohho import capabilities as caps
from ohho.registry import RobotSpec
from ohho.robot_spec import RobotClass

ROBOT_ID = "omnibot"

# SO-101 arm joints. Kept in sync with the arm driver's defaults.
ARM_JOINTS = (
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
)


def omnibot_spec() -> RobotSpec:
    """The shipped mobile-manipulator spec.

    ``source`` is this plugin's name so discovery does not rewrite it.
    """
    return RobotSpec(
        id=ROBOT_ID,
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
        joint_names=ARM_JOINTS,
        joint_groups={"arm": ARM_JOINTS},
        robot_class=RobotClass.MOBILE_MANIPULATOR,
        source="omnibot",
    )
