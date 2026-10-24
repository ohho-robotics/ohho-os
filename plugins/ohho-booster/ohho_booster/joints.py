"""Joint names copied from booster_assets ``T2_JOINT_NAMES``.

Source: https://github.com/BoosterRobotics/booster_assets
Commit: 12a97516f57469078707afb587a5f84c1cb110d1
File: src/booster_assets/motions.py
Licence: BSD-3-Clause. See NOTICE. These names are data; the mesh archive is not
vendored.
"""

from __future__ import annotations

from types import MappingProxyType

# Order is the upstream list, not a regrouping. Counts: head 2, left arm 7,
# right arm 7, waist 3, left leg 6, right leg 6.
T2_JOINT_NAMES: tuple[str, ...] = (
    "aa_head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

JOINT_GROUPS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "head": T2_JOINT_NAMES[0:2],
        "left_arm": T2_JOINT_NAMES[2:9],
        "right_arm": T2_JOINT_NAMES[9:16],
        "waist": T2_JOINT_NAMES[16:19],
        "left_leg": T2_JOINT_NAMES[19:25],
        "right_leg": T2_JOINT_NAMES[25:31],
    }
)

# Real-robot / MJCF motor order from booster_deploy ``K1_CFG.joint_names``
# (commit 7bb1462). Counts: head 2, left arm 4, right arm 4, left leg 6,
# right leg 6. No waist. The retarget list ``K1_JOINT_NAMES`` in
# booster_assets ``motions.py`` drops the ``aa`` prefix on the shoulder
# pitch joints; those names are not in the MJCF, so this list follows the
# model and the deployment config.
K1_JOINT_NAMES: tuple[str, ...] = (
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Isaac / ``K1_CFG.sim_joint_names`` order. Same names as ``K1_JOINT_NAMES``,
# different sequence. MuJoCo ``qpos`` for ``K1_22dof.xml`` follows the real
# order above, not this one. The policy indexes joints by name.
K1_SIM_JOINT_NAMES: tuple[str, ...] = (
    "aahead_yaw_joint",
    "aaleft_shoulder_pitch_joint",
    "aaright_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "aahead_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

K1_JOINT_GROUPS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "head": K1_JOINT_NAMES[0:2],
        "left_arm": K1_JOINT_NAMES[2:6],
        "right_arm": K1_JOINT_NAMES[6:10],
        "left_leg": K1_JOINT_NAMES[10:16],
        "right_leg": K1_JOINT_NAMES[16:22],
    }
)
