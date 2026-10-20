"""LeRobot feature names for a humanoid, taken from its joint groups.

The writer does not know a robot id. A recorder fills ``Episode.state_names``
and ``Episode.action_names`` from the spec, and states which arm-action form
was recorded. Joint order is the order of ``joint_groups``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .. import capabilities as caps

ARM_ACTION_EE_POSE = "ee_pose"
ARM_ACTION_ARM_JOINTS = "arm_joints"
ARM_ACTIONS = frozenset({ARM_ACTION_EE_POSE, ARM_ACTION_ARM_JOINTS})

# Position and quaternion, one side. The arm-joint form replaces these.
EE_POSE_AXES = ("x", "y", "z", "qw", "qx", "qy", "qz")
EE_POSE_WIDTH = len(EE_POSE_AXES)

HEAD_IMAGE_KEY = "observation.images.head"

# After the joints: orientation, angular rate, then planar odom.
BASE_STATE_NAMES = (
    "base_roll",
    "base_pitch",
    "base_yaw",
    "base_ang_vel_roll",
    "base_ang_vel_pitch",
    "base_ang_vel_yaw",
    "odom_x",
    "odom_y",
    "odom_yaw",
)

VELOCITY_NAMES = ("vx", "vy", "wz")
GRIPPER_NAMES = ("left_gripper", "right_gripper")


def is_humanoid(robot: Any) -> bool:
    """True when the spec is a humanoid or declares a humanoid capability."""
    spec = getattr(robot, "spec", None)
    kind = getattr(spec, "robot_class", None)
    if kind == "humanoid" or getattr(kind, "value", None) == "humanoid":
        return True
    has = getattr(robot, "has", None)
    if not callable(has):
        return False
    return bool(has(caps.HUMANOID_POSTURE) or has(caps.HUMANOID_LOCOMOTION))


def joint_groups(spec: Any) -> dict[str, tuple[str, ...]]:
    """Group name to joint names, in the spec's order."""
    raw = getattr(spec, "joint_groups", None)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        return {}
    groups: dict[str, tuple[str, ...]] = {}
    for name, joints in raw.items():
        if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
            continue
        groups[str(name)] = tuple(str(joint) for joint in joints)
    return groups


def joint_feature_names(spec: Any) -> list[str]:
    """``group.joint`` for every joint, in ``joint_groups`` order."""
    names: list[str] = []
    for group, joints in joint_groups(spec).items():
        for joint in joints:
            names.append(f"{group}.{joint}")
    return names


def state_names(spec: Any) -> list[str]:
    """Observation names: joints, then base orientation, rate, and odom."""
    return joint_feature_names(spec) + list(BASE_STATE_NAMES)


def arm_group_name(spec: Any, side: str) -> Optional[str]:
    """Left or right arm group, when the spec names one."""
    groups = joint_groups(spec)
    preferred = (f"{side}_arm", f"arm_{side}")
    for name in preferred:
        if name in groups:
            return name
    for key in groups:
        parts = key.lower().split("_")
        if side in parts and "arm" in parts:
            return key
    return None


def action_names(spec: Any, arm_action: str) -> list[str]:
    """Velocity, then either EE poses or arm-group joints, then grippers."""
    if arm_action not in ARM_ACTIONS:
        raise ValueError("arm_action must be 'ee_pose' or 'arm_joints'")
    names = list(VELOCITY_NAMES)
    if arm_action == ARM_ACTION_ARM_JOINTS:
        groups = joint_groups(spec)
        for side in ("left", "right"):
            group = arm_group_name(spec, side)
            if group is None:
                continue
            for joint in groups[group]:
                names.append(f"{group}.{joint}")
    else:
        for side in ("left", "right"):
            for axis in EE_POSE_AXES:
                names.append(f"{side}_ee_{axis}")
    names.extend(GRIPPER_NAMES)
    return names


def choose_arm_action(robot: Any, override: Optional[str]) -> str:
    """``arm_joints`` in custom mode, otherwise end-effector poses.

    An explicit ``override`` wins. Anything else is an error.
    """
    if override is not None:
        if override not in ARM_ACTIONS:
            raise ValueError("arm_action must be 'ee_pose' or 'arm_joints'")
        return override
    from ..humanoid import HumanoidMode, mode_of

    if mode_of(robot.telemetry()) is HumanoidMode.CUSTOM:
        return ARM_ACTION_ARM_JOINTS
    return ARM_ACTION_EE_POSE


def observation_state(robot: Any) -> list[float]:
    """Joint positions in group order, then roll/pitch/yaw, rate, and odom."""
    spec = robot.spec
    telemetry = robot.telemetry()
    block = _humanoid_block(telemetry)
    by_name = {
        joint.name: float(joint.position)
        for joint in (getattr(telemetry, "joints", None) or [])
    }
    reported = block.get("joints") if isinstance(block.get("joints"), Mapping) else {}
    positions: list[float] = []
    for group, joints in joint_groups(spec).items():
        sample = reported.get(group) if isinstance(reported, Mapping) else None
        q = sample.get("q") if isinstance(sample, Mapping) else None
        for index, joint in enumerate(joints):
            if joint in by_name:
                positions.append(by_name[joint])
            elif isinstance(q, Sequence) and not isinstance(q, (str, bytes)):
                if index < len(q):
                    positions.append(float(q[index]))
                else:
                    positions.append(0.0)
            else:
                positions.append(0.0)
    rpy = _three(block.get("base_rpy"))
    ang = _three(block.get("base_ang_vel"))
    odom = _planar_odom(block, telemetry)
    return positions + rpy + ang + odom


def gripper_value(cmd: object) -> float:
    """A gripper command as one float. A non-empty string is closed (1)."""
    if isinstance(cmd, bool) or not isinstance(cmd, (int, float)):
        if isinstance(cmd, str) and cmd.strip():
            return 1.0
        return 0.0
    return float(cmd)


def _humanoid_block(telemetry: Any) -> Mapping[str, Any]:
    custom = getattr(telemetry, "custom", None)
    if not isinstance(custom, Mapping):
        return {}
    block = custom.get("humanoid")
    if isinstance(block, Mapping):
        return block
    return {}


def _three(value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [0.0, 0.0, 0.0]
    out: list[float] = []
    for index in range(3):
        if index >= len(value):
            out.append(0.0)
            continue
        try:
            out.append(float(value[index]))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _planar_odom(block: Mapping[str, Any], telemetry: Any) -> list[float]:
    raw = block.get("odom")
    odom = raw if isinstance(raw, Mapping) else {}
    tele = getattr(telemetry, "odom", None)

    def pick(key: str, attr: str) -> float:
        if key in odom:
            try:
                return float(odom[key])
            except (TypeError, ValueError):
                return 0.0
        if tele is not None and hasattr(tele, attr):
            return float(getattr(tele, attr))
        return 0.0

    return [pick("x", "x"), pick("y", "y"), pick("yaw", "theta")]
