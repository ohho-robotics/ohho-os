"""Walk configuration for the generic humanoid-mujoco runner.

Gains, the joint order, the observation layout and the checkpoint path are
data for that runner. The numbers come from booster_deploy commit
7bb1462 (Apache-2.0), task ``t2_walk``. ``docs/t2-walk.md`` records the
mapping and why the serial MJCF is the one this config names.

``TRAINING_COMMAND`` is the checkpoint's training range. It is not an Ohho
safety cap. ``Robot.drive`` still clamps to the spec, and a gated link
clamps again in SafetyGate.
"""

from __future__ import annotations

import math
from types import MappingProxyType

from ohho.humanoid_sim import (
    HumanoidSimConfig,
    ObservationTerm,
    PolicyLayout,
    bind_sim_config,
    default_observation_terms,
)

from .assets import ensure_assets
from .joints import T2_JOINT_NAMES

# booster_deploy ControllerCfg.policy_dt and MujocoControllerCfg.decimation.
POLICY_DT_S = 0.02
PHYSICS_DECIMATION = 10
PHYSICS_DT_S = POLICY_DT_S / PHYSICS_DECIMATION
POLICY_RATE_HZ = 1.0 / POLICY_DT_S

# Deployment player init. The MJCF body origin is lower; this is the height
# the checkpoint was exercised at.
BASE_POS = (0.0, 0.0, 1.05)
BASE_BODY = "trunk"

HISTORY_LENGTH = 10
CLIP_OBSERVATION = 100.0
CLIP_ACTION = 10.0
DQ_SCALE = 1.0

# T2LocomotionPolicy.compute_observation. Not safety limits.
STANDING_COMMAND_NORM = 0.01
HEADING_ERROR_LIMIT = 0.5
COMMAND_CHANGE = 1.0e-6

# VelocityCommandCfg on the walk task. Training ranges, not Ohho caps.
TRAINING_COMMAND = MappingProxyType(
    {
        "vx_forward_max": 3.0,
        "vx_backward_max": 0.7,
        "vy_max": 0.8,
        "vyaw_max": 2.0,
    }
)

# Policy joint order from tasks/locomotion/robots/t2/__init__.py.
# Head and wrist joints are not in the action vector; PD holds their defaults.
WALK_POLICY_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "waist_yaw_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

# Same order as WALK_POLICY_JOINTS. T2LocomotionPolicyCfg.action_scale.
ACTION_SCALE: tuple[float, ...] = (
    0.2608,
    0.2608,
    0.2212,
    0.2773,
    0.2773,
    0.1740,
    0.2773,
    0.2773,
    0.3155,
    0.2773,
    0.2773,
    0.3155,
    0.3155,
    0.3155,
    0.3155,
    0.3155,
    0.3155,
    0.3155,
    0.3155,
    0.1059,
    0.1059,
    0.0950,
    0.0950,
)

# T2WalkControllerCfg joint_stiffness, joint_names order (all 31).
STIFFNESS: tuple[float, ...] = (
    20.0,
    30.0,
    67.1133,
    54.0854,
    54.0854,
    54.0854,
    2.0,
    2.0,
    2.0,
    67.1133,
    54.0854,
    54.0854,
    54.0854,
    2.0,
    2.0,
    2.0,
    155.9961,
    206.9129,
    110.9343,
    110.9343,
    110.9343,
    110.9343,
    110.9343,
    141.7042,
    52.6468,
    110.9343,
    110.9343,
    110.9343,
    110.9343,
    141.7042,
    52.6468,
)

# T2WalkControllerCfg joint_damping, same order.
DAMPING: tuple[float, ...] = (
    0.5,
    1.0,
    2.1362,
    1.7215,
    1.7215,
    1.7215,
    0.1,
    0.1,
    0.1,
    2.1362,
    1.7215,
    1.7215,
    1.7215,
    0.1,
    0.1,
    0.1,
    1.7215,
    1.7215,
    3.5311,
    3.5311,
    3.5311,
    3.5311,
    3.5311,
    1.7215,
    1.7215,
    3.5311,
    3.5311,
    3.5311,
    3.5311,
    1.7215,
    1.7215,
)

# T2WalkControllerCfg default_joint_pos, same order.
DEFAULT_Q: tuple[float, ...] = (
    0.0,
    0.0,
    0.2,
    -1.3,
    0.0,
    -0.5,
    0.0,
    0.0,
    0.0,
    0.2,
    1.3,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


class HeadingObservation:
    """Heading error and a standing flag, appended after the six named terms.

    This is the extra block from ``T2LocomotionPolicy.compute_observation``.
    The heading target resets when the command is near zero or changes, and
    otherwise integrates yaw rate over the policy period. The error is
    wrapped and limited to ±0.5 rad. The flag is 1 when the command norm is
    below 0.01.
    """

    def __init__(self) -> None:
        self._target: float | None = None
        self._last = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._target = None
        self._last = (0.0, 0.0, 0.0)

    def extend(
        self,
        *,
        yaw: float,
        command: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, float]:
        vx, vy, wz = command
        standing = math.sqrt(vx * vx + vy * vy + wz * wz) < STANDING_COMMAND_NORM
        changed = any(
            abs(current - previous) > COMMAND_CHANGE
            for current, previous in zip(command, self._last)
        )
        if self._target is None or standing or changed:
            self._target = yaw
        else:
            advanced = self._target + dt * wz
            self._target = math.atan2(math.sin(advanced), math.cos(advanced))
        self._last = (vx, vy, wz)
        error = math.atan2(math.sin(self._target - yaw), math.cos(self._target - yaw))
        error = max(-HEADING_ERROR_LIMIT, min(HEADING_ERROR_LIMIT, error))
        return (error, 1.0 if standing else 0.0)


def observation_terms() -> tuple[ObservationTerm, ...]:
    """The six runner terms. Joint velocity uses the deployment scale of 1."""
    terms: list[ObservationTerm] = []
    for term in default_observation_terms():
        scale = DQ_SCALE if term.name == "dq" else term.scale
        terms.append(ObservationTerm(term.name, scale=scale))
    return tuple(terms)


def walk_policy_layout(checkpoint: str) -> PolicyLayout:
    """Obs/action layout for the pinned checkpoint. The file need not exist yet."""
    if len(WALK_POLICY_JOINTS) != len(ACTION_SCALE):
        raise RuntimeError("walk action scale does not match the policy joints")
    return PolicyLayout(
        joint_names=WALK_POLICY_JOINTS,
        terms=observation_terms(),
        action_scale=ACTION_SCALE,
        rate_hz=POLICY_RATE_HZ,
        checkpoint=checkpoint,
        q_relative=True,
        history_length=HISTORY_LENGTH,
        history_init="repeat",
        clip_observation=CLIP_OBSERVATION,
        clip_action=CLIP_ACTION,
        extra_factory=HeadingObservation,
    )


def walk_sim_config(*, mjcf: str, checkpoint: str) -> HumanoidSimConfig:
    """One config for the serial MJCF and the walk checkpoint."""
    if len(STIFFNESS) != len(T2_JOINT_NAMES) or len(DAMPING) != len(T2_JOINT_NAMES):
        raise RuntimeError("walk gains do not match the joint list")
    if len(DEFAULT_Q) != len(T2_JOINT_NAMES):
        raise RuntimeError("default pose does not match the joint list")
    return HumanoidSimConfig(
        mjcf=mjcf,
        joint_names=T2_JOINT_NAMES,
        kp=STIFFNESS,
        kd=DAMPING,
        default_q=DEFAULT_Q,
        policy=walk_policy_layout(checkpoint),
        base_body=BASE_BODY,
        timestep=PHYSICS_DT_S,
        base_pos=BASE_POS,
    )


def bind_walk_config() -> HumanoidSimConfig:
    """Fetch the pin if needed and bind the walk config for ``booster-t2``."""
    mjcf, checkpoint = ensure_assets()
    config = walk_sim_config(mjcf=str(mjcf), checkpoint=str(checkpoint))
    bind_sim_config("booster-t2", config)
    return config
