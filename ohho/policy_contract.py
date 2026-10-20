"""Policy input and output layout for train and serve.

One contract describes a mobile base and a humanoid. Names and widths come
from the robot spec, or from the dataset feature names the recorder stored.

A mobile arm action is joint targets in spec order, then body velocity. A
legged action is body velocity, then a mode id that serve does not apply.
A humanoid action is body velocity, then the arm groups (or end-effector
poses), then grippers. Leg joints stay out of the action. Body velocity is
passed to :meth:`~ohho.robot.Robot.drive` so the locomotion policy keeps
the legs. Arm targets go through :meth:`~ohho.robot.Robot.move_joints`,
:meth:`~ohho.robot.Robot.move_ee` and :meth:`~ohho.robot.Robot.gripper`,
which enter :class:`~ohho.safety.SafetyGate` when the link is gated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from . import capabilities as caps
from .data.humanoid_features import (
    ARM_ACTION_ARM_JOINTS,
    ARM_ACTION_EE_POSE,
    ARM_ACTIONS,
    EE_POSE_AXES,
    action_names,
    state_names,
)
from .data.recorder import (
    BASE_ACTION_NAMES,
    BASE_STATE_NAMES,
    GO2_ACTION_NAMES,
    GO2_STATE_NAMES,
)
from .robot import Robot

_VELOCITY_KEYS = (
    ("vx", "vy", "wz"),
    ("base_vx", "base_vy", "base_omega"),
)


@dataclass(frozen=True)
class NamedSpan:
    """Indices of one slice of the action vector."""

    name: str
    kind: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class PolicyContract:
    """State and action names for one robot, plus how to apply the action."""

    kind: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    arm_action: Optional[str]
    spans: tuple[NamedSpan, ...]

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def spans_of(self, kind: str) -> tuple[NamedSpan, ...]:
        return tuple(span for span in self.spans if span.kind == kind)


@dataclass(frozen=True)
class ApplyResult:
    """What :func:`apply_policy_action` sent through the robot API.

    A gated link may still refuse a joint target. This records the request.
    """

    velocity: Optional[tuple[float, float, float]]
    arm_groups: tuple[tuple[str, tuple[float, ...]], ...]
    arm_sequence: Optional[tuple[float, ...]]
    end_effectors: tuple[tuple[str, tuple[float, ...]], ...]
    grippers: tuple[tuple[str, float], ...]


def contract_for_spec(spec: Any, arm_action: Optional[str] = None) -> PolicyContract:
    """Layout for ``spec``. A humanoid defaults to arm-joint actions."""
    if _is_humanoid(spec):
        form = arm_action or ARM_ACTION_ARM_JOINTS
        if form not in ARM_ACTIONS:
            raise ValueError("arm_action must be 'ee_pose' or 'arm_joints'")
        return contract_from_names(
            state_names(spec),
            action_names(spec, form),
            arm_action=form,
            kind="humanoid",
        )
    if _is_legged(spec):
        return contract_from_names(GO2_STATE_NAMES, GO2_ACTION_NAMES, kind="legged")
    if _has(spec, caps.MANIPULATION):
        joints = tuple(str(name) for name in getattr(spec, "joint_names", ()) or ())
        tail = ("base_vx", "base_vy", "base_omega")
        names = joints + tail
        return contract_from_names(names, names, kind="mobile_arm")
    return contract_from_names(BASE_STATE_NAMES, BASE_ACTION_NAMES, kind="base")


def contract_from_dataset(root: str) -> PolicyContract:
    """Layout stored in a dataset's ``meta/info.json``."""
    from .data.reader import DatasetReader

    return contract_from_reader(DatasetReader(root))


def contract_from_reader(reader: Any) -> PolicyContract:
    """Layout from a :class:`~ohho.data.reader.DatasetReader`."""
    info = reader.info
    return contract_from_names(
        _feature_names(info.features.get("observation.state")),
        _feature_names(info.features.get("action")),
        arm_action=getattr(info, "arm_action", None),
    )


def contract_from_names(
    state: Sequence[str],
    action: Sequence[str],
    *,
    arm_action: Optional[str] = None,
    kind: Optional[str] = None,
) -> PolicyContract:
    """Build a contract from feature names."""
    state_names_ = tuple(str(name) for name in state)
    action_names_ = tuple(str(name) for name in action)
    form = _infer_arm_action(action_names_, arm_action)
    resolved = kind or _infer_kind(state_names_, action_names_, form)
    return PolicyContract(
        kind=resolved,
        state_names=state_names_,
        action_names=action_names_,
        arm_action=form if resolved == "humanoid" else None,
        spans=_spans(resolved, action_names_, form),
    )


def apply_policy_action(
    robot: Robot,
    action: Sequence[float],
    contract: PolicyContract,
) -> ApplyResult:
    """Map one policy vector onto the robot API.

    Humanoid leg groups are not in ``contract`` and are not commanded.
    Body velocity is always :meth:`Robot.drive`. Arm motion uses the group
    form on a humanoid and the joint-sequence form on a mobile arm.
    """
    if len(action) < contract.action_dim:
        raise ValueError(
            f"action length {len(action)} is shorter than {contract.action_dim}"
        )
    values = tuple(_finite(item) for item in action[: contract.action_dim])
    velocity: Optional[tuple[float, float, float]] = None
    groups: list[tuple[str, tuple[float, ...]]] = []
    sequence: Optional[tuple[float, ...]] = None
    effectors: list[tuple[str, tuple[float, ...]]] = []
    grippers: list[tuple[str, float]] = []

    for span in contract.spans_of("velocity"):
        velocity = (
            values[span.indices[0]],
            values[span.indices[1]],
            values[span.indices[2]],
        )
        robot.drive(vx=velocity[0], vy=velocity[1], w=velocity[2])
    for span in contract.spans_of("arm_sequence"):
        sequence = tuple(values[index] for index in span.indices)
        robot.move_joints(sequence)
    for span in contract.spans_of("arm_joints"):
        target = tuple(values[index] for index in span.indices)
        groups.append((span.name, target))
        robot.move_joints(span.name, target)
    for span in contract.spans_of("ee_pose"):
        pose = tuple(values[index] for index in span.indices)
        effectors.append((span.name, pose))
        robot.move_ee(span.name, pose, 0.0)
    for span in contract.spans_of("gripper"):
        command = values[span.indices[0]]
        grippers.append((span.name, command))
        robot.gripper(span.name, command)
    return ApplyResult(
        velocity=velocity,
        arm_groups=tuple(groups),
        arm_sequence=sequence,
        end_effectors=tuple(effectors),
        grippers=tuple(grippers),
    )


def apply_prediction(
    robot: Robot,
    prediction: Any,
    contract: Optional[PolicyContract] = None,
) -> ApplyResult:
    """Apply a serve prediction. Names on the prediction select the contract."""
    vector = _prediction_vector(prediction)
    chosen = contract or _contract_from_prediction(robot, prediction)
    return apply_policy_action(robot, vector, chosen)


def _prediction_vector(prediction: Any) -> list[float]:
    if isinstance(prediction, Mapping):
        raw = prediction.get("vector")
    else:
        raw = prediction
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("prediction vector must be a sequence of numbers")
    return [float(item) for item in raw]


def _contract_from_prediction(robot: Robot, prediction: Any) -> PolicyContract:
    if isinstance(prediction, Mapping):
        names = prediction.get("names")
        if (
            isinstance(names, Sequence)
            and not isinstance(names, (str, bytes))
            and names
        ):
            state = prediction.get("state_names") or ()
            if isinstance(state, (str, bytes)) or not isinstance(state, Sequence):
                state = ()
            return contract_from_names(
                state,
                names,
                arm_action=(
                    prediction.get("arm_action")
                    if isinstance(prediction.get("arm_action"), str)
                    else None
                ),
            )
    return contract_for_spec(robot.spec)


def _spans(
    kind: str, names: tuple[str, ...], arm_action: Optional[str]
) -> tuple[NamedSpan, ...]:
    spans: list[NamedSpan] = []
    velocity = _velocity_indices(names)
    if velocity is not None:
        spans.append(NamedSpan("base", "velocity", velocity))
    if kind == "humanoid":
        spans.extend(_humanoid_arm_spans(names, arm_action))
        spans.extend(_gripper_spans(names))
    elif kind == "mobile_arm":
        used = set(velocity or ())
        joints = tuple(index for index in range(len(names)) if index not in used)
        if joints:
            spans.append(NamedSpan("arm", "arm_sequence", joints))
    elif kind == "legged" and "mode_id" in names:
        spans.append(NamedSpan("mode", "mode", (names.index("mode_id"),)))
    return tuple(spans)


def _humanoid_arm_spans(
    names: tuple[str, ...], arm_action: Optional[str]
) -> list[NamedSpan]:
    if arm_action == ARM_ACTION_EE_POSE:
        spans: list[NamedSpan] = []
        for side in ("left", "right"):
            keys = tuple(f"{side}_ee_{axis}" for axis in EE_POSE_AXES)
            if all(key in names for key in keys):
                spans.append(
                    NamedSpan(side, "ee_pose", tuple(names.index(key) for key in keys))
                )
        return spans
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, name in enumerate(names):
        if "." not in name:
            continue
        group, _joint = name.split(".", 1)
        if "arm" not in group.lower().split("_"):
            continue
        if group not in groups:
            order.append(group)
            groups[group] = []
        groups[group].append(index)
    return [NamedSpan(group, "arm_joints", tuple(groups[group])) for group in order]


def _gripper_spans(names: tuple[str, ...]) -> list[NamedSpan]:
    spans: list[NamedSpan] = []
    for side in ("left", "right"):
        key = f"{side}_gripper"
        if key in names:
            spans.append(NamedSpan(side, "gripper", (names.index(key),)))
    return spans


def _velocity_indices(names: Sequence[str]) -> Optional[tuple[int, int, int]]:
    pool = list(names)
    for keys in _VELOCITY_KEYS:
        if all(key in pool for key in keys):
            return (
                pool.index(keys[0]),
                pool.index(keys[1]),
                pool.index(keys[2]),
            )
    return None


def _infer_arm_action(names: Sequence[str], declared: Optional[str]) -> Optional[str]:
    if declared in ARM_ACTIONS:
        return declared
    if any("_ee_" in name for name in names):
        return ARM_ACTION_EE_POSE
    for name in names:
        if "." not in name:
            continue
        group = name.split(".", 1)[0]
        if "arm" in group.lower().split("_"):
            return ARM_ACTION_ARM_JOINTS
    return None


def _infer_kind(
    state: Sequence[str], action: Sequence[str], arm_action: Optional[str]
) -> str:
    if arm_action in ARM_ACTIONS or "base_roll" in state:
        return "humanoid"
    if "posture_id" in state or (
        len(state) == len(GO2_STATE_NAMES) and "fl_hip_q" in state
    ):
        return "legged"
    if len(action) > 3 and tuple(action[-3:]) == ("base_vx", "base_vy", "base_omega"):
        return "mobile_arm"
    return "base"


def _feature_names(feature: object) -> tuple[str, ...]:
    if not isinstance(feature, Mapping):
        return ()
    raw = feature.get("names")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    return tuple(str(name) for name in raw)


def _is_humanoid(spec: Any) -> bool:
    kind = getattr(spec, "robot_class", None)
    if kind == "humanoid" or getattr(kind, "value", None) == "humanoid":
        return True
    return _has(spec, caps.HUMANOID_POSTURE) or _has(spec, caps.HUMANOID_LOCOMOTION)


def _is_legged(spec: Any) -> bool:
    kind = getattr(spec, "robot_class", None)
    if kind == "quadruped" or getattr(kind, "value", None) == "quadruped":
        return True
    return _has(spec, caps.LEGGED_POSE)


def _has(spec: Any, capability: str) -> bool:
    has = getattr(spec, "has", None)
    if callable(has):
        return bool(has(capability))
    return capability in tuple(getattr(spec, "capabilities", ()) or ())


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"action value must be a real number, not {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("action value must be finite")
    return number


__all__ = [
    "ApplyResult",
    "NamedSpan",
    "PolicyContract",
    "apply_policy_action",
    "apply_prediction",
    "contract_for_spec",
    "contract_from_dataset",
    "contract_from_names",
    "contract_from_reader",
]
