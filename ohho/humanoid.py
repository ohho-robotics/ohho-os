"""Generic humanoid modes and the calls that use them.

A behavior asks for a capability (``humanoid.posture``, ``head.gaze``, and so
on). When the robot does not have it, the call is a no-op and returns the
robot. When it does, the call goes through :class:`~ohho.safety.SafetyGate`
if the transport is gated, and through ``send_humanoid`` otherwise.

Walking is :meth:`~ohho.robot.Robot.drive`. This module does not add a second
velocity command.

Modes are ``OFF``, ``DAMPING``, ``PREPARE``, ``WALK``, ``CUSTOM``, ``FALLEN``
and ``UNKNOWN``. A plugin maps its own mode names onto these.
:func:`parse_mode` turns anything it does not recognise into ``UNKNOWN``.
:func:`is_standing` is false for ``UNKNOWN``.

Fall state is separate from mode: ``upright``, ``falling``, ``fallen``,
``recovering`` and ``unknown``. :func:`is_upright` is true only for an
explicit ``upright`` value. ``unknown`` and a missing fall state are not
upright. The ``custom["humanoid"]`` block is built in
:mod:`ohho.humanoid_telemetry`.

Low-level :func:`move_joints` is refused unless the mode is ``CUSTOM``, the
link is simulated, and :func:`policy_allows_low_level_joints` is true. That
hook stays false on an ungated link. A gated link asks
:class:`ohho.humanoid_policy.HumanoidPolicy`, which allows the command only
in ``CUSTOM`` on a sim target and applies joint limits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Optional, Protocol, TypeVar

from . import capabilities as caps
from .safety import SafetyGate
from .schema import Telemetry

# Must match ``SafetyGate``'s humanoid action set.
ACTION_NAMES: tuple[str, ...] = (
    "prepare",
    "walk_mode",
    "damp",
    "get_up",
    "lie_down",
    "look",
    "move_ee",
    "gripper",
    "hand",
    "move_joints",
)

_SIDES = frozenset({"left", "right"})


class HumanoidMode(str, Enum):
    """Generic posture. Plugins map their own names onto these values.

    ``UNKNOWN`` is the fallback. It is not a standing mode.
    """

    OFF = "OFF"
    DAMPING = "DAMPING"
    PREPARE = "PREPARE"
    WALK = "WALK"
    CUSTOM = "CUSTOM"
    FALLEN = "FALLEN"
    UNKNOWN = "UNKNOWN"


class FallState(str, Enum):
    """Whether the humanoid is upright.

    ``UNKNOWN`` is the fallback. It is not upright, and neither is a missing
    value. Plugins map their own fall reports onto these names.
    """

    UPRIGHT = "upright"
    FALLING = "falling"
    FALLEN = "fallen"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


_STANDING = frozenset({HumanoidMode.PREPARE, HumanoidMode.WALK})
_FALL_BY_NAME = {member.value: member for member in FallState}
_MISSING = object()


class _Commanded(Protocol):
    """The slice of ``Robot`` these calls use."""

    @property
    def transport(self) -> object: ...

    def has(self, capability: str) -> bool: ...

    def telemetry(self) -> Telemetry: ...


_R = TypeVar("_R", bound=_Commanded)


def parse_mode(value: object) -> HumanoidMode:
    """Return the generic mode named by ``value``.

    Enum members pass through. Names match case-insensitively (``"walk"`` and
    ``"WALK"`` are :attr:`HumanoidMode.WALK`). Anything else, including
    ``None`` and an unrecognised string, is :attr:`HumanoidMode.UNKNOWN`.
    """
    if isinstance(value, HumanoidMode):
        return value
    if isinstance(value, str):
        key = value.strip().upper()
        try:
            return HumanoidMode[key]
        except KeyError:
            return HumanoidMode.UNKNOWN
    return HumanoidMode.UNKNOWN


def is_standing(mode: object) -> bool:
    """True only for :attr:`~HumanoidMode.PREPARE` and :attr:`~HumanoidMode.WALK`.

    ``UNKNOWN`` is never standing. ``OFF``, ``DAMPING``, ``FALLEN`` and
    ``CUSTOM`` are not standing either. A value that is not a
    :class:`HumanoidMode` — including a raw string — is not standing.
    """
    if not isinstance(mode, HumanoidMode) or mode is HumanoidMode.UNKNOWN:
        return False
    return mode in _STANDING


def parse_fall_state(value: object) -> FallState:
    """Return the fall state named by ``value``.

    Enum members pass through. Names match case-insensitively (``"Upright"``
    and ``"upright"`` are :attr:`FallState.UPRIGHT`). Anything else, including
    ``None`` and an unrecognised string, is :attr:`FallState.UNKNOWN`.
    """
    if isinstance(value, FallState):
        return value
    if isinstance(value, str):
        found = _FALL_BY_NAME.get(value.strip().lower())
        if found is not None:
            return found
    return FallState.UNKNOWN


def fall_state_of(value: object) -> FallState:
    """Fall state carried by ``value``. Missing or unrecognised is ``UNKNOWN``.

    A :class:`~ohho.schema.Telemetry` is read only at
    ``custom["humanoid"]["fall_state"]``. A mapping that contains ``humanoid``
    is unwrapped to that block. A mapping that itself has ``fall_state`` is
    the block. ``UNKNOWN`` is never upright.
    """
    if isinstance(value, FallState):
        return value
    if isinstance(value, Telemetry):
        custom = value.custom
        block = custom.get("humanoid") if isinstance(custom, Mapping) else None
        return _fall_state_in_block(block)
    if isinstance(value, str):
        return parse_fall_state(value)
    if isinstance(value, Mapping):
        if "humanoid" in value:
            return _fall_state_in_block(value.get("humanoid"))
        if "fall_state" in value:
            return parse_fall_state(value.get("fall_state"))
        return FallState.UNKNOWN
    state = getattr(value, "fall_state", _MISSING)
    if state is not _MISSING:
        if isinstance(state, FallState):
            return state
        return parse_fall_state(state)
    return FallState.UNKNOWN


def is_upright(value: object) -> bool:
    """True only when the fall state is explicitly ``upright``.

    ``unknown``, a missing block, a missing ``fall_state``, and every other
    state return False.
    """
    return fall_state_of(value) is FallState.UPRIGHT


def _fall_state_in_block(block: object) -> FallState:
    if isinstance(block, Mapping):
        return parse_fall_state(block.get("fall_state"))
    if isinstance(block, (str, bytes)) or block is None:
        return FallState.UNKNOWN
    state = getattr(block, "fall_state", _MISSING)
    if state is _MISSING:
        return FallState.UNKNOWN
    if isinstance(state, FallState):
        return state
    return parse_fall_state(state)


def mode_of(telemetry: Optional[Telemetry]) -> HumanoidMode:
    """Mode stored at ``telemetry.custom["humanoid"]["mode"]``.

    A missing or malformed block is :attr:`HumanoidMode.UNKNOWN`.
    """
    if telemetry is None:
        return HumanoidMode.UNKNOWN
    block = telemetry.custom.get("humanoid")
    if isinstance(block, (str, bytes)) or not isinstance(block, Mapping):
        return HumanoidMode.UNKNOWN
    return parse_mode(block.get("mode"))


def policy_allows_low_level_joints() -> bool:
    """Whether the class policy allows low-level joint targets.

    Always false. An ungated link therefore refuses low-level joints. The
    gate's humanoid policy is what allows them, and only in ``CUSTOM`` on a
    sim target. Callers still have to be in ``CUSTOM`` on a simulated link.
    """
    return False


def joints_permitted(mode: object, *, simulated: bool, policy_allows: bool) -> bool:
    """Low-level joints are allowed only when every condition holds.

    The mode must be :attr:`HumanoidMode.CUSTOM` (so ``UNKNOWN`` never
    permits them), the link must be simulated, and the policy must allow it.
    """
    if mode is not HumanoidMode.CUSTOM:
        return False
    if not simulated:
        return False
    return bool(policy_allows)


def prepare(robot: _R) -> _R:
    """Stand and hold balance, ready to walk. No-op without ``humanoid.posture``."""
    return _posture(robot, "prepare")


def walk_mode(robot: _R) -> _R:
    """Enable locomotion. No-op without ``humanoid.posture``.

    The body-velocity command after this is :meth:`Robot.drive`.
    """
    return _posture(robot, "walk_mode")


def damp(robot: _R) -> _R:
    """Put every joint in damping. The robot may collapse.

    No-op without ``humanoid.posture``. This is not an e-stop.
    """
    return _posture(robot, "damp")


def get_up(robot: _R) -> _R:
    """Run the get-up recovery. No-op without ``humanoid.posture``."""
    return _posture(robot, "get_up")


def lie_down(robot: _R) -> _R:
    """Run the lie-down recovery. No-op without ``humanoid.posture``."""
    return _posture(robot, "lie_down")


def look(robot: _R, pitch: object, yaw: object) -> _R:
    """Point the head. Angles are radians. No-op without ``head.gaze``."""
    if not robot.has(caps.HEAD_GAZE):
        return robot
    pitch_rad = _finite("pitch", pitch)
    yaw_rad = _finite("yaw", yaw)
    return _dispatch(robot, "look", (pitch_rad, yaw_rad), pitch=pitch_rad, yaw=yaw_rad)


def move_ee(robot: _R, side: object, pose: object, t: object) -> _R:
    """Move one hand to ``pose`` over ``t`` seconds.

    ``side`` is ``"left"`` or ``"right"``. ``pose`` is a sequence of finite
    numbers in the end-effector frame the transport owns. No-op without
    ``manipulation.dual_arm``.
    """
    if not robot.has(caps.MANIPULATION_DUAL_ARM):
        return robot
    which = _side(side)
    target = _float_list("pose", pose)
    duration = _finite("t", t)
    if duration < 0:
        raise ValueError("t must be >= 0")
    return _dispatch(
        robot,
        "move_ee",
        (which, target, duration),
        side=which,
        pose=target,
        t=duration,
    )


def gripper(robot: _R, side: object, cmd: object) -> _R:
    """Command one gripper. No-op without ``end_effector.gripper``.

    ``cmd`` is a finite number or a short command string.
    """
    if not robot.has(caps.END_EFFECTOR_GRIPPER):
        return robot
    which = _side(side)
    command = _gripper_cmd(cmd)
    return _dispatch(robot, "gripper", (which, command), side=which, cmd=command)


def hand(robot: _R, side: object, fingers: object) -> _R:
    """Command one dexterous hand. No-op without ``end_effector.dex_hand``.

    ``fingers`` is a sequence of finite joint targets.
    """
    if not robot.has(caps.END_EFFECTOR_DEX_HAND):
        return robot
    which = _side(side)
    targets = _float_list("fingers", fingers)
    return _dispatch(robot, "hand", (which, targets), side=which, fingers=targets)


def move_joints(
    robot: _R,
    group: object,
    q: object = None,
    kp: object = None,
    kd: object = None,
) -> _R:
    """Command one joint group. No-op without ``whole_body.joint``.

    Refused unless the robot is in ``CUSTOM``, the link is simulated, and the
    policy allows it. A gated transport still receives the call, and the gate
    refuses ``move_joints``. ``kp`` and ``kd`` are optional finite gains.
    """
    if not robot.has(caps.WHOLE_BODY_JOINT):
        return robot
    name = _group(group)
    targets = _float_list("q", q)
    kp_v = None if kp is None else _finite("kp", kp)
    kd_v = None if kd is None else _finite("kd", kd)
    args = (name, targets, kp_v, kd_v)
    transport = robot.transport
    if isinstance(transport, SafetyGate):
        transport.command("move_joints", *args)
        return robot
    if not joints_permitted(
        mode_of(robot.telemetry()),
        simulated=_is_simulated(transport),
        policy_allows=policy_allows_low_level_joints(),
    ):
        return robot
    return _send(
        robot,
        "move_joints",
        group=name,
        q=targets,
        kp=kp_v,
        kd=kd_v,
    )


def _posture(robot: _R, action: str) -> _R:
    if not robot.has(caps.HUMANOID_POSTURE):
        return robot
    return _dispatch(robot, action, ())


def _dispatch(robot: _R, action: str, args: tuple[object, ...], **fields: object) -> _R:
    transport = robot.transport
    if isinstance(transport, SafetyGate):
        transport.command(action, *args)
        return robot
    return _send(robot, action, **fields)


def _send(robot: _R, action: str, **fields: object) -> _R:
    transport = robot.transport
    send = getattr(transport, "send_humanoid", None)
    if not callable(send):
        raise AttributeError(
            f"{type(transport).__name__} does not implement send_humanoid"
        )
    send(action, **fields)
    return robot


def _is_simulated(transport: object) -> bool:
    return getattr(transport, "protocol", "") == "simulated"


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _float_list(name: str, values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    return tuple(_finite(name, item) for item in values)


def _side(side: object) -> str:
    if not isinstance(side, str):
        raise TypeError(f"side must be 'left' or 'right', not {type(side).__name__}")
    which = side.strip().lower()
    if which not in _SIDES:
        raise ValueError("side must be 'left' or 'right'")
    return which


def _group(group: object) -> str:
    if not isinstance(group, str):
        raise TypeError(f"group must be a string, not {type(group).__name__}")
    name = group.strip()
    if not name:
        raise ValueError("group must not be empty")
    return name


def _gripper_cmd(cmd: object) -> float | str:
    if isinstance(cmd, str):
        text = cmd.strip()
        if not text:
            raise ValueError("cmd must not be empty")
        return text
    return _finite("cmd", cmd)


__all__ = [
    "ACTION_NAMES",
    "FallState",
    "HumanoidMode",
    "damp",
    "fall_state_of",
    "get_up",
    "gripper",
    "hand",
    "is_standing",
    "is_upright",
    "joints_permitted",
    "lie_down",
    "look",
    "mode_of",
    "move_ee",
    "move_joints",
    "parse_fall_state",
    "parse_mode",
    "policy_allows_low_level_joints",
    "prepare",
    "walk_mode",
]
