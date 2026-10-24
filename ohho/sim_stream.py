"""Robot-agnostic sim-serve stream: pose, ordered joints, and status.

The WebSocket server still sends the existing ``odom`` and ``joints``
object. This module adds the documented frame beside them. Joint names
come from the robot descriptor when that document lists them, otherwise
from ``spec.joint_names``. Posture requests use the spec's declared
``humanoid.posture`` modes. Nothing here names a robot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import capabilities as caps
from .humanoid_policy import explicit_sim_target
from .humanoid_sim import rpy_to_quat_wxyz
from .registry import RobotSpec, UnknownRobot
from .safety import ARM_ENV, ARM_PHRASE

# Telemetry broadcast rate. About 30 Hz.
STREAM_HZ = 30.0

# Request token -> (declared posture mode it requires, transport action).
# ``walk_mode`` is the transport name for the declared mode ``walk``.
_MODE_REQUESTS: dict[str, tuple[str, str]] = {
    "prepare": ("prepare", "prepare"),
    "walk": ("walk", "walk_mode"),
    "walk_mode": ("walk", "walk_mode"),
    "damping": ("damping", "damp"),
    "damp": ("damping", "damp"),
    "custom": ("custom", "custom"),
    "get_up": ("get_up", "get_up"),
    "lie_down": ("lie_down", "lie_down"),
}


@dataclass(frozen=True)
class ModeDecision:
    """Whether a ``modes`` request may be sent, and the action to send."""

    accepted: bool
    action: str = ""
    reason: str = ""


def declared_posture_modes(spec: RobotSpec | None) -> frozenset[str]:
    """Posture modes the spec declares on ``humanoid.posture``.

    An absent capability, or a capability with no ``modes`` list, declares
    nothing. The robot id is not consulted.
    """
    if spec is None or not spec.has(caps.HUMANOID_POSTURE):
        return frozenset()
    params = spec.capability_params
    if not isinstance(params, Mapping):
        return frozenset()
    raw = params.get(caps.HUMANOID_POSTURE)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        return frozenset()
    modes = raw.get("modes")
    if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
        return frozenset()
    return frozenset(mode for mode in modes if isinstance(mode, str) and mode)


def mode_confirmation_required(backend: str, uri: str | None) -> bool:
    """True for a robot backend whose URI is not an explicit sim target.

    ``backend=sim`` never needs the phrase. ``sim://``, ``simulated://`` and a
    loopback humanoid-mujoco URI with ``sim_target=True`` are sim targets.
    """
    if backend != "robot":
        return False
    return not explicit_sim_target(uri)


def decide_mode_request(
    spec: RobotSpec | None,
    request: object,
    *,
    backend: str,
    uri: str | None,
    confirm: object = None,
    env: Mapping[str, str] | None = None,
) -> ModeDecision:
    """Accept a posture request or refuse it.

    The request must name a mode the spec declares. On a non-sim robot URI the
    arming phrase and ``OHHO_ARM_HARDWARE=1`` must already be present. This
    function only reads ``env``.
    """
    if env is None:
        env = {}
    if spec is None or not spec.has(caps.HUMANOID_POSTURE):
        return ModeDecision(False, reason="capability")
    if backend not in ("sim", "robot"):
        return ModeDecision(False, reason="backend")
    if not isinstance(request, str) or not request.strip():
        return ModeDecision(False, reason="request")
    name = request.strip()
    mapped = _MODE_REQUESTS.get(name)
    declared = declared_posture_modes(spec)
    if mapped is None or (name not in declared and mapped[0] not in declared):
        return ModeDecision(False, reason="undeclared")
    if mode_confirmation_required(backend, uri):
        if confirm != ARM_PHRASE:
            return ModeDecision(False, reason="confirm")
        if env.get(ARM_ENV) != "1":
            return ModeDecision(False, reason="arm_env")
    return ModeDecision(True, action=mapped[1])


def stream_joint_names(spec: RobotSpec | None) -> tuple[str, ...]:
    """Joint names in descriptor order.

    A descriptor that lists ``joints`` wins. Otherwise the spec's
    ``joint_names`` are the order. An id that is not registered uses the
    spec object in hand.
    """
    if spec is None:
        return ()
    documented = _descriptor_joints(spec)
    if documented:
        return documented
    return _name_tuple(getattr(spec, "joint_names", ()))


def ordered_joint_state(
    names: Sequence[str], joints: Mapping[str, float]
) -> dict[str, list[Any]]:
    """Positions parallel to ``names``. A missing joint is ``0.0``."""
    positions: list[float] = []
    for name in names:
        positions.append(_num(joints.get(name, 0.0)))
    return {"names": [str(name) for name in names], "positions": positions}


def pose_from_sources(
    odom: Mapping[str, float] | None,
    *,
    telemetry: Any = None,
    base_position: Sequence[float] | None = None,
) -> dict[str, Any]:
    """World pose. Orientation is a quaternion ``x, y, z, w``.

    Planar ``odom`` supplies ``x``, ``y`` and yaw. A humanoid telemetry
    block replaces roll, pitch and yaw from ``base_rpy``. ``base_position``
    is ``(x, y, z)`` in metres when the transport reports a floating base.
    """
    x = y = z = 0.0
    roll = pitch = yaw = 0.0
    if isinstance(odom, Mapping):
        x = _num(odom.get("x"))
        y = _num(odom.get("y"))
        yaw = _num(odom.get("yaw", odom.get("theta")))
    block = _humanoid_block(telemetry)
    if block is not None:
        rpy = block.get("base_rpy")
        if (
            isinstance(rpy, Sequence)
            and not isinstance(rpy, (str, bytes))
            and len(rpy) == 3
        ):
            roll, pitch, yaw = _num(rpy[0]), _num(rpy[1]), _num(rpy[2])
        planar = block.get("odom")
        if isinstance(planar, Mapping):
            if "x" in planar:
                x = _num(planar.get("x"))
            if "y" in planar:
                y = _num(planar.get("y"))
            if "z" in planar:
                z = _num(planar.get("z"))
            if "yaw" in planar:
                yaw = _num(planar.get("yaw"))
        height = block.get("base_position")
        if (
            isinstance(height, Sequence)
            and not isinstance(height, (str, bytes))
            and len(height) == 3
        ):
            z = _num(height[2])
    if (
        base_position is not None
        and not isinstance(base_position, (str, bytes))
        and len(base_position) == 3
    ):
        x = _num(base_position[0])
        y = _num(base_position[1])
        z = _num(base_position[2])
    quat_w, quat_x, quat_y, quat_z = rpy_to_quat_wxyz(roll, pitch, yaw)
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": quat_x, "y": quat_y, "z": quat_z, "w": quat_w},
    }


def posture_label(telemetry: Any) -> str:
    """Generic mode string on the humanoid block, or ``\"\"``."""
    block = _humanoid_block(telemetry)
    if block is None:
        return ""
    mode = block.get("mode")
    return mode if isinstance(mode, str) else ""


def build_stream_frame(
    *,
    backend: str,
    odom: Mapping[str, float],
    joints: Mapping[str, float],
    names: Sequence[str],
    pose: Mapping[str, Any],
    estop: bool,
    deadman: bool,
    connected: bool,
    posture: str,
    camera: str | None = None,
    modes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One server JSON object. Old keys stay; pose, joint_state and status are added."""
    frame: dict[str, Any] = {
        "odom": {
            "x": _num(odom.get("x")),
            "y": _num(odom.get("y")),
            "yaw": _num(odom.get("yaw", odom.get("theta"))),
        },
        "pose": dict(pose),
        "joints": {str(name): _num(value) for name, value in joints.items()},
        "joint_state": ordered_joint_state(names, joints),
        "status": {
            "backend": backend,
            "estop": bool(estop),
            "deadman": bool(deadman),
            "connected": bool(connected),
            "posture": posture,
        },
        "backend": backend,
    }
    if modes is not None:
        frame["modes"] = dict(modes)
    if camera:
        frame["camera"] = camera
    else:
        frame["no_camera"] = True
    return frame


def _descriptor_joints(spec: RobotSpec) -> tuple[str, ...]:
    try:
        from .descriptor import robot_descriptor

        document = robot_descriptor(spec.id, kind="sim")
    except (UnknownRobot, ValueError, TypeError, KeyError):
        return ()
    if not isinstance(document, dict):
        return ()
    return _name_tuple(document.get("joints"))


def _name_tuple(raw: object) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    names = tuple(name for name in raw if isinstance(name, str) and name)
    if len(names) != len(raw):
        return ()
    return names


def _humanoid_block(telemetry: Any) -> Mapping[str, Any] | None:
    custom = getattr(telemetry, "custom", None)
    if not isinstance(custom, dict):
        return None
    block = custom.get("humanoid")
    if isinstance(block, (str, bytes)) or not isinstance(block, Mapping):
        return None
    return block


def _num(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return number


__all__ = [
    "STREAM_HZ",
    "ModeDecision",
    "build_stream_frame",
    "decide_mode_request",
    "declared_posture_modes",
    "mode_confirmation_required",
    "ordered_joint_state",
    "pose_from_sources",
    "posture_label",
    "stream_joint_names",
]
