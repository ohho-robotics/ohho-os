"""Plugin configuration for the generic ``humanoid-mujoco`` sim backend.

The MJCF path, joint order, PD gains, observation layout and checkpoint are
data. Core does not name a robot. A plugin binds a :class:`HumanoidSimConfig`
for its spec id, or puts the MJCF on ``spec.assets.mjcf`` and the joint order
on ``spec.joint_groups``.

Observation terms, in the default order, are base angular velocity, projected
gravity, the velocity command, joint position, joint velocity and the previous
action. A plugin may reorder those terms or scale them. The policy runner does
not interpret them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, Union

from .humanoid import FallState

HUMANOID_MUJOCO_BACKEND = "humanoid-mujoco"
DEFAULT_POLICY_RATE_HZ = 50.0

# Starting values for telemetry only. They are not a safety latch.
UPRIGHT_TILT_RAD = 0.5
FALLEN_TILT_RAD = 1.0

OBS_TERM_NAMES: tuple[str, ...] = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "q",
    "dq",
    "last_action",
)
_VECTOR_TERMS = frozenset({"base_ang_vel", "projected_gravity", "command"})
_JOINT_TERMS = frozenset({"q", "dq", "last_action"})

Number = Union[float, Sequence[float]]
_HISTORY_INIT = frozenset({"zeros", "repeat"})


class ObservationExtra(Protocol):
    """Plugin state appended after the named observation terms.

    ``reset`` runs when locomotion starts. ``extend`` returns the extra
    values for one frame. The runner does not interpret them.
    """

    def reset(self) -> None:
        """Drop state so the next frame is a fresh start."""

    def extend(
        self,
        *,
        yaw: float,
        command: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, ...]:
        """Extra observation values for this policy step.

        ``yaw`` is the base yaw in radians. ``command`` is ``(vx, vy, yaw
        rate)``. ``dt`` is the policy period in seconds.
        """


@dataclass(frozen=True)
class ObservationTerm:
    """One block of the policy observation. ``scale`` multiplies every element."""

    name: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in OBS_TERM_NAMES:
            known = ", ".join(OBS_TERM_NAMES)
            raise ValueError(f"unknown observation term {self.name!r}; known: {known}")
        _finite("observation scale", self.scale)


def default_observation_terms() -> tuple[ObservationTerm, ...]:
    """The six terms in the default order, each with scale 1."""
    return tuple(ObservationTerm(name) for name in OBS_TERM_NAMES)


@dataclass(frozen=True)
class PolicyLayout:
    """Obs/action layout and checkpoint. All of it comes from the plugin.

    ``joint_names`` is the order of ``q``, ``dq`` and the action vector. It may
    differ from the PD joint order. ``action_scale`` maps the network output
    onto a joint-position offset: ``q_target = default_q + action * scale``.
    A single number applies to every policy joint. A sequence has one scale
    per policy joint. ``q_relative`` subtracts ``default_q`` from the position
    term. ``rate_hz`` is how often the network runs; the default is 50.

    ``history_length`` stacks that many frames, oldest first, before the
    network call. ``history_init`` ``zeros`` fills every frame but the latest
    with zeros on the first call. ``repeat`` copies the first frame. Length 1
    leaves the observation as one frame.     ``clip_observation`` and
    ``clip_action`` limit each element when set. ``extra_factory`` builds the
    object whose values are appended to each frame. An empty ``checkpoint``
    means no policy is loaded.

    ``actor_module`` selects a TorchScript ``actor`` submodule when the file
    has one. ``action_filter`` in ``(0, 1)`` low-passes each position target;
    ``1`` writes the target immediately. ``action_offset`` is subtracted from
    the network output before ``action_scale``. ``seed_last_action`` ``None``
    keeps the previous rule: seed the last action from the pose when
    ``q_relative`` is set, otherwise start at zero.
    """

    joint_names: tuple[str, ...] = ()
    terms: tuple[ObservationTerm, ...] = field(
        default_factory=default_observation_terms
    )
    action_scale: Number = 1.0
    rate_hz: float = DEFAULT_POLICY_RATE_HZ
    checkpoint: str = ""
    q_relative: bool = False
    history_length: int = 1
    history_init: str = "zeros"
    clip_observation: float | None = None
    clip_action: float | None = None
    extra_factory: Callable[[], ObservationExtra] | None = None
    actor_module: bool = False
    action_filter: float = 1.0
    action_offset: tuple[float, ...] = ()
    seed_last_action: bool | None = None

    def __post_init__(self) -> None:
        names = _names(self.joint_names, "policy joint_names")
        object.__setattr__(self, "joint_names", names)
        if not self.terms:
            raise ValueError("policy observation terms must not be empty")
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(
            self, "action_scale", _action_scale(self.action_scale, len(names))
        )
        if self.rate_hz <= 0 or not math.isfinite(self.rate_hz):
            raise ValueError("policy rate_hz must be a finite number > 0")
        if not isinstance(self.checkpoint, str):
            raise TypeError("checkpoint must be a string")
        if not isinstance(self.q_relative, bool):
            raise TypeError("q_relative must be a bool")
        if isinstance(self.history_length, bool) or not isinstance(
            self.history_length, int
        ):
            raise TypeError("history_length must be an int")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.history_init not in _HISTORY_INIT:
            raise ValueError("history_init must be 'zeros' or 'repeat'")
        object.__setattr__(
            self, "clip_observation", _optional_clip(self.clip_observation)
        )
        object.__setattr__(self, "clip_action", _optional_clip(self.clip_action))
        if self.extra_factory is not None and not callable(self.extra_factory):
            raise TypeError("extra_factory must be callable or None")
        if not isinstance(self.actor_module, bool):
            raise TypeError("actor_module must be a bool")
        if isinstance(self.action_filter, bool) or not isinstance(
            self.action_filter, (int, float)
        ):
            raise TypeError("action_filter must be a number")
        action_filter = float(self.action_filter)
        if not math.isfinite(action_filter) or action_filter <= 0 or action_filter > 1:
            raise ValueError("action_filter must be in (0, 1]")
        object.__setattr__(self, "action_filter", action_filter)
        if isinstance(self.action_offset, (str, bytes)) or not isinstance(
            self.action_offset, Sequence
        ):
            raise TypeError("action_offset must be a sequence of numbers")
        offset = tuple(_finite("action_offset", item) for item in self.action_offset)
        if offset and len(offset) != len(names):
            raise ValueError(
                f"action_offset must have {len(names)} values, got {len(offset)}"
            )
        object.__setattr__(self, "action_offset", offset)
        if self.seed_last_action is not None and not isinstance(
            self.seed_last_action, bool
        ):
            raise TypeError("seed_last_action must be a bool or None")


@dataclass(frozen=True)
class HumanoidSimConfig:
    """Everything the backend needs from a plugin, and nothing robot-specific.

    ``joint_names`` is the PD order. ``kp``, ``kd`` and ``default_q`` follow
    that order. A single number is applied to every joint. ``default_q`` is
    the pose ``prepare`` holds. ``policy`` is optional: ``prepare``, ``damp``
    and ``custom`` run with it left as ``None``. ``timestep`` replaces the
    MJCF timestep when set. ``base_pos`` is the floating-base ``(x, y, z)``
    written on connect, with the joints placed at ``default_q``.
    ``effort_limit`` clips each PD torque when set. ``None`` leaves the
    torque unclipped apart from the MJCF actuator range.
    """

    mjcf: str
    joint_names: tuple[str, ...] = ()
    kp: Number = 20.0
    kd: Number = 1.0
    default_q: Number = 0.0
    policy: PolicyLayout | None = None
    base_body: str = ""
    timestep: float | None = None
    base_pos: tuple[float, float, float] | None = None
    effort_limit: Number | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mjcf, str) or not self.mjcf.strip():
            raise ValueError("mjcf path must be a non-empty string")
        object.__setattr__(self, "mjcf", self.mjcf.strip())
        names = _names(self.joint_names, "joint_names")
        object.__setattr__(self, "joint_names", names)
        n = len(names)
        object.__setattr__(self, "kp", _broadcast(self.kp, n, "kp"))
        object.__setattr__(self, "kd", _broadcast(self.kd, n, "kd"))
        object.__setattr__(
            self, "default_q", _broadcast(self.default_q, n, "default_q")
        )
        if self.policy is not None and not isinstance(self.policy, PolicyLayout):
            raise TypeError("policy must be a PolicyLayout or None")
        if not isinstance(self.base_body, str):
            raise TypeError("base_body must be a string")
        if self.timestep is not None:
            step = _finite("timestep", self.timestep)
            if step <= 0:
                raise ValueError("timestep must be > 0")
            object.__setattr__(self, "timestep", step)
        object.__setattr__(self, "base_pos", _optional_vec3(self.base_pos))
        object.__setattr__(self, "effort_limit", _effort_limit(self.effort_limit, n))


_BOUND: dict[str, HumanoidSimConfig] = {}


def bind_sim_config(robot_id: str, config: HumanoidSimConfig) -> None:
    """Remember ``config`` for ``robot_id``. A later bind replaces it.

    Plugins call this while they load. The backend reads it when a spec does
    not pass a config into the transport directly.
    """
    if not isinstance(robot_id, str) or not robot_id.strip():
        raise ValueError("robot_id must be a non-empty string")
    if not isinstance(config, HumanoidSimConfig):
        raise TypeError("config must be a HumanoidSimConfig")
    _BOUND[robot_id] = config


def clear_sim_config(robot_id: str | None = None) -> None:
    """Drop one bound config, or every config when ``robot_id`` is omitted."""
    if robot_id is None:
        _BOUND.clear()
        return
    _BOUND.pop(robot_id, None)


def bound_sim_config(robot_id: str) -> HumanoidSimConfig | None:
    """The config bound for ``robot_id``, or ``None``."""
    return _BOUND.get(robot_id)


def resolve_sim_config(spec: object, address: str = "") -> HumanoidSimConfig:
    """Config for ``spec``: a bind, then assets and joint groups.

    An ``address`` that ends in ``.xml`` replaces the MJCF path. Any other
    address is a sim label and is ignored.
    """
    robot_id = str(getattr(spec, "id", "") or "")
    override = address.strip() if address.strip().lower().endswith(".xml") else ""
    bound = _BOUND.get(robot_id)
    if bound is not None:
        if override and override != bound.mjcf:
            return replace(bound, mjcf=override)
        return bound
    mjcf = override or _asset_mjcf(spec)
    if not mjcf:
        raise ValueError(
            "humanoid-mujoco needs an MJCF path on the plugin config or spec assets"
        )
    names = _spec_joint_names(spec)
    return HumanoidSimConfig(
        mjcf=mjcf, joint_names=names, kp=20.0, kd=1.0, default_q=0.0
    )


def policy_stride(timestep: float, rate_hz: float) -> int:
    """Model steps between policy evaluations.

    The count is ``round((1 / rate_hz) / timestep)``, and at least one. PD
    still runs on every model step.
    """
    if timestep <= 0 or not math.isfinite(timestep):
        raise ValueError("timestep must be a finite number > 0")
    if rate_hz <= 0 or not math.isfinite(rate_hz):
        raise ValueError("rate_hz must be a finite number > 0")
    return max(1, int(round((1.0 / rate_hz) / timestep)))


def fall_state_from_tilt(roll: float, pitch: float) -> FallState:
    """Classify base tilt. ``unknown`` is not produced here.

    ``max(|roll|, |pitch|) <= 0.5`` rad is upright. Above that, and below
    1 rad, is falling. At 1 rad or more the base is fallen.
    """
    tilt = max(abs(_finite("roll", roll)), abs(_finite("pitch", pitch)))
    if tilt >= FALLEN_TILT_RAD:
        return FallState.FALLEN
    if tilt > UPRIGHT_TILT_RAD:
        return FallState.FALLING
    return FallState.UPRIGHT


def projected_gravity(xmat: Sequence[float]) -> tuple[float, float, float]:
    """Gravity ``(0, 0, -1)`` expressed in the body frame.

    ``xmat`` is the 9-element row-major body-to-world rotation. An identity
    rotation yields ``(0, 0, -1)``.
    """
    if len(xmat) != 9:
        raise ValueError("xmat must have 9 elements")
    return (-float(xmat[6]), -float(xmat[7]), -float(xmat[8]))


def quat_wxyz_to_rpy(
    w: float, x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Roll, pitch, yaw in radians from a ``w, x, y, z`` quaternion."""
    _finite("quat w", w)
    _finite("quat x", x)
    _finite("quat y", y)
    _finite("quat z", z)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return (roll, pitch, yaw)


def rpy_to_quat_wxyz(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """``w, x, y, z`` quaternion from roll, pitch, yaw in radians."""
    half_roll = _finite("roll", roll) * 0.5
    half_pitch = _finite("pitch", pitch) * 0.5
    half_yaw = _finite("yaw", yaw) * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def build_observation(
    *,
    base_ang_vel: Sequence[float],
    projected_gravity: Sequence[float],
    command: Sequence[float],
    q: Sequence[float],
    dq: Sequence[float],
    last_action: Sequence[float],
    terms: Sequence[ObservationTerm],
) -> tuple[float, ...]:
    """Flatten the plugin's terms. Each term is scaled elementwise."""
    blocks = {
        "base_ang_vel": _vec3(base_ang_vel, "base_ang_vel"),
        "projected_gravity": _vec3(projected_gravity, "projected_gravity"),
        "command": _vec3(command, "command"),
        "q": tuple(float(item) for item in q),
        "dq": tuple(float(item) for item in dq),
        "last_action": tuple(float(item) for item in last_action),
    }
    out: list[float] = []
    for term in terms:
        values = blocks[term.name]
        out.extend(value * term.scale for value in values)
    return tuple(out)


def observation_size(n_joints: int, terms: Sequence[ObservationTerm]) -> int:
    """Length of :func:`build_observation` for ``n_joints`` and ``terms``."""
    if n_joints < 0:
        raise ValueError("n_joints must be >= 0")
    total = 0
    for term in terms:
        total += n_joints if term.name in _JOINT_TERMS else 3
    return total


def clip_vector(values: Sequence[float], limit: float | None) -> tuple[float, ...]:
    """Copy ``values``. When ``limit`` is set, each element is kept inside it."""
    if limit is not None and (limit < 0 or not math.isfinite(limit)):
        raise ValueError("clip limit must be >= 0")
    clipped: list[float] = []
    for item in values:
        number = float(item)
        if limit is not None:
            number = max(-limit, min(limit, number))
        clipped.append(number)
    return tuple(clipped)


def append_observation_frame(
    history: tuple[tuple[float, ...], ...] | None,
    frame: Sequence[float],
    *,
    length: int,
    init: str,
) -> tuple[tuple[float, ...], ...]:
    """Push one observation frame. The oldest frame is first.

    ``history`` is ``None`` on the first call. ``zeros`` starts with empty
    frames and puts ``frame`` last. ``repeat`` copies ``frame`` into every
    slot. Later calls drop the oldest frame.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("history length must be an int >= 1")
    if init not in _HISTORY_INIT:
        raise ValueError("history init must be 'zeros' or 'repeat'")
    current = tuple(float(item) for item in frame)
    if history is None:
        if init == "repeat" or length == 1:
            return tuple(current for _ in range(length))
        zero = tuple(0.0 for _ in current)
        return tuple(zero for _ in range(length - 1)) + (current,)
    if len(history) != length or any(len(item) != len(current) for item in history):
        raise ValueError("observation history does not match this frame")
    if length == 1:
        return (current,)
    return history[1:] + (current,)


def flatten_history(history: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """One vector, oldest frame first."""
    return tuple(float(value) for frame in history for value in frame)


def _asset_mjcf(spec: object) -> str:
    assets = getattr(spec, "assets", None)
    raw = getattr(assets, "mjcf", "") if assets is not None else ""
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def _spec_joint_names(spec: object) -> tuple[str, ...]:
    groups = getattr(spec, "joint_groups", None)
    names: list[str] = []
    if isinstance(groups, Mapping):
        for joints in groups.values():
            if isinstance(joints, (str, bytes)) or not isinstance(joints, Sequence):
                continue
            names.extend(str(joint) for joint in joints)
    if names:
        return tuple(names)
    raw = getattr(spec, "joint_names", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    return tuple(str(joint) for joint in raw)


def _names(values: Sequence[str], where: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{where} must be a sequence of joint names")
    names = tuple(str(item).strip() for item in values)
    if any(not name for name in names):
        raise ValueError(f"{where} must not contain an empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"{where} must not repeat a joint")
    return names


def _effort_limit(value: Number | None, n: int) -> tuple[float, ...] | None:
    """Per-joint torque clip. ``None`` means the PD torque is not clipped here."""
    if value is None:
        return None
    limits = _broadcast(value, n, "effort_limit")
    if any(item < 0 for item in limits):
        raise ValueError("effort_limit must be >= 0")
    return limits


def _broadcast(value: Number, n: int, where: str) -> tuple[float, ...]:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        number = _finite(where, float(value))
        return tuple(number for _ in range(n))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where} must be a number or a sequence of numbers")
    if n == 0:
        if len(value) != 0:
            raise ValueError(f"{where} must be empty when no joints are named")
        return ()
    if len(value) != n:
        raise ValueError(f"{where} must have {n} values, got {len(value)}")
    return tuple(_finite(where, item) for item in value)


def _action_scale(value: Number, n: int) -> float | tuple[float, ...]:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return _finite("action_scale", value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("action_scale must be a number or a sequence of numbers")
    if n == 0:
        raise ValueError("per-joint action_scale needs policy joint_names")
    if len(value) != n:
        raise ValueError(f"action_scale must have {n} values, got {len(value)}")
    return tuple(_finite("action_scale", item) for item in value)


def _optional_clip(value: float | None) -> float | None:
    if value is None:
        return None
    limit = _finite("clip", value)
    if limit < 0:
        raise ValueError("clip limit must be >= 0")
    return limit


def _optional_vec3(
    value: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError("base_pos must be x, y, z")
    return _vec3(value, "base_pos")


def _finite(where: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _vec3(values: Sequence[float], where: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{where} must have 3 values")
    return (
        _finite(where, values[0]),
        _finite(where, values[1]),
        _finite(where, values[2]),
    )


__all__ = [
    "DEFAULT_POLICY_RATE_HZ",
    "FALLEN_TILT_RAD",
    "HUMANOID_MUJOCO_BACKEND",
    "OBS_TERM_NAMES",
    "UPRIGHT_TILT_RAD",
    "HumanoidSimConfig",
    "ObservationExtra",
    "ObservationTerm",
    "PolicyLayout",
    "append_observation_frame",
    "bind_sim_config",
    "bound_sim_config",
    "build_observation",
    "clear_sim_config",
    "clip_vector",
    "default_observation_terms",
    "fall_state_from_tilt",
    "flatten_history",
    "observation_size",
    "policy_stride",
    "projected_gravity",
    "quat_wxyz_to_rpy",
    "resolve_sim_config",
    "rpy_to_quat_wxyz",
]
