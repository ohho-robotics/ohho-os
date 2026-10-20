"""Humanoid safety policy (OHH-328).

Selected for ``robot_class`` humanoid. A plugin may only tighten these caps.
Damping a standing humanoid drops it, so a stale command and an upright
e-stop never damp. Low-level joints stay in ``CUSTOM`` on a sim target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable

from .humanoid import (
    FallState,
    HumanoidMode,
    fall_state_of,
    is_upright,
    mode_of,
)
from .registry import RobotSpec
from .robot_spec import HumanoidLimits
from .safety_policy import ClampDecision, EstopAction, SafetyPolicy, _accel_step
from .schema import Telemetry, Velocity, clamp

SIM_MAX_VX_FORWARD = 0.5
SIM_MAX_VX_BACK = 0.3
SIM_MAX_VY = 0.3
SIM_MAX_WZ = 0.6
HARDWARE_MAX_VX_FORWARD = 0.3
HARDWARE_MAX_VX_BACK = 0.2
HARDWARE_MAX_VY = 0.2
HARDWARE_MAX_WZ = 0.5
MAX_JOINT_STEP_RAD = 0.15
TILT_LIMIT_RAD = 0.5
TILT_HOLD_S = 0.1
CMD_TIMEOUT_S = 0.3
_CLASS_ACCEL = 5.0

HUMANOID_ALLOWLIST = frozenset(
    {
        "move",
        "stop_move",
        "prepare",
        "walk_mode",
        "damp",
        "custom",
        "get_up",
        "lie_down",
        "look",
        "move_ee",
        "gripper",
        "hand",
        "move_joints",
    }
)
HUMANOID_MCP_TOOLS = frozenset(
    {
        "move",
        "stop_move",
        "prepare",
        "walk_mode",
        "get_up",
        "lie_down",
        "look",
        "move_ee",
        "gripper",
        "hand",
    }
)
_MCP_REFUSED = frozenset({"move_joints", "damp", "custom", "arm", "release_stop"})
_DENIED_EXTRAS = (
    "kick",
    "shoot",
    "handshake",
    "zero_torque",
    "trajectory",
    "trained_traj",
)
_HOLD_MODES = frozenset({HumanoidMode.PREPARE, HumanoidMode.WALK})


class _BadOrientation:
    """Marker for a present orientation that cannot be read as two angles."""


_BAD_ORIENTATION = _BadOrientation()
_CAP_FIELDS = (
    "max_vx_forward",
    "max_vx_back",
    "max_vy",
    "max_wz",
    "max_joint_step",
    "tilt_rad",
    "tilt_hold_s",
    "cmd_timeout_s",
)
_GAIN_FIELDS = ("max_kp", "max_kd", "max_tau")


def _safety() -> Any:
    from . import safety

    return safety


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0")


@dataclass(frozen=True)
class HumanoidCaps:
    """Asymmetric humanoid caps. Larger numbers are looser.

    ``max_kp``, ``max_kd`` and ``max_tau`` are ``None`` until a limit is
    declared. ``None`` refuses an explicit gain or torque rather than
    leaving it open.
    """

    max_vx_forward: float = SIM_MAX_VX_FORWARD
    max_vx_back: float = SIM_MAX_VX_BACK
    max_vy: float = SIM_MAX_VY
    max_wz: float = SIM_MAX_WZ
    max_joint_step: float = MAX_JOINT_STEP_RAD
    tilt_rad: float = TILT_LIMIT_RAD
    tilt_hold_s: float = TILT_HOLD_S
    cmd_timeout_s: float = CMD_TIMEOUT_S
    max_kp: float | None = None
    max_kd: float | None = None
    max_tau: float | None = None

    def __post_init__(self) -> None:
        for name in _CAP_FIELDS:
            _positive(name, float(getattr(self, name)))
        for name in _GAIN_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _positive(name, float(value))


SIM_CAPS = HumanoidCaps()
HARDWARE_CAPS = HumanoidCaps(
    max_vx_forward=HARDWARE_MAX_VX_FORWARD,
    max_vx_back=HARDWARE_MAX_VX_BACK,
    max_vy=HARDWARE_MAX_VY,
    max_wz=HARDWARE_MAX_WZ,
)


def class_profile(caps: HumanoidCaps) -> Any:
    """Safety profile whose symmetric fields are the sim-class ceiling.

    Forward speed is stored in ``max_vx``. The back cap stays on
    :class:`HumanoidCaps`. Timeouts and tilt stay inside ``SafetyProfile``'s
    own upper bounds.
    """
    safety = _safety()
    return safety.SafetyProfile(
        max_vx=min(caps.max_vx_forward, SIM_MAX_VX_FORWARD),
        max_vy=min(caps.max_vy, SIM_MAX_VY),
        max_wz=min(caps.max_wz, SIM_MAX_WZ),
        max_accel_x=_CLASS_ACCEL,
        max_accel_y=_CLASS_ACCEL,
        max_accel_wz=_CLASS_ACCEL,
        cmd_timeout_s=min(caps.cmd_timeout_s, CMD_TIMEOUT_S),
        tilt_limit_rad=min(caps.tilt_rad, TILT_LIMIT_RAD),
        max_joint_step=min(caps.max_joint_step, MAX_JOINT_STEP_RAD),
        allowlist=HUMANOID_ALLOWLIST,
    )


def _opt_min(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def intersect_caps(left: HumanoidCaps, right: HumanoidCaps) -> HumanoidCaps:
    """The tighter of two cap sets. A gain ceiling survives if either side has one."""
    return HumanoidCaps(
        max_vx_forward=min(left.max_vx_forward, right.max_vx_forward),
        max_vx_back=min(left.max_vx_back, right.max_vx_back),
        max_vy=min(left.max_vy, right.max_vy),
        max_wz=min(left.max_wz, right.max_wz),
        max_joint_step=min(left.max_joint_step, right.max_joint_step),
        tilt_rad=min(left.tilt_rad, right.tilt_rad),
        tilt_hold_s=min(left.tilt_hold_s, right.tilt_hold_s),
        cmd_timeout_s=min(left.cmd_timeout_s, right.cmd_timeout_s),
        max_kp=_opt_min(left.max_kp, right.max_kp),
        max_kd=_opt_min(left.max_kd, right.max_kd),
        max_tau=_opt_min(left.max_tau, right.max_tau),
    )


def cap_looser_reasons(plugin: HumanoidCaps, class_caps: HumanoidCaps) -> list[str]:
    """Ways ``plugin`` is looser than ``class_caps``."""
    reasons: list[str] = []
    for name in _CAP_FIELDS:
        plugin_value = float(getattr(plugin, name))
        class_value = float(getattr(class_caps, name))
        if plugin_value > class_value:
            reasons.append(f"{name} {plugin_value} exceeds class cap {class_value}")
    for name in _GAIN_FIELDS:
        plugin_value = getattr(plugin, name)
        class_value = getattr(class_caps, name)
        if plugin_value is None:
            continue
        if class_value is None:
            reasons.append(f"{name} enables a gain the class policy refuses")
        elif float(plugin_value) > float(class_value):
            reasons.append(f"{name} {plugin_value} exceeds class cap {class_value}")
    return reasons


def caps_from_spec(spec: RobotSpec | None) -> HumanoidCaps:
    """Sim caps, lowered when the spec declares a smaller step or a gain ceiling.

    Descriptive velocity fields on the spec are not safety caps.
    """
    caps = SIM_CAPS
    limits = getattr(spec, "limits", None) if spec is not None else None
    if not isinstance(limits, HumanoidLimits):
        return caps
    step = limits.max_joint_step
    if isinstance(step, (int, float)) and not isinstance(step, bool) and step > 0:
        caps = replace(caps, max_joint_step=min(caps.max_joint_step, float(step)))
    for attr, field_name in (("max_kp", "max_kp"), ("max_tau", "max_tau")):
        raw = getattr(limits, attr)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            caps = replace(caps, **{field_name: float(raw)})
    return caps


def explicit_sim_target(uri: str | None) -> bool:
    """True for ``sim://``, ``simulated://``, and loopback humanoid-mujoco.

    The mujoco flag must be the exact token ``sim_target=True``. Any other
    host still needs a human at a TTY.
    """
    if not uri:
        return False
    if uri.startswith("sim://") or uri.startswith("simulated://"):
        return True
    prefix = "humanoid-mujoco://"
    if not uri.startswith(prefix):
        return False
    rest = uri[len(prefix) :]
    host, _sep, query = rest.partition("?")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not _safety()._is_loopback(host):
        return False
    return "sim_target=True" in query.split("&")


def humanoid_denied(action: str) -> bool:
    """True for the shared denylist plus humanoid-only gestures and trajectories."""
    if _safety().is_denied(action):
        return True
    return any(fragment in action for fragment in _DENIED_EXTRAS)


def humanoid_mcp_allowed(action: object) -> bool:
    """High-level humanoid tools a remote caller may name.

    ``move_joints``, ``damp``, ``custom``, ``arm`` and ``release_stop`` are
    operator paths. This function does not touch a transport.
    """
    if not isinstance(action, str) or not action.strip():
        return False
    name = _safety().normalize_action(action)
    if humanoid_denied(name) or name in _MCP_REFUSED:
        return False
    return name in HUMANOID_MCP_TOOLS


def mcp_command(action: object, *args: object, mode: object = None) -> ClampDecision:
    """Refuse a remote humanoid tool, or echo an allowlisted action name.

    ``args`` and ``mode`` are ignored. The call never reaches a transport.
    """
    del args, mode
    if not humanoid_mcp_allowed(action):
        return ClampDecision(reject="mcp_refused")
    return ClampDecision(applied=(_safety().normalize_action(str(action)),))


class HumanoidPolicy(SafetyPolicy):
    """Generic humanoid rules. Vendor plugins may only tighten ``caps``."""

    name = "humanoid"

    def __init__(
        self,
        profile: Any,
        *,
        spec: RobotSpec | None = None,
        caps: HumanoidCaps | None = None,
    ) -> None:
        super().__init__(profile)
        self.spec = spec
        self.caps = SIM_CAPS if caps is None else caps
        source = getattr(spec, "source", "") if spec is not None else ""
        self.plugin_name = str(source or "")
        self.audit_fn: Callable[..., Any] | None = None
        self._sim_target = False
        self._damped = False
        self._last_mode: HumanoidMode | None = None
        self._tilt_since: float | None = None
        self._joint_cmds: dict[str, float] = {}

    @classmethod
    def for_spec(cls, spec: RobotSpec | None) -> HumanoidPolicy:
        """Class policy for ``spec``. ``None`` uses the sim ceiling."""
        caps = caps_from_spec(spec)
        return cls(class_profile(caps), spec=spec, caps=caps)

    def denied(self, action: str) -> bool:
        return humanoid_denied(action)

    def looser_than(self, class_policy: SafetyPolicy) -> list[str]:
        if not isinstance(class_policy, HumanoidPolicy):
            return []
        return cap_looser_reasons(self.caps, class_policy.caps)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        """True when a human must confirm. Sim targets arm without a phrase.

        ``sim://`` is judged as ``simulated://``, which is the existing
        simulator check. Loopback humanoid-mujoco arms only when the URI
        carries ``sim_target=True``.
        """
        target = explicit_sim_target(uri)
        mapped = uri
        if isinstance(uri, str) and uri.startswith("sim://"):
            mapped = "simulated://" + uri[len("sim://") :]
        if target and isinstance(uri, str) and uri.startswith("humanoid-mujoco://"):
            self._sim_target = True
            return False
        needed = super().arming(mapped, env)
        self._sim_target = bool(target and not needed)
        return needed

    def active_caps(self) -> HumanoidCaps:
        """Sim caps on a sim target, hardware caps otherwise, then any tighter bound."""
        runtime = SIM_CAPS if self._sim_target else HARDWARE_CAPS
        merged = intersect_caps(runtime, self.caps)
        profile = self.profile
        return replace(
            merged,
            max_vx_forward=min(merged.max_vx_forward, profile.max_vx),
            max_vx_back=min(merged.max_vx_back, profile.max_vx),
            max_vy=min(merged.max_vy, profile.max_vy),
            max_wz=min(merged.max_wz, profile.max_wz),
            max_joint_step=min(
                merged.max_joint_step, profile.max_joint_step, MAX_JOINT_STEP_RAD
            ),
            tilt_rad=min(merged.tilt_rad, profile.tilt_limit_rad),
            cmd_timeout_s=min(merged.cmd_timeout_s, profile.cmd_timeout_s),
        )

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        """Zero velocity. Stay in the current mode. Never damp."""
        del telemetry
        try:
            transport.send_velocity(Velocity())
        except Exception as exc:
            _safety_log().error("safety: zero-velocity send failed: %r", exc)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        """Zero velocity. Hold PREPARE while upright and standing. Damp only when down.

        Never calls the transport e-stop, never requests zero torque, and
        never drags. ``estop_damp`` does not damp a standing humanoid.
        """
        inner_error = False
        try:
            transport.send_velocity(Velocity())
        except Exception as exc:
            inner_error = True
            _safety_log().error("safety: zero-velocity send failed: %r", exc)
        mode = mode_of(telemetry)
        fall = fall_state_of(telemetry) if telemetry is not None else FallState.UNKNOWN
        if mode is HumanoidMode.CUSTOM:
            why = "custom"
        elif mode is HumanoidMode.FALLEN or fall is FallState.FALLEN:
            why = "fallen"
        elif is_upright(telemetry) and mode in _HOLD_MODES:
            self._send(transport, "prepare")
            return EstopAction(inner_error=inner_error, damp_why="")
        else:
            return EstopAction(inner_error=inner_error, damp_why="")
        if self._send(transport, "damp"):
            self._damped = True
            return EstopAction(inner_error=inner_error, damp_why=why)
        return EstopAction(inner_error=inner_error, damp_why="")

    def note_state(
        self,
        transport: Any,
        telemetry: Telemetry | None,
        *,
        latched: bool,
    ) -> None:
        """Record a mode change. Damp once if a latched robot is down."""
        if not latched:
            self._damped = False
        mode = mode_of(telemetry) if telemetry is not None else None
        if (
            mode is not None
            and self._last_mode is not None
            and mode is not self._last_mode
            and self.audit_fn is not None
        ):
            self.audit_fn(
                "mode_transition",
                mode_from=self._last_mode.value,
                mode_to=mode.value,
                plugin=self.plugin_name,
            )
        if mode is not None:
            self._last_mode = mode
        if not latched or self._damped or telemetry is None or mode is None:
            return
        fall = fall_state_of(telemetry)
        if fall is not FallState.FALLEN and mode is not HumanoidMode.FALLEN:
            return
        if self._send(transport, "damp"):
            self._damped = True
            if self.audit_fn is not None:
                self.audit_fn(
                    "damp",
                    why="fallen",
                    plugin=self.plugin_name,
                    mode=mode.value,
                )

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        """Latch on a fault, a non-upright fall state, or a held tilt.

        Unknown and missing fall states are not upright. Roll or pitch past
        the tilt cap must persist for ``tilt_hold_s`` on the sample timestamp.
        The first sample over the cap starts that window and does not latch.
        A non-finite orientation latches immediately. A missing timestamp does
        not latch on duration.
        """
        custom = telemetry.custom
        if "error_code" in custom:
            try:
                code = int(custom["error_code"])
            except (TypeError, ValueError):
                return "fault", self._details(
                    telemetry, error_code=repr(custom["error_code"])
                )
            if code != 0:
                return "fault", self._details(telemetry, error_code=code)
        fall = fall_state_of(telemetry)
        if fall is not FallState.UPRIGHT:
            return "fall", self._details(telemetry, fall_state=fall.value)
        orientation = _rpy(telemetry)
        if isinstance(orientation, _BadOrientation):
            return "tilt", self._details(telemetry, orientation="unreadable")
        if orientation is None:
            return None
        roll, pitch = orientation
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            return "tilt", self._details(telemetry, roll=roll, pitch=pitch)
        limit = self.active_caps().tilt_rad
        if abs(roll) <= limit and abs(pitch) <= limit:
            self._tilt_since = None
            return None
        started = _sample_time(telemetry)
        if started is None:
            return None
        if self._tilt_since is None:
            self._tilt_since = started
            return None
        if started - self._tilt_since >= self.active_caps().tilt_hold_s:
            return "tilt", self._details(telemetry, roll=roll, pitch=pitch)
        return None

    def low_level_joints(
        self,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        """Clamp one joint group. Refused outside CUSTOM or off a sim target."""
        if "move_joints" not in self.allowlist:
            return ClampDecision(reject="not_allowlisted")
        if not self._sim_target:
            return ClampDecision(reject="not_simulated")
        if mode_of(telemetry) is not HumanoidMode.CUSTOM:
            return ClampDecision(reject="not_custom")
        parsed = _parse_joint_args(args)
        if isinstance(parsed, str):
            return ClampDecision(reject=parsed)
        group, positions, kp, kd = parsed
        names = self._group_names(group)
        if names is None:
            return ClampDecision(reject="unknown_group")
        if len(positions) != len(names):
            return ClampDecision(reject="bad_joint_shape")
        limits = self.profile.joint_limits
        if any(name not in limits for name in names):
            return ClampDecision(reject="no_joint_limit")
        refs: list[float] = []
        for name in names:
            ref = _joint_position(telemetry, name)
            if ref is None:
                ref = self._joint_cmds.get(name)
            if ref is None:
                return ClampDecision(reject="no_joint_reference")
            refs.append(ref)
        caps = self.active_caps()
        gained = _apply_gains(kp, kd, caps)
        if isinstance(gained, str):
            return ClampDecision(reject=gained)
        applied_kp, applied_kd, gain_reasons = gained
        applied_q: list[float] = []
        reasons: list[str] = []
        step = caps.max_joint_step
        for name, position, reference in zip(names, positions, refs):
            lo, hi = limits[name]
            target = clamp(position, lo, hi)
            applied = clamp(target, reference - step, reference + step)
            if target != position and "joint_limit" not in reasons:
                reasons.append("joint_limit")
            if applied != target and "joint_step" not in reasons:
                reasons.append("joint_step")
            applied_q.append(applied)
        reasons.extend(gain_reasons)
        for name, position in zip(names, applied_q):
            self._joint_cmds[name] = position
        return ClampDecision(
            applied=(group, tuple(applied_q), applied_kp, applied_kd),
            reasons=tuple(reasons),
        )

    def _clamp_velocity(self, args: Sequence[Any]) -> ClampDecision:
        parsed = _parse_velocity(args)
        if parsed is None:
            return ClampDecision(reject="non_finite")
        requested, previous, raw_dt = parsed
        caps = self.active_caps()
        capped = (
            clamp(requested[0], -caps.max_vx_back, caps.max_vx_forward),
            clamp(requested[1], -caps.max_vy, caps.max_vy),
            clamp(requested[2], -caps.max_wz, caps.max_wz),
        )
        dt = min(max(raw_dt, 0.0), self.profile.cmd_timeout_s)
        accels = (
            self.profile.max_accel_x,
            self.profile.max_accel_y,
            self.profile.max_accel_wz,
        )
        applied = tuple(
            _accel_step(prev, target, accel * dt)
            for prev, target, accel in zip(previous, capped, accels)
        )
        reasons: list[str] = []
        if capped != requested:
            reasons.append("velocity_cap")
        if applied != capped:
            reasons.append("accel_cap")
        return ClampDecision(applied=applied, reasons=tuple(reasons))

    def _clamp_effort(self, args: Sequence[Any]) -> ClampDecision:
        if len(args) != 2:
            return ClampDecision(reject="non_finite")
        name, raw = args
        effort = _finite(raw)
        if effort is None or not isinstance(name, str):
            return ClampDecision(reject="non_finite")
        ceilings: list[float] = []
        if self.caps.max_tau is not None:
            ceilings.append(self.caps.max_tau)
        limit = self.profile.effort_limits.get(name)
        if limit is not None:
            ceilings.append(float(limit))
        if not ceilings:
            return ClampDecision(reject="no_effort_limit")
        cap = min(ceilings)
        applied = clamp(effort, -cap, cap)
        reasons = ("effort_cap",) if applied != effort else ()
        return ClampDecision(applied=(applied,), reasons=reasons)

    def _group_names(self, group: str) -> tuple[str, ...] | None:
        groups = getattr(self.spec, "joint_groups", None) or {}
        names = groups.get(group) if isinstance(groups, Mapping) else None
        if not names:
            return None
        return tuple(str(name) for name in names)

    def _details(self, telemetry: Telemetry, **extra: object) -> dict[str, Any]:
        details: dict[str, Any] = {
            "mode": mode_of(telemetry).value,
            "fall_state": fall_state_of(telemetry).value,
            "plugin": self.plugin_name,
        }
        details.update(extra)
        return details

    def _send(self, transport: Any, action: str) -> bool:
        if action not in ("prepare", "damp"):
            return False
        send = getattr(transport, "send_humanoid", None)
        if not callable(send):
            _safety_log().error("safety: humanoid %s unsupported", action)
            return False
        try:
            send(action)
        except Exception as exc:
            _safety_log().error("safety: humanoid %s failed: %r", action, exc)
            return False
        return True


def _safety_log() -> Any:
    return _safety().log


def _sample_time(telemetry: Telemetry) -> float | None:
    return _finite(telemetry.timestamp)


def _rpy(telemetry: Telemetry) -> tuple[float, float] | None | _BadOrientation:
    block = telemetry.custom.get("humanoid")
    if isinstance(block, Mapping) and "base_rpy" in block:
        rpy = block.get("base_rpy")
        if isinstance(rpy, (list, tuple)) and len(rpy) >= 2:
            return float(_as_float(rpy[0])), float(_as_float(rpy[1]))
        return _BAD_ORIENTATION
    orientation = _safety()._orientation(telemetry)
    if orientation is None:
        return None
    return orientation


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _parse_velocity(
    args: Sequence[Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float], float] | None:
    if len(args) != 3:
        return None
    requested, previous, raw_dt = args
    try:
        req = tuple(float(value) for value in requested)
        prev = tuple(float(value) for value in previous)
        dt = float(raw_dt)
    except (TypeError, ValueError):
        return None
    if len(req) != 3 or len(prev) != 3:
        return None
    if not all(math.isfinite(value) for value in (*req, *prev, dt)):
        return None
    return (
        (req[0], req[1], req[2]),
        (prev[0], prev[1], prev[2]),
        dt,
    )


def _parse_joint_args(
    args: Sequence[Any],
) -> tuple[str, tuple[float, ...], float | None, float | None] | str:
    if len(args) < 2:
        return "bad_arguments"
    group = args[0]
    raw = args[1]
    if not isinstance(group, str) or not group.strip():
        return "bad_arguments"
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return "bad_arguments"
    positions: list[float] = []
    for item in raw:
        number = _finite(item)
        if number is None:
            return "bad_arguments"
        positions.append(number)
    kp = None if len(args) < 3 or args[2] is None else _finite(args[2])
    kd = None if len(args) < 4 or args[3] is None else _finite(args[3])
    if len(args) >= 3 and args[2] is not None and kp is None:
        return "bad_gain"
    if len(args) >= 4 and args[3] is not None and kd is None:
        return "bad_gain"
    return group.strip(), tuple(positions), kp, kd


def _joint_position(telemetry: Telemetry | None, name: str) -> float | None:
    if telemetry is None:
        return None
    for reading in telemetry.joints:
        if reading.name == name:
            return _finite(reading.position)
    return None


def _apply_gains(
    kp: float | None,
    kd: float | None,
    caps: HumanoidCaps,
) -> tuple[float | None, float | None, tuple[str, ...]] | str:
    applied_kp, kp_reasons, kp_error = _one_gain(
        kp, caps.max_kp, "kp_cap", "no_kp_limit"
    )
    if kp_error:
        return kp_error
    applied_kd, kd_reasons, kd_error = _one_gain(
        kd, caps.max_kd, "kd_cap", "no_kd_limit"
    )
    if kd_error:
        return kd_error
    return applied_kp, applied_kd, kp_reasons + kd_reasons


def _one_gain(
    value: float | None,
    ceiling: float | None,
    reason: str,
    missing: str,
) -> tuple[float | None, tuple[str, ...], str]:
    if value is None:
        return None, (), ""
    if value < 0:
        return None, (), "bad_gain"
    if ceiling is None:
        return None, (), missing
    if value > ceiling:
        return ceiling, (reason,), ""
    return value, (), ""


__all__ = [
    "HARDWARE_CAPS",
    "HUMANOID_ALLOWLIST",
    "HUMANOID_MCP_TOOLS",
    "MAX_JOINT_STEP_RAD",
    "SIM_CAPS",
    "TILT_HOLD_S",
    "TILT_LIMIT_RAD",
    "HumanoidCaps",
    "HumanoidPolicy",
    "cap_looser_reasons",
    "caps_from_spec",
    "class_profile",
    "explicit_sim_target",
    "humanoid_denied",
    "humanoid_mcp_allowed",
    "intersect_caps",
    "mcp_command",
]
