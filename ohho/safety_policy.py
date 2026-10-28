"""Per-class safety policies for :class:`ohho.safety.SafetyGate` (OHH-313).

The gate keeps arming, the audit log, the e-stop latch and the watchdog
schedule. These classes decide what those mechanisms mean for a robot class:

- ``allowlist`` / ``denylist`` — which high-level actions pass
- ``clamp`` — velocity, joint, effort and euler caps
- ``on_watchdog`` — what a stale command does
- ``on_estop`` — how an e-stop is mapped onto the transport
- ``arming`` — whether this URI needs a human present
- ``fall_or_tilt`` — when incoming state latches the gate

``GroundPolicy`` (mobile base and the ground half of a mobile manipulator),
``ArmPolicy`` and ``LeggedPolicy`` (quadruped) are the rules that already
lived in the gate. Aerial (R-03) and humanoid (H-05) are not added here;
those classes keep today's generic rules until their own policies exist.

A plugin may return a stricter policy from ``safety_policy(spec)``.
``load_policy`` refuses one that loosens a class cap. ``None`` keeps the
class default. Discovery does not call the hook.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar, cast

from .registry import RobotSpec
from .robot_spec import RobotClass
from .schema import Telemetry, Velocity, clamp

log = logging.getLogger("ohho.safety")

_PolicyT = TypeVar("_PolicyT", bound="SafetyPolicy")

_CAP_FIELDS = (
    "max_vx",
    "max_vy",
    "max_wz",
    "max_accel_x",
    "max_accel_y",
    "max_accel_wz",
    "max_joint_step",
    "cmd_timeout_s",
    "state_timeout_s",
    "tilt_limit_rad",
    "euler_limit_rad",
)


class SafetyPolicyError(ValueError):
    """A plugin policy loosens a class cap, or is not a policy at all."""


@dataclass(frozen=True)
class ClampDecision:
    """Result of :meth:`SafetyPolicy.clamp`.

    ``reject`` is a gate refusal reason. Otherwise ``applied`` is what may be
    forwarded, and ``reasons`` names any cap that changed the command.
    """

    applied: tuple[Any, ...] = ()
    reasons: tuple[str, ...] = ()
    reject: str | None = None


@dataclass(frozen=True)
class EstopAction:
    """What :meth:`SafetyPolicy.on_estop` did to the transport.

    ``damp_why`` is empty when damp was not sent. The gate writes the audit.
    """

    inner_error: bool = False
    damp_why: str = ""


def _safety() -> Any:
    """The safety module. Imported lazily so this module can load first."""
    from . import safety

    return safety


def _accel_step(prev: float, target: float, dmax: float) -> float:
    """Braking toward zero is free; speeding up (or reversing) moves at most ``dmax``."""
    base = prev if prev * target > 0 else 0.0
    if abs(target) <= abs(base):
        return target
    return base + math.copysign(min(abs(target) - abs(base), dmax), target)


def _is_lying_down(telemetry: Telemetry | None, profile: Any) -> bool:
    if telemetry is None:
        return False
    height = telemetry.custom.get("body_height")
    if isinstance(height, bool) or not isinstance(height, (int, float)):
        return False
    if not math.isfinite(height):
        return False
    return height < profile.lying_body_height


class SafetyPolicy:
    """Decisions the gate delegates. Subclasses are one robot class each.

    ``profile`` holds the caps. The gate binds the effective profile (after
    the safety config, an explicit profile and the hard ceiling) before use.
    """

    name = "safety"

    def __init__(self, profile: Any) -> None:
        self.profile = profile
        self.from_plugin = False
        self.class_profile: Any = None

    def bind(self: _PolicyT, profile: Any) -> _PolicyT:
        """Return a copy that clamps with ``profile``. Hook overrides stay."""
        other = self.__class__.__new__(self.__class__)
        other.__dict__.update(self.__dict__)
        other.profile = profile
        return cast(_PolicyT, other)

    @property
    def allowlist(self) -> frozenset[str]:
        return self.profile.allowlist

    @property
    def denylist(self) -> frozenset[str]:
        """Action fragments and prefixes that are refused even if allowlisted."""
        safety = _safety()
        return frozenset(safety._DENIED_FRAGMENTS) | frozenset(safety._DENIED_PREFIXES)

    def denied(self, action: str) -> bool:
        """True when ``action`` matches the denylist (already normalised)."""
        return bool(_safety().is_denied(action))

    def clamp(
        self,
        action: str,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        """Cap ``action``. ``args`` depends on the action name.

        ``velocity`` — ``(requested, previous, dt)``
        ``joint`` — ``(name, position, reference, joint_names)``
        ``effort`` — ``(name, effort)``
        ``euler`` — three finite angles
        """
        del telemetry
        if action == "velocity":
            return self._clamp_velocity(args)
        if action == "joint":
            return self._clamp_joint(args)
        if action == "effort":
            return self._clamp_effort(args)
        if action == "euler":
            return self._clamp_euler(args)
        return ClampDecision(reject="unsupported")

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        """No fresh command while moving: zero velocity. Does not latch."""
        del telemetry
        try:
            transport.send_velocity(Velocity())
        except Exception as exc:
            log.error("safety: zero-velocity send failed: %r", exc)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        """Stop first. Damp only when already lying down, or when the profile says so.

        Unknown posture never damps. A missing ``sport_command`` skips damp.
        Transport errors are swallowed; the gate still latches.
        """
        inner_error = False
        try:
            transport.emergency_stop()
        except Exception as exc:
            inner_error = True
            log.error("safety: inner emergency_stop failed: %r", exc)
            try:
                transport.send_velocity(Velocity())
            except Exception as send_exc:
                log.error("safety: zero-velocity send failed: %r", send_exc)
        if self.profile.estop_damp:
            damp_why = "estop_damp"
        elif _is_lying_down(telemetry, self.profile):
            damp_why = "lying_down"
        else:
            damp_why = ""
        sent = ""
        sport = getattr(transport, "sport_command", None)
        if damp_why and callable(sport):
            try:
                sport("damp")
                sent = damp_why
            except Exception as exc:
                log.error("safety: damp failed: %r", exc)
        return EstopAction(inner_error=inner_error, damp_why=sent)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        """True when this URI needs a human (env var, TTY and the arming phrase)."""
        return bool(_safety().requires_hardware_arming(_transport_from_uri(uri), env))

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        """``(reason, details)`` when state should latch, else ``None``.

        A non-zero or unreadable ``error_code`` is a fault. Roll or pitch
        outside ``profile.tilt_limit_rad`` is tilt. Missing orientation is
        not itself a latch; the state watchdog handles that.
        """
        custom = telemetry.custom
        if "error_code" in custom:
            try:
                code = int(custom["error_code"])
            except (TypeError, ValueError):
                return "fault", {"error_code": repr(custom["error_code"])}
            if code != 0:
                return "fault", {"error_code": code}
        # The drone profile turns this off. Ground profiles leave it on, so
        # their 0.6 rad latch is unchanged.
        if getattr(self.profile, "tilt_guard", True):
            orientation = _safety()._orientation(telemetry)
            if orientation is not None:
                limit = self.profile.tilt_limit_rad
                if not (abs(orientation[0]) <= limit and abs(orientation[1]) <= limit):
                    return "tilt", {"roll": orientation[0], "pitch": orientation[1]}
        return None

    def _clamp_velocity(self, args: Sequence[Any]) -> ClampDecision:
        requested, previous, raw_dt = args
        profile = self.profile
        capped = (
            clamp(requested[0], -profile.max_vx, profile.max_vx),
            clamp(requested[1], -profile.max_vy, profile.max_vy),
            clamp(requested[2], -profile.max_wz, profile.max_wz),
        )
        # Cylinder clip for the aerial profile only. Ground max_horizontal is 0.
        horizontal = getattr(profile, "max_horizontal", 0.0)
        if getattr(profile, "aerial", False) and horizontal > 0.0:
            hx, hy = _safety().clip_horizontal(capped[0], capped[1], horizontal)
            capped = (hx, hy, capped[2])
        dt = min(max(float(raw_dt), 0.0), profile.cmd_timeout_s)
        accels = (profile.max_accel_x, profile.max_accel_y, profile.max_accel_wz)
        applied = tuple(
            _accel_step(prev, target, accel * dt)
            for prev, target, accel in zip(previous, capped, accels)
        )
        reasons: list[str] = []
        if capped != tuple(requested):
            reasons.append("velocity_cap")
        if applied != capped:
            reasons.append("accel_cap")
        return ClampDecision(applied=applied, reasons=tuple(reasons))

    def _clamp_joint(self, args: Sequence[Any]) -> ClampDecision:
        name, position, reference, joint_names = args
        if joint_names and name not in joint_names:
            return ClampDecision(reject="unknown_joint")
        limits = self.profile.joint_limits.get(name)
        if limits is None:
            return ClampDecision(reject="no_joint_limit")
        if reference is None:
            return ClampDecision(reject="no_joint_reference")
        target = clamp(position, limits[0], limits[1])
        applied = clamp(
            target,
            reference - self.profile.max_joint_step,
            reference + self.profile.max_joint_step,
        )
        reasons: list[str] = []
        if target != position:
            reasons.append("joint_limit")
        if applied != target:
            reasons.append("joint_step")
        return ClampDecision(applied=(applied,), reasons=tuple(reasons))

    def _clamp_effort(self, args: Sequence[Any]) -> ClampDecision:
        name, effort = args
        limit = self.profile.effort_limits.get(name)
        if limit is None:
            return ClampDecision(reject="no_effort_limit")
        applied = clamp(effort, -limit, limit)
        reasons = ("effort_cap",) if applied != effort else ()
        return ClampDecision(applied=(applied,), reasons=reasons)

    def _clamp_euler(self, args: Sequence[Any]) -> ClampDecision:
        limit = self.profile.euler_limit_rad
        applied = tuple(clamp(value, -limit, limit) for value in args)
        reasons = ("euler_limit",) if applied != tuple(args) else ()
        return ClampDecision(applied=applied, reasons=reasons)


class GroundPolicy(SafetyPolicy):
    """Wheeled base, and the base half of a mobile manipulator.

    A stale command zeros velocity. E-stop stops the base and damps only when
    the robot is already lying down (or the profile sets ``estop_damp``).
    A mobile manipulator's numbers come from its plugin profile; ``sim`` keeps
    the conservative profile used for any robot that has no built-in profile.
    """

    name = "ground"

    def clamp(
        self,
        action: str,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        return super().clamp(action, args, telemetry)

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        super().on_watchdog(transport, telemetry)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        return super().on_estop(transport, telemetry)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        return super().arming(uri, env)

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        return super().fall_or_tilt(telemetry)


class ArmPolicy(SafetyPolicy):
    """Arm: joint position and effort caps.

    Refused joint commands are dropped, not driven to zero. The watchdog still
    zeros velocity, which is what the gate does today for every class. An arm
    spec that is not in the built-in profile table keeps the conservative caps.
    """

    name = "arm"

    def clamp(
        self,
        action: str,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        return super().clamp(action, args, telemetry)

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        super().on_watchdog(transport, telemetry)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        return super().on_estop(transport, telemetry)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        return super().arming(uri, env)

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        return super().fall_or_tilt(telemetry)


class LeggedPolicy(SafetyPolicy):
    """Quadruped. Unitree Go2 keeps its built-in profile.

    The sport allowlist, orientation requirement and axis caps come from that
    profile. A stale command zeros velocity. E-stop calls the transport stop
    (StopMove on Go2) and damps only when the body is already down.
    """

    name = "legged"

    def clamp(
        self,
        action: str,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        return super().clamp(action, args, telemetry)

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        super().on_watchdog(transport, telemetry)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        return super().on_estop(transport, telemetry)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        return super().arming(uri, env)

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        return super().fall_or_tilt(telemetry)


class GroundArmPolicy(SafetyPolicy):
    """Mobile manipulator: ground rules for the base, arm rules for joints.

    Both halves share one profile, so velocity caps and joint limits stay a
    single safety profile. The policy name is ``ground+arm``.
    """

    name = "ground+arm"

    def __init__(self, profile: Any) -> None:
        super().__init__(profile)
        self.ground = GroundPolicy(profile)
        self.arm = ArmPolicy(profile)

    def bind(self, profile: Any) -> GroundArmPolicy:
        other = super().bind(profile)
        other.ground = self.ground.bind(profile)
        other.arm = self.arm.bind(profile)
        return other

    def clamp(
        self,
        action: str,
        args: Sequence[Any],
        telemetry: Telemetry | None,
    ) -> ClampDecision:
        if action in ("joint", "effort"):
            return self.arm.clamp(action, args, telemetry)
        return self.ground.clamp(action, args, telemetry)

    def on_watchdog(self, transport: Any, telemetry: Telemetry | None) -> None:
        self.ground.on_watchdog(transport, telemetry)

    def on_estop(self, transport: Any, telemetry: Telemetry | None) -> EstopAction:
        return self.ground.on_estop(transport, telemetry)

    def arming(self, uri: str | None, env: Mapping[str, str]) -> bool:
        return self.ground.arming(uri, env)

    def fall_or_tilt(self, telemetry: Telemetry) -> tuple[str, dict[str, Any]] | None:
        return self.ground.fall_or_tilt(telemetry)


def reference_profile(spec: RobotSpec | None) -> Any:
    """The class caps for ``spec``: today's :func:`ohho.safety.profile_for` result.

    Built-in ids keep their built-in profiles. Anything else stays on the
    conservative profile. This is the ceiling a plugin policy is compared to.
    """
    safety = _safety()
    if spec is None:
        return safety.profile_for(None, None)
    return safety.profile_for(spec.id, spec)


def policy_for_class(spec: RobotSpec | None) -> SafetyPolicy:
    """The default policy for ``spec.robot_class``.

    Aerial and humanoid do not have their own policy classes yet. They use
    :class:`GroundPolicy`. A built-in aerial id still gets
    :data:`ohho.safety.DRONE_PROFILE` from :func:`ohho.safety.profile_for`
    (tilt guard off, horizontal clip, flight e-stop). That does not raise
    ground, arm, or quadruped caps. An aerial or humanoid spec with no
    built-in profile keeps the conservative profile.
    """
    profile = reference_profile(spec)
    robot_class = RobotClass.MOBILE_BASE if spec is None else spec.robot_class
    if not isinstance(robot_class, RobotClass):
        robot_class = RobotClass.MOBILE_BASE
    if robot_class is RobotClass.MOBILE_MANIPULATOR:
        return GroundArmPolicy(profile)
    if robot_class is RobotClass.ARM:
        return ArmPolicy(profile)
    if robot_class is RobotClass.QUADRUPED:
        return LeggedPolicy(profile)
    return GroundPolicy(profile)


def _robot_id(spec: RobotSpec | None) -> str:
    if spec is None or not getattr(spec, "id", ""):
        return "unknown"
    return str(spec.id)


def _plugin_policy(spec: RobotSpec | None, plugin: Any) -> Any:
    if plugin is None:
        source = getattr(spec, "source", "") if spec is not None else ""
        if not source:
            return None
        from .plugins import plugin_named

        plugin = plugin_named(source)
        if plugin is None:
            return None
    return plugin.safety_policy(spec)


def looser_reasons(plugin_profile: Any, class_profile: Any) -> list[str]:
    """Human-readable ways ``plugin_profile`` is looser than ``class_profile``."""
    reasons: list[str] = []
    for name in _CAP_FIELDS:
        plugin_value = getattr(plugin_profile, name)
        class_value = getattr(class_profile, name)
        if plugin_value > class_value:
            reasons.append(f"{name} {plugin_value} exceeds class cap {class_value}")
    if not plugin_profile.allowlist <= class_profile.allowlist:
        extra = sorted(plugin_profile.allowlist - class_profile.allowlist)
        reasons.append(f"allowlist adds {extra}")
    if class_profile.require_orientation and not plugin_profile.require_orientation:
        reasons.append("require_orientation disabled")
    for name, (lo, hi) in plugin_profile.joint_limits.items():
        class_limit = class_profile.joint_limits.get(name)
        if class_limit is None:
            reasons.append(f"joint_limits[{name!r}] is not in the class policy")
        elif lo < class_limit[0] or hi > class_limit[1]:
            reasons.append(f"joint_limits[{name!r}] is wider than the class policy")
    for name, limit in plugin_profile.effort_limits.items():
        class_limit = class_profile.effort_limits.get(name)
        if class_limit is None:
            reasons.append(f"effort_limits[{name!r}] is not in the class policy")
        elif limit > class_limit:
            reasons.append(
                f"effort_limits[{name!r}] {limit} exceeds class cap {class_limit}"
            )
    if class_profile.estop_damp and not plugin_profile.estop_damp:
        reasons.append("estop_damp disabled")
    return reasons


def _narrow_joints(
    base: Any, plugin_profile: Any, class_profile: Any
) -> dict[str, tuple[float, float]] | None:
    if dict(plugin_profile.joint_limits) == dict(class_profile.joint_limits):
        return None
    joints = {name: (lim[0], lim[1]) for name, lim in base.joint_limits.items()}
    changed = False
    for name, lim in plugin_profile.joint_limits.items():
        if name not in class_profile.joint_limits or name not in joints:
            continue
        nlo = max(joints[name][0], lim[0])
        nhi = min(joints[name][1], lim[1])
        if nlo > nhi:
            raise SafetyPolicyError(
                "plugin safety policy refused at load: "
                f"joint_limits[{name!r}] does not overlap the class cap"
            )
        if (nlo, nhi) != joints[name]:
            joints[name] = (nlo, nhi)
            changed = True
    return joints if changed else None


def _narrow_efforts(
    base: Any, plugin_profile: Any, class_profile: Any
) -> dict[str, float] | None:
    if dict(plugin_profile.effort_limits) == dict(class_profile.effort_limits):
        return None
    efforts = dict(base.effort_limits)
    changed = False
    for name, limit in plugin_profile.effort_limits.items():
        if name not in class_profile.effort_limits or name not in efforts:
            continue
        narrowed = min(efforts[name], limit)
        if narrowed < efforts[name]:
            efforts[name] = narrowed
            changed = True
    return efforts if changed else None


def tighten_profile(base: Any, plugin_profile: Any, class_profile: Any) -> Any:
    """Lower ``base`` where the plugin is stricter than the class default.

    Fields the plugin leaves at the class cap are untouched, so a safety
    config file can still raise a cap up to the hard ceiling. A plugin value
    below the class cap wins over that raise.
    """
    updates: dict[str, Any] = {}
    for name in _CAP_FIELDS:
        plugin_value = getattr(plugin_profile, name)
        if plugin_value < getattr(class_profile, name):
            updates[name] = min(getattr(base, name), plugin_value)
    if plugin_profile.allowlist != class_profile.allowlist:
        updates["allowlist"] = base.allowlist & plugin_profile.allowlist
    if plugin_profile.require_orientation and not base.require_orientation:
        updates["require_orientation"] = True
    joints = _narrow_joints(base, plugin_profile, class_profile)
    if joints is not None:
        updates["joint_limits"] = joints
    efforts = _narrow_efforts(base, plugin_profile, class_profile)
    if efforts is not None:
        updates["effort_limits"] = efforts
    if not updates:
        return base
    return dataclasses.replace(base, **updates)


def load_policy(spec: RobotSpec | None, *, plugin: Any = None) -> SafetyPolicy:
    """Class policy for ``spec``, or the plugin's policy when it returns one.

    A plugin policy that loosens a class cap raises :class:`SafetyPolicyError`.
    That is the load check: :class:`~ohho.safety.SafetyGate` calls this before
    it subscribes to the transport. ``plugin`` overrides discovery; tests use
    it, and the gate passes ``None`` so the spec's plugin is looked up.
    """
    default = policy_for_class(spec)
    supplied = _plugin_policy(spec, plugin)
    if supplied is None:
        return default
    robot_id = _robot_id(spec)
    if not isinstance(supplied, SafetyPolicy):
        raise SafetyPolicyError(
            f"plugin safety policy for {robot_id!r} refused at load: "
            f"expected a SafetyPolicy, got {type(supplied).__name__}"
        )
    safety = _safety()
    profile = getattr(supplied, "profile", None)
    class_profile = default.profile
    if not isinstance(profile, safety.SafetyProfile):
        raise SafetyPolicyError(
            f"plugin safety policy for {robot_id!r} refused at load: "
            "caps must be a SafetyProfile"
        )
    reasons = looser_reasons(profile, class_profile)
    if reasons:
        raise SafetyPolicyError(
            f"plugin safety policy for {robot_id!r} refused at load: "
            f"loosens class caps ({'; '.join(reasons)})"
        )
    supplied.from_plugin = True
    supplied.class_profile = class_profile
    return supplied


def prepare_gate_profile(
    spec: RobotSpec | None,
    *,
    profile: Any = None,
    config: Any = None,
    estop_damp: bool = False,
) -> tuple[SafetyPolicy, Any]:
    """Policy plus the profile the gate will ceiling-check.

    An explicit ``profile`` is the caller's (tests and ``SafetyGate(profile=)``).
    Otherwise the profile is :func:`ohho.safety.profile_for`, then any plugin
    caps that are stricter than the class default. ``estop_damp`` is applied
    last, before the hard ceiling.
    """
    safety = _safety()
    chosen = load_policy(spec)
    if profile is None:
        robot_id = spec.id if spec is not None else None
        base = safety.profile_for(robot_id, spec, config)
        if chosen.from_plugin:
            base = tighten_profile(base, chosen.profile, chosen.class_profile)
    else:
        base = profile
    if estop_damp:
        base = dataclasses.replace(base, estop_damp=True)
    return chosen, base


def arming_uri(transport: Any) -> str:
    """Encode the transport fields :func:`ohho.safety.requires_hardware_arming` reads.

    DDS keeps the interface and the domain value, including a non-int domain,
    so a malformed domain still fails closed.
    """
    protocol = str(getattr(transport, "protocol", "") or "")
    if protocol == "dds":
        iface = str(getattr(transport, "iface", "") or "")
        domain = getattr(transport, "domain_id", 0)
        return f"dds://{iface}?domain={domain!r}"
    return f"{protocol}://"


class _UriTransport:
    """Minimal transport view rebuilt from :func:`arming_uri`."""

    def __init__(self, protocol: str, iface: str = "", domain_id: Any = 0) -> None:
        self.protocol = protocol
        self.iface = iface
        self.domain_id = domain_id


def _parse_domain(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError):
        return text


def _transport_from_uri(uri: str | None) -> _UriTransport:
    if not uri:
        return _UriTransport("unknown")
    if uri.startswith("dds://"):
        rest = uri[len("dds://") :]
        iface, marker, domain = rest.rpartition("?domain=")
        if marker:
            return _UriTransport("dds", iface, _parse_domain(domain))
        return _UriTransport("dds", rest, 0)
    protocol, sep, _rest = uri.partition("://")
    if not sep:
        return _UriTransport(uri)
    return _UriTransport(protocol)


__all__ = [
    "ArmPolicy",
    "ClampDecision",
    "EstopAction",
    "GroundArmPolicy",
    "GroundPolicy",
    "LeggedPolicy",
    "SafetyPolicy",
    "SafetyPolicyError",
    "arming_uri",
    "load_policy",
    "looser_reasons",
    "policy_for_class",
    "prepare_gate_profile",
    "reference_profile",
    "tighten_profile",
]
