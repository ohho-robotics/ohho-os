"""SafetyGate — a fail-closed ``Transport`` wrapper for real robots (OHH-117).

``Robot.connect`` wraps every transport whose ``protocol != "simulated"`` in a
``SafetyGate``. The gate implements the ordinary ``Transport`` interface, so
nothing above it changes; it decides what actually reaches the motors:

- **Caps.** Per-axis velocity caps and acceleration caps (clamped, never
  rejected), joint position limits + per-command step limits, and effort caps
  where the transport carries effort commands. Caps come from a per-robot
  ``SafetyProfile``; ``HARD_MAX_LIN`` / ``HARD_MAX_ANG`` bound every profile and
  can only be raised by an explicit safety config file.
- **Watchdogs.** No velocity command for 300 ms → zero velocity. No robot state
  for 500 ms while armed → stop + latch.
- **Latched e-stop.** Stop first; damp only when the robot is already lying
  down (or ``estop_damp`` is set). Release needs a human at a TTY and leaves the
  gate disarmed, so motion needs a fresh ``arm()``.
- **Allowlist.** ``command(action, *args)`` forwards only allowlisted sport
  actions; flips, jumps, pounce, dance, handstand, walk-upright, ``free_*`` and
  cross-step are refused even if a config file lists them. Aerial robots use a
  separate allowlist (``hold``, ``land``, ``rtl``, ``offboard_velocity``,
  ``offboard_position``). They do not reuse the Go2 set.
- **Tilt / fault guard.** On ground robots, ``|roll|`` or ``|pitch|`` over
  0.6 rad, a non-zero ``error_code``, or unreadable orientation values → stop
  + latch. The 0.6 rad tilt latch is quadruped-only. A non-zero ``error_code``
  still latches an aircraft.
- **Arming.** Motion is refused until ``arm()``. On hardware (anything except a
  simulator, a loopback interface on a non-zero DDS domain, or a loopback
  mavlink SITL under the GitHub CI sim opt-in) arming needs
  ``OHHO_ARM_HARDWARE=1``, an interactive TTY and the typed confirmation
  phrase. CI and non-TTY callers fail closed. Aerial arm and takeoff have one
  GitHub Actions opt-in for a proven simulator (``OHHO_CI_SIM_ARMING=1``). It
  never applies to hardware or ``release_stop()``.
- **Audit.** Arm/disarm, e-stops, releases, clamps, refusals and watchdog trips
  are appended as JSON Lines to ``~/.ohho/safety/<UTC date>.jsonl``.

Every refusal also sends a zero-velocity command. Joint *positions* are never
"zeroed" (that would move the arm); refused joint/effort commands are dropped.

``sim://`` bypasses the gate in ``Robot.connect``. Wrapping a ``SimTransport``
explicitly is supported (tests, demos) and arms without a TTY prompt.

This is a software guard inside the Python process. Code in the same process
can reach private attributes; it does not replace a hardware e-stop.
"""

from __future__ import annotations

import dataclasses
import getpass
import ipaddress
import json
import logging
import math
import os
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union
from urllib.parse import urlparse

from . import capabilities as caps
from .aerial import Frame
from .drone.estop import command_is_forbidden, disarm_allowed, estop_commands
from .drone.telemetry import positive_up_m
from .registry import RobotSpec
from .schema import ConnectionState, Telemetry, TransportStatus, Velocity, clamp
from .transport import Transport, BaseTransport

log = logging.getLogger("ohho.safety")

ARM_ENV = "OHHO_ARM_HARDWARE"
CI_SIM_ARMING_ENV = "OHHO_CI_SIM_ARMING"
CI_SITL_ENV = "OHHO_CI_SITL"
CONFIG_ENV = "OHHO_SAFETY_CONFIG"
AUDIT_DIR_ENV = "OHHO_SAFETY_DIR"
CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "JENKINS_URL",
    "TF_BUILD",
)
ARM_PHRASE = "I am physically present, the area is clear, the remote is in my hand"
RELEASE_PHRASE = "release"

HARD_MAX_LIN = 1.5  # m/s, every linear axis
HARD_MAX_ANG = 2.0  # rad/s
CMD_TIMEOUT_S = 0.3
STATE_TIMEOUT_S = 0.5
TILT_LIMIT_RAD = 0.6
EULER_LIMIT_RAD = 0.3

GO2_ACTIONS = frozenset(
    {
        "stand_up",
        "stand_down",
        "balance_stand",
        "recovery_stand",
        "stop_move",
        "move",
        "euler",
        "sit",
        "rise_sit",
    }
)
BASE_ACTIONS = frozenset({"move", "stop_move"})

# Aerial allowlist (OHH-288). Deliberately not GO2_ACTIONS: that set has no
# ``land``, and copying it would refuse a landing. ``arm`` / ``takeoff`` are
# human-at-a-TTY actions and are absent here so a config file cannot turn
# them into ordinary commands.
DRONE_ACTIONS = frozenset(
    {
        "hold",
        "land",
        "rtl",
        "offboard_velocity",
        "offboard_position",
    }
)
DRONE_TTY_ACTIONS = frozenset({"arm", "arm_flight", "takeoff"})
# Later MCP servers must not register these. ``rtl`` is a gate action and is
# still denied to MCP. ``land`` and ``hold`` are allowed.
DRONE_MCP_DENIED = frozenset(
    {
        "arm",
        "arm_flight",
        "drone_arm",
        "takeoff",
        "drone_takeoff",
        "rtl",
        "drone_rtl",
        "release_stop",
        "disarm",
        "disarm_flight",
        "drone_disarm",
    }
)

DRONE_MAX_HORIZONTAL = 1.5  # m/s
DRONE_MAX_VERTICAL = 1.0  # m/s
DRONE_MAX_YAW = 0.8  # rad/s
DRONE_TAKEOFF_ALT_M = 3.0  # m, teleop takeoff ceiling
DRONE_FENCE_RADIUS_M = 10.0  # m, default sim cylinder
DRONE_FENCE_ALT_M = 3.0  # m, default sim ceiling
TAKEOFF_MIN_BATTERY = 0.3
_FENCE_EPS = 1e-6
_CRAZYFLIE_BACKENDS = frozenset({"crazyflie", "crazyflie-cf21"})
_SIM_BACKENDS = frozenset({"drone-sim", "sim", "px4-sitl", "crazyflie-sim"})

_DENIED_FRAGMENTS = (
    "flip",
    "jump",
    "pounce",
    "dance",
    "handstand",
    "hand_stand",
    "upright",
    "cross_step",
    "crossstep",
)
_DENIED_PREFIXES = ("free",)
_LOOPBACK = frozenset({"lo", "lo0", "localhost", "127.0.0.1", "::1"})
_ZERO = (0.0, 0.0, 0.0)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[\s\-]+")


# ── actions ───────────────────────────────────────────────────────────────────
def normalize_action(name: str) -> str:
    """``"StandUp"`` / ``"stand-up"`` / ``"stand up"`` → ``"stand_up"``."""
    return _SEPARATORS.sub("_", _CAMEL.sub("_", name.strip())).lower()


def is_denied(action: str) -> bool:
    """True for actions that are refused no matter what the allowlist says."""
    return action.startswith(_DENIED_PREFIXES) or any(
        frag in action for frag in _DENIED_FRAGMENTS
    )


def drone_mcp_allowed(action: object) -> bool:
    """True only for aerial actions a later MCP server may register.

    ``hold``, ``land``, ``offboard_velocity`` and ``offboard_position`` pass.
    ``arm_flight``, ``takeoff``, ``rtl``, ``disarm`` and ``release_stop`` do
    not. Denied names such as ``flip`` do not. This does not arm anything.
    """
    if not isinstance(action, str) or not action.strip():
        return False
    name = normalize_action(action)
    if name in DRONE_MCP_DENIED or is_denied(name):
        return False
    return name in DRONE_ACTIONS


def clip_horizontal(vx: float, vy: float, limit: float) -> tuple[float, float]:
    """Scale ``(vx, vy)`` so the horizontal magnitude is at most ``limit``."""
    mag = math.hypot(vx, vy)
    if mag <= limit or mag == 0.0:
        return vx, vy
    scale = limit / mag
    return vx * scale, vy * scale


def fence_violation(
    x: float,
    y: float,
    z: float,
    frame: Frame,
    profile: "SafetyProfile",
) -> Optional[str]:
    """Why a local position is outside the cylinder, or None when it is inside.

    The fence is a cylinder about the local origin: horizontal radius and a
    positive-up altitude ceiling. The floor is the origin. ``BODY_FRD`` cannot
    be proved against that cylinder, so it is a violation.
    """
    if frame is Frame.BODY_FRD:
        return "body_frame"
    if frame is Frame.LOCAL_NED:
        radius = math.hypot(x, y)
        altitude = -z
    elif frame is Frame.LOCAL_ENU:
        radius = math.hypot(x, y)
        altitude = z
    else:
        return "frame"
    if radius > profile.fence_radius_m + _FENCE_EPS:
        return "radius"
    if altitude > profile.fence_max_alt_m + _FENCE_EPS:
        return "altitude"
    if altitude < -_FENCE_EPS:
        return "below_origin"
    return None


# ── profiles ──────────────────────────────────────────────────────────────────
def _check_number(name: str, value: Any, *, upper: Optional[float] = None) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not math.isfinite(value) or value < 0:
        raise ValueError(f"safety profile: {name} must be a finite number >= 0")
    if upper is not None and value > upper:
        raise ValueError(f"safety profile: {name} may not exceed {upper}")


@dataclass(frozen=True)
class SafetyProfile:
    """Per-robot safety limits. Immutable; validated on construction.

    Velocity caps are m/s (``max_vx``, ``max_vy``) and rad/s (``max_wz``);
    acceleration caps are per second. Timeouts, tilt and euler limits may only
    be made stricter than the module defaults. ``joint_limits`` maps joint name
    to ``(lo, hi)`` radians and ``effort_limits`` to a symmetric cap; joints
    without an entry are refused.

    ``aerial`` selects the drone allowlist, the cylinder geofence, and the
    flight e-stop mapping. ``tilt_guard`` is the 0.6 rad roll/pitch latch; it
    stays on for ground robots and off for the M1 drone profile. ``max_vz``
    and ``max_horizontal`` are the vertical and horizontal speed caps in m/s.
    """

    max_vx: float = 0.2
    max_vy: float = 0.2
    max_wz: float = 0.5
    max_accel_x: float = 0.5
    max_accel_y: float = 0.5
    max_accel_wz: float = 1.0
    cmd_timeout_s: float = CMD_TIMEOUT_S
    state_timeout_s: float = STATE_TIMEOUT_S
    tilt_limit_rad: float = TILT_LIMIT_RAD
    euler_limit_rad: float = EULER_LIMIT_RAD
    require_orientation: bool = False
    allowlist: frozenset = BASE_ACTIONS
    joint_limits: Mapping[str, tuple] = field(default_factory=dict, hash=False)
    max_joint_step: float = 0.25
    effort_limits: Mapping[str, float] = field(default_factory=dict, hash=False)
    estop_damp: bool = False
    lying_body_height: float = 0.15
    aerial: bool = False
    tilt_guard: bool = True
    max_vz: float = 0.0
    max_horizontal: float = 0.0
    takeoff_alt_m: float = 0.0
    fence_radius_m: float = 0.0
    fence_max_alt_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_vx",
            "max_vy",
            "max_wz",
            "max_accel_x",
            "max_accel_y",
            "max_accel_wz",
            "max_joint_step",
            "lying_body_height",
            "max_vz",
            "max_horizontal",
            "takeoff_alt_m",
            "fence_radius_m",
            "fence_max_alt_m",
        ):
            _check_number(name, getattr(self, name))
        _check_number("euler_limit_rad", self.euler_limit_rad, upper=EULER_LIMIT_RAD)
        for name, upper in (
            ("cmd_timeout_s", CMD_TIMEOUT_S),
            ("state_timeout_s", STATE_TIMEOUT_S),
            ("tilt_limit_rad", TILT_LIMIT_RAD),
        ):
            value = getattr(self, name)
            _check_number(name, value, upper=upper)
            if value <= 0:
                raise ValueError(f"safety profile: {name} must be > 0")

        if not isinstance(self.aerial, bool) or not isinstance(self.tilt_guard, bool):
            raise ValueError("safety profile: aerial and tilt_guard must be bools")
        if self.tilt_guard is False and not self.aerial:
            raise ValueError("safety profile: tilt guard cannot be disabled")
        if self.aerial and self.takeoff_alt_m > self.fence_max_alt_m + _FENCE_EPS:
            raise ValueError("safety profile: takeoff altitude exceeds the geofence")

        universe = DRONE_ACTIONS if self.aerial else GO2_ACTIONS
        allow = frozenset(normalize_action(str(a)) for a in self.allowlist)
        for a in allow:
            if is_denied(a) or a not in universe:
                raise ValueError(f"safety profile: action '{a}' cannot be allowlisted")

        joints: dict[str, tuple[float, float]] = {}
        for name, lim in dict(self.joint_limits).items():
            try:
                lo, hi = (float(v) for v in lim)
            except (TypeError, ValueError):
                raise ValueError(
                    f"safety profile: joint_limits[{name!r}] must be [lo, hi]"
                ) from None
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"safety profile: joint_limits[{name!r}] is invalid")
            joints[str(name)] = (lo, hi)

        efforts: dict[str, float] = {}
        for name, lim in dict(self.effort_limits).items():
            _check_number(f"effort_limits[{name!r}]", lim)
            efforts[str(name)] = float(lim)

        object.__setattr__(self, "allowlist", allow)
        object.__setattr__(self, "joint_limits", MappingProxyType(joints))
        object.__setattr__(self, "effort_limits", MappingProxyType(efforts))


# Mirrors the Feetech adapter's DEFAULT_JOINT_MIN/MAX — keep the two in sync.
_OMNIBOT_JOINT_LIMITS = {
    "arm_shoulder_pan": (-3.14, 3.14),
    "arm_shoulder_lift": (-1.57, 1.57),
    "arm_elbow_flex": (-1.57, 1.57),
    "arm_wrist_flex": (-1.57, 1.57),
    "arm_wrist_roll": (-3.14, 3.14),
    "arm_gripper": (-0.1, 0.8),
}

GO2_PROFILE = SafetyProfile(
    max_vx=0.5,
    max_vy=0.3,
    max_wz=1.0,
    max_accel_x=1.0,
    max_accel_y=0.6,
    max_accel_wz=2.0,
    require_orientation=True,
    allowlist=GO2_ACTIONS,
)

OMNIBOT_PROFILE = SafetyProfile(
    max_vx=0.2,
    max_vy=0.2,
    max_wz=1.0,
    max_accel_x=0.5,
    max_accel_y=0.5,
    max_accel_wz=2.0,
    joint_limits=_OMNIBOT_JOINT_LIMITS,
)

# M1 sim fence and caps. require_orientation stays off until drone-sim
# publishes a stable attitude. The 0.6 rad tilt latch stays off: it is the
# quadruped guard and trips in normal flight attitudes.
DRONE_PROFILE = SafetyProfile(
    max_vx=DRONE_MAX_HORIZONTAL,
    max_vy=DRONE_MAX_HORIZONTAL,
    max_wz=DRONE_MAX_YAW,
    max_accel_x=1.0,
    max_accel_y=1.0,
    max_accel_wz=1.6,
    require_orientation=False,
    allowlist=DRONE_ACTIONS,
    aerial=True,
    tilt_guard=False,
    max_vz=DRONE_MAX_VERTICAL,
    max_horizontal=DRONE_MAX_HORIZONTAL,
    takeoff_alt_m=DRONE_TAKEOFF_ALT_M,
    fence_radius_m=DRONE_FENCE_RADIUS_M,
    fence_max_alt_m=DRONE_FENCE_ALT_M,
)

PROFILES: Mapping[str, SafetyProfile] = MappingProxyType(
    {
        "unitree-go2": GO2_PROFILE,
        "omnibot": OMNIBOT_PROFILE,
        "px4-sitl": DRONE_PROFILE,
        "crazyflie-cf21": DRONE_PROFILE,
        "px4-x500": DRONE_PROFILE,
    }
)

_GENERIC_MAX_LIN = 0.3
_GENERIC_MAX_ANG = 1.0


@dataclass(frozen=True)
class SafetyConfig:
    """Operator-authored overrides, loaded only from an explicit JSON file."""

    max_lin_ceiling: float = HARD_MAX_LIN
    max_ang_ceiling: float = HARD_MAX_ANG
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, hash=False)
    source: Optional[str] = None


def profile_for(
    robot_id: Optional[str],
    spec: Optional[RobotSpec] = None,
    config: Optional[SafetyConfig] = None,
) -> SafetyProfile:
    """The safety profile for a robot: built-in, or conservative from its spec,
    with any overrides from ``config`` applied."""
    base = PROFILES.get(robot_id or "")
    if base is None:
        if spec is not None and spec.has(caps.AERIAL):
            base = DRONE_PROFILE
        elif spec is None:
            base = SafetyProfile()
        else:
            lin = min(spec.max_lin, _GENERIC_MAX_LIN)
            base = SafetyProfile(
                max_vx=lin,
                max_vy=lin if spec.has(caps.BASE_HOLONOMIC) else 0.0,
                max_wz=min(spec.max_ang, _GENERIC_MAX_ANG),
            )
    overrides = config.profiles.get(robot_id or "") if config is not None else None
    return dataclasses.replace(base, **overrides) if overrides else base


_PROFILE_FIELDS = frozenset(f.name for f in dataclasses.fields(SafetyProfile))


def _positive(where: str, value: Any) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not math.isfinite(value) or value <= 0:
        raise ValueError(f"safety config: {where} must be a positive number")
    return float(value)


def load_safety_config(path: Union[str, Path]) -> SafetyConfig:
    """Load a safety config JSON file. Anything malformed raises ``ValueError``.

    Shape::

        {"hard_ceiling": {"max_lin": 2.0, "max_ang": 3.0},
         "profiles": {"unitree-go2": {"max_vx": 0.8, "allowlist": ["move"]}}}
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read safety config {p}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("safety config: top level must be an object")
    unknown = set(data) - {"hard_ceiling", "profiles"}
    if unknown:
        raise ValueError(f"safety config: unknown keys {sorted(unknown)}")

    ceiling = data.get("hard_ceiling", {})
    if not isinstance(ceiling, dict) or set(ceiling) - {"max_lin", "max_ang"}:
        raise ValueError("safety config: hard_ceiling takes only max_lin, max_ang")
    lin = _positive("hard_ceiling.max_lin", ceiling.get("max_lin", HARD_MAX_LIN))
    ang = _positive("hard_ceiling.max_ang", ceiling.get("max_ang", HARD_MAX_ANG))

    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("safety config: profiles must be an object")
    profiles: dict[str, dict[str, Any]] = {}
    for rid, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ValueError(f"safety config: profiles.{rid} must be an object")
        bad = set(raw) - _PROFILE_FIELDS
        if bad:
            raise ValueError(
                f"safety config: profiles.{rid} unknown keys {sorted(bad)}"
            )
        ov = dict(raw)
        if "allowlist" in ov:
            if not isinstance(ov["allowlist"], list):
                raise ValueError(
                    f"safety config: profiles.{rid}.allowlist must be a list"
                )
            ov["allowlist"] = frozenset(ov["allowlist"])
        base_profile = PROFILES.get(rid, SafetyProfile())
        try:
            candidate = dataclasses.replace(base_profile, **ov)
        except (TypeError, ValueError) as e:
            raise ValueError(f"safety config: profiles.{rid}: {e}") from e
        if candidate.aerial != base_profile.aerial:
            raise ValueError(f"safety config: profiles.{rid} cannot change aerial")
        if max(candidate.max_vx, candidate.max_vy) > lin or candidate.max_wz > ang:
            raise ValueError(f"safety config: profiles.{rid} exceeds the hard ceiling")
        profiles[str(rid)] = ov
    return SafetyConfig(lin, ang, profiles, str(p))


def resolve_config(
    path: Optional[Union[str, Path]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> SafetyConfig:
    """``path``, else ``$OHHO_SAFETY_CONFIG``, else built-in defaults."""
    env = os.environ if environ is None else environ
    src = path or env.get(CONFIG_ENV)
    return load_safety_config(src) if src else SafetyConfig()


def _apply_ceiling(p: SafetyProfile, cfg: SafetyConfig) -> SafetyProfile:
    capped = dataclasses.replace(
        p,
        max_vx=min(p.max_vx, cfg.max_lin_ceiling),
        max_vy=min(p.max_vy, cfg.max_lin_ceiling),
        max_wz=min(p.max_wz, cfg.max_ang_ceiling),
        max_vz=min(p.max_vz, cfg.max_lin_ceiling),
        max_horizontal=min(p.max_horizontal, cfg.max_lin_ceiling),
    )
    if capped != p:
        log.warning(
            "safety: profile caps lowered to the hard ceiling (%.2f m/s, %.2f rad/s)",
            cfg.max_lin_ceiling,
            cfg.max_ang_ceiling,
        )
    return capped


# ── arming policy ─────────────────────────────────────────────────────────────
def _is_loopback(iface: str) -> bool:
    s = iface.strip().lower()
    return s in _LOOPBACK or s.startswith(("127.", "loopback"))


def _ros2_localhost_ci(environ: Mapping[str, str]) -> bool:
    """True when ROS 2 discovery is localhost-only on a non-zero domain (CI fake robot)."""
    loc = str(environ.get("ROS_LOCALHOST_ONLY", "")).strip().lower()
    if loc in ("1", "true", "yes"):
        localhost_only = True
    else:
        adr = str(environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "")).strip().upper()
        localhost_only = adr == "LOCALHOST"
    if not localhost_only:
        return False
    raw = str(environ.get("ROS_DOMAIN_ID", "0")).strip()
    if not raw:
        return False
    try:
        domain = int(raw, 10)
    except ValueError:
        return False
    if isinstance(domain, bool):
        return False
    return domain != 0


def _mavlink_sitl_ci(transport: object, env: Mapping[str, str]) -> bool:
    """True only for loopback mavlink SITL under the GitHub CI sim opt-in.

    Same shape as :func:`_ros2_localhost_ci`: any other mavlink link stays a
    hardware gate. An inspection error is false so the caller fails closed.
    """
    try:
        if not _token_true(env.get("CI", "")):
            return False
        if not _token_true(env.get("GITHUB_ACTIONS", "")):
            return False
        if env.get(CI_SIM_ARMING_ENV) != "1":
            return False
        if env.get(CI_SITL_ENV) != "1":
            return False
        if ARM_ENV in env:
            return False
        uri = getattr(transport, "uri", None)
        if not isinstance(uri, str) or not uri:
            return False
        parsed = urlparse(uri)
        if parsed.scheme != "mavlink+udp":
            return False
        if not _literal_loopback(parsed.hostname):
            return False
        if not _exposes(transport, "simulation_target"):
            return False
        return getattr(transport, "simulation_target") is True
    except Exception:
        return False


def requires_hardware_arming(
    transport: Transport,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """False for simulators, loopback DDS, localhost-only ROS 2 CI, and CI SITL.

    Mavlink is non-hardware only when ``CI`` and ``GITHUB_ACTIONS`` are
    ``true``, both CI sim opt-ins are exactly ``1``, ``OHHO_ARM_HARDWARE`` is
    absent, and the URI is ``mavlink+udp`` on a literal loopback host whose
    ``simulation_target`` is true. Any other mavlink link, and any inspection
    error, stays hardware.
    """
    env = os.environ if environ is None else environ
    proto = getattr(transport, "protocol", "")
    if proto == "simulated":
        return False
    if proto == "dds":
        iface = str(getattr(transport, "iface", "") or "")
        domain = getattr(transport, "domain_id", 0)
        is_int = isinstance(domain, int) and not isinstance(domain, bool)
        if _is_loopback(iface) and is_int and domain != 0:
            return False
    if proto == "ros2" and _ros2_localhost_ci(env):
        return False
    if proto == "mavlink" and _mavlink_sitl_ci(transport, env):
        return False
    return True


def _in_ci(env: Mapping[str, str]) -> bool:
    return any(
        str(env.get(v, "")).strip().lower() not in ("", "0", "false", "no")
        for v in CI_ENV_VARS
    )


class _CiSimArming(NamedTuple):
    reason: str
    backend: str = ""
    host: str = ""
    uri: str = ""
    target: str = ""


def _token_true(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == "true"


def _exposes(obj: object, name: str) -> bool:
    for cls in type(obj).__mro__:
        if name in cls.__dict__:
            return True
    try:
        return name in vars(obj)
    except TypeError:
        return False


def _literal_loopback(host: object) -> bool:
    """True for ``localhost``, ``127.0.0.0/8``, or the address ``::1``.

    IPv4-mapped addresses such as ``::ffff:127.0.0.1`` are not ``::1`` and are
    refused. Docker names are not loopback.
    """
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv4Address):
        return bool(parsed.is_loopback)
    return parsed == ipaddress.ip_address("::1")


def _transport_backend_id(inner: object) -> str:
    """Backend id used for aerial decisions.

    An explicit string ``backend_id`` wins. ``protocol == "simulated"`` with
    no explicit id is ``drone-sim``.
    """
    explicit = getattr(inner, "backend_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    proto = str(getattr(inner, "protocol", "") or "").strip().lower()
    if proto == "simulated":
        return "drone-sim"
    return proto or "unknown"


def _sitl_loopback(
    inner: object,
    env: Mapping[str, str],
    *,
    backend: str,
    scheme: str,
    flag: str,
) -> _CiSimArming:
    if env.get(CI_SITL_ENV) != "1":
        return _CiSimArming("sitl_opt_in", backend=backend)
    uri = getattr(inner, "uri", None)
    if not isinstance(uri, str) or not uri:
        return _CiSimArming("uri", backend=backend)
    parsed = urlparse(uri)
    if parsed.scheme != scheme:
        return _CiSimArming("scheme", backend=backend, uri=uri)
    host = parsed.hostname
    if not _literal_loopback(host):
        shown = host if isinstance(host, str) else ""
        return _CiSimArming("host", backend=backend, host=shown, uri=uri)
    if not _exposes(inner, flag) or getattr(inner, flag) is not True:
        return _CiSimArming("sim_flag", backend=backend, host=str(host), uri=uri)
    return _CiSimArming("", backend=backend, host=str(host), uri=uri, target=str(host))


def _ci_sim_arming_checked(
    inner: object, env: Mapping[str, str], *, hardware: bool
) -> _CiSimArming:
    if not _token_true(env.get("CI", "")):
        return _CiSimArming("not_github_actions")
    if not _token_true(env.get("GITHUB_ACTIONS", "")):
        return _CiSimArming("not_github_actions")
    if env.get(CI_SIM_ARMING_ENV) != "1":
        return _CiSimArming("opt_in")
    if ARM_ENV in env:
        return _CiSimArming("hardware_env")
    if hardware is not False:
        return _CiSimArming("hardware_gate")
    if _exposes(inner, "requires_hardware_arming") and (
        getattr(inner, "requires_hardware_arming") is not False
    ):
        return _CiSimArming("hardware_gate")
    backend = _transport_backend_id(inner)
    if backend == "drone-sim":
        return _CiSimArming("", backend=backend, target=backend)
    if backend == "px4-dds":
        return _sitl_loopback(
            inner, env, backend=backend, scheme="px4+dds", flag="sim_target"
        )
    if backend == "mavlink":
        return _sitl_loopback(
            inner,
            env,
            backend=backend,
            scheme="mavlink+udp",
            flag="simulation_target",
        )
    return _CiSimArming("not_sim_target", backend=backend)


def _ci_sim_arming(
    inner: object, env: Mapping[str, str], *, hardware: bool
) -> _CiSimArming:
    try:
        return _ci_sim_arming_checked(inner, env, hardware=hardware)
    except Exception:
        return _CiSimArming("inspect_error")


def ci_sim_arming_reason(
    inner: object, env: Mapping[str, str], *, hardware: bool
) -> str:
    """Return ``''`` only when GitHub Actions may arm a proven simulator.

    ``CI`` and ``GITHUB_ACTIONS`` must be ``true``, ``OHHO_CI_SIM_ARMING`` must
    be exactly ``1``, and ``OHHO_ARM_HARDWARE`` must be absent. ``hardware``
    must be false, and so must the transport's ``requires_hardware_arming``
    when it exposes one. The target is ``drone-sim``, or a loopback
    ``px4+dds`` / ``mavlink+udp`` link with its own sim flag and
    ``OHHO_CI_SITL=1``. Anything else, including an inspection error, is a
    short refusal reason. This does not arm a vehicle.
    """
    return _ci_sim_arming(inner, env, hardware=hardware).reason


def _stdin_isatty() -> bool:
    stdin = sys.stdin
    try:
        return bool(stdin is not None and stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _prompt(text: str) -> str:
    return input(text)


def _norm_phrase(s: Any) -> str:
    return " ".join(str(s).split()).casefold().rstrip(".")


def _operator() -> dict[str, str]:
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    return {"user": user, "host": socket.gethostname()}


# ── audit log ─────────────────────────────────────────────────────────────────
def default_audit_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``$OHHO_SAFETY_DIR`` or ``~/.ohho/safety``."""
    env = os.environ if environ is None else environ
    custom = env.get(AUDIT_DIR_ENV)
    return Path(custom) if custom else Path.home() / ".ohho" / "safety"


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if v is None or isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


class AuditLog:
    """Append-only JSON Lines audit trail, one file per UTC day."""

    def __init__(
        self,
        directory: Optional[Union[str, Path]] = None,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = (
            Path(directory) if directory is not None else default_audit_dir()
        )
        self._wall = wall_clock
        self._lock = threading.Lock()

    def path_for(self, t: float) -> Path:
        day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        return self.directory / f"{day}.jsonl"

    def write(self, event: str, **fields: Any) -> bool:
        """Append one record. Returns False (never raises) if it can't be written."""
        t = self._wall()
        rec: dict[str, Any] = {
            "ts": datetime.fromtimestamp(t, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "t": t,
            "event": event,
        }
        rec.update({k: _jsonable(v) for k, v in fields.items()})
        line = json.dumps(rec, sort_keys=True, allow_nan=False) + "\n"
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path_for(t).open("a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as e:
            log.error("safety: cannot write audit log in %s: %s", self.directory, e)
            return False
        return True


# ── helpers ───────────────────────────────────────────────────────────────────
def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _floats(args: Sequence[Any], n: int, *, finite: bool = False) -> Optional[tuple]:
    if len(args) != n:
        return None
    out = []
    for a in args:
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            return None
        out.append(float(a))
    if finite and not _finite(*out):
        return None
    return tuple(out)


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _accel_step(prev: float, target: float, dmax: float) -> float:
    """Braking toward zero is free; speeding up (or reversing) moves ≤ dmax."""
    base = prev if prev * target > 0 else 0.0
    if abs(target) <= abs(base):
        return target
    return base + math.copysign(min(abs(target) - abs(base), dmax), target)


def _orientation(t: Telemetry) -> Optional[tuple[float, float]]:
    c = t.custom
    if "roll" in c or "pitch" in c:
        return _as_float(c.get("roll")), _as_float(c.get("pitch"))
    rpy = c.get("imu_rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 2:
        return _as_float(rpy[0]), _as_float(rpy[1])
    return None


def _is_lying_down(t: Optional[Telemetry], p: SafetyProfile) -> bool:
    if t is None:
        return False
    h = t.custom.get("body_height")
    if isinstance(h, bool) or not isinstance(h, (int, float)) or not math.isfinite(h):
        return False
    return h < p.lying_body_height


# ── the gate ──────────────────────────────────────────────────────────────────
class SafetyGate(BaseTransport):
    """Wrap a real-robot ``Transport`` so every command passes the safety rules.

    Starts **disarmed**: motion is refused until :meth:`arm` succeeds. The
    watchdog runs on a daemon thread started by :meth:`connect` (pass
    ``auto_tick=False`` and call :meth:`tick` yourself for deterministic tests).
    ``clock``, ``environ``, ``isatty`` and ``input_fn`` are injectable for tests.
    """

    def __init__(
        self,
        inner: Transport,
        spec: Optional[RobotSpec] = None,
        *,
        profile: Optional[SafetyProfile] = None,
        config: Optional[SafetyConfig] = None,
        estop_damp: bool = False,
        audit: Optional[AuditLog] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        environ: Optional[Mapping[str, str]] = None,
        isatty: Optional[Callable[[], bool]] = None,
        input_fn: Optional[Callable[[str], str]] = None,
        auto_tick: bool = True,
        tick_period_s: float = 0.05,
    ) -> None:
        if isinstance(inner, SafetyGate):
            raise ValueError("transport is already wrapped in a SafetyGate")
        super().__init__()
        self._inner = inner
        self.spec = spec
        self.robot_id = spec.id if spec is not None else "unknown"
        self.protocol = getattr(inner, "protocol", "unknown")
        self._env: Mapping[str, str] = os.environ if environ is None else environ
        cfg = config if config is not None else resolve_config(environ=self._env)
        base = profile or profile_for(spec.id if spec else None, spec, cfg)
        if estop_damp:
            base = dataclasses.replace(base, estop_damp=True)
        self._profile = _apply_ceiling(base, cfg)
        self._audit_log = audit or AuditLog(
            default_audit_dir(self._env), wall_clock=wall_clock
        )
        self._clock = clock
        self._isatty = isatty or _stdin_isatty
        self._input = input_fn or _prompt
        self._auto_tick = auto_tick
        self._tick_period = tick_period_s
        self._hw = requires_hardware_arming(inner, self._env)

        self._lock = threading.RLock()
        self._armed = False
        self._latched = False
        self._latch_reason: Optional[str] = None
        self._latest: Optional[Telemetry] = None
        self._last_state_t: Optional[float] = None
        self._last_cmd_t = 0.0
        self._last_sent = _ZERO
        self._moving = False
        self._joint_cmds: dict[str, float] = {}
        self._audit_failing = False
        self._last_setpoint_t: Optional[float] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        inner.on_telemetry(self._on_inner_tele)
        inner.on_status(self._emit_status)

    # ── introspection ─────────────────────────────────────────────────────────
    @property
    def profile(self) -> SafetyProfile:
        return self._profile

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def latch_reason(self) -> Optional[str]:
        return self._latch_reason

    @property
    def requires_hardware_arming(self) -> bool:
        return self._hw

    @property
    def odom_received(self) -> bool:
        """Forwarded from the inner transport when it tracks live odometry."""
        inner = self._inner
        flag = getattr(inner, "odom_received", None)
        if flag is not None:
            return bool(flag)
        return False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        self._inner.connect()
        self._audit("connect", hardware=self._hw)
        if self._auto_tick and self._thread is None:
            self._stop_evt = threading.Event()
            self._thread = threading.Thread(
                target=self._watchdog_loop,
                args=(self._stop_evt,),
                name="ohho-safety-watchdog",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def disconnect(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            was_armed = self._armed
            self._armed = False
            self._zero_inner()
            if was_armed:
                self._audit("disarm", reason="disconnect")
        self._inner.disconnect()

    def status(self) -> TransportStatus:
        s = self._inner.status()
        if self._latched:
            mode = f"latched ({self._latch_reason})"
        else:
            mode = "armed" if self._armed else "disarmed"
        return dataclasses.replace(s, label=f"{s.label} · safety: {mode}")

    def read(self) -> Telemetry:
        return self._inner.read()

    # ── arming ────────────────────────────────────────────────────────────────
    def arm(self) -> bool:
        """Enable motion. On hardware: env var + TTY + typed phrase, else refused."""
        with self._lock:
            if self._latched:
                return self._refuse("arm_refused", "estop_latched")
            if self._armed:
                return True
            if self._inner.status().state != ConnectionState.CONNECTED:
                return self._refuse("arm_refused", "not_connected")
        if self._hw:
            reason = self._human_check(
                ARM_PHRASE,
                f"\n[ohho safety] Arming {self.robot_id} over {self.protocol}. "
                f"It will accept motion commands.\nType exactly: {ARM_PHRASE}\n> ",
                require_env=True,
            )
            if reason:
                return self._refuse("arm_refused", reason)
        with self._lock:
            if self._latched:
                return self._refuse("arm_refused", "estop_latched")
            now = self._clock()
            if self._state_stale(now):
                return self._refuse("arm_refused", "no_fresh_state")
            if not self._audit("arm", hardware=self._hw, **_operator()):
                return False
            self._armed = True
            self._last_cmd_t = now
            self._last_sent = _ZERO
            self._moving = False
        log.warning("safety: %s ARMED (%s)", self.robot_id, self.protocol)
        return True

    def disarm(self, reason: str = "operator") -> None:
        """Refuse further motion and send zero velocity."""
        with self._lock:
            self._armed = False
            self._zero_inner()
            self._audit("disarm", reason=reason)

    def _human_check(self, phrase: str, prompt: str, *, require_env: bool) -> str:
        """'' when a human confirmed at a TTY; otherwise the refusal reason."""
        if require_env and self._env.get(ARM_ENV) != "1":
            return "env"
        if _in_ci(self._env):
            return "ci"
        if not self._isatty():
            return "no_tty"
        try:
            answer = self._input(prompt)
        except (EOFError, KeyboardInterrupt, OSError):
            return "not_confirmed"
        if _norm_phrase(answer) != _norm_phrase(phrase):
            return "not_confirmed"
        return ""

    def _refuse(self, event: str, reason: str) -> bool:
        log.warning("safety: %s %s: %s", self.robot_id, event, reason)
        self._audit(event, reason=reason)
        return False

    # ── e-stop ────────────────────────────────────────────────────────────────
    def emergency_stop(self, reason: str = "operator") -> None:
        """Latch the e-stop. Stays latched until :meth:`release_stop`.

        Ground robots stop, then damp only if already lying down (or
        ``estop_damp`` is set). Aircraft reuse ``ohho.drone.estop``: land in
        the air, hold then land when the landed state is unknown, and disarm
        only on the ground. Crazyflie cuts motors. The sim backend descends.
        Force-disarm and flight termination are never sent.
        """
        self._trip(reason)

    def release_stop(self) -> bool:
        """Clear a latched e-stop. Needs a human at a TTY on hardware, refuses
        while the robot is still tilted/faulted, and leaves the gate disarmed.

        On an aircraft this clears the Ohho latch only. It does not arm the
        autopilot and it does not resume offboard.
        """
        with self._lock:
            if not self._latched:
                return True
        if self._hw:
            reason = self._human_check(
                RELEASE_PHRASE,
                f"\n[ohho safety] Release the e-stop on {self.robot_id}? "
                f'Type "{RELEASE_PHRASE}"\n> ',
                require_env=False,
            )
            if reason:
                return self._refuse("release_refused", reason)
        with self._lock:
            if self._latest is not None and self._fault(self._latest) is not None:
                return self._refuse("release_refused", "unsafe_state")
            if self._profile.aerial:
                # Ohho latch only. The autopilot stays disarmed until a later
                # human arm_flight, and offboard is not resumed.
                self._latched = False
                self._latch_reason = None
                self._armed = False
                self._audit(
                    "estop_release",
                    resumed_offboard=False,
                    autopilot_armed=False,
                    **self._flight_meta(None, None),
                    **_operator(),
                )
                return True
            try:
                self._inner.release_stop()
            except Exception as e:
                log.error("safety: inner release_stop failed: %r", e)
                return self._refuse("release_refused", "transport_error")
            self._latched = False
            self._latch_reason = None
            self._armed = False
            self._audit("estop_release", **_operator())
        return True

    def _trip(self, reason: str, **details: Any) -> None:
        with self._lock:
            self._latched = True
            self._armed = False
            self._latch_reason = reason
            self._moving = False
            self._last_sent = _ZERO
            inner_error = False
            if self._profile.aerial:
                estop_meta: dict[str, Any] = {
                    "flight_request": None,
                    "frame": None,
                    "backend": self._backend_id(),
                    "estop_layer": None,
                }
                try:
                    estop_meta.update(self._dispatch_aerial_estop())
                except Exception as e:
                    inner_error = True
                    log.error("safety: aerial emergency_stop failed: %r", e)
                    self._zero_inner()
                log.error("safety: %s E-STOP latched (%s)", self.robot_id, reason)
                fields = dict(estop_meta)
                fields.update(details)
                self._audit(
                    "estop",
                    reason=reason,
                    inner_error=inner_error,
                    **fields,
                )
                return
            try:
                self._inner.emergency_stop()
            except Exception as e:
                inner_error = True
                log.error("safety: inner emergency_stop failed: %r", e)
                self._zero_inner()
            if self._profile.estop_damp:
                damp_why = "estop_damp"
            elif _is_lying_down(self._latest, self._profile):
                damp_why = "lying_down"
            else:
                damp_why = ""
            sport = getattr(self._inner, "sport_command", None)
            if damp_why and callable(sport):
                try:
                    sport("damp")
                    self._audit("damp", why=damp_why)
                except Exception as e:
                    log.error("safety: damp failed: %r", e)
            log.error("safety: %s E-STOP latched (%s)", self.robot_id, reason)
            self._audit("estop", reason=reason, inner_error=inner_error, **details)

    # ── state + watchdogs ─────────────────────────────────────────────────────
    def _fault(self, t: Telemetry) -> Optional[tuple[str, dict]]:
        c = t.custom
        if "error_code" in c:
            try:
                code = int(c["error_code"])
            except (TypeError, ValueError):
                return "fault", {"error_code": repr(c["error_code"])}
            if code != 0:
                return "fault", {"error_code": code}
        if self._profile.tilt_guard:
            o = _orientation(t)
            if o is not None:
                lim = self._profile.tilt_limit_rad
                if not (abs(o[0]) <= lim and abs(o[1]) <= lim):
                    return "tilt", {"roll": o[0], "pitch": o[1]}
        return None

    def _on_inner_tele(self, t: Telemetry) -> None:
        with self._lock:
            self._latest = t
            fault = self._fault(t)
            if fault is not None:
                if not self._latched:
                    self._trip(fault[0], **fault[1])
            elif not self._profile.require_orientation or _orientation(t) is not None:
                self._last_state_t = self._clock()
        self._emit_telemetry(t)

    def _state_stale(self, now: float) -> bool:
        last = self._last_state_t
        return last is None or now - last >= self._profile.state_timeout_s

    def _trip_state_watchdog(self, now: float) -> None:
        stale = None if self._last_state_t is None else now - self._last_state_t
        self._audit("watchdog_trip", kind="state", stale_s=stale)
        self._trip("watchdog_state", stale_s=stale)

    def tick(self) -> None:
        """Run the watchdogs once (the background thread calls this ~20 Hz)."""
        with self._lock:
            if self._latched or not self._armed:
                return
            now = self._clock()
            if self._state_stale(now):
                self._trip_state_watchdog(now)
                return
            idle = now - self._last_cmd_t
            if self._moving and idle >= self._profile.cmd_timeout_s:
                self._audit("watchdog_trip", kind="command", idle_s=idle)
                self._zero_inner()

    def _watchdog_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self._tick_period):
            try:
                self.tick()
            except Exception as e:
                log.error("safety: watchdog error: %r", e)
                self._trip("watchdog_error", error=repr(e))

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _audit(self, event: str, **fields: Any) -> bool:
        ok = self._audit_log.write(
            event, robot=self.robot_id, protocol=self.protocol, **fields
        )
        if not ok and not self._audit_failing:
            self._audit_failing = True
            try:
                with self._lock:
                    self._armed = False
                    self._zero_inner()
            finally:
                self._audit_failing = False
        return ok

    def _zero_inner(self) -> None:
        """Best-effort zero velocity; never raises."""
        self._last_sent = _ZERO
        self._moving = False
        try:
            self._inner.send_velocity(Velocity())
        except Exception as e:
            log.error("safety: zero-velocity send failed: %r", e)

    def _reject(self, command: str, reason: str, **details: Any) -> bool:
        if self._profile.aerial:
            self.last_error = reason
        log.warning("safety: refused %s %s: %s", command, details, reason)
        self._audit("reject", command=command, reason=reason, **details)
        self._zero_inner()
        return False

    def _forward(self, fn: Callable[[], None], command: str) -> bool:
        try:
            fn()
        except Exception as e:
            log.error("safety: %s failed: %r", command, e)
            self._trip("transport_error", command=command, error=repr(e))
            return False
        return True

    def _motion_allowed(self, command: str, **details: Any) -> bool:
        if self._latched:
            return self._reject(command, "estop_latched", **details)
        if not self._armed:
            return self._reject(command, "not_armed", **details)
        now = self._clock()
        if self._state_stale(now):
            self._trip_state_watchdog(now)
            return False
        return True

    # ── commands ──────────────────────────────────────────────────────────────
    def send_velocity(self, vel: Velocity) -> None:
        self._velocity(vel, "send_velocity")

    def _velocity(self, vel: Velocity, command: str) -> bool:
        req = (vel.linear_x, vel.linear_y, vel.angular_z)
        with self._lock:
            if not _finite(*req):
                return self._reject(command, "non_finite", requested=req)
            if req == _ZERO:
                self._last_cmd_t = self._clock()
                return self._forward(lambda: self._zero_or_raise(), command)
            if not self._motion_allowed(command, requested=req):
                return False
            now = self._clock()
            p = self._profile
            capped = (
                clamp(req[0], -p.max_vx, p.max_vx),
                clamp(req[1], -p.max_vy, p.max_vy),
                clamp(req[2], -p.max_wz, p.max_wz),
            )
            if p.aerial and p.max_horizontal > 0.0:
                hx, hy = clip_horizontal(capped[0], capped[1], p.max_horizontal)
                capped = (hx, hy, capped[2])
            dt = min(max(now - self._last_cmd_t, 0.0), p.cmd_timeout_s)
            accels = (p.max_accel_x, p.max_accel_y, p.max_accel_wz)
            applied = tuple(
                _accel_step(prev, tgt, a * dt)
                for prev, tgt, a in zip(self._last_sent, capped, accels)
            )
            reasons = []
            if capped != req:
                reasons.append("velocity_cap")
            if applied != capped:
                reasons.append("accel_cap")
            if reasons and not self._audit(
                "clamp",
                command=command,
                requested=req,
                applied=applied,
                reasons=reasons,
            ):
                return False
            if not self._forward(
                lambda: self._inner.send_velocity(Velocity(*applied)), command
            ):
                return False
            self._last_sent = applied
            self._last_cmd_t = now
            self._moving = applied != _ZERO
            return True

    def _zero_or_raise(self) -> None:
        self._last_sent = _ZERO
        self._moving = False
        self._inner.send_velocity(Velocity())

    def send_joint_command(self, name: str, position: float) -> None:
        cmd = "send_joint_command"
        with self._lock:
            if not _finite(position):
                self._reject(cmd, "non_finite", joint=name, requested=position)
                return
            if not self._motion_allowed(cmd, joint=name, requested=position):
                return
            p = self._profile
            if self.spec is not None and self.spec.joint_names:
                if name not in self.spec.joint_names:
                    self._reject(cmd, "unknown_joint", joint=name)
                    return
            limits = p.joint_limits.get(name)
            if limits is None:
                self._reject(cmd, "no_joint_limit", joint=name)
                return
            ref = self._joint_reference(name)
            if ref is None:
                self._reject(cmd, "no_joint_reference", joint=name)
                return
            target = clamp(position, limits[0], limits[1])
            applied = clamp(target, ref - p.max_joint_step, ref + p.max_joint_step)
            reasons = []
            if target != position:
                reasons.append("joint_limit")
            if applied != target:
                reasons.append("joint_step")
            if reasons and not self._audit(
                "clamp",
                command=cmd,
                joint=name,
                requested=position,
                applied=applied,
                reasons=reasons,
            ):
                return
            if self._forward(
                lambda: self._inner.send_joint_command(name, applied), cmd
            ):
                self._joint_cmds[name] = applied

    def _joint_reference(self, name: str) -> Optional[float]:
        if self._latest is not None:
            for j in self._latest.joints:
                if j.name == name:
                    return j.position
        return self._joint_cmds.get(name)

    def send_joint_effort(self, name: str, effort: float) -> None:
        """Effort/torque command, clamped to ``profile.effort_limits[name]``.
        Refused when the joint has no limit or the transport has no effort path."""
        cmd = "send_joint_effort"
        with self._lock:
            if not _finite(effort):
                self._reject(cmd, "non_finite", joint=name, requested=effort)
                return
            if not self._motion_allowed(cmd, joint=name, requested=effort):
                return
            fn = getattr(self._inner, "send_joint_effort", None)
            if not callable(fn):
                self._reject(cmd, "effort_unsupported", joint=name)
                return
            limit = self._profile.effort_limits.get(name)
            if limit is None:
                self._reject(cmd, "no_effort_limit", joint=name)
                return
            applied = clamp(effort, -limit, limit)
            if applied != effort and not self._audit(
                "clamp",
                command=cmd,
                joint=name,
                requested=effort,
                applied=applied,
                reasons=["effort_cap"],
            ):
                return
            self._forward(lambda: fn(name, applied), cmd)

    def command(self, action: Any, *args: Any) -> bool:
        """Run a high-level (sport-mode) action if it is allowlisted.

        ``stop_move`` is always accepted. ``move`` goes through the velocity
        caps; ``euler`` angles are clamped to ``profile.euler_limit_rad``.
        Aerial profiles dispatch flight requests instead of sport actions.
        Returns True when the action was forwarded.
        """
        cmd = "command"
        with self._lock:
            if not isinstance(action, str) or not action.strip():
                return self._reject(cmd, "invalid_action", action=repr(action))
            a = normalize_action(action)
            if is_denied(a):
                return self._reject(cmd, "denied", action=a)
            if self._profile.aerial:
                return self._aerial_command(a, args)
            if a not in self._profile.allowlist:
                return self._reject(cmd, "not_allowlisted", action=a)
            if a == "stop_move":
                if args:
                    return self._reject(cmd, "bad_arguments", action=a)
                self._zero_inner()
                sport = getattr(self._inner, "sport_command", None)
                if callable(sport):
                    try:
                        sport("stop_move")
                    except Exception as e:
                        log.error("safety: stop_move failed: %r", e)
                return True
            if a == "move":
                nums = _floats(args, 3)
                if nums is None:
                    return self._reject(cmd, "bad_arguments", action=a)
                return self._velocity(Velocity(*nums), "command:move")
            if not self._motion_allowed(cmd, action=a):
                return False
            if a == "euler":
                nums = _floats(args, 3, finite=True)
                if nums is None:
                    return self._reject(cmd, "bad_arguments", action=a)
                lim = self._profile.euler_limit_rad
                applied = tuple(clamp(v, -lim, lim) for v in nums)
                if applied != nums and not self._audit(
                    "clamp",
                    command="euler",
                    requested=nums,
                    applied=applied,
                    reasons=["euler_limit"],
                ):
                    return False
                return self._sport(a, *applied)
            if args:
                return self._reject(cmd, "bad_arguments", action=a)
            return self._sport(a)

    # ── aerial (OHH-288) ──────────────────────────────────────────────────────
    def _backend_id(self) -> str:
        return _transport_backend_id(self._inner)

    def _flight_meta(self, request: Any, frame: Any) -> dict[str, Any]:
        if isinstance(frame, Frame):
            frame_out: Any = frame.value
        elif frame is None:
            frame_out = None
        else:
            frame_out = str(frame)
        return {
            "flight_request": request,
            "frame": frame_out,
            "backend": self._backend_id(),
        }

    def _drone_dict(self) -> dict[str, Any]:
        if self._latest is None:
            return {}
        drone = self._latest.custom.get("drone")
        if isinstance(drone, dict):
            return drone
        return {}

    def _landed_and_alt(self) -> tuple[Any, Any]:
        drone = self._drone_dict()
        state = drone.get("landed_state", "unknown")
        return state, self._altitude_m()

    def _altitude_m(self) -> Any:
        drone = self._drone_dict()
        pose = drone.get("pose")
        frame = drone.get("pose_frame")
        if not isinstance(pose, dict) or not isinstance(frame, str):
            return None
        try:
            return positive_up_m(pose.get("z"), frame)
        except (TypeError, ValueError):
            return None

    def _pose_local(self) -> Optional[tuple[float, float, float, Frame]]:
        drone = self._drone_dict()
        pose = drone.get("pose")
        frame_name = drone.get("pose_frame")
        if not isinstance(pose, dict) or not isinstance(frame_name, str):
            return None
        try:
            frame = Frame(frame_name)
        except ValueError:
            return None
        nums = _floats((pose.get("x"), pose.get("y"), pose.get("z")), 3, finite=True)
        if nums is None:
            return None
        return nums[0], nums[1], nums[2], frame

    def _aerial_reject(
        self, action: str, frame: Any, reason: str, **details: Any
    ) -> bool:
        return self._reject(
            "command",
            reason,
            action=action,
            **self._flight_meta(action, frame),
            **details,
        )

    def _invoke_flight(self, action: str, **fields: Any) -> None:
        fn = getattr(self._inner, "send_flight", None)
        if not callable(fn):
            raise RuntimeError(f"transport has no send_flight for {action}")
        fn(action, **fields)

    def _forward_flight(self, action: str, **fields: Any) -> bool:
        frame = fields.get("frame")
        audit_fields = {
            key: value
            for key, value in fields.items()
            if key != "frame" and not isinstance(value, Frame)
        }
        if not self._audit(
            "flight",
            command=action,
            **self._flight_meta(action, frame),
            **audit_fields,
        ):
            return False
        if not self._forward(
            lambda: self._invoke_flight(action, **fields), f"command:{action}"
        ):
            return False
        if action in {"offboard_position", "offboard_velocity"}:
            self._last_setpoint_t = self._clock()
        self.last_error = None
        return True

    def _reject_geofence(self, action: str, frame: Any, **details: Any) -> bool:
        """Refuse a setpoint that would leave the cylinder, and send hold."""
        self.last_error = "geofence"
        log.error("safety: geofence refused %s %s", action, details)
        self._audit(
            "reject",
            command=action,
            reason="geofence",
            action=action,
            **self._flight_meta(action, frame),
            **details,
        )
        try:
            self._invoke_flight("hold")
        except Exception as e:
            self._trip("transport_error", command="hold", error=repr(e))
        return False

    def _human_tty_reason(self, what: str) -> str:
        """'' when a human is at a TTY. Hardware also needs the arm phrase."""
        if _in_ci(self._env):
            return "ci"
        if not self._isatty():
            return "no_tty"
        if not self._hw:
            return ""
        return self._human_check(
            ARM_PHRASE,
            f"\n[ohho safety] {what} on {self.robot_id} needs a human at this "
            f"keyboard.\nType exactly: {ARM_PHRASE}\n> ",
            require_env=True,
        )

    def _setpoint_alive(self) -> bool:
        last = self._last_setpoint_t
        if last is None:
            return False
        return self._clock() - last < self._profile.cmd_timeout_s

    def _battery_fraction(self) -> Optional[float]:
        samples: list[float] = []
        if self._latest is not None:
            battery = self._latest.battery
            if (
                isinstance(battery, (int, float))
                and not isinstance(battery, bool)
                and math.isfinite(battery)
            ):
                samples.append(float(battery))
        drone_battery = self._drone_dict().get("battery")
        if (
            isinstance(drone_battery, (int, float))
            and not isinstance(drone_battery, bool)
            and math.isfinite(drone_battery)
        ):
            samples.append(float(drone_battery))
        if not samples:
            return None
        return min(samples)

    def _takeoff_block(self) -> Optional[str]:
        battery = self._battery_fraction()
        if battery is None or battery < TAKEOFF_MIN_BATTERY:
            return "battery"
        if self._drone_dict().get("link_ok") is not True:
            return "link"
        return None

    def _takeoff_altitude(self, args: tuple[Any, ...]) -> tuple[Optional[float], str]:
        nums = _floats(args, 1, finite=True)
        if nums is None or nums[0] <= 0.0:
            return None, "bad_arguments"
        return nums[0], ""

    def _altitude_exceeds(self, alt: float) -> bool:
        p = self._profile
        limit = min(p.takeoff_alt_m, p.fence_max_alt_m)
        return alt > limit + _FENCE_EPS

    def _motor_cut(self) -> None:
        fn = getattr(self._inner, "motor_cut", None)
        if callable(fn):
            fn()
            return
        self._inner.emergency_stop()

    def _dispatch_aerial_estop(self) -> dict[str, Any]:
        """Map one Ohho e-stop onto the backend. Reuses ``estop_commands``."""
        backend = self._backend_id()
        if backend in _CRAZYFLIE_BACKENDS:
            self._motor_cut()
            return {
                "backend": backend,
                "flight_request": "motor_cut",
                "frame": None,
                "estop_layer": "crazyflie_motor_cut",
            }
        state, alt = self._landed_and_alt()
        commands = estop_commands(state, alt)
        names: list[str] = []
        for cmd in commands:
            if command_is_forbidden(cmd):
                raise RuntimeError(
                    "refusing to emit force-disarm (param2 21196) or flight termination"
                )
            names.append(cmd.name)
            self._invoke_flight(cmd.name)
        if backend in _SIM_BACKENDS:
            descend = getattr(self._inner, "simulated_descent", None)
            if callable(descend):
                descend()
            layer = "simulated_descent"
        else:
            layer = "ohho_land"
        return {
            "backend": backend,
            "flight_request": names,
            "frame": None,
            "estop_layer": layer,
        }

    def _aerial_command(self, action: str, args: tuple[Any, ...]) -> bool:
        self.last_error = None
        if action in DRONE_TTY_ACTIONS or action == "disarm":
            return self._aerial_privileged(action, args)
        if action not in self._profile.allowlist:
            return self._aerial_reject(action, None, "not_allowlisted")
        if action in {"hold", "land", "rtl"}:
            if args:
                return self._aerial_reject(action, None, "bad_arguments")
            meta = self._flight_meta(action, None)
            if not self._motion_allowed("command", action=action, **meta):
                return False
            return self._forward_flight(action)
        if action == "offboard_position":
            return self._aerial_position(args)
        if action == "offboard_velocity":
            return self._aerial_velocity(args)
        return self._aerial_reject(action, None, "not_allowlisted")

    def _aerial_privileged(self, action: str, args: tuple[Any, ...]) -> bool:
        if action == "disarm":
            return self._aerial_disarm(args)
        canonical = "arm" if action in {"arm", "arm_flight"} else "takeoff"
        altitude: Optional[float] = None
        if canonical == "takeoff":
            altitude, why = self._takeoff_altitude(args)
            if altitude is None:
                return self._aerial_reject(action, None, why)
        elif args:
            return self._aerial_reject(action, None, "bad_arguments")
        meta = self._flight_meta(canonical, None)
        if not self._motion_allowed("command", action=canonical, **meta):
            return False
        human = self._human_tty_reason(canonical)
        if human in {"ci", "no_tty"}:
            decision = _ci_sim_arming(self._inner, self._env, hardware=self._hw)
            if decision.reason == "":
                if not self._audit(
                    "ci_sim_arming",
                    action=action,
                    backend=decision.backend,
                    host=decision.host,
                ):
                    return False
                log.warning(
                    "safety: CI sim-only arming exception used: %s %s on %s "
                    "(backend=%s, uri=%s)",
                    action,
                    self.robot_id,
                    decision.target,
                    decision.backend,
                    decision.uri,
                )
            else:
                return self._aerial_reject(canonical, None, human)
        elif human:
            return self._aerial_reject(canonical, None, human)
        if canonical == "arm":
            if not self._setpoint_alive():
                return self._aerial_reject("arm", None, "no_setpoint_stream")
            return self._forward_flight("arm")
        if altitude is None:
            return self._aerial_reject("takeoff", None, "bad_arguments")
        block = self._takeoff_block()
        if block:
            return self._aerial_reject("takeoff", None, block)
        if self._altitude_exceeds(altitude):
            return self._reject_geofence("takeoff", None, altitude_m=altitude)
        return self._forward_flight("takeoff", alt_m=altitude)

    def _aerial_disarm(self, args: tuple[Any, ...]) -> bool:
        if args:
            return self._aerial_reject("disarm", None, "bad_arguments")
        state, alt = self._landed_and_alt()
        if not disarm_allowed(state, alt):
            return self._aerial_reject(
                "disarm",
                None,
                "not_on_ground",
                landed_state=getattr(state, "value", state),
                altitude_m=alt,
            )
        meta = self._flight_meta("disarm", None)
        if not self._motion_allowed("command", action="disarm", **meta):
            return False
        return self._forward_flight("disarm")

    def _aerial_position(self, args: tuple[Any, ...]) -> bool:
        frame: Any = args[4] if len(args) == 5 else None
        nums = _floats(args[:4], 4, finite=True) if len(args) == 5 else None
        if nums is None or not isinstance(frame, Frame):
            return self._aerial_reject(
                "offboard_position",
                frame if isinstance(frame, Frame) else None,
                "bad_arguments",
            )
        meta = self._flight_meta("offboard_position", frame)
        if not self._motion_allowed("command", action="offboard_position", **meta):
            return False
        x, y, z, yaw = nums
        violation = fence_violation(x, y, z, frame, self._profile)
        if violation is not None:
            return self._reject_geofence(
                "offboard_position",
                frame,
                violation=violation,
                requested=[x, y, z, yaw],
            )
        return self._forward_flight(
            "offboard_position", x=x, y=y, z=z, yaw=yaw, frame=frame
        )

    def _aerial_velocity(self, args: tuple[Any, ...]) -> bool:
        frame: Any = args[4] if len(args) == 5 else None
        nums = _floats(args[:4], 4, finite=True) if len(args) == 5 else None
        if nums is None or not isinstance(frame, Frame):
            return self._aerial_reject(
                "offboard_velocity",
                frame if isinstance(frame, Frame) else None,
                "bad_arguments",
            )
        meta = self._flight_meta("offboard_velocity", frame)
        if not self._motion_allowed("command", action="offboard_velocity", **meta):
            return False
        pose = self._pose_local()
        if pose is not None:
            violation = fence_violation(
                pose[0], pose[1], pose[2], pose[3], self._profile
            )
            if violation is not None:
                return self._reject_geofence(
                    "offboard_velocity",
                    frame,
                    violation=violation,
                    requested=list(nums),
                )
        p = self._profile
        vx, vy = clip_horizontal(nums[0], nums[1], p.max_horizontal)
        vz = clamp(nums[2], -p.max_vz, p.max_vz)
        yaw_rate = clamp(nums[3], -p.max_wz, p.max_wz)
        applied = (vx, vy, vz, yaw_rate)
        if applied != nums and not self._audit(
            "clamp",
            command="offboard_velocity",
            requested=list(nums),
            applied=list(applied),
            reasons=["velocity_cap"],
            **meta,
        ):
            return False
        return self._forward_flight(
            "offboard_velocity",
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            frame=frame,
        )

    def _sport(self, action: str, *args: float) -> bool:
        fn = getattr(self._inner, "sport_command", None)
        if not callable(fn):
            return self._reject("command", "unsupported_by_transport", action=action)
        return self._forward(lambda: fn(action, *args), f"command:{action}")


__all__ = [
    "ARM_ENV",
    "ARM_PHRASE",
    "AUDIT_DIR_ENV",
    "AuditLog",
    "CI_ENV_VARS",
    "CI_SIM_ARMING_ENV",
    "CI_SITL_ENV",
    "CONFIG_ENV",
    "DRONE_ACTIONS",
    "DRONE_FENCE_ALT_M",
    "DRONE_FENCE_RADIUS_M",
    "DRONE_MAX_HORIZONTAL",
    "DRONE_MAX_VERTICAL",
    "DRONE_MAX_YAW",
    "DRONE_MCP_DENIED",
    "DRONE_PROFILE",
    "DRONE_TAKEOFF_ALT_M",
    "DRONE_TTY_ACTIONS",
    "GO2_PROFILE",
    "HARD_MAX_ANG",
    "HARD_MAX_LIN",
    "OMNIBOT_PROFILE",
    "PROFILES",
    "RELEASE_PHRASE",
    "SafetyConfig",
    "SafetyGate",
    "SafetyProfile",
    "TAKEOFF_MIN_BATTERY",
    "ci_sim_arming_reason",
    "clip_horizontal",
    "default_audit_dir",
    "drone_mcp_allowed",
    "fence_violation",
    "is_denied",
    "load_safety_config",
    "normalize_action",
    "profile_for",
    "requires_hardware_arming",
    "resolve_config",
]
