"""Pure e-stop decision for an aerial vehicle (OHH-285).

No I/O. Given a landed state and a positive-up altitude, return the ordered
command list a backend should send:

* ``in_air``, ``takeoff``, ``landing`` → land
* ``unknown`` (and any unrecognised report) → hold, then land
* disarm only when ``landed_state`` is ``on_ground`` and altitude is a finite
  number of metres ``<= 0.25``

``unknown`` is never on the ground, including when the altitude is high.
A missing or non-finite altitude does not count as ``<= 0.25``. An
``on_ground`` flag that disagrees with a higher altitude lands, and does
not disarm.

Commands are the backend-neutral names ``land``, ``hold`` and ``disarm``.
MAVLink command ids sit on the same objects for a later backend. This module
does not encode a frame. It never emits force-disarm (param2 ``21196``) and
never emits flight termination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .telemetry import LandedState, coerce_landed_state

# Metres. Both the landed-state flag and a finite altitude at or below this
# value are required before a disarm is even a candidate.
DISARM_ALTITUDE_M = 0.25

# MAVLink command ids for later backends. Not an encoder.
MAV_CMD_NAV_LAND = 21
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_FLIGHTTERMINATION = 185  # never emitted
MAV_CMD_COMPONENT_ARM_DISARM = 400

# PX4 treats param2=21196 on MAV_CMD_COMPONENT_ARM_DISARM as force-disarm.
# This decision never uses that value.
MAV_FORCE_DISARM_MAGIC = 21196

MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
PX4_CUSTOM_MAIN_MODE_AUTO = 4
PX4_CUSTOM_SUB_MODE_AUTO_LOITER = 3  # HOLD


def _px4_custom_mode(main_mode: int, sub_mode: int) -> int:
    """Pack a PX4 main mode and sub mode the way PX4 reads custom_mode."""
    return ((sub_mode & 0xFF) << 24) | ((main_mode & 0xFF) << 16)


PX4_MODE_AUTO_LOITER = _px4_custom_mode(
    PX4_CUSTOM_MAIN_MODE_AUTO, PX4_CUSTOM_SUB_MODE_AUTO_LOITER
)

_AIRBORNE = frozenset({LandedState.IN_AIR, LandedState.TAKEOFF, LandedState.LANDING})
_FORBIDDEN_NAMES = frozenset(
    {
        "flight_termination",
        "terminate",
        "kill",
        "force_disarm",
        "motor_kill",
    }
)
_FORCE_DISARM_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class EstopCommand:
    """One backend-neutral e-stop step, plus the MAVLink id a later backend may send."""

    name: str
    mav_command: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0


LAND_COMMAND = EstopCommand(name="land", mav_command=MAV_CMD_NAV_LAND)
HOLD_COMMAND = EstopCommand(
    name="hold",
    mav_command=MAV_CMD_DO_SET_MODE,
    param1=float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
    param2=float(PX4_MODE_AUTO_LOITER),
    param3=float(PX4_CUSTOM_SUB_MODE_AUTO_LOITER),
)
# Normal disarm. param1=0 disarms, param2=0 is not the force-disarm magic.
DISARM_COMMAND = EstopCommand(
    name="disarm",
    mav_command=MAV_CMD_COMPONENT_ARM_DISARM,
    param1=0.0,
    param2=0.0,
)


def _normalized_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _is_force_magic(value: object) -> bool:
    """True when ``value`` is the force-disarm magic, or not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    if not math.isfinite(value):
        return True
    return abs(float(value) - MAV_FORCE_DISARM_MAGIC) < _FORCE_DISARM_TOLERANCE


def command_is_forbidden(command: EstopCommand) -> bool:
    """True for flight termination, force-disarm, or a disarm that is not param2=0."""
    if not isinstance(command, EstopCommand):
        return True
    if command.mav_command in (MAV_CMD_DO_FLIGHTTERMINATION, MAV_FORCE_DISARM_MAGIC):
        return True
    if _normalized_name(command.name) in _FORBIDDEN_NAMES:
        return True
    if any(
        _is_force_magic(value)
        for value in (command.param1, command.param2, command.param3)
    ):
        return True
    if _normalized_name(command.name) == "disarm" and (
        command.param1 != 0.0 or command.param2 != 0.0
    ):
        return True
    if (
        command.mav_command == MAV_CMD_COMPONENT_ARM_DISARM
        and command.param1 == 0.0
        and command.param2 != 0.0
    ):
        return True
    return False


def _refuse_if_forbidden(
    commands: tuple[EstopCommand, ...],
) -> tuple[EstopCommand, ...]:
    for command in commands:
        if command_is_forbidden(command):
            raise RuntimeError(
                "refusing to emit force-disarm (param2 21196) or flight termination"
            )
    return commands


def _altitude_allows_disarm(altitude_m: object) -> bool:
    """True only for a finite altitude at or below ``DISARM_ALTITUDE_M``.

    Missing, boolean, non-numeric and non-finite samples do not pass. They
    are not treated as zero.
    """
    if isinstance(altitude_m, bool) or not isinstance(altitude_m, (int, float)):
        return False
    if not math.isfinite(altitude_m):
        return False
    return float(altitude_m) <= DISARM_ALTITUDE_M


def disarm_allowed(landed_state: object, altitude_m: object) -> bool:
    """True only for ``on_ground`` with a finite altitude ``<= 0.25`` m."""
    if coerce_landed_state(landed_state) != LandedState.ON_GROUND:
        return False
    return _altitude_allows_disarm(altitude_m)


def is_disarm_command(command: EstopCommand) -> bool:
    """True for a normal disarm. Force-disarm is forbidden and not counted as normal."""
    if command_is_forbidden(command):
        return False
    return command.name == "disarm" or (
        command.mav_command == MAV_CMD_COMPONENT_ARM_DISARM and command.param1 == 0.0
    )


def estop_commands(
    landed_state: object,
    altitude_m: object = None,
) -> tuple[EstopCommand, ...]:
    """Ordered e-stop commands for this state and positive-up altitude.

    ``altitude_m`` defaults to missing, which does not authorize a disarm.
    The return value is the whole decision: callers send every command, in
    order. This function does not talk to a vehicle.
    """
    state = coerce_landed_state(landed_state, strict=False)
    commands: tuple[EstopCommand, ...]
    if state in _AIRBORNE:
        commands = (LAND_COMMAND,)
    elif state == LandedState.ON_GROUND and _altitude_allows_disarm(altitude_m):
        commands = (DISARM_COMMAND,)
    elif state == LandedState.ON_GROUND:
        commands = (LAND_COMMAND,)
    else:
        # unknown, missing, or unrecognised: never a disarm
        commands = (HOLD_COMMAND, LAND_COMMAND)
    return _refuse_if_forbidden(commands)


_refuse_if_forbidden((LAND_COMMAND, HOLD_COMMAND, DISARM_COMMAND))


__all__ = [
    "DISARM_ALTITUDE_M",
    "DISARM_COMMAND",
    "HOLD_COMMAND",
    "LAND_COMMAND",
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_FLIGHTTERMINATION",
    "MAV_CMD_DO_SET_MODE",
    "MAV_CMD_NAV_LAND",
    "MAV_FORCE_DISARM_MAGIC",
    "MAV_MODE_FLAG_CUSTOM_MODE_ENABLED",
    "PX4_CUSTOM_MAIN_MODE_AUTO",
    "PX4_CUSTOM_SUB_MODE_AUTO_LOITER",
    "PX4_MODE_AUTO_LOITER",
    "EstopCommand",
    "command_is_forbidden",
    "disarm_allowed",
    "estop_commands",
    "is_disarm_command",
]
