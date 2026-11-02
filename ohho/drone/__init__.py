"""Experimental drone telemetry and the e-stop decision table (OHH-285).

Not flight-ready. This package does not encode MAVLink and does not open a
socket. OHH-286 adds the aerial capability and the frame enum; pose frames
here are the plain strings ``LOCAL_NED`` and ``LOCAL_ENU``.
"""

from __future__ import annotations

from .estop import (
    DISARM_ALTITUDE_M,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_FLIGHTTERMINATION,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_NAV_LAND,
    MAV_FORCE_DISARM_MAGIC,
    EstopCommand,
    command_is_forbidden,
    estop_commands,
)
from .mock_link import (
    MockCommandLog,
    SimulationTarget,
    simulation_refusal_reason,
    simulation_target_allowed,
)
from .telemetry import (
    BODY_FRD,
    DRONE_TELEMETRY_KEYS,
    LOCAL_ENU,
    LOCAL_NED,
    DronePose,
    DroneTelemetry,
    DroneVelocity,
    FlightMode,
    GpsFix,
    LandedState,
    coerce_landed_state,
    gps_fix_from_mav,
    landed_state_from_mav,
    positive_up_m,
)

__all__ = [
    "BODY_FRD",
    "DISARM_ALTITUDE_M",
    "DRONE_TELEMETRY_KEYS",
    "LOCAL_ENU",
    "LOCAL_NED",
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_FLIGHTTERMINATION",
    "MAV_CMD_DO_SET_MODE",
    "MAV_CMD_NAV_LAND",
    "MAV_FORCE_DISARM_MAGIC",
    "DronePose",
    "DroneTelemetry",
    "DroneVelocity",
    "EstopCommand",
    "FlightMode",
    "GpsFix",
    "LandedState",
    "MockCommandLog",
    "SimulationTarget",
    "coerce_landed_state",
    "command_is_forbidden",
    "estop_commands",
    "gps_fix_from_mav",
    "landed_state_from_mav",
    "positive_up_m",
    "simulation_refusal_reason",
    "simulation_target_allowed",
]
