"""OhhO OS — the open-source, robot-agnostic engine for any robot.

Public surface:

    from ohho import Robot, connect
    bot = Robot.connect("omnibot")
"""

from __future__ import annotations

__version__ = "1.1.3"

from .schema import (
    ConnectionState,
    JointReading,
    Odometry,
    Scan,
    Telemetry,
    TransportStatus,
    Velocity,
)
from .robot import Robot, connect
from .config import RobotProfile, load_robot_profile, resolve_robot_id
from .session import RobotSession
from .registry import RobotSpec, get_spec, list_specs, register

__all__ = [
    "__version__",
    "Robot",
    "connect",
    "RobotSpec",
    "get_spec",
    "list_specs",
    "register",
    "RobotProfile",
    "load_robot_profile",
    "resolve_robot_id",
    "RobotSession",
    "Velocity",
    "Odometry",
    "JointReading",
    "Scan",
    "Telemetry",
    "TransportStatus",
    "ConnectionState",
]
