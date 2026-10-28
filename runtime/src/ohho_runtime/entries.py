"""Declared start targets for class-default services.

The supervisor calls these later. Calling one here does not connect to a
robot, send a command, or arm.
"""

from __future__ import annotations


def safety_watchdog() -> None:
    """Start target for the safety-watchdog role."""
    return None


def transport() -> None:
    """Start target for the transport role."""
    return None


def telemetry() -> None:
    """Start target for the telemetry role."""
    return None


def teleop_gateway() -> None:
    """Start target for the teleop-gateway role."""
    return None


def autopilot_heartbeat() -> None:
    """Start target for the aerial autopilot-heartbeat role."""
    return None


def mode_monitor() -> None:
    """Start target for the humanoid mode-monitor role."""
    return None


__all__ = [
    "autopilot_heartbeat",
    "mode_monitor",
    "safety_watchdog",
    "telemetry",
    "teleop_gateway",
    "transport",
]
