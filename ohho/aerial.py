"""Aerial frames and the conversions flight backends share.

Experimental: nothing has flown. This module does not arm a vehicle, open a
link, or talk to a flight controller. Callers name a :class:`Frame` on every
conversion; there is no default that treats ENU numbers as NED.
"""

from __future__ import annotations

import math
from enum import Enum


class Frame(str, Enum):
    """Coordinate frame for an aerial setpoint.

    * ``LOCAL_NED`` — x north, y east, z down. Positive vz descends.
      This is the PX4 ``TrajectorySetpoint`` local frame.
    * ``LOCAL_ENU`` — x east, y north, z up. ROS local convention.
    * ``BODY_FRD`` — x forward, y right, z down. PX4 / MAVLink body frame
      (``MAV_FRAME_BODY_NED``). Velocity teleop uses this frame.

    Every setpoint names one of these. Omitting the frame is a ``TypeError``.
    A bare string such as ``"LOCAL_NED"`` is a ``TypeError`` as well.
    """

    LOCAL_NED = "LOCAL_NED"
    LOCAL_ENU = "LOCAL_ENU"
    BODY_FRD = "BODY_FRD"


_LOCAL = frozenset({Frame.LOCAL_NED, Frame.LOCAL_ENU})


def require_frame(frame: object, *, name: str = "frame") -> Frame:
    """Return ``frame`` when it is a :class:`Frame`.

    ``name`` is the parameter name used in the error. It is not a coordinate
    frame, and it does not select ``LOCAL_NED`` or any other default.
    """
    if isinstance(frame, Frame):
        return frame
    raise TypeError(
        f"{name} is required and must be ohho.aerial.Frame "
        "(LOCAL_NED, LOCAL_ENU, or BODY_FRD), "
        f"not {type(frame).__name__}"
    )


def finite_number(name: str, value: object) -> float:
    """Finite ``int`` or ``float``. Booleans, NaN, and infinities are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _local_pair(source: Frame, target: Frame) -> None:
    if source is target:
        return
    if source in _LOCAL and target in _LOCAL:
        return
    raise ValueError(
        f"cannot convert {source.value} to {target.value} without an attitude; "
        "BODY_FRD stays in the body frame"
    )


def convert_vector(
    x: float,
    y: float,
    z: float,
    *,
    source: Frame,
    target: Frame,
) -> tuple[float, float, float]:
    """Convert a position or linear velocity between local frames.

    ``LOCAL_ENU`` (east, north, up) and ``LOCAL_NED`` (north, east, down) swap
    the horizontal axes and negate z. That map is its own inverse, so one
    function converts either way. ``BODY_FRD`` converts only to itself: a body
    vector needs the vehicle attitude before it is a local vector.
    """
    source = require_frame(source, name="source")
    target = require_frame(target, name="target")
    x = finite_number("x", x)
    y = finite_number("y", y)
    z = finite_number("z", z)
    if source is target:
        return (x, y, z)
    _local_pair(source, target)
    return (y, x, -z)


def convert_yaw(yaw: float, *, source: Frame, target: Frame) -> float:
    """Convert a heading between ``LOCAL_NED`` and ``LOCAL_ENU``.

    NED yaw is zero facing north and positive clockwise from above (nose
    right). ENU yaw is zero facing east and positive counter-clockwise from
    above. ``yaw_enu = pi/2 - yaw_ned``, and the same expression converts
    back. The result is not wrapped into an interval.
    """
    source = require_frame(source, name="source")
    target = require_frame(target, name="target")
    yaw = finite_number("yaw", yaw)
    if source is target:
        return yaw
    _local_pair(source, target)
    return math.pi / 2.0 - yaw


def convert_yaw_rate(yaw_rate: float, *, source: Frame, target: Frame) -> float:
    """Convert a yaw rate between ``LOCAL_NED`` and ``LOCAL_ENU``.

    Positive NED yaw rate is nose-right. Positive ENU yaw rate is nose-left,
    so the sign flips. A ``BODY_FRD`` yaw rate is nose-right about the body
    z axis and is left unchanged only when the source and target are both
    ``BODY_FRD``.
    """
    source = require_frame(source, name="source")
    target = require_frame(target, name="target")
    yaw_rate = finite_number("yaw_rate", yaw_rate)
    if source is target:
        return yaw_rate
    _local_pair(source, target)
    return -yaw_rate


def body_z_from_climb(climb: float) -> float:
    """``BODY_FRD`` z for a positive-up climb request.

    Body z points down, so stick-up (positive ``climb``) is a negative body z.
    """
    return -finite_number("climb", climb)


def stick_to_body_frd(
    forward: float,
    right: float,
    climb: float,
    yaw_right: float,
) -> tuple[float, float, float, float]:
    """Mode-2 sticks to ``(vx, vy, vz, yaw_rate)`` in ``BODY_FRD``.

    ``forward`` is body +x. ``right`` is body +y. ``climb`` positive (stick
    up) becomes negative vz via :func:`body_z_from_climb`. ``yaw_right``
    positive is nose-right and stays a positive yaw rate: a mode-2 radio's
    left stick, rightward, and the VR aerial scheme both use that sign.
    The result is already in ``BODY_FRD``; this function does not take a frame.
    """
    return (
        finite_number("forward", forward),
        finite_number("right", right),
        body_z_from_climb(climb),
        finite_number("yaw_right", yaw_right),
    )


__all__ = [
    "Frame",
    "body_z_from_climb",
    "convert_vector",
    "convert_yaw",
    "convert_yaw_rate",
    "finite_number",
    "require_frame",
    "stick_to_body_frd",
]
