"""Headless stand, walk, turn and stop against a connected sim transport.

The transport is the generic humanoid-mujoco backend. This module only
supplies the scripted commands and the measurements. It does not open a
viewer and it does not start the live thread: the caller steps the model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ohho.schema import Velocity


@dataclass(frozen=True)
class WalkResult:
    """Measurements from one headless run. Distances are metres."""

    min_height_m: float
    stand_height_m: float
    walked_m: float
    yaw_change_rad: float
    stop_speed_m_s: float
    stayed_up: bool


def run_walk_check(
    transport: object,
    *,
    stand_s: float = 30.0,
    speed_m_s: float = 0.3,
    distance_m: float = 1.0,
    yaw_rate: float = 0.5,
    yaw_change_rad: float = 0.5,
    chunk_s: float = 0.2,
) -> WalkResult:
    """Stand, walk about ``distance_m``, turn, then stop.

    ``stand_s`` is simulated time with a zero command, so the locomotion
    policy is holding balance. Walking uses ``speed_m_s`` forward until the
    horizontal displacement reaches ``distance_m`` (or 20 s). The turn holds
    ``yaw_rate`` until the yaw change reaches ``yaw_change_rad`` (or 8 s).
    Stop speed is the horizontal speed over the half-second after a 2 s
    settle at zero command.
    """
    send_mode: Callable[..., object] = getattr(transport, "send_humanoid")
    send_velocity: Callable[[Velocity], object] = getattr(transport, "send_velocity")
    step: Callable[[float], object] = getattr(transport, "step")
    send_mode("walk_mode")
    send_velocity(Velocity())
    heights = _Heights(transport)
    _advance(transport, step, stand_s, chunk_s, heights)
    stand_height = _height(transport)
    origin = _xy(transport)
    send_velocity(Velocity(speed_m_s, 0.0, 0.0))
    walked = 0.0
    elapsed = 0.0
    while walked < distance_m and elapsed < 20.0:
        step(chunk_s)
        elapsed += chunk_s
        heights.sample()
        pos = _xy(transport)
        walked = math.hypot(pos[0] - origin[0], pos[1] - origin[1])
    yaw_start = _yaw(transport)
    send_velocity(Velocity(0.0, 0.0, yaw_rate))
    turned = 0.0
    elapsed = 0.0
    while abs(turned) < yaw_change_rad and elapsed < 8.0:
        step(chunk_s)
        elapsed += chunk_s
        heights.sample()
        turned = _wrap(_yaw(transport) - yaw_start)
    send_velocity(Velocity())
    _advance(transport, step, 2.0, chunk_s, heights)
    stopped = _xy(transport)
    step(0.5)
    heights.sample()
    after = _xy(transport)
    stop_speed = math.hypot(after[0] - stopped[0], after[1] - stopped[1]) / 0.5
    return WalkResult(
        min_height_m=heights.minimum,
        stand_height_m=stand_height,
        walked_m=walked,
        yaw_change_rad=turned,
        stop_speed_m_s=stop_speed,
        stayed_up=heights.stayed_up,
    )


class _Heights:
    def __init__(self, transport: object) -> None:
        self._transport = transport
        self.minimum = _height(transport)
        self.stayed_up = _is_up(transport)

    def sample(self) -> None:
        self.minimum = min(self.minimum, _height(self._transport))
        if not _is_up(self._transport):
            self.stayed_up = False


def _advance(
    transport: object,
    step: Callable[[float], object],
    seconds: float,
    chunk_s: float,
    heights: _Heights,
) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        dt = min(chunk_s, seconds - elapsed)
        step(dt)
        elapsed += dt
        heights.sample()


def _height(transport: object) -> float:
    return float(getattr(transport, "base_position")[2])


def _xy(transport: object) -> tuple[float, float]:
    pos = getattr(transport, "base_position")
    return (float(pos[0]), float(pos[1]))


def _yaw(transport: object) -> float:
    read: Callable[[], object] = getattr(transport, "read")
    telemetry = read()
    odom = getattr(telemetry, "odom")
    if odom is None:
        raise RuntimeError("sim telemetry has no odom")
    return float(odom.theta)


def _is_up(transport: object) -> bool:
    read: Callable[[], object] = getattr(transport, "read")
    telemetry = read()
    custom = getattr(telemetry, "custom")
    block = custom.get("humanoid")
    if isinstance(block, (str, bytes)) or not isinstance(block, Mapping):
        return False
    return block.get("fall_state") != "fallen"


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
