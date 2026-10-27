"""Kinematic multicopter stub (backend id ``drone-sim``).

A point mass plus yaw. ``step(dt)`` is the only clock: no thread, no socket,
no wall time, and no rotor model. This is not a physics engine. Nothing
has flown.

Internal pose is ``LOCAL_NED`` (x north, y east, z down, yaw clockwise from
north). ``LOCAL_ENU`` and ``BODY_FRD`` setpoints are converted with
``ohho.aerial`` before integration. Planar ``send_velocity`` is body-frame
forward, right, and yaw rate; it never climbs.

Ohho caps (not PX4 ``MPC_XY_VEL_MAX``): 2 m/s horizontal, 1 m/s climb, and
0.8 rad/s yaw. A spec may be stricter. ``emergency_stop`` freezes horizontal
motion and descends until the vehicle is on the ground, then disarms. It
does not cut motors, model free-fall, or zero the altitude in one step.

Telemetry is a plain dict at ``Telemetry.custom["drone"]``. OHH-285 adds the
typed schema; this module does not define ``DroneTelemetry``.
"""

from __future__ import annotations

import math

from ..aerial import (
    Frame,
    convert_vector,
    convert_yaw,
    convert_yaw_rate,
    finite_number,
    require_frame,
)
from ..registry import RobotSpec
from ..schema import (
    ConnectionState,
    Odometry,
    Telemetry,
    TransportStatus,
    Velocity,
    clamp,
)
from ..transport import BaseTransport

# Ohho caps for px4-sitl / crazyflie-sim. Not the PX4 parameter.
OHHO_MAX_HORIZONTAL_MPS = 2.0
OHHO_MAX_CLIMB_MPS = 1.0
OHHO_MAX_YAW_RATE_RAD_S = 0.8

_TAU_VEL = 0.25
_TAU_POS = 0.8
_TAU_YAW = 0.4
_RTL_RADIUS_M = 0.3
_DISARM_ALT_M = 0.25
_PACK_AH = 5.0
_V_FULL = 16.8
_V_EMPTY = 13.2
_CURRENT_IDLE = 0.2
_CURRENT_ARMED_GROUND = 1.5
_CURRENT_FLIGHT = 12.0
_BATTERY_FAILSAFE = 0.15

_PHASE_DISARMED = "disarmed"
_PHASE_READY = "ready"
_PHASE_TAKEOFF = "takeoff"
_PHASE_HOLD = "hold"
_PHASE_LAND = "land"
_PHASE_RTL = "rtl"
_PHASE_POSITION = "offboard_position"
_PHASE_VELOCITY = "offboard_velocity"

_FLIGHT_MODE = {
    _PHASE_DISARMED: "disarmed",
    _PHASE_READY: "ready",
    _PHASE_TAKEOFF: "takeoff",
    _PHASE_HOLD: "hold",
    _PHASE_LAND: "land",
    _PHASE_RTL: "rtl",
    _PHASE_POSITION: "offboard",
    _PHASE_VELOCITY: "offboard",
}

_Cmd = tuple[float, float, float, float]


def _limits(spec: RobotSpec) -> tuple[float, float, float]:
    """Horizontal, climb, and yaw caps. Never looser than the Ohho caps."""
    horizontal = spec.max_lin if spec.max_lin > 0.0 else OHHO_MAX_HORIZONTAL_MPS
    climb = spec.max_climb if spec.max_climb > 0.0 else OHHO_MAX_CLIMB_MPS
    yaw = spec.max_ang if spec.max_ang > 0.0 else OHHO_MAX_YAW_RATE_RAD_S
    return (
        min(horizontal, OHHO_MAX_HORIZONTAL_MPS),
        min(climb, OHHO_MAX_CLIMB_MPS),
        min(yaw, OHHO_MAX_YAW_RATE_RAD_S),
    )


def _lag(current: float, target: float, dt: float, tau: float) -> float:
    """First-order step toward ``target``. One update per ``step`` call."""
    if tau <= 0.0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


def _clamp_horizontal(vx: float, vy: float, limit: float) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    if speed <= limit or speed <= 0.0:
        return (vx, vy)
    scale = limit / speed
    return (vx * scale, vy * scale)


class DroneSimTransport(BaseTransport):
    """In-process kinematic multicopter. Drive it with ``step(dt)`` only."""

    protocol = "simulated"
    backend = "drone-sim"

    def __init__(self, spec: RobotSpec) -> None:
        super().__init__()
        self.spec = spec
        horizontal, climb, yaw = _limits(spec)
        self._max_horizontal = horizontal
        self._max_climb = climb
        self._max_yaw = yaw
        self._state = ConnectionState.IDLE
        self._connected_since: float | None = None
        self._reset()

    def _reset(self) -> None:
        self._sim_time = 0.0
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._yaw = 0.0
        self._zero_rates()
        self._armed = False
        self._phase = _PHASE_DISARMED
        self._landed_state = "on_ground"
        self._estop_latched = False
        self._home_x = 0.0
        self._home_y = 0.0
        self._hold_x = 0.0
        self._hold_y = 0.0
        self._hold_z = 0.0
        self._target_alt = 0.0
        self._sp_x = 0.0
        self._sp_y = 0.0
        self._sp_z = 0.0
        self._yaw_target = 0.0
        self._yaw_mode = "absolute"
        self._cmd_frame = Frame.LOCAL_NED
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_vz = 0.0
        self._cmd_yaw_rate = 0.0
        self._battery = 1.0
        self._current_a = _CURRENT_IDLE

    def _zero_rates(self) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw_rate = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        self._reset()
        self._state = ConnectionState.CONNECTED
        self._connected_since = self._sim_time
        status = self.status()
        self._emit_status(status)
        self._emit_telemetry(self._snapshot())
        return status

    def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        return TransportStatus(
            protocol=self.protocol,
            state=self._state,
            label=f"drone-sim · {self.spec.id}",
            connected_since=self._connected_since,
        )

    def read(self) -> Telemetry:
        return self._snapshot()

    # ── commands ─────────────────────────────────────────────────────────────
    def send_velocity(self, vel: Velocity) -> None:
        """Planar body command: forward, right, yaw rate. Altitude is held.

        Ignored on the ground and while takeoff, landing, RTL, or e-stop owns
        the vehicle. The vertical channel of this transport is not on
        ``Velocity``.
        """
        vx = finite_number("linear_x", vel.linear_x)
        vy = finite_number("linear_y", vel.linear_y)
        yaw_rate = finite_number("angular_z", vel.angular_z)
        if not self._setpoints_allowed():
            return
        self._cmd_frame = Frame.BODY_FRD
        self._cmd_vx = vx
        self._cmd_vy = vy
        self._cmd_vz = 0.0
        self._cmd_yaw_rate = yaw_rate
        self._yaw_mode = "rate"
        self._phase = _PHASE_VELOCITY
        self._landed_state = "in_air"

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def send_flight(self, action: str, **fields: object) -> None:
        """One autopilot request from ``Robot`` flight methods.

        ``disarm`` never force-disarms. Extra fields such as ``force`` are
        ignored. An airborne disarm is refused.
        """
        if action == "arm":
            self._cmd_arm()
        elif action == "disarm":
            self._cmd_disarm()
        elif action == "takeoff":
            self._cmd_takeoff(fields.get("alt_m"))
        elif action == "land":
            self._cmd_land()
        elif action == "rtl":
            self._cmd_rtl()
        elif action == "hold":
            self._cmd_hold()
        elif action == "offboard_position":
            self._cmd_position(fields)
        elif action == "offboard_velocity":
            self._cmd_velocity(fields)
        elif action == "set_yaw":
            self._cmd_yaw(fields)

    def emergency_stop(self) -> None:
        """Freeze horizontal motion and land, then disarm.

        Altitude is unchanged until a later ``step``. Descent is a capped
        downward speed, not a free-fall and not an instant drop to the ground.
        """
        self._estop_latched = True
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        if self._vz < 0.0:
            self._vz = 0.0
        on_ground = self._landed_state == "on_ground" and -self._z <= _DISARM_ALT_M
        if on_ground:
            self._z = 0.0 if self._z > 0.0 else self._z
            self._vz = 0.0
            self._disarm_on_ground()
            return
        self._begin_descent()

    def release_stop(self) -> None:
        """Clear the sim latch. Does not arm and does not resume a setpoint."""
        self._estop_latched = False

    def _setpoints_allowed(self) -> bool:
        if self._estop_latched or self._phase in (
            _PHASE_LAND,
            _PHASE_TAKEOFF,
            _PHASE_RTL,
        ):
            return False
        return self._armed and self._landed_state == "in_air"

    def _cmd_arm(self) -> None:
        if self._estop_latched or self._phase == _PHASE_LAND:
            return
        if self._landed_state != "on_ground" or -self._z > _DISARM_ALT_M:
            return
        self._armed = True
        self._phase = _PHASE_READY
        self._landed_state = "on_ground"
        self._home_x = self._x
        self._home_y = self._y
        self._yaw_target = self._yaw

    def _cmd_disarm(self) -> None:
        # Refused in the air. There is no force-disarm path.
        if self._landed_state != "on_ground" or -self._z > _DISARM_ALT_M:
            return
        self._disarm_on_ground()

    def _disarm_on_ground(self) -> None:
        self._armed = False
        self._phase = _PHASE_DISARMED
        self._landed_state = "on_ground"
        self._zero_rates()
        self._yaw_mode = "absolute"

    def _cmd_takeoff(self, alt_m: object) -> None:
        altitude = finite_number("alt_m", alt_m)
        if altitude <= 0.0:
            raise ValueError("alt_m must be positive up, in metres")
        if not self._armed or self._estop_latched or self._phase == _PHASE_LAND:
            return
        self._target_alt = altitude
        self._hold_x = self._x
        self._hold_y = self._y
        if self._landed_state == "in_air" and -self._z >= altitude:
            self._enter_hold()
            return
        self._phase = _PHASE_TAKEOFF
        self._landed_state = "takeoff"

    def _cmd_land(self) -> None:
        if self._phase == _PHASE_LAND:
            return
        if self._landed_state == "on_ground" and -self._z <= _DISARM_ALT_M:
            self._disarm_on_ground()
            return
        self._begin_descent()

    def _cmd_rtl(self) -> None:
        if self._estop_latched or self._phase == _PHASE_LAND or not self._armed:
            return
        if self._landed_state not in ("in_air", "takeoff"):
            return
        self._phase = _PHASE_RTL
        self._landed_state = "in_air"
        self._hold_z = self._z
        self._yaw_mode = "absolute"

    def _cmd_hold(self) -> None:
        if not self._armed or self._estop_latched or self._phase == _PHASE_LAND:
            return
        self._yaw_target = self._yaw
        self._yaw_mode = "absolute"
        if self._landed_state == "on_ground":
            self._phase = _PHASE_HOLD
            self._hold_x = self._x
            self._hold_y = self._y
            self._hold_z = self._z
            return
        self._enter_hold()

    def _enter_hold(self) -> None:
        self._phase = _PHASE_HOLD
        self._landed_state = "in_air"
        self._hold_x = self._x
        self._hold_y = self._y
        self._hold_z = self._z
        self._yaw_mode = "absolute"

    def _cmd_position(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        x = finite_number("x", fields.get("x"))
        y = finite_number("y", fields.get("y"))
        z = finite_number("z", fields.get("z"))
        yaw = finite_number("yaw", fields.get("yaw"))
        if not self._setpoints_allowed():
            return
        self._store_position(x, y, z, yaw, frame)
        self._phase = _PHASE_POSITION
        self._yaw_mode = "absolute"
        self._landed_state = "in_air"

    def _cmd_velocity(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        vx = finite_number("vx", fields.get("vx"))
        vy = finite_number("vy", fields.get("vy"))
        vz = finite_number("vz", fields.get("vz"))
        yaw_rate = finite_number("yaw_rate", fields.get("yaw_rate"))
        if not self._setpoints_allowed():
            return
        self._cmd_frame = frame
        self._cmd_vx = vx
        self._cmd_vy = vy
        self._cmd_vz = vz
        self._cmd_yaw_rate = yaw_rate
        self._yaw_mode = "rate"
        self._phase = _PHASE_VELOCITY
        self._landed_state = "in_air"

    def _cmd_yaw(self, fields: dict[str, object]) -> None:
        frame = require_frame(fields.get("frame"))
        yaw = finite_number("yaw", fields.get("yaw"))
        if not self._armed or self._estop_latched or self._phase == _PHASE_LAND:
            return
        if frame == Frame.LOCAL_NED:
            self._yaw_target = yaw
        elif frame == Frame.LOCAL_ENU:
            self._yaw_target = convert_yaw(yaw, source=frame, target=Frame.LOCAL_NED)
        else:
            self._yaw_target = self._yaw + yaw
        self._yaw_mode = "absolute"

    def _store_position(
        self, x: float, y: float, z: float, yaw: float, frame: Frame
    ) -> None:
        if frame == Frame.LOCAL_NED:
            self._sp_x, self._sp_y, self._sp_z = x, y, z
            self._yaw_target = yaw
            return
        if frame == Frame.LOCAL_ENU:
            self._sp_x, self._sp_y, self._sp_z = convert_vector(
                x, y, z, source=frame, target=Frame.LOCAL_NED
            )
            self._yaw_target = convert_yaw(yaw, source=frame, target=Frame.LOCAL_NED)
            return
        north, east, down = self._rotate_body(x, y, z)
        self._sp_x = self._x + north
        self._sp_y = self._y + east
        self._sp_z = self._z + down
        self._yaw_target = self._yaw + yaw

    def _begin_descent(self) -> None:
        self._phase = _PHASE_LAND
        self._landed_state = "landing"
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        if self._vz < 0.0:
            self._vz = 0.0

    def _touchdown(self) -> None:
        self._z = 0.0
        self._zero_rates()
        self._disarm_on_ground()

    # ── integration ──────────────────────────────────────────────────────────
    def step(self, dt: float) -> Telemetry:
        """Advance the model by ``dt`` seconds and return telemetry.

        Callers, including tests, pass ``dt``. Nothing in this method reads
        the wall clock.
        """
        dt = finite_number("dt", dt)
        if dt < 0.0:
            raise ValueError("dt must be >= 0")
        if dt == 0.0:
            return self._snapshot()
        self._sim_time += dt
        self._update_power(dt)
        self._integrate(dt)
        snap = self._snapshot()
        self._emit_telemetry(snap)
        return snap

    def _integrate(self, dt: float) -> None:
        vx_cmd, vy_cmd, vz_cmd, yaw_cmd = self._commands()
        vx_cmd, vy_cmd = _clamp_horizontal(vx_cmd, vy_cmd, self._max_horizontal)
        vz_cmd = clamp(vz_cmd, -self._max_climb, self._max_climb)
        yaw_cmd = clamp(yaw_cmd, -self._max_yaw, self._max_yaw)
        if self._phase == _PHASE_LAND:
            self._vx = 0.0
            self._vy = 0.0
            self._yaw_rate = 0.0
            vx_cmd = 0.0
            vy_cmd = 0.0
            yaw_cmd = 0.0
        self._vx = _lag(self._vx, vx_cmd, dt, _TAU_VEL)
        self._vy = _lag(self._vy, vy_cmd, dt, _TAU_VEL)
        self._vz = _lag(self._vz, vz_cmd, dt, _TAU_VEL)
        self._yaw_rate = _lag(self._yaw_rate, yaw_cmd, dt, _TAU_VEL)
        if self._phase == _PHASE_LAND:
            self._vx = 0.0
            self._vy = 0.0
            self._yaw_rate = 0.0
        self._x += self._vx * dt
        self._y += self._vy * dt
        self._yaw += self._yaw_rate * dt
        next_z = self._z + self._vz * dt
        if self._phase == _PHASE_TAKEOFF and -next_z >= self._target_alt:
            self._z = -self._target_alt
            self._zero_rates()
            self._enter_hold()
            return
        if self._phase == _PHASE_LAND and next_z >= 0.0:
            self._touchdown()
            return
        if next_z > 0.0:
            next_z = 0.0
            if self._vz > 0.0:
                self._vz = 0.0
        self._z = next_z

    def _commands(self) -> _Cmd:
        if self._phase == _PHASE_LAND:
            vz = self._max_climb if self._z < 0.0 else 0.0
            return (0.0, 0.0, vz, 0.0)
        if not self._armed or self._phase in (_PHASE_DISARMED, _PHASE_READY):
            return (0.0, 0.0, 0.0, 0.0)
        if self._phase == _PHASE_TAKEOFF:
            vx, vy, _vz = self._track(self._hold_x, self._hold_y, self._z)
            remaining = self._target_alt - (-self._z)
            vz = -self._max_climb if remaining > 0.0 else 0.0
            return (vx, vy, vz, self._absolute_yaw_rate())
        if self._phase == _PHASE_HOLD:
            vx, vy, vz = self._track(self._hold_x, self._hold_y, self._hold_z)
            return (vx, vy, vz, self._absolute_yaw_rate())
        if self._phase == _PHASE_POSITION:
            vx, vy, vz = self._track(self._sp_x, self._sp_y, self._sp_z)
            return (vx, vy, vz, self._absolute_yaw_rate())
        if self._phase == _PHASE_RTL:
            dist = math.hypot(self._x - self._home_x, self._y - self._home_y)
            if dist <= _RTL_RADIUS_M:
                self._begin_descent()
                vz = self._max_climb if self._z < 0.0 else 0.0
                return (0.0, 0.0, vz, 0.0)
            vx, vy, vz = self._track(self._home_x, self._home_y, self._hold_z)
            return (vx, vy, vz, self._absolute_yaw_rate())
        if self._phase == _PHASE_VELOCITY:
            vx, vy, vz, yaw_rate = self._velocity_ned()
            if self._yaw_mode == "absolute":
                yaw_rate = self._absolute_yaw_rate()
            return (vx, vy, vz, yaw_rate)
        return (0.0, 0.0, 0.0, 0.0)

    def _track(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        ex = x - self._x
        ey = y - self._y
        ez = z - self._z
        dist = math.hypot(ex, ey)
        if dist <= 1e-9:
            vx, vy = 0.0, 0.0
        else:
            speed = min(self._max_horizontal, dist / _TAU_POS)
            vx = ex / dist * speed
            vy = ey / dist * speed
        vz = clamp(ez / _TAU_POS, -self._max_climb, self._max_climb)
        return (vx, vy, vz)

    def _absolute_yaw_rate(self) -> float:
        err = (self._yaw_target - self._yaw + math.pi) % (2.0 * math.pi) - math.pi
        return err / _TAU_YAW

    def _velocity_ned(self) -> _Cmd:
        frame = self._cmd_frame
        vx, vy, vz = self._cmd_vx, self._cmd_vy, self._cmd_vz
        yaw_rate = self._cmd_yaw_rate
        if frame == Frame.BODY_FRD:
            vx, vy, vz = self._rotate_body(vx, vy, vz)
        elif frame == Frame.LOCAL_ENU:
            vx, vy, vz = convert_vector(
                vx, vy, vz, source=frame, target=Frame.LOCAL_NED
            )
            yaw_rate = convert_yaw_rate(yaw_rate, source=frame, target=Frame.LOCAL_NED)
        return (vx, vy, vz, yaw_rate)

    def _rotate_body(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """BODY_FRD to LOCAL_NED. Yaw is clockwise from north."""
        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        north = x * c - y * s
        east = x * s + y * c
        return (north, east, z)

    def _update_power(self, dt: float) -> None:
        if self._armed and self._landed_state != "on_ground":
            self._current_a = _CURRENT_FLIGHT
        elif self._armed:
            self._current_a = _CURRENT_ARMED_GROUND
        else:
            self._current_a = _CURRENT_IDLE
        used = (self._current_a / (_PACK_AH * 3600.0)) * dt
        self._battery = max(0.0, self._battery - used)

    # ── telemetry ────────────────────────────────────────────────────────────
    def _snapshot(self) -> Telemetry:
        # Plain dict on purpose. OHH-285 adds the typed DroneTelemetry schema.
        voltage = _V_EMPTY + (_V_FULL - _V_EMPTY) * self._battery
        current = self._current_a
        fraction = self._battery
        failsafe: list[str] = []
        if fraction <= _BATTERY_FAILSAFE:
            failsafe.append("battery")
        link_ok = self._state == ConnectionState.CONNECTED
        pose = {
            "x": self._x,
            "y": self._y,
            "z": self._z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": self._yaw,
        }
        velocity = {
            "vx": self._vx,
            "vy": self._vy,
            "vz": self._vz,
            "yaw_rate": self._yaw_rate,
        }
        battery = {
            "fraction": fraction,
            "voltage_v": voltage,
            "current_a": current,
        }
        drone = {
            "armed": self._armed,
            "flight_mode": _FLIGHT_MODE[self._phase],
            "landed_state": self._landed_state,
            "pose": pose,
            "pose_frame": "LOCAL_NED",
            "velocity": velocity,
            "velocity_frame": "LOCAL_NED",
            "battery": battery,
            "voltage_v": voltage,
            "current_a": current,
            "gps_fix": self.spec.gps_fix or "3d",
            "failsafe": failsafe,
            "link_ok": link_ok,
            "link_quality": 1.0 if link_ok else 0.0,
            # True until a person logs a real flight. Nothing here flies.
            "experimental": True,
            # Nothing in this stub sets flight_ready true.
            "flight_ready": False,
        }
        # Planar shadow so a recorder that only reads odom still sees x, y, yaw.
        # Altitude is pose["z"] in LOCAL_NED, not odom.
        odom = Odometry(
            x=self._x,
            y=self._y,
            theta=self._yaw,
            vx=self._vx,
            vy=self._vy,
            omega=self._yaw_rate,
        )
        return Telemetry(
            odom=odom,
            joints=[],
            battery=fraction,
            custom={"drone": drone},
            timestamp=self._sim_time,
        )
