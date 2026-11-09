# Kinematic drone sim (`drone-sim`)

Experimental: nothing has flown. This backend is a **kinematic stub, not a
physics engine**. It has no mass, no rotors, no drag, and no free-fall. It
does not talk to PX4, a radio, or a flight controller, and it does not start
Docker. Pose advances only when something calls `step(dt)`.

`Robot.connect("px4-sitl", "sim://")` and `Robot.connect("crazyflie-sim", "sim://")`
select this backend. The same ids with the transport omitted do too, because
their default transport is `sim://`. Ground robots on `sim://` (`omnibot`,
`unitree-go2`, `sim`) still use the wheeled `SimTransport`.

`px4-x500` is registry data only. `resolve_transport` refuses every URI for
it, including `sim://`. Nothing in this build connects to an X500.
`requires_hardware_arming` is true on that spec: a non-loopback `px4+dds://`
or `mavlink+serial://` URI needs a human present. `px4+dds://` is not
implemented. `mavlink+udp://` and `mavlink+serial://` are the MAVLink backend
(`docs/mavlink.md`); this spec still refuses them.

The link reports `protocol="simulated"`, so `Robot.connect` does not wrap it
in `SafetyGate`. A drone safety profile is a separate change. Do not point
this stub at a real aircraft.

## Caps

Ohho limits, not PX4 `MPC_XY_VEL_MAX`. A spec may be stricter. The stub never
exceeds these:

| Axis | Cap |
|---|---|
| Horizontal speed | 2 m/s |
| Climb and descent | 1 m/s |
| Yaw rate | 0.8 rad/s |

`px4-sitl`, `crazyflie-sim`, and `px4-x500` all use those numbers.
`flight_ready` is false on every one of them. Nothing here sets it true.
`experimental` on the spec is true. Telemetry `experimental` is always true.

`crazyflie-sim` reports `gps_fix: "none"`. `px4-sitl` reports `"3d"`.

## Frames

The integrator stores pose in `LOCAL_NED`: x north, y east, z down, yaw
clockwise from north. Callers still pass an explicit `ohho.aerial.Frame`.
There is no default.

| Frame | What the stub does |
|---|---|
| `LOCAL_NED` | Use the numbers as-is. Positive `vz` descends. |
| `LOCAL_ENU` | Swap x/y and negate z, and convert yaw, before integrating. |
| `BODY_FRD` | Rotate forward/right/down by the current NED yaw. A position in this frame is an offset from the current pose. A yaw in this frame is added to the current yaw. |

An ENU setpoint and the NED setpoint that names the same point both end on
that NED pose. `Robot.drive` and `send_velocity` stay planar: forward, right,
and yaw rate in the body frame, with climb held at zero.

## State machine

Arm on the ground, then `takeoff(alt_m)` with `alt_m` positive up. The stub
climbs at the climb cap and then holds. `land` and `rtl` are autopilot modes:
RTL flies back to the arm position, then lands. `hold` loiters.
`offboard_position` and `offboard_velocity` are first-order (a step change in
the command does not teleport the vehicle). `set_yaw` slews to an absolute
heading.

`disarm_flight` is refused while airborne. There is no force-disarm. Landing
and e-stop disarm only after `landed_state` is `on_ground` and the altitude
is on the ground.

`emergency_stop` freezes horizontal motion and descends at the climb cap
until touchdown, then disarms. Altitude does not jump to zero. It is a
simulated landing, not a motor kill. `release_stop` clears that latch only.
It does not arm and does not resume the old setpoint.

Battery fraction is `Telemetry.battery` and also
`custom["drone"]["battery"]["fraction"]`, with `voltage_v` and `current_a`.
It drains while the vehicle is armed, faster in the air.

## Telemetry

`Telemetry.odom` is a planar shadow (x, y, yaw, vx, vy, yaw rate) in the same
NED horizontal axes. Altitude is not in `odom`.

The aerial state is a plain dict at `Telemetry.custom["drone"]`. OHH-285 adds
a typed `DroneTelemetry`; this backend does not define that class. Keys:

| Key | Value |
|---|---|
| `armed` | Autopilot motors, not the SafetyGate. |
| `flight_mode` | `disarmed`, `ready`, `takeoff`, `hold`, `land`, `rtl`, or `offboard`. |
| `landed_state` | `on_ground`, `in_air`, `takeoff`, or `landing`. |
| `pose` | `{x, y, z, roll, pitch, yaw}` in `pose_frame`. Roll and pitch stay 0. |
| `pose_frame` | `LOCAL_NED`. |
| `velocity` | `{vx, vy, vz, yaw_rate}` in `velocity_frame` (`LOCAL_NED`). |
| `battery` | `{fraction, voltage_v, current_a}`. Fraction is also `Telemetry.battery`. |
| `gps_fix` | `none` on `crazyflie-sim`, `3d` on `px4-sitl`. |
| `failsafe` | List of flags. `battery` appears if the fraction is very low. |
| `link_ok` | True after `connect`, plus `link_quality` 0..1. |
| `experimental` | `true`. |
| `flight_ready` | `false`. |

## Driving it from a test

```python
from ohho.adapters import resolve_transport
from ohho.aerial import Frame
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime

spec = get_spec("px4-sitl")
transport = resolve_transport("sim://", spec)
transport.connect()
bot = Robot(spec, transport, NativeRuntime())
bot.arm_flight()
bot.takeoff(2.5)
transport.step(0.05)  # repeat; do not wait on a wall clock
bot.offboard_velocity(0.5, 0.0, 0.0, 0.0, Frame.LOCAL_NED)
bot.land()
```

A 4 m square at 0.5 m/s in local NED (north, then east, then south, then
west), with corners inside 0.8 m, is the acceptance path. It uses `step(dt)`
only.
