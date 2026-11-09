# Aerial capability (`ohho.aerial`)

Experimental: nothing has flown. The types and `Robot` methods below are the
API a later backend will implement. This package does not open a radio, arm a
flight controller, or set a vehicle ready to fly.

```python
from ohho.aerial import Frame
from ohho.robot import Robot

bot.offboard_velocity(0.3, 0.0, -0.2, 0.0, Frame.BODY_FRD)
```

`frame` is an argument on every setpoint. There is no default. Passing
`"LOCAL_NED"` as a string raises `TypeError`; pass `Frame.LOCAL_NED`.

## `arm_flight` and `arm`

`Robot.arm()` unlocks the SafetyGate so motion commands are allowed to pass.
On hardware that still needs the environment variable, a TTY, and the typed
phrase. A simulated robot reports itself armed without that ceremony.

`Robot.arm_flight()` asks the autopilot to arm the motors. It does not call
`arm()`, and it does not change the gate's armed flag. On a transport wrapped
in `SafetyGate` the call is `command("arm")`.

`Robot.disarm_flight()` asks the autopilot for a normal disarm. The action is
`disarm`. The method takes no force argument and does not send a force-disarm.
`Robot.disarm()` only disarms the gate.

## Flight methods

All of these return the robot. On a spec without `aerial` they do nothing.
`takeoff` checks its altitude only when `aerial` is present, so a ground robot
can call it and stay put. Setpoints that take a `frame` still require a
`Frame` value on every robot: a missing or non-`Frame` value raises
`TypeError` before the no-op.

| Method | What it asks the autopilot to do |
|---|---|
| `arm_flight()` | Arm the motors. |
| `disarm_flight()` | Disarm normally. |
| `takeoff(alt_m)` | Climb to `alt_m` metres above the local origin, positive up, then hold. |
| `land()` | Land at the current horizontal position. |
| `rtl()` | Return to launch. Ohho does not fly the path itself. |
| `hold()` | Loiter / hover. |
| `offboard_position(x, y, z, yaw, frame)` | Position setpoint in `frame`. |
| `offboard_velocity(vx, vy, vz, yaw_rate, frame)` | Velocity setpoint in `frame`. |
| `set_yaw(yaw, frame)` | Absolute yaw in `frame`. |

`Robot.drive(vx, vy, w)` stays planar. It commands forward, lateral, and yaw
rate. Climb is `takeoff` or `offboard_velocity`.

`takeoff(alt_m)` is always metres, positive up, in the autopilot's takeoff
mode. It does not take a frame, and a negative number is not "NED up".

`z` and `vz` on the offboard methods are in the frame you pass. Positive `z`
is down in `LOCAL_NED` and `BODY_FRD`, and up in `LOCAL_ENU`. `Robot` forwards
those numbers unchanged.

## Frames

| Frame | Axes |
|---|---|
| `Frame.LOCAL_NED` | x north, y east, z down. Positive `vz` descends. PX4 local setpoint. |
| `Frame.LOCAL_ENU` | x east, y north, z up. ROS local convention. |
| `Frame.BODY_FRD` | x forward, y right, z down. MAVLink `MAV_FRAME_BODY_NED`. Stick teleop uses this. |

Convert local frames at the backend, before publish, with `ohho.aerial`:

| Helper | Map |
|---|---|
| `convert_vector(x, y, z, source=, target=)` | ENU (east, north, up) ↔ NED (north, east, down): swap x/y and negate z. |
| `convert_yaw(yaw, source=, target=)` | `yaw_enu = pi/2 - yaw_ned`. Zero NED yaw faces north; zero ENU yaw faces east. |
| `convert_yaw_rate(yaw_rate, source=, target=)` | Sign flip. Positive NED yaw rate is nose-right; positive ENU yaw rate is nose-left. |

`source` and `target` are keyword-only and required. `BODY_FRD` converts only
to itself. Turning a body vector into a local vector needs the vehicle
attitude, which these helpers do not invent.

Mode-2 sticks become a body setpoint with `stick_to_body_frd(forward, right, climb, yaw_right)`:

| Stick axis | Setpoint in `BODY_FRD` |
|---|---|
| Right stick Y, up positive (`forward`) | `+vx` |
| Right stick X, right positive (`right`) | `+vy` |
| Left stick Y, up positive (`climb`) | negative `vz` (`body_z_from_climb`) |
| Left stick X, right positive (`yaw_right`) | positive yaw rate, nose-right |

That yaw sign matches a mode-2 radio and the VR aerial scheme. The helper
applies those signs and returns a `BODY_FRD` tuple `(vx, vy, vz, yaw_rate)`.

## Transport

An ungated aerial transport implements:

```python
def send_flight(self, action: str, **fields) -> None: ...
```

| `action` | `fields` |
|---|---|
| `"arm"` | none |
| `"disarm"` | none |
| `"takeoff"` | `alt_m` |
| `"land"`, `"rtl"`, `"hold"` | none |
| `"offboard_position"` | `x`, `y`, `z`, `yaw`, `frame` |
| `"offboard_velocity"` | `vx`, `vy`, `vz`, `yaw_rate`, `frame` |
| `"set_yaw"` | `yaw`, `frame` |

`frame` is a `Frame` member. A transport with no `send_flight` raises
`AttributeError` when the spec includes `aerial`.

When `Robot.transport` is a `SafetyGate`, the same request is
`gate.command(action, *args)` and is not sent around the gate. Positional
`args` are `(alt_m,)` for takeoff, `(x, y, z, yaw, frame)` for position,
`(vx, vy, vz, yaw_rate, frame)` for velocity, `(yaw, frame)` for yaw, and
empty otherwise. The current sport-action allowlist does not include these
actions, so the gate refuses them and commands zero velocity. A drone profile
has to allowlist them before a gated link will forward them.

Telemetry for an aircraft is not defined here. The shared `Telemetry` type is
unchanged.
