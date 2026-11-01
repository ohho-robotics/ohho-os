# Drone SafetyGate profile (OHH-288)

Experimental. Nothing has flown. `flight_ready` is false, and nothing in this
package sets it true. The rules below are a software guard in the Python
process. They are not a clearance to arm an aircraft.

`unitree-go2` and `omnibot` keep the profiles they already had. A drone does
not reuse `GO2_ACTIONS`: that set has no `land`, so a landing would be
refused. The 0.6 rad tilt latch is the quadruped guard. It is off on the
drone profile. `require_orientation` is off until a drone sim publishes a
stable attitude.

## Profile

`DRONE_PROFILE` in `ohho.safety` is the profile for `px4-sitl`,
`crazyflie-cf21`, and `px4-x500`, and for any other spec that declares the
`aerial` capability and has no ground profile of its own.

| | Drone profile |
|---|---|
| Horizontal speed | 1.5 m/s (vector magnitude) |
| Vertical speed | 1.0 m/s |
| Yaw rate | 0.8 rad/s |
| Teleop takeoff altitude | 3 m |
| Geofence | cylinder, radius 10 m, max altitude 3 m |
| `require_orientation` | off |
| Tilt latch | off |

A config file may tighten these. It may not switch a ground robot to
`aerial`, and it may not turn the tilt latch off on a ground robot. Takeoff
altitude may not be set above the geofence ceiling. The hard linear ceiling
still applies to horizontal and vertical speed. Only `hard_ceiling` in a
safety config file raises that ceiling.

### Allowlist

Ordinary commands the gate will forward once it is armed:

`hold`, `land`, `rtl`, `offboard_velocity`, `offboard_position`.

`flip` and every other unknown name are refused. The denylist (`flip`,
`jump`, `pounce`, `dance`, `handstand`, `upright`, `cross_step`, `free*`)
still wins over a config file.

`arm_flight` and `takeoff` are not on that allowlist. A human at a TTY may
request them. On hardware that is the same typed phrase `SafetyGate.arm`
already uses, plus `OHHO_ARM_HARDWARE=1`. A simulated link still needs a TTY,
so a headless caller cannot take off. They are not MCP tools.

GitHub Actions may arm and take off only a proven simulator, and only with
an explicit opt-in. All of these must hold: `CI` and `GITHUB_ACTIONS` are
`true` (any case, surrounding space ignored), `OHHO_CI_SIM_ARMING` is exactly
`1`, `OHHO_ARM_HARDWARE` is unset, and the gate is not a hardware gate. If
the transport exposes `requires_hardware_arming`, that must be false too.
The target is `drone-sim` (backend id `drone-sim`, or protocol `simulated`
with no other backend id), or a loopback `px4+dds://` link whose
`sim_target` is true, or a loopback `mavlink+udp://` link whose
`simulation_target` is true. Loopback means `127.0.0.0/8`, `::1`, or
`localhost`. The PX4 and MAVLink cases also need `OHHO_CI_SITL` exactly `1`.
Serial, radio, Crazyflie, docker names, non-loopback hosts, and every other
backend are refused. When the exception is used the gate logs
`safety: CI sim-only arming exception used` and writes a `ci_sim_arming`
audit line (action, backend, host). If that line cannot be written the
command is refused. Geofence, takeoff altitude, battery, link, the setpoint
stream, the allowlist, speed caps, and the e-stop latch still apply.
`release_stop()` and ground `arm()` are unchanged. This never arms hardware.

`arm_flight` also needs an offboard setpoint forwarded in the last 300 ms.
The gate does not arm the autopilot by calling `arm()`. `Robot.arm()` only
unlocks the gate.

`rtl` is a gate action. An MCP server must not register `drone_arm`,
`drone_takeoff`, `drone_rtl`, `release_stop`, or disarm. `drone_mcp_allowed`
is the check for that later server. `land` and `hold` pass it.

`disarm` is a normal disarm only. It is refused unless the landed state is
`on_ground` and the positive-up altitude is a finite number of metres at or
below 0.25. Extra arguments are refused. Force-disarm is not a parameter.

Takeoff is refused when the battery fraction is missing or below 0.3, or
when `link_ok` is not true. If both `Telemetry.battery` and
`custom["drone"]["battery"]` are present, the lower one is used.

## Caps

Velocity magnitude may be clipped. Position may not.

`offboard_velocity` clips the horizontal magnitude to 1.5 m/s, vertical
speed to 1.0 m/s, and yaw rate to 0.8 rad/s. A clipped command is forwarded
at the capped value. The audit record is `clamp`, with `requested`,
`applied`, and `reasons` containing `velocity_cap`, plus `flight_request`,
`frame`, and `backend`.

`send_velocity` on an aerial profile also clips the horizontal magnitude, on
top of the existing per-axis and acceleration limits. Ground profiles are
unchanged: Go2 is still 0.5 / 0.3 / 1.0, and a 2 m/s Go2 command is still
clamped per axis, not by the drone magnitude rule.

## Geofence

The fence is a cylinder about the local origin. Default sim radius 10 m,
max altitude 3 m. Altitude is positive up. The floor is the origin.

| Frame | Horizontal | Positive-up altitude |
|---|---|---|
| `LOCAL_NED` | `hypot(x, y)` | `-z` |
| `LOCAL_ENU` | `hypot(x, y)` | `z` |
| `BODY_FRD` | not a local position | rejected |

A position setpoint outside the cylinder, below the origin, or in
`BODY_FRD` is not forwarded. `command` returns false, `last_error` is
`geofence`, an error is logged, the audit reason is `geofence`, and `hold`
is sent. The position is not clipped onto the fence.

Takeoff above the teleop ceiling or the fence ceiling is the same refusal:
error and hold, not a climb to the capped height.

A velocity setpoint is clipped, not rejected, while the current pose is
inside the fence or not yet known. A pose that is already outside rejects
the velocity and sends hold.

On the boundary (radius 10 m, altitude 3 m, altitude 0) the setpoint is
inside.

## E-stop layers

These are not synonyms.

| Layer | Who | In flight |
|---|---|---|
| Ohho `emergency_stop` | `SafetyGate` latches and disarms the gate | `px4-dds` and `mavlink`: the D-01 decision. Land when `in_air`, `takeoff`, or `landing`. Hold, then land, when the landed state is unknown or missing. Disarm only when `on_ground` and altitude is finite and `<= 0.25` m. Unknown is never on the ground. |
| Autopilot failsafe | PX4 `COM_*` / `GF_*`, ArduPilot `FENCE_*` / `FS_*` | Runs even if this process is dead. Ohho does not disable it. |
| Hardware kill | The pilot's RC kill switch | Motor outputs stop. The vehicle falls. Not an Ohho function. Not an MCP tool. |

The Ohho decision is `ohho.drone.estop.estop_commands`. This profile does not
reimplement it. The gate never sends force-disarm (`MAV_CMD_COMPONENT_ARM_DISARM`
param2 `21196`) and never sends flight termination.

Backend id selects the Ohho mapping. Set `backend_id` on the transport.
`protocol == "simulated"` is `drone-sim` when the attribute is absent.

| Backend id | What the gate does | Audit `estop_layer` |
|---|---|---|
| `crazyflie`, `crazyflie-cf21` | Motor cut. Documented exception for a 30 g aircraft in a net. No land and no disarm from this path. | `crazyflie_motor_cut` |
| `drone-sim`, `sim`, `px4-sitl`, `crazyflie-sim` | The D-01 commands, then `simulated_descent` when the transport has it. Descent is simulated. Not a free-fall and not a motor cut. | `simulated_descent` |
| `px4-dds`, `mavlink`, anything else | The D-01 commands only, via `send_flight`. Not `emergency_stop`, so a zero setpoint is not used as a stand-in for land. | `ohho_land` |

The backend id is on the audit line, including the Crazyflie motor-cut line.

An in-air Ohho e-stop does not disarm. A ground e-stop with altitude above
0.25 m lands and does not disarm.

## `release_stop`

Clears the Ohho latch only. It does not arm the autopilot and does not
resume offboard. The gate stays disarmed; motion needs a new `arm()`.

On hardware the existing rule is unchanged: an interactive TTY and the word
`release`. CI and a missing TTY are refused. A non-zero `error_code` still
refuses the release. Roll and pitch do not, on this profile.

The audit record has `resumed_offboard: false` and `autopilot_armed: false`.

## Audit

Aerial records add three fields to flight, clamp, reject, e-stop, and
release lines in `~/.ohho/safety/<UTC date>.jsonl`:

| Field | Value |
|---|---|
| `flight_request` | The request (`land`, `hold`, `offboard_velocity`, …). An e-stop with several D-01 steps sends the list of names. Crazyflie sends `motor_cut`. |
| `frame` | `LOCAL_NED`, `LOCAL_ENU`, `BODY_FRD`, or null. |
| `backend` | The transport `backend_id`. |

Ground audit lines are unchanged.

## What this does not do

- It does not open a radio, start PX4, or set `OHHO_ARM_HARDWARE`.
- It does not fly a return-to-launch path. `rtl` asks the autopilot.
- It does not publish the autopilot geofence. PX4 `GF_*` and ArduPilot
  `FENCE_*` are the pilot's settings. The software fence rejects the
  setpoint first; the autopilot fence runs after a breach.
- It does not enable `require_orientation`. Turn that on only after a sim
  publishes a stable attitude, and tune any hardware tilt limit after a
  tethered hover with a person watching.
