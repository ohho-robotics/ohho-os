# PX4 over uXRCE-DDS (`px4+dds://`)

Experimental: nothing has flown. This adapter does not arm a vehicle by
itself, does not open a UDP socket, and does not start SITL or Gazebo.

The companion process is an rclpy node. PX4's `uxrce_dds_client` talks to
Micro XRCE-DDS Agent, and the agent bridges `px4_msgs` onto the ROS 2 graph.
`Px4DdsTransport` publishes and subscribes on that graph.

```text
PX4 SITL or Pixhawk
  uxrce_dds_client --udp 8888--> MicroXRCEAgent
                                      |
                                      v
                               rclpy node
                               Px4DdsTransport
                               SafetyGate
                               Robot
```

`dds://` is the Unitree adapter. PX4 uses `px4+dds://` and is not selected by
the Unitree scheme.

## Versions

| Piece | Pin |
|---|---|
| PX4 | **v1.17.0**. A v1.18 release candidate is refused. |
| `px4_msgs` | Branch **`release/1.17`**. |
| ROS 2 | **Jazzy**. Humble is the alternative if that distro is already sourced. |
| Micro XRCE-DDS Agent | **2.4.3**. `MicroXRCEAgent udp4 -p 8888`. |

The image line recorded for a later SITL workflow is
`px4io/px4-sitl-gazebo:v1.17.0`. This package does not run that image.
`dependencies` in `pyproject.toml` stays empty. rclpy and `px4_msgs` are not
pip packages; the `[ros2]` extra does not install them.

## URI

```text
px4+dds://127.0.0.1:8888
px4+dds://HOST:PORT?ns=NAME
```

`HOST:PORT` is the agent endpoint the operator started. The default port is
8888. `ns` is optional. With `ns=uav1` the topics are `/uav1/fmu/...`. Without
it they are `/fmu/...`. The query accepts `ns` only. A URI with a user or
password is rejected.

Loopback is a sim target. The check is the host: `localhost`, `::1`, or any
address in `127.0.0.0/8`. Port 8888 does not make a non-loopback host a sim
target, and the scheme string is not an input to the check.

`Px4DdsTransport.sim_target` is that result. `requires_hardware_arming` is the
opposite. A non-loopback host refuses `connect` unless `OHHO_ARM_HARDWARE` is
`1`. Connect still does not arm the motors. `1` is the only value that
counts.

## Topics and QoS

Subscribe:

| Topic | Message |
|---|---|
| `/fmu/out/vehicle_status` | `VehicleStatus` |
| `/fmu/out/vehicle_local_position` | `VehicleLocalPosition` |
| `/fmu/out/vehicle_attitude` | `VehicleAttitude` |
| `/fmu/out/battery_status` | `BatteryStatus` |
| `/fmu/out/vehicle_land_detected` | `VehicleLandDetected` |
| `/fmu/out/failsafe_flags` | `FailsafeFlags` |

Publish:

| Topic | Message |
|---|---|
| `/fmu/in/offboard_control_mode` | `OffboardControlMode` |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` |
| `/fmu/in/vehicle_command` | `VehicleCommand` |

QoS is best-effort, depth 1, on every endpoint. Publishers are volatile.
Subscriptions to PX4 are transient-local. The depth-10 reliable profile on
`Ros2Transport` is a different adapter.

## Message pin

On connect, `OffboardControlMode` fields are compared to release/1.17:

```text
timestamp, position, velocity, acceleration, attitude, body_rate,
thrust_and_torque, direct_actuator
```

`actuator` is the older field. A mismatch refuses connect before any
publisher is created, so arming never starts on a mismatched definition.
v1.18 is not accepted by widening the pin.

## Offboard heartbeat

A timer publishes `OffboardControlMode` at 20 Hz (at least 10 Hz) together
with a `TrajectorySetpoint`. Until a caller sends a setpoint, the trajectory
fields are NaN. NaN means that axis is not controlled. A zero velocity
setpoint is hover, and the heartbeat does not use it as a default.

`arm_flight` / `send_flight("arm")` is refused until that stream has published
and `vehicle_status.pre_flight_checks_pass` is true. The arm command is
`VEHICLE_CMD_COMPONENT_ARM_DISARM` (400) with param1 1 and param2 0.

## Commands

Trajectory setpoints are local NED. `LOCAL_ENU` is converted with
`ohho.aerial` before publish. `BODY_FRD` position and absolute yaw are
refused. `BODY_FRD` velocity is rotated with `vehicle_attitude` when a
quaternion has arrived.

| Action | What is published |
|---|---|
| `arm` | Command 400, param1 1, param2 0. Needs a live offboard stream and pre-arm. |
| `disarm` | Command 400, param1 0, param2 0. Only when on the ground and low. |
| `takeoff` | `VEHICLE_CMD_NAV_TAKEOFF` (22). param7 is the positive-up metres requested. |
| `land` | `VEHICLE_CMD_NAV_LAND` (21). |
| `rtl` | `VEHICLE_CMD_NAV_RETURN_TO_LAUNCH` (20). |
| `hold` | `DO_SET_MODE` auto-loiter (main mode 4, sub mode 3). |
| `offboard_position` / `offboard_velocity` / `set_yaw` | Trajectory setpoint, then `DO_SET_MODE` offboard (main mode 6). |

`VEHICLE_CMD_NAV_LAND` is command 21. `vehicle_status` nav_state 21 is orbit.
Those numbers are different enums. Land is the vehicle command.

`send_velocity` is planar local NED with vertical speed 0. `Velocity()` is
hover. It is not land.

param2 **21196** (force-disarm) is refused on every command. Vehicle command
**185** (flight termination) is refused.

## E-stop

`emergency_stop` on this transport:

| Landed state | Altitude | Sent |
|---|---|---|
| `on_ground` | finite and ≤ 0.25 m | Normal disarm, param2 0. |
| `on_ground` | missing, or above 0.25 m | NAV_LAND. |
| `unknown` | any | Auto-loiter, then NAV_LAND. |
| `in_air`, `takeoff`, `landing` | any | NAV_LAND. |

Altitude is the larger of valid `dist_bottom` and `max(0, -z)`. Unknown is
never treated as on the ground. The heartbeat does not publish a trajectory
while the e-stop is latched, so a zero setpoint is not what lands the vehicle.

`release_stop` clears that latch. It does not arm the autopilot and does not
send an offboard mode command.

## Telemetry

`Telemetry.custom["drone"]` uses `pose_frame` `LOCAL_NED`. `flight_ready` is
false. `experimental` is true. `gps_fix` is `none` because this subscription
set has no GPS fix message. `link_ok` becomes true once `vehicle_status` has
arrived; `link_quality` is 1.0 or 0.0 from that presence, not a radio
measurement. The planar `Telemetry.odom` shadow copies x, y, yaw, vx, and vy.
Its yaw rate is 0 because `vehicle_angular_velocity` is not subscribed.

## SafetyGate (OHH-288)

`Robot.connect` wraps this transport in `SafetyGate` because `protocol` is
`px4-dds`, not `simulated`. The profile is the aerial one when the robot spec
has `aerial` (`DRONE_PROFILE`: allowlist `hold`, `land`, `rtl`,
`offboard_velocity`, `offboard_position`; 1.5 m/s horizontal, 1.0 m/s
vertical, 0.8 rad/s yaw; cylinder geofence). `Robot` flight methods call
`SafetyGate.command`, which calls `send_flight`. They do not publish past
the gate.

`requires_hardware_arming` treats `sim_target` True as the loopback
exception. A non-loopback host still needs the hardware phrase. The gate
does not parse `px4+dds://`.

`command` behaviour the gate applies before `send_flight`:

| Request | Gate |
|---|---|
| `arm` / `arm_flight`, `takeoff` | Human at a TTY. Hardware also needs the arm phrase. `arm` needs a live setpoint stream. `takeoff` needs battery ≥ 0.3, `link_ok`, and an altitude inside the fence. |
| `land`, `rtl`, `hold` | Allowlisted. Forwarded when the gate is armed and state is fresh. |
| `offboard_velocity` | Clamped (horizontal magnitude 1.5 m/s, vertical 1.0 m/s, yaw 0.8 rad/s) and the clamp is audited. A pose already outside the fence is rejected and `hold` is sent. |
| `offboard_position` | A point outside the cylinder is rejected and `hold` is sent. The point is not clipped through the fence. |
| anything else (`flip`, `orbit`, …) | Refused. Denied names stay denied. |

E-stop is the D-01 decision, sent as `send_flight` (`land`, `hold` then
`land`, or `disarm`). In the air that is land. Unknown is hold, then land.
Disarm only on the ground at or below 0.25 m. param2 **21196** and command
**185** are not sent. `land`, `rtl`, `hold`, and `disarm` pause the offboard
heartbeat so a later timer tick does not publish a setpoint over the mode
command. `release_stop` clears the Ohho latch only.
