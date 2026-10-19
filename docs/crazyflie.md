# Crazyflie backend (OHH-303)

Experimental. Nothing in this package has flown. `flight_ready` stays false.
`gps_fix` stays `none`. CI never opens a Crazyradio and never starts
Crazyswarm2.

The registry id is `crazyflie-cf21`. Caps match the other aerial specs:
2 m/s horizontal, 1 m/s climb, 0.8 rad/s yaw. Those are Ohho caps, not a
Crazyflie parameter. The kinematic stub is still `crazyflie-sim` on `sim://`.
`Robot.connect("crazyflie-cf21")` with no URI, or with `sim://`, stays on the
ground simulator. Pass one of the URIs below.

`protocol` is `crazyflie`, so `Robot.connect` wraps the link in `SafetyGate`
and leaves it disarmed. The gate's e-stop for backend id `crazyflie` is a
motor cut. The audit line carries `backend: "crazyflie"`,
`flight_request: "motor_cut"`, and `estop_layer: "crazyflie_motor_cut"`.
See `docs/drone-safety.md`. A Crazyflie 2.1 is about 30 g. Cutting the motors
is the documented exception for a sub-250 g aircraft in a net. It is not the
PX4 land-then-disarm policy.

## URIs

```text
crazyflie://radio://0/80/2M/E7E7E7E7E7
crazyflie+ros2://cf21
```

| URI | Speaks | Extra |
|---|---|---|
| `crazyflie://radio://…` | `cflib` high-level commander, world-velocity setpoint, full state, supervisor emergency | `pip install 'ohho-os[crazyflie]'` (`cflib`). The base install's `dependencies` stay empty. |
| `crazyflie+ros2://<name>` | Crazyswarm2 services and topics for that one robot | ROS 2 plus the `crazyflie_interfaces` package. Connect with `runtime="ros2"`. |

`crazyflie://` must be a `radio://` URI. The address above is the Bitcraze
default from the drone plan, not a discovered radio. `crazyflie+ros2://cf21`
uses `/cf21/…`, not the swarm-wide `/all/…` services.

The world frame sent to Crazyswarm2 is z-up, yaw CCW from +x, treated as
`LOCAL_ENU` (the lab frame, not geographic east). `LOCAL_NED` is converted
with `ohho.aerial` before publish. `BODY_FRD` position is refused: a relative
`go_to` is a world offset, not a body offset. `GoTo.srv` comments yaw as
degrees; the Python client and the sim pass radians through, and so does
Ohho.

## What `backend:=sim` actually runs

Checked against `crazyflie_sim/crazyflie_server.py` on Crazyswarm2 `main`,
19 October 2026. The same split is `SIM_EXECUTED`, `SIM_LOG_ONLY`, and
`SIM_ABSENT` in `ohho/adapters/crazyflie.py`.

| Command | Sim server | What Ohho sends |
|---|---|---|
| `takeoff`, `land` | Executed (`CrazyflieSIL`) | Those services. Height is positive up. Duration is at least 2 s. |
| `go_to` | Executed only after a high-level mode is active (takeoff sets it). From idle the SIL raises and the server ignores the call. | `offboard_position` and `set_yaw`, yaw in radians, `relative=false`. |
| `cmd_full_state` | Executed | `offboard_velocity` on the ROS URI, and only when a pose or a prior `go_to` has xy. A missing pose is not filled with the origin. |
| `upload_trajectory`, `start_trajectory` | Executed | Not an Ohho flight method. |
| `emergency` | Service exists. Callback logs `emergency not yet implemented` and does not stop the sim. | Still called. On cflib and the cpp backend this is the motor cut. |
| `cmd_vel_legacy` | Subscribed. Callback logs `cmd_vel_legacy not yet implemented`. | Not sent by flight methods. The Twist is pitch, roll, yaw rate, and thrust, not a linear velocity. |
| `cmd_hover`, `notify_setpoints_stop` | Log only | Not called. Hover after takeoff is the high-level commander's own end state. |
| `arm` | Service is not created. Overview table: n/a | Not called on `crazyflie+ros2://`. The radio path sends `supervisor.send_arming_request`. |
| `rtl` | No such service | Not sent. Ohho does not fly a return. |
| pose, status, parameters | Default logging is off | No position and no battery until some other backend publishes `/<name>/pose` and `/<name>/status`. |

`Robot.drive` stays planar and is not turned into a Crazyflie setpoint. A
zero velocity from the gate is not a land and not an emergency.

Radio velocity uses `commander.send_velocity_world_setpoint` (m/s, yaw rate
converted to degrees at that edge). That command is not in the sim server.
The ROS path uses `cmd_full_state` instead, because the sim executes it.

E-stop on the radio is `supervisor.send_emergency_stop`, or
`loc.send_emergency_stop` when the supervisor port is absent. Further
setpoints are not sent. `release_stop` clears the Ohho latch only. It does
not reboot the Crazyflie and does not resume the last setpoint.

## Local recipe (not run by CI)

Do not point this at a Crazyradio. Do not set `OHHO_ARM_HARDWARE` on a
machine that has a radio plugged in. This recipe is a Crazyswarm2
`backend:=sim` graph on a machine with no aircraft.

Terminal 1, from a Crazyswarm2 workspace, with one enabled robot named
`cf21` in `crazyflies.yaml`:

```sh
ros2 launch crazyflie launch.py backend:=sim
```

Terminal 2, ROS sourced, from this repo. The flight a person can ask Ohho
for, once the gate is armed at a TTY, is:

```python
from ohho import Robot

bot = Robot.connect("crazyflie-cf21", "crazyflie+ros2://cf21", runtime="ros2")
bot.arm()             # human phrase; refused in CI and without a TTY
bot.takeoff(0.5)      # /cf21/takeoff height=0.5 duration=2.0
bot.hold()            # no extra service; takeoff already ends in a hover
bot.land()            # /cf21/land height=0 duration=2.0
bot.emergency_stop()  # /cf21/emergency even if the sim only logs it
bot.disconnect()
```

Two gates sit in front of that sequence today:

1. `SafetyGate.takeoff` refuses while telemetry has no battery fraction
   (`last_error == "battery"`). `backend:=sim` does not publish
   `/cf21/status`, so a real sim session hits this refusal. Unit tests
   supply a fraction the way a status message would (`3.0 V` empty,
   `4.2 V` full, 1S only). Do not invent a voltage to get past the gate.
2. The same gate refuses takeoff when `link_ok` is not true. Connect sets
   `link_ok` from the Ohho link, not from a radio RSSI.

`emergency_stop` does not need the gate to be armed. It calls
`/cf21/emergency` and the audit line names backend `crazyflie`. The sim
callback logs `emergency not yet implemented` and leaves the SIL running.
That gap is real; the Ohho side still makes the call.

High-level durations are at least 2 seconds (`HL_DURATION_FLOOR_S`), which
is the Crazyswarm hello-world takeoff. Land height is `0.0` m for this sim
recipe. A flow-deck hover on hardware often wants a few centimetres so the
controller does not drive into the floor. This backend does not guess that
offset.

## Tests

`python -m unittest tests.test_crazyflie` uses in-memory links and fake ROS
clients. It does not import `cflib` or `rclpy`, and it does not call
`SyncCrazyflie.open_link`.
