# Drone e-stop and telemetry (OHH-285)

Experimental. Not flight-ready. `flight_ready` is always false, and nothing
in this package sets it true. `ohho.drone` decides and describes. It does
not open a socket, encode MAVLink, or command hardware.

Pose frames are the plain strings `LOCAL_NED` and `LOCAL_ENU`. OHH-286 adds
the frame enum. This package does not import it.

`SafetyGate` is unchanged. A later change can call `estop_commands` when a
drone transport handles `emergency_stop`. The gate's own latch, arming, and
audit rules stay where they are.

## Telemetry

`DroneTelemetry.to_ohho_telemetry()` writes `ohho.schema.Telemetry`. The
battery fraction is `Telemetry.battery`. Everything else is
`Telemetry.custom["drone"]`, with these keys:

| Key | Meaning |
|---|---|
| `armed` | Autopilot armed, not the safety gate. |
| `flight_mode` | String from the backend's mode table. |
| `landed_state` | `on_ground`, `in_air`, `takeoff`, `landing`, or `unknown`. |
| `pose` | `{x, y, z, roll, pitch, yaw}` in `pose_frame`. Metres and radians. |
| `pose_frame` | `LOCAL_NED` or `LOCAL_ENU`. The backend sets it. |
| `velocity` | `{vx, vy, vz, yaw_rate}` in `velocity_frame`. |
| `velocity_frame` | `LOCAL_NED`, `LOCAL_ENU`, or `BODY_FRD`. |
| `battery` | Fraction `0..1`, the same number as `Telemetry.battery`. |
| `voltage_v` | Pack voltage, volts. |
| `current_a` | Pack current, amps. |
| `gps_fix` | `none`, `2d`, `3d`, `dgps`, `rtk_float`, `rtk_fixed`. |
| `failsafe` | Active flags: `offboard_loss`, `rc_loss`, `geofence`, `battery`, `datalink`. |
| `link_ok` | Link considered up. |
| `link_quality` | `0..1` when the backend has a figure, otherwise null. |
| `experimental` | Always true. |
| `flight_ready` | Always false. |

`unknown` is what a missing or unrecognised landed report becomes. It is
never rewritten to `on_ground`.

`LOCAL_NED` z is down, so positive-up height is `-pose.z`. `LOCAL_ENU` z is
already up. That local-origin height is not a global altitude above ground.
Pass the better figure into the e-stop when the backend has one.

`Telemetry.odom` may carry a planar shadow `(x, y, yaw, vx, vy, yaw rate)`
so a recorder that only reads odometry still sees horizontal motion. Build
it with `include_planar_odom=True` or pass `odom=` yourself. The shadow is
not the altitude. It is refused when pose and velocity are in different
frames, so body-frame velocity is not stored as a local velocity.

```python
from ohho.drone import DronePose, DroneTelemetry

snap = DroneTelemetry(
    pose_frame="LOCAL_NED",
    velocity_frame="LOCAL_NED",
    landed_state="in_air",
    pose=DronePose(z=-2.5),
)
tele = snap.to_ohho_telemetry(include_planar_odom=True)
tele.custom["drone"]["experimental"]  # True
tele.custom["drone"]["flight_ready"]  # False
```

## E-stop

`estop_commands(landed_state, altitude_m)` returns the ordered command list.
It does no I/O. Altitude is metres, positive up. The default altitude is
missing, and a missing sample is not treated as zero.

| Landed state | Altitude | Commands |
|---|---|---|
| `in_air`, `takeoff`, `landing` | any, including missing | `land` |
| `unknown`, or not a known report | any, including high | `hold`, then `land` |
| `on_ground` | finite and `<= 0.25` m | `disarm` |
| `on_ground` | missing, not finite, or `> 0.25` m | `land` |

A ground flag that disagrees with height lands. It does not disarm, and it
does not skip straight to a motor cut. Unknown always holds and then lands,
including when the vehicle is high. Altitude is a disarm gate. It is not a
reason to skip the hold, and it is not a reason to treat unknown as on the
ground.

| Name | MAVLink id | Parameters |
|---|---|---|
| `land` | `21` `MAV_CMD_NAV_LAND` | zeros |
| `hold` | `176` `MAV_CMD_DO_SET_MODE` | PX4 `AUTO.LOITER` hints (`param1=1`, packed loiter mode, sub-mode `3`) |
| `disarm` | `400` `MAV_CMD_COMPONENT_ARM_DISARM` | `param1=0`, `param2=0` |

The name is the contract. The MAVLink id and the hold-mode parameters are
there for a later backend. Other backends send `land`, `hold`, or `disarm`
and ignore the PX4 mode word.

Two commands are never produced, and the command log drops them if a caller
builds them anyway:

* force-disarm, `MAV_CMD_COMPONENT_ARM_DISARM` with `param2` `21196`
* flight termination, `MAV_CMD_DO_FLIGHTTERMINATION` (`185`), or a command
  named `flight_termination`, `terminate`, `kill`, or `force_disarm`

```python
from ohho.drone import LandedState, estop_commands

[step.name for step in estop_commands(LandedState.IN_AIR, 2.5)]
# ["land"]
[step.name for step in estop_commands(LandedState.UNKNOWN, 2.5)]
# ["hold", "land"]
[step.name for step in estop_commands(LandedState.ON_GROUND, 0.1)]
# ["disarm"]
```

## Simulation targets

`MockCommandLog` is the in-memory log tests use for "what would have been
sent". `connect` applies the allow-list and does not open a socket, resolve
DNS, or start a simulator.

| Host | `sitl=False` | `sitl=True` |
|---|---|---|
| `127.0.0.0/8`, `::1`, IPv4-mapped loopback, `localhost` | allowed | allowed |
| `host.docker.internal`, `*.docker.internal`, `*.containers.internal` | refused | allowed |
| any other name or address | refused | refused |

A refused link logs nothing. The decision function can still be called on
its own; the log is what blocks a command aimed at a non-simulation host.

```python
from ohho.drone import MockCommandLog

link = MockCommandLog(host="192.0.2.10")
link.connect().allowed  # False
link.apply_estop("on_ground", 0.0)  # ()
```
