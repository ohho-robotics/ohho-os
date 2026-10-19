# Aerial episodes (LeRobot v2.1)

Experimental. These episodes are recorded from the in-process kinematic
`drone-sim` through `ohho sim-serve`. Nothing here has flown. Do not record
hardware episodes until a person has flown and chosen to record. This does
not start Docker or PX4, and it does not set `OHHO_ARM_HARDWARE`.

## Schema

`meta/info.json` uses `codebase_version` `v2.1` and `robot_type` `px4-sitl`
(or `crazyflie-sim` / `px4-x500` when that spec is what was recorded).

| Column | Contents |
|---|---|
| `observation.state` | `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `vx`, `vy`, `vz`, `yaw_rate`, `battery`. 11 floats. Names are stored in `info.json`. |
| `action` | Velocity setpoint that was sent: `vx`, `vy`, `vz`, `yaw_rate`. 4 floats. |
| `action_frame` | Constant `BODY_FRD` on the dataset. Not a per-frame column. Positive `vz` is down. |
| `observation.images.front` | Optional. Present only when a camera frame was attached. Absent in M1 sim. |
| `observation.lidar` | Not in the dataset. A point cloud is not written into an image column. |

`action` is the setpoint `offboard_velocity` handed to the transport, read
back from the command the stub kept. On the ground, and while takeoff,
landing, or RTL owns the vehicle, the stub ignores a velocity setpoint and
the action is zeros. `action_raw` keeps the requested numbers.

`observation.state` is the pose, velocity, and battery fraction the backend
published in `Telemetry.custom["drone"]`, plus `Telemetry.battery`. `drone-sim`
publishes pose and velocity in `LOCAL_NED` (z down, so a hover at 0.4 m is
about `z = -0.4`). That is not the body-frame action.

Task strings for these sim episodes are `sim takeoff hover land` and
`sim square`.

## Record

```python
from ohho.data import record_aerial_teleop_episodes

path, stats = record_aerial_teleop_episodes("./datasets/px4-sitl")
```

That opens `SimServe` on `px4-sitl` / `sim://`, arms, climbs, records a few
episodes, and lands. The first episode hovers. Every second episode sends a
flat `BODY_FRD` square (forward, right, back, left). `websockets` is the
`[sim-serve]` extra.

A test or script can also wrap a `Robot` on `DroneSimTransport` with
`Recorder` and call `offboard_velocity` directly. `Recorder._wrap_robot`
wraps that method; without the wrap the action column stays zero in flight.

## Open

```python
from ohho.data import DatasetReader

reader = DatasetReader(path)
frames = reader.load_episode(0)
```

`DatasetReader` needs no torch. With the `[train]` extra, the same directory
opens in `LeRobotDataset` (`root` is the dataset directory, `revision` is
`v2.1`).
