# MAVLink backend (OHH-305)

Experimental. Nothing has flown. `flight_ready` stays false.

The backend is `ohho/adapters/mavlink_drone.py`. It talks to PX4 (and, later,
ArduPilot) through **native MAVSDK 4.x**. There is no `mavsdk_server` process
and no gRPC. Install it with the optional extra:

```bash
pip install -e '.[mavlink]'
```

That extra is `mavsdk>=4,<5` plus `pymavlink`. The base install stays
dependency-free: `dependencies = []`. Importing the adapter does not import
`mavsdk`. `connect()` without an injected link does, and raises
`AdapterUnavailable` when the extra is missing.

`px4-x500` is still a registry spec only. `resolve_transport` refuses every
URI for it, including these.

## URIs

| Ohho URI | MAVSDK 4 `add_any_connection` string |
|---|---|
| `mavlink+udp://127.0.0.1:14540` | `udpin://127.0.0.1:14540` |
| `mavlink+udp://HOST:PORT` | `udpin://HOST:PORT` |
| `mavlink+serial:///dev/ttyACM0?baud=921600` | `serial:///dev/ttyACM0:921600` |
| `mavlink+serial://COM3?baud=57600` | `serial://COM3:57600` |

Grammar checked against MAVSDK 4 (package `mavsdk` 4.x, not `mavsdk-grpc`):

* `udpin://[ip][:port]` — ground station listens. PX4 SITL sends the offboard
  stream to port `14540`, so `mavlink+udp://127.0.0.1:14540` is that listen.
* `serial://[path][:baudrate]` — UART. The Ohho query `baud` defaults to
  `921600` when it is omitted.

`udpin` is listen, not `udpout`. A companion that must dial out is not this
URI.

## Loopback and hardware arming

Loopback is decided by `ohho.drone.mock_link.simulation_target_allowed`, not
by hoping the scheme string looks like a simulator.

| Target | `simulation_target` | `requires_hardware_arming` on the transport |
|---|---|---|
| `mavlink+udp://127.0.0.1:14540`, `localhost`, `::1` | true | false |
| `mavlink+udp://host.docker.internal:14540?sitl=1` | true | false |
| any other UDP host | false | true |
| `mavlink+serial://…` | false | true |

`Robot.connect` still wraps `protocol="mavlink"` in `SafetyGate`, and the
gate's own `requires_hardware_arming` does not yet read the transport flag
(this change does not edit `ohho/safety.py`). Until that follow-up, arming
the gate on a loopback SITL link still follows the hardware rule. The
transport flag is the explicit check the follow-up should call.

The aerial profile for `px4-sitl` is `DRONE_PROFILE`. E-stop goes through
`ohho.drone.estop.estop_commands`:

* `in_air`, `takeoff`, `landing` → land
* `unknown` → hold, then land
* disarm only when `landed_state` is `on_ground` and positive-up altitude is
  a finite number of metres `<= 0.25`

Disarm is `MAV_CMD_COMPONENT_ARM_DISARM` with param1 `0` and param2 `0`.
The backend never calls MAVSDK `kill`, `terminate`, or `arm_force`. `kill`
is the call that sends force-disarm param2 `21196`. Flight termination is
never sent. A `force=True` argument on `disarm` is ignored.

`release_stop` clears the Ohho latch only. It does not arm the autopilot
and it does not resume offboard.

## Frames

Callers pass `ohho.aerial.Frame`. There is no default.

| Ohho frame | What is published |
|---|---|
| `BODY_FRD` | `MAV_FRAME_BODY_NED` (id 8). Numbers are not rotated. Velocity uses MAVSDK `set_velocity_body` (`VelocityBodyYawspeed`: forward, right, down, yaw speed in deg/s). Positive yaw rate is nose-right. |
| `LOCAL_NED` | `MAV_FRAME_LOCAL_NED` (id 1). Position uses MAVSDK `set_position_ned` (yaw in degrees, clockwise from north). Velocity keeps yaw **rate** on `SET_POSITION_TARGET_LOCAL_NED` (radians/s). The rate is not written into an absolute yaw field. |
| `LOCAL_ENU` | Converted to `LOCAL_NED` first (swap x/y, negate z, `yaw_ned = pi/2 - yaw_enu`, yaw rate sign flips), then published as `MAV_FRAME_LOCAL_NED`. |

`BODY_FRD` position and yaw-only setpoints are the same pymavlink gap: MAVSDK's
position setter is local NED, not body NED. pymavlink encodes
`SET_POSITION_TARGET_LOCAL_NED`. The mock used by unit tests records the frame
id and the numbers without importing either library.

`send_velocity` is planar body FRD (forward, right, yaw rate, climb held at
zero). A zero velocity is hover, not land.

Telemetry is `Telemetry.custom["drone"]` in `LOCAL_NED`, with `experimental`
true and `flight_ready` false. `Telemetry.odom` is the planar shadow.

## PX4 SITL square (local recipe, not run by CI)

The GitHub job `.github/workflows/sitl-mavlink.yml` only prints this recipe.
It does not start Docker, PX4, or a MAVLink socket. A later job can run the
square. Do not set `OHHO_ARM_HARDWARE` for loopback SITL.

1. Start PX4 SITL yourself, headless, model `gz_x500`, image pinned (not
   `latest`). One known image is `px4io/px4-sitl-gazebo:v1.17.0`. This
   recipe does not launch it.
2. `pip install -e '.[mavlink]'` (native `mavsdk` 4.x).
3. Connect with `mavlink+udp://127.0.0.1:14540`
   (`udpin://127.0.0.1:14540`).
4. Arm the safety gate (see the hardware-arming note above), stream an
   offboard setpoint, then `arm_flight`.
5. `takeoff(2.5)` and wait until the NED down coordinate is within 0.4 m of
   `-2.5` for 2 seconds.
6. Fly a square of 4 m sides in `LOCAL_NED` at 0.5 m/s, altitude held.
   Corners within 0.8 m horizontal.
7. `land`. Pass when `landed_state` is `on_ground`, motors are disarmed with
   a normal disarm, and no failsafe flag was set after takeoff.

`python scripts/sitl_mavlink_square.py --print-recipe` prints the same steps
and exits 0. It does not connect.

D-22 (ArduPilot SITL, the same square through this URI) waits on this backend.
