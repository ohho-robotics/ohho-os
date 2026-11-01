# Simulation WebSocket Server (`sim-serve`)

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`), a live robot via `Robot.connect` (`--backend robot`), or ROS 2 through rosbridge.

## Protocol & Message Schema

### Client Messages (JSON)

Clients command the robot by sending JSON messages over the WebSocket connection:

| Field | Type | Description |
|---|---|---|
| `velocity` | `object` | Target velocity with linear (`x`, `y`, and for aircraft `z`) and angular (`z`) components. |
| `frame` | `string` | Aircraft setpoint frame: `BODY_FRD` (teleop), `LOCAL_NED`, or `LOCAL_ENU`. Ground sessions ignore it. |
| `joints` | `object` | Mapping of joint names to target angles in radians (e.g. `{"arm_shoulder_pan": 0.5}`). |
| `deadman` | `boolean` | Deadman safety switch. Omitted means false. |
| `estop` | `boolean` | Emergency stop latch request. |
| `mode` | `string` | Optional legged posture command: `stand`, `lie_down`, or `recover`. |
| `flight` | `object` | Aircraft request. Ignored for a robot without the `aerial` capability. |

**Safety Invariants:**
- If `deadman` is false, or no client message arrives for 300 ms, the commanded velocity is zero.
- On an aircraft that zero is a single offboard velocity, then the server **stops the setpoint stream**. It does not keep publishing the last non-zero command, and it does not land. On PX4 the silence trips the offboard-loss failsafe (`COM_OBL_RC_ACT`). A zero setpoint left streaming would be hover, which is not what this window does.
- `estop` latches until a message with `estop: false` arrives. While latched, no setpoint is published. The server calls `emergency_stop`. In the in-process drone sim that lands the aircraft and then disarms on the ground.
- `takeoff` and `arm` on `--backend robot` with a non-sim URI require the SafetyGate phrase and `OHHO_ARM_HARDWARE=1` already set in the server process by a person. The WebSocket payload cannot set that variable and cannot arm the gate (`Robot.arm`). `backend=sim`, and `backend=robot` with `sim://`, may take off without the phrase.

Example client message:
```json
{
  "velocity": {
    "linear": {"x": 0.2, "y": 0.0},
    "angular": {"z": 0.1}
  },
  "joints": {
    "arm_shoulder_pan": 0.5
  },
  "deadman": true,
  "estop": false
}
```

### Server Messages (JSON)

The server periodically broadcasts robot telemetry to all connected clients:

| Field | Type | Description |
|---|---|---|
| `protocolVersion` | `string` | OhhO wire protocol semver (`ohho.PROTOCOL_VERSION`, see `schemas/manifest.json`). |
| `odom` | `object` | Current pose containing `{"x": float, "y": float, "yaw": float}`. |
| `joints` | `object` | Present joint positions `{"<name>": float, ...}`. |
| `backend` | `string` | Active backend identifier (`"sim"`, `"ros2"`, or `"robot"`). |
| `posture` | `string` | Optional human-readable posture when reported by the robot. |
| `battery` | `number` | Optional state of charge (0–1) when reported. |
| `drone` | `object` | Optional aerial state from `Telemetry.custom["drone"]` (pose, landed state, flight mode). |
| `streaming` | `boolean` | Aircraft only. True while offboard setpoints are being repeated. |
| `frame` | `string` | Aircraft only. Frame of the active velocity stream, once the client has named one. |
| `flight_status` | `object` | Result of the latest `flight` request: `request`, `accepted`, `reason`. `accepted` means the request was forwarded. On a SafetyGate it is false when the gate's `last_error` is set. |
| `command_error` | `string` | Present when a velocity was refused (`frame`, `bad_velocity`). |
| `camera` | `string` | Base64-encoded JPEG image frame (optional). |
| `no_camera` | `boolean` | Set to `true` when no camera feed is attached. |

Example server message:
```json
{
  "protocolVersion": "1.0.0",
  "odom": {
    "x": 0.125,
    "y": 0.0,
    "yaw": 0.05
  },
  "joints": {
    "arm_shoulder_pan": 0.5
  },
  "backend": "sim",
  "no_camera": true
}
```

---

## Commands

### 1. In-process simulation
```bash
ohho sim-serve --backend sim --port 8790
```
Serves the existing in-process `sim://` robot on that WebSocket, with no ROS.

Serves the existing in-process `sim://` robot on that WebSocket, with no ROS.

The default port is **8790**. Port 8765 is the Foxglove bridge in OmniBot (`robot.launch.py`, `simulation.launch.py`); sim-serve does not bind it unless you pass `--port 8765` yourself.

### 2. Live robot
```bash
ohho sim-serve --backend robot --robot unitree-go2 --transport 'dds://lo?domain=1'
```
Connects through `Robot.connect` with the given transport URI. Hardware links use `SafetyGate` and stay disarmed until an operator arms them. A loopback or simulated link (`dds://lo` on a domain other than 0, or `sim://`) is armed automatically so teleop can drive the sim.

### 3. ROS 2 bridge
```bash
ohho sim-serve --backend ros2 --rosbridge ws://localhost:9090
```
Forwards velocity to `/cmd_vel` and reads `/odom` via rosbridge. Depends on the `[ros2]` extra.

> **Note:** Gazebo itself is launched from the OmniBot repo, not from this package.

### 4. Unsupported backends
```bash
ohho sim-serve --backend isaac
ohho sim-serve --backend mujoco
```
Both exit non-zero and print `not built`.

---

## Aerial teleop

Use the in-process aircraft with the existing sim backend:

```bash
ohho sim-serve --backend sim --robot px4-sitl --port 8765
```

`px4-sitl` and `crazyflie-sim` select the kinematic `drone-sim` stub. The server calls `step(dt)` on that stub. Nothing here flies a real aircraft.

### Velocity and frame

Teleop velocity is `BODY_FRD`: `linear.x` forward, `linear.y` right, `linear.z` down, `angular.z` nose-right. Positive `angular.z` is not negated. Stick-up (climb) is a **negative** `linear.z`, and that conversion belongs to the client. The server does not invent a frame when `frame` is missing or unknown, and it does not mix ENU with NED.

`LOCAL_NED` and `LOCAL_ENU` are accepted when the client names them. They are passed through to `Robot.offboard_velocity`.

While `deadman` is true and a client message arrives at least every 300 ms, the server repeats the setpoint at the telemetry rate. That is the offboard stream. After a loss it publishes **one** zero in the same frame, drops the retained setpoint, and publishes nothing further until the next live deadman message. `land` is not called.

A ground robot still uses planar `drive(vx, vy, w)`. `linear.z` and `frame` do not change that path. The ros2 backend publishes `geometry_msgs/Twist` on `/cmd_vel`, including `linear.z`, and keeps publishing a zero Twist on loss because that backend has no offboard stream.

### Flight requests

```json
{
  "velocity": {"linear": {"x": 0.1, "y": 0.0, "z": 0.0}, "angular": {"z": 0.0}},
  "frame": "BODY_FRD",
  "deadman": true,
  "estop": false,
  "flight": {
    "request": "takeoff",
    "alt_m": 2.5,
    "confirm": "I am physically present, the area is clear, the remote is in my hand"
  }
}
```

`flight.request` is one of `takeoff`, `land`, `rtl`, `hold`, `arm`, `disarm`.

| Request | What the server calls |
|---|---|
| `arm` | `Robot.arm_flight` (autopilot arm). Never `Robot.arm`. |
| `disarm` | `Robot.disarm_flight` |
| `takeoff` | `Robot.takeoff(alt_m)` |
| `land` | `Robot.land` |
| `rtl` | `Robot.rtl` |
| `hold` | `Robot.hold` |

Omitted `alt_m` is 3 m. A request above 3 m is refused (`reason: "altitude"`). `alt_m` is positive up.

`drone-sim` ignores takeoff until `arm` has been accepted, and it ignores velocity until the vehicle is in the air and not in takeoff, land, or RTL. Deadman loss does not cancel an autopilot takeoff, land, or RTL; those are modes, not the offboard stream.

### Confirmation

| Session | `takeoff` / `arm` |
|---|---|
| `--backend sim` | Accepted without the phrase. |
| `--backend robot --transport sim://` | Accepted without the phrase. The transport is still the simulator. |
| `--backend robot` with any other URI | Refused unless `confirm` is the SafetyGate arm phrase **and** `OHHO_ARM_HARDWARE=1` is already in the server's environment. |

The phrase is `I am physically present, the area is clear, the remote is in my hand`. A person sets the variable in their own terminal before starting the server. A payload field named `OHHO_ARM_HARDWARE`, `env`, or `confirm` cannot create it. `land`, `rtl`, `hold`, and `disarm` do not require the phrase; the gate still applies its own rules when the transport is gated.

`flight_status.accepted: true` means the call was forwarded. On a gated robot the gate can still refuse it (`not_armed`, `no_tty`, `battery`, `link`, and the rest of `SafetyGate.last_error`). The socket does not arm the gate, does not type the phrase into the TTY prompt, and does not bypass the gate. A hardware flight therefore still needs a person at that TTY. This process never sets `OHHO_ARM_HARDWARE`.

### E-stop

`estop: true` calls `emergency_stop` and stops the stream without publishing another setpoint. In `drone-sim` the stub freezes horizontal motion and descends until it is on the ground, then disarms. It does not cut motors or drop the altitude in one step. `estop: false` calls `release_stop`, which clears the latch and does not arm the autopilot.

### Reading the aircraft

`odom` stays the planar shadow (`x`, `y`, `yaw`). Altitude is `drone.pose.z` in `drone.pose_frame`. For `drone-sim` that frame is `LOCAL_NED`, so height above the origin is `-drone.pose.z`. `drone.landed_state` is `on_ground`, `in_air`, `takeoff`, `landing`, or `unknown`. `unknown` is not on the ground. `drone.experimental` stays true. `drone.flight_ready` stays false.
