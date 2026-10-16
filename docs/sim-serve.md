# Simulation WebSocket Server (`sim-serve`)

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`) or connects to ROS 2 through rosbridge.

## Protocol & Message Schema

### Client Messages (JSON)

Clients command the robot by sending JSON messages over the WebSocket connection:

| Field | Type | Description |
|---|---|---|
| `velocity` | `object` | Target velocity. Twist style `linear.x` / `linear.y` / `angular.z`, or Connect fields `linearX` / `linearY` / `angularZ`. |
| `joints` | `object` | Mapping of joint names to target angles in radians (e.g. `{"arm_shoulder_pan": 0.5}`). |
| `deadman` | `boolean` | Deadman safety switch. |
| `estop` | `boolean` | Emergency stop latch request. |

**Safety Invariants:**
- `deadman` must be JSON `true`. A missing field, `false`, or any other value commands zero velocity immediately.
- If no client message arrives for 300 ms, the commanded velocity is zero. That watchdog is independent of the telemetry rate.
- If the WebSocket drops and no client remains, velocity is zero immediately (and always within 300 ms).
- `estop` latches until a message with `estop: false` arrives. While latched, commanded velocity remains zero regardless of `deadman` or `velocity` inputs.
- A server message includes `camera` (base64) only when a front-camera frame has arrived. Otherwise it includes `no_camera: true` and no `camera` field.

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
| `odom` | `object` | Current pose containing `{"x": float, "y": float, "yaw": float}`. |
| `joints` | `object` | Present joint positions `{"<name>": float, ...}`. |
| `backend` | `string` | Active backend identifier (`"sim"` or `"ros2"`). |
| `camera` | `string` | Base64-encoded JPEG image frame (optional). |
| `no_camera` | `boolean` | Set to `true` when no camera feed is attached. |

Example server message:
```json
{
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
ohho sim-serve --backend sim --port 8765
```
Serves the existing in-process `sim://` robot on that WebSocket, with no ROS.

### 2. ROS 2 bridge
```bash
ohho sim-serve --backend ros2 --rosbridge ws://localhost:9090
```
Forwards velocity to `/cmd_vel`, joint targets to `/arm/joint_commands`, and e-stop to `/emergency_stop`. It reads `/odom`, `/joint_states` (and `/arm/joint_states`), and `/camera/front/image_raw` (plus the compressed image on that prefix). Depends on the `[ros2]` extra. Protocol tests use an in-process fake rosbridge and do not launch Gazebo.

> **Note:** Gazebo itself is launched from the OmniBot repo, not from this package.

### 3. Unsupported backends
```bash
ohho sim-serve --backend isaac
ohho sim-serve --backend mujoco
```
Both exit non-zero, print `not built`, and the Python entry point returns `not-implemented`. Neither imports Isaac Sim or MuJoCo.
