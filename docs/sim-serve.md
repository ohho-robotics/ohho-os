# Simulation WebSocket Server (`sim-serve`)

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`) or connects to ROS 2 through rosbridge.

## Protocol & Message Schema

### Client Messages (JSON)

Clients command the robot by sending JSON messages over the WebSocket connection:

| Field | Type | Description |
|---|---|---|
| `velocity` | `object` | Target velocity with linear (`x`, `y`) and angular (`z`) components. |
| `joints` | `object` | Mapping of joint names to target angles in radians (e.g. `{"arm_shoulder_pan": 0.5}`). |
| `deadman` | `boolean` | Deadman safety switch. |
| `estop` | `boolean` | Emergency stop latch request. |

**Safety Invariants:**
- If `deadman` is false, or no client message arrives for 300 ms, the commanded velocity is zero.
- `estop` latches until a message with `estop: false` arrives. While latched, commanded velocity remains zero regardless of `deadman` or `velocity` inputs.

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
Forwards velocity to `/cmd_vel` and reads `/odom` via rosbridge. Depends on the `[ros2]` extra.

> **Note:** Gazebo itself is launched from the OmniBot repo, not from this package.

### 3. Unsupported backends
```bash
ohho sim-serve --backend isaac
ohho sim-serve --backend mujoco
```
Both exit non-zero and print `not built`. `--viewer` on those backends does
not change that.

### 4. Native MuJoCo window

```bash
ohho sim-serve --robot <id> --viewer
```

Headless is the default. `--viewer` opens MuJoCo's passive viewer
(`mujoco.viewer.launch_passive`) when `<id>` uses an in-process MuJoCo sim,
and the sim loop syncs the window. `--backend ros2` rejects the flag. A
session with no display exits with an error. On Windows, a process that
cannot create an OpenGL context exits with an error as well, because
``glfwCreateWindow`` would abort the process.

On macOS, run the same command under MuJoCo's launcher so the window is
created on the main thread:

```bash
mjpython -m ohho sim-serve --robot <id> --viewer
```
