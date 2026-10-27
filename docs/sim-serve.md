# Simulation WebSocket Server (`sim-serve`)

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`), a live robot via `Robot.connect` (`--backend robot`), or ROS 2 through rosbridge.

## Protocol & Message Schema

### Client Messages (JSON)

Clients command the robot by sending JSON messages over the WebSocket connection:

| Field | Type | Description |
|---|---|---|
| `velocity` | `object` | Target velocity with linear (`x`, `y`) and angular (`z`) components. |
| `joints` | `object` | Mapping of joint names to target angles in radians (e.g. `{"arm_shoulder_pan": 0.5}`). |
| `deadman` | `boolean` | Deadman safety switch. |
| `estop` | `boolean` | Emergency stop latch request. |
| `mode` | `string` | Optional legged posture command: `stand`, `lie_down`, or `recover`. |

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
| `protocolVersion` | `string` | OhhO wire protocol semver (`ohho.PROTOCOL_VERSION`, see `schemas/manifest.json`). |
| `odom` | `object` | Current pose containing `{"x": float, "y": float, "yaw": float}`. |
| `joints` | `object` | Present joint positions `{"<name>": float, ...}`. |
| `backend` | `string` | Active backend identifier (`"sim"`, `"ros2"`, or `"robot"`). |
| `posture` | `string` | Optional human-readable posture when reported by the robot. |
| `battery` | `number` | Optional state of charge (0–1) when reported. |
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
Connects through `Robot.connect` with the given transport URI. Hardware links use `SafetyGate` (disarmed until `bot.arm()` in an interactive session).

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
