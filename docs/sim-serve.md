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
| `modes` | `object` | Optional posture request: `{"request": "<mode>", "confirm": "<phrase>"}`. |

`modes.request` must be a `humanoid.posture` value that spec declares (`prepare`, `walk`, `damping`, `custom`, `get_up`, `lie_down`). `walk_mode` is accepted when `walk` is declared, and `damp` when `damping` is declared. Anything else is refused. Selection uses the capability and that list, not a robot id. `backend=sim` does not need `confirm`. A released deadman, or 300 ms of silence, sends zero velocity and leaves the posture unchanged.

**Safety Invariants:**
- If `deadman` is false, or no client message arrives for 300 ms, the commanded velocity is zero. Posture is not changed on that path.
- A posture request is applied only while the deadman is held and e-stop is clear.
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

The server broadcasts robot telemetry to all connected clients at about 30 Hz (`--rate`, default 30).

| Field | Type | Description |
|---|---|---|
| `odom` | `object` | Planar pose `{"x": float, "y": float, "yaw": float}`. |
| `pose` | `object` | `position` (`x`, `y`, `z` metres) and `orientation` quaternion (`x`, `y`, `z`, `w`). |
| `joints` | `object` | Present joint positions `{"<name>": float, ...}`. |
| `joint_state` | `object` | `names` in descriptor order and `positions` in radians, one value per name. A joint the transport has not reported is `0`. |
| `status` | `object` | `backend`, `estop`, `deadman`, `connected`, and `posture` (generic mode string, or the last accepted action). |
| `backend` | `string` | Active backend identifier (`"sim"` or `"ros2"`). |
| `modes` | `object` | Last posture decision, only after a `modes` request. |
| `camera` | `string` | Base64-encoded JPEG image frame (optional). |
| `no_camera` | `boolean` | Set to `true` when no camera feed is attached. |

`joint_state.names` is the humanoid descriptor's `joints` list when that document has one, otherwise `spec.joint_names`. The same order is what a viewer should apply.

A browser page on `https://` cannot open `ws://` to a host that is not loopback (mixed content). Chromium treats `ws://127.0.0.1` and `ws://localhost` as trustworthy, and a public https site may still have to ask before it connects to loopback. `ohho sim-serve` speaks `ws://` only. If the page cannot connect, run it from `http://127.0.0.1`.

`--backend sim --robot <id>` uses that spec's sim backend. A humanoid spec whose sim backend is MuJoCo is served that way. The CLI names `isaac` and `mujoco` are still not built and exit non-zero. Plugin ids and the start command are in `plugins/ohho-booster/README.md`.

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
Both exit non-zero and print `not built`.
