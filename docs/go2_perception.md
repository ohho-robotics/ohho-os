# Unitree Go2 Perception & Spatio-Temporal Memory (OHH-125)

This document describes the front-camera perception pipeline for the Unitree Go2 quadruped, wiring camera frames to Claude-vision (`VlmPerceptor`), projecting 2D detections into world coordinates using robot odometry, and persisting them in spatio-temporal memory (`SpatialMemory`).

---

## 1. Architecture Overview

```
   Go2 Front Camera / MuJoCo Render
                 │ (JPEG bytes)
                 ▼
      UnitreeDdsTransport.get_camera_frame()
                 │
                 ▼
           VlmPerceptor (≤ 1 Hz rate-limited)
                 │ Claude Vision (JSON detections with label, conf, image_x)
                 ▼
      Coordinate Projection (odom + bearing + fixed range)
                 │ (x_world, y_world, bearing, range)
                 ▼
      SpatialMemory (~/.ohho/memory/unitree-go2.json)
```

1. **Camera Frame Extraction**: `bot.get_camera_frame()` or `UnitreeDdsTransport.get_camera_frame()` fetches compressed JPEG bytes.
2. **Vision-Language Model (VLM)**: `VlmPerceptor` sends the image frame to Claude vision, prompting for detected objects with normalized horizontal coordinates `image_x` in $[0, 1]$ (0 = left, 0.5 = center, 1 = right).
3. **World Coordinate Projection**: `project_detection` computes the horizontal bearing relative to the robot heading from `image_x` and the camera field-of-view, projecting the detection into world coordinates using current robot odometry and a fixed-range assumption.
4. **Spatio-Temporal Memory**: `remember_detections` commits the detections to `SpatialMemory`, maintaining object permanence, position tracking, and historical event timelines.

---

## 2. Sim vs Real Go2 Camera Differences

| Mode | Frame Source | Transport / Mechanism |
|---|---|---|
| **Real Go2 Hardware** | Front camera sensor | `unitree_sdk2py.go2.video.video_client.VideoClient` over CycloneDDS (domain 0). Requires hardware arming. |
| **MuJoCo Simulation** | Head-camera offscreen context | Offscreen render buffer or sim-only ROS 2 / DDS topic / local IPC socket (e.g. `dds://lo?domain=1`). Injected via `frame_factory`. |
| **Test & Mock Mode** | Synthetic JPEG frames | Loopback on domain != 0 (`dds://lo?domain=1 --mock` or `OHHO_STUB_DDS=1`). |

### Hardware Safety Notice
- **Never run tests against domain 0 hardware.** Domain 0 connects to physical robots and requires interactive hardware arming (`OHHO_ARM_HARDWARE=1`).
- Simulated and test environments run on loopback interfaces with `domain != 0` (e.g. `lo?domain=1`).

---

## 3. World Coordinate Projection

The front camera on the Unitree Go2 provides a wide horizontal field of view ($\approx 90^\circ$ to $120^\circ$, default $\text{FOV} = \frac{\pi}{2}$ rad):

Given:
- Normalized horizontal detection center: $x \in [0, 1]$
- Camera horizontal FOV: $\text{FOV}$ (default $90^\circ$ = $1.5708$ rad)
- Robot odometry pose: $(p_x, p_y, \theta)$
- Range assumption: $r$ (default $1.5$ m, or detection range if provided)

### Bearing Calculation
In standard robotics conventions, counter-clockwise (left) is positive:
$$\text{bearing} = (0.5 - x) \times \text{FOV}$$

- $x = 0.5$ (image center) $\implies \text{bearing} = 0.0$ rad
- $x = 0.25$ (left of center) $\implies \text{bearing} = +0.25 \times \text{FOV} > 0$
- $x = 0.75$ (right of center) $\implies \text{bearing} = -0.25 \times \text{FOV} < 0$

### World Coordinates
$$\theta_{\text{world}} = \theta + \text{bearing}$$
$$x_{\text{world}} = p_x + r \times \cos(\theta_{\text{world}})$$
$$y_{\text{world}} = p_y + r \times \sin(\theta_{\text{world}})$$

---

## 4. Rate-Limiting & Cost Monitoring

- **Rate Limit**: Calls to the VLM are throttled to at most **1 Hz** (`rate_limit_hz=1.0`) by default. Successive calls within the minimum interval sleep for the remaining duration.
- **Latency Tracking**: Each inference records duration in seconds (`duration_s`) and milliseconds in `VlmCallStats`.
- **Token & Cost Estimation**: Token counts from Claude's usage metadata are tracked and cost estimated via `estimate_vlm_cost(model, input_tokens, output_tokens)`.
- **Hooks**: Callers can register a `stats_hook: Callable[[VlmCallStats], None]` to receive live telemetry for monitoring dashboards and audit logging.

---

## 5. CLI Usage

### Look around and remember objects:
```bash
# On simulated Go2 (loopback domain 1):
ohho look unitree-go2 --transport dds://lo?domain=1 --mock

# Query remembered memory:
ohho memory show unitree-go2
ohho memory where unitree-go2 chair
ohho memory near unitree-go2 --x 2.0 --y 3.0 --radius 1.5
```

---

## 6. Evaluation Status & TODOs

- [x] UnitreeDdsTransport `get_camera_frame()` capability with `frame_factory` injection
- [x] VlmPerceptor wiring with normalized coordinate parsing
- [x] Odometry + bearing world projection with fixed-range assumption
- [x] Rate-limiting to $\le 1$ Hz and latency/cost telemetry hooks
- [x] Integration with `SpatialMemory` and `ohho look` CLI
- [x] Safety: domain 0 protection and loopback testing on domain 1
- [ ] **HUMAN BENCH TODO**: Run full office-scene 4/5 memory accuracy evaluation in live MuJoCo environment / bench hardware. (Do not claim benchmark accuracy until bench validation is run by a human operator).
