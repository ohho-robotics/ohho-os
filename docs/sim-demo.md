# Simulation Demo Harness (OHH-134)

This document specifies the run harness, shot list, video recording checklist,
and evaluation JSON schema for the **OHH-134 Sim Demo** (typed goal -> agent ->
Unitree Go2 quadruped in MuJoCo simulation).

> **WARNING: SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE.**
> All behaviors shown run in MuJoCo simulation via local DDS loopback
> (`dds://lo?domain=1`) or in-process simulation (`sim://`).
> Nothing in this repository or demo has been validated on a physical Unitree Go2.
> Real hardware bring-up is tracked as **Vision** until a physical Go2 EDU is
> available and an operator is physically present (**OHH-135**).

---

## 1. Demo Overview

The OHH-134 demo demonstrates the full OhhO stack on a quadruped form factor:
1. **System readiness & discovery**: Probing the environment, runtimes, adapters, and `unitree-go2` capabilities (`ohho doctor`).
2. **Simulation loopback connection**: Connecting over DDS loopback (`dds://lo?domain=1`) to the MuJoCo sport shim without touching hardware domain 0.
3. **Natural language typed goal**: Accepting high-level intent (`"Stand up, drive forward 0.3 m, then stop and lie down."`).
4. **Agent reasoning & streamed tool calls**: Breaking down the goal into typed capability tools (`stand_up`, `drive`, `stop`, `lie_down`) and driving locomotion in MuJoCo with live odometry feedback.
5. **SafetyGate enforcement**: Demonstrating the three core safety invariants:
   - Forward speed capped to Go2 limit (0.5 m/s ceiling).
   - Prohibited dangerous sport actions (`backflip`, `jump`) refused by safety policy.
   - Emergency stop latches and disables all motion commands.
6. **Honest end card**: Prominently declaring simulation-only status, safety guarantees, and open-source repository links.

Target clip duration: **60--90 seconds** (scripted timing: ~85 seconds).

---

## 2. Shot List & Video Overlays

| Shot | Timecode | Duration | Title / Description | Visual Layout | On-Screen Graphic Overlay |
|---|---|---|---|---|---|
| **1** | `0:00 - 0:10` | 10s | **Doctor & System Readiness**<br>Probe runtimes, adapters, Go2 spec | Terminal full screen | `[OHHO OS v1.1.3] [SIMULATION ONLY -- NOT TESTED ON HARDWARE]`<br>*Lower third:* "Shot 1: System Readiness & Robot Discovery" |
| **2** | `0:10 - 0:22` | 12s | **Sim & Sport Shim Start**<br>Connect to MuJoCo on domain 1 | Split screen (50% Terminal, 50% MuJoCo) | `[SIMULATION ONLY -- DOMAIN 1 LOOPBACK] [HARDWARE MOTION: DISARMED]`<br>*Callout:* "SafetyGate: Fail-closed until armed on loopback" |
| **3** | `0:22 - 0:34` | 12s | **Natural Language Typed Goal**<br>User enters prompt | Split screen | `Goal: "Stand up, drive forward 0.3 m, then stop and lie down."`<br>*Tag:* `[AGENT HARNESS REASONING]` |
| **4** | `0:34 - 0:54` | 20s | **Tool-Call Stream & Motion**<br>Go2 stands, walks 0.3m, stops, lies down | Split screen (Left: tool calls, Right: Go2 motion) | `[TOOL CALLS & MOTION] stand_up() -> drive(0.2) -> stop() -> lie_down()`<br>*Live odom counter:* `x = 0.10 -> 0.20 -> 0.30 m` |
| **5** | `0:54 - 1:12` | 18s | **SafetyGate Clamp & E-Stop**<br>2.0 m/s clamped to 0.5; backflip denied; E-stop | Split screen | `[SAFETYGATE DEMO: 2.0 m/s CLAMPED TO 0.5 m/s \| BACKFLIP DENIED \| E-STOP LATCHED]`<br>*Alert banner:* "Safety Invariants Verified" |
| **6** | `1:12 - 1:25` | 13s | **End-Card Disclaimer & Links**<br>Full honest disclaimer and repos | Full screen graphic / terminal card | `[SIMULATION ONLY -- NOT TESTED ON PHYSICAL HARDWARE]`<br>`github.com/ohho-robotics/ohho-os`<br>`github.com/ohho-robotics/ohho-quadruped` |

---

## 3. How Varun Records the 60--90s Clip

### Screen Setup & OBS Configuration
1. **Resolution**: 1920x1080 @ 60 FPS.
2. **Window Arrangement**:
   - **Left half (50%)**: Terminal (width 80--90 cols, font size ~18pt, high contrast theme).
   - **Right half (50%)**: MuJoCo simulation window (Go2 model centered, camera angled at ~45 degrees).
3. **Persistent Watermark**: Place top-right text banner: `SIMULATION ONLY -- NOT TESTED ON HARDWARE`.

### Execution Steps

#### Terminal 1: Launch MuJoCo sim and sport shim (from `ohho-quadruped`)
```bash
# In WSL2 or Linux dev environment with unitree_mujoco installed:
python sim/run_go2.py --scene flat
# In a second tab or background job:
python -m ohho_go2.sport_shim --domain 1 --iface lo
```

#### Terminal 2: Run the OhhO OS Demo Harness (from `ohho-os`)
```bash
# Print overlay cues and production checklist:
ohho sim-demo --check-overlays

# Run the live demo against the MuJoCo sport shim:
ohho sim-demo --transport dds://lo?domain=1 --out evals/logs/ohh-134-demo-run.json

# Offline / automated CI mode (simulated loopback, no MuJoCo window required):
ohho sim-demo --sim --fast --out evals/logs/ci-eval.json
```

---

## 4. Evaluation JSON Schema & Example Log

Every demo run produces an evaluation report structured per the standard schema:
- **Schema file**: [`evals/schemas/sim_demo_schema.json`](../evals/schemas/sim_demo_schema.json)
- **Example log**: [`evals/examples/ohh-134-go2-mujoco-demo.json`](../evals/examples/ohh-134-go2-mujoco-demo.json)

### Validate a Run Against the Schema
```python
import json
import jsonschema

schema = json.load(open("evals/schemas/sim_demo_schema.json"))
report = json.load(open("evals/examples/ohh-134-go2-mujoco-demo.json"))
jsonschema.validate(report, schema)
print("Evaluation log schema check: PASS")
```

---

## 5. Link Instructions for `ohho-quadruped` README

To maintain repo hygiene without conflicting with concurrent quad PRs, do **not**
modify `ohho-quadruped` directly while other PRs are open.
Once the sibling PRs merge, insert the following section into `ohho-quadruped/README.md`
under `## Simulation`:

```markdown
### OhhO OS Sim Demo Harness (OHH-134)

The agent-side simulation harness connects OhhO OS to the Go2 MuJoCo sim and sport shim:

```bash
# Run the demo harness against the local sport shim (domain 1, interface lo):
ohho sim-demo --transport dds://lo?domain=1

# Run offline / automated CI validation:
ohho sim-demo --sim --fast
```

See [docs/sim-demo.md](https://github.com/ohho-robotics/ohho-os/blob/main/docs/sim-demo.md)
for the shot list, OBS recording checklist, and eval JSON schema.
```
