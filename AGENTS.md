# OhhO OS — Engineering Handoff & Roadmap

> **Read this first.** This file is the single source of truth for continuing
> work on the OhhO OS SDK. It is written so that a fresh AI agent (or a human)
> with no prior context can understand what exists, why it's built this way, and
> exactly what to do next. Keep it up to date as milestones land.

> **This checkout is the ohho-sdk tree.** The Python package is at the
> repository root. Paths written below as `sdk/` refer to that root
> (`tests/`, not `sdk/tests/`). The TypeScript workspace is in `ts/` and
> is unchanged; `tsc` fails there. The licence is Apache-2.0 (OHH-18).

- **Package:** `ohho-os` (imports as `ohho`) — repository root of
  [ohho-sdk](https://github.com/ohho-robotics/ohho-sdk).
- **What it is:** the open-source, robot-agnostic engine that powers the OhhO
  platform. One API controls any robot, **with or without ROS**, and carries the
  whole stack from perception to training.
- **Status:** **M0–M5 complete — v1.0.1 (verified).** All milestones shipped and
  independently verified from a fresh install: 151 unittest cases green in a
  fully-equipped env; in a **bare** env the optional-dep tests skip cleanly
  (agent_engine/numpy → brains tests; fastapi → the record→train→serve e2e; 4 HIL
  tests skip unless `OHHO_HIL=1`). `ohho` CLI works end-to-end with `doctor list
  version connect sim drive agent serve market profile`.
- **v1.0.1 hardening:** `Agent`/`ohho agent` no longer crash on a bare install —
  `brains.harness_available()` probes the real import chain and `HarnessBrain.run`
  falls back to `ScriptedBrain` with an explanatory log line; `ohho doctor` now
  reports agent-brain/serve/data/train availability dynamically (the stale
  "ROS 2 arrives later" line is gone — auto prefers ros2 when rclpy is present).
- **v1.1.0 — native nav + spatio-temporal memory:** the two former DimOS-parity
  gaps are closed. `ohho/nav.py` (occupancy-grid mapping from `Telemetry.scan`,
  costmap inflation, A*, frontier exploration, closed-loop `Navigator` with
  stuck detection + goal diversion; CLI `ohho nav goto|explore|map`) and
  `ohho/memory.py` + `ohho/perception.py` (entity store with object permanence,
  spatial/temporal queries, JSON persistence at `~/.ohho/memory/<robot>.json`;
  `SimPerceptor` + optional Claude-vision `VlmPerceptor`; CLI `ohho look`,
  `ohho memory show|where|near|clear`). The sim gained labeled obstacles,
  ray-cast scans and **solid-object collision**; the agent's tool registry gained
  `navigate_to, explore, look_around, where_is, objects_near, remember_note` and
  injects the memory summary into every goal. All stdlib — base stays dep-free.
- **Current version:** `1.1.3` (see `ohho/__init__.py` `__version__`).

---

## 0. TL;DR — current state

| Area | State |
|---|---|
| Robot Abstraction Layer (`Robot`) | ✅ working |
| Capability model + registry | ✅ working (OmniBot, Unitree Go2, generic sim) |
| Runtime port + **native** backend | ✅ working (threaded scheduler, pub/sub, timers, params) |
| **ROS 2** runtime backend | ✅ **M4 complete** — `Ros2Runtime` (rclpy node: timers, pub/sub via topics, params via node params) + `Ros2Transport` (bridges /cmd_vel, /odom, /arm/joint_states). `get_runtime("auto")` prefers ros2 when rclpy present. `ros2://` scheme registered. Fully testable with `FakeRos2Node` (no ROS 2 needed). |
| Simulator adapter | ✅ working (holonomic physics + joint servoing) |
| **Hardware adapters** | ✅ M1 software complete — `serial://` (Yahboom base) + `feetech://` (SO-101 arm) + `composite` (base+arm merge for mobile-manipulators) + `dds://` (Unitree Go2 with real state polling) all built and mock-tested. **On-robot bring-up still pending** (hardware-gated). DJI/others not built. |
| Agent loop | ✅ **real brain** — `HarnessBrain` wires `agent_engine.AgentHarness` (perceive→reason→act→reflect). `ToolRegistry` built from robot capabilities. `ClaudeToolCallingReasoner` over `ReasoningRouter` (cloud Claude when `ANTHROPIC_API_KEY` + `[agent]` extra, echo fallback otherwise). `ScriptedBrain` remains the no-dep fallback. |
| Training / data / serve | ✅ **M3 complete** — `ohho.data.Recorder` (capture state+action from Robot → LeRobot v2.0), `ohho.train.finetune()` (delegates to lerobot_engine, mock mode for sim), `ohho.serve` (FastAPI server, `ohho serve` CLI). Record→train→serve loop works on sim. |
| **Navigation (native, no-ROS)** | ✅ **v1.1.0** — `ohho.nav`: grid mapping, costmap, A*, frontiers, `Navigator` (collision-aware). CLI `ohho nav goto/explore/map` |
| **Memory + perception** | ✅ **v1.1.0** — `ohho.memory` (object permanence, temporal queries, JSON persistence) + `ohho.perception` (`SimPerceptor`, Claude-vision `VlmPerceptor`). CLI `ohho look`, `ohho memory` |
| **Simulation WebSocket (`sim-serve`)** | ✅ **OHH-91 complete** — `ohho.sim_serve` (JSON WebSocket session: velocity/joints/deadman/estop, 300ms timeout, estop latch; backends: `sim` and `ros2` via rosbridge; isaac/mujoco exit non-zero). CLI `ohho sim-serve` |
| **Hardware self-test (`selftest`)** | ⏳ **OHH-87 code + sim done; real-robot run pending (Varun)** — `ohho.selftest`: Bench v0 bring-up suite (serial link, wheel spin direction with encoder integration, IMU 20+ Hz rate, STS3215 servo diagnostics [ID/voltage/temp/pos], camera frames) with dated JSON report + `--sim` CI mode. Real-robot bench run pending (Varun). CLI `ohho selftest` |
| **Nightly CPU sim loop (`sim-loop`)** | ✅ **OHH-86 complete** — `ohho.sim_loop` (teleop record >=5 episodes -> LeRobot v2.0 dataset schema check + LeRobotDataset loading -> tiny ACT CPU train -> FastAPI serve -> >=50 closed-loop sim steps + latency p50/p95). CLI `ohho sim-loop` + `python scripts/sim_loop.py` + `.github/workflows/sim-loop.yml`. |
| **SafetyGate (`ohho.safety`)** | ✅ **OHH-117** — fail-closed `Transport` wrapper inserted by `Robot.connect` for every `protocol != "simulated"`: velocity/accel/joint/effort caps, 300 ms command + 500 ms state watchdogs, latched e-stop, sport-action allowlist, tilt/fault latch, hardware arming, JSONL audit. Rules in §9a. CLI `--arm --estop-damp --safety-config` |
| CLI (`ohho`) | ✅ `doctor list version connect sim drive agent serve sim-serve sim-loop market profile nav look memory selftest` |
| Tests | Python 3.12, Windows: `python -m unittest discover -s tests` → Ran 383 tests, OK (skipped=7). |
| CI | `.github/workflows/ci.yml` (multi-OS, Py 3.10-3.13; includes `ohho selftest --sim` smoke check and a ≥95% line+branch coverage gate on `ohho/safety.py`) + `.github/workflows/sim-loop.yml` (nightly CPU sim loop designed for CPU runners, first run pending). |

**M1 (two-robot hardware vertical slice) — software complete.** All four adapters
(`yahboom`, `feetech`, `composite`, `unitree`) are built and unit-tested against
the protocols with mock/loopback bytes (no hardware). The Unitree DDS state
subscription is wired (polling reader thread → `_on_state` → `Telemetry`). The
OmniBot arm adapter (Feetech bus) is built and composes with the Yahboom base via
`CompositeTransport`. What remains is **on-robot bring-up** only — running the HIL
tests against real OmniBot + Go2 hardware. See §8.

---

## 1. What OhhO OS is (vision & positioning)

OhhO OS is the **engine** beneath the OhhO product family — it is **not** a
product itself and does **not** rename or absorb the products (Build, Train,
etc.). It is the open-core that those consoles run on. Open-core model:

```
   ohho.com           OhhO Cloud (the ~19 paid consoles)         ← Stripe tiers
                                 │ same APIs
   pip install →      OhhO OS  (this package; Apache-2.0)  ← you are here
```

**The four differentiators (competitor-agnostic, as advertised on `/os`):**
1. **ROS optional — never required.** Native runtime *or* ROS 2, switch with one
   argument. (Most agentic robot stacks force one or the other.)
2. **Truly robot-agnostic.** Write a behavior once; run it on a wheeled base, a
   quadruped, a humanoid, or an arm. Swapping hardware ≠ rewriting code.
3. **The whole lifecycle, one engine.** Not just control — perception, data,
   training, sim, fleet, safety.
4. **Open source, zero lock-in.**

Marketing + user docs already shipped: `website/app/os/page.tsx` and
`website/docs/ohho-os/*.md` (install, quickstart, runtimes, robots, training,
architecture). **Keep the SDK and those docs in sync.**

---

## 2. Architecture

The core idea: **the robot and the middleware are swappable backends, not forks
of your code.**

```
   your code  ·  Agent (Mind)  ·  Train / Data / Serve     ← runtime-agnostic
            │
   ┌────────┴─────────┐  Robot Abstraction Layer (RAL = ohho.robot.Robot)
   │  Runtime (port)  │  pub/sub · timers · params
   └────────┬─────────┘
      ┌──────┴───────┐
  native runtime    ros2 runtime (stub)
  (no ROS)          (rclpy + ROS 2 stack)
  asyncio/threads   Nav2 · SLAM · MoveIt 2
  direct drivers    DDS multi-machine
            │
      ┌─────┴──────┐  Adapters (one per robot family)
   sim · yahboom · unitree · dji · …   (only `sim` exists today)
```

**THE ONE INVARIANT RULE (do not break):**
> Engines (Robot, agent, future train/perception) depend **only** on the
> `Runtime` port and the `Transport` interface — they must **never import
> `rclpy`** or a specific adapter directly. This is what makes "switch runtime /
> robot with one argument" true. New backends/adapters implement the interface;
> nothing above them changes.

**Capability-typed commands.** Commands are gated by capability. A robot that
lacks a capability turns the call into a safe no-op (e.g. `move_joints` on a
robot with no arm) instead of crashing. This is what makes one behavior portable
across heterogeneous hardware.

---

## 3. Current file map (`sdk/`)

```
sdk/
├── pyproject.toml        # name=ohho-os; extras: serial/arm/agent/data/unitree/dji/serve/train/ros2/yaml/dev/all; console_script `ohho`
├── README.md             # user-facing quickstart
├── AGENTS.md             # ← THIS FILE (handoff + roadmap)
├── .gitignore            # egg-info, __pycache__, caches
├── ohho/
│   ├── __init__.py       # exports Robot, connect, schema types, registry fns; __version__
│   ├── schema.py         # Velocity, Odometry, JointReading, Telemetry, TransportStatus, ConnectionState, clamp()
│   ├── capabilities.py   # capability string constants + ALL + is_known()
│   ├── registry.py       # RobotSpec (frozen dataclass) + _BUILTINS + get_spec/list_specs/register/load_manifest
│   ├── transport.py      # Transport (ABC) + BaseTransport (callback bookkeeping)
│   ├── runtime/
│   │   ├── __init__.py   # get_runtime("auto"|"native"|"ros2"), available_runtimes()
│   │   ├── base.py       # Runtime (ABC): pub/sub + params (concrete) + now/create_timer/start/stop (abstract); TimerHandle; RuntimeUnavailable
│   │   ├── native.py     # NativeRuntime — threaded scheduler
│   │   └── ros2.py       # Ros2Runtime — rclpy node (timers, pub/sub, params); node_factory injectable for tests
│   ├── adapters/
│   │   ├── __init__.py        # resolve_transport(uri, spec, runtime); available_adapters(); scheme map; composite auto-compose
│   │   ├── errors.py          # AdapterUnavailable
│   │   ├── sim.py             # SimTransport — holonomic physics + joint servoing; step(dt) deterministic
│   │   ├── _yahboom_proto.py  # vendored Yahboom packet codec (pure stdlib): encode + parse_stream
│   │   ├── yahboom.py         # YahboomTransport — serial:// (OmniBot base); pyserial lazy; _ingest() test seam
│   │   ├── feetech.py         # FeetechTransport — feetech:// (SO-101 arm); lerobot lazy; rad↔tick; torque; reader thread
│   │   ├── composite.py       # CompositeTransport — merges base + arm into one Transport (mobile-manipulators)
│   │   ├── ros2.py            # Ros2Transport — ros2:// (bridges /cmd_vel, /odom, /joint_states); msg_factory injectable
│   │   └── unitree.py         # UnitreeDdsTransport — dds:// (Go2); SDK lazy; state polling thread; state_factory injectable
│   ├── robot.py          # Robot (RAL) — connect/drive/move_joints/telemetry/emergency_stop/has/disconnect; connect() shortcut
│   ├── agent.py          # Agent + ScriptedBrain (fallback); _default_brain() lazily loads HarnessBrain
│   ├── brains.py         # HarnessBrain — wires agent_engine (ToolRegistry from caps, RobotPerceptor, ClaudeToolCallingReasoner)
│   ├── data/             # ohho.data — Recorder (capture state+action), LeRobot v2.0 writer + reader
│   │   ├── __init__.py   # exports Recorder, Episode
│   │   ├── recorder.py   # Recorder — wraps Robot, intercepts drive/move_joints, captures frames
│   │   ├── writer.py     # write_dataset — LeRobot v2.0 (Parquet via pyarrow, JSON Lines fallback)
│   │   └── reader.py     # DatasetReader — lightweight reader (no torch needed), stats()
│   ├── train/            # ohho.train — finetune() delegates to lerobot_engine, mock mode for sim
│   │   ├── __init__.py   # finetune(), SUPPORTED_POLICIES, TrainUnavailable, _mock_train()
│   │   └── act.py        # TinyACTPolicy + train_act() CPU Action Chunking Transformer
│   ├── serve/            # ohho.serve — FastAPI inference server, `ohho serve` CLI
│   │   ├── __init__.py   # serve(), build_app(), ServeUnavailable, _MockModel
│   │   └── act.py        # ACTModel inference adapter
│   ├── sim_serve.py      # ohho.sim_serve — WebSocket teleop and telemetry server (`sim` and `ros2`)
│   ├── sim_loop.py       # ohho.sim_loop — record -> validate -> tiny ACT train -> serve -> eval pipeline
│   ├── selftest.py       # ohho.selftest — hardware self-test suite (motors, encoders, IMU, servos, camera)
│   ├── safety.py         # ohho.safety — SafetyGate(Transport), SafetyProfile, AuditLog, arming policy
│   ├── profiles.py       # HardwareProfile, detect_profile(), `ohho profile` CLI
│   ├── market.py         # Skill registry, @skill decorator, run_skill(), `ohho market` CLI
│   ├── hardware.py       # resolve_device("auto") -> cuda/mps/cpu (lazy torch)
│   ├── cli.py            # argparse: doctor/list/version/connect/sim/drive/agent/serve/sim-serve/sim-loop/selftest; main()
│   └── robots/
│       └── example.json  # example manifest (nested dof/limits) for load_manifest()
├── docs/
│   ├── safety.md         # SafetyGate rules, profiles, config file, audit format
│   └── selftest.md       # Bench v0 self-test documentation, safety rules, JSON schema
├── scripts/
│   ├── check_classifiers.py
│   └── sim_loop.py       # one-liner reproducible sim loop runner
└── tests/                # unittest suite including test_sim_loop.py, test_sim_serve.py, test_selftest.py
```

---

## 4. Public API quick reference (as built — keep stable)

```python
from ohho import Robot, connect

bot = Robot.connect("omnibot")                 # transport=None -> spec adapter, falls back to sim
bot = Robot.connect("omnibot", "sim://")       # explicit transport URI
bot = Robot.connect("omnibot", runtime="native")  # "auto" | "native" | "ros2"

bot.drive(vx=0.2, vy=0.0, w=0.3)               # clamped to spec limits; vy ignored if not holonomic; no-op if no base.drive
bot.stop()
bot.move_joints([0, -0.5, 0.5, 0, 0, 0.2])     # no-op unless capabilities include "manipulation"
bot.emergency_stop(); bot.release_stop()
t = bot.telemetry()                            # Telemetry(odom, joints, battery, custom, timestamp)
bot.has("manipulation")                        # capability check
bot.status()                                   # TransportStatus
bot.disconnect()                               # also: `with Robot.connect(...) as bot:`

from ohho.agent import Agent
Agent(bot).run("explore the room")             # returns list[str] log lines (scripted brain for now)
```

**Schema (`ohho/schema.py`):** `Velocity(linear_x, linear_y, angular_z)`,
`Odometry(x, y, theta, vx, vy, omega)`, `JointReading(name, position, velocity?)`,
`Telemetry(odom?, joints[], battery?, custom{}, timestamp)`,
`TransportStatus(protocol, state, label, latency_ms, msg_rate, error?, connected_since?)`,
`ConnectionState` enum, `clamp(value, lo, hi)`.

**RobotSpec (`ohho/registry.py`):** `id, name, category, capabilities(tuple),
adapter, dof_base, dof_arm, max_lin, max_ang, joint_names(tuple)`, `.has(cap)`.
Built-ins: `omnibot` (mecanum mobile-manip, 6-DOF arm, max_lin 0.2),
`unitree-go2` (quadruped, max_lin 1.5), `sim` (generic holonomic).

**Capabilities (`ohho/capabilities.py`):** `base.drive`, `base.holonomic_drive`,
`legged.pose`, `manipulation`, `perception.rgb`, `perception.depth`.

**Transport interface (`ohho/transport.py`):** `connect()`, `disconnect()`,
`status()`, `read()->Telemetry`, `send_velocity(Velocity)`,
`send_joint_command(name, position)`, `emergency_stop()`, `release_stop()`,
`on_telemetry(cb)->unsub`, `on_status(cb)->unsub`. Subclass `BaseTransport` (it
does the callback bookkeeping). **New adapters = subclass `BaseTransport`.**

**Runtime port (`ohho/runtime/base.py`):** `subscribe/publish` (concrete),
`get_param/set_param` (concrete), `now()`, `create_timer(period_s, cb)->TimerHandle`,
`start()`, `stop()` (abstract). **New runtimes = subclass `Runtime`.**

---

## 5. Develop / run / test

```bash
# from repo root
pip install -e .                         # editable, dependency-free base
# (add extras as they gain code: pip install -e '.[unitree]')

python3 -m unittest discover -s tests
ohho doctor                              # environment / runtimes / adapters / robots
ohho sim --robot omnibot --seconds 5     # drive a pattern in simulation
ohho agent omnibot "explore the room"

# Lint exactly like CI (run from repo root — see §7):
pip install ruff
ruff format --check --exclude robot_ws/src/omnibot_firmware .
ruff check        --exclude robot_ws/src/omnibot_firmware .
```

---

## 6. Conventions & invariants (follow these)

- **Dependency-free `base`.** Core (`schema`, `capabilities`, `registry`,
  `transport`, `runtime/native`, `adapters/sim`, `robot`, `cli`) must import only
  the standard library. Anything heavier (torch, rclpy, cyclonedds, pymavlink,
  numpy, fastapi) goes behind an **extra** and is **imported lazily** inside the
  component that needs it (see `hardware.resolve_device` and
  `registry.load_manifest` for the pattern).
- **Style:** `from __future__ import annotations` at the top of every module;
  full type hints; docstrings on public classes/functions. **Code must pass
  `ruff format` and `ruff check`** (CI runs both repo-wide — see §7). After
  editing, run `ruff format ohho tests` before committing.
- **Tests:** stdlib `unittest` (no pytest dependency), under `tests/`,
  deterministic (build `Robot` over a `SimTransport` and call `step(dt)`
  manually rather than relying on the live thread — see `tests/test_robot.py`).
  Add tests with every new module.
- **The invariant rule from §2** — engines depend only on `Runtime`/`Transport`.
- **Keep `RobotSpec` constants synced** with the real platform: OmniBot arm joint
  names mirror the ROS arm driver; OmniBot `max_lin=0.2` mirrors the Yahboom
  driver clamp. If you change one, change both and note it here.
- **Keep website docs in sync** (`website/docs/ohho-os/*`, `website/app/os`).

---

## 7. CI / git notes

- This repository's workflow is `.github/workflows/ci.yml`. It installs with
  `pip install -e .`, runs `python -m unittest discover -s tests`, then
  `ohho doctor` and `ohho sim --robot omnibot --seconds 2`, on Ubuntu, macOS,
  and Windows for Python 3.10, 3.11, 3.12, and 3.13.
- There is no PyPI publish workflow here (that is OHH-20).
- Editable-install artifacts (`ohho_os.egg-info`, `__pycache__`) are gitignored.
- The notes that used to live in this section described OmniBotPro's ROS
  workflow (`ros2_ci.yml`, Vercel, `sdk/` paths). They do not apply to this
  repository.

---

## 8. Roadmap (M0 → M5)

Estimates are split into **software time** (compresses well with an AI agent) and
**hardware-gated time** (needs the physical robots on a bench — does *not*
compress). M0 is done.

### ✅ M0 — Scaffold + sim slice (DONE, merged)
RAL, native runtime, sim adapter, registry, CLI, 40 tests. Acceptance: met.

### ✅ M1 — Two-robot hardware vertical slice (SOFTWARE COMPLETE)
**Goal:** the *same* code drives two very different real robots, proving
robot-agnosticism. Pair: **OmniBot** (wheeled, USB serial) + **Unitree Go2**
(quadruped, DDS).

**Done (v0.3.0):**
- `ohho/adapters/yahboom.py` — `YahboomTransport(BaseTransport)` over pyserial,
  `serial://` scheme. Reuses the vendored codec `_yahboom_proto.py`. Reader thread
  decodes `0xFB` packets → odometry integration + IMU yaw fusion. Clean
  `AdapterUnavailable` when `[serial]` missing.
- `ohho/adapters/feetech.py` — `FeetechTransport(BaseTransport)` for the SO-101
  6-DOF arm, `feetech://` scheme. Lazy `lerobot` import (`[arm]` extra). rad↔tick
  conversion (4096 ticks/rev, home 2048), joint clamping, torque enable/disable,
  reader thread for present positions. Fake bus injectable for tests.
- `ohho/adapters/composite.py` — `CompositeTransport(base, arm)` merges a base
  link + arm link into one `Transport`: `send_velocity`→base,
  `send_joint_command`→arm, `read`→merged telemetry, estop→both. This is what
  lets a single `Robot` drive the full OmniBot mobile-manipulator.
- `ohho/adapters/unitree.py` — `UnitreeDdsTransport(BaseTransport)` over
  `unitree_sdk2py`, `dds://` scheme. **DDS state subscription wired**: polling
  reader thread calls `SportClient.GetState(SportModeState)` → `_on_state` →
  `Telemetry` at 20 Hz. `state_factory` injectable for mock tests. Maps
  `Velocity` → `SportClient.Move(vx, vy, vyaw)`.
- `resolve_transport` auto-composes: `serial://<base>,<arm>` for a robot with
  `manipulation` capability returns a `CompositeTransport(Yahboom, Feetech)`.
- Extras: `[serial]` (pyserial), `[arm]` (lerobot), `[unitree]` (cyclonedds).
- **Mock/loopback tests** (27 new, 85 total): `test_feetech.py`,
  `test_composite.py`, updated `test_unitree.py` + `test_adapters.py`.
- **HIL test harness**: `tests/hil/test_hil.py` — 4 tests (OmniBot base,
  OmniBot arm, OmniBot composite, Go2 DDS), skipped unless `OHHO_HIL=1`.

**Remaining (hardware-gated only):** on-robot bring-up against the real OmniBot
+ Go2 — run `OHHO_HIL=1 python -m unittest discover -s tests/hil -v` on the
bench. DJI / MAVLink adapter not started.

### ✅ M2 — Real agent brain (DONE)
Wired `ohho.agent._default_brain()` to **`agent_engine`** via the new
`ohho/brains.py` module. `HarnessBrain` builds a `ToolRegistry` from the robot's
capabilities (drive, move_joints, stop, emergency_stop, get_telemetry,
get_status), creates a `RobotPerceptor` that turns `Robot.telemetry()` into a
`WorldState`, and runs a `ClaudeToolCallingReasoner` over a `ReasoningRouter`
(cloud Claude when `ANTHROPIC_API_KEY` + `anthropic` present, echo fallback
otherwise). `ScriptedBrain` remains the no-dependency fallback when
`agent_engine` isn't installed. Acceptance met: `Agent(bot).run(goal)` performs
a real perceive→reason→act→reflect loop; 11 new tests in `test_brains.py`.

### ✅ M3 — Training & data pipelines (DONE)
Built `ohho.data`, `ohho.train`, `ohho.serve` — the record→train→serve loop.

- **`ohho.data.Recorder`** — wraps a `Robot`, intercepts `drive()` and
  `move_joints()` to capture actions, polls `telemetry()` for state. Frames are
  9-D (6 arm + 3 base) for mobile-manipulators, 3-D for base-only robots. Manual
  mode (`start_episode`/`capture_frame`/`stop_episode`) and timer mode
  (`record_episode`/`stop_recording`). Writes LeRobot v2.0 layout (Parquet via
  pyarrow `[data]` extra, JSON Lines fallback without it) with `info.json`,
  `tasks.jsonl`, `episodes.jsonl`.
- **`ohho.data.DatasetReader`** — lightweight reader (no torch needed) that
  loads meta + episode data (Parquet or JSON Lines), provides `stats()`,
  `iter_episodes()`, `all_frames()`.
- **`ohho.train.finetune()`** — delegates to `lerobot_engine` when torch + lerobot
  are installed (`[train]` extra). Supports `smolvla`, `act`, `diffusion`,
  `openvla`. `mock=True` writes a dummy checkpoint from dataset stats (no GPU
  needed — for the sim acceptance loop).
- **`ohho.serve`** — `build_app()` returns a FastAPI inference server
  (testable with `TestClient`); `serve()` launches it via uvicorn. Endpoints:
  `GET /health`, `POST /load_model`, `POST /predict`. `mock_model=True` uses a
  no-op model (for the sim loop). Added `ohho serve` CLI command.
- **Acceptance test** (`test_data.py::TestEndToEndRecordTrainServe`): records 10
  frames from a sim OmniBot → mock-trains a smolvla checkpoint → launches a mock
  serve app → `POST /predict` returns a 9-D action vector. 12 new tests.

### ✅ M4 — ROS 2 runtime backend (DONE)
Implemented `Ros2Runtime` (rclpy node behind the existing `Runtime` port) and
`Ros2Transport` (bridges the unified Transport to ROS 2 topics).

- **`Ros2Runtime`** (`runtime/ros2.py`) — creates an rclpy node (lazy import,
  injectable `node_factory` for tests). Overrides `subscribe`/`publish` to bridge
  to ROS 2 topics (JSON-encoded `std_msgs/String` for the generic bus).
  Overrides `get_param`/`set_param` to use the node parameter API.
  `create_timer` uses rclpy timers. `start()` spins the node in a background
  thread; `stop()` cleans up timers, publishers, subscriptions, and shuts down
  rclpy.
- **`Ros2Transport`** (`adapters/ros2.py`) — bridges to real ROS 2 topics:
  publishes `geometry_msgs/Twist` on `/cmd_vel`, subscribes to
  `nav_msgs/Odometry` on `/odom`, subscribes to `sensor_msgs/JointState` on
  `/arm/joint_states`, publishes joint commands on `/arm/joint_commands`.
  Supports namespace prefixes (`ros2:///robot1`). All message types are
  imported lazily; `msg_factory` injectable for tests.
- **`get_runtime("auto")`** — now prefers `ros2` when rclpy is available, falls
  back to `native` otherwise. Same code, zero changes above the port.
- **`ros2://` scheme** registered in `resolve_transport`; `available_adapters()`
  probes for `rclpy`.
- **Tests** (`test_ros2.py`, 21 new): `FakeRos2Node` simulates the full rclpy
  node interface — runtime pub/sub, params, timers, transport publishers/
  subscribers, odometry + joint state callbacks, namespace prefix, estop, auto
  runtime selection. All run without ROS 2 installed.
- **Acceptance met:** `runtime="ros2"` runs the identical Robot/agent/train code
  with zero changes above the port. `Robot.connect("omnibot", "ros2://",
  runtime="ros2")` drives through `/cmd_vel` and reads `/odom`.

### ✅ M5 — Packaging, profiles & community (DONE)
- **`ohho.profiles`** — `HardwareProfile` + `NodeSpec` with 5 built-in profiles
  (`pi_workstation`, `jetson_single`, `workstation_single`, `mac_dev`,
  `edge_cpu`). `detect_profile()` auto-detects via `OHHO_HW_PROFILE` env var or
  platform probes. `describe()` for logs/W&B. Added `ohho profile list/show/detect`
  CLI.
- **`ohho.market`** — skill registry: `@skill` decorator, `register_skill()`,
  `list_skills()`, `run_skill()`. 4 built-in skills (`patrol`, `wave`, `stop`,
  `status`). Capability-gated execution (`SkillRequirementsNotMet` if the robot
  lacks required caps). Added `ohho market list/run` CLI.
- **`CONTRIBUTING.md`** — "add a robot = one manifest + one adapter + one test"
  guide with code examples, the architecture invariants, and a PR checklist.
- **Root `CLAUDE.md`** — updated to document the `sdk/` area.
- **Version bumped to `1.0.0`** — all milestones (M0–M5) complete.
- 22 new tests (`test_market.py` + `test_profiles.py`).
- **PyPI:** not published from this repository. OHH-20 covers trusted
  publishing. Do not add a publish workflow here.

---

## 9. START HERE: post-1.0 development

All milestones (M0–M5) are complete. OhhO OS is at v1.1.3. To continue:

1. `pip install -e .` and run `python -m unittest discover -s tests`.
   On a base install (Python 3.12.3, Linux) that reported
   `Ran 197 tests` / `OK (skipped=18)`.
2. **PyPI publish** is OHH-20. This repository does not publish.
3. **Installer:** `curl | sh` behind `ohho.com/install.sh`.
4. **New robots/skills:** follow `CONTRIBUTING.md` — one manifest + one
   adapter + one test.
5. **On-robot M1 bring-up:** `OHHO_HIL=1 python -m unittest discover -s
   tests/hil -v` on the bench.
6. Keep `AGENTS.md` §0/§3/§8 in sync as you add features.

### M1 on-robot bring-up (when hardware is available)

1. Install extras: `pip install -e '.[serial,arm,unitree]'`.
2. Set env vars: `OHHO_HIL=1`, `OHHO_OMNIBOT_PORT=/dev/ttyUSB0`,
   `OHHO_OMNIBOT_ARM=/dev/ttyACM0`, `OHHO_GO2_IFACE=eth0`.
3. Run `python -m unittest discover -s tests/hil -v`.
4. Or drive manually:
   ```python
   from ohho import Robot
   # OHHO_ARM_HARDWARE=1 and an interactive terminal; arm() asks for the phrase
   with Robot.connect("omnibot", "serial:///dev/ttyUSB0,/dev/ttyACM0") as bot:
       bot.arm(); bot.drive(vx=0.05); bot.move_joints([0, -0.5, 0.5, 0, 0, 0.2])
   with Robot.connect("unitree-go2", "dds://eth0") as bot:
       bot.arm(); bot.drive(vx=0.3, w=0.2); print(bot.telemetry().odom)
   ```

### Bench v0 hardware self-test (OHH-87)

Run the bring-up diagnostic suite for OmniBot or simulated test bench:
1. **Simulation / CI:** `ohho selftest --sim` (runs all 5 checks against simulated hardware, exits 0).
2. **On-robot bench:** `ohho selftest --robot omnibot --allow-spin` (checks base serial, wheel encoders with short pulse, IMU rate >= 20Hz, STS3215 arm bus diagnostics, and camera frames). Generates dated JSON report (`selftest-YYYYMMDD-HHMMSS.json`).
3. **Real-robot run:** Pending on-bench execution by Varun (hardware-gated).

---

## 9a. Safety rules (SafetyGate, OHH-117) — do not weaken

`ohho/safety.py`; user docs in `docs/safety.md`. These are invariants. Changing
any of them needs an explicit Linear issue, not a drive-by edit.

1. **Always gated.** `Robot.connect` wraps every transport whose
   `protocol != "simulated"` in `SafetyGate`. Do not add a bypass flag. Do not
   give the gate a `__getattr__` passthrough (it would expose `send_motor`,
   raw `sport_command`, etc.).
2. **Fail closed.** Any refusal sends zero velocity. Transport exceptions,
   watchdog errors → latched e-stop. Unwritable audit log → disarm. Refused
   joint/effort commands are dropped, never "zeroed".
3. **Caps come from the per-robot `SafetyProfile`.** Go2: 0.5 m/s forward,
   0.3 m/s lateral, 1.0 rad/s yaw, acceleration-limited. Hard ceiling 1.5 m/s /
   2.0 rad/s; only a safety config file (`hard_ceiling`) raises it, never the
   caller or an agent. Clamp, don't reject.
4. **Watchdogs.** No command for 300 ms → zero velocity. No state for 500 ms
   while armed → stop + latch. Profiles may make these stricter, never looser.
5. **E-stop.** Latched. Stop first (`StopMove()` on Go2), then `Damp()` only
   if the robot is already lying down or `estop_damp` / `--estop-damp` is set.
   Unknown posture never damps. `release_stop()` needs a human at a TTY,
   refuses while tilted/faulted, and leaves the gate disarmed (re-arm needed).
6. **Agents and policies never get `arm` or `release_stop` as tools**
   (`brains.py`, `market.py`). They may call `emergency_stop`.
7. **Allowlist.** `stand_up, stand_down, balance_stand, recovery_stand,
   stop_move, move, euler (±0.3 rad), sit, rise_sit`. Always refuse flips,
   jumps, pounce, dance, handstand, walk_upright, `free_*`, cross_step — the
   denylist wins over any config file.
8. **Tilt / fault.** `|roll|` or `|pitch|` > 0.6 rad, or `error_code != 0` →
   stop + latch, armed or not.
9. **Arming.** Never automatic, including tests and CI. Any non-loopback
   interface or DDS domain 0 needs `OHHO_ARM_HARDWARE=1` **and** an interactive
   TTY confirmation ("I am physically present, the area is clear, the remote is
   in my hand"), logged with a timestamp. CI env or non-TTY → refused.
10. **Audit.** Every arm/disarm, e-stop, release, clamp, refusal and watchdog
    trip goes to `~/.ohho/safety/<UTC date>.jsonl`.
11. **Tests.** Fake clock + fake transports only; no sleeps, no hardware. CI
    fails below 95 % line + branch coverage on `ohho/safety.py`.

Not gated: `ohho selftest` and `tests/hil/` construct adapters directly (they
have their own consent: `--allow-spin`, `OHHO_HIL=1`). The MuJoCo 2 m/s →
0.5 m/s and 300 ms client-kill check is OHH-121. The `ohho-quadruped` repo's
AGENTS.md needs the same rules section (not edited from here).

---

## 10. Decisions made (don't re-litigate without reason)

- **Name:** "OhhO OS" is the **engine**, sits beside the products, renames
  nothing. Confirmed by the user.
- **Runtime posture:** **both** ROS and no-ROS, user chooses; the chooser
  explains trade-offs (shipped on `/os` + `docs/ohho-os/runtimes.md`).
- **Launch robots:** OmniBot + Unitree Go2 (max contrast: wheeled/serial/has-arm
  vs legged/DDS/no-arm).
- **Language:** Python-first; the web `Transport` types are the shared contract.
- **License:** Apache-2.0 (OHH-18). `pyproject.toml` sets
  `license = "Apache-2.0"` and `license-files = ["LICENSE"]`.
- **OSS home:** the canonical public package is
  https://github.com/ohho-robotics/ohho-sdk. `pyproject.toml` `Homepage`
  and `Source` point there.

---

## 11. Gotchas

- `get_runtime("auto")` returns **native** in M0 (ros2 backend unimplemented);
  flip this in M4.
- `Robot.connect("<any>")` with no transport **silently runs in simulation**
  today (auto fallback). An **explicit** non-sim URI raises `AdapterUnavailable`
  — that's intentional (honest error vs convenient default).
- An explicit non-sim URI that does connect comes back **gated and disarmed**:
  `bot.drive()` sends zero velocity until `bot.arm()` succeeds (see §9a).
  `bot.transport` is the `SafetyGate`, not the adapter.
- The sim's `start_sim()` runs a background thread; for deterministic tests
  construct `Robot(spec, SimTransport(spec), NativeRuntime())` directly and call
  `transport.step(dt)` (don't start the thread).
- `ohho_os.egg-info` / `__pycache__` are gitignored — if `git add sdk` ever tries
  to stage them, the `.gitignore` is missing.

---

## 12. References

- **Website surfaces:** `website/app/os/page.tsx` (marketing),
  `website/docs/ohho-os/{README,install,quickstart,runtimes,robots,training,architecture}.md`.
- **Transport contract (TS source):** `website/lib/connect/types.ts`;
  **adapter catalog / joint maps:** `website/lib/bridge/adapters.ts`;
  **robot taxonomy:** `website/lib/garage/types.ts`.
- **Reuse targets:** `packages/yahboom_ros2`, `packages/mecanum_drive_ros2`,
  `agent_engine/`, `learning_engine/`, `lerobot_engine/`, `vla_engine/`,
  `packages/vla_serve`, `data_engine/`, `robot_ws/` (ROS backend).
- **Merged PRs:** #163 (brand + docs), #164 (SDK M0 core).
- **Repo guide:** root `CLAUDE.md` (monorepo conventions, physical constants,
  topic map).

---

_Last updated: M5 complete — OhhO OS v1.0.0. All milestones (M0–M5) shipped.
151 tests pass. Hardware profiles, skill market, CONTRIBUTING.md, root CLAUDE.md
updated. Ready for PyPI publish._
