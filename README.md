# OhhO OS

**Engine:** [ohho-os](https://github.com/ohho-robotics/ohho-os) · **Client SDK:** [ohho-sdk](https://github.com/ohho-robotics/ohho-sdk)

Python package `ohho-os` (import `ohho`), version 1.2.0.
Homepage: [ohho-robotics.com](https://ohho-robotics.com).
Source: [ohho-robotics/ohho-os](https://github.com/ohho-robotics/ohho-os).

[![CI](https://github.com/ohho-robotics/ohho-os/actions/workflows/ci.yml/badge.svg)](https://github.com/ohho-robotics/ohho-os/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Install

```bash
pip install ohho-os
ohho doctor
ohho sim --robot omnibot --seconds 2
ohho sim --robot unitree-go2 --seconds 2
```

`ohho doctor --agent` reports AI-agent setup; `ohho verify --sim` drives the simulator only. `ohho rules install` writes the agent safety rules for the same clients ([`docs/rules.md`](docs/rules.md)). Full CLI reference: [`docs/cli.md`](docs/cli.md).

`pip install ohho-os` installs the dependency-free base (stdlib only) and the
`ohho` console script. `ohho doctor` prints the interpreter, the runtimes and
adapters it can import, and the built-in robot ids. `ohho sim` connects with
transport `sim://` and runtime `native`, then drives an in-process pattern for
the given number of seconds.

From a git checkout, `pip install -e .` is the same base install, editable.

## Protocol version and SDK compatibility

**Package semver** (`ohho-os` on PyPI) and **protocol version** (`ohho.PROTOCOL_VERSION`,
JSON schemas in `schemas/`) are independent. Bump the protocol only when committed
schemas change in a contract-visible way.

| ohho-os | Protocol | ohho-sdk (`@ohho/schemas`) |
| --- | --- | --- |
| 1.1.3+ | 1.0.0 | 1.0.0+ |

TypeScript types for transport shapes are generated in
[ohho-sdk](https://github.com/ohho-robotics/ohho-sdk) from a pinned copy of
`schemas/`. Regenerate with `python scripts/export_json_schemas.py` here, then
sync the pin in ohho-sdk.

Sign-in is optional. `ohho sim` and `ohho mcp serve` keep working with no
account. `ohho login` uses the console device flow and stores a token in
`~/.ohho/credentials.json` (mode `0600`, or a user-only ACL on Windows).
`ohho whoami` prints the account; `ohho whoami --json` is the agent schema
in [`docs/cli.md`](docs/cli.md). `OHHO_TOKEN` overrides the file and is never
written to disk. The token is not printed.

CI (`.github/workflows/ci.yml`) runs `python -m unittest discover -s tests`,
then `ohho doctor`, `ohho sim --robot omnibot --seconds 2`, and
`ohho sim --robot unitree-go2 --seconds 2`, on Ubuntu,
macOS, and Windows for Python 3.10, 3.11, 3.12, and 3.13. That workflow is
green on `main`. Tag `v1.1.2` published `ohho-os` 1.1.2 to PyPI.

## Protocol pin (not yet tagged)

`protocolVersion` `1.0.0` is not in this commit (`85db848268168fa27b4ebb1b5992220b3bfacfcd`). The package version on that baseline is `1.1.3`. This tree's package version is `1.2.0`. Do not push a `v*` tag for that pin: [`.github/workflows/release.yml`](.github/workflows/release.yml) publishes PyPI on `v*` when the tag matches `pyproject.toml`. The paperwork name is `protocol-v1.0.0`. `ohho.telemetry.v1` and `RobotDescriptor` live in the ohho-sdk npm repository, also not yet tagged. Draft: [CHANGELOG.md](CHANGELOG.md). Human commands: [docs/releases/protocol-tag-commands.md](docs/releases/protocol-tag-commands.md).

## Tests

```bash
python -m unittest discover -s tests
```

On Python 3.12.3, Linux, base install (no extras):

```
Ran 197 tests in 23.041s
OK (skipped=18)
```

The 18 skips were: 4 hardware-in-the-loop tests (`OHHO_HIL` unset),
13 tests that need `agent_engine` and numpy, and 1 test that needs
fastapi (`[serve]`). No failures. A missing `rclpy` is logged from a
background ROS 2 runtime thread; the suite still exits 0.

## Needs hardware

`ohho doctor`, `ohho sim --robot omnibot`, and the default test run do
not open a serial port, a camera, or a GPU. The simulator is in-process.

`tests/hil/` talks to real robots. Every test in that package is
skipped unless `OHHO_HIL=1`. The ports they use:

| Test | Device | Environment variable | Default |
|---|---|---|---|
| OmniBot base | Yahboom serial | `OHHO_OMNIBOT_PORT` | `/dev/ttyUSB0` |
| OmniBot arm | Feetech bus | `OHHO_OMNIBOT_ARM` | `/dev/ttyACM0` |
| OmniBot base + arm | both of the above | both | both defaults |
| Unitree Go2 | DDS (**untested on hardware** — use `ohho sim --robot unitree-go2` first; see [Install](#install)) | `OHHO_GO2_IFACE` | `eth0` |

Those adapters are not installed by `pip install ohho-os`. The extras are
`serial` (pyserial), `arm` (lerobot), and `unitree` (cyclonedds — you also
need [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
from GitHub; it is not published on PyPI as `unitree_sdk2py`). The
`ros2` extra does not pip-install `rclpy`; that module comes from a ROS 2
distro. `train` needs torch. None of those were installed for the test
run above.

## Safety gate on real robots

`Robot.connect` wraps every non-simulated transport (`serial://`,
`feetech://`, `dds://`, `ros2://`) in `ohho.safety.SafetyGate`. The robot
starts **disarmed**: motion commands are refused (and a zero velocity is
sent) until you arm it.

```bash
OHHO_ARM_HARDWARE=1 ohho drive unitree-go2 --transport dds://eth0 --arm --vx 0.3
```

`--arm` asks you to type
`I am physically present, the area is clear, the remote is in my hand`
at an interactive terminal. Without the environment variable, without a TTY,
or under CI, arming is refused. Once armed, the gate clamps velocity to the
robot's profile (Go2: 0.5 m/s forward, 0.3 m/s lateral, 1.0 rad/s yaw, with
acceleration limits), zeroes velocity after 300 ms without a command, and
latches the e-stop after 500 ms without robot state, when roll or pitch passes
0.6 rad, or when the robot reports an error code. A latched e-stop is cleared
only by `release_stop()` from a human at a terminal, and the robot must then
be armed again. Sport actions go through `bot.transport.command(...)`, which
refuses anything outside the allowlist (flips, jumps, dances, handstands and
`free_*` modes are always refused). Every arm, disarm, e-stop, clamp and
refusal is appended to `~/.ohho/safety/<date>.jsonl`.

`sim://` is not gated. See [`docs/safety.md`](docs/safety.md) for the rules,
profiles, config file and audit format. For Unitree Go2 DDS (`dds://`), honest
sim-vs-hardware status and HIL expectations are in [`docs/unitree.md`](docs/unitree.md).

## Pipeline in sim

The complete Data -> Train -> Serve sim loop can be reproduced locally with a single command:

```bash
python scripts/sim_loop.py
# or
ohho sim-loop
```

This single command executes the full five-stage CPU sim loop end-to-end:
1. **Record**: Connects to the simulated robot (`omnibot` over `sim://`), captures >= 5 scripted teleoperation episodes (100 total frames), and writes a standardized LeRobot v2.0 dataset (`meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/stats.json`, `data/chunk-000/episode_*.parquet`).
2. **Validate**: Validates the dataset against the LeRobot v2.0 schema and loads it via `LeRobotDataset`.
3. **Train**: Trains a tiny Action Chunking Transformer (ACT, CVAE + Transformer) policy on CPU for 200 steps and saves the checkpoint (`policy.pt`, `config.json`, `metrics.json`).
4. **Serve & Evaluate**: Boots the FastAPI inference endpoint, loads the trained ACT checkpoint, and executes >= 50 policy-driven closed-loop steps on the simulated robot without errors.
5. **Summary**: Computes serve request latency percentiles (p50 and p95) and formats the run metrics and train loss curve. In GitHub Actions, the summary is published to `$GITHUB_STEP_SUMMARY`.

To run in dry-run / mock mode on machines without PyTorch installed:
```bash
python scripts/sim_loop.py --mock
# or
ohho sim-loop --mock
```

To run with custom parameters:
```bash
python scripts/sim_loop.py --episodes 5 --steps-per-episode 20 --train-steps 200 --eval-steps 50 --output-dir ./sim_loop_output
```

The nightly CI workflow (`.github/workflows/sim-loop.yml`) is designed to run this pipeline on CPU-only runners, verify LeRobotDataset loading, and upload the trained checkpoint and dataset artifacts (first run pending).

## Simulation WebSocket Server (`sim-serve`)

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`), a live robot via the unified `Robot` API (`--backend robot --transport <uri>`), or ROS 2 through rosbridge.

### Message Schema

#### Client sends (JSON):
- `velocity`: Target linear and angular velocity, e.g. `{"linear": {"x": 0.2, "y": 0.0}, "angular": {"z": 0.1}}`.
- `joints`: Mapping of joint name to radians, e.g. `{"arm_shoulder_pan": 0.5}`.
- `deadman` (`bool`): Deadman switch. If `false` or if no client message arrives for 300 ms, the commanded velocity is zero.
- `estop` (`bool`): Emergency stop. If `true`, estop latches until a message with `estop: false` arrives.
- `mode` (optional): `"stand"`, `"lie_down"`, or `"recover"` — forwarded as allowlisted high-level actions on legged robots (via `SafetyGate` when armed).

#### Server sends (JSON):
- `odom`: Current odometry pose `{"x": float, "y": float, "yaw": float}`.
- `joints`: Current joint positions `{"<name>": float, ...}`.
- `protocolVersion`: OhhO wire protocol semver (`ohho.PROTOCOL_VERSION`). Present on every telemetry frame, including frames that add `posture` or `battery`.
- `backend`: Backend identifier (`"sim"`, `"ros2"`, or `"robot"`).
- `posture` / `battery` (optional): when the connected robot reports them.
- `camera`: Base64 JPEG string (optional) or `no_camera: true`.

### Commands

1. **In-process simulation (no ROS required):**
   ```bash
   ohho sim-serve --backend sim --port 8790
   ```
   Serves the existing in-process `sim://` robot on port 8790 with zero external dependencies. The default is 8790 because port 8765 is OmniBot's Foxglove bridge.

2. **Live robot (Go2, OmniBot, any registered spec):**
   ```bash
   ohho sim-serve --backend robot --robot unitree-go2 --transport 'dds://lo?domain=1'
   ```
   Uses `Robot.connect` with the given transport URI (same stack as `ohho connect` / agents). Non-sim transports are wrapped in `SafetyGate` and start **disarmed** — arm at the terminal before motion. Use `dds://lo?domain=1` for local DDS simulators; domain `0` is the default on-robot DDS domain.

3. **ROS 2 bridge:**
   ```bash
   ohho sim-serve --backend ros2 --rosbridge ws://localhost:9090
   ```
   Forwards velocity commands to `/cmd_vel` and reads `/odom` via rosbridge. Depends on the `[ros2]` extra.
   > **Note:** Gazebo itself is launched from the OmniBot repo, not from this package.

4. **Unsupported backends:**
   ```bash
   ohho sim-serve --backend isaac
   ohho sim-serve --backend mujoco
   ```
   Both exit non-zero and print `not built`.

## Session registry (`sim up`, `robot announce`)

`ohho sim up --robot omnibot --target local` heartbeats a row in Supabase every 10 seconds so the VR garage can find this PC's sim. `ohho sim down` removes it. `ohho robot announce` does the same for a real robot through the OHH-196 gateway. `--target cloud` exits with an error that names OHH-216 and does not create a VM.

The gateway token is stored only as a SHA-256 hash. Stale rows (no heartbeat for 30 seconds) are hidden. Apply `supabase/migrations/20261014120000_robot_sessions.sql` in the Ohho Supabase project before the first `sim up`. The OhhO-VR `SessionClient` contract is in [`docs/sessions.md`](docs/sessions.md).

## Simulation Demo (OHH-134)

**SIMULATION ONLY -- NOT TESTED ON PHYSICAL ROBOT HARDWARE.**
All quadruped behaviors execute in MuJoCo simulation via local DDS loopback
(`dds://lo?domain=1`) or in-process simulation (`sim://`). Real hardware
bring-up is tracked as **Vision** until a physical Unitree Go2 EDU is available
and an operator is physically present (**OHH-135**).

`ohho sim-demo` executes the complete 6-shot demo sequence (typed goal -> agent
-> Go2 in MuJoCo) and generates an evaluation report conforming to
[`evals/schemas/sim_demo_schema.json`](evals/schemas/sim_demo_schema.json):

```bash
# Print OBS recording cues and the 60-90s shot list checklist:
ohho sim-demo --check-overlays

# Run the live demo against the MuJoCo sport shim (domain 1, interface lo):
ohho sim-demo --transport dds://lo?domain=1 --out evals/logs/demo-run.json

# Offline / automated CI execution (simulated loopback, fast execution):
python scripts/sim_demo_harness.py --sim --fast
```

### The 6-Shot Demo Sequence (~85s target)
1. **Shot 1 -- Doctor & Readiness (`0:00 - 0:10`)**: `ohho doctor` probes runtimes, adapters, and `unitree-go2` capabilities.
2. **Shot 2 -- Sim & Sport Shim (`0:10 - 0:22`)**: Connects over DDS loopback (`dds://lo?domain=1`) to local MuJoCo sport shim; SafetyGate arms on loopback.
3. **Shot 3 -- Typed Goal (`0:22 - 0:34`)**: Prompts: `"Stand up, drive forward 0.3 m, then stop and lie down."`
4. **Shot 4 -- Tool-Call Stream (`0:34 - 0:54`)**: Streams tool calls (`stand_up`, `drive`, `stop`, `lie_down`) with live odometry feedback in MuJoCo.
5. **Shot 5 -- SafetyGate Invariants (`0:54 - 1:12`)**: Demonstrates velocity ceiling clamp (2.0 m/s -> 0.5 m/s), dangerous command refusal (`backflip` denied), and latched E-Stop.
6. **Shot 6 -- Honest End-Card (`1:12 - 1:25`)**: Full-screen graphic end card declaring simulation-only status, OHH-135 hardware gate, and repository links.

### How Varun Records the 60--90s Video Clip
- **Screen Layout**: 1920x1080 @ 60 FPS in OBS Studio. Left 50% Terminal (font size ~18pt), Right 50% MuJoCo Simulation Window.
- **Top-right Watermark**: `SIMULATION ONLY -- NOT TESTED ON HARDWARE`.
- **Terminal 1 (`ohho-quadruped`)**: `python sim/run_go2.py --scene flat` and `python -m ohho_go2.sport_shim --domain 1 --iface lo`.
- **Terminal 2 (`ohho-os`)**: `ohho sim-demo --transport dds://lo?domain=1 --out evals/logs/demo-run.json`.
- Full recording checklist and link instructions: see [`docs/sim-demo.md`](docs/sim-demo.md).

## Hardware Bring-Up Self-Test (`selftest`)

`ohho selftest` evaluates hardware health across 5 core subsystems (serial link, 4-wheel spin with encoder feedback, IMU publishing rate, STS3215 arm servos, and camera frame streaming) and outputs a dated JSON report (`selftest-YYYYMMDD-HHMMSS.json`).

```bash
# In simulation (runs in CI without hardware)
ohho selftest --sim

# On hardware (wheel spin safety lock requires explicit confirmation or --allow-spin)
ohho selftest --robot omnibot --allow-spin
```

See [`docs/selftest.md`](docs/selftest.md) for full usage, safety invariants, and the JSON schema.

## TypeScript

The TypeScript SDK (`@ohho/schemas`, `@ohho/connect`, `@ohho/client`) lives in
[ohho-sdk](https://github.com/ohho-robotics/ohho-sdk). See [`ts/README.md`](ts/README.md).

## Licence

Apache-2.0. The full text is [`LICENSE`](LICENSE).

## Releasing

See [docs/RELEASING.md](docs/RELEASING.md) for PyPI trusted-publisher settings
(owner `ohho-robotics`, repo `ohho-os`, workflow `release.yml`, environment
`pypi`) and the TestPyPI `workflow_dispatch` dry run.
The 1.2.0 checklist (merge order, tag, post-release checks) is
[`docs/releases/ohho-os-1.2.0.md`](docs/releases/ohho-os-1.2.0.md).
Do not push a `v*` tag until that checklist says to. Pushing a `v*` tag
runs `.github/workflows/release.yml`, which publishes to PyPI.

```bash
git tag v1.2.0
git push origin v1.2.0
```

The tag version must match `version` in `pyproject.toml`.
`.github/workflows/release.yml` runs on tags matching `v*`. It fails the
release when those versions differ, checks every classifier against
`trove-classifiers`, builds the sdist and wheel with `python -m build`,
runs `twine check`, and publishes to PyPI with trusted publishing
(`pypa/gh-action-pypi-publish`, `environment: pypi`, no API token). A
following job, in a fresh virtualenv on Ubuntu and macOS, retries
`pip install ohho-os==<tag version>` until that version is installable,
then runs `ohho doctor`.

`ohho-os` 1.1.2 is the release already on PyPI (tag `v1.1.2`). Tag
`v1.1.1` points at `286d408` and was not published: PyPI returned 400
because `Topic :: Scientific/Engineering :: Robotics` is not a trove
classifier.
