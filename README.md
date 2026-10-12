# OhhO OS

Python package `ohho-os` (import `ohho`), version 1.1.3.
Homepage: [ohho-robotics.com](https://ohho-robotics.com).
Source: [ohho-robotics/ohho-sdk](https://github.com/ohho-robotics/ohho-sdk).

[![CI](https://github.com/ohho-robotics/ohho-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/ohho-robotics/ohho-sdk/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Install

```bash
pip install ohho-os
ohho doctor
ohho sim --robot omnibot --seconds 2
```

`pip install ohho-os` installs the dependency-free base (stdlib only) and the
`ohho` console script. `ohho doctor` prints the interpreter, the runtimes and
adapters it can import, and the built-in robot ids. `ohho sim` connects with
transport `sim://` and runtime `native`, then drives an in-process pattern for
the given number of seconds.

From a git checkout, `pip install -e .` is the same base install, editable.

CI (`.github/workflows/ci.yml`) runs `python -m unittest discover -s tests`,
then `ohho doctor` and `ohho sim --robot omnibot --seconds 2`, on Ubuntu,
macOS, and Windows for Python 3.10, 3.11, 3.12, and 3.13. That workflow is
green on `main`. Tag `v1.1.2` published `ohho-os` 1.1.2 to PyPI.

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
| Unitree Go2 | DDS interface | `OHHO_GO2_IFACE` | `eth0` |

Those adapters are not installed by `pip install ohho-os`. The extras are
`serial` (pyserial), `arm` (lerobot), and `unitree` (cyclonedds). The
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
profiles, config file and audit format.

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

`ohho sim-serve` exposes a JSON WebSocket session for teleoperation and telemetry streaming. It supports the in-process simulator (`sim://`) or connects to ROS 2 through rosbridge.

### Message Schema

#### Client sends (JSON):
- `velocity`: Target linear and angular velocity, e.g. `{"linear": {"x": 0.2, "y": 0.0}, "angular": {"z": 0.1}}`.
- `joints`: Mapping of joint name to radians, e.g. `{"arm_shoulder_pan": 0.5}`.
- `deadman` (`bool`): Deadman switch. If `false` or if no client message arrives for 300 ms, the commanded velocity is zero.
- `estop` (`bool`): Emergency stop. If `true`, estop latches until a message with `estop: false` arrives.

#### Server sends (JSON):
- `odom`: Current odometry pose `{"x": float, "y": float, "yaw": float}`.
- `joints`: Current joint positions `{"<name>": float, ...}`.
- `backend`: Backend identifier (`"sim"` or `"ros2"`).
- `camera`: Base64 JPEG string (optional) or `no_camera: true`.

### Commands

1. **In-process simulation (no ROS required):**
   ```bash
   ohho sim-serve --backend sim --port 8765
   ```
   Serves the existing in-process `sim://` robot on port 8765 with zero external dependencies.

2. **ROS 2 bridge:**
   ```bash
   ohho sim-serve --backend ros2 --rosbridge ws://localhost:9090
   ```
   Forwards velocity commands to `/cmd_vel` and reads `/odom` via rosbridge. Depends on the `[ros2]` extra.
   > **Note:** Gazebo itself is launched from the OmniBot repo, not from this package.

3. **Unsupported backends:**
   ```bash
   ohho sim-serve --backend isaac
   ohho sim-serve --backend mujoco
   ```
   Both exit non-zero and print `not built`.

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

`ts/` is the previous TypeScript workspace (`package.json`,
`tsconfig.json`, `packages/`), moved with the same file contents. It does
not typecheck. `tsc --noEmit -p ts/tsconfig.json` (TypeScript 5.6.3)
exits with 246 `error TS` diagnostics, including missing modules
(`react`, `vitest`, `@/lib/...`) and files that are not in this tree
(`packages/schemas/src/types.ts`, `packages/schemas/src/robot-catalog.ts`).
See `ts/AGENTS.md`. Paths in that file still describe the old repository
root.

## Licence

Apache-2.0. The full text is [`LICENSE`](LICENSE).

## Releasing

```bash
git tag v1.1.3
git push origin v1.1.3
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
