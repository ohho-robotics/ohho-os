# DRAFT — pending Varun review — CV, robotics software

**Status:** DRAFT — pending Varun review. Do not send, export to PDF, or publish until the contact block and any employment or education lines are filled in by hand.

**Varun Vaidhiya**

Robotics software. One Python API for a wheeled mobile manipulator, a quadruped, and an in-process simulator, with ROS 2 optional.

| | |
|---|---|
| GitHub org | https://github.com/ohho-robotics |
| Site | https://ohho-robotics.com |
| YouTube | https://www.youtube.com/@varun.vaidhiya/videos |
| X | https://x.com/varunvaidhiya |
| Email / phone / location | _not in the repositories — fill in_ |
| Employment / education | _not in the repositories — fill in_ |

Public handles above are the ones already printed on the OmniBot README. The git author name on this checkout is Varun Vaidhiya.

## Summary

Author of OhhO OS (`ohho-os`, import name `ohho`): an Apache-2.0, robot-agnostic engine. Callers depend on a `Robot` abstraction, a `Runtime` port, and a `Transport` interface. Swapping the robot or the middleware is a connect argument, not a rewrite. The base install imports only the Python standard library.

## OhhO OS

- Repository (this checkout's `origin`): https://github.com/ohho-robotics/ohho-os
- `README.md` and `pyproject.toml` still list Source as https://github.com/ohho-robotics/ohho-sdk
- Tree version **1.1.3** (`ohho/__init__.py`, `pyproject.toml`).
- **PyPI:** `ohho-os` **1.1.2** is the release already published, from tag `v1.1.2`. Tag `v1.1.1` was not published: PyPI returned 400 because `Topic :: Scientific/Engineering :: Robotics` is not a trove classifier. Tag `v1.1.3` is the documented release command; this draft does not claim that 1.1.3 is on PyPI.
- Licence: Apache-2.0.

**Abstraction**

- Capability-gated commands. A robot without `manipulation` turns `move_joints` into a no-op instead of an exception.
- Built-in specs in `ohho/registry.py`: **OmniBot** (mobile manipulator, holonomic base, 6-DOF arm, `max_lin` 0.2 m/s, `max_ang` 1.0 rad/s), **Unitree Go2** (legged, `max_lin` 1.5 m/s, `max_ang` 2.0 rad/s), **sim** (generic holonomic).
- Transports: `sim://`, `serial://` (Yahboom base; `serial://<base>,<arm>` composes the Feetech arm), `feetech://`, `dds://` (Unitree), `ros2://` (`/cmd_vel`, `/odom`, arm joint topics).
- Runtimes: threaded `NativeRuntime`, and `Ros2Runtime` (rclpy timers, pub/sub, params). `get_runtime("auto")` prefers ROS 2 when `rclpy` is importable. Tests use `FakeRos2Node`; they do not need a ROS 2 install.
- Non-simulated connects are wrapped in `SafetyGate` and start disarmed. `sim://` is not gated.

**CLI** (`ohho`): `doctor`, `list`, `version`, `connect`, `sim`, `drive`, `agent`, `serve`, `sim-serve`, `sim-loop`, `market`, `profile`, `nav`, `look`, `memory`, `selftest`.

**Same script, two transports.** `examples/demo_drive.py` changes only `--transport`. `docs/evidence/ohh-32-demo-drive.md` records a headless run on a developer machine (ROS 2 Jazzy, no Gazebo): both `ros2://` against `examples/fake_ros2_robot_node.py` and `sim://` were invoked as `--seconds 2 --vx 0.2 --min-dx 0.15`. The GitHub Actions job `demo-drive-fake-robot` (`.github/workflows/demo-drive-ros2.yml`) runs that path inside `ros:jazzy`. That evidence file states a Gazebo GIF was not produced.

## Tests and CI (as written down)

| Source | What it records |
|---|---|
| `README.md` | Python 3.12.3, Linux, base install, no extras: `Ran 197 tests in 23.041s`, `OK (skipped=18)`. Skips: 4 hardware-in-the-loop (`OHHO_HIL` unset), 13 that need `agent_engine` and numpy, 1 that needs fastapi. |
| `AGENTS.md` | A later local note: Windows, Python 3.12, `Ran 383 tests, OK (skipped=7)`. This draft does not collapse the two counts into one figure. |
| `.github/workflows/ci.yml` | On push to `main` and on pull requests: Ubuntu, macOS, and Windows, Python 3.10, 3.11, 3.12, and 3.13. Steps: trove-classifier check, `pip install -e .`, `python -m unittest discover -s tests`, `ohho/safety.py` line and branch coverage at least 95%, `ohho doctor`, `ohho sim --robot omnibot --seconds 2`, `ohho selftest --sim`. README states that workflow is green on `main`. |

## Also in the engine (software, not field results)

- Native navigation (`ohho/nav.py`): occupancy grid from `Telemetry.scan`, costmap inflation, A*, frontier exploration, closed-loop `Navigator`.
- Memory and perception (`ohho/memory.py`, `ohho/perception.py`): entity store, spatial and temporal queries, JSON at `~/.ohho/memory/<robot>.json`, `SimPerceptor`, optional Claude-vision `VlmPerceptor`.
- Skill registry (`ohho/market.py`) and hardware profiles (`ohho/profiles.py`).
- WebSocket session `ohho sim-serve` for `sim` and `ros2` (rosbridge). `isaac` and `mujoco` backends exit non-zero and print `not built`.

## Do not claim

- On-robot HIL (`tests/hil/`, four tests, skipped unless `OHHO_HIL=1`) has no recorded bench pass in these docs.
- Real-robot `ohho selftest` is marked pending in `docs/selftest.md`.
- `ts/` does not typecheck. README: `tsc --noEmit` with TypeScript 5.6.3 exits with 246 `error TS` diagnostics.
- The `[dji]` extra lists `pymavlink`. The DJI / MAVLink adapter is not built (`AGENTS.md`).
- Nightly CPU sim-loop workflow is designed and checked in; README says its first run is pending.
