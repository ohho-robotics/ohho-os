# Changelog

All notable changes to **ohho-os** (import `ohho`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Status:** entries below are **draft prep for OHH-71**. Nothing in this file
implies a published release until tags and PyPI artifacts exist.

## [1.3.0] — not released yet

Planned release tag: `v1.3.0` (must match `version` in `pyproject.toml` and
`ohho/__init__.py::__version__` before tagging). As of this draft, `main` still
reports **1.1.3** in those files; bump them as part of the release checklist in
[`RELEASE_PLAN.md`](RELEASE_PLAN.md).

Baseline for this noteset: `origin/main` after merge of PR #14 (`85db848`).
Last version on PyPI: **1.1.2** ([`v1.1.2`](https://pypi.org/project/ohho-os/1.1.2/)).
The 1.1.3 metadata commit on `main` was never tagged or published.

### Added

- **`ohho sim-serve`** — WebSocket teleop and telemetry server (`ohho/sim_serve.py`,
  CLI `ohho sim-serve`). Backends: in-process `sim://` and `ros2://` via rosbridge;
  Isaac/MuJoCo backends exit with a clear “not built” message. Session safety:
  300 ms command timeout and latched e-stop. Docs: [`docs/sim-serve.md`](docs/sim-serve.md).
  (OHH-91, PRs #7–#8.)
- **`ohho sim-loop`** — Nightly-style CPU pipeline: sim teleop record → LeRobot v2.0
  dataset validation → tiny ACT train → FastAPI serve → closed-loop sim evaluation with
  latency p50/p95. CLI `ohho sim-loop`, workflow
  [`.github/workflows/sim-loop.yml`](.github/workflows/sim-loop.yml). (OHH-86, PR #10.)
- **`ohho selftest`** — Bench v0 hardware bring-up suite (serial link, wheel encoders,
  IMU rate, Feetech STS3215 diagnostics, camera frames) with dated JSON report; `--sim`
  mode for CI. Docs: [`docs/selftest.md`](docs/selftest.md). (OHH-87, PRs #9, #11.)
- **`ohho.safety.SafetyGate`** — Fail-closed `Transport` wrapper on every non-sim
  connection from `Robot.connect`: velocity/accel/joint/effort caps, command and state
  watchdogs, latched e-stop, sport-action allowlist, tilt/fault latch, interactive
  arming, JSONL audit under `~/.ohho/safety/`. Docs: [`docs/safety.md`](docs/safety.md);
  engineering rules in [`AGENTS.md`](AGENTS.md) §9a. CI enforces ≥95% line+branch
  coverage on `ohho/safety.py`. (OHH-117, PR #12.)
- **Demo drive + ROS 2 Gazebo CI** — `ohho/demo_drive.py`, example
  [`examples/demo_drive.py`](examples/demo_drive.py), fake ROS 2 robot node for CI,
  workflow [`.github/workflows/demo-drive-ros2.yml`](.github/workflows/demo-drive-ros2.yml).
  (OHH-32, PR #13.)
- **ACT training/serve hardening** — Lazy exports in `ohho.train`, safer checkpoint
  auto-detection, `weights_only=True` loading, device forwarding during serve auto-load.
  (OHH-86 follow-ups on `main`.)

### Fixed

- **Demo drive on ROS 2** — Odometry readiness when telemetry flows through
  `SafetyGate`; loopback `ros2://` arming rule for CI. (PR #14.)
- **Selftest** — Cross-platform COM port handling (`\\.\COMn`), honesty fixes for
  skipped checks and pyserial-missing CI. (OHH-87 honesty, PR #11.)

### Changed

- **PyPI package metadata (1.1.3, unpublished)** — Accurate long description and
  classifiers in `pyproject.toml` (merged PR #6; not released to PyPI).

### Documentation

- Safety, selftest, and sim-serve user docs under [`docs/`](docs/).
- [`README.md`](README.md) — safety gate section, selftest/sim-serve pointers.

---

## [1.1.2] — 2025 (PyPI)

- Published to PyPI as [`ohho-os` 1.1.2](https://pypi.org/project/ohho-os/1.1.2/).
- Tag: `v1.1.2`.
- Removed invalid PyPI trove classifier that blocked `v1.1.1` publish.

## [1.1.1] — not published

- Tag `v1.1.1` exists in git history; PyPI rejected the robotics classifier.

## [1.1.0] and earlier

- Native navigation (`ohho.nav`), spatio-temporal memory and perception
  (`ohho.memory`, `ohho.perception`), M0–M5 robot-agnostic stack (RAL, sim +
  hardware adapters, agent brain, record→train→serve, ROS 2 runtime port, profiles,
  skill market). See [`AGENTS.md`](AGENTS.md) for milestone detail.
