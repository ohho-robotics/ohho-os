# Changelog — OmniBot (draft for v1.0.0)

**Canonical target path after you copy this into the OmniBot repo:**
[`CHANGELOG.md`](https://github.com/ohho-robotics/OmniBot/blob/main/CHANGELOG.md)
(local checkout: `C:\Users\varun\ohho\OmniBot\CHANGELOG.md`).

This file lives in **ohho-sdk** only as OHH-71 prep. It is **not released** until
you tag `v1.0.0` on [ohho-robotics/OmniBot](https://github.com/ohho-robotics/OmniBot)
and publish a GitHub Release (optional notes).

Baseline: `origin/main` at `d296c9b` (after PR #29). No `v1.0.0` tag exists yet.

## [1.0.0] — not released yet

First tagged open-source release of the OhhO reference mobile-manipulation stack:
ROS 2 Jazzy packages, simulation, teleop, learning/agent engines, Android and Quest
projects, and CI that matches the README “needs hardware / needs GPU” table.

### Added

- **Repository layout** — Split monorepo into OmniBot with `src/` colcon workspace,
  [`mecanum-kinematics/`](mecanum-kinematics/), [`yahboom-python-driver/`](yahboom-python-driver/),
  [`omnibot-ros2/`](omnibot-ros2/), [`omnibot-ai-ros2/`](omnibot-ai-ros2/),
  [`omnibot-ai-engines/`](omnibot-ai-engines/), [`omnibot-digital-twin/`](omnibot-digital-twin/),
  [`omnibot-android/`](omnibot-android/), [`omnibot-vr/`](omnibot-vr/).
- **CI** — Multi-job [`.github/workflows/ci.yml`](.github/workflows/ci.yml): CPU
  pytest/unittest suites, README docs-as-tests (`scripts/docs_test_readme.py`), ROS 2
  Jazzy colcon build+test, Gazebo Harmonic `sim-smoke`, nav-sim scenarios. Gitleaks
  scan and Linguist vendored markers for Unity assets.
- **Gazebo Harmonic simulation** — Headless `sim.launch.py`, flat-world camera test
  hooks, rosbridge on sim for browser teleop (OHH-27, OHH-92).
- **Nav2 + SLAM in simulation** — `slam_toolbox`, map, named waypoints, three-scenario
  nav-sim CI (OHH-30, PR #18).
- **Agent → Nav2 goals** — `/agent/goal` through scripted brain to mission planner with
  safety verify and loop trace; mux preempt expects base stop on teleop→nav2 (OHH-31,
  PRs #28–#29).
- **Safety and teleop** — Arm e-stop path (OHH-93); stop base on every control-mode
  switch (OHH-96); teleop default mode (OHH-96 follow-up).
- **Learning data quality** — Camera schema (OHH-95); episode filter / frozen-frame /
  skew and joint-jump fixes (OHH-99); policy guards — zero base twist on skipped ticks,
  arm limit clamping (OHH-100).
- **Documentation** — README refresh (pitch, quickstart, BOM, architecture, roadmap);
  deeper reference moved under `docs/`. Apache-2.0 licence across first-party packages
  (OHH-18).

### Integration notes (honest scope)

- **CPU-only path** documented with PyPI [`ohho-os`](https://pypi.org/project/ohho-os/)
  (`ohho doctor`, `ohho sim`); does not require this repo’s ROS workspace.
- **On-robot bring-up** (serial arm, live Nav2 on hardware, LeRobot recording with
  physical cameras) is supported by launch files but **not** fully verified in CI;
  treat bench validation as post-tag follow-up.
- **GPU stacks** (OpenVLA/SmolVLA, Isaac Lab RL) remain optional and documented as
  GPU-required in the README table.

### Known limitations (call out in release notes)

- Unity Quest project pins editor **6000.5.2f1**; Android build needs JDK 17 + SDK.
- Some launch entry points intentionally stub with a clear exit when dependencies are
  not built (`test_unbuilt_launch.py`).

---

## Pre-1.0 history

Commits from initial split (`187dec2`) through Apache-2.0 and colcon CI landings are
included in **1.0.0** above; no earlier semver tags were published.
