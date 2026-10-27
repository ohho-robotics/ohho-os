# Changelog

All notable changes to **ohho-os** (PyPI package `ohho-os`, import `ohho`) are
documented here. The TypeScript SDK is versioned independently in
[ohho-sdk](https://github.com/ohho-robotics/ohho-sdk).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - unreleased

Prep for OHH-281. This section is not a published release. PyPI still has
[ohho-os 1.1.2](https://pypi.org/project/ohho-os/1.1.2/). Version 1.1.3 was
set in the tree and was never tagged or published. Do not push tag `v1.2.0`
until the checklist in [`docs/releases/ohho-os-1.2.0.md`](docs/releases/ohho-os-1.2.0.md)
is done. Pushing a `v*` tag runs `.github/workflows/release.yml`.

### Added

- **Protocol version** — `ohho.PROTOCOL_VERSION` (`1.0.0`) decoupled from package
  semver; `ohho version` / `ohho doctor` print it.
- **JSON schemas** — `schemas/` (generated via `python scripts/export_json_schemas.py`);
  CI fails when stale. Release workflow uploads them as a GitHub artifact.
- **sim-serve** — WebSocket telemetry includes `protocolVersion` on every frame.
- **`ohho login` / `ohho whoami` / `ohho logout`** — RFC 8628 device-flow client
  for the OhhO console (OHH-273). `login` requests a code, prints the user code
  and verification URL, and polls until the code is approved, denied, or
  expired. The access token is stored in `~/.ohho/credentials.json` (POSIX mode
  `0600`; Windows: current-user DACL). `OHHO_TOKEN` overrides the file and is
  never written. `whoami` prints the account; `whoami --json` is the agent
  schema in [`docs/cli.md`](docs/cli.md). `logout` deletes the local file only
  (no CLI revoke endpoint). The token is not printed. Sign-in is optional:
  `ohho sim` and `ohho mcp serve` work with no account. Docs:
  [`docs/cli.md`](docs/cli.md).
- **`ohho mcp serve`** — stdio JSON-RPC MCP server, stdlib only (OHH-47, on the
  OHH-173 server). Real simulator tools: `connect`, `drive`, `telemetry`,
  `nav_goto`, `look`, plus `emergency_stop`. Default transport is `sim://`.
  `ohho mcp` with no subcommand still serves. Docs: [`docs/mcp.md`](docs/mcp.md).
- **`ohho mcp connect --client`** — merges that local stdio server into seven
  AI clients: Claude Code, Codex, Cursor, VS Code, Gemini CLI, Windsurf, and
  OpenCode (OHH-275). File configs are merged (only the `ohho` entry), backed
  up to `.bak`, and written atomically. The command is the absolute path of
  the `ohho` script plus `mcp serve`. `--hosted` exits with an error until
  hosted MCP auth (OHH-274). Docs: [`docs/mcp.md`](docs/mcp.md).
- **`ohho rules install --client`** — writes one agent-rules text,
  `ohho/agent_rules/ohho-rules.md`, into those same seven clients, user scope
  or `--project DIR` (OHH-279). A marked block is merged; text outside the
  markers is kept. Also `rules list`, `rules show`, and `rules remove`.
  Docs: [`docs/rules.md`](docs/rules.md).
- **`ohho doctor --agent`** — readiness for the CLI version, Python, login,
  rules, MCP config, and `sim://` telemetry (OHH-280). Offline unless
  `--online`. `--json` prints the report in [`docs/cli.md`](docs/cli.md).
  The sim check reads telemetry and does not drive.
- **`ohho verify --sim`** — drives the in-process simulator only (OHH-280).
  `--sim` is required. A non-sim URI from a flag, environment variable, or
  robot config is refused before a transport is created. On success it drives
  0.1 m/s forward for 1 second through `SafetyGate` and prints the distance.
  `--json` prints the result. `--console` optionally checks the console `/sim`
  page and does not fail a drive that passed.

### Changed

- **Package version `1.2.0`** in `pyproject.toml` and `ohho.__version__`.
  `dependencies` stays empty: the base install is still the standard library.
- **Agent rules ship in the wheel.** `[tool.setuptools.package-data]` includes
  `ohho/agent_rules/*.md` and `ohho/robots/*.json`. `.github/workflows/release.yml`
  is unchanged.

### Security

- **MCP stays on `sim://` unless a human passes `--allow-hardware`** (OHH-278).
  `OHHO_ALLOW_HARDWARE`, `OHHO_ARM_HARDWARE`, and tool arguments do not enable
  hardware. `ohho mcp connect` records `mcp serve` with neither
  `--allow-hardware` nor `--arm`. The server never calls `arm()` or
  `release_stop()`. A non-sim URI goes through `Robot.connect` (`SafetyGate`)
  and stays disarmed, because stdin is the JSON-RPC pipe, not a TTY. Motion
  tools then return `not_armed` and send zero velocity. `emergency_stop`
  latches; no tool clears it. Tool registration rejects names that arm,
  release an e-stop, or change a profile or a limit. `look` and `nav_goto`
  stay on the simulator even when hardware is allowed. Results that use a
  robot include `backend`; velocity results include the requested velocity,
  the applied velocity, whether it was clamped, and the cap.

All notable changes to the ohho-os protocol pin and the schema notes that go with it are documented here.

The protocol-pin draft below was written while the package version stayed **1.1.3** (`pyproject.toml`, `ohho.__version__` on that draft). That draft is not a package bump. This branch's package version is **1.2.0**.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Status:** the protocol pin is not yet tagged. Tags `v1.1.1` and `v1.1.2` already exist for the package. `v1.1.2` is the published PyPI release named by the README. `1.1.3` on `origin/main` has not been tagged.

**Status:** entries below are **draft prep for OHH-71**. Nothing in this file
implies a published release until tags and PyPI artifacts exist.

## [protocol 1.0.0] — not released yet

Intended git tag: `protocol-v1.0.0`.

This is not package `1.0.0` and it must not be git tag `v1.0.0` or any other `v*` tag. `.github/workflows/release.yml` publishes to PyPI on `v*` and requires the tag to equal `version` in `pyproject.toml` (`1.2.0` on this tree; the protocol draft recorded `1.1.3`).

Baseline of this paperwork branch: `origin/main` at `85db848268168fa27b4ebb1b5992220b3bfacfcd` (merge of PR #14). That commit has no `ohho/protocol.py` and no `schemas/manifest.json`.

Commands a human would run later: [docs/releases/protocol-tag-commands.md](docs/releases/protocol-tag-commands.md).

### Added

Prepared on other branches. Not in `85db848`. Not tagged.

- **`protocolVersion` `1.0.0`** — `PROTOCOL_VERSION = "1.0.0"` in `ohho/protocol.py`, independent of `ohho.__version__`. `schemas/manifest.json` records `"protocolVersion": "1.0.0"` and `"ohhoOsVersion": "1.1.3"`. Those JSON files are `connection_state`, `joint_reading`, `odometry`, `scan`, `telemetry`, `transport_status`, and `velocity`. They are the engine wire pin. They are not the `ohho.telemetry.v1` package. Branch `ohh-174-protocol-version`, tip `6f690d50b5991b13a35430581f84626529ab870d`. `ohho/sim_serve.py` on that commit puts `protocolVersion` on WebSocket frames.
- **`RobotDescriptor` loader** — `ohho/descriptor.py` loads and emits descriptor v1 from the OmniBot safety profile. Branch `ohh-214-robot-descriptor`, tip `389c589cd00619cd12b18b97c636d6fac556337e`. That tip is not the same commit as `6f690d50`.

### ohho-sdk schema package (other repository)

GitHub `ohho-robotics/ohho-sdk` (npm `@ohho/schemas`), not this Python tree. Branch `ohh-214-robot-descriptor` at `5a20f1c23c10c814b8a790536a80b21695b3ed82`:

- `ohho.telemetry.v1` manifest version `1.0.0` (commit `d3b54df9ec8310f56c77e34ce31c98887bfbf5b5` is contained there).
- `ohho.descriptor.v1` / `RobotDescriptor`, field `v` pattern `1.0.x`.
- `PINNED_PROTOCOL_VERSION = "1.0.0"`.
- `packages/schemas/package.json` `"version": "1.0.0"` on that commit.

That repository has no git tags in the local checkout used for this draft. This paperwork does not publish npm. Intended git tag, still not created: `schemas-v1.0.0`.

### Known limits

- Tagging `85db848` as `protocol-v1.0.0` would point at a commit that does not define `PROTOCOL_VERSION`.
- A `v*` tag of this commit would try to publish package `1.2.0` (the protocol draft recorded `1.1.3`), which is a different release from this pin.

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
