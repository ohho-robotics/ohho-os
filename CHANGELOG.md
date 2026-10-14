# Changelog

All notable changes to the ohho-os protocol pin and the schema notes that go with it are documented here.

The Python package on this branch stays **1.1.3** (`pyproject.toml`, `ohho.__version__`). This file is not a package bump.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Status:** the protocol pin is not yet tagged. Tags `v1.1.1` and `v1.1.2` already exist for the package. `v1.1.2` is the published PyPI release named by the README. `1.1.3` on `origin/main` has not been tagged.

## [protocol 1.0.0] — not released yet

Intended git tag: `protocol-v1.0.0`.

This is not package `1.0.0` and it must not be git tag `v1.0.0` or any other `v*` tag. `.github/workflows/release.yml` publishes to PyPI on `v*` and requires the tag to equal `version` in `pyproject.toml` (`1.1.3` on this commit).

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
- A `v*` tag of this commit would try to publish package `1.1.3`, which is a different release from this pin.
