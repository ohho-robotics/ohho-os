# Changelog

All notable changes to **ohho-os** (PyPI package `ohho-os`, import `ohho`) are
documented here. The TypeScript SDK is versioned independently in
[ohho-sdk](https://github.com/ohho-robotics/ohho-sdk).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.3] - unreleased

### Added

- **Protocol version** — `ohho.PROTOCOL_VERSION` (`1.0.0`) decoupled from package
  semver; `ohho version` / `ohho doctor` print it.
- **JSON schemas** — `schemas/` (generated via `python scripts/export_json_schemas.py`);
  CI fails when stale. Release workflow uploads them as a GitHub artifact.
- **sim-serve** — WebSocket telemetry includes `protocolVersion` on every frame.
