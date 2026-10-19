# Changelog

All notable changes to **ohho-os** (import `ohho`) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - unreleased

Prep for OHH-281. This section is not a published release. PyPI still has
[ohho-os 1.1.2](https://pypi.org/project/ohho-os/1.1.2/). Version 1.1.3 was
set in the tree and was never tagged or published. Do not push tag `v1.2.0`
until the checklist in [`docs/releases/ohho-os-1.2.0.md`](docs/releases/ohho-os-1.2.0.md)
is done. Pushing a `v*` tag runs `.github/workflows/release.yml`.

### Added

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
