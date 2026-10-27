# OhhO OS 1.2.0

**Not released.** This note prepares `ohho-os` 1.2.0. It does not publish to
PyPI and it does not create a git tag. [PyPI](https://pypi.org/project/ohho-os/)
still has **1.1.2**. Version 1.1.3 was in the tree and was never published.

The commands below are in this package. They talk to the **in-process
simulator** (`sim://`) unless a person, at a real terminal, chooses hardware
and arms it. An agent install should stop at simulation.

Setup playbook: [https://ohho-robotics.com/onboard.md](https://ohho-robotics.com/onboard.md)
is **coming** (site PR #53, OHH-282). That page is not live with this package.
Do not treat the URL as published instructions until OHH-282 lands.

## Install

The base install is still dependency-free (Python 3.10+):

```bash
pip install ohho-os
```

`pipx install ohho-os` is the install the setup playbook will recommend.
Either install provides the `ohho` command. From a git checkout,
`pip install -e .` is the same base install, editable.

Sign-in is optional. `ohho sim` and `ohho mcp serve` work with no account.

## What you can run

### Sign in

`ohho login` is the console device-flow client (OHH-273, RFC 8628). It prints
a user code and a verification URL, opens a browser unless you pass
`--no-browser`, and waits until the code is approved, denied, or expired.
The access token is stored in `~/.ohho/credentials.json` (mode `0600` on
POSIX; a current-user-only ACL on Windows). It is not printed.

```bash
ohho login
ohho whoami
ohho whoami --json
ohho logout
```

`OHHO_TOKEN`, when set, is used instead of the file and is never written.
`ohho logout` deletes the local file only. It does not revoke the token on
the console, and it does not unset `OHHO_TOKEN`. Details and the JSON schema
are in [cli.md](../cli.md).

A successful login needs the console device endpoints (OHH-272) at
`https://console.ohho-robotics.com`, or at `OHHO_CONSOLE_URL`. This repository
does not call that host from tests.

### Simulator MCP server

`ohho mcp serve` is a stdio MCP server (OHH-47). No extra package is required.
The tools that drive the simulator are `connect`, `drive`, `telemetry`,
`nav_goto`, and `look`. `emergency_stop` latches a stop. The default
transport is `sim://`.

```bash
ohho mcp serve
ohho mcp serve --sim
```

`ohho mcp` with no subcommand still serves, so an existing `ohho mcp --sim`
entry keeps working.

### Connect an AI client

`ohho mcp connect --client` writes that local server into one of seven
clients (OHH-275): Claude Code, Codex, Cursor, VS Code, Gemini CLI, Windsurf,
OpenCode. It merges the `ohho` entry into the existing config, copies the
old file to `.bak`, and does not remove other servers. The recorded command
is the absolute path of `ohho`, then `mcp serve`, with no hardware flag.

```bash
ohho mcp connect --list
ohho mcp connect --client cursor
ohho mcp connect --client claude-code --dry-run
```

`ohho mcp connect --hosted` exits with an error. Hosted MCP auth is OHH-274
and is not in this release. Paths and the merge rules are in [mcp.md](../mcp.md).

### Agent rules

`ohho rules install --client` copies one safety text into those same seven
clients (OHH-279). The text lives in `ohho/agent_rules/ohho-rules.md` and is
shipped inside the wheel. It tells an agent to use `sim://`, not to arm, not
to release an e-stop, and to stop when safety refuses a command. The rules
text does not itself block a hardware URI; the server does (below).

```bash
ohho rules install --client cursor
ohho rules list
ohho rules show
```

Paths and markers are in [rules.md](../rules.md).

### Check the setup

`ohho doctor --agent` reports CLI version, Python, login, rules, MCP config,
and whether `omnibot` on `sim://` returns telemetry. It does not drive.
Offline is the default: it does not call the console unless you pass
`--online`.

```bash
ohho doctor --agent
ohho doctor --agent --json
```

`ohho verify --sim` is the drive check. `--sim` is required. Any non-sim URI
is refused before a connection is opened. On success it drives the simulator
forward at 0.1 m/s for 1 second, stops, and prints the distance. The simulator
is wrapped in `SafetyGate` for that drive. The command does not read
`OHHO_ARM_HARDWARE` and does not call `release_stop`.

```bash
ohho verify --sim
ohho verify --sim --json
```

`--console` optionally requests the console `/sim` page. A 404 is reported
as unavailable and does not fail a drive that passed. Production `/sim` is
404 today. The doctor and verify schemas are in [cli.md](../cli.md).

## Safety

Hardware is not the default.

- `ohho mcp serve` accepts `sim://` only, unless the person who started the
  process passed `--allow-hardware` (OHH-278). Environment variables and tool
  arguments cannot turn that on. `ohho mcp connect` does not write the flag.
- With `--allow-hardware`, a non-sim URI is opened through `Robot.connect`,
  which wraps it in `SafetyGate` (caps, 300 ms command watchdog, 500 ms state
  watchdog, latched e-stop, sport-action allowlist, audit log). The MCP
  process never calls `arm()` or `release_stop()`. Its stdin is the JSON-RPC
  pipe, so it cannot complete the interactive arming prompt. Velocity tools
  return `not_armed` and send zero velocity until a person arms the robot
  from their own terminal.
- `emergency_stop` is allowed on the simulator and on a gated link. It
  latches. No MCP tool clears the latch. Tools whose names arm, release a
  stop, or change a profile or a limit are not registered.
- `look` and `nav_goto` stay on the simulator even when hardware is allowed.
- `ohho verify --sim` never opens a hardware transport.

The engineering rules for `SafetyGate` are in [safety.md](../safety.md) and
in `AGENTS.md` §9a. This release does not weaken them.

## Not in this release

- No PyPI publish and no `v1.2.0` tag. Those wait for the checklist below.
- `ohho mcp connect --hosted` (OHH-274).
- A CLI revoke call. `ohho logout` is local only (OHH-272).
- The public setup page `onboard.md` (OHH-282, site PR #53).
- PR #20 (OHH-171), which re-points project URLs at `ohho-os` and adds
  `docs/RELEASING.md`. Merge that before tagging. Do not re-do it here.

## Release checklist

Human steps. Do this in order. Do not tag early.

### 1. Merge order

Merge onto `main` from the bottom of the stack. This branch
(`ohh-281-release-1-2-0`) was cut from `ohh-278-mcp-safety` and already
contains that stack. Review each PR on its own, then merge this one.

1. **#22** (OHH-173, `ohh-173-mcp-split`) and **#35** (OHH-47,
   `ohh-47-mcp-real-tools`). #35 needs #22. Together they are `ohho mcp serve`.
2. **#49** (OHH-275, `ohh-275-mcp-connect-clients`) — `ohho mcp connect --client`.
3. **#50** (OHH-273, `ohh-273-cli-login`) — `ohho login` / `whoami` / `logout`.
4. **#51** (OHH-280, `ohh-271-p7-doctor-verify`) — `ohho doctor --agent` and
   `ohho verify --sim`.
5. **#52** (OHH-279, `ohh-271-p6-rules-install`) — `ohho rules install --client`.
6. **OHH-278 PR** (branch `ohh-278-mcp-safety`). As of 19 Oct 2026 that branch
   had no open pull request number. It keeps MCP on `sim://` unless a human
   passes `--allow-hardware`.
7. **Merge #20 before tagging** (OHH-171, `ohh-171-pypi-repoint`). It re-points
   project URLs, adds `docs/RELEASING.md` and `tests/test_packaging.py`, and
   adds a TestPyPI `workflow_dispatch` on `release.yml`. Trusted-publisher
   steps for the `ohho-sdk` → `ohho-os` rename are in that PR's
   `docs/RELEASING.md`. Do not copy them into this note.
8. **This PR** (OHH-281, `ohh-281-release-1-2-0`) — version `1.2.0`, this
   note, and `CHANGELOG.md`.

### 2. Go-ahead, then tag

Varun confirms the stack above is on `main`, `pyproject.toml` `version` is
`1.2.0`, and the trusted publisher in #20's `docs/RELEASING.md` points at
repository `ohho-os`. Then Varun, not this prep, pushes the tag:

```bash
git tag v1.2.0
git push origin v1.2.0
```

That push runs `.github/workflows/release.yml`: classifier check, tag version
must match `pyproject.toml`, `python -m build`, `twine check`, publish with
trusted publishing (environment `pypi`, no API token), then `pip install
ohho-os==<version>` on Ubuntu and macOS and `ohho doctor`.

This prep does not change that workflow. The wheel already includes
`ohho/agent_rules/ohho-rules.md` and `ohho/robots/example.json` via
`[tool.setuptools.package-data]`. There is no TOML template in the package;
Codex config is written at connect time by `ohho/mcp_toml.py`.

### 3. After PyPI has 1.2.0

In a clean virtualenv:

```bash
python -m pip install "ohho-os==1.2.0"
ohho login --help
ohho mcp serve --help
ohho mcp connect --help
ohho rules install --help
ohho doctor --agent
```

Then `pipx install ohho-os` on Windows, macOS, and Linux, and run
`ohho version`.

Flip the "not released" markers in the site's `onboard.md` (OHH-282, site
PR #53) only after that install works.

Also still open, from the command docs, and not done by CI:

- Run `ohho login`, `ohho whoami`, and `ohho logout` once against a console
  preview (`OHHO_CONSOLE_URL`). Tests do not call the production console.
- Run `ohho doctor --agent --online` once against that preview and confirm
  the token is absent from the output.
- Run `ohho verify --sim --console` once. Against production, `/sim` is 404
  and the drive should still pass.
- Supervised hardware check (Varun, robot present, area clear): `ohho mcp
  serve --allow-hardware` refuses motion until the robot is armed from a
  separate terminal. Do not arm from the MCP client.
