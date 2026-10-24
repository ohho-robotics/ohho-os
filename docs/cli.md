# OhhO OS CLI reference

The `ohho` command is the entry point for simulation, hardware bring-up, navigation,
memory, skills, and the record→train→serve loop. Install with `pip install ohho-os`
(or `pip install -e .` from a checkout).

## Environment and robots

| Command | Description |
|---------|-------------|
| `ohho doctor` | Python version, runtimes, adapters, robots, optional extras |
| `ohho list` | Built-in robot specs and capabilities |
| `ohho version` | Package version (`ohho -V` is equivalent) |

## Connect and drive

Robot subcommands take a **robot id** (from `ohho list`). Use `--transport sim://`
for in-process simulation (no hardware, no safety gate).

| Command | Description |
|---------|-------------|
| `ohho connect <robot> [--transport sim://]` | Connect, print status and telemetry, disconnect |
| `ohho sim --robot omnibot --seconds 2` | Drive a scripted pattern in simulation |
| `ohho drive <robot> [--transport sim://] --vx … --seconds …` | Open-loop velocity for a duration |
| `ohho agent <robot> [--transport sim://] "<goal>"` | Run the agent loop (scripted brain without `[agent]` extra) |

Non-sim transports are wrapped in **SafetyGate** and start disarmed; see
[`safety.md`](safety.md).

## Navigation, perception, memory

| Command | Description |
|---------|-------------|
| `ohho nav map <robot> [--transport sim://] [--world demo]` | Spin-scan and print occupancy grid |
| `ohho nav explore <robot> … [--frontiers N] [--world demo]` | Frontier exploration |
| `ohho nav goto <robot> … --x X --y Y [--timeout SEC]` | Closed-loop goto |
| `ohho look <robot> [--transport sim://] [--world demo]` | Sim perception → spatio-temporal memory |
| `ohho memory show <robot>` | Summary of remembered objects |
| `ohho memory where <robot> <label>` | Last known position of a label |
| `ohho memory near <robot> --x X --y Y [--radius R]` | Objects near a point |
| `ohho memory clear <robot>` | Erase persisted memory for that robot |

## Local skill registry (`skills`)

Skills are **registered in-process** (Python `@skill` decorators and imports). There is
no network catalog and no package manager — only behaviors available in your
environment.

| Command | Description |
|---------|-------------|
| `ohho skills list` | List registered skills |
| `ohho skills run <robot> [--transport sim://] <skill_name>` | Execute a skill |

`ohho market` is a **deprecated alias** for `ohho skills` (same subcommands). Register
skills with `ohho.market.skill`; see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Profiles and inference

| Command | Description |
|---------|-------------|
| `ohho profile list` | Hardware deployment profiles |
| `ohho profile show <name>` | Profile details |
| `ohho profile detect` | Auto-detect this machine |
| `ohho serve [--mock] [--port 8000]` | FastAPI policy server (`[serve]` extra; `--mock` for CI/sim) |

With `--mock`, endpoints include `GET /health`, `POST /load_model`, and `POST /predict`.

## Simulation services

| Command | Description |
|---------|-------------|
| `ohho sim-serve --backend sim` | WebSocket teleop + telemetry (`[sim-serve]` / websockets) |
| `ohho sim-serve --backend isaac\|mujoco` | Exit non-zero with `not built` |
| `ohho sim-loop [--mock]` | Record → validate → train → serve → eval on CPU (see README) |
| `ohho sim-demo --sim --fast` | Scripted Go2 sim demo and eval log (simulation only; see [`sim-demo.md`](sim-demo.md)) |
| `ohho selftest --sim` | Bench diagnostics against simulated hardware (CI-safe) |

## Smoke tests

CI runs `python -m unittest discover -s tests/cli_smoke` after installing
`ohho-os[serve]`, covering the commands above on `sim://` and `--mock` paths only.

# OhhO CLI account (`ohho login`, `ohho whoami`, `ohho logout`)

Sign-in is optional. `sim://`, `ohho sim`, and the local `ohho mcp serve` /
`ohho mcp connect` stdio server keep working with no account. `ohho mcp
connect --hosted` stays refused until hosted MCP auth (OHH-274).

The access token is never printed by `login`, `whoami`, `logout`,
`ohho doctor`, or `ohho doctor --agent`, and it is not written to the log.

`ohho doctor --agent` and `ohho verify --sim` are documented below.

## Console

Device flow from OHH-272 (RFC 8628):

| Step | Method | Path |
|------|--------|------|
| Request a code | POST | `/api/cli/device/code` |
| Poll for a token | POST | `/api/cli/device/token` |
| Current account | GET | `/api/cli/whoami` |

The default API origin is `https://console.ohho-robotics.com`. Set
`OHHO_CONSOLE_URL` to an `http` or `https` origin (no path, no userinfo) to
talk to a preview host. The CLI sends `scope=mcp` and `client_name=OhhO CLI`.

`OHHO_TOKEN`, when set to a non-empty value, is the access token for
`ohho whoami`. It wins over the credentials file. It is never written to
disk, and `ohho logout` does not clear it.

## `ohho login`

```bash
ohho login
ohho login --no-browser
```

The command prints the user code and the verification URL from the console,
then opens a browser unless `--no-browser` is set or no browser is available.
It polls `POST /api/cli/device/token` until the code is approved, denied, or
expired. The wait between polls is the console's `interval`. A `slow_down`
error (or HTTP 429) switches to the returned `interval` or `Retry-After`,
or adds five seconds when the console does not send one. The wait is capped
at 60 seconds. Polling also stops at `expires_in` (at most one hour).

On success the access token is stored in `~/.ohho/credentials.json`. The
file mode is `0600` on POSIX. On Windows the DACL grants the current user
full control and removes inherited entries. The token is not printed.

If `OHHO_CONSOLE_URL` points at a different host than the verification URL,
the CLI still opens the URL the console returned, and also prints
`{OHHO_CONSOLE_URL}/device`. OHH-272 currently advertises
`https://console.ohho-robotics.com/device` from every deployment.

## `ohho whoami`

```bash
ohho whoami
ohho whoami --json
```

Text, when the console accepts the token:

```text
email: ada@example.com
scopes: mcp
```

`scopes` is space-separated. A missing email is printed as `unknown`.

Text, when no token is configured or the console returns 401:

```text
not signed in
```

The exit code is 1. A credentials file that cannot be parsed, a bad
`OHHO_CONSOLE_URL`, or a console that cannot be reached prints `error: …`
on stderr and exits 1. Those cases do not print the token or the file body.

### `ohho whoami --json`

UTF-8 JSON, one object, one line, no other stdout. Field order is fixed.

Signed in (exit 0):

| Field | Type | Meaning |
|-------|------|---------|
| `signed_in` | `true` | The console accepted the token. |
| `email` | string or null | Account email from `GET /api/cli/whoami`. |
| `scopes` | array of strings | Scopes from that response. The device flow requests `mcp`. |
| `user_id` | string or null | Console user id. |

```json
{"signed_in": true, "email": "ada@example.com", "scopes": ["mcp"], "user_id": "5b1c0b0e-0000-4000-8000-000000000001"}
```

Not signed in, or the console call failed (exit 1):

| Field | Type | Meaning |
|-------|------|---------|
| `signed_in` | `false` | |
| `email` | null | |
| `scopes` | `[]` | |
| `user_id` | null | |
| `error` | string | Why `signed_in` is false. |

`error` is one of:

| Value | When |
|-------|------|
| `not_signed_in` | No `OHHO_TOKEN` and no credentials file (or the file has no token). |
| `invalid_token` | The console returned 401 (unknown or revoked). |
| `credentials_unreadable` | The credentials file or home directory could not be read. |
| `console_unavailable` | `OHHO_CONSOLE_URL` is invalid, or the console could not be reached, or it returned a non-401 error. |

```json
{"signed_in": false, "email": null, "scopes": [], "user_id": null, "error": "not_signed_in"}
```

The access token is not a field. `email`, `user_id`, and `scopes` are dropped
when they contain the token.

## `ohho logout`

```bash
ohho logout
```

Deletes `~/.ohho/credentials.json` when that file exists. Running it again
still exits 0.

```text
Signed out. Local credentials removed.
```

OHH-272 has no CLI revoke endpoint. Logout does not call the console. A
token stays valid until it is revoked on the console account page (the
signed-in browser session sets `cli_tokens.revoked_at`). When `OHHO_TOKEN`
is set, logout also prints that the variable is still in effect and was not
written to disk. Later `ohho whoami` calls keep using it.

## Credentials file

Path: `~/.ohho/credentials.json` (`USERPROFILE` on Windows, otherwise
`HOME`). The CLI does not use `Path.home()`, so a process can point the file
elsewhere by setting those variables.

```json
{
  "version": 1,
  "token": "ohho_…",
  "token_type": "Bearer",
  "scope": "mcp",
  "console_url": "https://console.ohho-robotics.com"
}
```

`token` is the secret. Do not commit this file. `console_url` is the origin
that issued it. `OHHO_CONSOLE_URL` still wins for the next API call.

## `ohho doctor --agent`

```bash
ohho doctor
ohho doctor --agent
ohho doctor --agent --json
ohho doctor --agent --online
ohho doctor --agent --online --json
```

`ohho doctor` with no flag is unchanged: it prints the interpreter, runtimes,
adapters, and robots. `--json` and `--online` do nothing unless `--agent` is
set.

`ohho doctor --agent` is a readiness report for AI-agent setup. It does not
move the robot. Each check is `pass`, `fail`, or `skip`. The process exits 0
when no check failed. A `skip` is not a failure. Exit code 1 means at least
one check failed.

| Check | Pass | Fail | Skip |
|-------|------|------|------|
| `cli` | `ohho-os` version and install method | version string is missing or not numeric | the install layout could not be read |
| `python` | 3.10 or newer | older than 3.10 | the version could not be read |
| `login` | a token is configured (`OHHO_TOKEN` or the credentials file) | the credentials file cannot be parsed; or `--online` was asked and the console could not be reached | not signed in. Login is optional for `sim://` |
| `rules` | `ohho.agent_rules.installed_clients()` reports a path for every client | a client has no rules block, the hook raised, or it returned no clients | the `ohho.agent_rules` module or `installed_clients` is missing |
| `mcp` | at least one of the seven clients has an `ohho` MCP server entry | the home directory was readable and none of the clients have that entry | the home directory could not be resolved |
| `sim` | `omnibot` on `sim://` returned telemetry | connect or telemetry failed | the simulator is not connected |

Install method is a best-effort label: `pipx`, `pip --user`, `venv`,
`editable source`, or `unknown`. `unknown` is still a pass.

Login reports the email and the source (`env` or `file`) only. Offline (the
default) does not call the network, so the email is null: the credentials
file does not store one. `--online` calls `GET /api/cli/whoami` on
`OHHO_CONSOLE_URL` or `https://console.ohho-robotics.com`. A rejected token
is `skip` (signed out), not a crash. The token is not a JSON field and is
stripped from every string if it would otherwise appear.

### Rules check

`ohho rules install` writes the agent rules and `ohho.agent_rules.installed_clients()`
reports them. Paths, markers, and per-client files are in [rules.md](rules.md).

```python
def installed_clients() -> dict[str, str | None]:
    """Client id → rules path, or None when that client has no rules."""
```

Pass means every client has a path. Fail means at least one client is missing,
or the hook raised. The skip detail below is only used when this module cannot
be imported:

```text
skip: ohho rules install not available in this build
```

### MCP check

The seven clients are the ones `ohho mcp connect` knows (Claude Code, Codex,
Cursor, VS Code, Gemini CLI, Windsurf, OpenCode). Paths come from that
command's config table. The check only reads files. It does not create
backups or write config. One configured client is enough to pass; the JSON
still lists every client.

### `sim` check

Connects `omnibot` with transport `sim://`, reads telemetry once, and
disconnects. It does not send a velocity.

Text (illustrative; install method and Python version vary):

```text
OhhO OS 1.1.3 agent readiness
  cli     pass  ohho-os 1.1.3 (editable source)
  python  pass  3.12.10
  login   skip  not signed in (optional for sim)
  rules   fail  rules missing: claude-code, codex, cursor, vscode, gemini, windsurf, opencode
  mcp     fail  no ohho MCP server configured
  sim     pass  sim:// telemetry ok
```

That example exits 1 because `rules` and `mcp` failed.

### `ohho doctor --agent --json`

UTF-8 JSON, one object, no other stdout. Field order is fixed: `ok`,
`version`, `checks`. Each check starts with `id`, `status`, `detail`. Any
further keys are sorted by name.

| Field | Type | Meaning |
|-------|------|---------|
| `ok` | bool | True when no check has status `fail`. |
| `version` | string | `ohho-os` `__version__`. |
| `checks` | array | One object per check, in the order cli, python, login, rules, mcp, sim. |
| `checks[].id` | string | `cli`, `python`, `login`, `rules`, `mcp`, or `sim`. |
| `checks[].status` | string | `pass`, `fail`, or `skip`. |
| `checks[].detail` | string | Short human text. Never contains the access token. |
| `checks[].version` | string | Present on `cli` and `python`. |
| `checks[].install` | string | Present on `cli`. |
| `checks[].source` | string or null | `login`: `env`, `file`, or null. |
| `checks[].email` | string or null | `login`: account email when `--online` returned one. |
| `checks[].clients` | object | `rules`: client → path or null. `mcp`: client → `{configured, path, detail}`. |
| `checks[].robot` | string | `sim`: the robot id that was connected. |

```json
{
  "ok": false,
  "version": "1.1.3",
  "checks": [
    {
      "id": "cli",
      "status": "pass",
      "detail": "ohho-os 1.1.3 (editable source)",
      "install": "editable source",
      "version": "1.1.3"
    },
    {
      "id": "python",
      "status": "pass",
      "detail": "3.12.10",
      "version": "3.12.10"
    },
    {
      "id": "login",
      "status": "skip",
      "detail": "not signed in (optional for sim)",
      "email": null,
      "source": null
    },
    {
      "id": "rules",
      "status": "fail",
      "detail": "rules missing: claude-code, codex, cursor, vscode, gemini, windsurf, opencode",
      "clients": {
        "claude-code": null,
        "codex": null,
        "cursor": null,
        "vscode": null,
        "gemini": null,
        "windsurf": null,
        "opencode": null
      }
    },
    {
      "id": "mcp",
      "status": "fail",
      "detail": "no ohho MCP server configured",
      "clients": {
        "cursor": {
          "configured": false,
          "path": "/home/ada/.cursor/mcp.json",
          "detail": "not configured"
        }
      }
    },
    {
      "id": "sim",
      "status": "pass",
      "detail": "sim:// telemetry ok",
      "robot": "omnibot"
    }
  ]
}
```

The real `mcp.clients` object has all seven clients. The example shows the
shape with `cursor` only. There is no `token` field.

Signed in, offline:

```json
{"id": "login", "status": "pass", "detail": "signed in (source: env)", "email": null, "source": "env"}
```

Signed in, `--online`:

```json
{"id": "login", "status": "pass", "detail": "signed in as ada@example.com (source: file)", "email": "ada@example.com", "source": "file"}
```

## `ohho verify --sim`

```bash
ohho verify --sim
ohho verify --sim --json
ohho verify --sim --robot omnibot
ohho verify --sim --console
```

`--sim` is required. Without it the command prints an error and exits 2.
Hardware is never the default. The only backend is `sim://`.

There is no transport option that selects a real robot. `--transport` with
any value other than `sim://` exits 2. The same refusal applies when any of
these name a non-sim URI:

| Source | Name |
|--------|------|
| Environment | `OHHO_TRANSPORT`, `OHHO_ROBOT_TRANSPORT`, `OHHO_VERIFY_TRANSPORT`, `OHHO_URI` |
| Flag | `--transport`, or a `--robot` value that is itself a non-sim URI |
| Robot config | `OHHO_ROBOT_CONFIG` pointing at JSON whose `transport`, `uri`, or `url` is not `sim://` |

The error names the flag or variable. It does not repeat the URI. A manifest
that only sets `adapter` (and no URI) is not a hardware connection: verify
still builds `sim://`. `OHHO_TRANSPORT=sim://` is ignored and the command
still uses its own simulator.

On success the command connects `omnibot` (or `--robot`) on `sim://`, reads
telemetry, drives 0.1 m/s forward for 1 second, stops, reads telemetry again,
and prints the distance moved. The pass band is distance > 0 and distance ≤
0.1 m plus a 0.05 m tolerance (acceleration limiting means the robot does not
spend the whole second at 0.1 m/s). Exit 0 on pass, 1 on fail.

The sim transport is wrapped in `SafetyGate` before any motion. Caps, the
300 ms command watchdog, and the action allowlist apply. Commands are sent
every 50 ms so the deadman stays fed, then an explicit zero-velocity stop is
sent. The stop also runs if the drive raises or the process is interrupted.
`release_stop` is never called. Limits are the built-in profile;
`OHHO_SAFETY_CONFIG` is not applied. `OHHO_ARM_HARDWARE=1` does not select
hardware and does not change the distance.

`Robot.connect` does not put `sim://` behind the gate. verify wraps the
simulator itself and calls `arm()`, which succeeds without a TTY prompt
because a simulated link does not require hardware arming. The prompt is
disabled on this command, so a mis-detected hardware link cannot be armed.

The safety log may print `safety: <robot> ARMED (simulated)` on stderr. That
line is the simulator gate, not a hardware arm.

`--console` (off by default) sends `GET {origin}/sim`. The origin is
`OHHO_CONSOLE_URL` or `https://console.ohho-robotics.com`. HTTP 404, any
other non-200, and network errors are reported as `skip` / `unavailable`.
They do not crash the process and they do not fail a drive that otherwise
passed. Production `/sim` is 404 today. No HTTP call is made unless
`--console` is set. `ohho doctor --agent` makes no HTTP call unless
`--online` is set.

Text:

```text
verify sim:// omnibot: pass
  distance: 0.086 m
  tolerance: 0.050 m
  commanded: 0.1 m/s for 1 s
```

### `ohho verify --sim --json`

One JSON object. Field order is fixed. `console` is null unless `--console`
was set.

| Field | Type | Meaning |
|-------|------|---------|
| `ok` | bool | True when the distance is inside the pass band. |
| `robot` | string | Robot id. Default `omnibot`. |
| `transport` | string | Always `sim://`. |
| `distance_m` | number | Straight-line distance from the start pose, metres. |
| `tolerance_m` | number | 0.05. |
| `commanded_mps` | number | 0.1. |
| `duration_s` | number | 1. |
| `detail` | string | `moved 0.086 m` or an outside-tolerance / arming failure line. |
| `console` | object or null | `{status, detail}` when `--console` is set. `status` is `pass` or `skip`. |

```json
{
  "ok": true,
  "robot": "omnibot",
  "transport": "sim://",
  "distance_m": 0.086,
  "tolerance_m": 0.05,
  "commanded_mps": 0.1,
  "duration_s": 1.0,
  "detail": "moved 0.086 m",
  "console": null
}
```

With `--console` when production returns 404:

```json
{"status": "skip", "detail": "unavailable (HTTP 404)"}
```

A refusal (missing `--sim`, or a non-sim URI) exits 2. With `--json`, stdout
is one object and the URI is not copied into it:

```json
{
  "ok": false,
  "robot": "omnibot",
  "transport": "sim://",
  "error": "ohho verify accepts only sim:// (refusing --transport)"
}
```

Without `--json`, the same message is only on stderr (`error: …`).

## Human TODO

- Run `ohho login`, `ohho whoami`, and `ohho logout` once against a console
  preview (`OHHO_CONSOLE_URL=https://<preview>`). This repository does not
  call the production console from tests. If the printed verification URL is
  the production `/device` page, open `<preview>/device` and enter the same
  user code.
- Server-side revoke from the CLI needs a console endpoint. Until that
  exists, revoke the token from the console account page after `ohho logout`.
- Run `ohho doctor --agent --online` once against a console preview. Tests
  mock `GET /api/cli/whoami` and do not call production. Confirm the email
  matches the account and that the token is absent from stdout and stderr.
- Run `ohho verify --sim --console` once. Against production, `/sim` is 404
  and the report should say `skip` / `unavailable` while the sim drive still
  passes. Against a preview where `/sim` exists, it should say `reachable`.
  Tests mock that GET.
- `ohho rules install` is documented in [rules.md](rules.md). `ohho doctor
  --agent` reads `ohho.agent_rules.installed_clients()` and fails the rules
  check until every client file contains the Ohho block.
