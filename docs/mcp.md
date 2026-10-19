# OhhO OS MCP server (Link v0)

`ohho mcp serve` is the on-robot stdio server from OHH-173 (`ohho/mcp_server.py`,
`ohho/mcp_tools.py`). OHH-47 adds **five tools** backed by the real Python SDK
against **`sim://` only**. `ohho mcp` with no subcommand still serves, so
existing `ohho mcp --sim` invocations keep working.

| Tool | What it does |
|------|----------------|
| `connect` | `Robot.connect`. `sim://` unless the human passed `--allow-hardware`. Never arms. |
| `drive` | Velocity for a duration, then stop. Clamped to the sim or profile cap. |
| `telemetry` | Odometry, battery, joints |
| `nav_goto` | Native `Navigator.navigate_to` (simulation) |
| `look` | `SimPerceptor.look` + spatial memory (simulation) |
| `emergency_stop` | Latch e-stop on sim or a gated link. Always allowed. On an aircraft it follows the drone e-stop map (land, or hold then land). |

Aircraft tools (OHH-293) use the same server and the same `--allow-hardware`
switch. They are listed in [Drone tools](#drone-tools-ohh-293).

The same process also serves the OHH-173 on-robot catalog (`connect.listRobots`,
`connect.connect`, `connect.emergencyStop`, `bridge.listAdapters`, and the
other on-robot tools). Those tools follow the same transport rule.

No extra package is required — the server is stdlib:

```bash
ohho mcp serve
ohho mcp serve --sim
# human only — opens a non-sim URI through SafetyGate, still disarmed:
# ohho mcp serve --allow-hardware --transport serial:///dev/ttyUSB0
```

`ohho mcp connect` (OHH-275) registers that local server with an AI client.
The default entry is this stdio server (`sim://` only). `--hosted` is reserved
and exits with an error: it needs `ohho login` (OHH-273) and hosted MCP auth
(OHH-274). Account commands are documented in [cli.md](cli.md).

## `ohho mcp connect`

```bash
ohho mcp connect --list
ohho mcp connect --client cursor
ohho mcp connect --client claude-code --client codex
ohho mcp connect --client vscode --dry-run
ohho mcp connect --client cursor --print
ohho mcp connect --client gemini --remove
```

| Flag | Effect |
|------|--------|
| `--client` | Client id. Repeat the flag to configure more than one. |
| `--list` | Print the client ids and where each config lives. |
| `--dry-run` | Print a unified diff (or the `claude` / `codex` command) and change nothing. |
| `--print` | Print the merged config, or the CLI command when a client CLI is used. |
| `--remove` | Delete only the `ohho` entry. Other servers stay. |
| `--hosted` | Refused. Needs `ohho login` (OHH-273) and hosted MCP auth (OHH-274). |

The command stored in file-based configs is the absolute path of the `ohho`
console script (`pipx` shim, venv `bin/ohho` or `Scripts/ohho.exe`), then
`mcp serve`. GUI clients often do not inherit a shell `PATH`.

File updates **merge**. The writer parses the file, adds or replaces only
`ohho`, and leaves every other key alone. An existing file is copied to
`<name>.bak` first (for example `mcp.json.bak`), then replaced atomically
(temp file in the same directory, then rename). An empty or missing file is
created. A file that does not parse as JSON or TOML is left untouched, with
no backup, and the error names the path. A second run that would not change
the bytes does not write another backup.

`claude` and `codex` are invoked when those programs are on `PATH`. Otherwise
the same entry is written to the fallback file below. Tests inject a fake
runner; they do not execute the real CLIs.

Client ids: `claude-code` (alias `claude`), `codex`, `cursor`, `vscode`
(alias `vs-code`), `gemini` (alias `gemini-cli`), `windsurf`, `opencode`.

### Claude Code

Source: [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp).

User scope is the top-level `mcpServers` object in `~/.claude.json` (not
`~/.claude/settings.json`, and not a project entry under `projects`). When
`claude` is on `PATH`:

```bash
claude mcp add --scope user ohho -- /absolute/path/to/ohho mcp serve
```

Current docs use `claude mcp add [options] <name> -- <command> [args...]`,
so `--scope user` comes before the server name. If `claude` is not installed,
the fallback file is `~/.claude.json`:

```json
{
  "mcpServers": {
    "ohho": {
      "command": "/absolute/path/to/ohho",
      "args": ["mcp", "serve"]
    }
  }
}
```

An entry with no `type` is stdio. Other top-level keys, including `projects`,
are preserved.

### Codex

Source: [Model Context Protocol](https://learn.chatgpt.com/docs/extend/mcp)
(Codex CLI, ChatGPT desktop, and the IDE extension share this file).

When `codex` is on `PATH`:

```bash
codex mcp add ohho -- /absolute/path/to/ohho mcp serve
```

Otherwise `~/.codex/config.toml` gains or replaces only `[mcp_servers.ohho]`:

```toml
[mcp_servers.ohho]
command = "/absolute/path/to/ohho"
args = ["mcp", "serve"]
```

Comments and other tables stay in place. The editor is stdlib-only (Python
3.10 has no `tomllib`). A document it cannot parse is refused.

### Cursor

Source: [Model Context Protocol (MCP)](https://cursor.com/docs/mcp).

Global file: `~/.cursor/mcp.json` (`%USERPROFILE%\.cursor\mcp.json` on
Windows). Key: `mcpServers.ohho`, with `command` and `args`.

### VS Code

Sources: [Add and manage MCP servers](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
and the user settings directory in
[User and workspace settings](https://code.visualstudio.com/docs/configure/settings).

User `mcp.json` uses `servers` (not `mcpServers`) and requires `"type": "stdio"`.

| OS | Path |
|----|------|
| Windows | `%APPDATA%\Code\User\mcp.json` |
| macOS | `~/Library/Application Support/Code/User/mcp.json` |
| Linux | `$XDG_CONFIG_HOME/Code/User/mcp.json`, or `~/.config/Code/User/mcp.json` |

On Linux, `XDG_CONFIG_HOME=$HOME/.config`, `${HOME}/.config`, or `~/.config` follows the same home directory as the other client files. Any other absolute path is used as written.

```json
{
  "servers": {
    "ohho": {
      "type": "stdio",
      "command": "/absolute/path/to/ohho",
      "args": ["mcp", "serve"]
    }
  }
}
```

### Gemini CLI

Sources: [MCP servers with Gemini CLI](https://geminicli.com/docs/tools/mcp-server/)
and [settings](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/settings.md).

User file: `~/.gemini/settings.json`. Key: `mcpServers.ohho`, with `command`
and `args`. Workspace `.gemini/settings.json` is not modified.

### Windsurf

Source: [Cascade MCP](https://docs.windsurf.com/windsurf/cascade/mcp).

File: `~/.codeium/windsurf/mcp_config.json` on macOS, Linux, and Windows.
Key: `mcpServers.ohho`, with `command` and `args`. This is the legacy Cascade
agent file.

### OpenCode

Sources: [MCP servers (v1)](https://opencode.ai/docs/mcp-servers/),
[MCP servers (v2)](https://opencode.ai/v2/docs/mcp-servers/),
and [Migrate from V1](https://opencode.ai/v2/docs/migrate-v1/).

Global file, every OS: `~/.config/opencode/opencode.json`.

OHH-275 names the v1 shape: `mcp.ohho`, `"type": "local"`, and `command` as
an array. New files and v1 configs are written that way. OpenCode v2 reads
that shape in memory. When the file already has a `mcp.servers` object, the
entry is written at `mcp.servers.ohho` so the other v2 servers stay put.

```json
{
  "mcp": {
    "ohho": {
      "type": "local",
      "command": ["/absolute/path/to/ohho", "mcp", "serve"]
    }
  }
}
```

## Claude Desktop

Copy or merge into `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).
Use the absolute path to `ohho` if it is not on `PATH`. `ohho mcp connect`
does not write the Claude Desktop file; it writes Claude Code's
`~/.claude.json`.

See [examples/mcp-claude-desktop.json](../examples/mcp-claude-desktop.json).

## Cursor (manual)

`ohho mcp connect --client cursor` writes the global file. To edit it by
hand, use [examples/mcp-cursor.json](../examples/mcp-cursor.json) and replace
`command` with the absolute path of `ohho`.

## MCP Inspector

With Node installed:

```bash
npx @modelcontextprotocol/inspector ohho mcp serve --sim
```

Connect, call `connect` with `{"robot":"omnibot","transport":"sim://"}`, then
`telemetry` or `drive`.

## Safety (OHH-278)

Default is `sim://`. `connect` and `connect.connect` refuse any other URI
with `hardware_refused` / `sim_only`. The text still says the tools only
support `sim://`.

`--allow-hardware` is the only switch, and only on the `ohho mcp serve`
command a human starts. `OHHO_ALLOW_HARDWARE`, `OHHO_ARM_HARDWARE`, and tool
arguments (`allowHardware`, `allow_hardware`) do not enable hardware.
`ohho mcp connect` writes `mcp serve` with neither `--allow-hardware` nor
`--arm`.

With the flag, a non-sim URI goes through `Robot.connect`, which wraps it in
`SafetyGate` (caps, 300 ms command watchdog, 500 ms state watchdog, latched
e-stop, sport allowlist, audit). The server does not call `arm()` or
`release_stop()`. Stdin is the JSON-RPC pipe, so there is no interactive TTY
and arming cannot succeed. `drive`, `connect.sendVelocity`,
`pilot.applyJoystick`, and `autonomy.navigateTo` return `motion_refused` /
`not_armed` and send zero velocity. The message tells the agent to ask the
human to arm in their own terminal.

`emergency_stop` and `connect.emergencyStop` work on simulation and on a
gated link. They latch. A later velocity call returns `estop_latched`. No
tool clears the latch.

Registration rejects a tool whose name arms, releases an e-stop, or changes
a profile or a limit. `tools/list` has no `arm`, `release_stop`,
`release-stop`, `profile`, or `limit` tool. `emergency_stop` is allowed.

Results that use a robot include `"backend": "sim"` or `"backend": "hardware"`.
Velocity results include `requested`, `applied`, `clamped`, and `cap`. A
2 m/s request comes out at or below the simulator cap (OmniBot 0.2 m/s) or
the safety-profile cap (Go2 forward 0.5 m/s), whichever applies.

`look` and `nav_goto` stay on the simulator even when hardware is allowed.
`nav_goto` on a disarmed hardware link is refused before it drives.

`ohho mcp connect` only records how to launch the server. It does not arm
hardware and it does not call `Robot.connect`. See [safety.md](safety.md).

## Drone tools (OHH-293)

Experimental. Nothing has flown. `flight_ready` stays false.

| Tool | What it does |
|------|----------------|
| `drone_connect` | `Robot.connect` for an aircraft. Default robot `px4-sitl`, default URI `sim://`. Never calls `arm()` or `arm_flight()`. |
| `drone_telemetry` | Read `custom["drone"]`, battery, and the planar odometry shadow. No setpoint. |
| `drone_velocity` | `offboard_velocity`. Clamped before the backend. Hardware only after a human has armed the gate from a TTY. The tool does not arm. |
| `drone_hold` | Autopilot hold. Does not arm. |
| `drone_land` | Land. Always attempted, including when the server was started without `--allow-hardware`. Does not arm and does not clear an e-stop. |
| `emergency_stop` | The existing tool. Always allowed. Latches. On a gated aircraft it uses the drone e-stop map. On `drone-sim` it freezes horizontal motion and descends. |

`drone_arm`, `drone_takeoff`, `drone_rtl`, and `release_stop` are not
registered. The same denylist rejects a future tool whose name is `takeoff`,
`rtl`, `arm_flight`, or a fence change (`set_fence`, `fence_change`).
`drone_mcp_allowed` is checked again when the drone tools are registered:
`hold`, `land`, and `offboard_velocity` must pass, or the server refuses to
start those tools.

`--allow-hardware` is the OHH-278 flag. It is not a drone-only switch.
`OHHO_ARM_HARDWARE`, `OHHO_ALLOW_HARDWARE`, and tool arguments do not enable
hardware and do not arm. The process stdin is the JSON-RPC pipe, so there is
no interactive TTY.

`drone_connect` to `serial://`, `px4+dds://`, or any other non-`sim` URI
without that flag returns `hardware_refused` / `sim_only` and does not open
the link. A ground robot id such as `omnibot` returns `not_aerial`.

Every drone result includes `"backend": "sim"` or `"backend": "hardware"`.

### Caps and geofence

Clamps are applied in the tool, before `offboard_velocity`:

| | Cap |
|--|-----|
| Horizontal speed | 1.5 m/s (vector magnitude) |
| Vertical speed | 1.0 m/s |
| Yaw rate | 0.8 rad/s |
| Altitude | 3 m on `sim://`. On hardware, the gate's geofence ceiling |

A 3 m/s request is clamped to 1.5 m/s. The sim backend's own horizontal cap
is 2 m/s; the tool does not hand it a value above the drone profile.

The fence is the cylinder from the drone profile (radius 10 m, and the
altitude ceiling above). The check uses the clamped velocity over the
profile's command window (0.3 s).

- The vehicle is already outside: error `geofence`, and hold is requested.
  The position is not clipped onto the fence.
- The clamped velocity would leave the cylinder inside that window: same
  error and hold. A climb is not silently shortened so the craft stops on
  the ceiling.
- The pose is not known yet: the velocity is clamped and sent. Hardware
  still has to be armed or the tool returns `not_armed` and sends nothing.

`frame` is required: `LOCAL_NED`, `LOCAL_ENU`, or `BODY_FRD`. There is no
default. In `LOCAL_NED` and `BODY_FRD`, positive `vz` is down.

On a disarmed hardware gate, `drone_land` and `drone_hold` are still
invoked. The gate drops them until a human arms from a TTY (`not_armed`).
This server does not call `arm()` to make them succeed. `emergency_stop`
still latches and, in the air, asks the backend to land.

## Manual check

A person should confirm, in Cursor and in Claude Code, that the `ohho` server
shows up and that `tools/list` returns `connect`, `drive`, `telemetry`,
`nav_goto`, `look`, and `emergency_stop`, and does not return `arm` or
`release_stop`. That check is not run by CI.

Supervised hardware check (Varun, not CI, not an agent): with the robot
present and the area clear, start `ohho mcp serve --allow-hardware` against
the real transport and confirm motion is refused until the robot is armed
from a separate terminal. Do not arm from the MCP client. Confirm
`emergency_stop` latches. This checkout does not run that check.
