# OhhO OS MCP server (Link v0)

`ohho mcp serve` is the on-robot stdio server from OHH-173 (`ohho/mcp_server.py`,
`ohho/mcp_tools.py`). OHH-47 adds **five tools** backed by the real Python SDK
against **`sim://` only**. `ohho mcp` with no subcommand still serves, so
existing `ohho mcp --sim` invocations keep working.

| Tool | What it does |
|------|----------------|
| `connect` | `Robot.connect` in simulation (`sim://` enforced) |
| `drive` | Velocity command for a duration, then stop |
| `telemetry` | Odometry, battery, joints |
| `nav_goto` | Native `Navigator.navigate_to` |
| `look` | `SimPerceptor.look` + spatial memory |

The same process also serves the OHH-173 on-robot catalog (`connect.listRobots`,
`connect.connect`, `bridge.listAdapters`, and the other on-robot tools). The
five tools above refuse any session that is not simulated.

No extra package is required — the server is stdlib:

```bash
ohho mcp serve
ohho mcp serve --sim
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

## Safety

The Link v0 tools **refuse** non-`sim` transports at `connect`, and refuse
`drive`, `telemetry`, `nav_goto`, and `look` unless the session robot is
simulated. They do not expose `arm`, `release_stop`, or raw adapter commands.
`ohho mcp connect` only records how to launch that server. It does not arm
hardware and it does not call `Robot.connect`.

## Manual check

A person should confirm, in Cursor and in Claude Code, that the `ohho` server
shows up and that `tools/list` returns the five sim tools (`connect`, `drive`,
`telemetry`, `nav_goto`, `look`). That check is not run by CI.
