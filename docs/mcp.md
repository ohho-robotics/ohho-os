# OhhO OS MCP server (Link v0)

Honest, small surface: **five tools** backed by the real Python SDK against **`sim://` only**. No hardware links, no fictional tool catalog.

| Tool | What it does |
|------|----------------|
| `connect` | `Robot.connect` in simulation (`sim://` enforced) |
| `drive` | Velocity command for a duration, then stop |
| `telemetry` | Odometry, battery, joints |
| `nav_goto` | Native `Navigator.navigate_to` |
| `look` | `SimPerceptor.look` + spatial memory |

Install:

```bash
pip install -e '.[mcp]'
ohho doctor   # should list mcp: ready — 5 tools …
```

Run (stdio — how Claude Desktop and Cursor spawn MCP servers):

```bash
ohho mcp
```

## Claude Desktop

Copy or merge into `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS). Use the absolute path to `ohho` if it is not on `PATH`.

See [examples/mcp-claude-desktop.json](../examples/mcp-claude-desktop.json).

## Cursor

Add an MCP server in **Cursor Settings → MCP** or edit your user `mcp.json`. See [examples/mcp-cursor.json](../examples/mcp-cursor.json).

## MCP Inspector

With Node installed:

```bash
npx @modelcontextprotocol/inspector ohho mcp
```

Connect, call `connect` with `{"robot":"omnibot","transport":"sim://"}`, then `telemetry` or `drive`.

## Human TODO

- Record a short GIF of an LLM (via MCP) driving the sim robot and attach it to the OHH-47 issue / marketing — **not** generated or faked in CI.

## Safety

This server **refuses** non-`sim` transports at `connect`. It does not expose `arm`, `release_stop`, or raw adapter commands.
