# OhhO OS MCP server (Link v0)

`ohho mcp` is the on-robot stdio server from OHH-173 (`ohho/mcp_server.py`,
`ohho/mcp_tools.py`). OHH-47 adds **five tools** backed by the real Python SDK
against **`sim://` only**.

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
ohho mcp --sim
```

## Claude Desktop

Copy or merge into `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS). Use the absolute path to `ohho` if it is not on `PATH`.

See [examples/mcp-claude-desktop.json](../examples/mcp-claude-desktop.json).

## Cursor

Add an MCP server in **Cursor Settings → MCP** or edit your user `mcp.json`. See [examples/mcp-cursor.json](../examples/mcp-cursor.json).

## MCP Inspector

With Node installed:

```bash
npx @modelcontextprotocol/inspector ohho mcp --sim
```

Connect, call `connect` with `{"robot":"omnibot","transport":"sim://"}`, then `telemetry` or `drive`.

## Safety

The Link v0 tools **refuse** non-`sim` transports at `connect`, and refuse `drive`, `telemetry`, `nav_goto`, and `look` unless the session robot is simulated. They do not expose `arm`, `release_stop`, or raw adapter commands.
