# @ohho/mcp (removed)

MCP tool implementations were split in **OHH-173**:

- **Hosted tools** (garage, fleet, market, train, …) live in the private
  [console.ohho-robotics.com](https://github.com/ohho-robotics/console.ohho-robotics.com)
  repo under `lib/<product>/mcp-tools.ts` and are exposed at `/api/mcp`.
- **On-robot tools** (connect, bridge, autonomy, pilot, …) live in
  [ohho-os](https://github.com/ohho-robotics/ohho-os) — run `ohho mcp` on the
  robot.

See [docs/mcp-split.md](../../docs/mcp-split.md) for the full classification table.

This package is intentionally not published from ohho-sdk.
