#!/usr/bin/env bash
# Print the protocol tag commands. Does not create tags or releases.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
echo "OHH-249 print-only. No tag is created. No GitHub Release is created."
cat "$root/docs/releases/protocol-tag-commands.md"
