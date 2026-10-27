"""Committed JSON schemas stay in sync with ohho.protocol and ohho.schema."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestJsonSchemas(unittest.TestCase):
    def test_manifest_protocol_matches_module(self) -> None:
        from ohho.protocol import PROTOCOL_VERSION

        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "schemas" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocolVersion"], PROTOCOL_VERSION)

    def test_export_script_check_passes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "export_json_schemas.py"), "--check"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=proc.stdout + proc.stderr,
        )


if __name__ == "__main__":
    unittest.main()
