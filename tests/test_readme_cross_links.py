"""README cross-links for OHH-175 (engine vs client SDK)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

ENGINE = re.compile(r"https://github\.com/ohho-robotics/ohho-os")
CLIENT = re.compile(r"https://github\.com/ohho-robotics/ohho-sdk")
BAD_ENGINE_CI = re.compile(r"github\.com/ohho-robotics/ohho-sdk/actions/workflows/ci\.yml")


class TestReadmeCrossLinks(unittest.TestCase):
    def test_readme_links_engine_and_client_sdk(self) -> None:
        text = README.read_text(encoding="utf-8")
        self.assertRegex(text, ENGINE)
        self.assertRegex(text, CLIENT)
        self.assertNotRegex(text, BAD_ENGINE_CI)
        self.assertIn("License-Apache_2.0", text)


if __name__ == "__main__":
    unittest.main()
