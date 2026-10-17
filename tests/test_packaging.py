"""Packaging metadata invariants (PyPI project URLs after ohho-os rename)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class TestPackagingMetadata(unittest.TestCase):
    def test_project_urls_point_at_ohho_os_repo(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        self.assertEqual(project["name"], "ohho-os")
        urls = project["urls"]
        self.assertIn("Source", urls)
        self.assertIn("Issues", urls)
        self.assertIn("Documentation", urls)
        github_repo = re.compile(r"^https://github\.com/ohho-robotics/ohho-os(?:/|$)")
        self.assertRegex(urls["Source"], github_repo)
        self.assertRegex(urls["Issues"], github_repo)
        self.assertNotIn("ohho-sdk", urls["Source"])
        self.assertNotIn("ohho-sdk", urls["Issues"])


if __name__ == "__main__":
    unittest.main()
