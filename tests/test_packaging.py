"""Packaging metadata invariants (PyPI project URLs after ohho-os rename)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

# Python 3.10 has no tomllib. tomli is not a base dependency, and the L4T
# r36.2.0 job does not install it. Skip only when neither parser imports.
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


@unittest.skipUnless(tomllib is not None, "neither tomllib nor tomli is available")
class TestPackagingMetadata(unittest.TestCase):
    def test_project_urls_point_at_ohho_os_repo(self) -> None:
        assert tomllib is not None
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
