"""The protocol paperwork must stay unpublished and name the real pins."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseNotesTest(unittest.TestCase):
    def test_protocol_notes_are_not_a_published_release(self) -> None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        commands = (ROOT / "docs" / "releases" / "protocol-tag-commands.md").read_text(
            encoding="utf-8"
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        combined = "\n".join((changelog, commands, readme))
        for needle in (
            "not yet tagged",
            "not released yet",
            "protocol-v1.0.0",
            "PROTOCOL_VERSION",
            "1.0.0",
            "ohho.telemetry.v1",
            "RobotDescriptor",
            "6f690d50b5991b13a35430581f84626529ab870d",
            "5a20f1c23c10c814b8a790536a80b21695b3ed82",
            "git tag -a protocol-v1.0.0",
            "gh release create protocol-v1.0.0",
            "Do not run",
            "1.1.3",
        ):
            self.assertIn(needle, combined)
        folded = combined.lower()
        for needle in (
            "v1.0.0 is released",
            "release is published",
            "has been tagged",
        ):
            self.assertNotIn(needle, folded)
        script = (ROOT / "scripts" / "print-release-commands.sh").read_text(encoding="utf-8")
        for line in script.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("git tag"))
            self.assertFalse(stripped.startswith("git push"))
            self.assertFalse(stripped.startswith("gh release create"))


if __name__ == "__main__":
    unittest.main()
