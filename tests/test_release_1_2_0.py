"""OHH-281: 1.2.0 version, changelog, and wheel contents.

Does not publish and does not create a git tag. The wheel is built in a
temporary directory and discarded.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import ohho
from ohho.agent_rules import rules_source

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.2.0"
RULES_IN_WHEEL = "ohho/agent_rules/ohho-rules.md"
EXAMPLE_IN_WHEEL = "ohho/robots/example.json"


def _project_fields(text: str) -> dict[str, str]:
    """``key = value`` lines from the ``[project]`` table.

    Python 3.10 has no ``tomllib``, and ``tomli`` is not a base dependency.
    This only accepts the single-line assignments this repository uses.
    """
    fields: dict[str, str] = {}
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project or "=" not in stripped or stripped.startswith("#"):
            continue
        key, _, raw = stripped.partition("=")
        fields[key.strip()] = raw.strip()
    return fields


class Release120Tests(unittest.TestCase):
    def test_version_matches_pyproject(self) -> None:
        fields = _project_fields((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        quoted = fields["version"].strip("\"'")
        self.assertEqual(quoted, VERSION)
        self.assertEqual(ohho.__version__, VERSION)
        self.assertEqual(ohho.__version__, quoted)
        self.assertEqual(fields["dependencies"], "[]")

    def test_changelog_has_unreleased_1_2_0_section(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertRegex(text, re.compile(r"(?m)^## \[1\.2\.0\] - unreleased$"))
        self.assertIn("OHH-273", text)
        self.assertIn("OHH-278", text)

    def test_wheel_contains_agent_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "-w",
                    tmp,
                    str(ROOT),
                ],
                check=False,
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + "\n" + completed.stderr,
            )
            wheels = list(Path(tmp).glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            self.assertIn(VERSION, wheels[0].name)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                self.assertIn(RULES_IN_WHEEL, names)
                self.assertIn(EXAMPLE_IN_WHEEL, names)
                body = archive.read(RULES_IN_WHEEL).decode("utf-8")
                metadata = archive.read(
                    next(name for name in names if name.endswith(".dist-info/METADATA"))
                ).decode("utf-8")
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        metadata = metadata.replace("\r\n", "\n").replace("\r", "\n")
        self.assertEqual(normalized, rules_source())
        self.assertIn("sim://", body)
        self.assertRegex(metadata, re.compile(rf"(?m)^Version: {re.escape(VERSION)}$"))


if __name__ == "__main__":
    unittest.main()
