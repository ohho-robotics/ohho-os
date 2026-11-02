"""Vendor-name grep for ``runtime/``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vendor_grep import scan_runtime

_ROOT = Path(__file__).resolve().parents[2]


class VendorGrepTest(unittest.TestCase):
    def test_runtime_tree_has_no_vendor_names(self) -> None:
        self.assertEqual(scan_runtime(_ROOT), [])

    def test_planted_vendor_string_under_runtime_src_fails(self) -> None:
        token = "Uni" + "tree"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planted = root / "runtime" / "src" / "ohho_runtime" / "planted.py"
            planted.parent.mkdir(parents=True)
            planted.write_text(f"name = {token!r}\n", encoding="utf-8")
            findings = scan_runtime(root)
        self.assertTrue(findings)
        self.assertIn("planted.py", findings[0])
        self.assertIn("vendor name", findings[0])
        self.assertIn(token, findings[0])

    def test_planted_vendor_string_in_runtime_project_fails(self) -> None:
        token = "ya" + "hboom"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "runtime" / "pyproject.toml"
            project.parent.mkdir(parents=True)
            project.write_text(f"description = {token!r}\n", encoding="utf-8")
            findings = scan_runtime(root)
        self.assertTrue(findings)
        self.assertIn("pyproject.toml", findings[0])
        self.assertIn(token, findings[0])

    def test_markdown_and_tests_are_not_scanned(self) -> None:
        token = "px" + "4"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "runtime" / "src" / "ohho_runtime"
            src.mkdir(parents=True)
            (src / "clean.py").write_text("value = 1\n", encoding="utf-8")
            (root / "runtime" / "pyproject.toml").write_text(
                "[project]\nname = 'ohho-runtime'\n",
                encoding="utf-8",
            )
            (root / "runtime" / "README.md").write_text(token + "\n", encoding="utf-8")
            tests = root / "runtime" / "tests"
            tests.mkdir()
            (tests / "planted.py").write_text(token + "\n", encoding="utf-8")
            self.assertEqual(scan_runtime(root), [])


if __name__ == "__main__":
    unittest.main()
