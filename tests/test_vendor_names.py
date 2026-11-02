"""Vendor names stay inside their built-in plugin packages.

``omnibot`` / ``yahboom`` stay in ``ohho/robots/omnibot/``. ``unitree`` /
``go2`` stay in ``ohho/robots/unitree/``. ``tests/`` and markdown are not
scanned. ``pyproject.toml`` may name the ``ohho.robots`` entry points, and
the published ``unitree`` install extra.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCANNER = _ROOT / "runtime" / "tests" / "vendor_grep.py"


def _scan_core():
    spec = importlib.util.spec_from_file_location("vendor_grep", _SCANNER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {_SCANNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scan_core


class VendorNameTest(unittest.TestCase):
    def test_vendor_names_stay_in_their_plugins(self) -> None:
        findings = _scan_core()(_ROOT)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
