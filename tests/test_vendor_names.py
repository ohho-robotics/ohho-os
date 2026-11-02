"""Mobile-manipulator names stay inside the built-in plugin package.

``tests/`` and markdown are not scanned. The ``ohho.robots`` entry-point
line in ``pyproject.toml`` is the one allowed occurrence outside
``ohho/robots/omnibot/``.
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
    def test_omnibot_and_yahboom_stay_in_the_plugin(self) -> None:
        findings = _scan_core()(_ROOT)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
