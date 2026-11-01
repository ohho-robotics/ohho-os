"""Guardrails for docs/unitree.md (OHH-133 honest Go2 claims)."""

from __future__ import annotations

import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "docs" / "unitree.md"

_REQUIRED_PHRASES = (
    "Nothing in this repository has been validated on a physical Unitree Go2",
    "dds://lo?domain=1",
    "OHHO_ARM_HARDWARE=1",
    "TestHilUnitreeGo2",
    "OHHO_HIL=1",
    "OHH-135",
    "flip",
    "jump",
    "unitree_sdk2",
)


class TestUnitreeDoc(unittest.TestCase):
    def test_unitree_doc_exists_and_covers_claims(self):
        self.assertTrue(_DOC.is_file(), f"missing {_DOC}")
        text = _DOC.read_text(encoding="utf-8")
        for phrase in _REQUIRED_PHRASES:
            self.assertIn(phrase, text, msg=f"missing required phrase: {phrase}")


if __name__ == "__main__":
    unittest.main()
