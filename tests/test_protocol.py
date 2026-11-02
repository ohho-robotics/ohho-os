from __future__ import annotations

import unittest

from ohho.protocol import PROTOCOL_VERSION


class TestProtocolVersion(unittest.TestCase):
    def test_protocol_version_is_semver_triplet(self) -> None:
        parts = PROTOCOL_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertTrue(part.isdigit())


if __name__ == "__main__":
    unittest.main()
