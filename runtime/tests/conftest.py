"""Make ``ohho`` and ``ohho_runtime`` importable without an editable install."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "runtime" / "src"
_TESTS = Path(__file__).resolve().parent
for _path in (str(_SRC), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
if str(_TESTS) not in sys.path:
    sys.path.append(str(_TESTS))

# Keep crash bundles from test crashes out of the default state directory.
_CRASH_TMP = tempfile.TemporaryDirectory(prefix="ohho-rt-crash-")
os.environ.setdefault("OHHO_RUNTIME_STATE_DIR", _CRASH_TMP.name)
