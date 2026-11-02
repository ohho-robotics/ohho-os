"""Vendor-name scan for the onboard runtime tree.

Scans ``runtime/src/**/*.py`` and ``runtime/pyproject.toml``.
``runtime/tests/`` is not scanned. Markdown is not scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

# Same tokens the core vendor scan uses. Kept in tests so the pattern
# itself is not a hit under ``runtime/src``.
VENDOR_RE = re.compile(
    r"\b(?:booster|t2|unitree|go2|px4|crazyflie|yahboom|omnibot)\b",
    re.IGNORECASE,
)


def scan_runtime(root: Path) -> list[str]:
    """Return vendor-name findings under ``root/runtime``.

    ``root`` is a repository root (the directory that contains ``runtime/``).
    An empty list means the scanned files are clean.
    """
    runtime = root / "runtime"
    findings: list[str] = []
    src = runtime / "src"
    if src.is_dir():
        for path in sorted(src.rglob("*.py")):
            findings.extend(_scan_file(path))
    project = runtime / "pyproject.toml"
    if project.is_file():
        findings.extend(_scan_file(project))
    return findings


def _scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = VENDOR_RE.search(line)
        if match:
            findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
    return findings
