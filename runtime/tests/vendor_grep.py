"""Vendor-name scans.

``scan_runtime`` covers ``runtime/src/**/*.py`` and ``runtime/pyproject.toml``.
``runtime/tests/`` is not scanned. Markdown is not scanned.

``scan_core`` covers ``ohho/**/*.py`` for vendor tokens. ``omnibot`` and
``yahboom`` may appear only in ``ohho/robots/omnibot/``. ``unitree`` and
``go2`` may appear only in ``ohho/robots/unitree/``. ``tests/`` and markdown
are not scanned: tests name the robot and import the moved adapter, and
public docs keep the product name.

``pyproject.toml`` may contain the ``ohho.robots`` entry-point assignments.
One further allowance: the optional-dependency extra ``unitree``
(``pip install 'ohho-os[unitree]'``). Renaming that extra would change the
published install contract. No other occurrence is allowed.
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

# Substring match so identifiers such as a profile constant count.
_OMNI_RE = re.compile(r"omnibot|yahboom", re.IGNORECASE)
_UNITREE_RE = re.compile(r"unitree|go2", re.IGNORECASE)
_OMNI_PARTS = ("ohho", "robots", "omnibot")
_UNITREE_PARTS = ("ohho", "robots", "unitree")
_OMNI_ENTRY = re.compile(r'^omnibot\s*=\s*"ohho\.robots\.omnibot:plugin"\s*(?:#.*)?$')
_UNITREE_ENTRY = re.compile(
    r'^unitree\s*=\s*"ohho\.robots\.unitree:plugin"\s*(?:#.*)?$'
)
# Published install extra. The key is the allowance; the comment may not add
# another vendor token beyond that line.
_UNITREE_EXTRA = re.compile(r'^unitree\s*=\s*\[\s*"cyclonedds>=0\.10"\s*\]\s*(?:#.*)?$')


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


def _in_package(path: Path, package: tuple[str, ...]) -> bool:
    parts = path.parts
    width = len(package)
    for index in range(len(parts) - width + 1):
        if parts[index : index + width] == package:
            return True
    return False


def scan_core(root: Path) -> list[str]:
    """Return vendor-name findings outside each plugin package.

    ``root`` is a repository root. ``tests/`` and markdown are not findings.
    ``omnibot`` / ``yahboom`` may appear in ``ohho/robots/omnibot/``.
    ``unitree`` / ``go2`` may appear in ``ohho/robots/unitree/``. The
    ``ohho.robots`` entry-point lines in ``pyproject.toml`` are allowed, as is
    the ``unitree`` optional-dependency extra.
    """
    findings: list[str] = []
    package = root / "ohho"
    if package.is_dir():
        for path in sorted(package.rglob("*.py")):
            findings.extend(_scan_core_file(path))
    project = root / "pyproject.toml"
    if project.is_file():
        findings.extend(_scan_project(project))
    return findings


def _scan_lines(path: Path, text: str, pattern: re.Pattern[str]) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = pattern.search(line)
        if match:
            findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
    return findings


def _scan_core_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []
    if not _in_package(path, _OMNI_PARTS):
        findings.extend(_scan_lines(path, text, _OMNI_RE))
    if not _in_package(path, _UNITREE_PARTS):
        findings.extend(_scan_lines(path, text, _UNITREE_RE))
    return findings


def _scan_project(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not _OMNI_ENTRY.match(stripped):
            match = _OMNI_RE.search(line)
            if match:
                findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
        if _UNITREE_ENTRY.match(stripped) or _UNITREE_EXTRA.match(stripped):
            continue
        match = _UNITREE_RE.search(line)
        if match:
            findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
    return findings
