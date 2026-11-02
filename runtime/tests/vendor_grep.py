"""Vendor-name scans.

``scan_runtime`` covers ``runtime/src/**/*.py`` and ``runtime/pyproject.toml``.
``runtime/tests/`` is not scanned. Markdown is not scanned.

``scan_core`` covers ``ohho/**/*.py`` for the mobile-manipulator tokens
``omnibot`` and ``yahboom``. Those names may appear only in
``ohho/robots/omnibot/``. ``tests/`` and markdown are not scanned: tests name
the robot and import the moved adapter, and public docs keep the product
name. ``pyproject.toml`` may contain the ``ohho.robots`` entry-point
assignment and no other occurrence.
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
CORE_VENDOR_RE = re.compile(r"omnibot|yahboom", re.IGNORECASE)
_PLUGIN_PARTS = ("ohho", "robots", "omnibot")
_ENTRY_ALLOW = re.compile(r'^omnibot\s*=\s*"ohho\.robots\.omnibot:plugin"\s*(?:#.*)?$')


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


def _in_plugin(path: Path) -> bool:
    parts = path.parts
    for index in range(len(parts) - len(_PLUGIN_PARTS) + 1):
        if parts[index : index + len(_PLUGIN_PARTS)] == _PLUGIN_PARTS:
            return True
    return False


def scan_core(root: Path) -> list[str]:
    """Return ``omnibot`` / ``yahboom`` findings outside the plugin package.

    ``root`` is a repository root. ``tests/``, markdown, and
    ``ohho/robots/omnibot/`` are not findings. The ``ohho.robots`` entry-point
    line in ``pyproject.toml`` is the one allowed occurrence outside the
    package.
    """
    findings: list[str] = []
    package = root / "ohho"
    if package.is_dir():
        for path in sorted(package.rglob("*.py")):
            if _in_plugin(path):
                continue
            findings.extend(_scan_core_file(path))
    project = root / "pyproject.toml"
    if project.is_file():
        findings.extend(_scan_project(project))
    return findings


def _scan_core_file(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = CORE_VENDOR_RE.search(line)
        if match:
            findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
    return findings


def _scan_project(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _ENTRY_ALLOW.match(line.strip()):
            continue
        match = CORE_VENDOR_RE.search(line)
        if match:
            findings.append(f"{path}:{lineno}: vendor name {match.group(0)!r}")
    return findings
