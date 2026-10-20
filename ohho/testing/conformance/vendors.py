"""Vendor-name grep and the import-time SDK sentinel (OHH-315).

Core source is ``ohho/**/*.py`` plus ``pyproject.toml``. Plugin packages
(``ohho/robots/`` and ``plugins/``) may name a vendor. Docs are not scanned.
A path under ``tests/`` is scanned only when it is listed in
:data:`FIXTURE_WHITELIST`. This module is not scanned: it is the denylist.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# Token -> issue that moves that name out of core. ``booster`` and ``t2`` are
# enforced immediately and cannot appear in the allowlist.
TOKEN_ISSUE: dict[str, str] = {
    "omnibot": "R-01",
    "yahboom": "R-01",
    "feetech": "R-01",
    "unitree": "R-02",
    "go2": "R-02",
    "px4": "R-03",
    "ardupilot": "R-03",
    "crazyflie": "R-03",
}
UNALLOWABLE = frozenset({"booster", "t2"})
TOKENS = tuple(sorted(set(TOKEN_ISSUE) | UNALLOWABLE))

# Import roots a plugin must not load while its module is imported.
VENDOR_SDK_ROOTS = frozenset(
    {
        "serial",
        "lerobot",
        "cyclonedds",
        "unitree_sdk2py",
        "rclpy",
        "pymavlink",
        "booster_robotics_sdk",
        "booster_robotics_sdk_python",
    }
)

EXCLUDED_PREFIXES = ("ohho/robots/", "plugins/")
EXCLUDED_FILES = ("ohho/testing/conformance/vendors.py",)
# Empty until a fixture is named here. Unlisted tests and docs are not scanned.
FIXTURE_WHITELIST: tuple[str, ...] = ()

_ALLOWLIST_PATH = Path(__file__).with_name("vendor_allowlist.txt")
_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])("
    + "|".join(re.escape(token) for token in TOKENS)
    + r")(?![A-Za-z0-9_])"
)


@dataclass(frozen=True)
class VendorHit:
    """One vendor token on one source line."""

    path: str
    line: int
    token: str

    def format(self) -> str:
        """``file:line: token`` for a failing grep."""
        return f"{self.path}:{self.line}: {self.token}"


def repo_root() -> Path:
    """Repository root (the directory that contains ``ohho/`` and ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def _excluded(rel: str) -> bool:
    norm = rel.replace("\\", "/")
    if norm in EXCLUDED_FILES:
        return True
    return any(norm.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def iter_source_files(root: Path | None = None) -> list[tuple[Path, str]]:
    """Core files the grep reads, in stable order."""
    root = repo_root() if root is None else root
    found: list[tuple[Path, str]] = []
    ohho = root / "ohho"
    if ohho.is_dir():
        for path in sorted(ohho.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if _excluded(rel):
                continue
            found.append((path, rel))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        found.append((pyproject, "pyproject.toml"))
    for rel in FIXTURE_WHITELIST:
        path = root / rel
        norm = rel.replace("\\", "/")
        if path.is_file() and not _excluded(norm):
            found.append((path, norm))
    return found


def scan_text(path: str, text: str) -> list[VendorHit]:
    """Return every vendor token in ``text``, with 1-based line numbers."""
    hits: list[VendorHit] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _PATTERN.finditer(line):
            hits.append(VendorHit(path, lineno, match.group(1).casefold()))
    return hits


def scan_paths(paths: Iterable[Path], *, root: Path | None = None) -> list[VendorHit]:
    """Scan explicit files. Paths outside ``root`` stay absolute in the report."""
    root = repo_root() if root is None else root
    hits: list[VendorHit] = []
    for path in paths:
        rel = _relative(root, path)
        if _excluded(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(scan_text(rel, text))
    return hits


def scan_repository(root: Path | None = None) -> list[VendorHit]:
    """Scan core source. Plugin packages, docs, and unlisted tests are omitted."""
    root = repo_root() if root is None else root
    hits: list[VendorHit] = []
    for path, rel in iter_source_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(scan_text(rel, text))
    return hits


def load_allowlist(path: Path | None = None) -> dict[tuple[str, str], str]:
    """Load ``path token issue`` rows. A bad row or a forbidden token raises.

    Each issue is the one that removes that token (R-01 OmniBot, R-02 Go2,
    R-03 drones). ``booster`` and ``t2`` cannot be listed.
    """
    path = _ALLOWLIST_PATH if path is None else path
    table: dict[tuple[str, str], str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{lineno}: expected path, token, issue")
        rel, token, issue = (part.strip() for part in parts)
        token = token.casefold()
        rel = rel.replace("\\", "/")
        if token in UNALLOWABLE:
            raise ValueError(f"{path}:{lineno}: {token} cannot be allowlisted")
        expected = TOKEN_ISSUE.get(token)
        if expected is None:
            raise ValueError(f"{path}:{lineno}: unknown token {token!r}")
        if issue != expected:
            raise ValueError(
                f"{path}:{lineno}: {token} is removed by {expected}, not {issue}"
            )
        table[(rel, token)] = issue
    return table


def violations(
    hits: Sequence[VendorHit], allowlist: dict[tuple[str, str], str]
) -> list[VendorHit]:
    """Hits whose ``(path, token)`` is not on the allowlist."""
    return [hit for hit in hits if (hit.path, hit.token) not in allowlist]


def format_violation(hit: VendorHit) -> str:
    """``file:line: token``."""
    return hit.format()


class _SentinelLoader(importlib.abc.Loader):
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        raise ImportError(f"vendor SDK imported at plugin load: {self.fullname}")


class _SentinelFinder:
    def __init__(self, hits: list[str]) -> None:
        self.hits = hits

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        root = fullname.split(".", 1)[0]
        if root not in VENDOR_SDK_ROOTS:
            return None
        self.hits.append(fullname)
        return importlib.machinery.ModuleSpec(fullname, _SentinelLoader(fullname))


def imported_vendor_sdks(load: Callable[[], None]) -> list[str]:
    """Import roots that ``load`` touches, using a sentinel for vendor SDKs.

    Already-imported vendor modules are hidden for the call, so a plugin that
    imports one is reported even when an earlier test imported it first.
    """
    hits: list[str] = []
    saved: dict[str, ModuleType] = {}
    for key in list(sys.modules):
        if key.split(".", 1)[0] in VENDOR_SDK_ROOTS:
            module = sys.modules.pop(key)
            if isinstance(module, ModuleType):
                saved[key] = module
    finder = _SentinelFinder(hits)
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        load()
    except ImportError as exc:
        if "vendor SDK imported at plugin load" not in str(exc) and not hits:
            raise
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for key in list(sys.modules):
            if key not in before and key.split(".", 1)[0] in VENDOR_SDK_ROOTS:
                sys.modules.pop(key, None)
        sys.modules.update(saved)
    roots = sorted({name.split(".", 1)[0] for name in hits})
    return roots
