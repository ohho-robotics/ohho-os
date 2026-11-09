"""Fail CI when dependency scans report CRITICAL vulnerabilities (Shield v0, OHH-63)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# NIST CVSS v3.0+ band: 9.0–10.0 is Critical.
CRITICAL_CVSS_THRESHOLD = 9.0


@dataclass(frozen=True)
class CriticalFinding:
    source: str
    vuln_id: str
    package: str
    version: str
    cvss: float | None

    def line(self) -> str:
        score = f" CVSS={self.cvss}" if self.cvss is not None else ""
        return f"{self.source}: {self.vuln_id} affects {self.package}@{self.version}{score}"


def _parse_cvss_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    # OSV may use "CVSS:3.1/AV:N/..." or a bare number.
    if text[0].isdigit() or text.startswith("."):
        try:
            return float(text.split()[0])
        except ValueError:
            return None
    return None


def max_cvss_from_osv_vuln(vuln: dict[str, Any]) -> float | None:
    best: float | None = None
    for item in vuln.get("severity") or []:
        if not isinstance(item, dict):
            continue
        score = _parse_cvss_score(item.get("score"))
        if score is not None:
            best = score if best is None else max(best, score)
    db = vuln.get("database_specific")
    if isinstance(db, dict):
        for key in ("cvss_score", "cvss", "severity_score"):
            score = _parse_cvss_score(db.get(key))
            if score is not None:
                best = score if best is None else max(best, score)
        sev = str(db.get("severity", "")).strip().upper()
        if sev == "CRITICAL":
            return CRITICAL_CVSS_THRESHOLD
    return best


def is_critical_osv_vuln(vuln: dict[str, Any]) -> bool:
    score = max_cvss_from_osv_vuln(vuln)
    if score is not None:
        return score >= CRITICAL_CVSS_THRESHOLD
    db = vuln.get("database_specific")
    if isinstance(db, dict) and str(db.get("severity", "")).upper() == "CRITICAL":
        return True
    return False


def critical_from_osv_scanner(
    data: dict[str, Any], *, source_label: str
) -> list[CriticalFinding]:
    findings: list[CriticalFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for result in data.get("results") or []:
        if not isinstance(result, dict):
            continue
        src_path = ""
        src = result.get("source")
        if isinstance(src, dict):
            src_path = str(src.get("path") or "")
        label = source_label if not src_path else f"{source_label}:{src_path}"
        for pkg_entry in result.get("packages") or []:
            if not isinstance(pkg_entry, dict):
                continue
            pkg = pkg_entry.get("package") or {}
            name = str(pkg.get("name") or "?")
            version = str(pkg.get("version") or "?")
            for vuln in pkg_entry.get("vulnerabilities") or []:
                if not isinstance(vuln, dict):
                    continue
                if not is_critical_osv_vuln(vuln):
                    continue
                vuln_id = str(vuln.get("id") or "unknown")
                key = (label, vuln_id, f"{name}@{version}")
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    CriticalFinding(
                        source=label,
                        vuln_id=vuln_id,
                        package=name,
                        version=version,
                        cvss=max_cvss_from_osv_vuln(vuln),
                    )
                )
    return findings


def critical_from_pip_audit(
    data: dict[str, Any], *, source_label: str
) -> list[CriticalFinding]:
    """pip-audit JSON rarely includes CVSS; only flag explicit CRITICAL severities."""
    findings: list[CriticalFinding] = []
    for dep in data.get("dependencies") or []:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name") or "?")
        version = str(dep.get("version") or "?")
        for vuln in dep.get("vulns") or []:
            if not isinstance(vuln, dict):
                continue
            sev = str(vuln.get("severity") or "").upper()
            if sev != "CRITICAL":
                continue
            vuln_id = str(vuln.get("id") or "unknown")
            findings.append(
                CriticalFinding(
                    source=source_label,
                    vuln_id=vuln_id,
                    package=name,
                    version=version,
                    cvss=None,
                )
            )
    return findings


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_allowlist(path: Path) -> set[str]:
    """Load allowed/ignored vulnerability IDs from a file.

    Supports:
    - TOML files (e.g. osv-scanner.toml) with [[IgnoredVulns]] blocks
    - Plain text files with one vulnerability ID per line (# comments supported)
    """
    if not path.is_file():
        return set()

    text = path.read_text(encoding="utf-8")
    ignored: set[str] = set()

    if path.suffix == ".toml" or "IgnoredVulns" in text:
        try:
            import tomllib

            data = tomllib.loads(text)
            for entry in data.get("IgnoredVulns", []):
                if isinstance(entry, dict) and "id" in entry:
                    ignored.add(str(entry["id"]).strip())
        except Exception:
            for match in re.finditer(r'id\s*=\s*["\']([^"\']+)["\']', text):
                ignored.add(match.group(1).strip())
        return ignored

    for line in text.splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if cleaned:
            ignored.add(cleaned)
    return ignored


def merge_findings(groups: Iterable[list[CriticalFinding]]) -> list[CriticalFinding]:
    merged: list[CriticalFinding] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group:
            key = (item.source, item.vuln_id, f"{item.package}@{item.version}")
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exit non-zero when scan JSON reports CRITICAL vulnerabilities.",
    )
    parser.add_argument(
        "--osv",
        action="append",
        default=[],
        metavar="PATH",
        help="osv-scanner JSON output (repeatable)",
    )
    parser.add_argument(
        "--pip-audit",
        action="append",
        default=[],
        metavar="PATH",
        help="pip-audit JSON output (repeatable)",
    )
    parser.add_argument(
        "--allowlist",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to osv-scanner.toml or text file of allowed vulnerability IDs (repeatable)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="ID",
        help="Specific vulnerability ID to ignore (repeatable)",
    )
    args = parser.parse_args(argv)

    all_findings: list[CriticalFinding] = []
    for path_str in args.osv:
        path = Path(path_str)
        if not path.is_file():
            print(f"missing osv scan file: {path}", file=sys.stderr)
            return 2
        all_findings.extend(
            critical_from_osv_scanner(load_json(path), source_label=path.name)
        )
    for path_str in args.pip_audit:
        path = Path(path_str)
        if not path.is_file():
            print(f"missing pip-audit file: {path}", file=sys.stderr)
            return 2
        all_findings.extend(
            critical_from_pip_audit(load_json(path), source_label=path.name)
        )

    allowed_ids: set[str] = set(args.ignore)
    for path_str in args.allowlist:
        p = Path(path_str)
        if not p.is_file():
            print(f"missing allowlist file: {p}", file=sys.stderr)
            return 2
        allowed_ids.update(load_allowlist(p))

    if not args.allowlist:
        default_toml = Path("osv-scanner.toml")
        if default_toml.is_file():
            allowed_ids.update(load_allowlist(default_toml))

    critical = merge_findings([all_findings])
    unignored: list[CriticalFinding] = []
    for item in critical:
        if item.vuln_id in allowed_ids:
            print(f"vuln_gate: ignoring documented vulnerability: {item.line()}")
        else:
            unignored.append(item)

    if not unignored:
        print("vuln_gate: no CRITICAL vulnerabilities in scan inputs")
        return 0

    print(
        "vuln_gate: CRITICAL vulnerabilities must be fixed or documented:",
        file=sys.stderr,
    )
    for item in unignored:
        print(f"  {item.line()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
