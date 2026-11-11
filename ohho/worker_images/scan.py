"""Read a Grype JSON document and list fixed CRITICAL findings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ohho.worker_images.errors import WorkerImageError

EXCEPTION_SCHEMA = "ohho.worker_scan_exceptions.v1"


def critical_failures(
    document: Any,
    ignore: frozenset[str] = frozenset(),
    *,
    types: frozenset[str] | None = None,
) -> list[str]:
    """Lines for fixed CRITICAL findings that are not ignored.

    Unfixed findings are reported by the scanner and do not fail the gate.
    ``types`` limits the gate to those artifact types (for example Python
    packages). A document without a ``matches`` array raises
    ``WorkerImageError``.
    """
    matches = _matches(document)
    lines: list[str] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        vuln = match.get("vulnerability")
        if not isinstance(vuln, dict):
            continue
        severity = str(vuln.get("severity", "")).upper()
        if severity != "CRITICAL":
            continue
        vuln_id = str(vuln.get("id", "")).strip()
        if vuln_id and vuln_id in ignore:
            continue
        if not _fix_available(vuln.get("fix")):
            continue
        artifact = match.get("artifact")
        name = "unknown"
        version = "unknown"
        art_type = ""
        if isinstance(artifact, dict):
            name = str(artifact.get("name") or "unknown")
            version = str(artifact.get("version") or "unknown")
            art_type = str(artifact.get("type") or "").lower()
        if types is not None and art_type not in types:
            continue
        lines.append(f"{vuln_id or 'CRITICAL'}: {name}@{version}")
    return lines


def load_exceptions(path: str | Path) -> frozenset[str]:
    """Vulnerability ids the gate ignores. An empty list ignores nothing."""
    target = Path(path)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerImageError(f"Scan exceptions {target} are unreadable.") from exc
    if not isinstance(document, dict) or document.get("schema") != EXCEPTION_SCHEMA:
        raise WorkerImageError(f"Scan exceptions {target} are not {EXCEPTION_SCHEMA}.")
    ignore = document.get("ignore")
    if not isinstance(ignore, list) or not all(
        isinstance(item, str) for item in ignore
    ):
        raise WorkerImageError("Scan exceptions ignore must be a list of ids.")
    return frozenset(ignore)


def _matches(document: Any) -> list[Any]:
    if not isinstance(document, dict) or not isinstance(document.get("matches"), list):
        raise WorkerImageError("scan file has no matches array")
    return list(document["matches"])


def _fix_available(fix: Any) -> bool:
    if not isinstance(fix, dict):
        return False
    state = str(fix.get("state", "")).lower()
    versions = fix.get("versions") or []
    if state in {"not-fixed", "wont-fix", "unknown"} and not versions:
        return False
    return state == "fixed" or bool(versions)
