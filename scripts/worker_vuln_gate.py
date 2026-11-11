#!/usr/bin/env python3
"""Fail when a worker-image Grype scan has a fixed CRITICAL finding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from ohho.worker_images.errors import WorkerImageError
    from ohho.worker_images.scan import critical_failures, load_exceptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan")
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="fail only on Python packages; the full scan is still uploaded",
    )
    parser.add_argument(
        "--exceptions",
        default=str(ROOT / "docker" / "serve-worker" / "scan-exceptions.json"),
    )
    args = parser.parse_args(argv)
    scan_path = Path(args.scan)
    try:
        document = json.loads(scan_path.read_text(encoding="utf-8"))
        ignore = load_exceptions(args.exceptions)
        types = frozenset({"python", "pypi"}) if args.python_only else None
        found = critical_failures(document, ignore, types=types)
    except (OSError, json.JSONDecodeError, WorkerImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if found:
        print("error: fixed CRITICAL vulnerabilities:", file=sys.stderr)
        for line in found:
            print(f"error: {line}", file=sys.stderr)
        return 1
    print(f"ok: {scan_path.name} has no fixed CRITICAL findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
