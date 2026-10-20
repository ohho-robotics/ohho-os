"""Headless upper-body collection and eval.

Usage::

    python scripts/upper_body_vla.py collect ROBOT --output DIR
    python scripts/upper_body_vla.py eval ROBOT --checkpoint DIR --output report.json

See ``docs/humanoid-vla.md``.
"""

from __future__ import annotations

import sys

from ohho.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] != "vla":
        argv = ["vla", *argv]
    raise SystemExit(main(argv))
