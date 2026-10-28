#!/usr/bin/env python3
"""Run the OHH-35 HIL harness.

Dry run (no hardware)::

    python scripts/hil_run.py

Fake transports (CI)::

    python scripts/hil_run.py --mock

Hardware is refused unless OHHO_HIL=1, an explicit device variable,
``--preflight-ok``, and an interactive terminal are all present.
See docs/hil-preflight.md. This script does not open a port on its own.
"""

from __future__ import annotations

import sys

from ohho.hil import main

if __name__ == "__main__":
    sys.exit(main())
