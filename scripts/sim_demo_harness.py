#!/usr/bin/env python3
"""Run the OHH-134 Sim Demo Harness (typed goal -> agent -> Go2 in MuJoCo).

Executes the 6-shot demo sequence (doctor -> sim/shim start -> typed goal ->
tool-call log -> SafetyGate clamp/e-stop demo -> end card) and records an
eval JSON log.

SIMULATION ONLY — NOT TESTED ON PHYSICAL ROBOT HARDWARE.

Usage:
    python scripts/sim_demo_harness.py --check-overlays
    python scripts/sim_demo_harness.py --sim --fast
    python scripts/sim_demo_harness.py --out evals/examples/ohh-134-go2-mujoco-demo.json
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ohho.sim_demo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
