"""Headless stand, walk, turn and stop for booster-t2.

Uses the same sim backend as ``ohho sim --robot booster-t2`` and steps the
model directly, so the check is simulated time rather than a live thread.

    python plugins/ohho-booster/scripts/walk_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ohho.adapters import resolve_transport
    from ohho_booster.sim import bind_walk_config
    from ohho_booster.spec import booster_t2_spec
    from ohho_booster.walk import run_walk_check

    bind_walk_config()
    spec = booster_t2_spec()
    transport = resolve_transport("sim://", spec)
    transport.connect()
    try:
        result = run_walk_check(transport)
    finally:
        transport.disconnect()
    print(
        f"min_height_m={result.min_height_m:.3f} "
        f"stand_height_m={result.stand_height_m:.3f} "
        f"walked_m={result.walked_m:.3f} "
        f"yaw_change_rad={result.yaw_change_rad:.3f} "
        f"stop_speed_m_s={result.stop_speed_m_s:.3f} "
        f"stayed_up={result.stayed_up}"
    )
    ok = (
        result.stayed_up
        and result.min_height_m > 0.9
        and result.stand_height_m > 0.9
        and 0.9 <= result.walked_m <= 1.2
        and abs(result.yaw_change_rad) > 0.45
        and result.stop_speed_m_s < 0.05
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
