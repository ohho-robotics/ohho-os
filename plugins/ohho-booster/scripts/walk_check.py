"""Headless stand, walk, turn and stop for a Booster humanoid.

Uses the same sim backend as ``ohho sim --robot <id>`` and steps the model
directly, so the check is simulated time rather than a live thread.

    python plugins/ohho-booster/scripts/walk_check.py
    python plugins/ohho-booster/scripts/walk_check.py --robot booster-k1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    plugin = Path(__file__).resolve().parents[1]
    checkout = plugin.parents[1]
    # The script directory is sys.path[0]. Put this checkout and the plugin
    # ahead of an older installed copy so the check runs the tree it lives in.
    for entry in (str(checkout), str(plugin)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="booster-t2",
        choices=("booster-t2", "booster-k1"),
        help="spec to simulate (default: booster-t2)",
    )
    args = parser.parse_args(argv)
    from ohho.adapters import resolve_transport
    from ohho_booster.sim import bind_walk_config
    from ohho_booster.spec import booster_k1_spec, booster_t2_spec
    from ohho_booster.walk import run_walk_check, walk_limits, within_limits

    bind_walk_config(args.robot)
    spec = booster_k1_spec() if args.robot == "booster-k1" else booster_t2_spec()
    transport = resolve_transport("sim://", spec)
    transport.connect()
    try:
        result = run_walk_check(transport)
    finally:
        transport.disconnect()
    print(
        f"robot={args.robot} "
        f"min_height_m={result.min_height_m:.3f} "
        f"stand_height_m={result.stand_height_m:.3f} "
        f"walked_m={result.walked_m:.3f} "
        f"yaw_change_rad={result.yaw_change_rad:.3f} "
        f"stop_speed_m_s={result.stop_speed_m_s:.3f} "
        f"stayed_up={result.stayed_up}"
    )
    return 0 if within_limits(result, walk_limits(args.robot)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
