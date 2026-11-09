"""Print the PX4 SITL square recipe for mavlink+udp://127.0.0.1:14540.

This stub does not start PX4, Docker, or a MAVLink socket, and it does not
set OHHO_ARM_HARDWARE. The GitHub workflow sitl-mavlink.yml runs it with
--print-recipe. A later job can fly the square; D-22 (ArduPilot) waits on
the same URI.

Native MAVSDK 4.x only. There is no mavsdk_server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running `python scripts/sitl_mavlink_square.py` puts scripts/ on sys.path,
# not the repo root. Prefer this checkout over another installed ohho.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ohho.adapters.mavlink_drone import to_mavsdk_url  # noqa: E402

SITL_URI = "mavlink+udp://127.0.0.1:14540"


def recipe() -> str:
    """Steps for a 2.5 m square. Descriptive only."""
    url = to_mavsdk_url(SITL_URI)
    return "\n".join(
        [
            "PX4 SITL square via native MAVSDK 4 (no mavsdk_server).",
            f"Ohho URI: {SITL_URI}",
            f"MAVSDK 4 connection: {url}",
            "This process does not connect and does not start SITL.",
            "",
            "1. Start PX4 SITL yourself (headless gz_x500, image pinned, not latest).",
            "   Example image: px4io/px4-sitl-gazebo:v1.17.0. Not launched here.",
            "2. pip install -e '.[mavlink]'   # mavsdk>=4,<5 and pymavlink",
            "3. Connect Robot to mavlink+udp://127.0.0.1:14540 (udpin listen).",
            "4. Arm the safety gate. Loopback is a simulation target on the",
            "   transport; SafetyGate.requires_hardware_arming does not read",
            "   that flag yet. Do not set OHHO_ARM_HARDWARE in CI.",
            "5. Stream offboard, arm_flight, takeoff(2.5). Wait until NED down",
            "   is within 0.4 m of -2.5 for 2 s.",
            "6. Square, 4 m sides, LOCAL_NED, 0.5 m/s, altitude held.",
            "   Corners within 0.8 m horizontal.",
            "7. land. Pass when landed_state is on_ground, motors disarmed with",
            "   a normal disarm (param2 0, never 21196), and no failsafe flag",
            "   was set after takeoff.",
            "",
            "See docs/mavlink.md.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-recipe",
        action="store_true",
        help="print the square steps and exit 0 (the default)",
    )
    parser.add_argument(
        "--fly",
        action="store_true",
        help="refused: this stub does not fly",
    )
    args = parser.parse_args(argv)
    if args.fly:
        print(
            "Refusing to start PX4 SITL or open a MAVLink socket.",
            file=sys.stderr,
        )
        return 2
    url = to_mavsdk_url(SITL_URI)
    if url != "udpin://127.0.0.1:14540":
        print(f"unexpected MAVSDK URL {url}", file=sys.stderr)
        return 1
    print(recipe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
