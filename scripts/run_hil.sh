#!/usr/bin/env bash
# OHH-46 — hardware-in-the-loop runner for the OmniBot + Go2 bench.
# Run on the robot workstation only (requires physical USB / DDS link).
set -euo pipefail

export OHHO_HIL=1

: "${OHHO_OMNIBOT_PORT:=/dev/ttyUSB0}"
: "${OHHO_OMNIBOT_ARM:=/dev/ttyACM0}"
: "${OHHO_GO2_IFACE:=eth0}"

echo "OHHO_HIL=1"
echo "  OHHO_OMNIBOT_PORT=${OHHO_OMNIBOT_PORT}"
echo "  OHHO_OMNIBOT_ARM=${OHHO_OMNIBOT_ARM}"
echo "  OHHO_GO2_IFACE=${OHHO_GO2_IFACE}"
echo
echo "Install extras as needed: pip install -e '.[serial,arm,unitree]'"
echo

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
python -m unittest discover -s tests/hil -v
