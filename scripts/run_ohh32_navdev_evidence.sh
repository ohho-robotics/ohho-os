#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
cd "$(dirname "$0")/.."
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-199}"
pip install --break-system-packages -e ".[dev]" -q
python3 examples/fake_ros2_robot_node.py &
FAKE=$!
sleep 2
echo "=== ros2:// ==="
python3 examples/demo_drive.py --transport ros2:// --seconds 2 --vx 0.2 --min-dx 0.15
echo "=== sim:// ==="
python3 examples/demo_drive.py --transport sim:// --seconds 2 --vx 0.2 --min-dx 0.15
kill "$FAKE" 2>/dev/null || true
wait "$FAKE" 2>/dev/null || true
python3 -m pytest tests/test_demo_drive.py -v
