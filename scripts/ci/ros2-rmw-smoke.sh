#!/usr/bin/env bash
# Source this row's ROS distro and load the pinned RMW.
# RT-08 QoS catalogue tests belong in the workflow step after this smoke.
# They need the sourced distro and this row's RMW_IMPLEMENTATION.
set -eo pipefail

: "${ROS_DISTRO:?ROS_DISTRO is required}"
: "${RMW_IMPLEMENTATION:?RMW_IMPLEMENTATION is required}"

pinned="${RMW_IMPLEMENTATION}"

script_dir=$(
  CDPATH=''
  cd -- "$(dirname -- "$0")" && pwd
)

# setup.bash reads variables that are unset in a minimal container, and a
# sourced helper can return non-zero without meaning the environment failed.
set +e
set +u
# shellcheck disable=SC1090,SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source_status=$?
set -e
if [[ "${source_status}" -ne 0 ]]; then
  echo "failed to source /opt/ros/${ROS_DISTRO}/setup.bash (status ${source_status})" >&2
  exit "${source_status}"
fi
set -u

export RMW_IMPLEMENTATION="${pinned}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-199}"
# shellcheck disable=SC1090,SC1091
source "${script_dir}/ros2-venv.sh"
# shellcheck disable=SC1090,SC1091
source "${script_dir}/ros2-loopback.sh"

mapfile -t libs < <(find "/opt/ros/${ROS_DISTRO}" -name "lib${pinned}.so" -print)
if [[ ${#libs[@]} -eq 0 ]]; then
  echo "pinned RMW library lib${pinned}.so is not under /opt/ros/${ROS_DISTRO}" >&2
  exit 1
fi
printf 'pinned RMW library: %s\n' "${libs[@]}"

# Nounset stays a shell option. Python only sees the exported environment.
set +u
timeout 60 python3 - <<'PY'
import os
import platform

import ohho
import rclpy
from ohho.runtime.ros2 import Ros2Runtime

expected = os.environ["RMW_IMPLEMENTATION"]
if not Ros2Runtime.is_available():
    raise SystemExit("Ros2Runtime.is_available() is false after sourcing ROS")
print(
    "ohho",
    ohho.__version__,
    "python",
    platform.python_version(),
    "rmw",
    expected,
    "discovery",
    os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE"),
    "localhost_only",
    os.environ.get("ROS_LOCALHOST_ONLY"),
)
rclpy.init()
try:
    node = rclpy.create_node("ohho_ci_rmw_pin")
    node.destroy_node()
finally:
    if rclpy.ok():
        rclpy.shutdown()
print("created an rclpy node with", expected)
PY
