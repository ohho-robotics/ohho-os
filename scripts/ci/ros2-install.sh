#!/usr/bin/env bash
# Install ohho-os and this row's RMW package inside a ROS 2 container.
set -eo pipefail

: "${ROS_DISTRO:?ROS_DISTRO is required}"
: "${RMW_IMPLEMENTATION:?RMW_IMPLEMENTATION is required}"
: "${RMW_APT:?RMW_APT is required}"

script_dir=$(
  CDPATH=''
  cd -- "$(dirname -- "$0")" && pwd
)
root=$(
  CDPATH=''
  cd -- "${script_dir}/../.." && pwd
)
cd "${root}"

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends python3-pip python3-venv python3-yaml "${RMW_APT}"

# Humble ships pip 22, which cannot editable-install this pyproject.
# A virtualenv gets a current pip and leaves the Debian pip in place.
# --system-site-packages keeps apt modules rclpy imports (PyYAML).
python3 -m venv --system-site-packages /opt/ohho-ci
/opt/ohho-ci/bin/pip install -U pip
/opt/ohho-ci/bin/pip install -e "${root}"
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo /opt/ohho-ci/bin >> "${GITHUB_PATH}"
fi

echo "installed ohho-os on ROS ${ROS_DISTRO} with ${RMW_APT} (${RMW_IMPLEMENTATION})"
