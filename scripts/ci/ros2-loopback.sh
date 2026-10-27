#!/usr/bin/env bash
# Loopback DDS for this ROS distro. Source this file after setup.bash.
# Humble reads ROS_LOCALHOST_ONLY. Jazzy warns on that variable and reads
# ROS_AUTOMATIC_DISCOVERY_RANGE instead.
if [[ "${ROS_DISTRO:-}" == "humble" ]]; then
  export ROS_LOCALHOST_ONLY=1
  unset ROS_AUTOMATIC_DISCOVERY_RANGE || true
else
  unset ROS_LOCALHOST_ONLY || true
  # Force the value. setup.bash may already have set SUBNET.
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
fi
