# 2-minute MCAP bag (OHH-35)

Record this on the Pi 5 after `docs/hil-preflight.md`. `ohho hil` does not
record a bag. Put the bag path in `OHHO_HIL_BAG` or in the Notes section of
`docs/hil-results.md`.

## Topics

This package's ROS 2 transport uses:

- `/cmd_vel` (`geometry_msgs/Twist`)
- `/odom` (`nav_msgs/Odometry`)
- `/arm/joint_states` (`sensor_msgs/JointState`)
- `/arm/joint_commands`

Lidar and cameras are published by the robot image. Confirm the names before
recording:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
ros2 topic hz /scan
ros2 topic hz /camera/image_raw
```

Use the camera topic that list actually shows. Common names are
`/camera/image_raw` and `/camera/color/image_raw`. Drop a topic that has no
publisher so the bag info is easy to read.

## Record for 120 seconds

```bash
source /opt/ros/jazzy/setup.bash
sudo apt-get install -y ros-jazzy-rosbag2-storage-mcap
mkdir -p "$HOME/ohho-hil-bags"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BAG="$HOME/ohho-hil-bags/hil-$STAMP"
# SIGINT at 120s lets rosbag2 finish the mcap file.
timeout --signal=INT 120 ros2 bag record -s mcap -o "$BAG" \
  /cmd_vel \
  /odom \
  /arm/joint_states \
  /arm/joint_commands \
  /scan \
  /camera/image_raw
echo "$BAG"
```

`timeout` sends SIGINT so rosbag2 can write the mcap footer. If `timeout` is
missing, start the same `ros2 bag record` and press Ctrl-C at two minutes.
Ctrl-C is SIGINT as well.

If `ros2 bag info` says the file is unindexed, record again and stop with
Ctrl-C. Do not copy a partial file into the results as a finished bag.

## Check

```bash
ros2 bag info "$BAG"
```

You want:

- Storage identifier `mcap`
- Duration about 120 seconds
- Messages on `/odom` and on the scan topic if those sensors were up
- The camera topic present when you included one that was publishing

## Write the path down

```bash
OHHO_HIL=1 \
OHHO_OMNIBOT_PORT=/dev/ttyUSB0 \
OHHO_OMNIBOT_ARM=/dev/ttyACM0 \
OHHO_GO2_IFACE=eth0 \
OHHO_HIL_BAG="$BAG" \
ohho hil --hardware --preflight-ok
```

The bag stays on the Pi. This repository does not upload it. The results
file only stores the path.
