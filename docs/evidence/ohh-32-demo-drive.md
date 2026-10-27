# OHH-32 evidence: one script, sim:// vs ros2://

## Claim

The same `examples/demo_drive.py`, changing only `--transport`, drives the in-process
sim adapter and a ROS 2 robot bridged on `/cmd_vel` and `/odom`.

## Code

- Demo entrypoint: `examples/demo_drive.py` → `ohho/demo_drive.py`
- Fake integrator (CI + local headless): `examples/fake_ros2_robot_node.py`
- CI workflow: `.github/workflows/demo-drive-ros2.yml`

## Local headless (navdev, ros:jazzy)

Recorded on the `navdev` container on the developer machine (no Gazebo required for
the integrator test). Commands:

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/ohho-os
pip install -e .
python3 examples/fake_ros2_robot_node.py &
sleep 2
python3 examples/demo_drive.py --transport ros2:// --seconds 2 --vx 0.2 --min-dx 0.15
python3 examples/demo_drive.py --transport sim:// --seconds 2 --vx 0.2 --min-dx 0.15
```

Log output is saved beside this file as `ohh-32-navdev.log`.

## Gazebo / OmniBot apartment world

A headless GIF of Gazebo was **not** produced in this pass (requires the OmniBot
Gazebo launch from OHH-30 and a display or offscreen render pipeline). The
acceptance path for CI uses the fake robot node instead of Gazebo, per the issue
tasks.

## CI

GitHub Actions job `demo-drive-fake-robot` in `.github/workflows/demo-drive-ros2.yml`
runs the same commands as above inside `ros:jazzy` and asserts `--min-dx`.
