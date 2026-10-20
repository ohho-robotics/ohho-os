# ohho-booster

Separately installable OhhO robot plugin. It registers the `booster-t2`
humanoid spec (31 joints, generic mode map, pinned description assets).
Core OhhO OS does not import this package.

```bash
pip install -e plugins/ohho-booster
ohho list
```

Uninstalling the package removes `booster-t2` from `ohho list`.

The plugin does not import `booster_robotics_sdk_python` at import time.
`booster://<ip>` is the optional extra `booster_robotics_sdk_python==1.6.3`
(Linux x86_64/aarch64). That SDK is imported only when the transport factory
runs. `booster+ros2://` is registered and refuses to connect: the ROS 2
distro on the robot is an open question, so that path is not implemented.

`ohho connect --robot booster-t2 --read-only booster://<ip>` subscribes to
mode, fall state, joints and battery and publishes nothing. Read-only is
enforced in the session: state-changing RPCs and joint commands are refused
before they are sent. `ohho doctor --robot booster-t2` runs this plugin's
doctor hook (SDK installed, robot reachable, model, firmware 1.7.2 or later,
network interface) through the shared robot-id resolver. Those checks are
mock-tested and do not command the robot.

```bash
pip install -e "plugins/ohho-booster[booster]"
```

Description files come from
[booster_assets](https://github.com/BoosterRobotics/booster_assets) at the
commit recorded in `ohho_booster/assets_manifest.json` (BSD-3-Clause; see
`NOTICE`). Fetch them with:

```bash
python plugins/ohho-booster/scripts/fetch_assets.py
```

The download is cached under `~/.ohho/assets/booster`. Meshes are not
committed.
