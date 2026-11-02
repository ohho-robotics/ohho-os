# ohho-booster

Separately installable OhhO robot plugin. It registers the `booster-t2`
humanoid spec (31 joints, generic mode map, pinned description assets).
Core OhhO OS does not import this package.

```bash
pip install -e plugins/ohho-booster
ohho list
```

Uninstalling the package removes `booster-t2` from `ohho list`.

The plugin does not import `booster_robotics_sdk_python`. There is no
`booster://` transport in this package.

Description files come from
[booster_assets](https://github.com/BoosterRobotics/booster_assets) at the
commit recorded in `ohho_booster/assets_manifest.json` (BSD-3-Clause; see
`NOTICE`). Fetch them with:

```bash
python plugins/ohho-booster/scripts/fetch_assets.py
```

The download is cached under `~/.ohho/assets/booster`. Meshes and the
locomotion checkpoint are not committed. The pin is a commit plus a sha256
for the description archive and for `t2_walk.pt` (booster_deploy, Apache-2.0).

`ohho sim --robot booster-t2` is headless MuJoCo: it prepares, enters walk
mode, and drives. The checkpoint's training command range (vx −0.7..3.0 m/s,
vy 0.8 m/s, yaw rate 2.0 rad/s) is not an Ohho cap. The spec stays at the
generic humanoid limits, and SafetyGate is unchanged. See
[docs/t2-walk.md](docs/t2-walk.md) for the observation mapping and why the
serial MJCF is used.

```bash
python plugins/ohho-booster/scripts/walk_check.py
```

That check stands 30 s, walks about 1 m at 0.3 m/s, turns, and stops.
