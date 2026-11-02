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

The download is cached under `~/.ohho/assets/booster`. Meshes are not
committed.
