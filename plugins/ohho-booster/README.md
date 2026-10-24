# ohho-booster

Separately installable OhhO robot plugin. It registers two humanoid specs
and a generic mode map. Description assets are pinned, not vendored.
Core OhhO OS does not import this package.

| Robot | Joints | Status |
|---|---|---|
| `booster-t2` | 31 | sim only, not hardware-tested |
| `booster-k1` | 22 (head 2, arms 4+4, legs 6+6, no waist) | sim only, not hardware-tested |

```bash
pip install -e plugins/ohho-booster
ohho list
```

Uninstalling the package removes `booster-t2` and `booster-k1` from `ohho list`.

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
locomotion checkpoints are not committed. The pin is a commit plus a sha256
for the description archive and for `t2_walk.pt` and `k1_walk.pt`
(booster_deploy, Apache-2.0).

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
`booster-k1` uses the same script with `--robot booster-k1`. Its
checkpoint, observation layout and the reason the serial MJCF is loaded
are in [docs/k1-walk.md](docs/k1-walk.md). Isaac Lab's K1 task is not
wired here.

`ohho sim-serve` exposes either spec on the local WebSocket. The process
is the in-process MuJoCo sim (`--backend sim`), not the unimplemented
`mujoco` backend name. Commands are velocity plus the declared posture
modes (`prepare`, `walk`, `damping`, `custom`, `get_up`, `lie_down`).
The server sends pose, joint positions in descriptor order, and status
at about 30 Hz.

```bash
ohho sim-serve --backend sim --robot booster-t2
ohho sim-serve --backend sim --robot booster-k1
```

Default port is 8765. A browser page served over https can open
`ws://127.0.0.1` only when the browser allows loopback from a secure
page; a non-loopback `ws://` host is blocked. See `docs/sim-serve.md`.
