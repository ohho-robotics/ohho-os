# T2 walk in MuJoCo

`ohho sim --robot booster-t2` uses the generic `humanoid-mujoco` backend.
This package supplies the MJCF, PD gains, observation layout and the
`t2_walk.pt` checkpoint. Core code does not name this robot.

The checkpoint and the numbers below are from
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy) commit
`7bb1462e2ac742eba5c01659b815e2e99fbe617f` (Apache-2.0):

- `tasks/locomotion/robots/t2/__init__.py` (`T2WalkControllerCfg`)
- `tasks/locomotion/locomotion.py` (`T2LocomotionPolicy`, `T2LocomotionPolicyCfg`)
- `booster_deploy/robots/t2.py` (`T2_31DOF_CFG.mjcf_path`)
- `tasks/locomotion/robots/t2/models/t2_walk.pt`

The description archive is booster_assets commit
`12a97516f57469078707afb587a5f84c1cb110d1` (BSD-3-Clause). Both pins, with
sha256, are in `ohho_booster/assets_manifest.json`. Nothing is vendored.

## Which MJCF

The serial model `robots/T2/T2_31dof/T2_31dof.xml` is the one this config
loads. At that commit it has 32 joints (the free joint plus 31 hinges) and
31 motors, one per hinge, in `T2_JOINT_NAMES` order.

`T2_31dof_parallel.xml` is not used. It has 44 joints and 31 motors: the
extra joints are the closed-loop ankle and waist linkages. The checkpoint
observes 23 of the serial hinges and never sees those constraint joints.
`T2_31DOF_CFG.mjcf_path` in booster_deploy points at the serial file, which
is the model the checkpoint was deployed against. Loading the parallel file
under the same action vector would change the kinematics and the contacts.

## Command ranges are not Ohho caps

The walk task's `VelocityCommandCfg` is the training range:

| Command | Training range |
|---|---|
| forward `vx` | 0 to 3.0 m/s |
| backward `vx` | 0 to 0.7 m/s |
| `vy` | ±0.8 m/s |
| yaw rate | ±2.0 rad/s |

Those numbers are what the checkpoint was trained to track. They are not
OhhO limits. The spec keeps the generic humanoid description (0.5 m/s,
1.0 rad/s). `Robot.drive` clamps to that spec. On a link that is not
simulated, SafetyGate clamps again. This package does not raise either cap.
H-05 (the humanoid safety policy) is a separate change and is not
implemented here.

The observation stores the command in m/s and rad/s, the same values
`drive` sent after that clamp. It does not divide by the training range.
Booster's joystick helper can map a normalized stick to that range before
the command exists; the network does not see the normalized stick.

## Control timing

| | |
|---|---|
| Policy period | 0.02 s (50 Hz) |
| Physics step | 0.002 s (`policy_dt / 10`) |
| Policy input | 10 frames × 80 values = 800 |
| Policy output | 23 joint-position offsets |

The MJCF file itself sets a 0.001 s step. The deployment controller replaces
that with 0.002 s, and so does this config, so the PD loop matches the
decimation the checkpoint was run with. PD still runs on every physics step:

```text
tau = kp (q_target - q) - kd dq
```

`kp` and `kd` are `T2WalkControllerCfg.joint_stiffness` and `joint_damping`
(all 31 joints, `T2_JOINT_NAMES` order). The pose held before the first
policy update, and the pose added to each action, is that task's
`default_joint_pos`. The floating base starts at `(0, 0, 1.05)`.

Head and wrist joints are not in the action vector. Their targets stay at
the default pose. Motors in the MJCF already limit torque to the effort
range; the runner writes torque to those motors.

## Observation, one frame

Each frame is clipped to ±100, then stacked. The first call repeats the
frame 10 times (`history_init = "repeat"`). Later calls drop the oldest
frame. The network also has its own normalizer; this package does not
normalize again.

| Block | Values | Definition |
|---|---|---|
| `base_ang_vel` | 3 | Base angular velocity in the body frame, rad/s. Scale 1. |
| `projected_gravity` | 3 | Gravity `(0, 0, -1)` expressed in the body frame. Scale 1. |
| `command` | 3 | `(vx, vy, yaw rate)` in m/s and rad/s. Scale 1. |
| `q` | 23 | Joint position minus `default_q`, policy joint order. Scale 1. |
| `dq` | 23 | Joint velocity, policy joint order, scale 1 (`obs_dof_vel_scale`). |
| `last_action` | 23 | Previous clipped network output. Zeros at the default pose. |
| heading error | 1 | Wrapped yaw error, limited to ±0.5 rad. See below. |
| standing | 1 | `1` when the command norm is below 0.01, else `0`. |

Policy joint order (23), which is not `T2_JOINT_NAMES` order:

```text
left_shoulder_pitch, right_shoulder_pitch, waist_pitch,
left_shoulder_roll, right_shoulder_roll, waist_roll,
left_elbow_pitch, right_elbow_pitch, waist_yaw,
left_elbow_yaw, right_elbow_yaw,
left_hip_pitch, right_hip_pitch,
left_hip_roll, right_hip_roll,
left_hip_yaw, right_hip_yaw,
left_knee_pitch, right_knee_pitch,
left_ankle_pitch, right_ankle_pitch,
left_ankle_roll, right_ankle_roll
```

Heading: the target resets to the current yaw when it is unset, when the
command norm is below 0.01, or when any command element changes by more
than `1e-6`. Otherwise it integrates `yaw_rate * 0.02`. The error is
`wrap(target - yaw)`, then limited to ±0.5 rad.

## Action

The network returns 23 values, clipped to ±10. For policy joint `i`:

```text
q_target = default_q + action * action_scale[i]
```

`action_scale` is the 23-element list on `T2LocomotionPolicyCfg` (about
0.26 on the shoulders down to 0.095 on the ankle rolls). Joints the policy
does not name keep `default_q`.

## Run

Fetch once (cached under `~/.ohho/assets/booster`, not committed):

```bash
python -m ohho_booster.assets
```

Then, headless (the default; the process is a physics thread):

```bash
ohho sim --robot booster-t2
```

That prepares, enters walk mode, and drives the usual timed pattern.

The same loop can open MuJoCo's native window. The viewer is
`mujoco.viewer.launch_passive`, synced from each model step:

```bash
ohho sim --robot booster-t2 --viewer
ohho sim-serve --robot booster-t2 --viewer
```

On macOS, `python` does not run that window on the main thread. Use the
launcher installed with MuJoCo:

```bash
mjpython -m ohho sim --robot booster-t2 --viewer
```

A session with no display (CI, or SSH without a display) makes `--viewer`
exit with an error. The headless command is what CI runs.

`plugins/ohho-booster/scripts/walk_check.py` is the scripted check: stand
30 s, walk about 1 m at 0.3 m/s, turn, and stop. The same function is what
the unit test asserts.
