# K1 walk in MuJoCo

`ohho sim --robot booster-k1` uses the generic `humanoid-mujoco` backend.
This package supplies the MJCF, PD gains, observation layout and the
`k1_walk.pt` checkpoint. Core code does not name this robot. The result is
sim only, not hardware-tested.

The checkpoint and the numbers below are from
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy) commit
`7bb1462e2ac742eba5c01659b815e2e99fbe617f` (Apache-2.0):

- `tasks/locomotion/robots/k1/__init__.py` (`K1WalkControllerCfg`)
- `tasks/locomotion/locomotion.py` (`K1LocomotionPolicyCfg`, `LocomotionPolicy`)
- `booster_deploy/robots/k1.py` (`K1_CFG.mjcf_path`, `joint_names`, `sim_joint_names`)
- `tasks/locomotion/robots/k1/models/k1_walk.pt`

The description archive is booster_assets commit
`12a97516f57469078707afb587a5f84c1cb110d1` (BSD-3-Clause). Both pins, with
sha256, are in `ohho_booster/assets_manifest.json`. Nothing is vendored.

## Which MJCF

The serial model `robots/K1/K1_22dof.xml` is the one this config loads.
`K1_CFG.mjcf_path` points at that file. It has 22 hinge motors plus the
free joint. Joint names, PD gains, the default pose and the effort clip
use `K1_CFG.joint_names` (the real-robot order). MuJoCo `qpos` for this
serial file follows that same order.

`K1_22dof_parallel.xml` is not used. It names the same 22 actuated joints
and adds the closed-loop ankle linkages (31 joint names, 22 motors). The
checkpoint never sees those constraint joints. Loading it under the same
action vector would change the kinematics and the contacts.

`K1_CFG.sim_joint_names` is the Isaac joint order: the same 22 names in a
different sequence. This runner indexes joints by name, so the policy
order is `K1WalkControllerCfg.policy_joint_names`, not the Isaac order.
The retarget list in booster_assets `motions.py` drops the `aa` prefix on
the shoulder pitch joints. Those names are not in the MJCF, so they are
not used here.

Head joints are not in the 20-value action vector. Their targets stay at
the default pose.

## Command ranges are not Ohho caps

The walk task's `VelocityCommandCfg` is the training range:

| Command | Training range |
|---|---|
| forward `vx` | 0 to 1.6 m/s |
| backward `vx` | 0 to 0.3 m/s |
| `vy` | ±0.3 m/s |
| yaw rate | ±1.8 rad/s |

Those numbers are what the checkpoint was trained to track. They are not
OhhO limits. The spec keeps the generic humanoid description (0.5 m/s,
1.0 rad/s). `Robot.drive` clamps to that spec. On a link that is not
simulated, SafetyGate clamps again. This package does not raise either cap.

The spec does not declare `manipulation` or `manipulation.dual_arm`. Each
arm is four joints and there is no end effector. `kSoccer` stays
`UNKNOWN`; the posture modes are the same closed set as T2.

## Control timing

| | |
|---|---|
| Policy period | 0.02 s (50 Hz) |
| Physics step | 0.002 s (`policy_dt / 10`) |
| Policy input | 10 frames × 69 values = 690 |
| Policy output | 20 joint-position offsets |

One frame is angular velocity (3), projected gravity (3), command (3),
relative joint position (20), joint velocity × 0.1 (20) and the previous
action (20). There is no heading or standing extra. History starts by
repeating the first frame. The last action starts at zero. The TorchScript
`actor` submodule is the network that runs; the observation normalizer on
this file is identity, and selecting `actor` matches the deployment runner.

The MJCF file itself sets a 0.001 s step. The deployment controller
replaces that with 0.002 s, and so does this config. PD still runs on
every physics step, then each torque is clipped to
`K1WalkControllerCfg.effort_limit`. That clip is a deployment limit. It is
not an Ohho safety cap. The floating base starts at `(0, 0, 0.6)`, which
is `MujocoControllerCfg.init_pos` (this task does not override it).

Before the action scale of 0.25, the right-elbow pitch action is reduced
by 0.2 (`arm_action_fix`). Each position target is then low-passed with
weight 0.8, starting from the default pose:

```text
q_target = previous + 0.8 * ((default_q + (action - offset) * 0.25) - previous)
tau = clip(kp (q_target - q) - kd dq, ±effort_limit)
```

## Follow-up

Isaac Lab's K1 task is not part of this package.
