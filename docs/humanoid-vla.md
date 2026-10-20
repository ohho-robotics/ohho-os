# Upper-body policy on a humanoid

The mobile base and a humanoid share `ohho.train.finetune` and `ohho.serve`.
`ohho.policy_contract` reads state and action names from the robot spec, or
from the dataset the recorder wrote. A humanoid action is body velocity, then
the arm groups, then the grippers. Leg joints stay in the observation and out
of the action. Body velocity is `Robot.drive`. Arm targets are
`Robot.move_joints` on the arm groups.

Those calls go through `SafetyGate` when the link is gated. `move_joints` is
forwarded only in `custom` mode, only on a simulated target, and only for
joints that have limits. The 0.15 rad step cap still applies. `custom` is not
a remote tool. Legs are not given limits by the demo session, so a leg group
stays refused.

A GPU fine-tune has not been run. The numbers below are commands, not a
trained success rate.

## Collect

Record a scripted reach without an operator. The arm group is the first one
named on the spec (left, then right). The command ramps one joint and leaves
body velocity at zero.

```bash
ohho vla collect ROBOT --episodes 2 --steps 8 --output ./upper-body-data
```

`ROBOT` is a registered spec whose sim backend is `humanoid-mujoco` and whose
assets include an MJCF. The same entry point is
`record_upper_body_episodes`. `python scripts/upper_body_vla.py collect ...`
calls the same command.

The dataset is LeRobot v2.1. `observation.state` and `action` names come from
`joint_groups`. `arm_action` is `arm_joints`.

## Train

Smoke check (a couple of ACT steps on CPU when torch is installed, otherwise
the mock checkpoint). SmolVLA, diffusion, and openvla validate a recipe and
do not start training:

```bash
ohho train --dataset ./upper-body-data --policy act --smoke --steps 2 --output ./checkpoints
ohho train --dataset ./upper-body-data --policy smolvla --smoke --output ./checkpoints
```

The GPU fine-tune is the same entry point without `--smoke`. It has not been
run:

```bash
ohho train --dataset ./upper-body-data --policy act --output ./checkpoints --device cuda
```

`training_recipe` records `gpu_finetune: not_run` until that run exists.

## Evaluate

Run a checkpoint in the headless humanoid simulator and write a JSON report.
A scripted checkpoint is a stand-in so the harness runs without a trained
policy. Its success rate is not a fine-tune result.

```bash
ohho vla eval ROBOT --checkpoint ./checkpoints/act --episodes 2 --steps 16 --output ./eval.json
```

The report has `success_rate`, `gpu_finetune` (`not_run`), and a note that a
GPU fine-tune has not been run. The console line is
`upper-body eval: <successes>/<episodes> success_rate=<rate> policy=<name>`.

## Still to do

- Run the GPU fine-tune with `ohho train` and without `--smoke`.
- Log the eval success rate of that checkpoint.
