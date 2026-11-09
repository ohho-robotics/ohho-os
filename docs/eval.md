# Checkpoint eval (`ohho eval`)

`ohho eval` scores a checkpoint by running seeded episodes in simulation and
writing an `ohho.eval/v1` report. The CPU backend is the in-process kinematic
sim (`backend` is `kinematic-sim`). It does not launch Gazebo and it does not
arm hardware.

## Command

```bash
ohho eval CHECKPOINT --suite gazebo-nav-regression --episodes 5 --seed 0 --out report.json
```

| Flag | Meaning |
|---|---|
| `CHECKPOINT` | File or directory. A directory is digested from its sorted relative paths and file hashes. |
| `--suite` | Suite name. `gazebo-nav-regression` is the Proof v0 Gazebo nav regression suite. |
| `--episodes` | Episodes per scenario (default 5). Episode `i` of scenario `s` uses `seed + s * episodes + i`. |
| `--seed` | Base seed. The same checkpoint and seed repeat the same success rate. |
| `--scenarios` | Proof scenario YAML to read. Overrides the checkout lookup. |
| `--videos DIR` | Write a pose trace per episode and list those paths on the report. Omit it and `videos` is `[]`. |
| `--out` | Write the report JSON. A one-line summary still prints to stdout. |

Exit 0 means the report was produced. A missing checkpoint or scenario file exits 1. The success rate is a result, not an exit code.

## Proof suites are referenced

Scenario geometry is not vendored in ohho-os. The suite pointer lives in
`ohho/eval/suites.json`. `gazebo-nav-regression` points at:

`omnibot-ros2/omnibot_navigation/config/nav_regression_scenarios.yaml`

in [ohho-robotics/OmniBot](https://github.com/ohho-robotics/OmniBot).

Resolution order:

1. `--scenarios PATH`
2. `OHHO_EVAL_SCENARIOS` (a file path)
3. `OHHO_PROOF_ROOT`, plus the relative path above. Point it at the OmniBot checkout.

When the checkout lookup is used, `scenario_source` is the repository reference
(`https://github.com/ohho-robotics/OmniBot#omnibot-ros2/omnibot_navigation/config/nav_regression_scenarios.yaml`).
The report stores scenario ids, not the poses.

The kinematic rollout reads each scenario's start, goal, and obstacle box. XY
success is 0.30 m and yaw success is 0.50 rad, the same limits as the Proof
nav regression checker. The box is expanded by a 0.18 m body radius. Contact
is `collision`. Still short of the goal after 400 steps of 0.05 s is
`timeout`.

A JSON checkpoint may set controller gains:

```json
{"eval": {"speed_mps": 0.4, "heading_gain": 2.5}}
```

Otherwise the cruise speed is taken from the checkpoint digest. The episode
seed scales that speed. Other checkpoint formats (a directory, a weight file)
are still scored; only the digest and the optional `eval` object are read.

## Report schema `ohho.eval/v1`

The contract is [`schemas/ohho.eval.v1.json`](../schemas/ohho.eval.v1.json)
(`$id` `https://ohho-robotics.com/schemas/ohho.eval/v1.json`). The same file
ships as package data on `ohho.eval`.

```python
from importlib.resources import files

text = files("ohho.eval").joinpath("schemas/ohho.eval.v1.json").read_text(encoding="utf-8")
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | Constant `ohho.eval/v1`. |
| `checkpoint` | yes | Path that was evaluated. |
| `checkpoint_sha256` | yes | 64-char hex digest. |
| `suite` | yes | Suite name. |
| `scenario_source` | yes | Repository reference or file path. Not the scenario body. |
| `scenario_ids` | yes | Ids in run order. |
| `seed` | yes | Base seed. |
| `backend` | yes | `kinematic-sim` on this runner. |
| `episodes_per_scenario` | yes | Integer ≥ 1. |
| `success_rate` | yes | Successes / episodes, in `[0, 1]`. |
| `success_ci95` | yes | Wilson score interval. `method` is `wilson`, `level` is `0.95`, plus `low`, `high`, `successes`, `trials`. |
| `episode_length` | yes | `mean`, `min`, `max` in `steps`. |
| `failure_modes` | yes | Counts for unsuccessful episodes (`collision`, `timeout`). |
| `videos` | yes | Trace paths. Empty when `--videos` was omitted. |
| `episodes` | yes | One row per episode: `scenario_id`, `seed`, `success`, `length`, `failure_mode`, `video_path`. |

The root object allows additional properties. Removing, renaming, or retyping a
field is `ohho.eval/v2`. `success_rate` is 0 when a run has no episodes; the
Wilson interval is then `(0, 1)`.

`z` for the 95% interval is `statistics.NormalDist().inv_cdf(0.975)`.

## Training and sweep hook

Training and sweep code live in other changes. They should call this package
at the end of a trial. This package does not import them.

```python
from ohho.eval import JOB_TYPE, evaluate_after_train, run_eval_job

report = evaluate_after_train(
    checkpoint,
    "gazebo-nav-regression",
    enabled=True,  # default False, so a caller can always invoke the hook
    episodes=5,
    seed=trial_seed,
    scenarios=None,  # or a path; otherwise OHHO_PROOF_ROOT
)

report = run_eval_job(
    {
        "type": JOB_TYPE,  # "eval"
        "checkpoint": checkpoint,
        "suite": "gazebo-nav-regression",
        "episodes": 5,
        "seed": trial_seed,
        "enabled": True,
    }
)
```

`evaluate_after_train(..., enabled=False)` returns `None` and does not read
the checkpoint. `run_eval_job` defaults `enabled` to true because the job
itself is an eval.
