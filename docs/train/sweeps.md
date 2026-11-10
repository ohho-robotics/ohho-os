# Training sweeps (`ohho.sweep/v1`)

`ohho sweep plan` reads one search-space document and prints the concrete
trials a later training run would launch. The same JSON Schema is package
data, so a console can validate the file with the same contract. `strategy`
is `grid`, `random`, `tpe`, `bayesian`, or `multi`. `tpe` and `bayesian`
are the built-in Tree-structured Parzen Estimator. `multi` is the
multi-objective form of that estimator and stores a Pareto set.

```bash
ohho sweep plan sweep.yaml
ohho sweep plan sweep.yaml --json
```

The text plan prints `trials:` and `estimated_gpu_hours:`. The GPU-hour
figure is an estimate computed from per-trial values in the spec. Planning
does not train and does not start a GPU job.

`sweep.yaml` may be JSON. A file that starts with `{` or `[` is parsed as
JSON. A `.json` suffix is always parsed as JSON. Other files use the YAML
subset below. The base install does not need PyYAML.

## Recipe connection

`recipe` is an opaque mapping. `ohho.sweep/v1` copies it onto every trial
and writes dotted overrides on top. A trial's `sets` lines use the form
`path=value` (JSON scalars), the same form `ohho train --set` uses:

```text
model="small"
train.batch_size=8
optimizer.lr=1e-05
```

The recipe document has its own schema (OHH-507). This sweep schema does
not define those fields and does not validate them. A trainer that already
has the recipe schema can validate each trial's resolved `recipe` after
expansion. Until that schema is applied, the sweep tools leave the mapping
unchecked on purpose.

Overrides create missing intermediate mappings. A path that would descend
through a value that is already a non-mapping is an error.

## Document

| Field | Role |
| --- | --- |
| `schema` | Required. Always `ohho.sweep/v1`. |
| `recipe` | Opaque base recipe. Default `{}`. |
| `dimensions` | Search axes, expanded in list order. Default `[]` (one trial, the base recipe). |
| `constraints` | Drop invalid combinations after expansion. |
| `objectives` | Metrics a later runner optimizes. At most one `primary`. |
| `early_stopping` | `rule` (`patience`, `asha`, or `median`), `rungs`, `reduction_factor` (default 4), `grace_period` (default 0), plus `metric`, `patience`, `min_delta`, and `mode`. It does not change which trials are proposed. |
| `strategy` | `grid` (default), `random`, `tpe`, `bayesian`, or `multi`. |
| `seed` | Integer ≥ 0. Derives the sampler seed and each trial's training seed. Grid order ignores it. |
| `parallelism` | Integer ≥ 1. Recorded on the plan. It does not change the trial list. |
| `budget.max_trials` | Keep this many surviving trials. Required for `random`, `tpe`, `bayesian`, and `multi` when `dimensions` is non-empty. `0` yields an empty plan. |
| `sampler` | Knobs for `tpe`, `bayesian`, and `multi`. `grid` and `random` ignore them. |
| `budget.max_gpu_hours` | Stop before the next trial would push the estimated total above this value. |
| `estimate` | Per-trial GPU-hours used to compute the estimate. |

The schema file is `ohho/train/schemas/sweep.v1.json`
(`$id` `https://ohho-robotics.com/schemas/ohho.sweep.v1.json`).
Examples live in `ohho/train/schemas/examples/`.
`ohho.train.sweep.validate_sweep` is the validator the CLI uses. It checks
the JSON Schema and the semantic rules named in that file's `description`.

## Dimensions

Each dimension has a `name`, a `type`, and an optional `path`. `path`
defaults to `name`. Paths are dotted keys (`optimizer.lr`). Names and paths
are unique.

| Type | Grid levels | Random draw |
| --- | --- | --- |
| `categorical` | `values` (required), in listed order | uniform over `values` |
| `int` | `values`, or `low`/`high` with `step` (default 1) or `n` points | uniform over that same set; `low`/`high` alone is uniform over the integers |
| `float` | `values`, or `n` linspace points from `low` to `high` (endpoints included) | those points when `n` or `values` is set; otherwise continuous uniform |
| `log_uniform` | `values`, or `n` log-spaced points from `low` to `high` (endpoints included, both bounds positive) | those points when `n` or `values` is set; otherwise continuous log-uniform |

`n: 1` is the midpoint (float) or geometric mean (log-uniform). An `int`
`n` that cannot produce that many strictly increasing points is rejected;
pass `values` or `step` instead. `values` cannot be combined with `low`,
`high`, `step`, or `n`.

Grid order is the cartesian product in dimension order, values in listed
or numeric order. The first dimension varies slowest. The same grid is
produced for every seed. A grid whose cartesian size, before constraints,
exceeds `ohho.train.sweep.MAX_TRIALS` (100000) is rejected.

`random` draws `budget.max_trials` trials with
`random.Random(sampler_seed(seed))`. `ohho.train.sweep.sampler_seed`
returns the sweep seed, so this is `random.Random(seed)`. The same sweep
seed reproduces the same trials. A draw that fails a constraint is rejected
and another is taken. If the constraints cannot be satisfied, the expander
raises. `tpe`, `bayesian`, and `multi` use this same generator for their
startup batch (`sampler.startup_trials`, capped by `budget.max_trials`).
Later trials are described under Strategies. Each trial's training seed is
separate; see Seeds.

## Conditional parameters

`when` is a condition. The dimension is applied only to trials where the
condition matches the base recipe plus earlier dimensions. Its key is
absent on the other trials, including the resolved recipe. A conditional
dimension with several levels multiplies only the matching trials.

`when` sees the base recipe and dimensions listed above it. Put a
dimension after the axes it depends on.

LoRA rank only on a VLA family (the packaged `conditional.json` example):

```yaml
dimensions:
  - name: family
    type: categorical
    values: [act, vla]
  - name: lora_rank
    path: model.lora.rank
    type: int
    values: [4, 8]
    when:
      path: family
      eq: vla
```

The `act` trial has no `model.lora.rank` override. Each `vla` trial has
one rank.

Precision, pruning, dataset, and model choice are ordinary categorical
dimensions. Nothing in the schema is specific to one robot.

## Conditions and constraints

A condition is one of:

- `{path, eq|ne|in|not_in|lt|lte|gt|gte|exists}` — operators on one path are AND. A missing path makes every operator false except `exists: false`.
- `{all: [condition, ...]}`
- `{any: [condition, ...]}`
- `{not: condition}`

`1` and `1.0` compare equal. `true` and `1` do not.

Constraints run after the grid is built (and inside random sampling):

```yaml
constraints:
  - name: int8-small-only
    prune_when:
      all:
        - path: precision
          eq: int8
        - path: model
          eq: large
```

`keep_when` drops a trial when its condition is false. A constraint has
one of `prune_when` or `keep_when`.

## Estimate, budget, early stopping

```yaml
estimate:
  gpu_hours_per_trial: 0.25
  by_value:
    - when:
        path: model
        eq: large
      gpu_hours: 1.0
```

The first matching `by_value` rule sets that trial's hours. Otherwise the
trial uses `gpu_hours_per_trial` (default 0). The plan sums the trials it
kept. `budget.max_gpu_hours` stops at the first trial that would exceed
the running total. `budget.max_trials` keeps a prefix of the surviving
grid, then the hour cap is applied.

`objectives` are copied onto the plan. For `grid` and `random` they do not
change which trials exist. For `tpe`, `bayesian`, and `multi`, objective
values reported to `SweepSampler.tell` choose later trials. `early_stopping`
does not change which trials are proposed. The trial runner and the sweep
controller use it to stop a trial early. See Early stopping.

## 2 × 3 × 4 grid

`ohho/train/schemas/examples/grid_2x3x4.yaml` is two models, three batch
sizes, and four log-spaced learning rates. The plan contains 24 trials.
Each trial's `recipe` is the base recipe with those three overrides
applied, so `optimizer.weight_decay` from the base recipe is still there
and `optimizer.lr` is the sampled rate. Twelve `small` trials are estimated
at 0.25 GPU-hours and twelve `large` trials at 1.0, for a total of 15.

```bash
ohho sweep plan ohho/train/schemas/examples/grid_2x3x4.yaml
```

`--json` prints the same plan. `trials` is the count. `trial_recipes`
holds each resolved recipe, its `sets`, and its estimated hours.

## Strategies

| Strategy | What `ohho sweep plan` prints | What proposes the next trial |
| --- | --- | --- |
| `grid` | The constrained cartesian product, then the budget caps | `SweepSampler` walks that same list |
| `random` | `budget.max_trials` seeded draws, then the budget caps | `SweepSampler` continues that same generator |
| `tpe` | The random startup batch | Built-in Tree-structured Parzen Estimator |
| `bayesian` | The same estimator as `tpe` | The same estimator as `tpe` |
| `multi` | The random startup batch | That estimator with a Pareto split |

`tpe` and `bayesian` need one objective. When several objectives are set,
one of them is `primary`. `multi` needs at least two objectives. A float or
`log_uniform` dimension may omit `n` for every strategy except `grid`.

`sampler` defaults are `startup_trials: 10`, `gamma: 0.25`,
`candidates: 64`, and `prior_weight: 1`. The startup trials are random
draws from the derived sampler seed. After that many objectives have been
recorded (and at least two), the estimator builds a Gaussian Parzen density
on the better `gamma` fraction of trials and another on the rest. It draws
`candidates` from the better density and samples the next trial with
probability proportional to the ratio of the two densities. The bandwidth
shrinks as more trials are told. `multi` puts the current Pareto front in
the better group when that front is smaller than the whole run.

The estimator is standard-library code. It does not import Optuna, NumPy,
or PyTorch.

```python
from ohho.train.sweep_sampler import SweepSampler

sampler = SweepSampler.from_sweep("sweep.yaml")
trial = sampler.ask()[0]
sampler.tell(trial.index, {"val_loss": 0.4})
```

`ask(n)` proposes up to `n` trials from the model built so far. Those
trials do not see one another's objectives until `tell`. `ask` raises
`SweepError` with the text `sweep budget is exhausted` when the trial cap,
the GPU-hour cap, or the grid has nothing left. One `ask` that returns a
short tuple has used the last trials that fit.

A conditional dimension is omitted when its `when` does not match, the same
way the expander omits it. Constraints reject a draw and another draw is
taken.

`grid` and `random` samplers propose the same overrides, in the same order,
as `expand` on that spec. `tpe`, `bayesian`, and `multi` plans from
`expand` are only the startup prefix of that sampler.

## Pareto leaderboard

`SweepSampler.leaderboard()` returns an `ohho.sweep.leaderboard/v1`
document. The same object is stored on sampler state under `leaderboard`.
The schema file is `ohho/train/schemas/sweep.leaderboard.v1.json`.

A completed trial is on the Pareto set when no other completed trial
dominates it. Trial A dominates trial B when A is no worse on every
objective and strictly better on at least one. `maximize` prefers a larger
number. `minimize` prefers a smaller number. Equal values do not dominate.
`pareto` lists that set in ascending trial index. `trials` lists every told
trial and sets `dominated` when it is off the set. Proposals that have not
been told are omitted.

`success_rate`, `latency_p95_ms`, and `memory_mb` are ordinary objective
names. Nothing in the schema is specific to those metrics.

```json
{
  "schema": "ohho.sweep.leaderboard/v1",
  "strategy": "multi",
  "seed": 1,
  "objectives": [
    {"name": "success_rate", "direction": "maximize", "primary": false},
    {"name": "latency_p95_ms", "direction": "minimize", "primary": false},
    {"name": "memory_mb", "direction": "minimize", "primary": false}
  ],
  "trials": [
    {
      "trial": 0,
      "values": {"success_rate": 0.9, "latency_p95_ms": 10, "memory_mb": 100},
      "overrides": {"tag": "a"},
      "sets": ["tag=\"a\""],
      "dominated": false
    }
  ],
  "pareto": [
    {
      "trial": 0,
      "values": {"success_rate": 0.9, "latency_p95_ms": 10, "memory_mb": 100},
      "overrides": {"tag": "a"},
      "sets": ["tag=\"a\""]
    }
  ]
}
```

`overrides` are the dotted path values for that trial. `sets` uses the
`path=value` form. `trial` is the sampler's trial index.

## Seeds

The spec has one `seed`. Two integers are derived from it:

| Value | Function | Role |
| --- | --- | --- |
| Sampler seed | `ohho.train.sweep.sampler_seed` | The sweep seed. It seeds the proposal generator. `grid` order ignores it. |
| Trial training seed | `ohho.train.sweep.trial_training_seed` | Seed a trainer should use for that trial index. |

The sampler seed is the sweep seed, so the proposal generator stays
`random.Random(sweep_seed)`. The training seed is SHA-256 of the JSON array
`["ohho.sweep.seed/v1", sweep_seed, "trial", trial_index]`. The integer is
the first four digest bytes, unsigned and big-endian, so it is in
`0 .. 2**32 - 1`. It does not draw from the proposal generator. Asking,
telling, or resuming does not change the training seed of a trial index.

The same spec and the same sweep seed propose the same trial sequence.
`grid` order is the cartesian product for every seed; the training seeds
still change when the sweep seed changes. `random` repeats when the sweep
seed repeats. `tpe`, `bayesian`, and `multi` also repeat when the told
objective values repeat in the same order.

`expand` and `SweepSampler.ask` store the training seed on each trial as
`seed`. `ohho sweep plan` prints it. `ohho.train.sweep.derived_trial_seed`
reads that field from a trial-table row. `ohho.run/v1` run manifests are
not defined in this package. A later manifest writer copies this integer
onto the run record. This sweep record does not invent that schema.

## Resume

`SweepSampler.dump_state()` returns an `ohho.sweep.sampler/v1` document.
The schema file is `ohho/train/schemas/sweep.sampler.v1.json`. It holds the
normalized spec, the random generator state, the trial log, a GPU-hour stop
flag, and the leaderboard derived from told trials. `values: null` means
the trial was proposed and not told.

```python
import json
from ohho.train.sweep_sampler import SweepSampler

sampler = SweepSampler.from_sweep(spec)
trial = sampler.ask()[0]
sampler.tell(trial.index, {"val_loss": 0.4})
state = json.loads(json.dumps(sampler.dump_state()))
restored = SweepSampler.load_state(state)
assert restored.ask()[0].overrides == sampler.ask()[0].overrides
```

Loading that state and asking again proposes the same next trials as the
sampler that kept running, including later tells of the same objective
values. `grid` order does not use the generator. `random` continues it.
`tpe`, `bayesian`, and `multi` continue it and refit the densities from the
told trials. Each restored trial keeps the training seed derived from the
sweep seed and its index.

## Persisted runs

A stopped sweep continues from disk. The file is `ohho.sweep.run/v1`
(`ohho/train/schemas/sweep.run.v1.json`). It holds the sampler state and
the trial table. The table row's `seed` is that trial's training seed.
`values: null` means the trial was proposed and not told.

```bash
ohho sweep start sweep.yaml --id push-lr
ohho sweep resume push-lr
ohho sweep resume push-lr --count 4 --json
```

`OHHO_SWEEP_DIR` overrides the default directory `~/.ohho/sweeps`. The run
is `<dir>/<id>/run.json`. `start` refuses an id that already exists.
`resume` loads that file and proposes the next trials. Indexes already in
the table are not proposed again. One process should resume a given id at
a time.

```python
from ohho.train.sweep_run import resume_sweep, start_sweep

run = start_sweep("sweep.yaml", sweep_id="push-lr")
trial = run.ask()[0]
run.tell(trial.index, {"val_loss": 0.4})
stopped = resume_sweep("push-lr")
assert stopped.ask()[0].index == 1
```

The Connect console Resume button is a separate surface. This package
exposes the run and `ohho sweep resume`.

## Early stopping

`early_stopping` names the scalar a trial runner reports while a trial is
still training, and the rule that may stop that trial before the full
budget. The sampler still proposes the same trials. Stopping is a decision
on a trial that has already been asked.

The sweep controller that fans those trials out across jobs is not in this
package. `EarlyStopping` and `SweepRun.report` / `SweepRun.decide` are the
library that controller calls. They do not launch training processes.

| Field | Meaning |
| --- | --- |
| `metric` | Name of the scalar the runner reports. It does not have to be an objective. |
| `rule` | `patience` (default), `asha`, or `median`. |
| `mode` | `min` (smaller is better) or `max` (larger is better). |
| `rungs` | Strictly increasing steps where a decision is made. Required for `asha`. Empty means every reported step can decide for `patience` and `median`. |
| `reduction_factor` | Integer ≥ 2, default 4. For `asha` this is eta. For `median` it is the minimum number of reports at a step before anyone can be stopped. `patience` ignores it. |
| `grace_period` | Integer ≥ 0, default 0. A trial is not stopped before this step. Every rung must be ≥ this value. |
| `patience` | For `patience` only. Stop after this many decision steps with no improvement. Default 1. |
| `min_delta` | For `patience` only. An improvement must be strictly better than the best value by this amount. Default 0. |

`asha` is successive halving. At a rung, once `n >= reduction_factor`
trials have reported that rung, a trial stops when its metric is strictly
worse than the best `floor(n / reduction_factor)` values. A value that
ties that cutoff continues. The first decision at a rung is kept: a trial
that was told to continue is not stopped at that same rung when later
peers report. Reporting every active trial at the rung, then calling
`decide`, is the synchronous bracket and is what the benchmark below uses.
Calling `decide` as each report arrives is the asynchronous rule, and an
early arrival can keep training even if a later full rung would have
stopped it.

`median` stops a trial that is strictly worse than the median of the
reports at that step. A value equal to the median continues. `patience`
watches one trial. The first decision step sets the best value. Later
decision steps that do not improve it by `min_delta` count toward
`patience`.

```yaml
early_stopping:
  metric: val_loss
  mode: min
  rule: asha
  rungs: [1, 3, 9, 27, 81]
  reduction_factor: 3
  grace_period: 1
```

`ohho/train/schemas/examples/asha.json` is that rule on a 32-trial random
sweep. A median rule omits `rungs` when every step after the grace period
should be checked:

```yaml
early_stopping:
  metric: val_loss
  mode: min
  rule: median
  grace_period: 5
  reduction_factor: 2
```

The trial runner calls `report` at a step. The controller calls `decide`.
`decide` returns `continue` or `stop`.

```python
from ohho.train.sweep_run import start_sweep

run = start_sweep("sweep.yaml", sweep_id="push-lr")
trials = run.ask(32)
active = [trial.index for trial in trials]
for step in run.spec.early_stopping["rungs"]:
    for index in list(active):
        value = read_validation_metric(index, step)
        run.report(index, step, value, checkpoint=f"checkpoints/{index}/{step}")
    active = [
        index
        for index in active
        if run.decide(index).action == "continue"
    ]
for index in active:
    run.tell(index, {"val_loss": read_validation_metric(index, step)})
```

`stop` sets the trial-table status to `pruned`. The last checkpoint path
and the intermediate metrics stay on the row. `values` stays null, so a
partial curve is not a completed objective and does not enter the
leaderboard or the Parzen estimator. `tell` is what marks a trial
`completed` and records those objectives. `mark_failed(index, error)` sets
`failed` instead. A pruned trial is not failed, and neither call rewrites
the other status. `ohho sweep resume` loads the saved row, so a pruned or
failed trial stays that way and its index is not proposed again.

## Simulated ASHA benchmark

`ohho.train.sweep_early.simulate_asha_budget` runs 32 synthetic trials.
Trial `i` at step `t` reports `(i + 1) / t`. The loss falls as the step
grows, trial 0 is best at every step, and the curves do not cross. The
rungs are 1, 3, 9, 27, and 81. The reduction factor is 3. The loop reports
every trial that is still active at a rung before it decides that rung.

A trial stopped at a rung has used that many steps. The same 32 trials
with no early stopping use `32 * 81 = 2592` trial-steps. The ASHA run uses
142 trial-steps, which is under half of 2592. Thirty-one trials are
`pruned` and keep a checkpoint path and their metrics. The best trial
reaches step 81. `tests/test_sweep_early.py` checks that bound.

## YAML subset

Spaces only. Indent nested blocks deeper than the line that introduces
them. List items start with `- `. Flow lists and mappings fit on one line
(`values: [8, 16, 32]`, `when: {path: family, eq: vla}`). Quotes use JSON
escapes in double quotes, and `''` inside single quotes. Scalars are
strings, integers, floats (`1.0e-5`), `true`, `false`, and `null`. `#`
starts a comment outside quotes. A leading `---` is accepted. Tabs,
anchors, tags, and block scalars (`|`, `>`) are rejected.

## Package data

```python
from importlib.resources import files

text = files("ohho.train").joinpath("schemas", "sweep.v1.json").read_text(encoding="utf-8")
```

The wheel lists those files in `[tool.setuptools.package-data]`.
