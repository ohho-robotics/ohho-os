# Training sweeps (`ohho.sweep/v1`)

`ohho sweep plan` reads one search-space document and prints the concrete
trials a later training run would launch. The same JSON Schema is package
data, so a console can validate the file with the same contract.

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
| `early_stopping` | `metric`, `patience` (default 1), `min_delta` (default 0), `mode` (`min` or `max`). |
| `strategy` | `grid` (default) or `random`. |
| `seed` | Integer ≥ 0. Grid order ignores it. `random` draws from this seed. |
| `parallelism` | Integer ≥ 1. Recorded on the plan. It does not change the trial list. |
| `budget.max_trials` | Keep this many surviving trials. Required for `strategy: random` when `dimensions` is non-empty. `0` yields an empty plan. |
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

`random` draws `budget.max_trials` trials with `random.Random(seed)`. The
same seed reproduces the same trials. A draw that fails a constraint is
rejected and another is taken. If the constraints cannot be satisfied, the
expander raises.

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

`objectives` and `early_stopping` are copied onto the plan for the runner.
They do not change which trials exist.

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
