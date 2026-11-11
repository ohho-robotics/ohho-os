# Run manifest (`ohho.run/v1`)

`ohho train` writes `run_manifest.json` in the recipe output directory when the family emits `train_start`. It finalises that same file when the run ends. The schema file is package data:

`ohho/train/schema/ohho.run.v1.json`

Its `$id` is `https://ohho-robotics.com/schemas/ohho.run/v1.json` and its title is `ohho.run/v1`. `"schema"` inside the document is the constant `ohho.run/v1`.

```bash
ohho run show checkpoints/act
ohho run show checkpoints/act --json
```

The human command prints the console summary. `--json` prints the whole document. Exit `0` when the file is valid, `1` when it is missing or invalid, and `2` when the command line is wrong. A path to `run_manifest.json` itself is accepted as well as the directory that contains it.

## When the file is written

| Moment | `status` | `finished_at` | `error` |
| --- | --- | --- | --- |
| `train_start` | `running` | null | null |
| `run` returns and the checkpoint and metrics exist | `succeeded` | UTC timestamp | null |
| The trainer raises after `train_start` | `failed` | UTC timestamp | the error text |
| Keyboard interrupt, `SystemExit`, or a broken stdout pipe after `train_start` | `interrupted` | UTC timestamp | `interrupted` |

The interrupt is finalised and then propagated. A recipe that fails before `train_start` (parse error, schema error, unavailable family, missing dataset, ACT preflight) does not get a file. A family that returns without emitting `train_start` still gets a final manifest, written when `run` returns.

Times are UTC `YYYY-MM-DDTHH:MM:SSZ`. `git.commit` is the commit at the start. `git.dirty` is true when that checkout was dirty at the start or became dirty before the end. Artifacts written inside the recipe checkout therefore flag the tree dirty.

## Console run page

The console run page reads `run_manifest.json` from the run directory. The summary renders these fields. `ohho run show` prints the same lines, in this order.

| Manifest field | Page label | Human line |
| --- | --- | --- |
| `status` | Status | `status` |
| `started_at` | Started | `started_at` |
| `finished_at` | Finished | `finished_at` |
| `family` | Family | `family` |
| `device` | Device | `device` |
| `precision` | Precision | `precision` |
| `loss_scaling` | Loss scaling | `loss_scaling` |
| `precision_backend` | Precision backend | `precision_backend` |
| `seeds.recipe` | Seed | `seeds.recipe` |
| `seeds.python` | Python seed | `seeds.python` |
| `seeds.numpy` | NumPy seed | `seeds.numpy` |
| `seeds.torch` | Torch seed | `seeds.torch` |
| `seeds.pythonhashseed` | PYTHONHASHSEED | `seeds.pythonhashseed` |
| `seeds.extra` | Other seeds | `seeds.extra` |
| `git.commit` | Git commit | `git.commit` |
| `git.dirty` | Dirty tree | `git.dirty` |
| `dataset.version_id` | Dataset version | `dataset.version_id` |
| `dataset.hash` | Dataset hash | `dataset.hash` |
| `dataset.path` | Dataset path | `dataset.path` |
| `base_weights.uri` | Base weights | `base_weights.uri` |
| `base_weights.revision` | Base weights revision | `base_weights.revision` |
| `base_weights.hash` | Base weights hash | `base_weights.hash` |
| `base_weights.loaded` | Base weights loaded | `base_weights.loaded` |
| `container_image_digest` | Container image | `container_image_digest` |
| `library_versions` | Libraries | `library_versions` |
| `gpu` | GPU | `gpu` |
| `final_loss` | Final loss | `final_loss` |
| `precision_metrics` | Precision metrics | `precision_metrics` |
| `qat` | QAT | `qat` |
| `error` | Error | `error` |
| `resolved_config` | Resolved config | `resolved_config` |

`gpu` is null when no CUDA device is reported. When a device is reported the value is `{name, driver, cuda}`. `container_image_digest` is null when the process is not inside a container. `dataset.hash` is null when the recipe names only a version id, or the local path is not present. `base_weights.hash` is null when the uri is not a local file or directory. Remote URIs are stored and are not downloaded.

`seeds.recipe` is `hyperparameters.seed`. When that integer is set, the runner also seeds Python's `random`, and NumPy and torch when those libraries import. A null library seed means that library was not seeded. Other hyperparameters whose names end in `_seed` are listed under `seeds.extra`. `PYTHONHASHSEED` is recorded and is not changed after the process has started.

`git` describes the checkout that contains the recipe file. `commit` is null when that directory is not a git checkout or the checkout has no commit yet. `dirty` is true when `git status --porcelain` shows any change, including untracked files. When git cannot read the status of a checkout that does exist, `dirty` is true.

`library_versions` always includes `python` and `ohho` (the `ohho-os` version). `torch`, `numpy`, and `lerobot` are included when those distributions are installed. The version lookup uses package metadata.

`precision` is the recipe value the device check accepted (`fp32`, `fp16`, `bf16`, or `fp8`). `loss_scaling` is true only for fp16, where the step uses GradScaler. `precision_backend` is `fp32`, `torch.amp` (fp16 and bf16), or `transformer_engine` (fp8). `precision_metrics` has one key per precision. The precision that ran holds `throughput_steps_per_s` and `peak_memory_bytes`. The others are null. While the run is still going, and when the trainer did not write those keys, the numbers are null. Throughput is training steps per second over the step loop. On CUDA, peak memory is `max_memory_allocated` in bytes after the loop starts. On CPU and MPS it is the process resident set sampled during the loop, which includes the interpreter. Details are in [Mixed precision](precision.md).

The JSON document also carries `recipe_path`, `overrides`, `checkpoint_dir`, `metrics_path`, `declared_hooks`, `git.root`, and, when a run recorded a parent, `lineage`. Those are available to a details panel. They are not summary fields, except `lineage.teacher.version`, which `ohho run show` prints when a distillation run set it. `--json` prints them with the summary.
`qat` is null when the recipe has no `qat` block. When the block is set, the field is the resolved settings: bit width, scope, granularity, symmetry, calibration length, fine-tune length, and `export_onnx`. Defaults are filled, including activation granularity and symmetry when the recipe omits them. Phase losses and the ONNX path are in `metrics.json`, not in this object. Details are in [Quantisation-aware training](qat.md). A manifest that omits `qat` is still valid.

`lineage` is omitted when the run has no parent. A distillation run sets `lineage.teacher` to `{version, source, model, checkpoint}`. `version` is the registry model version, the explicit checkpoint version, or `sha256:` plus the sha256 of the teacher `policy.pt`. `source` is `registry` or `checkpoint`. `model` is null for a local checkpoint reference. Other compression methods may add their own keys beside `teacher`.
The JSON document also carries `recipe_path`, `overrides`, `checkpoint_dir`, `metrics_path`, `declared_hooks`, `git.root`, and, when this run was produced by `ohho run clone`, `cloned_from`. Those are available to a details panel. They are not summary fields. `--json` prints them with the summary. `ohho run show` does not add a `cloned_from` line to the summary.

## Database hook

The manifest file is the stored record. ohho-os does not open a database connection and does not read credentials. A process that wants its own copy installs a hook before training:

```python
from ohho.train.run_manifest import set_run_store

class LocalStore:
    def save(self, manifest: dict) -> None:
        """Persist manifest locally."""

set_run_store(LocalStore())
```

`save` receives a JSON object after the running file is written and again after the final file is written. A function of one argument works in place of an object with `save`. `set_run_store(None)` clears the hook. An exception from the hook is printed on stderr and does not change the process exit code. The training result still depends on the manifest file, not on the hook.

## Container digest

Outside a container, `container_image_digest` is null. Inside a container (`/.dockerenv`, `/run/.containerenv`, or a cgroup that names docker, containerd, podman, or Kubernetes) the digest is the first non-empty value of:

1. `OHHO_IMAGE_DIGEST`
2. the first line of `/etc/ohho/image-digest`
3. the first line of `/.ohho-image-digest`

When none of those is set, the field stays null. `OHHO_IMAGE_DIGEST` on a host that is not a container is ignored.

## Clone and tweak

`ohho run clone` builds a new run config from a source run's `resolved_config`. It does not start training.

```bash
ohho run clone checkpoints/act --set hyperparameters.lr=0.0003 --set hyperparameters.seed=2
ohho run clone checkpoints/act --set output.dir=checkpoints/act-lr --dest checkpoints/act-lr
```

`<id>` is a run directory or a path to `run_manifest.json`, the same paths `ohho run show` accepts. `--set` is repeatable. Keys are dotted, including numeric list indexes (`compression_hooks.0.params.rank`). Values use the same parser as `ohho train --set`: booleans, numbers, `null` (delete a mapping key), JSON arrays and objects, quoted strings, or plain strings. Later repeats of the same key win.

The tweaked document is validated against `ohho.recipe/v1`. Recipe defaults are filled after validation, matching `ohho train`. A config the schema rejects is not written. Exit `1` prints the summary on stderr and the schema errors under it. Exit `2` is an argparse usage error.

The new run directory defaults to a sibling named `<source>-clone`. `--dest DIR` chooses another empty directory. An existing non-empty destination is left untouched. The directory contains:

| File | Contents |
| --- | --- |
| `recipe.json` | The new run config. This is the tweaked `resolved_config`. |
| `run_manifest.json` | An `ohho.run/v1` document. `resolved_config` is that config. `overrides` is the `--set` list used for this clone (the source run's overrides are already baked into its resolved config). `recipe_path` is `recipe.json`. `family` and `device` follow the new config. `cloned_from` points at the source. |

`cloned_from` is an object `{run, manifest}`. `run` is the source run directory and `manifest` is its `run_manifest.json`, both absolute. The field is omitted on a run that was not cloned. Training does not write it.

Status, seeds, git, dataset, base weights, library versions, GPU, loss, checkpoints, and the other execution fields are copied from the source manifest. They describe the source execution, not a new training result. When the new config's `output.dir` is the source run directory, the command prints:

`note: output.dir is the source run; set output.dir before training this recipe`

Training `recipe.json` with `ohho train` writes a separate execution manifest in `output.dir`. That manifest does not copy `cloned_from`.

```python
from ohho.train.run_clone import clone_run, cloned_config

preview = cloned_config(manifest.resolved_config, ["hyperparameters.lr=0.0003"])
result = clone_run("checkpoints/act", ["hyperparameters.lr=0.0003"])
```

`cloned_config` is the pure preview. `clone_run` writes the directory and returns the config, the manifest, and the diff against the source.

## Config diff

`ohho run diff` compares `resolved_config` for any two runs. It does not compare status, loss, or `cloned_from`.

```bash
ohho run diff checkpoints/act checkpoints/act-clone
ohho run diff checkpoints/act checkpoints/act-clone --json
```

The first run is the before side. Exit `0` when both manifests load, including when the configs differ. Exit `1` when a run is missing or invalid. Exit `2` is a usage error.

Text and `--json` are two renderings of one value. The console imports that value directly:

```python
from ohho.train.run_clone import diff_run_configs, format_config_diff

diff = diff_run_configs("checkpoints/act", "checkpoints/act-clone")
text = format_config_diff(diff)
payload = diff.to_dict()
```

`--json` prints `payload` with two-space indentation. `payload` is `ohho.run.diff/v1`:

```json
{
  "schema": "ohho.run.diff/v1",
  "left": {"run": "/path/act", "manifest": "/path/act/run_manifest.json"},
  "right": {"run": "/path/act-clone", "manifest": "/path/act-clone/run_manifest.json"},
  "equal": false,
  "changes": [
    {"path": "hyperparameters.lr", "op": "replace", "before": 0.0001, "after": 0.0003}
  ]
}
```

`changes` is empty when `equal` is true. Each change has `path` and `op`:

| `op` | Meaning | Fields |
| --- | --- | --- |
| `replace` | Both sides have the path and the values differ | `before`, `after` |
| `add` | The path exists only on the right | `after` |
| `remove` | The path exists only on the left | `before` |

Paths match `--set` keys. Nested objects and lists are walked. A list index is a decimal segment (`eval_hooks.0.name`). A dotted key inside one object is not escaped. `before` and `after` are JSON values, so booleans stay booleans and `1` does not match `true`.

The text form prints the same changes, in the same order, after the run paths:

```text
ohho.run.diff/v1
left: /path/act
left_manifest: /path/act/run_manifest.json
right: /path/act-clone
right_manifest: /path/act-clone/run_manifest.json
equal: false
~ hyperparameters.lr: 0.0001 -> 0.0003
```

`+` is add, `-` is remove, `~` is replace. Values are compact JSON with object keys sorted. `equal: true` means there are no change lines.

## Artifact retention

Retention does not rewrite this file or `metrics.json`. Checkpoint deletion is recorded in `artifact_index.json` beside the manifest. See [artifacts.md](artifacts.md).
