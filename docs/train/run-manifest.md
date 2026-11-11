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
| `pruning` | Pruning | `pruning` |
| `error` | Error | `error` |
| `resolved_config` | Resolved config | `resolved_config` |

`gpu` is null when no CUDA device is reported. When a device is reported the value is `{name, driver, cuda}`. `container_image_digest` is null when the process is not inside a container. `dataset.hash` is null when the recipe names only a version id, or the local path is not present. `base_weights.hash` is null when the uri is not a local file or directory. Remote URIs are stored and are not downloaded.

`seeds.recipe` is `hyperparameters.seed`. When that integer is set, the runner also seeds Python's `random`, and NumPy and torch when those libraries import. A null library seed means that library was not seeded. Other hyperparameters whose names end in `_seed` are listed under `seeds.extra`. `PYTHONHASHSEED` is recorded and is not changed after the process has started.

`git` describes the checkout that contains the recipe file. `commit` is null when that directory is not a git checkout or the checkout has no commit yet. `dirty` is true when `git status --porcelain` shows any change, including untracked files. When git cannot read the status of a checkout that does exist, `dirty` is true.

`library_versions` always includes `python` and `ohho` (the `ohho-os` version). `torch`, `numpy`, and `lerobot` are included when those distributions are installed. The version lookup uses package metadata.

`pruning` is null when the recipe has no pruning block. A manifest that omits the key means the same thing and still validates. When a block was set, the value is the pruning report: method, schedule, target, per-layer sparsity, achieved sparsity, and, for a structured shrink, parameter counts and exported file sizes before and after. The human summary line prints method, schedule, target, achieved sparsity, parameter counts, and export sizes. Per-layer rows stay in the JSON. Details are in [Pruning](pruning.md).

`precision` is the recipe value the device check accepted (`fp32`, `fp16`, `bf16`, or `fp8`). `loss_scaling` is true only for fp16, where the step uses GradScaler. `precision_backend` is `fp32`, `torch.amp` (fp16 and bf16), or `transformer_engine` (fp8). `precision_metrics` has one key per precision. The precision that ran holds `throughput_steps_per_s` and `peak_memory_bytes`. The others are null. While the run is still going, and when the trainer did not write those keys, the numbers are null. Throughput is training steps per second over the step loop. On CUDA, peak memory is `max_memory_allocated` in bytes after the loop starts. On CPU and MPS it is the process resident set sampled during the loop, which includes the interpreter. Details are in [Mixed precision](precision.md).

The JSON document also carries `recipe_path`, `overrides`, `checkpoint_dir`, `metrics_path`, `declared_hooks`, and `git.root`. Those are available to a details panel. They are not summary fields. `--json` prints them with the summary.

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

## Artifact retention

Retention does not rewrite this file or `metrics.json`. Checkpoint deletion is recorded in `artifact_index.json` beside the manifest. See [artifacts.md](artifacts.md).
