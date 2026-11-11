# Knowledge distillation

An [`ohho.recipe/v1`](recipes.md) document can set `distillation`. The recipe family is the student. The teacher is a model reference: a registry model version, or a local checkpoint. Without the block, training is unchanged.

This build trains a tiny CPU regressor. It does not load SmolVLA, Diffusion Policy, OpenVLA, or the TinyACT transformer as the student or the teacher. Those runs need a GPU and are not executed here.

## Student

The student is any imitation-learning recipe.

| Family | In this build |
| --- | --- |
| `act` | Trains the tiny student. ACT hyperparameters are still checked. `d_model` is the student hidden size (default 8). `train_steps`, `batch_size`, `lr`, and `seed` are used. |
| `diffusion`, `smolvla`, `openvla` | IL families, and no trainer is registered. The run exits 3 before the teacher is resolved. |
| `ppo` and other non-IL families | Exit 1. The student must be an imitation-learning recipe. |
| A plugin family | A student when `metadata()` sets `imitation` to true and `available()` is true. Otherwise exit 1, or exit 3 when it is IL but unavailable. |

The IL check happens before the availability check. A non-IL family exits 1 even when it also has no trainer.

The checkpoint `config.json` records `"policy": "tiny-student"` and the recipe family. `base_weights` is recorded and is not loaded. Declared hooks are recorded and are not executed.

## Teacher

Set exactly one of `teacher.checkpoint` and `teacher.registry`.

`teacher.checkpoint` is a local directory, or the `policy.pt` inside it. `config.json` must say `"policy": "tiny-teacher"` and the directory must contain `policy.pt`. `save_tiny_teacher` in `ohho.train.compression.distillation` writes that pair.

`teacher.version` is optional. When it is set, that string is the teacher version. When it is omitted, the version is `sha256:` plus the sha256 of `policy.pt`.

`teacher.registry` is `{model, version}`. A process installs a client with `set_model_registry`. The client is any object with `resolve(model, version) -> {"checkpoint": <local path>, "version": <canonical version>}`. The returned checkpoint must already be on disk. This build does not download weights, and it does not ship a registry client. With no client, the run exits 1 and names the model version. The version recorded in the manifest is the client's `version` when it returns one, otherwise the requested version.

A checkpoint whose policy is not `tiny-teacher` (for example a SmolVLA directory) exits 1. The error names the checkpoint and says that distilling a full model such as SmolVLA into ACT needs a GPU. No fallback teacher is loaded.

## Loss

`loss` is `action_regression` (default) or `kl`.

| Term | Definition |
| --- | --- |
| Supervised | Mean squared error between student actions and dataset actions. |
| `action_regression` | Mean squared error between student actions and teacher actions. Temperature is recorded and is not applied. |
| `kl` | `T^2` times KL of `softmax(student / T)` against `softmax(teacher / T)`, over the action vector. `temperature` must be greater than 0. The default is 1. |
| Feature matching | Optional. Mean squared error between student and teacher hidden features, multiplied by `feature_weight` (default 1). |

`mix_weight` is from 0 to 1, default 0.5:

```text
total = (1 - mix_weight) * supervised + mix_weight * teacher_loss
```

Feature matching is added after that mix. It requires `hyperparameters.d_model` (or the default 8) to equal the teacher hidden size.

## Teacher-output cache

Teacher actions and features are stored under `distillation.cache_dir`. The default is `~/.ohho/cache/teacher-outputs`, which is shared by every trial on the machine. The file name is the sha256 of the teacher version, a NUL byte, and the dataset version.

The dataset version is `version:<dataset.version_id>` when the recipe sets a version id. A local dataset, which is what training requires in this build, uses `sha256:` plus the directory hash. Do not write run artifacts inside the dataset directory. That changes the hash and misses the cache.

A later trial with the same teacher version and dataset version reads the file and does not construct the teacher or run its forward. A different version, a different dataset hash, a different frame count, or an unreadable file is a miss. The teacher forward that fills a miss runs on CPU. The student still trains on the recipe device.

`metrics.json` records `cache_reused`, `teacher_forward_passes` (1 on a miss, 0 on a hit), `teacher_version`, `dataset_version`, and `cache_key`.

## Manifest

The student [`ohho.run/v1`](run-manifest.md) manifest sets `lineage.teacher`:

| Field | Meaning |
| --- | --- |
| `version` | Registry version, explicit checkpoint version, or `sha256:` of `policy.pt`. |
| `source` | `registry` or `checkpoint`. |
| `model` | Registry model id, or null for a checkpoint reference. |
| `checkpoint` | Local directory that was resolved. |

Runs that do not distill omit `lineage`. `ohho run show` prints `lineage.teacher.version` when it is present.

## Follow-up

These are not run in this tree:

- A real SmolVLA (or other full VLA) teacher distilled into ACT, on a GPU.
- Training the family's own architecture as the student instead of the tiny regressor.
- A console editor for the `distillation` block and the lineage field.
- TensorRT, or any other compiled student.
