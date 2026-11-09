# Training recipes (`ohho.recipe/v1`)

`ohho train` reads one recipe file and runs it on this machine. The same document is what a UI, a sweep, or a job worker submits. Families that do not have a trainer in this build are still valid recipes. Their run stops before training and does not invent a result.

Every run that reaches `train_start` writes an [`ohho.run/v1`](run-manifest.md) manifest next to its artifacts. `ohho run show <run-dir>` prints it. `ohho run clone <run-dir> --set k=v` writes a tweaked config from that manifest's resolved recipe, and `ohho run diff <a> <b>` compares two resolved configs. See [run-manifest.md](run-manifest.md).

The schema file is package data:

`ohho/train/schema/ohho.recipe.v1.json`

Its `$id` is `https://ohho-robotics.com/schemas/ohho.recipe/v1.json` and its title is `ohho.recipe/v1`. Shipped examples, one per family, live in `ohho/train/examples/`:

| File | Family | Trainer in this build |
| --- | --- | --- |
| `act.yaml` | `act` | Yes. Calls `ohho.train.act.train_act` on CPU or GPU. |
| `diffusion.yaml` | `diffusion` | No. |
| `smolvla.yaml` | `smolvla` | No. |
| `openvla.yaml` | `openvla` | No. |
| `ppo.yaml` | `ppo` | No. |

```bash
ohho train ohho/train/examples/act.yaml --set dataset.path=/path/to/dataset --set output.dir=/path/to/run
ohho train ohho/train/examples/diffusion.yaml
```

The diffusion command is schema-valid and exits `3`. It does not train.

## Document

A recipe is a YAML mapping, or JSON when the file name ends in `.json`. The loader accepts indented mappings and lists, scalars, comments, and inline `[]` / `{}`. Other YAML needs the `yaml` extra (`pip install 'ohho-os[yaml]'`), which is used only for syntax this subset rejects.

Required fields are `schema`, `family`, `dataset`, and `output`. `schema` is the constant `ohho.recipe/v1`.

| Field | Meaning |
| --- | --- |
| `family` | Registry id. Built-ins: `act`, `diffusion`, `smolvla`, `openvla`, `ppo`. A plugin may register another id matching `^[a-z][a-z0-9_-]{0,63}$`. |
| `base_weights` | Optional `{uri, revision}`. The ACT trainer records the uri and does not load it. |
| `dataset` | Exactly one of `path` (local directory) or `version_id`. |
| `hyperparameters` | Object. Known keys are typed. Other keys are allowed so a plugin can read them. |
| `precision` | `fp32` (default), `fp16`, or `bf16`. ACT runs `fp32` only. |
| `device` | `cpu` (default), `cuda`, `mps`, or `auto`. |
| `compression_hooks` | List of `{name, params}`. Recorded. Not executed. |
| `eval_hooks` | List of `{name, params}`. Recorded. Not executed. |
| `output.dir` | Directory for the checkpoint, `metrics.json`, and `run_manifest.json`. |

`dataset.version_id` is not resolved in this build. A family that has a trainer, and a recipe that sets only `version_id`, exits `4` before training.

Relative paths are resolved from the process working directory.

## Checks, in order

1. Parse the file.
2. Apply every `--set` override.
3. Validate against the published JSON Schema.
4. Look up `family`. A missing or unavailable family stops here.
5. Family preflight (for ACT: precision, hyperparameters, local dataset, device, torch).
6. On `train_start`, write `run_manifest.json` with `status: running`.
7. Call the trainer.
8. Finalise that file as `succeeded`, `failed`, or `interrupted`.

A failure in steps 1–5 does not call the trainer, does not emit `train_start`, and does not write a manifest. Schema validation happens before the availability check, so an invalid field on an unavailable family is exit `1`, not exit `3`.

## `--set`

Repeat the flag. The key is a dotted path. The value replaces that field. `null` deletes a mapping key.

```bash
ohho train recipe.yaml \
  --set dataset.path=/data/demos \
  --set output.dir=/runs/act-01 \
  --set hyperparameters.lr=0.0002 \
  --set hyperparameters.train_steps=2 \
  --set device=cpu
```

Parsing rules for the value:

| Text | Result |
| --- | --- |
| `null`, `Null`, `NULL`, `~` | delete the key |
| `true` / `false` | boolean |
| an integer or a float, including `1e-4` | number |
| `{...}` or `[...]` | JSON |
| `"..."` or `'...'` | string, so `"null"` stays the string null |
| anything else | string |

`--set dataset.path=null --set dataset.version_id=dv_123` switches the dataset field from a path to a version id. A numeric path segment indexes a list that already exists (`eval_hooks.0.params.episodes=9`). Unknown fields and wrong types fail schema validation and exit `1` before training.

ACT hyperparameters that are passed through to `train_act` are `train_steps`, `batch_size`, `lr`, `chunk_size`, `d_model`, `nhead`, `num_encoder_layers`, `num_decoder_layers`, `dim_feedforward`, `latent_dim`, `kl_weight`, and `log_every`. `seed` is applied with `torch.manual_seed` and is not forwarded. When `train_steps` is omitted and `num_epochs` is set, the step count is `num_epochs * 2`. When both are set, `train_steps` is used. When neither is set, ACT uses `200` steps. `d_model` must be divisible by `nhead` (defaults `64` and `2` when omitted). `hyperparameters.mock` is rejected. ACT does not run a fake trainer.

## What a successful ACT run writes

`train_act` writes the checkpoint directory (`policy.pt`, `config.json`, `metrics.json`). `ohho train` writes `run_manifest.json` in the output directory when ACT emits `train_start`, then finalises that file in the checkpoint directory when training ends. `config.json` has `"mock": false` for this command.

The manifest is an [`ohho.run/v1`](run-manifest.md) document. It records the resolved recipe, every seed, the git commit and dirty flag for the checkout that contains the recipe, the dataset version id and content hash, the base-weights hash, the container image digest, library versions, GPU name / driver / CUDA, UTC start and finish times, and the status. `base_weights.loaded` is false for ACT in this build, including when `base_weights.uri` is set. Remote weight URIs are not downloaded, so their hash stays null. Hook names are listed under `declared_hooks`. The runner does not execute them. The progress stream says `status: "skipped"` for each one.

## Progress stream

Stdout is one JSON object per line and nothing else. Every object includes `"stream": "ohho.train.progress/v1"`. Stderr repeats the human-readable error text. Workers should parse stdout and use the process exit code.

| `event` | When |
| --- | --- |
| `error` | The process is failing. Fields: `exit_code`, `message`, and when relevant `family` and `errors` (a list of strings). |
| `base_weights` | ACT, after preflight, when `base_weights` is set. `status` is `not_loaded`. |
| `hook` | ACT, one object per declared hook, before training. `status` is `skipped`. |
| `train_start` | Immediately before `train_act`. Fields: `family`, `device`, `train_steps`. |
| `metrics` | After a successful train. Fields: `path`, `final_loss`. |
| `checkpoint` | Checkpoint directory. |
| `manifest` | `run_manifest.json` path. `status` is `running` at start and `succeeded`, `failed`, or `interrupted` when the file is finalised. |
| `done` | Success. Fields: `exit_code` 0, `checkpoint_dir`, `manifest`. |

A run that never starts training emits a single `error` line and no `train_start`. A trainer that raises after `train_start` emits `error` with exit code `5`, does not emit `done`, and leaves `run_manifest.json` with `status: failed`. An interrupt leaves `status: interrupted` and then propagates. A missing `run_manifest.json` means the run never started. Argparse usage errors (for example, no recipe path) write no JSON line.

```json
{"stream": "ohho.train.progress/v1", "event": "error", "exit_code": 3, "message": "family not available in this build", "family": "diffusion"}
```

## Exit codes

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | ok | Checkpoint, metrics, and a final `run_manifest.json` (`status: succeeded`) were written. |
| 1 | invalid | Parse error, `--set` error, schema error, or ACT preflight refusal (unsupported precision, unknown hyperparameter, `d_model` not divisible by `nhead`, device not available, torch missing). The trainer was not called. |
| 2 | usage | The process was invoked incorrectly. This is argparse (the recipe path was missing). No JSON line. |
| 3 | unavailable | The recipe is schema-valid and the family has no trainer in this build. `message` is `family not available in this build`. The trainer was not called. |
| 4 | dataset | The family can train, but the dataset is not a local directory (`version_id` only, or `path` is not a directory). The trainer was not called. |
| 5 | failed | `train_start` was emitted and the trainer then failed. The manifest is finalised with `status: failed`. |

## Adding a family

Register an object with `id`, `available()`, and `run(recipe, emit)`. `available()` returns false when this build cannot train that family. `run` trains for real, writes `metrics.json`, and returns a `TrainResult`. Emit `train_start` before training. The runner writes `run_manifest.json` on that event and finalises it when `run` returns or raises. The family does not write the manifest. Use `emit` for progress objects. Do not print to stdout. That stream is JSON lines.

```python
from ohho.train.recipe_registry import TrainResult, register

class MyFamily:
    id = "my_family"

    def available(self) -> bool:
        return True

    def run(self, recipe, emit):
        emit({"event": "train_start", "family": self.id, "device": "cpu"})
        # Train, then write metrics.json under recipe["output"]["dir"].
        raise NotImplementedError

register(MyFamily(), source="my_package")
```

An installed distribution can use the `ohho.recipes` entry point instead of calling `register`. The value is the object, or a class with no required constructor arguments. A second implementation of an id that is already registered is refused, so a plugin cannot replace `act`.

```toml
[project.entry-points."ohho.recipes"]
my_family = "my_package.recipes:FAMILY"
```

The built-in objects are `ohho.train.builtin_recipes:ACT`, `DIFFUSION`, `SMOLVLA`, `OPENVLA`, and `PPO`. They are registered even when the distribution's entry points are not installed.

From Python, `ohho.train.run_cli.execute(path, ["hyperparameters.lr=0.001"])` is the same function the `ohho train` command calls.
