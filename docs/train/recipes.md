# Training recipes (`ohho.recipe/v1`)

`ohho train` reads one recipe file and runs it on this machine. The same document is what a UI, a sweep, or a job worker submits. Families that do not have a trainer in this build are still valid recipes. Their run stops before training and does not invent a result.

The schema file is package data:

`ohho/train/schema/ohho.recipe.v1.json`

Its `$id` is `https://ohho-robotics.com/schemas/ohho.recipe/v1.json` and its title is `ohho.recipe/v1`. Shipped examples, one per family, live in `ohho/train/examples/`:

| File | Family | Trainer in this build |
| --- | --- | --- |
| `act.yaml` | `act` | Yes. Calls `ohho.train.act.train_act` on CPU or GPU. |
| `diffusion.yaml` | `diffusion` | Yes. Calls `ohho.train.il.diffusion_policy.train_diffusion`. |
| `smolvla.yaml` | `smolvla` | Preflight only, then exit 3. Does not download weights. |
| `openvla.yaml` | `openvla` | Preflight only, then exit 3. Does not download weights. |
| `ppo.yaml` | `ppo` | No. |

```bash
ohho train ohho/train/examples/il/act-tiny-cpu.yaml --set dataset.path=/path/to/dataset --set output.dir=/path/to/run
ohho train ohho/train/examples/il/diffusion-tiny-cpu.yaml --set dataset.path=/path/to/dataset --set output.dir=/path/to/run
```

ACT and Diffusion Policy parameters, defaults, ranges, validation events, and the checkpoint layout serve loads are in [`il-recipes.md`](il-recipes.md). Tiny CPU configs and single-GPU examples live in `ohho/train/examples/il/`. SmolVLA and OpenVLA parameters, the memory estimate, LoRA merge export, and base-model licence are in [`vla-recipes.md`](vla-recipes.md). Launch examples live in `ohho/train/examples/vla/`. The files in `ohho/train/examples/*.yaml` are one schema example per family. Running `diffusion.yaml` as shipped exits `4` until `dataset.path` is a real directory or an `hf://` hub id. `smolvla` and `openvla` are schema-valid, check their family parameters, and exit `3` without downloading weights. `ppo` is schema-valid and exits `3`. They do not train.

## Document

A recipe is a YAML mapping, or JSON when the file name ends in `.json`. The loader accepts indented mappings and lists, scalars, comments, and inline `[]` / `{}`. Other YAML needs the `yaml` extra (`pip install 'ohho-os[yaml]'`), which is used only for syntax this subset rejects.

Required fields are `schema`, `family`, `dataset`, and `output`. `schema` is the constant `ohho.recipe/v1`.

| Field | Meaning |
| --- | --- |
| `family` | Registry id. Built-ins: `act`, `diffusion`, `smolvla`, `openvla`, `ppo`. A plugin may register another id matching `^[a-z][a-z0-9_-]{0,63}$`. |
| `base_weights` | Optional `{uri, revision}`. The ACT trainer records the uri and does not load it. |
| `dataset` | Exactly one of `path` (local directory) or `version_id`. |
| `hyperparameters` | Object. Known keys are typed. Other keys are allowed so a plugin can read them. |
| `precision` | `fp32` (default), `fp16`, or `bf16`. ACT and diffusion run `fp32` only. SmolVLA and OpenVLA store BF16 on `hyperparameters.precision` and do not read this field. OHH-524 will unify the two. |
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
4. Look up `family`. A missing family, or one whose `available()` is false (`ppo`), stops here.
5. Family preflight. ACT and diffusion check the shared precision field, hyperparameters, dataset, device, and torch. SmolVLA and OpenVLA check family hyperparameters (BF16 is `hyperparameters.precision`; the shared precision field is not read) and build a memory estimate, then stop. They do not download weights and do not emit `train_start`.
6. Emit `train_start` and call the trainer. SmolVLA and OpenVLA do not reach this step.

A failure in steps 1–5 does not call the trainer and does not emit `train_start`. Schema validation happens before the availability check, so an invalid field on an unavailable family is exit `1`, not exit `3`.

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

ACT and diffusion hyperparameters, their defaults, and their ranges are listed in [`il-recipes.md`](il-recipes.md). The family checks them before training. `hyperparameters.mock` is rejected. When `train_steps` is omitted and `num_epochs` is set, the step count is `num_epochs * 2`. When both are set, `train_steps` is used. When neither is set, the step count is 200. For ACT, `d_model` must be divisible by `nhead` (defaults `64` and `2` when omitted). `seed` is applied with `torch.manual_seed` and is not forwarded to the optimizer constructor. ACT and diffusion do not run a fake trainer.

## What a successful ACT run writes

`train_act` writes the checkpoint directory (`policy.pt`, `config.json`, `metrics.json`). `ohho train` then writes `run_manifest.json` in that same directory. `config.json` has `"mock": false` for this command.

The manifest is a small JSON object with `"schema": "ohho.train.run_manifest/v0"`. It records the resolved recipe, the overrides, the checkpoint directory, the metrics path, `final_loss`, `base_weights_loaded`, the declared hook names, and UTC start and finish timestamps. It is not an `ohho.run/v1` manifest. `base_weights_loaded` is false for ACT in this build, including when `base_weights.uri` is set. Hook names are listed under `declared_hooks`. The runner does not execute them. The progress stream says `status: "skipped"` for each one.

## Progress stream

Stdout is one JSON object per line and nothing else. Every object includes `"stream": "ohho.train.progress/v1"`. Stderr repeats the human-readable error text. Workers should parse stdout and use the process exit code.

| `event` | When |
| --- | --- |
| `error` | The process is failing. Fields: `exit_code`, `message`, and when relevant `family` and `errors` (a list of strings). |
| `base_weights` | ACT, after preflight, when `base_weights` is set. `status` is `not_loaded`. |
| `hook` | ACT, one object per declared hook, before training. `status` is `skipped`. |
| `train_start` | Immediately before the trainer. Fields: `family`, `device`, `train_steps`, and `val_every` for ACT and diffusion. |
| `validation` | During ACT or diffusion training, after every `val_every` optimizer step and after the final step. Fields: `family`, `step` (completed updates, starting at 1), `validation_loss`, `best_validation_loss`, `train_loss`. |
| `early_stop` | Patience stopped ACT or diffusion before `train_steps`. Fields: `family`, `step`, `validation_loss`, `best_validation_loss`, `patience`. The process still exits 0 when the checkpoint is written. |
| `metrics` | After a successful train. Fields: `path`, `final_loss`. |
| `checkpoint` | Checkpoint directory. |
| `manifest` | `run_manifest.json` path. |
| `done` | Success. Fields: `exit_code` 0, `checkpoint_dir`, `manifest`. |

A run that never starts training emits a single `error` line and no `train_start`. A trainer that raises after `train_start` emits `error` with exit code `5` and does not emit `done`. A missing `run_manifest.json` means the run did not succeed. Argparse usage errors (for example, no recipe path) write no JSON line.

```json
{"stream": "ohho.train.progress/v1", "event": "error", "exit_code": 3, "message": "family not available in this build", "family": "smolvla"}
```

## Exit codes

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | ok | Checkpoint, metrics, and `run_manifest.json` were written. |
| 1 | invalid | Parse error, `--set` error, schema error, ACT/diffusion preflight refusal (unsupported precision, unknown hyperparameter, value outside its range, `d_model` not divisible by `nhead`, `vision_backbone` other than `none`, device not available, torch missing), or a SmolVLA/OpenVLA hyperparameter outside its range. The trainer was not called. |
| 2 | usage | The process was invoked incorrectly. This is argparse (the recipe path was missing). No JSON line. |
| 3 | unavailable | PPO has no trainer. A schema-valid SmolVLA or OpenVLA recipe whose family parameters are in range also exits 3: this build does not download base weights or launch the fine-tune. `message` is `family not available in this build`. The trainer was not called. |
| 4 | dataset | The family can train, but the dataset is not a local directory (`version_id` only, or `path` is not a directory). The trainer was not called. |
| 5 | failed | `train_start` was emitted and the trainer then failed. No success manifest is written. |

## Adding a family

Register an object with `id`, `available()`, and `run(recipe, emit)`. `available()` returns false when this build cannot train that family. `run` trains for real, writes `metrics.json`, and returns a `TrainResult`. Use `emit` for progress objects. Do not print to stdout. That stream is JSON lines. SmolVLA and OpenVLA set `available()` true so parameter preflight runs, then `run` raises `FamilyUnavailable` because this build does not load their base weights.

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
