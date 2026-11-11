# Deterministic training and re-run comparison

`deterministic` is an optional boolean on `ohho.recipe/v1`. The default is `false`, which keeps the current training loop. `true` is a recipe flag, not a new family and not a precision mode.

```bash
ohho train ohho/train/examples/il/act-tiny-cpu.yaml \
  --set dataset.path=/path/to/lerobot-dataset \
  --set output.dir=/path/to/run \
  --set deterministic=true
ohho run rerun /path/to/run
ohho run rerun /path/to/run --json
```

## What `deterministic: true` does

ACT and Diffusion Policy honor the flag. Other families still accept it on the recipe. This build does not train them, so the flag is only stored on `resolved_config` if a run starts.

1. **Seeded batch order.** Windows stay in episode order for the train/validation split. A shuffled `DataLoader` draws its permutation from a `torch.Generator` seeded with `hyperparameters.seed`. When the flag is true and that seed is omitted, the runner sets the seed to `0` and the manifest records it.
2. **Fixed DataLoader workers.** `num_workers` is `0` (`ohho.train.determinism.FIXED_DATALOADER_WORKERS`). Worker processes are not started, so a worker's own RNG cannot change the batch order.
3. **Torch and cuDNN, when available.** The trainer calls `torch.use_deterministic_algorithms(True)` when that function exists. If the strict call raises, it tries `warn_only=True`. When cuDNN is present it sets `deterministic` and turns `benchmark` off. TF32, the non-math scaled-dot-product kernels, and MKLDNN are turned off when those switches exist. Flags are restored when the trainer returns. ACT also disables the transformer nested-tensor fast path for that run. A knob the installed torch does not have is skipped.

`train_start` includes `"deterministic": true` or `false`. `config.json` and `metrics.json` record the same boolean.

CUDA reruns that must match bit for bit also need `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment before the process starts. This package does not set that variable. The CPU contract below does not use it.

## Re-run

```bash
ohho run rerun <id> [--output DIR] [--json]
```

`<id>` is the run directory, or the path of `run_manifest.json`. Those are the same strings `ohho run show` accepts. The command reads the manifest, trains `resolved_config` into a new directory, and compares final metrics with the original. The original directory is not modified. The dataset path recorded on the manifest is used when that directory still exists.

The default output is a sibling of the original run (`<name>-rerun`, then `<name>-rerun-2`, and so on). `--output` chooses the directory. It must be empty or absent, and it must not be the original run directory.

| Exit | Meaning |
| --- | --- |
| 0 | The re-run finished and the compared final metrics match exactly. |
| 1 | The manifest is missing or invalid, the run did not succeed, the output directory was refused, or the final metrics differ. |
| 2 | The command line is wrong. |
| 3, 4, 5 | The re-run's trainer returned that `ohho train` exit code. |

`--json` prints one `ohho.run.rerun/v1` object. The human lines are the same fields: `original_final_loss`, `rerun_final_loss`, `abs_delta`, `exact`, `expected_variance`, `tolerance`, `within_noise`.

`exact` is true only when the re-run succeeded and the shared final metrics match: `final_loss`, `validation_loss`, `loss_curve`, `validation_curve`, `completed_steps`, `train_steps`, and `stopped_early`, for each key both files have. `within_noise` is only the `final_loss` hint below. A gap inside the tolerance can still be `exact: false`.

## Expected variance

These figures are published data. Read them with `ohho.train.variance.recipe_noise`, `all_recipe_noise`, `within_noise`, and `comparison_hint`. Deterministic mode publishes `0` because the CPU trainers are required to match exactly. The default mode publishes `1e-4` as the expected variance (the grain `final_loss` is recorded at) and `1e-3` as the absolute band a comparison can treat as within noise. That default pair is a hint, not a statistic computed from a sample of runs. Recipes with no trainer in this build publish no number.

| Recipe | Metric | Deterministic expected variance | Deterministic tolerance | Default expected variance | Default tolerance |
| --- | --- | --- | --- | --- | --- |
| `act` | `final_loss` | 0 | 0 | 1e-4 | 1e-3 |
| `diffusion` | `final_loss` | 0 | 0 | 1e-4 | 1e-3 |
| `smolvla` | `final_loss` | n/a | n/a | n/a | n/a |
| `openvla` | `final_loss` | n/a | n/a | n/a | n/a |
| `ppo` | `final_loss` | n/a | n/a | n/a | n/a |

```python
from ohho.train.variance import comparison_hint

hint = comparison_hint("act", original_loss, rerun_loss, deterministic=True)
hint["within_noise"]
hint["tolerance"]
```

A console comparison view can call comparison_hint and read tolerance to show a within-noise hint. That view is not in this build.

The CI check for this contract re-runs `ohho/train/examples/il/act-tiny-cpu.yaml` on CPU with `deterministic: true` and requires the final loss to match exactly. It skips when torch is not installed. Ubuntu CI with Python 3.12 installs a CPU torch wheel before the unit tests, so the check runs there. `lerobot` is required only to download an `hf://` dataset, not for that local CPU re-run.
