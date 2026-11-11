# ACT and Diffusion Policy recipes

`ohho train` runs these two families on a LeRobot-format dataset. The dataset is a local directory, or a Hugging Face Hub id written as `hf://org/name` (a bare `org/name` is also a Hub id when that path is not an existing directory and the first segment is not a workspace folder such as `datasets`). `dataset.version_id` is not resolved.

Tiny CPU configs and single-GPU examples:

| File | Family | What it is |
| --- | --- | --- |
| `ohho/train/examples/il/act-tiny-cpu.yaml` | `act` | State-only ACT, `device: cpu`, two steps. This is the CI-sized config. |
| `ohho/train/examples/il/diffusion-tiny-cpu.yaml` | `diffusion` | State-only diffusion, `device: cpu`, two steps, four diffusion steps. |
| `ohho/train/examples/il/act-gpu.yaml` | `act` | Single-GPU document: `device: cuda`, chunk size 100, `vision_backbone: resnet18`, Hub dataset `hf://lerobot/aloha_sim_insertion_human`. |
| `ohho/train/examples/il/diffusion-gpu.yaml` | `diffusion` | Single-GPU document: `device: cuda`, DDIM, 100 diffusion steps, `vision_backbone: resnet18`, Hub dataset `hf://lerobot/pusht`. |

Replace `dataset.path` before a real run. The tiny files ship with the placeholder `datasets/sim-loop`.

```bash
ohho train ohho/train/examples/il/act-tiny-cpu.yaml \
  --set dataset.path=/path/to/lerobot-dataset \
  --set output.dir=/path/to/run
ohho train ohho/train/examples/il/diffusion-tiny-cpu.yaml \
  --set dataset.path=/path/to/lerobot-dataset \
  --set output.dir=/path/to/run
```

The trainer reads LeRobot v2 layout through `DatasetReader`: `meta/info.json`, `meta/episodes.jsonl`, and `data/chunk-000/episode_*.parquet` or `.jsonl`. A Hub id is downloaded with `lerobot` into that layout before the first training step. Without `lerobot`, preflight exits `4` and does not train. Installing torch is `pip install 'ohho-os[train]'`.

## Vision backbone

`vision_backbone` validates as `none`, `resnet18`, or `resnet34`. This build trains `none` only: a state-and-action policy with no ResNet stem. `resnet18` and `resnet34` are legal names, and the single-GPU examples record `resnet18`, but preflight then exits `1` before `train_start` with `not trainable`. To train the shipped policy on a GPU, set `device: cuda` and `vision_backbone: none`.

`beta_start` (`0.0001`) and `beta_end` (`0.02`) are fixed diffusion constants. They are stored in the checkpoint and are not recipe fields.

## ACT parameters

| Parameter | Default | Range |
| --- | --- | --- |
| `train_steps` | 200 | integer in [1, 1000000] |
| `num_epochs` | unset | integer in [1, 1000000]; if train_steps is omitted, steps = num_epochs * 2 |
| `batch_size` | 8 | integer in [1, 1024] |
| `lr` | 0.0001 | float in (0, 1] |
| `vision_backbone` | none | one of none, resnet18, resnet34 |
| `val_every` | 50 | integer in [1, 1000000] |
| `val_fraction` | 0.2 | float in (0, 0.5] |
| `early_stop_patience` | 0 | integer in [0, 100000]; 0 disables early stopping |
| `early_stop_min_delta` | 0.0 | float in [0, 1e12] |
| `seed` | unset | integer in [-2147483648, 2147483647] |
| `log_every` | 25 | integer in [1, 1000000] |
| `chunk_size` | 10 | integer in [1, 1024] |
| `d_model` | 64 | integer in [1, 1024], and divisible by nhead |
| `nhead` | 2 | integer in [1, 16] |
| `num_encoder_layers` | 1 | integer in [1, 12] |
| `num_decoder_layers` | 1 | integer in [1, 12] |
| `dim_feedforward` | 128 | integer in [1, 4096] |
| `latent_dim` | 16 | integer in [1, 256] |
| `kl_weight` | 10.0 | float in [0, 1000000] |

## Diffusion Policy parameters

`horizon` is an alias of `chunk_size`. If both are set they must be equal. If only one is set, the other takes that value. If neither is set, both are 16.

`num_inference_steps` is the DDIM stride count. When it is omitted, the value is `min(10, diffusion_steps)`. `scheduler: ddpm` denoises every training timestep. `scheduler: ddim` denoises `num_inference_steps` evenly spaced timesteps. Training loss is noise-prediction MSE for both.

| Parameter | Default | Range |
| --- | --- | --- |
| `train_steps` | 200 | integer in [1, 1000000] |
| `num_epochs` | unset | integer in [1, 1000000]; if train_steps is omitted, steps = num_epochs * 2 |
| `batch_size` | 8 | integer in [1, 1024] |
| `lr` | 0.0001 | float in (0, 1] |
| `vision_backbone` | none | one of none, resnet18, resnet34 |
| `val_every` | 50 | integer in [1, 1000000] |
| `val_fraction` | 0.2 | float in (0, 0.5] |
| `early_stop_patience` | 0 | integer in [0, 100000]; 0 disables early stopping |
| `early_stop_min_delta` | 0.0 | float in [0, 1e12] |
| `seed` | unset | integer in [-2147483648, 2147483647] |
| `log_every` | 25 | integer in [1, 1000000] |
| `chunk_size` | 16 | integer in [1, 1024] |
| `horizon` | 16 | integer in [1, 1024]; alias of chunk_size; both must match when set |
| `diffusion_steps` | 100 | integer in [1, 1000] |
| `scheduler` | ddpm | one of ddpm, ddim |
| `num_inference_steps` | 10 | integer in [1, diffusion_steps]; omitted values use min(10, diffusion_steps) |
| `d_model` | 64 | integer in [8, 1024] |

Unknown keys and `hyperparameters.mock` are rejected before training. A value outside the range above is exit `1` and does not emit `train_start`.

## Validation stream and early stopping

Stdout remains `ohho.train.progress/v1`, one JSON object per line. After each completed optimizer step that is a multiple of `val_every`, and after the final step when that step is not a multiple, the trainer emits:

```json
{"stream": "ohho.train.progress/v1", "event": "validation", "family": "act", "step": 2, "validation_loss": 0.42, "best_validation_loss": 0.42, "train_loss": 0.5}
```

`step` is the number of optimizer updates completed, starting at 1. The last `val_fraction` of windows (at least one, and never the whole set when there are two or more windows) is the validation split. `early_stop_patience` of 0 disables early stopping. Otherwise training stops when validation loss fails to beat the best value by more than `early_stop_min_delta` for that many validation reports. The stop is also a progress event:

```json
{"stream": "ohho.train.progress/v1", "event": "early_stop", "family": "diffusion", "step": 4, "validation_loss": 0.4, "best_validation_loss": 0.2, "patience": 2}
```

An early stop still writes the checkpoint and exits 0.

## Checkpoint layout loaded by serve

Both families write the same directory shape. `ohho train` then adds `run_manifest.json` (the OHH-507 manifest, not `ohho.run/v1`).

| File | ACT | Diffusion Policy |
| --- | --- | --- |
| `config.json` | `policy` is `act`. Also `state_dim`, `action_dim`, `chunk_size`, `d_model`, `nhead`, `num_encoder_layers`, `num_decoder_layers`, `dim_feedforward`, `latent_dim`, `kl_weight`, `mock: false`. | `policy`, `model_type`, and `model_family` are `diffusion`. Also `state_dim`, `action_dim`, `chunk_size`, `diffusion_steps`, `scheduler`, `num_inference_steps`, `d_model`, `beta_start`, `beta_end`, `mock: false`. |
| `policy.pt` | `TinyACTPolicy.state_dict`. | `TinyDiffusionPolicy.state_dict`. |
| `metrics.json` | `final_loss`, `loss_curve`, `train_steps`, `completed_steps`, `stopped_early`, and `validation_loss` / `validation_curve` when validation ran. | Same metric keys. |

Serve loads weights with `torch.load(..., weights_only=True)`.

ACT is selected with no `model_class` when `config.json` has `policy`, `model_family`, or `model_type` equal to `act` (or `model_type` `tiny_act`). `POST /load_model` and `build_app(..., auto_load=True)` then use `ohho.serve.act.ACTModel`, which reads `policy.pt` and `config.json` and returns `action.vector` from `POST /predict`.

A diffusion directory does not match that marker, so serve does not route it through `ACTModel`. Pass the dotted class serve already imports:

`model_class=ohho.train.il.diffusion_policy.DiffusionPolicyModel`

`DiffusionPolicyModel.load_model` reads the same two files and `predict_action` returns `{"vector": [...]}` of length `action_dim` (the first action in the chunk). No serve module is edited for that path.
