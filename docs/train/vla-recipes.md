# SmolVLA and OpenVLA fine-tune recipes

`ohho train` checks these two families and then stops. A schema-valid recipe whose parameters are in range exits `3` with `family not available in this build`. It does not download base weights, does not emit `train_start`, and does not write a checkpoint. An out-of-range parameter exits `1` before that. PPO is unchanged and still exits `3` with no parameter check of its own.

BF16 is the family default on `hyperparameters.precision`. The shared recipe `precision` field is left as it is. OHH-524 will unify the two. This family does not read the shared field, so the shipped `openvla.yaml` value `precision: fp16` does not select the compute dtype. The family dtype stays `bf16` until `hyperparameters.precision` is set.

Launch documents (replace `dataset.path` before a real run):

| File | Family | What it is |
| --- | --- | --- |
| `ohho/train/examples/vla/smolvla-24gb.yaml` | `smolvla` | Full fine-tune, `device: cuda`, batch 8, BF16, gradient checkpointing on. The launch class is one 24 GB GPU. |
| `ohho/train/examples/vla/openvla-lora-80gb.yaml` | `openvla` | LoRA, `device: cuda`, batch 16, rank 32, `all-linear`, BF16. The launch class is one single A100 80GB. |
| `ohho/train/examples/vla/openvla-full.yaml` | `openvla` | `tune_mode: full`. Full OpenVLA fine-tune is multi-GPU/cloud only. |

The 24 GB figure is the SmolVLA default launch class. The OpenVLA LoRA class is a single A100 80GB, which is the GPU in the upstream LoRA example (batch 16, rank 32, about 72 GB). Upstream also says a smaller GPU can run LoRA when it has at least about 27 GB and the batch is reduced. Full fine-tune is the upstream FSDP path on a node of 8 A100 80GB.

Calling `ohho train` on these files does not confirm any of that on a GPU. The checks below run on CPU.

## Parameters

`train_steps` wins when both `train_steps` and `num_epochs` are set. When only `num_epochs` is set, the step count is `num_epochs * 2`. When neither is set, the step count is 200. The launch files above set `train_steps` to 20000 (SmolVLA) and 200000 (OpenVLA, the upstream `max_steps` default).

`lora_alpha` defaults to 16, which is upstream `min(rank, 16)` at the default rank of 32. Changing `lora_rank` does not change `lora_alpha`. `lora_target_modules` defaults to `all-linear`. A comma-separated string or a list of names selects specific modules. `grad_accumulation_steps` is the number of microbatches; the memory estimate uses `batch_size` and does not multiply by the accumulation count.

Unknown keys and `hyperparameters.mock` are rejected. A value outside the range below is exit `1` and does not emit `train_start`.

### SmolVLA

Default `tune_mode` is `full`. LoRA is the same parameter set with `tune_mode: lora`.

| Parameter | Default | Range |
| --- | --- | --- |
| `train_steps` | 200 | integer in [1, 1000000] |
| `num_epochs` | unset | integer in [1, 1000000]; if train_steps is omitted, steps = num_epochs * 2 |
| `batch_size` | 8 | integer in [1, 256] |
| `grad_accumulation_steps` | 1 | integer in [1, 1024] |
| `lr` | 0.0001 | float in (0, 1] |
| `precision` | bf16 | one of bf16, fp16, fp32 |
| `gradient_checkpointing` | true | boolean |
| `tune_mode` | full | one of lora, full |
| `lora_rank` | 32 | integer in [1, 256] |
| `lora_alpha` | 16 | integer in [1, 1024] |
| `lora_dropout` | 0.0 | float in [0, 0.5] |
| `lora_target_modules` | all-linear | all-linear, or 1 to 32 names (letters, digits, underscore; each at most 64 characters) |
| `warmup_steps` | 0 | integer in [0, 1000000] |
| `max_grad_norm` | 1.0 | float in (0, 100] |
| `seed` | unset | integer in [-2147483648, 2147483647] |
| `log_every` | 25 | integer in [1, 1000000] |

### OpenVLA

Default `tune_mode` is `lora`. `tune_mode: full` is a valid recipe for a multi-GPU job. This build still exits `3` and does not launch it.

| Parameter | Default | Range |
| --- | --- | --- |
| `train_steps` | 200 | integer in [1, 1000000] |
| `num_epochs` | unset | integer in [1, 1000000]; if train_steps is omitted, steps = num_epochs * 2 |
| `batch_size` | 16 | integer in [1, 256] |
| `grad_accumulation_steps` | 1 | integer in [1, 1024] |
| `lr` | 0.0005 | float in (0, 1] |
| `precision` | bf16 | one of bf16, fp16, fp32 |
| `gradient_checkpointing` | true | boolean |
| `tune_mode` | lora | one of lora, full |
| `lora_rank` | 32 | integer in [1, 256] |
| `lora_alpha` | 16 | integer in [1, 1024] |
| `lora_dropout` | 0.0 | float in [0, 0.5] |
| `lora_target_modules` | all-linear | all-linear, or 1 to 32 names (letters, digits, underscore; each at most 64 characters) |
| `warmup_steps` | 0 | integer in [0, 1000000] |
| `max_grad_norm` | 1.0 | float in (0, 100] |
| `seed` | unset | integer in [-2147483648, 2147483647] |
| `log_every` | 25 | integer in [1, 1000000] |

`fp16` is the same width as `bf16` in the memory estimate. `fp32` doubles weight, gradient, and activation bytes. AdamW moments stay fp32 (8 bytes per trainable parameter) in every dtype.

## Memory estimate

```python
from ohho.train.vla import estimate_memory

estimate = estimate_memory("smolvla", {"batch_size": 8, "tune_mode": "full"})
if estimate.warning:
    print(estimate.warning)
```

`estimate_memory(family, hyperparameters)` accepts a raw mapping or an already resolved one. `estimate.as_dict()` is the same fields for a UI. `warning` is empty when `total_gib` fits the launch class. `launch_limit_gib` is `24.0` for SmolVLA, `80.0` for OpenVLA LoRA, and null for full OpenVLA.

| Config | Class | `gpu_class` | What the estimate is for |
| --- | --- | --- | --- |
| SmolVLA, any `tune_mode` | one 24 GB GPU | `24gb` | Default full fine-tune (batch 8, BF16, gradient checkpointing on) is under 24 GiB. Batch 16, or checkpointing off, warns. |
| OpenVLA LoRA | single A100 80GB | `a100-80gb` | At rank 32, `all-linear`, BF16, and gradient checkpointing, the estimate is 24 GiB plus 3 GiB per sample: 72 GiB at batch 16 and 27 GiB at batch 1. Those two points are the OpenVLA README's published LoRA example and its published floor. They are not a measurement from this package. |
| OpenVLA full | multi-GPU/cloud only | `multi-gpu-cloud` | `fits_launch_class` is false. The warning says the fine-tune does not fit one GPU. Upstream runs it with FSDP on 8 A100 80GB. |

`recommended_gpu` is `single 24 GB GPU`, `single A100 80GB`, or `multi-GPU/cloud (8x A100 80GB, FSDP)`.

The parameter counts are planning constants: 450 million for `lerobot/smolvla_base`, and 7.5 billion for `openvla/openvla-7b` (the README's "7.5 billion parameters"). LoRA trainable size is 1.4% of that count at rank 32 with `all-linear`, scaled by rank, and scaled by `len(modules) / 7` for an explicit list. Weights are not loaded to count parameters.

`ohho train` does not print `warning`. Call `estimate_memory` before launch.

## Licence metadata and LoRA export

`checkpoint_metadata(family, hyperparameters, base_weights_uri="")` records the published base model. `prepare(family, recipe)` returns the resolved parameters, the memory estimate, and that metadata. Neither downloads weights.

| Family | `base_model` | `base_model_licence` | `base_model_card_licence` |
| --- | --- | --- | --- |
| `smolvla` | `lerobot/smolvla_base` | Apache-2.0 | Apache-2.0 |
| `openvla` | `openvla/openvla-7b` | Llama 2 Community License | MIT |

The OpenVLA README states that `openvla-7b` is derived from Llama-2 and those weights are subject to the Llama 2 Community License. The Hugging Face model card also lists MIT. Both strings are stored. `base_weights_uri` is the recipe uri (`local:smolvla-base` on the shipped schema example, or the published id on the launch files). It does not change the licence.

```python
from ohho.train.vla import merge_lora_for_serve

merge_lora_for_serve(
    model,
    "checkpoints/openvla-merged",
    family="openvla",
    hyperparameters={"tune_mode": "lora", "lora_rank": 32},
    base_weights_uri="openvla/openvla-7b",
)
```

`merge_lora_for_serve` writes:

| File | Contents |
| --- | --- |
| `config.json` | `policy` / `model_type` / `model_family` are the family id, so serve does not treat the directory as ACT. Also the resolved dtype, LoRA fields, `merged_lora`, `base_model`, `base_model_licence`, `base_model_card_licence`, `base_model_licence_note`, `base_weights_uri`, and `parameter_count`. `mock` is false. |
| `policy.pt` | Plain `state_dict`. For `tune_mode: lora`, a plain module is wrapped with the recipe LoRA config and `merge_and_unload` folds the adapter. A module that already has a PEFT adapter is merged as it stands. Adapter keys are not in the file. For `tune_mode: full`, the module state dict is written unchanged. |

Torch and, for a LoRA wrap, `peft` are imported inside the call. `pip install 'ohho-os[train]'` installs torch. `pip install peft` installs the adapter library. This function does not call `from_pretrained` and does not download a 7B or 450M checkpoint. The CPU test merges a tiny randomly initialised module when both libraries are installed, and skips when they are not.

## What a GPU has to confirm

This package does not run the fine-tune. A GPU job still has to confirm:

- SmolVLA at the 24 GB default (the `smolvla-24gb.yaml` batch, BF16, gradient checkpointing on) actually fits and trains.
- OpenVLA LoRA at batch 16 on a single A100 80GB matches the README's about-72 GB example closely enough for that card.
- Full OpenVLA fine-tune on a multi-GPU node (upstream: 8 A100 80GB, FSDP).
- A merged `policy.pt` from a real OpenVLA or SmolVLA adapter loads in the serve class you run in production. The CPU check only folds LoRA on a tiny random module.
