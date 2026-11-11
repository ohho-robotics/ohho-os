# Pruning

`ohho train` reads an optional `pruning` block from an [`ohho.recipe/v1`](recipes.md) document. Omit the block and the run stays dense. The block is sweepable with `--set`: `target_sparsity`, `schedule`, `include`, and `exclude` are ordinary fields.

```yaml
pruning:
  method: unstructured          # unstructured | structured_channel | structured_head | nm_2_4
  schedule: one_shot            # one_shot | gradual | iterative
  target_sparsity: 0.5          # 0 inclusive, 1 exclusive. nm_2_4 must be 0.5
  include: []                   # module-name patterns. empty selects every prunable layer
  exclude: ["action_head"]
  start_step: 0                 # gradual only
  end_step: 100                 # gradual only. omit to use the last optimizer step
  iterations: 3                 # iterative only. default 3
  finetune_steps: 20            # iterative only. run length becomes iterations * finetune_steps
```

`include` and `exclude` match module names. `encoder.*` matches `encoder.layers.0.linear1`. `encoder` matches that module and its children. `exclude` wins. A name that matches nothing fails the run after `train_start`.

## Methods

| Method | What is removed | Checkpoint |
| --- | --- | --- |
| `unstructured` | Smallest-magnitude weights, per tensor. Biases stay dense. | Same module shapes. Zeros stay in the file. |
| `structured_channel` | Output channels of a `Linear` or `Conv2d`, and the matching input channels of the next sibling layer. | Those modules are rebuilt. Parameter count and `policy.pt` get smaller. |
| `structured_head` | Whole attention heads. | Each `MultiheadAttention` is rebuilt with fewer heads. The residual width stays `embed_dim`. |
| `nm_2_4` | In every group of 4 along the last dimension, the two smallest weights. | Same shapes, with a 2:4 mask. See the export below. |

Channel pruning only links two sibling layers when that width has one producer and one consumer in the parent. A feed-forward pair (`linear1` then `linear2`) matches. Two heads that merely share an output width do not. Grouped convolutions are left dense. Attention projections are not channel-pruned; use `structured_head`.

Head pruning keeps at least one head. The saved attention module is smaller than `nn.MultiheadAttention`, which cannot represent a head width narrower than `embed_dim`. `load_act_checkpoint` still builds the original dense ACT graph, so a structurally pruned ACT checkpoint does not load back through that helper.

`nm_2_4` refuses any `target_sparsity` other than `0.5`. A last dimension shorter than 4 is left dense. A remainder that is not a multiple of 4 stays dense, and the export is marked incomplete.

## Schedules

`one_shot` trains dense, then prunes once to the target.

`gradual` follows the cubic schedule (Zhu and Gupta). With initial sparsity 0 and final sparsity `target_sparsity`:

```text
s(step) = target + (0 - target) * (1 - (step - start) / (end - start)) ** 3
```

`start_step` defaults to 0. `end_step` defaults to the last optimizer step. At `start` the mask is still dense. At `end` it is the target. The ranking is frozen the first time the mask becomes sparse, and a pruned weight stays pruned.

`iterative` splits the run into `iterations` cycles (default 3). Cycle `i` prunes to `target * (i + 1) / iterations`, then fine-tunes. Without `finetune_steps`, the recipe `train_steps` are split across the cycles, and each cycle needs at least one step. With `finetune_steps`, the run length is `iterations * finetune_steps`.

For `nm_2_4`, the schedule moves the fraction of groups that are patterned, from none to all. Each patterned group is exactly 2:4, so a finished run is about 50% sparse.

## What a run records

`metrics.json` and the [`ohho.run/v1`](run-manifest.md) manifest both carry a `pruning` object. `layers` is one row per pruned module: `name`, `kind`, `sparsity`, `params_before`, `params_after`, and `nonzero`. `achieved_sparsity` is the fraction of those weights that were removed. `skipped` lists selected layers this method left dense, with a reason.

Structured runs also set `params_before`, `params_after`, `export_bytes_before`, and `export_bytes_after`. The byte counts are `pruning_export/before.pt` (the dense state dict at the start) and `pruning_export/after.pt` (the state dict after the shrink). `policy.pt` is the shrunk checkpoint. Unstructured and 2:4 runs keep the same parameter count; their files do not get smaller just because values are zero.

While the run is still going, `achieved_sparsity` and the byte counts are null. A recipe with no pruning block stores `pruning: null`. A manifest that omits the key is still valid.

## 2:4 export

A `nm_2_4` run writes `pruning_export/nm24/`:

| File | Contents |
| --- | --- |
| `manifest.json` | `schema` is `ohho.pruning.2to4/v1`. `layout` is groups of 4 along the last dimension. `complete` is true only when every exported tensor is fully patterned and its last dimension is a multiple of 4. |
| `<parameter>.values.f32` | Little-endian float32. Two values per group, in ascending index order. |
| `<parameter>.bitmask.u8` | One byte per group. Bits 0–1 are the lower kept index. Bits 2–3 are the higher kept index. |

Example group `[0.1, -0.4, 0.2, -0.9]` keeps indexes 1 and 3. The byte is `1 | (3 << 2) = 13`. The values are `-0.4` then `-0.9`.

`ohho.train.compression.pruning.load_2to4_tensor(checkpoint_dir, "fc.weight")` rebuilds the dense tensor. The checkpoint directory, `pruning_export`, or `nm24` directory all resolve the manifest.

## TensorRT

`build_tensorrt_sparse_engine` does not build an engine. Without a CUDA GPU and an installed TensorRT it raises `PruningCapabilityError` and writes nothing. With both present it still raises: constructing the engine is a GPU follow-up, and this tree does not run it. The 2:4 directory above is the input that build consumes. `ohho train` records that error on `pruning.tensorrt_error` and leaves `pruning.tensorrt_engine` null. The training run still succeeds.

An incomplete export (a tensor that is not fully 2:4) is refused as well, once CUDA and TensorRT are both available.

A later build can pass `builder(export_dir) -> engine_path`. The trainer does not pass one.
