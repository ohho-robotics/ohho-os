# Parity report (`ohho.parity/v1`)

`ohho parity` scores an optimized artifact against a reference model on the
same inputs. `ohho optimize` runs the same gate before it writes
`manifest.json`. A console, a harness, or CI can render the JSON directly.

The schema id inside the document is `ohho.parity/v1`. The JSON Schema file
is `ohho/parity/parity_report.schema.json`
(`https://ohho-robotics.com/schemas/parity-report.v1.json`).

```bash
ohho parity --reference reference-act --candidate build/act --inputs random --n 256 --json
ohho parity --reference policy.pt --candidate build/act/model.onnx --inputs ./dataset --n 256
ohho parity --reference reference-mlp --candidate candidate.pt --inputs recorded --seed 0
```

Exit 0 when the gate passes. Exit 1 when a scored metric misses its limit.
Exit 2 when the pair cannot be scored (missing file, shape mismatch, bad
threshold file). `--json` prints the report on both pass and fail. A request
error with `--json` prints `{"ok": false, "error": "..."}`.

## What is compared

Both sides run on CPU. The reference is a registry id (`reference-act`,
`reference-mlp`) or a checkpoint path. The candidate is an `.onnx` file, a
TorchScript file, a `.pt` module, or a directory containing one of those
(`model.onnx` is preferred).

A TensorRT `.engine` is not executed. That comparison needs a GPU and
TensorRT, which this process does not do. Pass the ONNX file exported beside
the engine. `ohho optimize` still lists the engine under `unchecked` and
scores the ONNX graph. If the output directory has no ONNX or TorchScript
file, the gate fails closed.

## Inputs

`--inputs` is one of:

| Value | Samples |
| --- | --- |
| `random` | `--n` vectors from `random.Random(--seed)`, each coordinate in `[-1, 1]`. |
| A LeRobot dataset directory, or `dataset:PATH` | `observation.state` rows. A dataset longer than `--n` is subsampled with `--seed` and then sorted back into time order. A shorter dataset uses every frame. |
| `recorded` | One episode from `ohho.sim_loop.record_sim_episodes` (at most 8 steps), then the same reader. |
| `recorded:PATH` or an episode `.jsonl` | Frames already written by that recorder. |

The report records `inputs.source`, `inputs.path`, `inputs.seed`,
`inputs.n_requested`, `inputs.n`, `inputs.indices` (null for random), and
`inputs.sha256`. The hash is SHA-256 of the little-endian float32 values,
row-major, input names in sorted order. `inputs.dtype` is `float32`.

## Metrics

Every output tensor reports:

| Field | Meaning |
| --- | --- |
| `mean_abs` | Mean of `\|reference - candidate\|`. |
| `max_abs` | Largest absolute element difference. |
| `rmse` | Root mean square of the element difference. |
| `cosine` | Cosine of the two flattened tensors. Two zero tensors score 1. |

Action heads (family `act`, `rl_mlp`, `diffusion`, `smolvla`, `openvla`, or
`policy`, or an output whose name contains `action`) also report:

| Field | Meaning |
| --- | --- |
| `per_joint` | Mean absolute error of each joint. The last axis is the joint axis. |
| `action_chunk` | Mean, over the batch, of the L2 norm of that row's action chunk. |

Classifiers and LLMs (family `classifier`, `llm`, or `vlm`, or an output
whose name contains `logit`) also report:

| Field | Meaning |
| --- | --- |
| `top1` | Fraction of rows whose last-axis argmax agrees. Ties take the lower index. |
| `kl` | Mean KL(softmax(reference) \|\| softmax(candidate)), in nats. |

`shape` is the reference output shape, batch included.

## Limits

The default gate is `mean_abs <= 0.05`. A value equal to the limit passes.
Other metrics are reported and gated only when a limit is set.

Classifier, LLM, and VLM families also start with `top1_min` 0.99 and
`kl_max` 0.05. A threshold file replaces those fields; it does not have to
repeat ones it leaves alone.

```json
{
  "default": {"mean_abs": 0.05},
  "families": {"act": {"mean_abs": 0.02}},
  "targets": {
    "jetson-orin-nx": {
      "mean_abs": 0.08,
      "precisions": {"INT8": {"mean_abs": 0.12}},
      "families": {"act": {"mean_abs": 0.03}}
    }
  }
}
```

Later layers override only the fields they name:

1. Built-in default (`mean_abs` 0.05).
2. Built-in family extras (classifier, llm, vlm).
3. File `default`.
4. File `families.<family>`.
5. File `targets.<target>` metric fields.
6. File `targets.<target>.precisions.<PRECISION>`.
7. File `targets.<target>.families.<family>`.
8. `--mean-abs`, which replaces `mean_abs` only.

`--family` selects the family. It defaults to the model kind. `--target` and
`--precision` select the target layers. `ohho optimize` passes the profile id
and the precision it actually applied (`FP32`, `FP16`, `INT8`, `FP8`, `INT4`).

## Result

`result` is `pass` or `fail`. `reasons` is empty on pass. On fail each entry
names the output, the metric, the value, and the limit, for example
`actions: mean abs diff 0.42 > 0.05`.

`allow_parity_fail` is false for `ohho parity`. `ohho optimize` copies the
`--allow-parity-fail` flag into this field and also into `flags` and `inputs`
of `manifest.json`. The full report is `parity.json` in the output directory
and the `parity` object of the manifest. A failing gate without that flag
removes the outputs and does not write the manifest. The process exits 1.

`unchecked` lists outputs that were written and not executed, with a reason.
An empty list means every output was compared.

## Example

```json
{
  "schema": "ohho.parity/v1",
  "result": "pass",
  "reasons": [],
  "reference": {"ref": "reference-act", "kind": "act", "family": "act"},
  "candidate": {"path": "model.onnx", "format": "onnx"},
  "target": "x86-cpu",
  "precision": "FP32",
  "thresholds": {
    "mean_abs": 0.05,
    "max_abs": null,
    "rmse": null,
    "cosine_min": null,
    "top1_min": null,
    "kl_max": null,
    "per_joint": null,
    "action_chunk": null
  },
  "inputs": {
    "source": "random",
    "path": null,
    "seed": 0,
    "n_requested": 256,
    "n": 256,
    "indices": null,
    "sha256": "…64 lowercase hex characters…",
    "dtype": "float32"
  },
  "outputs": [
    {
      "name": "actions",
      "shape": [256, 3, 4],
      "mean_abs": 0.0,
      "max_abs": 0.0,
      "rmse": 0.0,
      "cosine": 1.0,
      "per_joint": [0.0, 0.0, 0.0, 0.0],
      "action_chunk": 0.0,
      "top1": null,
      "kl": null
    }
  ],
  "unchecked": [],
  "allow_parity_fail": false
}
```

`candidate.format` is `onnx`, `torchscript`, `module`, or null when nothing
could be executed. `reference.family` is the family used for limits and for
the extra metrics.

## Task accuracy

`--tasks` is an optional stage after the tensor gate. It does not change
`ohho.parity/v1`. The stage writes an `ohho.bench/v1` document whose
`accuracy` block compares a reference score with an optimised score.

```bash
ohho parity --reference reference-act --candidate build/act --inputs random \
  --tasks action_error,sim_success --task-seeds 20 --json
ohho optimize --model reference-act --target x86-cpu --precision fp32 \
  --out build/act --tasks action_error,sim_success
```

`--tasks all` also runs detection mAP and token match. Those two need
`--task-detections` and `--task-tokens`. Omitting `--tasks` leaves the tensor
gate unchanged.

Evaluators are entry points in the `ohho.parity.tasks` group:

| Name | Metric | What is scored |
| --- | --- | --- |
| `action_error` | `action_mae`, `action_l2` | Predicted action against the held-out episode's ground truth. MAE is the mean absolute joint error. L2 is the mean Euclidean norm of each frame. |
| `sim_success` | `sim_success` | Closed-loop success on seeds `0 .. N-1` (default 20) through `ohho.sim_loop.evaluate_policy_sim`. |
| `detection_map` | `map` | Mean average precision at IoU 0.5. |
| `token_match` | `top1`, `token_match` | Top-1 rate and token match rate for an LLM or VLM. |

Each row stores `reference`, `optimized`, `delta` (`optimized - reference`),
`tolerance`, and `result`. Error metrics pass when `delta` is at most the
tolerance. Rate metrics pass when the drop (`reference - optimized`) is at
most the tolerance. A value equal to the tolerance passes. The bench
`accuracy.result` is `fail` when any row fails. `accuracy.primary` stays the
schema's required headline and is the first optimised score.

Default tolerances are action MAE 0.05, action L2 0.05, sim success drop 0,
mAP drop 0.01, top-1 drop 0.01, and token-match drop 0.01. A JSON file
replaces only the metrics it names, in order `default`, `families.<family>`,
`models.<model>`.

Sim episodes use the `sim` robot. The policy sees the sim state with
`(goal_x, goal_y, seed)` appended. The goal lies 0.4 m from the origin.
`pursuit_action` reaches it in 8 steps of 0.1 s at 0.5 m/s. Success is a
final position within 0.02 m. Nothing in this stage runs on a real robot.

`ohho optimize` writes the document to `accuracy.json` next to `manifest.json`
after the tensor gate. A task miss is not published unless
`--allow-parity-fail` is set. The manifest schema is unchanged.

With `--json`, `ohho parity` prints `{"parity": ..., "accuracy": ...}` only
when `--tasks` is set. Without `--tasks` the printed document is still the
tensor report alone. Exit 1 when either gate fails. Exit 2 when a pair cannot
be scored.

The ACT FP16 comparison is a CPU check. Weight and value rounding uses IEEE
binary16. A GPU float16 run is not executed here. The same action-error and
sim-success evaluators run on deterministic fake policies when torch is absent.
