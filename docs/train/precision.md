# Mixed precision

`ohho train` reads `precision` from an [`ohho.recipe/v1`](recipes.md) document and checks it before the family runs. The same check is what a later trainer picks up. This build does not rewrite the request to a different precision.

| Value | What runs | Where |
| --- | --- | --- |
| `fp32` | Ordinary PyTorch. Master weights stay fp32. | CPU, CUDA, and MPS. This is the default. |
| `fp16` | `torch.amp` autocast in float16, and `GradScaler` loss scaling. Master weights stay fp32. | CUDA, compute capability 5.3 or newer. |
| `bf16` | `torch.amp` autocast in bfloat16. No GradScaler. | CUDA compute capability 8.0 or newer, and CPU or MPS when a linear step under autocast succeeds. |
| `fp8` | Transformer Engine `fp8_autocast` with a hybrid delayed-scaling recipe. | CUDA compute capability 8.9 or newer, and an installed `transformer_engine` package. |

fp16 on CPU or MPS is refused. bfloat16's exponent range is the reason fp16 needs the scaler and bf16 does not. fp8 is refused when the GPU is older than compute capability 8.9 or when `transformer_engine` is not installed. The error names the precision, the device, and the requirement, and it says that no fallback is applied. The process exits `1`. `train_start` is not emitted and `run_manifest.json` is not written.

A family can narrow the set with `metadata()`:

```python
def metadata(self) -> dict:
    return {"precisions": ["fp32", "bf16"]}
```

Omitting `metadata` accepts all four values. The device check still runs.

## What the trainer does

`ohho.train.precision.PrecisionSession` wraps one step loop.

- fp32 calls `loss.backward()` and `optimizer.step()`.
- fp16 runs the forward under CUDA autocast and steps with `GradScaler` (PyTorch's default initial scale, 65536, unless that torch build uses another default).
- bf16 runs the forward under autocast and an unscaled backward.
- fp8 enters `transformer_engine.pytorch.fp8_autocast`. The package is imported only when a run has already passed the fp8 check.

ACT keeps its existing modules (`nn.Linear` and `nn.Transformer*`). Transformer Engine applies FP8 kernels to its own modules inside that autocast. The manifest records `precision_backend: transformer_engine` for fp8. A GPU with the package installed is what confirms that an ACT step under that context matches the installed Transformer Engine.

SmolVLA, Diffusion Policy, OpenVLA, and PPO are registered and are not trainable in this build. Their recipes can already name a precision. When a trainer is registered for one of them, the runner applies this check before `run`.

## Metrics

A run records one precision. `metrics.json` includes `throughput_steps_per_s` and `peak_memory_bytes` for that precision. The [`ohho.run/v1`](run-manifest.md) manifest copies them into `precision_metrics`, which has a key for every precision. Only the precision that ran is an object. The others are null.

Throughput is training steps divided by the wall time of the step loop (`time.perf_counter`). It is null when the trainer did not write it.

Peak memory is bytes:

- CUDA: `torch.cuda.max_memory_allocated()` after the counter is reset at the start of the loop, so resident weights and later activations are included.
- CPU and MPS: the process resident set sampled during the loop (Windows working set, Linux `/proc/self/statm`, or `ru_maxrss` elsewhere). That figure includes the interpreter, not only tensors.

`loss_scaling` in the manifest is true only for fp16. fp8's delayed scaling stays inside Transformer Engine and does not set that flag.

## Without a GPU

fp32 on CPU is what `ohho train` runs in the unit tests. bf16 on CPU runs when the autocast probe succeeds and is skipped when it does not. fp16 and fp8 are refused on CPU by the capability check, including under mocked devices (a CPU, a GPU without bf16, a GPU without fp8). Training ACT or SmolVLA at fp16, bf16, or fp8 on a real GPU is not confirmed from this tree.
