# Quantisation-aware training

`ohho train` reads an optional `qat` block from an [`ohho.recipe/v1`](recipes.md) document. When the block is absent, training is unchanged and the run manifest records `qat: null`. When the block is present, linear layers are wrapped with PyTorch `torch.ao.quantization.FakeQuantize` and the observers in `torch.ao.quantization.observer` (`MinMaxObserver`, `PerChannelMinMaxObserver`). The fake-quant ops are `torch.fake_quantize_per_tensor_affine` and `torch.fake_quantize_per_channel_affine`.

The block is plain fields, so a sweep can set them with `--set`:

```bash
ohho train recipe.yaml \
  --set qat.bit_width=4 \
  --set qat.granularity=per_tensor \
  --set qat.calibration_steps=8 \
  --set qat.finetune_steps=40
```

| Field | Meaning | Default |
| --- | --- | --- |
| `enabled` | When false, the block is recorded and the phases do not run. | `true` |
| `bit_width` | `8` or `4`, for both weights and activations. | `8` |
| `scope` | `weight_only` or `weight_and_activation`. | `weight_only` |
| `granularity` | `per_tensor` or `per_channel`, for weights. | `per_channel` |
| `activation_granularity` | Same enum. Omitted means it follows `granularity`. | weight granularity |
| `symmetric` | `true` keeps zero-point 0 (signed). `false` is affine (unsigned). | `true` |
| `activation_symmetric` | Omitted means it follows `symmetric`. | weight symmetry |
| `calibration_steps` | Warm-up forwards. Observers on, fake quant off, no optimizer step. | `0` |
| `finetune_steps` | Fake-quant steps after the warm-up. Observers frozen. | `0` |
| `export_onnx` | Write `qdq.onnx` for a linear chain. Refused for family `act`. | `false` |

INT8 symmetric uses the signed range `-128..127`. INT8 asymmetric uses `0..255`. INT4 symmetric uses `-8..7`. INT4 asymmetric uses `0..15`. Per-channel weights use output-channel axis 0. Per-channel activations use axis 1 of the linear output.

`weight_and_activation` needs `calibration_steps >= 1`. With `calibration_steps` of 0 the weight tensors are still observed once, which is enough for `weight_only`. There is no stand-in scale.

For family `act`, the precision loop (`hyperparameters.train_steps`) runs first. Calibration and the QAT fine-tune run after it, at the recipe learning rate, with AdamW. The fine-tune uses the fake-quant straight-through estimator. `policy.pt` stays the fp32 master weights so the existing loader still reads it. The quantized graph is the ONNX file, not the checkpoint. Each wrapped linear keeps `.weight` and `.bias` and registers a forward hook so the transformer encoder's eval fast path still runs the fake-quant module.

## Q/DQ ONNX

`ohho.compression.qat.export_qdq_onnx` lowers `TinyActLike` (state projection, latent projection, action head) to ONNX. Each linear is a float weight, `QuantizeLinear`, `DequantizeLinear`, and `Gemm`. Weight-and-activation adds the same pair on the linear output. INT8 uses opset 13. INT4 uses opset 21. TensorRT's parser consumes this Q/DQ pattern. The unit test checks that ONNX Runtime on CPU matches the fake-quant PyTorch module.

Family `act` includes a transformer. That module is not a linear chain, so `qat.export_onnx: true` is refused before `train_start` and no graph is written. Full-transformer lowering is a follow-up.

`build_tensorrt_engine` does not build an engine. Without the `tensorrt` package, or without a CUDA GPU, it raises `QatCapabilityError` and names what is missing. A machine that has both still does not call the TensorRT builder in this version; the ONNX file is what a GPU host should pass to `trtexec`. GPU engine timing is a follow-up.

## PTQ and QAT

Post-training quantisation (PTQ) and QAT share one calibration warm-up and one fake-quant configuration. PTQ stops when the warm-up ends. QAT then takes `finetune_steps` of fake-quant updates. Compare them like this:

1. Fix `bit_width`, `scope`, `granularity`, `symmetric`, and the calibration batches. Do not change those between the two arms.
2. PTQ: run `calibration_steps` forwards with observers on and the optimizer off. Freeze the scales. Evaluate task loss (for ACT, the chunk loss) on a held-out batch. Export Q/DQ ONNX if the module is a linear chain.
3. QAT: repeat the same warm-up from the same initial weights, then run `finetune_steps` with fake quant on and observers frozen. Evaluate the same held-out batch. Export the same way.
4. Report both losses, the shared settings, and, on a GPU host later, TensorRT latency of the two ONNX files.

`ohho.compression.qat.compare_ptq_qat(spec)` runs that procedure on a tiny `TinyActLike` on CPU and returns `ptq_eval_loss`, `qat_eval_loss`, and `gpu_measured: false`. It does not produce dataset or TensorRT numbers. Those measurements need a GPU host and are a follow-up.

The manifest field `qat` is the resolved block (defaults filled). `metrics.json` adds `finetune_loss`, `onnx_path`, and `ran` for an ACT run that executed the phases. The console run page can show the manifest field; a dedicated compression panel is a follow-up.
