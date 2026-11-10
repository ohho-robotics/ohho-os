# ohho optimize

`ohho optimize` exports a policy and, when the machine can do it, compiles
that export for one hardware target from `ohho targets`. The base install
stays dependency-free. Export and ONNX Runtime need:

```bash
pip install 'ohho-os[optimize]'
```

TensorRT, a CUDA execution provider, AWQ, GPTQ, bitsandbytes, and FP8 are
optional. If the selected path needs one of them and it is not installed, the
command exits 2 and names the missing piece. It does not write a stand-in
engine or a fake quantized checkpoint.

```bash
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --json
ohho optimize --model policy.pt --submodule actor --target x86-cuda --precision fp16 --tensorrt --workspace-gb 2 --max-batch 4 --dynamic-batch --out build/fp16
ohho optimize --model policy.pt --target jetson-orin-nx --precision int8 --tensorrt --calib-data calib --calib-batch 8 --calib-cache build/int8.cache --out build/int8
ohho optimize --model hf-model/ --target x86-cuda --precision int8 --quant awq --out build/awq
```

`--model` is a checkpoint path or a registry id. `reference-act` and
`reference-mlp` are tiny untrained fixtures so a CPU check can run without a
download. A directory may contain `model.json` plus `model.pt`, `model.onnx`,
or `model.ts`. A Hugging Face directory (a `config.json` and no weight file
above) is a weight-quant source. `--submodule` is a dotted path or one of
`vision`, `actor`, `encoder`.

`--target` is a built-in profile id or a `.json` / `.yaml` profile file.
Precision and the execution provider are checked with `ohho.targets`. FP8 on
an Orin profile, INT8 TensorRT on a CPU profile, and a workspace larger than
the memory budget are rejected before any export.

## What each precision does

| Request | What is written |
| --- | --- |
| `fp32` | ONNX opset 17 with a dynamic batch, then an ONNX Runtime session on the selected provider. |
| `fp16` | A TensorRT engine (`--tensorrt`). The ONNX graph stays FP32. |
| `int8` + `--tensorrt` | A TensorRT engine. The entropy calibrator is assigned to `config.int8_calibrator` before the build. |
| `int8` or `int4` + `--quant awq\|gptq` | A weight-quant directory when that package is installed. |
| `fp8` + `--tensorrt` | A TensorRT engine when that builder flag exists. |

ONNX export always uses a dynamic batch axis. `--dynamic-batch` widens the
TensorRT optimisation profile from batch 1 to `--max-batch`. Without it, the
profile is fixed at `--max-batch`. `--workspace-gb` overrides the profile
workspace and must stay inside `memory_budget_mib`.

`--ort-ep` is `cpu`, `cuda`, or `tensorrt`. Omit it and the session uses the
first provider on the profile. The session is opened with that provider only.
If ONNX Runtime falls back to a different provider, the run fails.

## INT8 calibration

`--calib-data` is a directory of float32 samples in the model input shape,
without the batch axis. Either:

- `calib.json` with `{"shape": [8], "dtype": "float32"}` and one `.bin` file per sample, or
- `.npy` float32 files when NumPy is installed.

`--calib-batch` groups those samples. A short directory is an error. The
cache path is read at the start of the build and written when TensorRT
returns a cache. A later calibration change can replace the batch source
without moving the `config.int8_calibrator` assignment.

## Exporters

Matchers run in order: ACT-style (a module with an encoder and an action
head), MLP actor (`rl_mlp`, including an `actor.*` state dict), then generic
`torch.onnx.export`. `register_exporter` inserts another matcher ahead of the
generic one. `export_torchscript` traces the same wrapped module and scripts
it if tracing fails. A TorchScript input is saved again as `model.ts` next
to the ONNX file.

An actor checkpoint is a state dict whose Linear weights use keys
`actor.0.weight`, `actor.2.weight`, and so on (ELU sits on the odd indexes).

## Weight quantizers

AWQ and GPTQ are `--quant`. Both require a local directory that contains
`config.json`. GPTQ also needs calibration examples passed to `quantize()`;
the CLI does not download a dataset. bitsandbytes and FP8 are on the same
Python interface and need a CUDA device. `probe_quant` reports that without
importing a missing package.

## Manifest

The output directory gets `manifest.json`. The schema id is
`ohho.optimize_manifest.v1` (`ohho/optimize/optimize_manifest.schema.json`).
The document records the model, the target, the precision, the execution
provider, the flags, the torch / ONNX / ONNX Runtime / TensorRT versions, the
request inputs, the session providers that were actually enabled, the sha256
and size of each output, and the parity report.

## Parity gate

After a successful export, `ohho optimize` compares the reference model to the
ONNX or TorchScript file it just wrote. The default fail is a mean absolute
difference above 0.05. A failure deletes the outputs and exits 1, so the
artifact is not published. `--allow-parity-fail` publishes anyway and records
that flag on `flags`, `inputs`, and `parity` in the manifest.

```bash
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --parity-n 256
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --allow-parity-fail
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --parity-set build/act-golden
```

`--parity-set` scores the export on a captured set. The manifest's
`parity.parity_set` records that set's id and sha256. Omitting it uses a
synthetic set when `--parity-inputs` is `random`, with `synthetic` true.
`--parity-inputs` is `random` (seeded, `--parity-seed`), a LeRobot dataset
directory, `recorded` (one short episode from the sim-loop recorder), or
`camera:PATH`.
TensorRT engines and weight-quant directories are not executed here. When the
directory has an ONNX graph, that graph is what the gate scores and the engine
is listed as unchecked. When nothing runnable was written, the gate fails
closed. The report schema is [`docs/serve/parity.md`](serve/parity.md).

## GPU follow-up

CPU tests cover ONNX versus PyTorch for the ACT fixture and the MLP fixture,
and they attach a mock TensorRT calibrator. A real TensorRT engine, a CUDA
execution provider, and the quantizer kernels are not run in CPU CI.
