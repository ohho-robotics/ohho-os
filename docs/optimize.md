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
ohho optimize --model policy.pt --target jetson-orin-nx --precision int8 --tensorrt --calib-data calib --calib-batch 8 --calib-max-images 32 --calib-cache build/int8.cache --out build/int8
ohho optimize --model hf-model/ --target x86-cuda --precision int8 --quant awq --out build/awq
ohho optimize --model smolvla/ --submodule lm --precision int4-awq --target PROFILE --out build/awq
ohho optimize --model reference-act --target x86-cpu --precision int8 --out build/int8
```

`--model` is a checkpoint path or a registry id. `reference-act` and
`reference-mlp` are tiny untrained fixtures so a CPU check can run without a
download. A directory may contain `model.json` plus `model.pt`, `model.onnx`,
or `model.ts`. A Hugging Face directory (a `config.json` and no weight file
above) is a weight-quant source. `--submodule` is a dotted path or one of
`vision`, `actor`, `encoder`, `lm`. `lm` selects the SmolVLA text backbone
(`model.vlm_with_expert.vlm.model.text_model` and the same role on a
shorter attribute path).

`--target` is a built-in profile id or a `.json` / `.yaml` profile file.
Precision and the execution provider are checked with `ohho.targets`. FP8 on
an Orin profile, INT8 TensorRT on a CPU profile, and a workspace larger than
the memory budget are rejected before any export.

## What each precision does

| Request | What is written |
| --- | --- |
| `fp32` | ONNX opset 17 with a dynamic batch, then an ONNX Runtime session on the selected provider. |
| `fp16` | A TensorRT engine (`--tensorrt`). The ONNX graph stays FP32. |
| `int8` + `--tensorrt` | A TensorRT engine. TensorRT 8 and 9 assign the entropy calibrator to `config.int8_calibrator` before the build. TensorRT 10 and newer deprecate that implicit calibrator and build explicit Q/DQ instead. |
| `int8` on `x86-cpu` or `rpi5-cpu` | ORT static INT8 (QDQ). `--quant awq` still selects AWQ instead. |
| `int8` or `int4` + `--quant awq\|gptq` | A weight-quant directory when that package is installed. |
| `int4-awq`, `int4-gptq` | The same INT4 weight quant, without a separate `--quant` flag. |
| `int4-bnb`, `nf4` | bitsandbytes NF4. Research baseline only. |
| `fp8` | A TensorRT FP8 engine. Rejected unless the profile is Ada or Hopper (`cloud-l4`, `cloud-h100`). |

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

`--calib-batch` groups those samples. A short directory is an error.
`--calib-max-images N` reads only the first N samples. N must be at least 1
and at least `--calib-batch`.

`calib.json` may include a `preprocessing` object with `mean`, `std`,
`resize` (`[height, width]`), and `layout` (`nchw` or `nhwc`). Those fields,
the sample shape, the dtype, `--calib-max-images`, and `--submodule` are the
preprocessing config.

The cache path is read at the start of the build and written when calibration
finishes. The build also writes `{cache}.meta.json` with the model sha256 and
that preprocessing config. A second build reuses the cache when both match.
A different model sha256 or preprocessing config invalidates the cache, logs
the reason, and calibrates again.

TensorRT 10 and newer do not attach `config.int8_calibrator`. They insert
`QuantizeLinear` / `DequantizeLinear` nodes (or keep the ones already in the
graph) and build that explicit graph. An INT8 build with no calibration data
is refused unless the ONNX model already contains those Q/DQ nodes.

On TensorRT 8 and 9 the entropy calibrator is still assigned to
`config.int8_calibrator` before the build.

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

Backends register on the `ohho.quantizers` entry-point group. Each one
implements `requirements`, `supports(target, model)`, and
`quantize(model, calib)`. `supports` runs against the target profile
before a checkpoint is exported. A third-party package can add a backend by
publishing an entry point; the builtins are AWQ, GPTQ, bitsandbytes NF4,
TensorRT FP8, and ORT static INT8.

`--precision int4-awq` and `--quant awq` both select AWQ. The older
`--quant` flag still works. AWQ and GPTQ need a local directory that
contains `config.json`, or a SmolVLA module selected with `--submodule lm`.
GPTQ also needs calibration examples passed to `quantize()`; the CLI does
not download a dataset. bitsandbytes NF4 (`int4-bnb`) is a research
baseline and needs a CUDA device. `probe_quant` reports a missing package
without importing it.

Every published output, including each quantizer artifact, is passed to
`ohho.optimize.publish.run_parity`. That calls `ohho.parity.run_parity`
when OHH-430 is installed. The manifest records `backend` and
`backend_version` on each output.

## Manifest

The output directory gets `manifest.json`. The schema id is
`ohho.optimize_manifest.v1` (`ohho/optimize/optimize_manifest.schema.json`).
The document records the model sha256, the target profile, the precision,
every applied flag, the calibration-cache sha256, the request inputs, the
session providers that were actually enabled, the sha256, size, backend
name, and backend version of each output, and the parity report. It also
records the build duration and the host GPU (device name and compute
capability `major.minor`).

Toolchain versions on that manifest are torch, ONNX, ONNX Runtime, TensorRT,
CUDA, the driver, and JetPack. CUDA comes from `torch.version.cuda`. The
driver comes from `nvidia-smi`. JetPack is the L4T release parsed from
`/etc/nv_tegra_release` when that file exists. A probe that cannot see a
piece stores null.

## Parity gate

After a successful export, `ohho optimize` compares the reference model to the
ONNX or TorchScript file it just wrote. The default fail is a mean absolute
difference above 0.05. A failure deletes the outputs and exits 1, so the
artifact is not published. `--allow-parity-fail` publishes anyway and records
that flag on `flags`, `inputs`, and `parity` in the manifest.

```bash
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --parity-n 256
ohho optimize --model reference-act --target x86-cpu --precision fp32 --out build/act --allow-parity-fail
```

`--parity-inputs` is `random` (seeded, `--parity-seed`), a LeRobot dataset
directory, or `recorded` (one short episode from the sim-loop recorder).
TensorRT engines and weight-quant directories are not executed here. When the
directory has an ONNX graph, that graph is what the gate scores and the engine
is listed as unchecked. When nothing runnable was written, the gate fails
closed. The report schema is [`docs/serve/parity.md`](serve/parity.md).

## Reproducible builds

A second `ohho optimize` with the same model bytes, target profile,
precision, flags, and toolchain versions copies the previous artifacts and
skips export, calibration, and the engine build. The cache key is the sha256
of those values. The local cache lives at `~/.ohho/optimize-cache`, or at
`OHHO_OPTIMIZE_CACHE` when that variable is set. `--no-cache` rebuilds and
replaces the cache entry.

```bash
ohho optimize --explain build/fp16
ohho optimize --explain build/fp16/model.engine
ohho optimize --model policy.pt --target x86-cuda --precision fp16 --tensorrt --out build/fp16 --no-cache
```

`--explain ARTIFACT` prints the `manifest.json` beside a build directory, an
artifact in that directory, or the manifest file itself.

An engine is refused when this host's GPU architecture or TensorRT version
differs from the manifest. The check runs when a cached engine is restored
and when `require_engine_host` opens an engine file. Pass `--no-cache` to
build a new engine for the current host.

`RegistryCache` (`find`, `publish`) is the hook for a console registry.
`optimize(..., registry=client)` asks it after a local miss and passes it
the build directory after a local store. This package does not call a
registry of its own.

## GPU follow-up

CPU tests cover ONNX versus PyTorch for the ACT fixture and the MLP fixture,
and they attach a mock TensorRT calibrator. A real TensorRT engine, a CUDA
execution provider, and the quantizer kernels are not run in CPU CI.
