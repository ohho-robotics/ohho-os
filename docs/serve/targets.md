# Hardware target profiles

A target profile tells an optimisation or benchmark run which precisions,
ONNX Runtime execution providers, TensorRT versions, memory budget and
default latency budgets apply. It is not a deployment topology.
`ohho profile` still answers which process role runs on which machine.
`ohho targets` answers which hardware the compiled model must fit.

Import the API as `ohho.targets`. The serve package does not re-export it.

Profiles are JSON package data in `ohho/targets/profiles/`. The JSON Schema
for one profile is `ohho/targets/target_profile.schema.json`. That schema is
not part of the transport protocol files under `schemas/`. A console can
read the nine JSON files to render a target picker. YAML documents of the
same shape load when PyYAML is installed (`pip install 'ohho-os[yaml]'`).

`slos_p95_ms` and `memory_budget_mib` are planning budgets, not measurements.
The memory budget is about half of the module the summary names, so the OS
and cameras keep room. A 4GB Orin Nano, an 8GB Orin NX, a 32GB AGX Orin or
a smaller host needs a lower budget than the built-in value.

## Fields

| Field | Meaning |
| --- | --- |
| `id` | Stable slug, for example `rpi5-cpu`. |
| `arch` | `aarch64` or `x86_64`. |
| `accelerator` | `none`, `orin_gpu` (Orin GPU), `dgpu` (discrete GPU) or `npu` (placeholder). |
| `precisions` | `FP32`, `FP16`, `INT8`, `FP8`, `INT4`. `FP32` is always required. |
| `execution_providers` | ONNX Runtime providers, best first. `CPUExecutionProvider` is last. |
| `tensorrt` | Inclusive `min_version`, exclusive `before_version`, and optional excluded windows. `null` when TensorRT is not used. |
| `memory_budget_mib` | Memory the optimiser may plan to use. |
| `workspace_size_mib` | Default TensorRT workspace. `0` when TensorRT is not used. |
| `slos_p95_ms` | Default p95 milliseconds for `act`, `smolvla`, `diffusion` and `openvla`. |

The NPU value is a placeholder. A profile that uses it can list `FP32` and
`INT8` on `CPUExecutionProvider` only. It cannot select CUDA or TensorRT.
No built-in profile uses it yet.

## Built-in profiles

| Id | Arch | Accelerator | Precisions | Providers | TensorRT | Memory / workspace (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| `jetson-orin-nano` | aarch64 | Orin GPU | FP32, FP16, INT8 | TensorRT, CUDA, CPU | `>=8.5.2,<11.0.0` | 4096 / 512 |
| `jetson-orin-nx` | aarch64 | Orin GPU | FP32, FP16, INT8 | TensorRT, CUDA, CPU | `>=8.5.2,<11.0.0` | 8192 / 1024 |
| `jetson-agx-orin` | aarch64 | Orin GPU | FP32, FP16, INT8 | TensorRT, CUDA, CPU | `>=8.5.2,<12.0.0` excluding `>=11.0.0,<11.4.1` | 32768 / 4096 |
| `rpi5-cpu` | aarch64 | none | FP32, INT8 | CPU | none | 4096 / 0 |
| `x86-cpu` | x86_64 | none | FP32, INT8 | CPU | none | 8192 / 0 |
| `x86-cuda` | x86_64 | dGPU | FP32, FP16, INT8 | CUDA, TensorRT, CPU | `>=8.5.0,<12.0.0` | 8192 / 2048 |
| `cloud-l4` | x86_64 | dGPU | FP32, FP16, INT8, FP8 | TensorRT, CUDA, CPU | `>=8.6.0,<12.0.0` | 20480 / 4096 |
| `cloud-a10g` | x86_64 | dGPU | FP32, FP16, INT8 | TensorRT, CUDA, CPU | `>=8.5.0,<12.0.0` | 20480 / 4096 |
| `cloud-h100` | x86_64 | dGPU | FP32, FP16, INT8, FP8 | TensorRT, CUDA, CPU | `>=8.6.0,<12.0.0` | 65536 / 8192 |

Orin profiles assume the 8GB Nano, 16GB NX and 64GB AGX modules. `rpi5-cpu`
assumes an 8GB Raspberry Pi 5. `x86-cuda` is an RTX-class board and does not
list FP8, because Turing and Ampere RTX boards do not have it. `cloud-l4`
(Ada) and `cloud-h100` (Hopper) list FP8. `cloud-a10g` is Ampere and does
not. INT4 is in the schema for a later discrete-GPU profile. None of the
built-in profiles advertise it.

Orin Nano and Orin NX stay on the TensorRT 10.x line (`<11.0.0`), which
includes the TensorRT 10.16.2 stack shipped with current JetPack 7.2 for the
Orin family. AGX Orin also allows TensorRT 11.4.1 and later, before 12.
TensorRT 11.0.0 through 11.4.0 did not support JetPack, so that window is
excluded. Versions are dotted triples such as `10.16.2`.

## Execution providers

The provider order is the learning_engine hardware order, kept in
`ohho.targets.providers` so serve code does not keep a second copy:

- Orin: TensorRT, then CUDA, then CPU.
- A discrete GPU: CUDA, then CPU. `prefer_tensorrt=True` promotes TensorRT
  and is the order stored for `cloud-l4`, `cloud-a10g` and `cloud-h100`.
- `x86-cuda` keeps CUDA first and lists TensorRT second, so a benchmark can
  select it without putting the engine build ahead of CUDA.
- CPU and the NPU placeholder: CPU only.

`onnx_providers` can filter that order through the providers installed in
this process. The stored profile order does not change when a provider is
missing on the machine running the tool.

## CLI

```bash
ohho targets list
ohho targets list --json
ohho targets show rpi5-cpu
ohho targets detect
ohho targets detect --json
ohho targets check --profile rpi5-cpu --precision INT8 --provider tensorrt
ohho targets check --profile jetson-orin-nano --precision FP8 --provider tensorrt
ohho targets check --profile cloud-h100 --precision FP8 --provider tensorrt --tensorrt-version 11.4.1
```

`list --json` prints `{"schema": "ohho.target_profile.v1", "profiles": [...]}`.
`check` prints `ok: ...` and exits 0, or prints `error: ...` and exits 2.

Rejected examples:

- INT8 TensorRT on `rpi5-cpu` or `x86-cpu`: those targets have no TensorRT
  execution provider. INT8 on the CPU provider is valid.
- FP8 on any Orin profile: Orin GPUs support FP32, FP16 and INT8. FP8 needs
  an Ada or Hopper GPU (`cloud-l4` or `cloud-h100`).
- FP8 on `cloud-a10g` or `x86-cuda`: not in that profile's precision list.
- A TensorRT version outside the profile window, including the JetPack gap
  on AGX Orin.

## Detection

`ohho targets detect` only reads. It reuses the existing Jetson, Raspberry Pi
and Apple Silicon probes, and `ohho.hardware.resolve_device` for the torch
device. A discrete GPU name comes from `nvidia-smi --query-gpu` when that
program is on `PATH`. ONNX Runtime and TensorRT are imported only to ask
whether they are installed. The command does not set `OHHO_ARM_HARDWARE`,
does not write a file and does not open a robot.

On an x86_64 machine with no CUDA device, the result is `x86-cpu`. The report
says CUDA is missing and TensorRT is missing, including when PyTorch and
ONNX Runtime are not installed. A Jetson board is matched from the device-tree
model to Orin Nano, Orin NX or AGX Orin. A Raspberry Pi 5 selects `rpi5-cpu`.
GPU names select `cloud-h100`, `cloud-a10g`, `cloud-l4` or `x86-cuda` (RTX).
Any other discrete NVIDIA GPU on x86_64 uses `x86-cuda` and says that it has
no cloud profile. `OHHO_TARGET` forces the profile id and still reports the
real CUDA and TensorRT probe.

Apple Silicon, a Raspberry Pi that is not a Pi 5, and a Jetson module that is
not Nano, NX or AGX Orin produce `detected: none` plus a note. CUDA and
TensorRT are still reported.

`ohho optimize` compiles a model for one of these profiles. See
[optimize.md](../optimize.md).
