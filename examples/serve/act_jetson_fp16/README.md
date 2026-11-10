# ACT on a Jetson-class board (FP16 TensorRT)

Placeholder. `ohho optimize` is not in this tree (OHH-429). This script checks
that, names the built-in `jetson_single` profile, and stops. It does not
compile an ACT checkpoint, compare a parity set, or run the bench harness.
It prints no latency, agreement, or throughput numbers.

The ACT checkpoint format already lives in `ohho.train.act` and
`ohho.serve.act`. Pointing this example at a real `ohho optimize` invocation
waits until that command is on main.

## Hardware

Jetson-class GPU. Built-in profile `jetson_single`. This run does not use a GPU.

## Runtime

Under 5 seconds. The compile, parity check, and bench do not run.

## Expected output

```text
example: act-jetson-fp16
status: placeholder
hardware: Jetson-class GPU (profile jetson_single)
pending: ohho optimize (OHH-429) FP16 TensorRT, parity, and the bench harness
runtime: not-run
result: none
```

## Pending

FP16 TensorRT compile, a parity check, and a bench run. The nightly workflow
`.github/workflows/serve-examples-gpu.yml` runs this script on a self-hosted
runner labeled `linux` and `gpu` when one is online. Until the command exists,
that job prints the same placeholder.
