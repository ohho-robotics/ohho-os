# CPU ORT INT8 on a Raspberry Pi 5

Placeholder. `ohho optimize` is not in this tree (OHH-429). This script checks
that and names the built-in CPU profile `edge_cpu`. It does not quantize an
ACT checkpoint and it prints no benchmark numbers.

`edge_cpu` is the built-in CPU-only profile (a Pi-class board, a mini PC, or
a laptop). `pi_workstation` keeps inference on a separate GPU machine, so it
is not the target of this CPU-only example.

## Hardware

Raspberry Pi 5, CPU only. Built-in profile `edge_cpu`. No accelerator.

## Runtime

Under 5 seconds. The INT8 compile does not run.

## Expected output

```text
example: ort-int8-cpu
status: placeholder
hardware: Raspberry Pi 5 CPU (profile edge_cpu)
pending: ohho optimize (OHH-429) CPU ORT INT8
runtime: not-run
result: none
```

## Pending

CPU ONNX Runtime INT8 export. When `ohho optimize` is on main, this example
has to be pointed at that command. It will not invent flags or results before
then.
