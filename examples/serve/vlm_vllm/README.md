# Benchmark a small VLM on vLLM

Placeholder. `ohho bench` in this tree only has `validate`. The LLM bench
command (OHH-436), including a vLLM backend, is not on main. This script
checks for `ohho bench llm` and `ohho bench launch`. It does not start vLLM,
does not download a model, and prints no token-rate or accuracy numbers.

`ohho/bench/examples/llm.json` is a schema fixture. This example does not
print that file and does not treat it as a measured VLM result.

## Hardware

A GPU host that can run vLLM. This run does not use a GPU and does not open
a network connection.

## Runtime

Under 5 seconds. No model is loaded.

## Expected output

```text
example: vlm-vllm
status: placeholder
hardware: GPU host for vLLM
pending: ohho bench llm on vLLM (OHH-436)
runtime: not-run
result: none
```

## Pending

`ohho bench llm` on vLLM. The nightly workflow
`.github/workflows/serve-examples-gpu.yml` runs this script on a self-hosted
runner labeled `linux` and `gpu` when one is online. Until the command exists,
that job prints the same placeholder.
