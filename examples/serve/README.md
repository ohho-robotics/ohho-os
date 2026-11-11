# Serve examples

Short programs for the Serve path. Each directory has a `README.md` (hardware,
runtime, expected output), a `run.py`, and a notebook that runs that script.
CI executes the notebooks with the standard library. Jupyter is not required.

```bash
python examples/serve/run_cpu.py
python examples/serve/run_gpu.py
```

`run_cpu.py` is the release gate. It exits non-zero when any example does.
`.github/workflows/release.yml` runs it before publish. `.github/workflows/ci.yml`
runs it on every pull request. A broken example fails that job.

| Example | Hardware | Runtime | What runs today |
| --- | --- | --- | --- |
| `act_jetson_fp16` | Jetson-class GPU, profile `jetson_single` | under 5 s | Placeholder. `ohho optimize` (OHH-429) is not on main. No FP16 TensorRT, parity, or bench numbers. |
| `ort_int8_cpu` | Raspberry Pi 5 CPU, profile `edge_cpu` | under 5 s | Placeholder. CPU ORT INT8 is the same missing `ohho optimize` command. |
| `vlm_vllm` | GPU host for vLLM | under 5 s | Placeholder. `ohho bench llm` (OHH-436) is not on main. No vLLM call and no token rates. |
| `ci_gate` | CPU | under 5 s | `ohho bench validate` on the schema fixtures, and a check that the release job is wired. |
| `export_wandb_prometheus` | CPU | under 5 s | Placeholder. W&B and Prometheus export are not on main. No metric series is written. |

Commands that exist here and are safe to run: `ohho bench validate` and
`ohho profile` (the scripts only look up `jetson_single` and `edge_cpu`).
`ohho optimize`, the bench harness beyond `validate`, the vLLM bench, and the
metric exporters are not in this tree. Those examples say `status: placeholder`
and `result: none`.

The files under `ohho/bench/examples/` are schema fixtures. The CI gate
validates them and does not print them as measurements.

GPU targets (`act_jetson_fp16`, `vlm_vllm`) also run inside `run_cpu.py`,
because the placeholder needs no GPU. The nightly workflow
`.github/workflows/serve-examples-gpu.yml` runs `run_gpu.py` on a self-hosted
runner labeled `linux` and `gpu` when one is online. If the runners API
reports none, that job is skipped.

The whole CPU suite finishes in under a minute on a laptop. Each example's
own runtime line is the bound for that script.
