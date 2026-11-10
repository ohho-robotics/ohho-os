# Serve bench harness

`ohho bench` measures one inference artifact and writes an
[`ohho.bench/v1`](bench-schema.md) result. The schema module is
`ohho.bench.schema`. This command uses that module. It does not define a
second result format.

`--target` is a hardware profile id from `ohho targets` (`x86-cpu`,
`jetson-orin-nano`, ...). The schema field `run.target` is the workload
class (`inference` by default, or `--workload vision`). Those two names are
not the same field.

```bash
ohho bench --artifact model.onnx --target x86-cpu \
  --warmup 10 --iters 50 --batch 1,4,8 --inputs random --json out.json

ohho bench --artifact model.onnx --target x86-cpu --baseline prev.json \
  --slo p95<=200ms --max-regression p95=10% --markdown "$GITHUB_STEP_SUMMARY"

ohho bench validate out.json
```

`--inputs` accepts `random`. Any other value, including a dataset name, exits
2 with `not yet supported`. A registry id (a token that is not a file path)
exits 2 with the same phrase. Pass a `.onnx` file. ONNX Runtime and NumPy are
imported only when that file is loaded (`pip install 'ohho-os[bench]'`).

The result's top-level latency is the primary batch: batch size 1 when `--batch`
includes 1, otherwise the first listed size. `batches` holds every size.
Gates read the primary batch.

## Stats

End-to-end and each stage report mean, median, p50, p90, p95, p99, min, max,
and population std, in milliseconds. `latency.cold_start_ms` is the first
call, before warmup. Warmup iterations are discarded. `latency.achievable_hz`
is `1000 / mean_ms` (0 when the mean is 0). `throughput` uses the same number
with unit `Hz`.

Random inputs have no labels. `accuracy.primary` is `not_measured`.

`telemetry.memory_mb_max` is the process peak RSS when the OS reports it, and
`environment` records the target profile, git SHA, the execution provider that
ran, and CUDA / TensorRT / ONNX Runtime / driver versions when those are
installed. `telemetry.gpu_util_percent` and `telemetry.gpu_power_w` are `null`
when `nvidia-smi` does not return a number.

## Stages

Each sample is `preprocess`, `h2d`, `inference`, `d2h`, `postprocess`.

On CPU the clock is `time.perf_counter`. The harness stamps before and after
each stage, and the end-to-end sample is the sum of those five durations, so
the stage means add up to the end-to-end mean (float rounding only).

When the selected provider is CUDA or TensorRT and PyTorch reports a CUDA
device, `h2d`, `inference`, and `d2h` use CUDA events
(`torch.cuda.Event(enable_timing=True)`). `preprocess` and `postprocess` stay
on `perf_counter`. The published end-to-end figure is still the sum of the
five stage times. A CUDA event does not observe work queued in another
library's CUDA context, so an ONNX Runtime GPU session is on that timeline
only when it shares the device the events record. If a GPU provider was
requested and this process has no CUDA device, the run falls back to
`perf_counter` and sets `environment.timing_note`.

`tests/test_bench_harness.py` checks the CPU sum on a pure-Python runner.
The CUDA event path in that file uses a fake event clock. A development run
without a CUDA device does not execute events on a GPU.

## Baseline and gates

`--baseline prev.json` adds `comparison.metrics`. Each row has `baseline`,
`current`, `delta`, `pct_change` (omitted when the baseline is 0 and the
current value is not), and `regression`. Latency and peak memory regress when
they increase. Achievable Hz and throughput regress when they decrease.

`--slo p95<=200ms` (also `p95<=0.2s`) fails when the primary latency is over
the ceiling. An equal value passes. `--max-regression p95=10%` fails when the
percent change is worse than the limit. An equal limit passes.
`--max-regression` without `--baseline` exits 2. Any failed gate exits 1.
The JSON and Markdown are still written.

`--parity path.json` embeds that JSON on the result. It is not interpreted.

## Reports

`--json` writes the result. Without it, the JSON is printed to stdout.
`--markdown` writes a GitHub step summary (a heading and pipe tables).
`--prometheus-push URL` POSTs a text exposition. `--wandb` logs to Weights &
Biases. Both exporters are off unless the flag is set, and the tests never
require them. A missing `wandb` install errors only when `--wandb` is passed.

## Library

```python
from ohho.bench import BenchConfig, execute

document = execute(
    BenchConfig(
        artifact="model.onnx",
        target="x86-cpu",
        warmup=10,
        iters=50,
        batches=(1, 4, 8),
    )
)
```

A custom runner implements `preprocess`, `h2d`, `inference`, `d2h`, and
`postprocess`. Pass it as `execute(config, runner=runner)`. That is the
surface another package can call.

## Generate mode

`--mode generate` measures an autoregressive model: a language model, a
vision-language model, or action tokens. It does not take `--artifact`.
`--target` is still the hardware profile. `--workload` is `llm`, `vlm`, or
`vla` (an omitted workload is `llm`).

```bash
ohho bench --mode generate --target x86-cpu --workload llm \
  --concurrency 1,4,16 --prompt-tokens 16,64 \
  --precision fp16,bf16,bnb-int4,awq,gptq \
  --slo ttft_p95<=50ms,tpot_p95<=20ms --json out.json

ohho bench --mode generate --target x86-cpu --workload vlm \
  --image-sizes 224,448 --json out.json
```

Each cell records:

| Field | Meaning |
|---|---|
| TTFT | Time to the first output token. mean, median, and p95. |
| TPOT | Time per later output token. mean, median, and p95. |
| end-to-end | First token plus the following tokens. This is the top-level latency. |
| tokens/s per request | Output tokens divided by that request's end-to-end time. |
| tokens/s | All output tokens divided by the wall time of the cell. |

The default sweep is concurrency `1,4,16` and prompt lengths `16,64`.
Concurrency is the batching axis: several requests share the generator.
Static `--batch` stays on inference mode. A `vlm` or `vla` run also records
an image size (`224` unless `--image-sizes` is set). `llm` rejects a
non-zero image size. A cell runs at least as many requests as its
concurrency, so the level is saturated. Higher concurrency raises TTFT and
TPOT and still raises aggregate tokens/s on the stand-in.

`--slo ttft_p95<=50ms` and `--slo tpot_p95<=20ms` read `llm.ttft.p95` and
`llm.tpot.p95`. A comma in one `--slo` is two gates. A failed gate exits 1
after the JSON is written. End-to-end gates such as `p95<=200ms` still apply
to the primary latency.

### Backends

| `--backend` | When |
|---|---|
| `stand-in` (default) | Deterministic CPU generator. No extra packages. |
| `transformers` | Hugging Face causal LM. Needs `torch` and `transformers`, which are not base dependencies. |
| `vllm`, `sglang`, `tensorrt-llm` | External engines from the serving adapters. Only on a target whose accelerator is `dgpu` or `orin_gpu`. |

The stand-in advances a virtual clock (`environment.timing_source` is
`virtual`). FP16, BF16, and FP32 share greedy action tokens. `bnb-int4`,
`awq`, and `gptq` disagree on every fourth token. Agreement is top-1
action-token match against the FP16 sequence. `ohho.parity` supplies that
score when it is installed; otherwise the fraction is computed locally.
`peak_vram_mb` for the stand-in is the weight buffer it allocates
(`vram_source` is `stand-in-weights`): FP16 and BF16 are larger than the
4-bit buffers. That number is not a GPU probe.

`llm.points` is every cell. `llm.precisions` is the comparison at concurrency
1 (or the smallest concurrency), the shortest prompt, and the smallest image.
The required `llm` fields stay the primary cell, so an older reader still
sees TTFT, inter-token latency, and tokens/s. TPOT is also copied to
`llm.inter_token`.

On CPU CI the stand-in is the measured path. The transformers import and the
external engines are not loaded unless that backend is selected. A 4-bit
transformers load needs a CUDA device. GPU runs are outside CI.

## LLM serving sweep

`ohho bench llm --sweep` varies concurrency and the server options the
selected backend can change: max batch tokens, prefix caching, chunked
prefill, and speculative decoding. An option the backend cannot change is
skipped. The result records the reason, and the command still exits 0.

```bash
ohho bench llm --sweep --backend stand-in \
  --concurrency 1,2,4,8 --prefix-cache on,off \
  --target-p95-ttft 40ms \
  --output result.json --chart pareto.svg
```

The grid is crossed with three workloads unless `--workloads` narrows them:

| Workload | What is sent |
|---|---|
| `short` | Distinct short commands and no system prompt. On the stand-in, prefix caching does not change TTFT. |
| `long-context` | A long prompt, the shared system prompt, and a 1x1 image part. |
| `shared-prefix` | The same system prompt on every request, plus a short command. Prefix caching cuts stand-in TTFT and raises tokens/s. |

`--target-p95-ttft` is milliseconds (`40` or `40ms`). The recommendation is
the shared-prefix cell at or under that p95 TTFT with the highest tokens/s.
An equal p95 meets the target. `llm.recommendations` has one row per
workload. `llm.pareto` is the non-dominated set inside each workload: lower
p95 TTFT and higher tokens/s. `--chart` writes an SVG of that scatter.

The CPU backend is `stand-in`. `environment.timing_source` is `virtual`.
Those numbers are a deterministic model of batching and cache hits, not a
GPU measurement. The command does not launch an engine.
`llm.sweep.engine_launched` is false. A one-GPU run is outside this
command. Point `--endpoint` at a server you already started to measure
concurrency and the workloads against it. Launch settings on that process
are skipped, because the sweep does not relaunch it.

Adapter flags for a later launch are in [`llm-backends.md`](llm-backends.md).

## Cost estimate

Every inference run, generate run (`llm`, `vlm`, and `vla`), `ohho bench llm`
measurement, and `ohho bench llm --sweep` writes a `cost` object on the
`ohho.bench/v1` result. The same object is repeated on each batch and on
each LLM cell, using that row's measured throughput. `cost.estimate` is
always true and `cost.label` is `estimate`. `cost.inputs` is the price,
the watts, and the throughput that produced the figure. These numbers are
not an invoice, a cloud bill, or a hardware quote.

The shipped table is `ohho/targets/cost_prices.json`, one row per target
profile id. It is labelled `estimate`. A row with `pricing`
`cloud_gpu_hour` contributes `gpu_hour_usd` per hour. Electricity is not
added on top of that rate. Any other row is `edge_amortised`:
`hardware_usd / amortisation_hours`, plus measured watts times
`electricity_usd_per_kwh`. The built-in amortisation is 26280 hours.
The built-in electricity rate is 0.15 USD/kWh. The built-in control rate
is 50 Hz. When the result has no `telemetry.power_w_mean` or
`telemetry.gpu_power_w`, the row's `typical_watts` is used and
`watts_source` is `profile_default`.

```text
usd_per_hour (edge) = hardware_usd / amortisation_hours
                    + (watts / 1000) * electricity_usd_per_kwh
usd_per_1m_output_tokens = usd_per_hour * 1000000 / (tokens_per_s * 3600)
usd_per_1k_actions = usd_per_hour * 1000 / (actions_per_s * 3600)
usd_per_robot_hour = usd_per_hour * control_rate_hz / actions_per_s
actions_per_watt = actions_per_s / watts
robots_per_device = actions_per_s / control_rate_hz
```

`usd_per_1k_actions` is the cost of 1000 actions at `control_rate_hz` when
the device is shared across `robots_per_device` streams. The control rate
cancels in that USD figure and remains on `meets_control_rate` (true when
measured actions/s are at least the control rate) and on
`usd_per_robot_hour`. Token throughput comes from `tokens/s`. Action
throughput comes from `Hz`, `steps/s`, or `frames/s`. A `vla` generate run
counts one completed request as one action and still prices output tokens.

A result with no target profile uses the `x86-cpu` price row and says so
in `cost.inputs.note`.

Changing a price does not rerun the benchmark. `ohho bench cost` reads a
saved result, replaces `cost`, and leaves latency and throughput as
measured.

```bash
ohho bench cost result.json --gpu-hour-usd 2.5 --output repriced.json
ohho bench cost result.json --target cloud-l4 --prices my-prices.json
ohho bench cost result.json --hardware-usd 400 --amortise-hours 20000 \
  --electricity-usd-per-kwh 0.12 --watts 15 --control-rate-hz 20
ohho bench prices
ohho bench --mode generate --target x86-cpu --gpu-hour-usd 2.5 --json out.json
```

`--gpu-hour-usd` and `--hardware-usd` are mutually exclusive. `--prices`
is a JSON document with `"schema": "ohho.bench.cost_prices/v1"` and
`"estimate": true`. Profile objects in that file replace the named fields
and keep the rest of the built-in row. Pass `--gpu-hour-usd` or
`--hardware-usd` for a profile id that is not in the table.

`rank_by_cost` orders results by `usd_per_1k_actions`,
`usd_per_1m_output_tokens`, or `usd_per_robot_hour`. `max_p95_ms` keeps
rows whose `latency.p95` is at or under that ceiling. Each row is marked
`estimate` and includes `inputs`. The console comparison view that sorts
targets by cost at a latency SLO is a separate console change. It can call
`rank_by_cost` and must show the estimate label and the inputs.

