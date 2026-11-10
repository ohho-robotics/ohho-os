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

## Power, memory, and thermal

During the measured iterations (after warmup) a sampler thread reads optional
sensors at 10 Hz. The first sample is taken when the window opens and another
when it closes. Each source is detected at runtime and skipped when it is
not there.

| Source | What it can report |
| --- | --- |
| NVML | Accelerator power, device memory, temperature, clocks, throttle bits |
| tegrastats, or jtop when that binary is absent | Board input power, host RAM, temperature, clocks, throttling |
| RAPL package energy | CPU package power, from two counter readings |
| vcgencmd | Temperature, clocks, and the current throttled bits. Not watts |

Headline power prefers board input, then accelerator power, then package
power. Those domains are not added together. tegrastats power is the
instantaneous `VDD_IN` rail, or `POM_5V_IN` when that is the input rail.
Other rails are left out of the headline. vcgencmd voltage is not turned
into watts.

`telemetry.sensors` always carries:

| Field | Meaning |
| --- | --- |
| `power_w_mean`, `power_w_peak` | Headline power in watts, or null |
| `energy_per_inference_mj` | Mean power times mean latency in milliseconds |
| `device_memory_mb_peak` | Peak accelerator memory, or null |
| `host_memory_mb_peak` | Peak host memory from a sensor, otherwise process peak RSS |
| `temperature_c_max` | Hottest reading, or null |
| `throttling_events` | Rising edges. Each has `source`, `t_s`, and `detail` |
| `throttling_flagged` | True when any event was recorded |
| `clocks_mhz` | Max observed clock per source and name, or null |
| `reason` | Why a headline field is null. Null when every field was measured |
| `sources` | One row per source: `id`, `available`, `reason`, `power_domain` |

A missing sensor is null with a reason. The run still validates and exits 0
when no gate fails. `telemetry.power_w_mean` and
`telemetry.accelerator_memory_mb_max` are set only for real numbers, because
ohho.bench/v1 defines those gauges as non-null.

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

## Sustained

`--sustained 30` measures for 30 seconds instead of `--iters`. Warmup still
runs first and is discarded. The result's `sustained` object lists p95 for
each time window (one second when the run is at least two seconds, otherwise
two equal windows), `p95_drift_ms` (last window minus first), and
`throttling_flagged`. Each requested batch size gets its own window. The
top-level `sustained` object is the primary batch.

```bash
ohho bench --artifact model.onnx --target x86-cpu --sustained 30 --json out.json
```

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
