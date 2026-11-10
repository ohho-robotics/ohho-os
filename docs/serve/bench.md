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

## Pipeline (`--pipeline`)

`ohho bench --pipeline` times a camera frame through seven stages and writes
the same [`ohho.bench/v1`](bench-schema.md) document plus a `pipeline` object
whose `version` is `ohho.bench.pipeline/v1`.

```bash
ohho bench --pipeline --hz 30 --json out.json
ohho bench --pipeline --fixture stream.json --sink inprocess --warmup 0 --iters 8
ohho bench --pipeline --dataset ./dataset --episode 0 --hz 30
ohho bench --pipeline --bag ./recording --hz 30
```

Stages, in order: `decode`, `preprocess`, `h2d`, `inference`, `d2h`,
`postprocess`, `publish`. End-to-end latency is the sum of those seven.
`latency.p50`, `p95`, and `p99` are that sum. `pipeline.jitter_ms` is
`latency.std`. `latency.achievable_hz` stays "iterations per second implied
by the mean latency". The paced control rate is `pipeline.achieved_hz` and
`throughput` (`frames/s`).

The default source is the in-process simulator (`--source sim`, `--robot sim`).
Each tick reads a small RGB frame from that sim, runs the built-in `tiny-cpu`
policy, and publishes a clamped body velocity back into the sim. The result
is labelled `pipeline.world = sim` and `environment.world = sim`. It does not
train a model and it does not open hardware. A LeRobot episode recorded by
`record_sim_episodes` can be replayed with `--dataset`; an episode with no
stored camera is decoded from a state raster and marked
`pipeline.frame_origin = state-raster`.

`--fixture` reads `ohho.pipeline.stream/v1` (a JSON list of base64 frames).
`--bag` reads `ohho.pipeline.rosbag/v1` (`metadata.json` plus `frames.jsonl`,
or one JSON file). A ROS 2 bag directory or an `.mcap` file is refused: this
bench does not download rosbag2 and does not decode `sensor_msgs/Image`.

`publish` goes to a sink. `sim` calls `Robot.drive` on the in-process
simulator. `inprocess` records the action. `ros2` publishes
`geometry_msgs/Twist` and imports `rclpy` only when that sink is selected.
`postprocess` clamps vx, vy, and yaw rate to the robot spec (`--robot`,
default `sim`) before the sink sees them. Non-holonomic vy is zero.

Frames are paced at `--hz` (default 30). A frame is dropped when its
control period has already elapsed before the pipeline can start it.
`pipeline.rate_sustained` is true when nothing was dropped and every
published frame finished inside its period. `pipeline.dropped_frames` is
the skip count.

H2D and D2H are host copies. `pipeline.device` is `cpu` and
`pipeline.gpu_follow_up` lists `h2d` and `d2h`. CUDA copies are not timed.

`--source hardware` is refused unless `--allow-hardware` is set. With that
flag the bench still only replays `--fixture`, `--dataset`, or `--bag`.
`pipeline.live_robot` is false. A `serial://`, `dds://`, or other device URI
is refused. This command does not arm a robot and does not set
`OHHO_ARM_HARDWARE`.

`--artifact` stays the five-stage ONNX bench. Do not pass it with
`--pipeline`.
