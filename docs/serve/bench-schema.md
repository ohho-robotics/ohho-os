# ohho.bench/v1 result schema

The Serve bench harness, the results database, CI, and metric exporters all
read one document. That document is `ohho.bench/v1`. The JSON Schema is the
contract:

`ohho/bench/schemas/bench.v1.json`

`$id` is `https://ohho-robotics.com/schemas/ohho.bench/v1.json`. The file is
package data on `ohho.bench`, so a later console ingestion API can load the
same bytes the bench wrote.

```python
from importlib.resources import files

text = files("ohho.bench").joinpath("schemas/bench.v1.json").read_bytes()
```

The result is robot-agnostic. It names a workload (`target`), not a robot
product.

## Result

| Path | Required | Meaning |
|---|---|---|
| `schema` | yes | Constant `ohho.bench/v1`. |
| `run.model_sha` | yes | Digest of the model weights. |
| `run.artifact_sha` | yes | Digest of the artifact that ran. |
| `run.target` | yes | Workload class (`vision`, `vla`, `llm`, or a later name). |
| `run.manifest_ref` | yes | Manifest path, URI, or digest. |
| `run.host.name` | yes | Host or CI runner id. |
| `latency.mean`, `median`, `p50`, `p90`, `p95`, `p99`, `min`, `max`, `std` | yes | End-to-end latency. Non-negative. |
| `latency.unit` | yes | Scale for those numbers. Checked-in examples use `ms`. |
| `latency.stages` | yes | Per-stage distributions. `[]` means no breakdown. |
| `latency.stages[].name` | yes | Stage name, plus the same latency fields as the parent. |
| `throughput.value`, `throughput.unit` | yes | Work per time (`frames/s`, `steps/s`, `tokens/s`, ...). |
| `accuracy.primary.name`, `accuracy.primary.value` | yes | Primary quality metric. |
| `telemetry.duration_s`, `telemetry.samples` | yes | How long the run was observed, and how many samples. |
| `environment.versions.python`, `environment.versions.ohho` | yes | Interpreter version and ohho-os version. |

Digests are hex, 7 to 128 digits, with an optional `sha256:` prefix. Shorter
git ids and full sha256 digests both fit.

Optional fields already named in the schema (`run.id`, `latency.samples`,
`throughput.batch_size`, `accuracy.metrics`, resource gauges on `telemetry`,
host `platform` / `arch` / `cpu` / `accelerator`, and extra version keys)
may be omitted. Extra version keys are strings.

The harness (see [`bench.md`](bench.md)) adds more optional v1 fields. None of
them are required, and none change the type of a field v1 already defined:

| Path | Meaning |
|---|---|
| `latency.cold_start_ms` | First iteration before warmup, in milliseconds. |
| `latency.achievable_hz` | Iterations per second from the mean latency. |
| `environment.target_profile` | Hardware profile id (`x86-cpu`, `jetson-orin-nano`, ...). |
| `environment.git_sha` | Git commit, or `unknown`. |
| `environment.timing_source` | `perf_counter`, `cuda_event`, or `virtual` for a scripted clock. |
| `environment.execution_provider` | ONNX Runtime provider that actually ran. |
| `environment.driver` | GPU driver version when a driver is visible. |
| `comparison` | Per-metric delta against a baseline result. |
| `gates` | SLO and regression checks. `passed` is false when any check failed. |
| `batches` | One latency report per requested batch size. |

`telemetry.gpu_util_percent` and `telemetry.gpu_power_w` are additional
properties and are JSON `null` when a GPU sample is not available. The typed
gauges `accelerator_util_mean` and `power_w_mean` stay non-null numbers, so
they are omitted in that case. Making those gauges nullable would be an
`ohho.bench/v2` change. `telemetry.memory_mb_max` is the peak process
memory when it can be read.

`telemetry.sensors` is an additional object written by the harness. Power,
energy, device memory, temperature, and clocks inside it are JSON `null`
when no source reported them, and `reason` says why. `throttling_events` is
an array. `sustained` is an additional top-level object present only for
`--sustained`. Both stay on v1 because they do not change a field v1 already
defined.

Checked-in results:

- `ohho/bench/examples/vision.json` — vision policy
- `ohho/bench/examples/vla.json` — VLA
- `ohho/bench/examples/llm.json` — LLM

## Versioning

v1 readers accept additional properties on every object. A new optional
property, a new `target` string, a new metric name, a new stage name, and a
new component in `environment.versions` stay on v1. Writers still send every
required field. Readers ignore fields they do not know.

A v2 change is any of the following:

- removing a field v1 defined
- renaming a field
- changing a field's type
- tightening a constraint (`minimum`, `pattern`, `minLength`, `enum`, `const`)
- changing the meaning of an existing field

That change is published as `ohho.bench/v2` (a new schema document) and a
converter from the previous id. The `schema` field is the discriminator.

```python
from ohho.bench import SCHEMA_ID, convert

same = convert(document, to=SCHEMA_ID)  # deep copy; extra fields kept
```

`convert(document, to="ohho.bench/v2")` raises `BenchConversionError` until
that converter is registered. Identity conversion checks `schema` and copies
the document. It is not a second validator.

## Validate

```bash
ohho bench validate path/to/result.json
```

Exit 0 prints `valid ohho.bench/v1: <path>`. Exit 1 prints one JSON path per
problem, and lists every problem rather than stopping at the first:

```text
$.run.model_sha: required property is missing
$.latency.stages[0].p99: required property is missing
```

Library entry points are `ohho.bench.validate` and `ohho.bench.validate_path`.
`jsonschema` is not installed with the base package. The checker in
`ohho/bench/schema.py` implements the draft 2020-12 keywords this schema
uses:

`$schema`, `$id`, `$defs`, `$ref`, `$comment`, `title`, `description`,
`default`, `examples`, `type`, `properties`, `required`, `items`,
`additionalProperties`, `allOf`, `const`, `enum`, `minimum`, `maximum`,
`exclusiveMinimum`, `exclusiveMaximum`, `minLength`, `maxLength`, `minItems`,
`maxItems`, `pattern`.

`$ref` resolves only against the root `$defs`. A keyword outside that list
fails validation, so CI cannot go green on a constraint the checker ignores.
Other languages can still validate the same JSON Schema with a full
draft 2020-12 library.

Whole-number JSON values count as integers (`1` and `1.0`). Booleans do not
count as numbers. Non-finite values are rejected.

## Generated types

Python and TypeScript types are generated from the schema. Edit the schema,
then regenerate. The generated files are committed.

```bash
python scripts/generate_bench_types.py
python scripts/generate_bench_types.py --check
```

| Output | Role |
|---|---|
| `ohho/bench/generated/types.py` | `TypedDict` view, import `ohho.bench.generated.types` |
| `ohho/bench/generated/bench.v1.ts` | TypeScript interfaces for the same document |

`BENCH_SCHEMA_ID` in both outputs is the schema's `const`. The test suite
runs `--check` and fails when the generated files drift.
