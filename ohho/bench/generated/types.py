# Generated from ohho/bench/schemas/bench.v1.json by scripts/generate_bench_types.py.
# Do not edit. Regenerate with: python scripts/generate_bench_types.py
"""Typed view of an ohho.bench/v1 benchmark result.

Generated from ``ohho/bench/schemas/bench.v1.json``. Additional object fields
are allowed. Validate with :func:`ohho.bench.schema.validate`.
"""

from __future__ import annotations

from typing import Literal, TypedDict

BENCH_SCHEMA_ID: Literal["ohho.bench/v1"] = "ohho.bench/v1"

__all__ = [
    "BENCH_SCHEMA_ID",
    "BenchHost",
    "BenchRun",
    "Metric",
    "LatencyStats",
    "LatencyStage",
    "LatencyReport",
    "Throughput",
    "Accuracy",
    "Telemetry",
    "EnvironmentVersions",
    "Environment",
    "ComparisonMetric",
    "Comparison",
    "GateCheck",
    "Gates",
    "BatchResult",
    "LlmPoint",
    "LlmPrecision",
    "LlmBench",
    "PipelineBench",
    "BenchResult",
]


class _BenchHostRequired(TypedDict):
    name: str


class BenchHost(_BenchHostRequired, total=False):
    """Machine that executed the run.

    Additional properties are allowed.
    """

    platform: str
    arch: str
    cpu: str
    accelerator: str


class _BenchRunRequired(TypedDict):
    model_sha: str
    artifact_sha: str
    target: str
    manifest_ref: str
    host: BenchHost


class BenchRun(_BenchRunRequired, total=False):
    """Identity of the model, artifact, workload, manifest, and host.

    Additional properties are allowed.
    """

    id: str
    started_at: str
    finished_at: str


class _MetricRequired(TypedDict):
    name: str
    value: float


class Metric(_MetricRequired, total=False):
    """One named accuracy figure.

    Additional properties are allowed.
    """

    unit: str


class _LatencyStatsRequired(TypedDict):
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    std: float
    unit: str


class LatencyStats(_LatencyStatsRequired, total=False):
    """Latency distribution for one sample set.

    Additional properties are allowed.
    """

    samples: int


class _LatencyStageRequired(TypedDict):
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    std: float
    unit: str
    name: str


class LatencyStage(_LatencyStageRequired, total=False):
    """Latency distribution for one named stage.

    Additional properties are allowed.
    """

    samples: int


class _LatencyReportRequired(TypedDict):
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    std: float
    unit: str
    stages: list[LatencyStage]


class LatencyReport(_LatencyReportRequired, total=False):
    """End-to-end latency and the per-stage breakdown.

    Additional properties are allowed.
    """

    samples: int
    cold_start_ms: float
    achievable_hz: float


class _ThroughputRequired(TypedDict):
    value: float
    unit: str


class Throughput(_ThroughputRequired, total=False):
    """Work completed per unit of time.

    Additional properties are allowed.
    """

    samples: int
    batch_size: int
    duration_s: float


class _AccuracyRequired(TypedDict):
    primary: Metric


class Accuracy(_AccuracyRequired, total=False):
    """Quality of the model on the bench set.

    Additional properties are allowed.
    """

    metrics: list[Metric]
    samples: int


class _TelemetryRequired(TypedDict):
    duration_s: float
    samples: int


class Telemetry(_TelemetryRequired, total=False):
    """Resource and runtime measurements for the run.

    Additional properties are allowed.
    """

    cpu_percent_mean: float
    memory_mb_max: float
    accelerator_memory_mb_max: float
    accelerator_util_mean: float
    power_w_mean: float
    memory_source: str


class EnvironmentVersions(TypedDict):
    """Component name to version string. python and ohho are required.

    Additional properties are allowed and must be strings.
    """

    python: str
    ohho: str


class _EnvironmentRequired(TypedDict):
    versions: EnvironmentVersions


class Environment(_EnvironmentRequired, total=False):
    """Software environment the run executed in.

    Additional properties are allowed.
    """

    target_profile: str
    git_sha: str
    timing_source: str
    timing_note: str
    execution_provider: str
    driver: str
    world: Literal["sim", "recorded", "hardware"]


class _ComparisonMetricRequired(TypedDict):
    name: str
    baseline: float
    current: float
    delta: float
    regression: bool


class ComparisonMetric(_ComparisonMetricRequired, total=False):
    """One baseline delta. Latency metrics treat an increase as a regression.

    Additional properties are allowed.
    """

    unit: str
    pct_change: float
    higher_is_worse: bool


class Comparison(TypedDict):
    """Per-metric delta against a baseline ohho.bench/v1 result.

    Additional properties are allowed.
    """

    baseline_ref: str
    metrics: list[ComparisonMetric]


class _GateCheckRequired(TypedDict):
    name: str
    expr: str
    passed: bool


class GateCheck(_GateCheckRequired, total=False):
    """One SLO or regression gate.

    Additional properties are allowed.
    """

    metric: str
    value: float
    limit: float
    detail: str


class Gates(TypedDict):
    """SLO and regression gates for the primary batch.

    Additional properties are allowed.
    """

    passed: bool
    checks: list[GateCheck]


class BatchResult(TypedDict):
    """Latency and throughput for one batch size.

    Additional properties are allowed.
    """

    batch_size: int
    latency: LatencyReport
    throughput: Throughput


class _LlmPointRequired(TypedDict):
    concurrency: int
    prompt_tokens: int
    image_tokens: int
    precision: str
    ttft: LatencyStats
    tpot: LatencyStats
    e2e: LatencyStats
    tokens_per_s: float
    tokens_per_request_s: float


class LlmPoint(_LlmPointRequired, total=False):
    """One generate-mode cell: latency, throughput, and optional memory and
    agreement.

    Additional properties are allowed.
    """

    image_size: int
    requests_per_s: float
    requests: int
    output_tokens: int
    peak_vram_mb: float
    vram_source: str
    agreement: Metric


class _LlmPrecisionRequired(TypedDict):
    precision: str
    ttft: LatencyStats
    tpot: LatencyStats
    e2e: LatencyStats
    tokens_per_s: float
    tokens_per_request_s: float


class LlmPrecision(_LlmPrecisionRequired, total=False):
    """One precision compared at the primary prompt, image size, and concurrency.

    Additional properties are allowed.
    """

    peak_vram_mb: float
    vram_source: str
    agreement: Metric
    concurrency: int
    prompt_tokens: int
    image_tokens: int


class _LlmBenchRequired(TypedDict):
    schema: Literal["ohho.bench.llm/v1"]
    backend: str
    concurrency: int
    ttft: LatencyStats
    inter_token: LatencyStats
    tokens_per_s: float
    requests_per_s: float


class LlmBench(_LlmBenchRequired, total=False):
    """LLM serving measurements (ohho.bench.llm/v1). Optional on the parent
    ohho.bench/v1 result. Required fields are the primary point: time to first
    token, inter-token latency, token throughput, and request throughput at one
    concurrency. Optional tpot, points, and precisions record a generate-mode
    sweep.

    Additional properties are allowed.
    """

    backend_version: str
    precision: str
    model: str
    model_sha_of: str
    prompt: str
    output_tokens: int
    requests: int
    requests_ok: int
    workload: str
    tpot: LatencyStats
    e2e: LatencyStats
    tokens_per_request_s: float
    prompt_tokens: int
    image_tokens: int
    peak_vram_mb: float
    vram_source: str
    agreement: Metric
    points: list[LlmPoint]
    precisions: list[LlmPrecision]


class _PipelineBenchRequired(TypedDict):
    version: Literal["ohho.bench.pipeline/v1"]
    world: Literal["sim", "recorded", "hardware"]
    source: str
    target_hz: float
    achieved_hz: float
    rate_sustained: bool
    dropped_frames: int
    frames_presented: int
    frames_published: int
    jitter_ms: float
    device: Literal["cpu", "cuda"]
    live_robot: bool


class PipelineBench(_PipelineBenchRequired, total=False):
    """End-to-end camera-to-action pipeline (ohho.bench.pipeline/v1). Additive on
    ohho.bench/v1: a new field here stays on v1, and a break is a new pipeline
    version plus a converter.

    Additional properties are allowed.
    """

    interval_jitter_ms: float
    control_period_ms: float
    sink: str
    gpu_follow_up: list[str]
    gpu_note: str
    policy: str
    episode: int
    frame_origin: str
    recording: str


class _BenchResultRequired(TypedDict):
    schema: Literal["ohho.bench/v1"]
    run: BenchRun
    latency: LatencyReport
    throughput: Throughput
    accuracy: Accuracy
    telemetry: Telemetry
    environment: Environment


class BenchResult(_BenchResultRequired, total=False):
    """Serve benchmark result (ohho.bench/v1). Objects allow additional properties.
    Removing, renaming, or retyping a field requires ohho.bench/v2 and a
    converter.

    Additional properties are allowed.
    """

    comparison: Comparison
    gates: Gates
    batches: list[BatchResult]
    llm: LlmBench
    pipeline: PipelineBench
