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
    "CostInputs",
    "BenchCost",
    "BatchResult",
    "LlmParetoPoint",
    "LlmPoint",
    "LlmPrecision",
    "LlmRecommendation",
    "LlmSkip",
    "LlmSweep",
    "LlmBench",
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


class _CostInputsRequired(TypedDict):
    target_profile: str
    profile_source: Literal["run", "flag", "default"]
    pricing: Literal["cloud_gpu_hour", "edge_amortised"]
    price_source: Literal["built-in", "file", "flag"]
    control_rate_hz: float
    watts_source: Literal["measured", "profile_default", "user", "unmeasured"]
    electricity_usd_per_kwh: float
    electricity_applied: bool
    throughput_value: float
    throughput_unit: str
    note: str


class CostInputs(_CostInputsRequired, total=False):
    """Price inputs used for one estimate. Edit them and recompute without
    rerunning the benchmark. Every cost is an estimate, not an invoice.

    Additional properties are allowed.
    """

    measured_target_profile: str
    pricing_detail: str
    watts: float
    gpu_hour_usd: float
    hardware_usd: float
    amortisation_hours: float
    hardware_usd_per_hour: float
    electricity_usd_per_hour: float
    tokens_per_s: float
    actions_per_s: float


class _BenchCostRequired(TypedDict):
    schema: Literal["ohho.bench.cost/v1"]
    estimate: Literal[True]
    label: Literal["estimate"]
    inputs: CostInputs
    usd_per_hour: float


class BenchCost(_BenchCostRequired, total=False):
    """Estimated USD from measured throughput and a user-editable price. estimate
    is always true. label is estimate. These figures are not an invoice, a cloud
    bill, or a hardware quote.

    Additional properties are allowed.
    """

    usd_per_1m_output_tokens: float
    usd_per_1k_actions: float
    usd_per_robot_hour: float
    actions_per_watt: float
    meets_control_rate: bool
    robots_per_device: float


class _BatchResultRequired(TypedDict):
    batch_size: int
    latency: LatencyReport
    throughput: Throughput


class BatchResult(_BatchResultRequired, total=False):
    """Latency and throughput for one batch size.

    Additional properties are allowed.
    """

    cost: BenchCost


class _LlmParetoPointRequired(TypedDict):
    workload: str
    ttft_p95_ms: float
    tokens_per_s: float


class LlmParetoPoint(_LlmParetoPointRequired, total=False):
    """One non-dominated cell: p95 time to first token against tokens/s.

    Additional properties are allowed.
    """

    concurrency: int
    prompt_tokens: int
    image_tokens: int
    max_batch_tokens: int
    prefix_cache: bool
    chunked_prefill: bool
    speculative_decoding: bool
    cost: BenchCost


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
    workload: str
    prefix_cache: bool
    max_batch_tokens: int
    chunked_prefill: bool
    speculative_decoding: bool
    pareto: bool
    cost: BenchCost


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
    cost: BenchCost


class _LlmRecommendationRequired(TypedDict):
    target_p95_ttft_ms: float
    met: bool
    workload: str
    reason: str


class LlmRecommendation(_LlmRecommendationRequired, total=False):
    """The configuration at or under a p95 TTFT target with the highest tokens/s.
    met is false when no cell qualifies.

    Additional properties are allowed.
    """

    concurrency: int
    max_batch_tokens: int
    prefix_cache: bool
    chunked_prefill: bool
    speculative_decoding: bool
    ttft_p95_ms: float
    tokens_per_s: float
    cost: BenchCost


class LlmSkip(TypedDict):
    """One sweep option that was not measured, with the reason. A skip does not
    fail the run.

    Additional properties are allowed.
    """

    option: str
    reason: str


class _LlmSweepRequired(TypedDict):
    workloads: list[str]
    cells: int
    timing_source: str
    engine_launched: bool


class LlmSweep(_LlmSweepRequired, total=False):
    """Batching, prefix-cache, chunked-prefill, and speculative-decoding grid.
    engine_launched is false when this command did not start an engine.

    Additional properties are allowed.
    """

    note: str
    primary: str
    chart: str
    concurrencies: list[int]
    max_batch_tokens: list[int]
    prefix_cache: list[bool]
    chunked_prefill: list[bool]
    speculative_decoding: list[bool]


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
    sweep. Optional sweep, pareto, recommendation, recommendations, and skipped
    record an llm --sweep run.

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
    sweep: LlmSweep
    pareto: list[LlmParetoPoint]
    recommendation: LlmRecommendation
    recommendations: list[LlmRecommendation]
    skipped: list[LlmSkip]


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
    cost: BenchCost
