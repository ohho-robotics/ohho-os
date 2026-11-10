// Generated from ohho/bench/schemas/bench.v1.json by scripts/generate_bench_types.py.
// Do not edit. Regenerate with: python scripts/generate_bench_types.py
// Known fields are listed below. ohho.bench/v1 objects also allow additional fields.

/** Schema id for an ohho.bench/v1 result. */
export const BENCH_SCHEMA_ID = "ohho.bench/v1" as const;

/**
 * Machine that executed the run.
 * Additional properties are allowed.
 */
export interface BenchHost {
  name: string;
  platform?: string;
  arch?: string;
  cpu?: string;
  accelerator?: string;
}

/**
 * Identity of the model, artifact, workload, manifest, and host.
 * Additional properties are allowed.
 */
export interface BenchRun {
  model_sha: string;
  artifact_sha: string;
  target: string;
  manifest_ref: string;
  host: BenchHost;
  id?: string;
  started_at?: string;
  finished_at?: string;
}

/**
 * One named accuracy figure.
 * Additional properties are allowed.
 */
export interface Metric {
  name: string;
  value: number;
  unit?: string;
}

/**
 * Latency distribution for one sample set.
 * Additional properties are allowed.
 */
export interface LatencyStats {
  mean: number;
  median: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  std: number;
  unit: string;
  samples?: number;
}

/**
 * Latency distribution for one named stage.
 * Additional properties are allowed.
 */
export interface LatencyStage {
  mean: number;
  median: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  std: number;
  unit: string;
  samples?: number;
  name: string;
}

/**
 * End-to-end latency and the per-stage breakdown.
 * Additional properties are allowed.
 */
export interface LatencyReport {
  mean: number;
  median: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  std: number;
  unit: string;
  samples?: number;
  stages: LatencyStage[];
  cold_start_ms?: number;
  achievable_hz?: number;
}

/**
 * Work completed per unit of time.
 * Additional properties are allowed.
 */
export interface Throughput {
  value: number;
  unit: string;
  samples?: number;
  batch_size?: number;
  duration_s?: number;
}

/**
 * Quality of the model on the bench set.
 * Additional properties are allowed.
 */
export interface Accuracy {
  primary: Metric;
  metrics?: Metric[];
  samples?: number;
}

/**
 * Resource and runtime measurements for the run.
 * Additional properties are allowed.
 */
export interface Telemetry {
  duration_s: number;
  samples: number;
  cpu_percent_mean?: number;
  memory_mb_max?: number;
  accelerator_memory_mb_max?: number;
  accelerator_util_mean?: number;
  power_w_mean?: number;
  memory_source?: string;
}

/**
 * Component name to version string. python and ohho are required.
 * Additional properties are allowed and must be strings.
 */
export interface EnvironmentVersions {
  python: string;
  ohho: string;
  [key: string]: string;
}

/**
 * Software environment the run executed in.
 * Additional properties are allowed.
 */
export interface Environment {
  versions: EnvironmentVersions;
  target_profile?: string;
  git_sha?: string;
  timing_source?: string;
  timing_note?: string;
  execution_provider?: string;
  driver?: string;
}

/**
 * One baseline delta. Latency metrics treat an increase as a regression.
 * Additional properties are allowed.
 */
export interface ComparisonMetric {
  name: string;
  unit?: string;
  baseline: number;
  current: number;
  delta: number;
  pct_change?: number;
  regression: boolean;
  higher_is_worse?: boolean;
}

/**
 * Per-metric delta against a baseline ohho.bench/v1 result.
 * Additional properties are allowed.
 */
export interface Comparison {
  baseline_ref: string;
  metrics: ComparisonMetric[];
}

/**
 * One SLO or regression gate.
 * Additional properties are allowed.
 */
export interface GateCheck {
  name: string;
  expr: string;
  passed: boolean;
  metric?: string;
  value?: number;
  limit?: number;
  detail?: string;
}

/**
 * SLO and regression gates for the primary batch.
 * Additional properties are allowed.
 */
export interface Gates {
  passed: boolean;
  checks: GateCheck[];
}

/**
 * Price inputs used for one estimate. Edit them and recompute without
 * rerunning the benchmark. Every cost is an estimate, not an invoice.
 * Additional properties are allowed.
 */
export interface CostInputs {
  target_profile: string;
  profile_source: "run" | "flag" | "default";
  measured_target_profile?: string;
  pricing: "cloud_gpu_hour" | "edge_amortised";
  price_source: "built-in" | "file" | "flag";
  pricing_detail?: string;
  control_rate_hz: number;
  watts?: number;
  watts_source: "measured" | "profile_default" | "user" | "unmeasured";
  gpu_hour_usd?: number;
  hardware_usd?: number;
  amortisation_hours?: number;
  hardware_usd_per_hour?: number;
  electricity_usd_per_kwh: number;
  electricity_usd_per_hour?: number;
  electricity_applied: boolean;
  tokens_per_s?: number;
  actions_per_s?: number;
  throughput_value: number;
  throughput_unit: string;
  note: string;
}

/**
 * Estimated USD from measured throughput and a user-editable price. estimate
 * is always true. label is estimate. These figures are not an invoice, a cloud
 * bill, or a hardware quote.
 * Additional properties are allowed.
 */
export interface BenchCost {
  schema: "ohho.bench.cost/v1";
  estimate: true;
  label: "estimate";
  inputs: CostInputs;
  usd_per_hour: number;
  usd_per_1m_output_tokens?: number;
  usd_per_1k_actions?: number;
  usd_per_robot_hour?: number;
  actions_per_watt?: number;
  meets_control_rate?: boolean;
  robots_per_device?: number;
}

/**
 * Latency and throughput for one batch size.
 * Additional properties are allowed.
 */
export interface BatchResult {
  batch_size: number;
  latency: LatencyReport;
  throughput: Throughput;
  cost?: BenchCost;
}

/**
 * One non-dominated cell: p95 time to first token against tokens/s.
 * Additional properties are allowed.
 */
export interface LlmParetoPoint {
  workload: string;
  concurrency?: number;
  ttft_p95_ms: number;
  tokens_per_s: number;
  prompt_tokens?: number;
  image_tokens?: number;
  max_batch_tokens?: number;
  prefix_cache?: boolean;
  chunked_prefill?: boolean;
  speculative_decoding?: boolean;
  cost?: BenchCost;
}

/**
 * One generate-mode cell: latency, throughput, and optional memory and
 * agreement.
 * Additional properties are allowed.
 */
export interface LlmPoint {
  concurrency: number;
  prompt_tokens: number;
  image_tokens: number;
  image_size?: number;
  precision: string;
  ttft: LatencyStats;
  tpot: LatencyStats;
  e2e: LatencyStats;
  tokens_per_s: number;
  tokens_per_request_s: number;
  requests_per_s?: number;
  requests?: number;
  output_tokens?: number;
  peak_vram_mb?: number;
  vram_source?: string;
  agreement?: Metric;
  workload?: string;
  prefix_cache?: boolean;
  max_batch_tokens?: number;
  chunked_prefill?: boolean;
  speculative_decoding?: boolean;
  pareto?: boolean;
  cost?: BenchCost;
}

/**
 * One precision compared at the primary prompt, image size, and concurrency.
 * Additional properties are allowed.
 */
export interface LlmPrecision {
  precision: string;
  ttft: LatencyStats;
  tpot: LatencyStats;
  e2e: LatencyStats;
  tokens_per_s: number;
  tokens_per_request_s: number;
  peak_vram_mb?: number;
  vram_source?: string;
  agreement?: Metric;
  concurrency?: number;
  prompt_tokens?: number;
  image_tokens?: number;
  cost?: BenchCost;
}

/**
 * The configuration at or under a p95 TTFT target with the highest tokens/s.
 * met is false when no cell qualifies.
 * Additional properties are allowed.
 */
export interface LlmRecommendation {
  target_p95_ttft_ms: number;
  met: boolean;
  workload: string;
  reason: string;
  concurrency?: number;
  max_batch_tokens?: number;
  prefix_cache?: boolean;
  chunked_prefill?: boolean;
  speculative_decoding?: boolean;
  ttft_p95_ms?: number;
  tokens_per_s?: number;
  cost?: BenchCost;
}

/**
 * One sweep option that was not measured, with the reason. A skip does not
 * fail the run.
 * Additional properties are allowed.
 */
export interface LlmSkip {
  option: string;
  reason: string;
}

/**
 * Batching, prefix-cache, chunked-prefill, and speculative-decoding grid.
 * engine_launched is false when this command did not start an engine.
 * Additional properties are allowed.
 */
export interface LlmSweep {
  workloads: string[];
  cells: number;
  timing_source: string;
  engine_launched: boolean;
  note?: string;
  primary?: string;
  chart?: string;
  concurrencies?: number[];
  max_batch_tokens?: number[];
  prefix_cache?: boolean[];
  chunked_prefill?: boolean[];
  speculative_decoding?: boolean[];
}

/**
 * LLM serving measurements (ohho.bench.llm/v1). Optional on the parent
 * ohho.bench/v1 result. Required fields are the primary point: time to first
 * token, inter-token latency, token throughput, and request throughput at one
 * concurrency. Optional tpot, points, and precisions record a generate-mode
 * sweep. Optional sweep, pareto, recommendation, recommendations, and skipped
 * record an llm --sweep run.
 * Additional properties are allowed.
 */
export interface LlmBench {
  schema: "ohho.bench.llm/v1";
  backend: string;
  backend_version?: string;
  precision?: string;
  model?: string;
  model_sha_of?: string;
  prompt?: string;
  concurrency: number;
  ttft: LatencyStats;
  inter_token: LatencyStats;
  tokens_per_s: number;
  requests_per_s: number;
  output_tokens?: number;
  requests?: number;
  requests_ok?: number;
  workload?: string;
  tpot?: LatencyStats;
  e2e?: LatencyStats;
  tokens_per_request_s?: number;
  prompt_tokens?: number;
  image_tokens?: number;
  peak_vram_mb?: number;
  vram_source?: string;
  agreement?: Metric;
  points?: LlmPoint[];
  precisions?: LlmPrecision[];
  sweep?: LlmSweep;
  pareto?: LlmParetoPoint[];
  recommendation?: LlmRecommendation;
  recommendations?: LlmRecommendation[];
  skipped?: LlmSkip[];
}

/**
 * Serve benchmark result (ohho.bench/v1). Objects allow additional properties.
 * Removing, renaming, or retyping a field requires ohho.bench/v2 and a
 * converter.
 * Additional properties are allowed.
 */
export interface BenchResult {
  schema: "ohho.bench/v1";
  run: BenchRun;
  latency: LatencyReport;
  throughput: Throughput;
  accuracy: Accuracy;
  telemetry: Telemetry;
  environment: Environment;
  comparison?: Comparison;
  gates?: Gates;
  batches?: BatchResult[];
  llm?: LlmBench;
  cost?: BenchCost;
}
