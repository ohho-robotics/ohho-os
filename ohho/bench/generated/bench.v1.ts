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
 * Reference and optimised error for one action joint.
 * Additional properties are allowed.
 */
export interface JointAccuracy {
  index: number;
  name?: string;
  reference: number;
  optimized: number;
  delta: number;
}

/**
 * Reference versus optimised score for one task metric, and whether the delta
 * is inside the tolerance.
 * Additional properties are allowed.
 */
export interface AccuracyComparison {
  task: string;
  metric: string;
  reference: number;
  optimized: number;
  delta: number;
  tolerance: number;
  result: "pass" | "fail";
  direction?: "lower_is_better" | "higher_is_better";
  unit?: string;
  samples?: number;
  per_joint?: JointAccuracy[];
  seeds?: number[];
}

/**
 * Quality of the model on the bench set. A task gate may add
 * reference-versus-optimised comparisons; primary stays the headline figure.
 * Additional properties are allowed.
 */
export interface Accuracy {
  primary: Metric;
  metrics?: Metric[];
  samples?: number;
  result?: "pass" | "fail";
  reasons?: string[];
  comparisons?: AccuracyComparison[];
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
}
