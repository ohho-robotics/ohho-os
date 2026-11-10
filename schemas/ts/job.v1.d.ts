/**
 * Generated from ohho/serve/job/schema.json. Do not edit.
 *
 * Schema: ohho.job/v1
 * Regenerate: python scripts/generate_job_types.py
 */

export const JOB_SCHEMA_VERSION = "ohho.job/v1" as const;
export const JOB_SCHEMA_ID = "https://ohho-robotics.com/schemas/ohho.job/v1.json" as const;

export type JobSchemaVersion = "ohho.job/v1";
export type JobStep = "optimize" | "parity" | "bench";
export type JobPrecision = "fp32" | "fp16" | "bf16" | "int8";
export type JobBackend = "onnx" | "torchscript" | "cpu";

/** Model version reference. */
export interface JobModel {
  version: string;
}

/** Parity thresholds. Omitted fields stay unset. */
export interface JobParity {
  max_abs_error?: number;
  max_rel_error?: number;
  min_cosine?: number;
  samples?: number;
}

/** Bench settings. Omitted fields stay unset. */
export interface JobBench {
  iterations?: number;
  warmup?: number;
  batch_size?: number;
  timeout_s?: number;
}

/** Budget caps. Omitted fields stay unset. */
export interface JobBudget {
  max_seconds?: number;
  max_usd?: number;
  max_gpu_seconds?: number;
}

/** Weights and Biases export. */
export interface JobWandbExport {
  enabled: boolean;
  project?: string;
  entity?: string;
}

/** Prometheus export. */
export interface JobPrometheusExport {
  enabled: boolean;
  port?: number;
  job_name?: string;
}

/** Export targets. Omitted exports stay unset. */
export interface JobOutputs {
  wandb?: JobWandbExport;
  prometheus?: JobPrometheusExport;
}

/** Shared optimize, parity and bench request (ohho.job/v1). */
export interface Job {
  schema_version: JobSchemaVersion;
  model: JobModel;
  targets: string[];
  steps: JobStep[];
  precision?: JobPrecision;
  backend?: JobBackend;
  parity?: JobParity;
  bench?: JobBench;
  budget?: JobBudget;
  outputs?: JobOutputs;
}
