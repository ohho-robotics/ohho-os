"""Benchmark results and the Serve bench harness (ohho.bench/v1).

The JSON Schema, validator, and converter live in :mod:`ohho.bench.schema`.
The harness times a five-stage runner and writes a document that validates
against that schema. During the measured window it samples optional power,
memory, and thermal sources. Optional result exporters live in
:mod:`ohho.bench.exports`. Import the harness from here so another package
can call :func:`execute` without going through the CLI.

``execute_generate`` measures TTFT, TPOT, and tokens/s for an autoregressive
model.

:func:`execute_pipeline` is the camera-to-action bench. It writes the same
``ohho.bench/v1`` document plus a ``pipeline`` block.
"""

from __future__ import annotations

from ohho.bench.errors import BenchHarnessError
from ohho.bench.exports import export_result
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.harness import BenchConfig, execute, finalize_result, run_benchmark
from ohho.bench.pipeline import (
    PIPELINE_SCHEMA_ID,
    PIPELINE_STAGE_NAMES,
    PipelineConfig,
    execute_pipeline,
)
from ohho.bench.report import render_markdown, render_prometheus
from ohho.bench.runner import STAGE_NAMES, StageRunner
from ohho.bench.schema import (
    SCHEMA_ID,
    BenchConversionError,
    BenchValidationError,
    ValidationIssue,
    convert,
    load_schema,
    validate,
    validate_path,
)

__all__ = [
    "PIPELINE_SCHEMA_ID",
    "PIPELINE_STAGE_NAMES",
    "SCHEMA_ID",
    "STAGE_NAMES",
    "BenchConfig",
    "PipelineConfig",
    "BenchConversionError",
    "BenchHarnessError",
    "BenchValidationError",
    "GenerateConfig",
    "StageRunner",
    "ValidationIssue",
    "convert",
    "execute",
    "execute_generate",
    "execute_pipeline",
    "export_result",
    "finalize_result",
    "load_schema",
    "render_markdown",
    "render_prometheus",
    "run_benchmark",
    "validate",
    "validate_path",
]
