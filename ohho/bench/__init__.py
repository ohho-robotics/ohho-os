"""Benchmark results and the Serve bench harness (ohho.bench/v1).

The JSON Schema, validator, and converter live in :mod:`ohho.bench.schema`.
The harness times a five-stage runner and writes a document that validates
against that schema. ``execute_generate`` measures TTFT, TPOT, and tokens/s
for an autoregressive model. Import either entry point from here so another
package can call it without going through the CLI.
"""

from __future__ import annotations

from ohho.bench.errors import BenchHarnessError
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.harness import BenchConfig, execute, finalize_result, run_benchmark
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
    "SCHEMA_ID",
    "STAGE_NAMES",
    "BenchConfig",
    "BenchConversionError",
    "BenchHarnessError",
    "BenchValidationError",
    "GenerateConfig",
    "StageRunner",
    "ValidationIssue",
    "convert",
    "execute",
    "execute_generate",
    "finalize_result",
    "load_schema",
    "render_markdown",
    "render_prometheus",
    "run_benchmark",
    "validate",
    "validate_path",
]
