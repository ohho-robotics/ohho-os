"""Benchmark results and the Serve bench harness (ohho.bench/v1).

The JSON Schema, validator, and converter live in :mod:`ohho.bench.schema`.
The harness times a five-stage runner and writes a document that validates
against that schema. ``execute_generate`` measures TTFT, TPOT, and tokens/s
for an autoregressive model. ``execute_sweep`` measures a serving sweep
(concurrency, batch tokens, prefix cache, chunked prefill, speculative
decoding) and writes a Pareto table on the same document. ``recompute_cost``
fills USD estimates from measured throughput and a user-editable price.
Import these entry points from here so another package can call them without
going through the CLI.
"""

from __future__ import annotations

from ohho.bench.cost import (
    CostOverrides,
    format_estimate,
    price_catalog,
    rank_by_cost,
    recompute_cost,
)
from ohho.bench.errors import BenchHarnessError
from ohho.bench.generate import GenerateConfig, execute_generate
from ohho.bench.sweep import SweepConfig, execute_sweep
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
    "CostOverrides",
    "GenerateConfig",
    "SweepConfig",
    "StageRunner",
    "ValidationIssue",
    "convert",
    "execute",
    "execute_generate",
    "execute_sweep",
    "finalize_result",
    "format_estimate",
    "load_schema",
    "price_catalog",
    "rank_by_cost",
    "recompute_cost",
    "render_markdown",
    "render_prometheus",
    "run_benchmark",
    "validate",
    "validate_path",
]
