"""Checkpoint sim evaluation (``ohho eval``) and the ohho.eval/v1 report.

Proof scenario suites are loaded by reference from ``suites.json``.
This package does not copy the scenario files.

Training and sweep code, which ship in other changes, can call
:func:`evaluate_after_train` at the end of a trial or dispatch
:func:`run_eval_job` for a job whose type is :data:`JOB_TYPE`.
"""

from __future__ import annotations

from .ci import Z_95, wilson_interval
from .errors import EvalError, EvalValidationError
from .report import SCHEMA_ID, build_report, load_schema, validate_report
from .runner import (
    BACKEND,
    JOB_TYPE,
    checkpoint_sha256,
    evaluate_after_train,
    run_eval,
    run_eval_job,
)
from .scenarios import (
    SUITES,
    ProofScenario,
    SuiteRef,
    list_suites,
    load_scenarios,
    resolve_scenario_file,
)

__all__ = [
    "BACKEND",
    "JOB_TYPE",
    "SCHEMA_ID",
    "SUITES",
    "EvalError",
    "EvalValidationError",
    "ProofScenario",
    "SuiteRef",
    "Z_95",
    "build_report",
    "checkpoint_sha256",
    "evaluate_after_train",
    "list_suites",
    "load_scenarios",
    "load_schema",
    "resolve_scenario_file",
    "run_eval",
    "run_eval_job",
    "validate_report",
    "wilson_interval",
]
