"""ohho.job/v1 — the shared optimize, parity and bench job spec.

``schema.json`` is the only source. ``generated.py`` and
``schemas/ts/job.v1.d.ts`` are produced from it. A job that validates
always carries ``schema_version``.
"""

from __future__ import annotations

from .errors import JobSpecError, JobSpecInvalid, format_error
from .generated import SCHEMA_VERSION, Job
from .load import load_path, parse_text
from .run import BUILTIN_ENGINES, JobRunResult, StepResult, run_job
from .validate import load_schema, validate_data, validate_path, validate_text

__all__ = [
    "BUILTIN_ENGINES",
    "SCHEMA_VERSION",
    "Job",
    "JobRunResult",
    "JobSpecError",
    "JobSpecInvalid",
    "StepResult",
    "format_error",
    "load_path",
    "load_schema",
    "parse_text",
    "run_job",
    "validate_data",
    "validate_path",
    "validate_text",
]
