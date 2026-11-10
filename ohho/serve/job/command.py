"""``ohho job validate`` and ``ohho job run``."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .errors import JobSpecInvalid
from .generated import SCHEMA_VERSION
from .load import load_path
from .run import exit_code, run_job
from .validate import validate_data


def main(args: Any) -> int:
    """Run the job subcommand selected by ``ohho``."""
    as_json = bool(getattr(args, "json", False))
    command = args.job_command
    try:
        data = load_path(args.spec)
        job = validate_data(data)
    except JobSpecInvalid as exc:
        _emit_invalid(exc, as_json)
        return 1
    if command == "validate":
        _emit_job(job.to_dict(), as_json)
        return 0
    if command == "run":
        result = run_job(job)
        if as_json:
            _dump(result.to_dict())
        else:
            print(f"schema_version: {result.schema_version}")
            for step in result.steps:
                print(step.message)
        return exit_code(result)
    print(f"unknown job command {command}", file=sys.stderr)
    return 2


def _emit_job(job: dict[str, Any], as_json: bool) -> None:
    if as_json:
        _dump({"ok": True, "schema_version": job["schema_version"], "job": job})
        return
    print("ok")
    print(f"schema_version: {job['schema_version']}")


def _emit_invalid(exc: JobSpecInvalid, as_json: bool) -> None:
    if as_json:
        _dump(
            {
                "ok": False,
                "schema_version": SCHEMA_VERSION,
                "errors": [error.to_dict() for error in exc.errors],
            }
        )
        return
    _write_errors(exc, sys.stderr)


def _write_errors(exc: JobSpecInvalid, stream: TextIO) -> None:
    for error in exc.errors:
        print(str(error), file=stream)


def _dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
