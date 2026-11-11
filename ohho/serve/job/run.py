"""Run a validated ohho.job/v1 spec against engines that exist in this build.

``optimize``, ``parity`` and ``bench`` are not registered. A missing engine
is reported as ``not available in this build`` and is never given a
synthetic metric.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .generated import Job

StepStatus = Literal["ok", "not_available", "error"]


def not_available_message(step: str) -> str:
    """Stable sentence for a step whose engine is not in this build."""
    return f"{step} is not available in this build"


@dataclass(frozen=True)
class StepResult:
    """Outcome of one step. No metric fields: a missing engine cannot invent them."""

    step: str
    status: StepStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"step": self.step, "status": self.status, "message": self.message}


StepEngine = Callable[[Job], StepResult]

# Engines shipped with this build. The three schema steps are absent on purpose.
BUILTIN_ENGINES: Mapping[str, StepEngine] = MappingProxyType({})


@dataclass(frozen=True)
class JobRunResult:
    """Local run of a validated job. ``schema_version`` is copied from the job."""

    schema_version: str
    ok: bool
    steps: tuple[StepResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "steps": [step.to_dict() for step in self.steps],
        }


def run_job(
    job: Job,
    engines: Mapping[str, StepEngine] | None = None,
) -> JobRunResult:
    """Run ``job.steps`` in order.

    Only names present in ``engines`` are called. The default table is empty.
    Unregistered steps are ``not_available``. An engine exception is ``error``.
    """
    table = BUILTIN_ENGINES if engines is None else engines
    results: list[StepResult] = []
    for step in job.steps:
        engine = table.get(step)
        if engine is None:
            results.append(
                StepResult(
                    step=step,
                    status="not_available",
                    message=not_available_message(step),
                )
            )
            continue
        try:
            produced = engine(job)
        except Exception as exc:
            results.append(
                StepResult(
                    step=step,
                    status="error",
                    message=f"{step} failed: {exc}",
                )
            )
            continue
        status = produced.status
        message = produced.message
        if status not in ("ok", "error", "not_available"):
            status = "error"
            message = f"{step} failed: unrecognized step status"
        results.append(StepResult(step=step, status=status, message=message))
    ok = bool(results) and all(item.status == "ok" for item in results)
    return JobRunResult(
        schema_version=job.schema_version,
        ok=ok,
        steps=tuple(results),
    )


def exit_code(result: JobRunResult) -> int:
    """0 when every step is ok, 2 when any step is not available, else 1."""
    if result.ok:
        return 0
    if any(step.status == "not_available" for step in result.steps):
        return 2
    return 1
