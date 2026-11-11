"""Write task-gate scores as an ``ohho.bench/v1`` document.

The accuracy block carries ``primary`` (required by the schema) plus the
additive ``comparisons``, ``result``, and ``reasons`` fields. The rest of
the document is the run that produced those scores. :func:`ohho.bench.schema.validate`
checks the whole document before it is returned.
"""

from __future__ import annotations

import hashlib
import math
import platform
from datetime import datetime, timezone
from typing import Any

from ohho.bench.schema import BenchValidationError, validate
from ohho.parity.errors import ParityError
from ohho.parity.tasks.types import Comparison


def build_bench_document(
    comparisons: list[Comparison],
    *,
    model: str,
    candidate: str,
    family: str | None,
    manifest_ref: str,
    latencies_ms: list[float],
    duration_s: float,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    """Assemble a bench result and reject a document the schema refuses."""
    if not comparisons:
        raise ParityError("the task gate produced no metrics.")
    reasons = [_reason(item) for item in comparisons if not item.passed()]
    result = "fail" if reasons else "pass"
    primary = comparisons[0]
    metrics: list[dict[str, Any]] = []
    for item in comparisons:
        metrics.append(
            {
                "name": f"{item.task}.{item.metric}",
                "value": item.optimized,
                "unit": item.unit,
            }
        )
        metrics.append(
            {
                "name": f"{item.task}.{item.metric}.delta",
                "value": item.delta,
                "unit": item.unit,
            }
        )
    accuracy: dict[str, Any] = {
        "primary": {
            "name": f"{primary.task}.{primary.metric}",
            "value": primary.optimized,
            "unit": primary.unit,
        },
        "metrics": metrics,
        "samples": primary.samples,
        "result": result,
        "reasons": reasons,
        "comparisons": [_comparison(item) for item in comparisons],
    }
    samples = _latency_samples(latencies_ms, duration_s)
    latency = _latency(samples)
    document = {
        "schema": "ohho.bench/v1",
        "run": {
            "model_sha": _digest(f"reference:{model}"),
            "artifact_sha": _digest(f"optimized:{candidate}"),
            "target": _workload(family, comparisons),
            "manifest_ref": manifest_ref or "ohho.parity/v1",
            "host": _host(),
            "id": "task-accuracy",
            "started_at": _stamp(started_at),
            "finished_at": _stamp(finished_at),
        },
        "latency": latency,
        "throughput": {
            "value": _throughput(primary.samples, duration_s),
            "unit": "samples/s",
            "samples": primary.samples,
            "duration_s": max(0.0, duration_s),
        },
        "accuracy": accuracy,
        "telemetry": {
            "duration_s": max(0.0, duration_s),
            "samples": primary.samples,
        },
        "environment": {"versions": _versions()},
    }
    try:
        validate(document)
    except BenchValidationError as exc:
        raise ParityError(f"task accuracy report is invalid: {exc}") from exc
    return document


def _comparison(item: Comparison) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": item.task,
        "metric": item.metric,
        "reference": item.reference,
        "optimized": item.optimized,
        "delta": item.delta,
        "tolerance": item.tolerance,
        "result": "pass" if item.passed() else "fail",
        "direction": item.direction,
        "unit": item.unit,
        "samples": item.samples,
    }
    if item.per_joint is not None:
        row["per_joint"] = [dict(joint) for joint in item.per_joint]
    if item.seeds is not None:
        row["seeds"] = list(item.seeds)
    return row


def _reason(item: Comparison) -> str:
    if item.direction == "lower_is_better":
        return (
            f"{item.task} {item.metric}: delta {_fmt(item.delta)} > "
            f"tolerance {_fmt(item.tolerance)} "
            f"(reference {_fmt(item.reference)}, optimized {_fmt(item.optimized)})"
        )
    drop = item.reference - item.optimized
    return (
        f"{item.task} {item.metric}: drop {_fmt(drop)} > "
        f"tolerance {_fmt(item.tolerance)} "
        f"(reference {_fmt(item.reference)}, optimized {_fmt(item.optimized)})"
    )


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _latency_samples(latencies_ms: list[float], duration_s: float) -> list[float]:
    samples = [value for value in latencies_ms if value >= 0.0 and math.isfinite(value)]
    if samples:
        return samples
    return [max(0.0, duration_s * 1000.0)]


def _latency(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    stats = {
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.50),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
        "std": _std(ordered),
        "unit": "ms",
        "samples": len(ordered),
    }
    stage = {"name": "task", **stats}
    return {**stats, "stages": [stage]}


def _percentile(ordered: list[float], quantile: float) -> float:
    index = int(round(quantile * (len(ordered) - 1)))
    return ordered[index]


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _throughput(samples: int, duration_s: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    return samples / duration_s


def _workload(family: str | None, comparisons: list[Comparison]) -> str:
    tasks = {item.task for item in comparisons}
    if tasks <= {"detection_map"}:
        return "vision"
    if tasks <= {"token_match"}:
        return "llm"
    if family in {"llm", "vlm"} and "action_error" not in tasks:
        return "llm"
    if family in {"vision", "detection"} and "action_error" not in tasks:
        return "vision"
    return "vla"


def _host() -> dict[str, str]:
    name = platform.node().strip() or "local"
    return {
        "name": name,
        "platform": platform.platform() or platform.system() or "unknown",
        "arch": platform.machine() or "unknown",
    }


def _versions() -> dict[str, str]:
    from ohho import __version__

    return {"python": platform.python_version(), "ohho": __version__}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stamp(moment: datetime) -> str:
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
