"""SLO and regression gates.

``--slo p95<=200ms`` compares the primary batch latency, converted to
milliseconds. ``--max-regression p95=10%`` fails when that metric's
percent change is worse than the limit. An exact limit passes. A latency
increase is worse. A throughput or Hz decrease is worse.

Gates read the primary batch on the top-level result. Per-batch numbers
stay in ``batches`` and are not gated individually.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ohho.bench.compare import canonical_metric
from ohho.bench.errors import BenchHarnessError

_NUMBER = r"(?:\d+\.?\d*|\.\d+)"
_SLO_RE = re.compile(
    rf"^(?P<metric>[A-Za-z_][A-Za-z0-9_.]*)\s*(?P<op><=|<)\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<unit>ms|s)?$",
    re.IGNORECASE,
)
_REG_RE = re.compile(
    rf"^(?P<metric>[A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(?P<number>{_NUMBER})\s*%$"
)


@dataclass(frozen=True)
class Slo:
    """One parsed latency ceiling."""

    expr: str
    field: str
    op: str
    limit_ms: float


@dataclass(frozen=True)
class RegressionLimit:
    """One parsed percent ceiling on a comparison row."""

    expr: str
    metric: str
    limit_pct: float


def parse_slo(expr: str) -> Slo:
    """Parse ``p95<=200ms``. A bare number is milliseconds."""
    match = _SLO_RE.fullmatch(expr.strip())
    if match is None:
        raise BenchHarnessError(f"invalid --slo {expr!r}; expected p95<=200ms")
    canonical = canonical_metric(match.group("metric"))
    if not canonical.startswith("latency.") or canonical == "latency.achievable_hz":
        raise BenchHarnessError(
            f"invalid --slo {expr!r}; the metric must be a latency such as p95"
        )
    unit = (match.group("unit") or "ms").lower()
    number = float(match.group("number"))
    limit_ms = number * 1000.0 if unit == "s" else number
    return Slo(
        expr=expr.strip(),
        field=canonical.split(".", 1)[1],
        op=match.group("op"),
        limit_ms=limit_ms,
    )


def parse_regression(expr: str) -> RegressionLimit:
    """Parse ``p95=10%``."""
    match = _REG_RE.fullmatch(expr.strip())
    if match is None:
        raise BenchHarnessError(f"invalid --max-regression {expr!r}; expected p95=10%")
    return RegressionLimit(
        expr=expr.strip(),
        metric=canonical_metric(match.group("metric")),
        limit_pct=float(match.group("number")),
    )


def check_gate_expressions(
    slos: Sequence[str],
    max_regressions: Sequence[str],
    *,
    has_baseline: bool,
) -> None:
    """Fail before a run when a gate expression is unusable."""
    for expr in slos:
        parse_slo(expr)
    for expr in max_regressions:
        parse_regression(expr)
    if max_regressions and not has_baseline:
        raise BenchHarnessError("--max-regression requires --baseline")


def evaluate_gates(
    document: dict,
    slos: Sequence[str],
    max_regressions: Sequence[str],
) -> dict[str, object]:
    """Return ``{passed, checks}`` for the primary batch.

    An empty request is a passing gate list. Callers that were not asked
    for gates can omit the object.
    """
    checks: list[dict[str, object]] = []
    for expr in slos:
        checks.append(_slo_check(document, parse_slo(expr)))
    for expr in max_regressions:
        checks.append(_regression_check(document, parse_regression(expr)))
    return {"passed": all(bool(item["passed"]) for item in checks), "checks": checks}


def _slo_check(document: dict, slo: Slo) -> dict[str, object]:
    latency = document.get("latency")
    if not isinstance(latency, dict) or slo.field not in latency:
        raise BenchHarnessError(f"result has no latency.{slo.field} for {slo.expr}")
    value_ms = _as_ms(float(latency[slo.field]), latency, slo.field)
    if slo.op == "<=":
        passed = value_ms <= slo.limit_ms
        relation = "<="
    else:
        passed = value_ms < slo.limit_ms
        relation = "<"
    detail = f"p95-class {slo.field} {value_ms:.6g} ms {relation} {slo.limit_ms:.6g} ms"
    return {
        "name": "slo",
        "expr": slo.expr,
        "passed": passed,
        "metric": f"latency.{slo.field}",
        "value": value_ms,
        "limit": slo.limit_ms,
        "detail": detail,
    }


def _as_ms(value: float, latency: dict, field: str) -> float:
    if field == "cold_start_ms":
        return value
    unit = str(latency.get("unit") or "ms")
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    raise BenchHarnessError(f"SLO gates need latency.unit ms or s, got {unit!r}")


def _regression_check(document: dict, limit: RegressionLimit) -> dict[str, object]:
    comparison = document.get("comparison")
    if not isinstance(comparison, dict):
        raise BenchHarnessError("--max-regression requires --baseline")
    rows = comparison.get("metrics")
    if not isinstance(rows, list):
        raise BenchHarnessError("baseline comparison has no metrics")
    row = next((item for item in rows if item.get("name") == limit.metric), None)
    if not isinstance(row, dict):
        raise BenchHarnessError(f"baseline comparison has no metric {limit.metric!r}")
    higher_is_worse = bool(row.get("higher_is_worse", True))
    change = row.get("pct_change")
    if isinstance(change, bool) or not isinstance(change, (int, float)):
        worse = bool(row.get("regression"))
        passed = not worse
        detail = (
            f"{limit.metric} has no percent change "
            f"(baseline {row.get('baseline')}); regression={worse}"
        )
        value = float(row.get("delta") or 0.0)
    else:
        value = float(change)
        if higher_is_worse:
            passed = value <= limit.limit_pct
            detail = (
                f"{limit.metric} changed {value:.6g}% (limit +{limit.limit_pct:.6g}%)"
            )
        else:
            passed = value >= -limit.limit_pct
            detail = (
                f"{limit.metric} changed {value:.6g}% (limit -{limit.limit_pct:.6g}%)"
            )
    return {
        "name": "max_regression",
        "expr": limit.expr,
        "passed": passed,
        "metric": limit.metric,
        "value": value,
        "limit": limit.limit_pct,
        "detail": detail,
    }
