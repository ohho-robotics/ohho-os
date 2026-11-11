"""Baseline deltas for an ohho.bench/v1 result.

An increase is a regression for latency and peak memory. A decrease is a
regression for achievable Hz and throughput. ``pct_change`` is
``(current - baseline) / baseline * 100``. A zero baseline omits
``pct_change`` unless the current value is also zero, in which case the
change is 0.
"""

from __future__ import annotations

from ohho.bench.errors import BenchHarnessError

_LATENCY_FIELDS = (
    "mean",
    "median",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
    "std",
    "cold_start_ms",
)

HIGHER_IS_WORSE = frozenset(
    {
        "latency.mean",
        "latency.median",
        "latency.p50",
        "latency.p90",
        "latency.p95",
        "latency.p99",
        "latency.min",
        "latency.max",
        "latency.std",
        "latency.cold_start_ms",
        "telemetry.memory_mb_max",
    }
)
HIGHER_IS_BETTER = frozenset({"latency.achievable_hz", "throughput.value"})

_ALIASES = {
    "mean": "latency.mean",
    "median": "latency.median",
    "p50": "latency.p50",
    "p90": "latency.p90",
    "p95": "latency.p95",
    "p99": "latency.p99",
    "min": "latency.min",
    "max": "latency.max",
    "std": "latency.std",
    "cold_start": "latency.cold_start_ms",
    "cold_start_ms": "latency.cold_start_ms",
    "achievable_hz": "latency.achievable_hz",
    "hz": "latency.achievable_hz",
    "throughput": "throughput.value",
    "throughput.value": "throughput.value",
    "memory": "telemetry.memory_mb_max",
    "memory_mb_max": "telemetry.memory_mb_max",
    "telemetry.memory_mb_max": "telemetry.memory_mb_max",
}
for _name in HIGHER_IS_WORSE | HIGHER_IS_BETTER:
    _ALIASES.setdefault(_name, _name)
_ALIASES["ttft_p95"] = "llm.ttft.p95"
_ALIASES["tpot_p95"] = "llm.tpot.p95"
_ALIASES["llm.ttft.p95"] = "llm.ttft.p95"
_ALIASES["llm.tpot.p95"] = "llm.tpot.p95"


def canonical_metric(name: str) -> str:
    """Map a CLI metric such as ``p95`` onto the comparison row name."""
    key = name.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        known = ", ".join(sorted(_ALIASES))
        raise BenchHarnessError(
            f"unknown metric {name!r}. Expected one of: {known}."
        ) from None


def compare_results(
    current: dict, baseline: dict, *, baseline_ref: str
) -> dict[str, object]:
    """Per-metric delta of ``current`` against ``baseline``.

    Latency fields are compared in milliseconds. A baseline whose
    ``latency.unit`` is ``s`` is scaled. Throughput is compared only when
    the units match.
    """
    metrics: list[dict[str, object]] = []
    current_latency = _block(current, "latency")
    baseline_latency = _block(baseline, "latency")
    for field in _LATENCY_FIELDS:
        if field not in current_latency or field not in baseline_latency:
            continue
        metrics.append(
            _row(
                f"latency.{field}",
                _latency_ms(baseline_latency, field),
                _latency_ms(current_latency, field),
                unit="ms",
                higher_is_worse=True,
            )
        )
    if "achievable_hz" in current_latency and "achievable_hz" in baseline_latency:
        metrics.append(
            _row(
                "latency.achievable_hz",
                float(baseline_latency["achievable_hz"]),
                float(current_latency["achievable_hz"]),
                unit="Hz",
                higher_is_worse=False,
            )
        )
    current_rate = _block(current, "throughput")
    baseline_rate = _block(baseline, "throughput")
    if current_rate.get("unit") != baseline_rate.get("unit"):
        raise BenchHarnessError(
            "baseline throughput unit "
            f"{baseline_rate.get('unit')!r} does not match "
            f"{current_rate.get('unit')!r}"
        )
    metrics.append(
        _row(
            "throughput.value",
            float(baseline_rate["value"]),
            float(current_rate["value"]),
            unit=str(current_rate.get("unit") or "1"),
            higher_is_worse=False,
        )
    )
    current_memory = current.get("telemetry", {}).get("memory_mb_max")
    baseline_memory = baseline.get("telemetry", {}).get("memory_mb_max")
    if isinstance(current_memory, (int, float)) and isinstance(
        baseline_memory, (int, float)
    ):
        if isinstance(current_memory, bool) or isinstance(baseline_memory, bool):
            pass
        else:
            metrics.append(
                _row(
                    "telemetry.memory_mb_max",
                    float(baseline_memory),
                    float(current_memory),
                    unit="MiB",
                    higher_is_worse=True,
                )
            )
    return {"baseline_ref": baseline_ref, "metrics": metrics}


def _block(document: dict, key: str) -> dict:
    block = document.get(key)
    if not isinstance(block, dict):
        raise BenchHarnessError(f"bench result is missing {key}")
    return block


def _latency_ms(block: dict, field: str) -> float:
    """Return one latency field in milliseconds.

    ``cold_start_ms`` is already milliseconds. The other fields follow
    ``latency.unit``.
    """
    value = float(block[field])
    if field == "cold_start_ms":
        return value
    unit = str(block.get("unit") or "ms")
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    raise BenchHarnessError(f"cannot compare latency unit {unit!r}; expected ms or s")


def _row(
    name: str,
    baseline: float,
    current: float,
    *,
    unit: str,
    higher_is_worse: bool,
) -> dict[str, object]:
    delta = current - baseline
    row: dict[str, object] = {
        "name": name,
        "unit": unit,
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "regression": _regression(delta, higher_is_worse=higher_is_worse),
        "higher_is_worse": higher_is_worse,
    }
    change = _pct(baseline, current)
    if change is not None:
        row["pct_change"] = change
    return row


def _pct(baseline: float, current: float) -> float | None:
    if baseline == 0:
        if current == 0:
            return 0.0
        return None
    return (current - baseline) / baseline * 100.0


def _regression(delta: float, *, higher_is_worse: bool) -> bool:
    if higher_is_worse:
        return delta > 0
    return delta < 0
