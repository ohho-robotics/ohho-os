"""Numbers and grouping labels read from a bench or optimize result.

Adapters turn these into a vendor payload. This module does not name a
vendor or open a connection.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_LATENCY = ("mean", "median", "p95", "p99", "min", "max")
_KIND = re.compile(r"[^A-Za-z0-9_.-]+")


def summary_numbers(document: Mapping[str, Any]) -> dict[str, Any]:
    """Latency, accuracy, energy, and power values worth publishing.

    Missing measurements are omitted. Accuracy keeps its name beside the
    number so a later reader can tell a real score from ``not_measured``.
    """
    found: dict[str, Any] = {}
    latency = _mapping(document.get("latency"))
    for stat in _LATENCY:
        number = _finite(latency.get(stat))
        if number is not None:
            found[f"latency/{stat}"] = number
    rate = _finite(latency.get("achievable_hz"))
    if rate is None:
        rate = _finite(_mapping(document.get("throughput")).get("value"))
    if rate is not None:
        found["throughput/hz"] = rate
    name, score = _accuracy(document)
    if score is not None:
        found["accuracy"] = score
    if name:
        found["accuracy_name"] = name
    energy = energy_mj(document)
    if energy is not None:
        found["energy/mj"] = energy
    power = power_w(document)
    if power is not None:
        found["power/w"] = power
    return found


def energy_mj(document: Mapping[str, Any]) -> float | None:
    """Millijoules per inference, or ``None`` when the run did not measure it."""
    direct = _finite(document.get("energy_mj"))
    if direct is not None:
        return direct
    sensors = _mapping(_mapping(document.get("telemetry")).get("sensors"))
    return _finite(sensors.get("energy_per_inference_mj"))


def power_w(document: Mapping[str, Any]) -> float | None:
    """Mean watts, preferring the telemetry gauge over the sensor block."""
    telemetry = _mapping(document.get("telemetry"))
    direct = _finite(telemetry.get("power_w_mean"))
    if direct is not None:
        return direct
    return _finite(_mapping(telemetry.get("sensors")).get("power_w_mean"))


def grouping_labels(
    document: Mapping[str, Any],
    *,
    kind: str,
    spec: Mapping[str, Any],
) -> dict[str, str]:
    """``job``, ``model``, ``target``, and ``run_id`` for one result."""
    run = _mapping(document.get("run"))
    environment = _mapping(document.get("environment"))
    return {
        "job": job_name(kind),
        "model": _first(
            spec.get("model"),
            run.get("model_sha"),
        ),
        "target": _first(
            spec.get("target"),
            environment.get("target_profile"),
            run.get("target"),
        ),
        "run_id": _first(run.get("id"), spec.get("run_id")),
    }


def job_name(kind: str) -> str:
    """Grouping job label for a bench run, an optimize run, or another kind."""
    if kind == "bench":
        return "ohho-bench"
    if kind == "optimize":
        return "ohho-optimize"
    cleaned = _KIND.sub("-", kind).strip("-.")
    if not cleaned:
        return "ohho"
    return f"ohho-{cleaned}"[:128]


def _accuracy(document: Mapping[str, Any]) -> tuple[str, float | None]:
    accuracy = _mapping(document.get("accuracy"))
    primary = accuracy.get("primary")
    if isinstance(primary, Mapping):
        name = primary.get("name")
        label = name.strip() if isinstance(name, str) else ""
        return label, _finite(primary.get("value"))
    return "", _finite(accuracy.get("value"))


def _first(*values: object) -> str:
    for value in values:
        if isinstance(value, str):
            text = value.replace("\r", " ").replace("\n", " ").strip()
            if text:
                return text[:512]
    return "unknown"


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
