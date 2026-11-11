"""Build and check an ohho.eval/v1 report."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from ohho.bench.schema import BenchValidationError, validate

from .ci import Z_95, wilson_interval
from .errors import EvalValidationError

SCHEMA_ID = "ohho.eval/v1"
_SCHEMA_NAME = "schemas/ohho.eval.v1.json"


def load_schema() -> dict[str, Any]:
    """Return the packaged ohho.eval/v1 JSON Schema."""
    raw = files("ohho.eval").joinpath(_SCHEMA_NAME).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("eval schema must be a JSON object")
    const = parsed.get("properties", {}).get("schema", {}).get("const")
    if const != SCHEMA_ID:
        raise RuntimeError(f"shipped schema id {const!r} does not match {SCHEMA_ID!r}")
    return parsed


def validate_report(report: Any) -> None:
    """Raise :class:`EvalValidationError` when ``report`` misses ohho.eval/v1."""
    try:
        validate(report, load_schema())
    except BenchValidationError as exc:
        raise EvalValidationError(str(exc)) from exc


def build_report(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    suite: str,
    scenario_source: str,
    scenario_ids: list[str],
    seed: int,
    backend: str,
    episodes_per_scenario: int,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a report from episode rows and validate it before returning."""
    successes = sum(1 for episode in episodes if episode["success"])
    trials = len(episodes)
    low, high = wilson_interval(successes, trials, z=Z_95)
    lengths = [int(episode["length"]) for episode in episodes]
    modes: dict[str, int] = {}
    for episode in episodes:
        mode = episode["failure_mode"]
        if episode["success"] or not mode:
            continue
        modes[mode] = modes.get(mode, 0) + 1
    videos = [episode["video_path"] for episode in episodes if episode["video_path"]]
    if lengths:
        mean = sum(lengths) / len(lengths)
        shortest = min(lengths)
        longest = max(lengths)
    else:
        mean = 0.0
        shortest = 0
        longest = 0
    report: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "suite": suite,
        "scenario_source": scenario_source,
        "scenario_ids": list(scenario_ids),
        "seed": seed,
        "backend": backend,
        "episodes_per_scenario": episodes_per_scenario,
        "success_rate": (successes / trials) if trials else 0.0,
        "success_ci95": {
            "method": "wilson",
            "level": 0.95,
            "low": low,
            "high": high,
            "successes": successes,
            "trials": trials,
        },
        "episode_length": {
            "mean": mean,
            "min": shortest,
            "max": longest,
            "unit": "steps",
        },
        "failure_modes": modes,
        "videos": videos,
        "episodes": episodes,
    }
    validate_report(report)
    return report
