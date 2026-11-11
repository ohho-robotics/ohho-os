"""Per-model tolerances for the task gate.

A metric passes when the regression is less than or equal to its tolerance.
Error metrics (``lower_is_better``) use ``optimized - reference``. Rate
metrics (``higher_is_better``) use ``reference - optimized``.

Overlay order, each layer replacing only the metrics it names:

1. Built-in defaults.
2. File ``default``.
3. File ``families.<family>``.
4. File ``models.<model>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError

METRIC_NAMES = (
    "action_mae",
    "action_l2",
    "sim_success",
    "map",
    "top1",
    "token_match",
)
DEFAULTS: dict[str, float] = {
    "action_mae": 0.05,
    "action_l2": 0.05,
    "sim_success": 0.0,
    "map": 0.01,
    "top1": 0.01,
    "token_match": 0.01,
}
_FILE_KEYS = frozenset({"default", "families", "models"})
_METRIC_SET = frozenset(METRIC_NAMES)


def resolve_tolerances(
    model: str | None,
    family: str | None,
    path: str | None,
) -> dict[str, float]:
    """Return one finite tolerance for every task metric."""
    values = dict(DEFAULTS)
    if path:
        _overlay_file(values, Path(path), family, model)
    return values


def tolerance_for(values: dict[str, float], metric: str) -> float:
    """Look up one metric. A missing name is a request error."""
    try:
        return values[metric]
    except KeyError:
        raise ParityError(f"no tolerance for {metric}.") from None


def _overlay_file(
    values: dict[str, float],
    path: Path,
    family: str | None,
    model: str | None,
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParityError(f"could not read tolerances {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParityError(f"tolerances {path} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ParityError(f"tolerances {path} must be a JSON object.")
    extra = [key for key in document if key not in _FILE_KEYS]
    if extra:
        raise ParityError(
            f"tolerances {path} has unknown keys: {', '.join(sorted(extra))}."
        )
    _apply(values, document.get("default"), f"{path} default")
    families = document.get("families")
    if families is not None:
        _expect_object(families, f"{path} families")
        if family and family in families:
            _apply(values, families[family], f"{path} families.{family}")
    models = document.get("models")
    if models is not None:
        _expect_object(models, f"{path} models")
        if model and model in models:
            _apply(values, models[model], f"{path} models.{model}")


def _apply(values: dict[str, float], overlay: object, label: str) -> None:
    if overlay is None:
        return
    _expect_object(overlay, label)
    assert isinstance(overlay, dict)
    for key, raw in overlay.items():
        if key not in _METRIC_SET:
            known = ", ".join(METRIC_NAMES)
            raise ParityError(f"{label} has unknown metric {key!r}. Expected: {known}.")
        values[key] = _limit(raw, f"{label}.{key}")


def _expect_object(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be an object.")


def _limit(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} must be a number >= 0.")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")} or number < 0:
        raise ParityError(f"{label} must be a number >= 0.")
    return number
