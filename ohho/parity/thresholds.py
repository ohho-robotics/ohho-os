"""Parity thresholds.

The default gate is mean absolute difference ``<= 0.05``. A JSON file can
set a different limit per model family and per target profile. ``--mean-abs``
replaces only that one field after the file is applied.

Overlay order, each layer filling only the fields it names:

1. Built-in default (``mean_abs`` 0.05).
2. Built-in family extras (classifier and llm top-1 and KL).
3. File ``default``.
4. File ``families.<family>``.
5. File ``targets.<target>`` metric fields.
6. File ``targets.<target>.precisions.<PRECISION>``.
7. File ``targets.<target>.families.<family>``.
8. ``--mean-abs``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError

DEFAULT_MEAN_ABS = 0.05

_METRIC_KEYS = (
    "mean_abs",
    "max_abs",
    "rmse",
    "cosine_min",
    "top1_min",
    "kl_max",
    "per_joint",
    "action_chunk",
)
_METRIC_SET = frozenset(_METRIC_KEYS)
_FILE_KEYS = frozenset({"default", "families", "targets"})
_TARGET_NESTED = frozenset({"precisions", "families"})
_BUILTIN_FAMILY: dict[str, dict[str, float]] = {
    "classifier": {"top1_min": 0.99, "kl_max": 0.05},
    "llm": {"top1_min": 0.99, "kl_max": 0.05},
    "vlm": {"top1_min": 0.99, "kl_max": 0.05},
}


@dataclass(frozen=True)
class Thresholds:
    """Effective limits for one comparison. ``None`` means that metric is reported only."""

    mean_abs: float = DEFAULT_MEAN_ABS
    max_abs: float | None = None
    rmse: float | None = None
    cosine_min: float | None = None
    top1_min: float | None = None
    kl_max: float | None = None
    per_joint: float | None = None
    action_chunk: float | None = None

    def to_json(self) -> dict[str, float | None]:
        """The eight fields, with unused limits as null."""
        return dict(asdict(self))


def resolve_thresholds(
    family: str | None,
    target: str | None,
    precision: str | None,
    *,
    mean_abs: float | None = None,
    path: str | None = None,
) -> Thresholds:
    """Resolve the limits for this family, target, and precision."""
    values: dict[str, float | None] = {"mean_abs": DEFAULT_MEAN_ABS}
    _overlay(values, _BUILTIN_FAMILY.get(family or "", {}))
    if path:
        _overlay_file(values, Path(path), family, target, precision)
    if mean_abs is not None:
        _check_metric("mean_abs", mean_abs)
        values["mean_abs"] = float(mean_abs)
    return Thresholds(**values)


def _overlay_file(
    values: dict[str, float | None],
    path: Path,
    family: str | None,
    target: str | None,
    precision: str | None,
) -> None:
    document = _read_file(path)
    _overlay(values, _metric_object(document.get("default"), f"{path.name} default"))
    families = _nested(document.get("families"), f"{path.name} families")
    if family and family in families:
        _overlay(values, _metric_object(families[family], f"families.{family}"))
    targets = _nested(document.get("targets"), f"{path.name} targets")
    if not target or target not in targets:
        return
    block = targets[target]
    if not isinstance(block, dict):
        raise ParityError(f"targets.{target} must be an object.")
    _overlay(values, _metric_fields(block, f"targets.{target}"))
    precisions = block.get("precisions")
    if precision and precisions is not None:
        table = _nested(precisions, f"targets.{target}.precisions")
        key = precision.upper()
        if key in table:
            _overlay(
                values, _metric_object(table[key], f"targets.{target}.precisions.{key}")
            )
    nested_families = block.get("families")
    if family and nested_families is not None:
        table = _nested(nested_families, f"targets.{target}.families")
        if family in table:
            _overlay(
                values,
                _metric_object(table[family], f"targets.{target}.families.{family}"),
            )


def _read_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ParityError(f"Threshold file does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParityError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ParityError(f"{path} must be an object.")
    extra = set(document) - _FILE_KEYS
    if extra:
        raise ParityError(
            f"{path.name} has unknown fields: {', '.join(sorted(extra))}."
        )
    return document


def _nested(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be an object.")
    return value


def _metric_object(value: object, label: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be an object.")
    extra = set(value) - _METRIC_SET
    if extra:
        raise ParityError(f"{label} has unknown fields: {', '.join(sorted(extra))}.")
    return _metric_fields(value, label)


def _metric_fields(value: dict[str, Any], label: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for key in _METRIC_KEYS:
        if key not in value:
            continue
        found[key] = _check_metric(key, value[key], label=label)
    return found


def _overlay(values: dict[str, float | None], layer: dict[str, float]) -> None:
    for key, item in layer.items():
        values[key] = item


def _check_metric(key: str, value: object, *, label: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} {key} must be a number.")
    number = float(value)
    if key == "cosine_min":
        if not -1.0 <= number <= 1.0:
            raise ParityError(f"{label} cosine_min must be between -1 and 1.")
        return number
    if key == "top1_min":
        if not 0.0 <= number <= 1.0:
            raise ParityError(f"{label} top1_min must be between 0 and 1.")
        return number
    if number < 0.0:
        raise ParityError(f"{label} {key} must be >= 0.")
    return number
