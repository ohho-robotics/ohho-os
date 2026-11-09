"""Entry points for task evaluators.

The group is ``ohho.parity.tasks``. The four built-ins are loaded from this
tree even when the checkout has not been reinstalled. An installed entry
point whose target is not one of those built-ins is an extra evaluator. A
duplicate name in an extra entry point replaces the built-in.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError

ENTRY_GROUP = "ohho.parity.tasks"
BUILTIN_ORDER = (
    "action_error",
    "sim_success",
    "detection_map",
    "token_match",
)
_BUILTIN_VALUES = {
    "action_error": "ohho.parity.tasks.action_error:evaluator",
    "sim_success": "ohho.parity.tasks.sim_success:evaluator",
    "detection_map": "ohho.parity.tasks.detection_map:evaluator",
    "token_match": "ohho.parity.tasks.token_match:evaluator",
}
_ENTRY_LINE = re.compile(
    r"""(?m)^(?P<name>[A-Za-z0-9_.-]+)\s*=\s*["'](?P<value>[^"']+)["']\s*$"""
)


class TaskEvaluator:
    """One pluggable scorer. ``run`` returns comparison rows."""

    name: str

    def run(self, context: Any) -> list[Any]:
        raise NotImplementedError


def load_evaluators() -> dict[str, TaskEvaluator]:
    """Built-ins plus any extra entry points, keyed by name."""
    found: dict[str, TaskEvaluator] = {}
    for name in BUILTIN_ORDER:
        found[name] = _load_target(_BUILTIN_VALUES[name], name)
    builtin_targets = set(_BUILTIN_VALUES.values())
    for entry in _read_entry_points():
        value = str(getattr(entry, "value", ""))
        name = str(getattr(entry, "name", ""))
        if not name or value in builtin_targets:
            continue
        try:
            found[name] = _load_target(value, name)
        except ParityError as exc:
            raise ParityError(f"task evaluator {name!r} failed to load: {exc}") from exc
    return found


def select_evaluators(
    spec: str | None, available: Mapping[str, TaskEvaluator] | None = None
) -> list[TaskEvaluator]:
    """Resolve ``all`` or a comma-separated list. ``None`` selects nothing."""
    table = dict(available) if available is not None else load_evaluators()
    if spec is None:
        return []
    text = spec.strip()
    if not text:
        raise ParityError("--tasks is empty.")
    if text.lower() == "all":
        ordered = [table[name] for name in BUILTIN_ORDER if name in table]
        extras = [table[name] for name in sorted(table) if name not in BUILTIN_ORDER]
        return ordered + extras
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ParityError("--tasks is empty.")
    missing = [name for name in names if name not in table]
    if missing:
        known = ", ".join(sorted(table)) or "(none)"
        raise ParityError(
            f"unknown task evaluator {', '.join(missing)}. Known: {known}."
        )
    return [table[name] for name in names]


def _load_target(value: str, name: str) -> TaskEvaluator:
    module_name, sep, attr = value.partition(":")
    if not sep or not module_name or not attr:
        raise ParityError(f"{name} entry point {value!r} must be 'module:attr'.")
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
    except Exception as exc:
        raise ParityError(str(exc)) from exc
    if isinstance(obj, type):
        obj = obj()
    if not hasattr(obj, "run"):
        raise ParityError(f"{name} evaluator has no run method.")
    obj.name = getattr(obj, "name", name)
    return obj


def _read_entry_points() -> list[Any]:
    found = importlib.metadata.entry_points(group=ENTRY_GROUP)
    if isinstance(found, Mapping):
        found = found.get(ENTRY_GROUP, ())
    loaded = list(found)
    if loaded:
        return loaded
    return _fallback_from_pyproject()


def _fallback_from_pyproject() -> list[Any]:
    path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    marker = '[project.entry-points."ohho.parity.tasks"]'
    start = text.find(marker)
    if start < 0:
        return []
    rows: list[Any] = []
    for line in text[start + len(marker) :].splitlines():
        if line.strip().startswith("["):
            break
        match = _ENTRY_LINE.match(line.strip())
        if match:
            rows.append(
                importlib.metadata.EntryPoint(
                    name=match.group("name"),
                    value=match.group("value"),
                    group=ENTRY_GROUP,
                )
            )
    return rows
