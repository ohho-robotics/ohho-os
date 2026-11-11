"""Quantizer registry.

Built-in backends ship on the ``ohho.quantizers`` entry-point group. A
checkout that has not been reinstalled still loads those builtins. Any
other installed entry point is a third-party backend.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from ohho.optimize.errors import OptimizeError
from ohho.optimize.quantizers.base import Quantizer

ENTRY_GROUP = "ohho.quantizers"

_BUILTIN: tuple[tuple[str, str], ...] = (
    ("awq", "ohho.optimize.quantizers.awq:backend"),
    ("gptq", "ohho.optimize.quantizers.gptq:backend"),
    ("bitsandbytes", "ohho.optimize.quantizers.bitsandbytes:backend"),
    ("fp8", "ohho.optimize.quantizers.fp8:backend"),
    ("ort-int8", "ohho.optimize.quantizers.ort_int8:backend"),
)

_CACHE: dict[str, Quantizer] | None = None


def list_quantizers() -> list[str]:
    """Registered backend names in sorted order."""
    return sorted(quantizers())


def get_quantizer(name: str) -> Quantizer:
    """Return one backend. Unknown names name the registry."""
    try:
        return quantizers()[name]
    except KeyError:
        known = ", ".join(list_quantizers()) or "(none)"
        raise OptimizeError(
            f"Unknown quantizer {name!r}. Expected one of: {known}."
        ) from None


def quantizers() -> dict[str, Quantizer]:
    """Load the registry once."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return _CACHE


def reset_quantizers() -> None:
    """Drop the cache. Tests use this after patching entry points."""
    global _CACHE
    _CACHE = None


def _load() -> dict[str, Quantizer]:
    declared: dict[str, str] = {}
    for entry in _entry_points():
        name = str(getattr(entry, "name", ""))
        value = str(getattr(entry, "value", ""))
        if not name or not value:
            continue
        previous = declared.get(name)
        if previous is not None and previous != value:
            raise OptimizeError(
                f"Quantizer {name!r} is declared by {previous} and {value}."
            )
        declared[name] = value
    for name, value in _BUILTIN:
        previous = declared.get(name)
        if previous is None:
            declared[name] = value
        elif previous != value:
            raise OptimizeError(
                f"Quantizer {name!r} is declared by {previous} and {value}."
            )
    loaded: dict[str, Quantizer] = {}
    for name in sorted(declared):
        plugin = _load_entry(name, declared[name])
        loaded[plugin.name or name] = plugin
    return loaded


def _entry_points() -> list[Any]:
    try:
        found = importlib.metadata.entry_points(group=ENTRY_GROUP)
    except TypeError:
        found = importlib.metadata.entry_points().get(ENTRY_GROUP, ())
    if isinstance(found, dict):
        found = found.get(ENTRY_GROUP, ())
    return list(found)


def _load_entry(name: str, value: str) -> Quantizer:
    entry = importlib.metadata.EntryPoint(name=name, value=value, group=ENTRY_GROUP)
    try:
        obj = entry.load()
    except Exception as exc:
        raise OptimizeError(
            f"Quantizer {name!r} failed to load ({value}): {exc}"
        ) from exc
    if not isinstance(obj, Quantizer):
        raise OptimizeError(
            f"Quantizer {name!r} ({value}) must be a Quantizer instance."
        )
    if obj.name and obj.name != name:
        raise OptimizeError(
            f"Quantizer entry {name!r} loaded a backend named {obj.name!r}."
        )
    return obj
