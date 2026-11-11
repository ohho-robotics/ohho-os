"""Registry of training recipe families.

Built-in families register from :mod:`ohho.train.builtin_recipes`. Another
distribution can add a family with an ``ohho.recipes`` entry point whose
value is an object (or a zero-argument class) with ``id``, ``available``,
and ``run``. A second implementation of an id that is already registered is
refused.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .recipe_schema import family_pattern

ENTRY_GROUP = "ohho.recipes"
Emit = Callable[[dict[str, Any]], None]

_REGISTRY: dict[str, tuple[Any, str]] = {}
_LOAD_ERRORS: dict[str, str] = {}
_LOADED = False


class RecipeRegistryError(RuntimeError):
    """A family could not be registered."""


@dataclass(frozen=True)
class TrainResult:
    """What a family runner returns after it writes its outputs.

    ``base_weights_loaded`` is true only when this run actually loaded the
    recipe's base weights. The ACT runner in this build leaves it false.
    """

    checkpoint_dir: str
    metrics_path: str
    device: str
    base_weights_loaded: bool = False


def register(family: Any, *, source: str) -> None:
    """Register ``family``. The same object or class and id may be repeated."""
    family_id = _check_family(family)
    current = _REGISTRY.get(family_id)
    if current is not None:
        previous, previous_source = current
        if previous is family or _same_implementation(previous, family):
            return
        raise RecipeRegistryError(
            f"recipe family {family_id!r} is already registered "
            f"({previous_source}); refusing {source}"
        )
    _REGISTRY[family_id] = (family, source)


def unregister(family_id: str) -> None:
    """Drop one family. Tests use this to remove a temporary plugin."""
    _REGISTRY.pop(family_id, None)


def ensure_loaded() -> None:
    """Register built-ins, then ``ohho.recipes`` entry points."""
    global _LOADED
    if _LOADED:
        return
    from .builtin_recipes import register_builtins

    register_builtins()
    _load_entry_points()
    _LOADED = True


def get_family(family_id: str) -> Any | None:
    """Return the registered family, or ``None`` when the id is absent."""
    ensure_loaded()
    slot = _REGISTRY.get(family_id)
    if slot is None:
        return None
    return slot[0]


def family_ids() -> tuple[str, ...]:
    """Registered family ids, sorted."""
    ensure_loaded()
    return tuple(sorted(_REGISTRY))


def load_error(family_id: str) -> str:
    """Import error for an entry point that failed to load, or ``\"\"``."""
    ensure_loaded()
    return _LOAD_ERRORS.get(family_id, "")


def family_source(family_id: str) -> str:
    """Registration source string, or ``\"\"`` when the id is absent."""
    ensure_loaded()
    slot = _REGISTRY.get(family_id)
    if slot is None:
        return ""
    return slot[1]


def family_metadata(family_id: str) -> dict[str, Any]:
    """Registry record for one family, including the precisions it accepts.

    Missing families return ``{}``. A family with no ``metadata()`` method
    accepts the shared precision set, so a trainer added later picks the
    check up without a new field of its own.
    """
    ensure_loaded()
    slot = _REGISTRY.get(family_id)
    if slot is None:
        return {}
    family, source = slot
    from .precision import supported_precisions

    return {
        "id": family_id,
        "source": source,
        "available": bool(family.available()),
        "precisions": list(supported_precisions(family)),
    }


def _check_family(family: Any) -> str:
    family_id = getattr(family, "id", None)
    if not isinstance(family_id, str) or family_pattern().fullmatch(family_id) is None:
        raise RecipeRegistryError(f"recipe family id {family_id!r} is not allowed")
    if not callable(getattr(family, "available", None)):
        raise RecipeRegistryError(f"{family_id}: available() is missing")
    if not callable(getattr(family, "run", None)):
        raise RecipeRegistryError(f"{family_id}: run() is missing")
    return family_id


def _same_implementation(previous: Any, family: Any) -> bool:
    """True when both objects are the same family implementation.

    An editable install can load the class twice. Matching module, qualname,
    and id is still one implementation. A different class for that id is not.
    """
    if previous.id != family.id:
        return False
    if type(previous) is type(family):
        return True
    previous_type = type(previous)
    family_type = type(family)
    return (
        previous_type.__module__ == family_type.__module__
        and previous_type.__qualname__ == family_type.__qualname__
    )


def _load_entry_points() -> None:
    for entry in _entry_points():
        name = str(getattr(entry, "name", ""))
        try:
            loaded = entry.load()
        except Exception as exc:
            _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(loaded, type):
            try:
                loaded = loaded()
            except Exception as exc:
                _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
                continue
        value = str(getattr(entry, "value", name))
        register(loaded, source=f"entry-point:{value}")


def _entry_points() -> list[Any]:
    found: Any = importlib.metadata.entry_points(group=ENTRY_GROUP)
    if isinstance(found, Mapping):
        found = found.get(ENTRY_GROUP, ())
    return list(found)
