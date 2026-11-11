"""Load, override, and validate an ``ohho.recipe/v1`` document.

The schema file ships as package data next to this module. Validation uses
that file, so a recipe the published schema rejects does not reach a trainer.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .schema_validate import validate_instance
from .yaml_subset import YamlError, loads

SCHEMA_ID = "ohho.recipe/v1"
SCHEMA_FILENAME = "ohho.recipe.v1.json"
_SCHEMA: dict[str, Any] | None = None
_MISSING = object()


class RecipeError(Exception):
    """Base error for recipe loading and runs."""


class RecipeConfigError(RecipeError):
    """The recipe or a ``--set`` override is not usable.

    Raised before a trainer is called.
    """

    def __init__(self, message: str, errors: Sequence[str] | None = None) -> None:
        self.summary = message
        self.errors = list(errors) if errors is not None else [message]
        text = message if errors is None else message + "\n" + "\n".join(self.errors)
        super().__init__(text)


class FamilyUnavailable(RecipeError):
    """The family id is known to the schema but has no trainer in this build."""

    def __init__(self, family: str) -> None:
        self.family = family
        super().__init__("family not available in this build")


class DatasetUnavailable(RecipeError):
    """The dataset cannot be trained from a local directory in this build."""


class TrainFailed(RecipeError):
    """The selected trainer started and did not finish."""


def schema_path() -> Path:
    """Path of the shipped ``ohho.recipe/v1`` JSON Schema."""
    return Path(__file__).resolve().parent / "schema" / SCHEMA_FILENAME


def examples_dir() -> Path:
    """Directory of the shipped family examples."""
    return Path(__file__).resolve().parent / "examples"


def schema_document() -> dict[str, Any]:
    """Return the parsed schema document."""
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))
    return _SCHEMA


def iter_examples() -> list[Path]:
    """Shipped ``*.yaml`` examples, sorted by file name."""
    return sorted(examples_dir().glob("*.yaml"))


def example_path(family: str) -> Path:
    """Path of the shipped example for ``family``."""
    path = examples_dir() / f"{family}.yaml"
    if not path.is_file():
        raise RecipeConfigError(f"no shipped example for family {family!r}")
    return path


def family_pattern() -> re.Pattern[str]:
    """Compiled family-id pattern from the published schema."""
    raw = str(schema_document()["properties"]["family"]["pattern"])
    return re.compile(raw)


def load_recipe_file(
    path: str | Path, overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    """Read a recipe, apply overrides, validate, and fill defaults.

    ``.json`` files are parsed as JSON. Every other suffix is parsed as YAML.
    """
    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise RecipeConfigError(f"recipe file not found: {recipe_path}")
    text = recipe_path.read_text(encoding="utf-8")
    try:
        if recipe_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = loads(text)
    except json.JSONDecodeError as exc:
        raise RecipeConfigError(f"could not parse {recipe_path.name}: {exc}") from exc
    except YamlError as exc:
        raise RecipeConfigError(f"could not parse {recipe_path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise RecipeConfigError("recipe must be a mapping")
    data = apply_overrides(data, list(overrides or []))
    errors = validate_recipe(data)
    if errors:
        raise RecipeConfigError("invalid recipe", errors)
    return apply_defaults(data)


def apply_overrides(
    document: dict[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """Return a copy of ``document`` with dotted ``k=v`` overrides applied.

    ``null`` deletes a mapping key. Values are parsed as booleans, numbers,
    JSON arrays or objects, quoted strings, or plain strings.
    """
    updated = copy.deepcopy(document)
    for item in overrides:
        _apply_one(updated, item)
    return updated


def validate_recipe(document: Any) -> list[str]:
    """Validate ``document`` against ``ohho.recipe/v1``."""
    errors = validate_instance(document, schema_document())
    dataset = document.get("dataset") if isinstance(document, dict) else None
    if isinstance(dataset, dict) and (("path" in dataset) == ("version_id" in dataset)):
        errors = [
            error
            for error in errors
            if not error.startswith("$.dataset: expected exactly one")
        ]
        clearer = "$.dataset: set exactly one of path or version_id"
        if clearer not in errors:
            errors.append(clearer)
    return errors


def apply_defaults(recipe: dict[str, Any]) -> dict[str, Any]:
    """Fill optional fields that have schema defaults."""
    filled = copy.deepcopy(recipe)
    filled.setdefault("precision", "fp32")
    filled.setdefault("device", "cpu")
    filled.setdefault("deterministic", False)
    filled.setdefault("hyperparameters", {})
    filled.setdefault("compression_hooks", [])
    filled.setdefault("eval_hooks", [])
    block = filled.get("distillation")
    if isinstance(block, dict):
        block.setdefault("loss", "action_regression")
        block.setdefault("feature_matching", False)
        block.setdefault("feature_weight", 1.0)
        block.setdefault("mix_weight", 0.5)
        block.setdefault("temperature", 1.0)
    return filled


def parse_override_value(raw: str) -> Any:
    """Parse the value half of a ``--set key=value`` pair."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text[:1] in {"{", "["}:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RecipeConfigError(f"invalid JSON in --set value: {exc}") from exc
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+", text):
        return float(text)
    return text


def _apply_one(document: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise RecipeConfigError(f"--set expects k=v, got {item!r}")
    raw_key, raw_value = item.split("=", 1)
    key = raw_key.strip()
    if not key or key.startswith(".") or key.endswith(".") or ".." in key:
        raise RecipeConfigError(f"invalid --set key {raw_key!r}")
    parts = key.split(".")
    value = parse_override_value(raw_value)
    cursor: Any = document
    for part in parts[:-1]:
        cursor = _descend(cursor, part, key)
    _write(cursor, parts[-1], value, key)


def _descend(cursor: Any, part: str, key: str) -> Any:
    if isinstance(cursor, dict):
        current = cursor.get(part, _MISSING)
        if current is _MISSING or current is None:
            cursor[part] = {}
            return cursor[part]
        if isinstance(current, (dict, list)):
            return current
        raise RecipeConfigError(f"--set {key!r} cannot descend through {part!r}")
    if isinstance(cursor, list):
        if not part.isdigit():
            raise RecipeConfigError(f"--set {key!r} needs a numeric index at {part!r}")
        index = int(part)
        if index < 0 or index >= len(cursor):
            raise RecipeConfigError(f"--set {key!r} index {index} is out of range")
        return cursor[index]
    raise RecipeConfigError(f"--set {key!r} cannot descend through {part!r}")


def _write(cursor: Any, part: str, value: Any, key: str) -> None:
    if isinstance(cursor, dict):
        if value is None:
            cursor.pop(part, None)
        else:
            cursor[part] = value
        return
    if isinstance(cursor, list):
        if not part.isdigit():
            raise RecipeConfigError(f"--set {key!r} needs a numeric index at {part!r}")
        index = int(part)
        if value is None:
            raise RecipeConfigError(f"--set {key!r} cannot delete a list element")
        if index < 0 or index >= len(cursor):
            raise RecipeConfigError(f"--set {key!r} index {index} is out of range")
        cursor[index] = value
        return
    raise RecipeConfigError(f"--set {key!r} cannot assign {part!r}")
