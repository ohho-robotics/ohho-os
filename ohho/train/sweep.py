"""Training-sweep schema ``ohho.sweep/v1`` and the trial expander.

A sweep document names an opaque base recipe plus a search space. Expanding
it yields one trial recipe per concrete combination: a deep copy of that
recipe with dotted ``path=value`` overrides applied. That is the same form
``ohho train --set`` uses. The recipe mapping itself is a separate contract
and is left unchecked here so this module does not depend on it.

``strategy`` is ``grid``, ``random``, ``tpe``, ``bayesian``, or ``multi``.
``tpe`` and ``bayesian`` are the built-in Tree-structured Parzen Estimator
(a Bayesian sampler; Optuna is not imported). ``multi`` is multi-objective
TPE. Those three are sequential: :func:`expand` returns the random startup
batch, and ``ohho.train.sweep_sampler.SweepSampler`` proposes later trials
from completed objectives. Sampler state and the Pareto leaderboard are
documented in ``docs/train/sweeps.md``.

``estimated_gpu_hours`` is an estimate computed from per-trial values in the
spec (the first matching ``estimate.by_value`` rule, otherwise
``estimate.gpu_hours_per_trial``). Planning does not train and does not
measure a run.

The sweep ``seed`` derives the sampler seed and each trial's training seed
(:mod:`ohho.train.sweep_seed`). Grid order ignores the seed. The training
seed is stored on the trial and does not advance the proposal generator.

The JSON Schema shipped at ``ohho/train/schemas/sweep.v1.json`` is package
data. :func:`validate_sweep` is the validator the CLI uses.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .sweep_seed import derived_trial_seed, sampler_seed, trial_training_seed

SCHEMA_ID = "ohho.sweep/v1"
SCHEMA_FILENAME = "sweep.v1.json"
STRATEGIES = ("grid", "random", "tpe", "bayesian", "multi")
ADAPTIVE_STRATEGIES = frozenset({"tpe", "bayesian", "multi"})
ADAPTIVE_PLAN_NOTE = (
    "strategy {strategy} is sequential; this plan is the startup batch. "
    "Later trials come from SweepSampler after completed objectives are recorded."
)
ESTIMATE_NOTE = (
    "estimated_gpu_hours is an estimate computed from per-trial values in the spec"
)
# Cartesian size cap, counted on the grid before constraints remove rows.
MAX_TRIALS = 100_000
_SAMPLE_MIN_ATTEMPTS = 1000
_SAMPLE_ATTEMPT_FACTOR = 200
_ANNOTATIONS = frozenset(
    {
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
    }
)
_NUMERIC_KEYS = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
)


class SweepError(ValueError):
    """A sweep spec is missing, unreadable, or invalid, or expansion failed."""


@dataclass(frozen=True)
class Dimension:
    """One axis of the search space, in declaration order."""

    name: str
    path: str
    type: str
    values: tuple[Any, ...] | None
    low: int | float | None
    high: int | float | None
    step: int | float | None
    n: int | None
    when: Mapping[str, Any] | None


@dataclass(frozen=True)
class SweepSpec:
    """A validated ``ohho.sweep/v1`` document with defaults filled in."""

    schema: str
    seed: int
    strategy: str
    parallelism: int
    recipe: dict[str, Any]
    dimensions: tuple[Dimension, ...]
    constraints: tuple[dict[str, Any], ...]
    objectives: tuple[dict[str, Any], ...]
    early_stopping: dict[str, Any] | None
    budget: dict[str, Any]
    estimate: dict[str, Any]
    sampler: dict[str, Any]


@dataclass(frozen=True)
class Trial:
    """One concrete trial: dotted overrides plus the resolved recipe.

    ``seed`` is the training seed derived from the sweep seed and ``index``.
    """

    index: int
    overrides: dict[str, Any]
    sets: tuple[str, ...]
    recipe: dict[str, Any]
    estimated_gpu_hours: float
    seed: int = 0


@dataclass(frozen=True)
class SweepPlan:
    """Expanded trials and the estimated GPU-hour total."""

    schema: str
    strategy: str
    seed: int
    sampler_seed: int
    parallelism: int
    trials: tuple[Trial, ...]
    estimated_gpu_hours: float
    objectives: tuple[dict[str, Any], ...]
    early_stopping: dict[str, Any] | None
    budget: dict[str, Any]
    estimate_note: str
    note: str = ""


def load_schema() -> dict[str, Any]:
    """Return a copy of the ``ohho.sweep/v1`` JSON Schema."""

    return copy.deepcopy(_cached_schema())


def parse_sweep_text(text: str) -> dict[str, Any]:
    """Parse a JSON document or the YAML subset accepted by ``ohho sweep``."""

    if text.startswith("\ufeff"):
        text = text[1:]
    data = _parse_document(text)
    if not isinstance(data, dict):
        raise SweepError("sweep document must be a mapping")
    return data


def validate_sweep(doc: Mapping[str, Any]) -> SweepSpec:
    """Validate ``doc`` with the packaged schema and the semantic rules.

    This is the single validator the CLI calls.
    """

    if not isinstance(doc, Mapping):
        raise SweepError("sweep document must be a mapping")
    data = copy.deepcopy(dict(doc))
    schema = _cached_schema()
    _validate(data, schema, schema, "$", 0)
    _check_semantics(data)
    return _build_spec(data)


def load_sweep(source: str | Path | Mapping[str, Any]) -> SweepSpec:
    """Load a mapping or a ``.json`` / ``.yaml`` file and validate it."""

    if isinstance(source, Mapping):
        return validate_sweep(source)
    path = Path(source)
    if not path.is_file():
        raise SweepError(f"sweep spec not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise SweepError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise SweepError(f"{path}: sweep document must be a mapping")
    else:
        data = parse_sweep_text(text)
    return validate_sweep(data)


def expand(source: str | Path | Mapping[str, Any]) -> SweepPlan:
    """Expand a sweep spec into concrete trial recipes."""

    return _expand_spec(load_sweep(source))


def format_plan(plan: SweepPlan) -> str:
    """Render a plan as text. The GPU-hour line is labeled as an estimate."""

    lines = [
        f"schema: {plan.schema}",
        f"strategy: {plan.strategy}",
        f"seed: {plan.seed}",
        f"sampler_seed: {plan.sampler_seed}",
        f"parallelism: {plan.parallelism}",
        f"trials: {len(plan.trials)}",
        f"estimated_gpu_hours: {_format_number(plan.estimated_gpu_hours)}",
        f"estimate: {plan.estimate_note}",
    ]
    if plan.note:
        lines.append(f"note: {plan.note}")
    if plan.objectives:
        lines.append("objectives:")
        for obj in plan.objectives:
            primary = " primary" if obj.get("primary") else ""
            lines.append(f"  {obj['name']}: {obj['direction']}{primary}")
    if plan.early_stopping is not None:
        early = plan.early_stopping
        lines.append(
            "early_stopping: "
            f"metric={early['metric']} patience={early['patience']} "
            f"min_delta={_format_number(float(early['min_delta']))} "
            f"mode={early['mode']}"
        )
    if plan.budget:
        lines.append("budget:")
        for key in ("max_trials", "max_gpu_hours"):
            if key in plan.budget:
                value = plan.budget[key]
                rendered = (
                    _format_number(float(value))
                    if isinstance(value, float)
                    else str(value)
                )
                lines.append(f"  {key}: {rendered}")
    for trial in plan.trials:
        lines.append("---")
        lines.append(f"trial: {trial.index}")
        lines.append(f"seed: {trial.seed}")
        lines.append(
            f"estimated_gpu_hours: {_format_number(trial.estimated_gpu_hours)}"
        )
        lines.append("sets:")
        if trial.sets:
            lines.extend(f"  {item}" for item in trial.sets)
        else:
            lines.append("  (none)")
        lines.append("recipe:")
        rendered = json.dumps(trial.recipe, indent=2, sort_keys=True, ensure_ascii=True)
        lines.extend(f"  {row}" for row in rendered.splitlines())
    return "\n".join(lines) + "\n"


def plan_as_dict(plan: SweepPlan) -> dict[str, Any]:
    """JSON-ready plan. ``trials`` is the count; recipes live in ``trial_recipes``."""

    payload = {
        "schema": plan.schema,
        "strategy": plan.strategy,
        "seed": plan.seed,
        "sampler_seed": plan.sampler_seed,
        "parallelism": plan.parallelism,
        "trials": len(plan.trials),
        "estimated_gpu_hours": plan.estimated_gpu_hours,
        "estimate": plan.estimate_note,
        "objectives": [dict(obj) for obj in plan.objectives],
        "early_stopping": (
            None if plan.early_stopping is None else dict(plan.early_stopping)
        ),
        "budget": dict(plan.budget),
        "trial_recipes": [
            {
                "index": trial.index,
                "seed": trial.seed,
                "overrides": dict(trial.overrides),
                "sets": list(trial.sets),
                "recipe": trial.recipe,
                "estimated_gpu_hours": trial.estimated_gpu_hours,
            }
            for trial in plan.trials
        ],
    }
    if plan.note:
        payload["note"] = plan.note
    return payload


def run_plan(path: str | Path, *, as_json: bool = False) -> int:
    """Print the plan for ``path``. Returns a process status code."""

    plan = expand(path)
    if as_json:
        json.dump(plan_as_dict(plan), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_plan(plan))
    return 0


def _cached_schema() -> dict[str, Any]:
    schema = getattr(_cached_schema, "_value", None)
    if schema is None:
        schema = json.loads(_read_resource("schemas", SCHEMA_FILENAME))
        _assert_refs(schema, schema, "$")
        _cached_schema._value = schema  # type: ignore[attr-defined]
    return schema


def _read_resource(*parts: str) -> str:
    from importlib.resources import files

    resource = files("ohho.train").joinpath(*parts)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, TypeError, UnicodeError):
        path = Path(__file__).resolve().parent.joinpath(*parts)
        if not path.is_file():
            joined = "/".join(parts)
            raise SweepError(f"missing package data: {joined}") from None
        return path.read_text(encoding="utf-8")


def _assert_refs(node: Any, root: Mapping[str, Any], path: str) -> None:
    if isinstance(node, dict):
        if "$ref" in node:
            _resolve_ref(root, node["$ref"], path)
        for key, value in node.items():
            _assert_refs(value, root, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_refs(value, root, f"{path}[{index}]")


def _resolve_ref(root: Mapping[str, Any], ref: str, path: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise SweepError(f"{path}: unsupported $ref {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            raise SweepError(f"{path}: unresolvable $ref {ref}")
        node = node[part]
    if not isinstance(node, Mapping):
        raise SweepError(f"{path}: $ref {ref} is not an object schema")
    return node


def _fail(path: str, message: str) -> None:
    raise SweepError(f"{path}: {message}")


def _validate(
    instance: Any,
    schema: Any,
    root: Mapping[str, Any],
    path: str,
    depth: int,
) -> None:
    if depth > 64:
        _fail(path, "schema nesting is too deep")
    if isinstance(schema, bool):
        if not schema:
            _fail(path, "rejected by schema")
        return
    if not isinstance(schema, Mapping):
        _fail(path, "schema must be an object or a boolean")
    schema = _expand_ref(schema, root, path, ())
    if "const" in schema and not _json_equal(instance, schema["const"]):
        _fail(path, f"must be {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(instance, item) for item in schema["enum"]
    ):
        _fail(path, f"must be one of {list(schema['enum'])!r}")
    if "type" in schema:
        _check_type(instance, schema["type"], path)
    if any(key in schema for key in _NUMERIC_KEYS):
        _check_numeric(instance, schema, path)
    if isinstance(instance, str):
        _check_string(instance, schema, path)
    if isinstance(instance, list) and any(
        key in schema for key in ("items", "minItems", "maxItems", "uniqueItems")
    ):
        _check_array(instance, schema, root, path, depth)
    if isinstance(instance, dict) and any(
        key in schema for key in ("properties", "required", "additionalProperties")
    ):
        _check_object(instance, schema, root, path, depth)
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate(instance, sub, root, path, depth + 1)
    if "anyOf" in schema:
        errors: list[str] = []
        for sub in schema["anyOf"]:
            err = _try_validate(instance, sub, root, path, depth + 1)
            if err is None:
                break
            errors.append(str(err))
        else:
            detail = errors[0] if errors else "no alternatives"
            _fail(path, f"did not match any alternative; {detail}")
    if "oneOf" in schema:
        matched = 0
        errors = []
        for sub in schema["oneOf"]:
            err = _try_validate(instance, sub, root, path, depth + 1)
            if err is None:
                matched += 1
            else:
                errors.append(str(err))
        if matched != 1:
            detail = errors[0] if errors else "more than one alternative matched"
            _fail(
                path,
                f"must match exactly one alternative ({matched} matched); {detail}",
            )
    if (
        "not" in schema
        and _try_validate(instance, schema["not"], root, path, depth + 1) is None
    ):
        _fail(path, "must not match the negated schema")
    if "if" in schema:
        if _try_validate(instance, schema["if"], root, path, depth + 1) is None:
            if "then" in schema:
                _validate(instance, schema["then"], root, path, depth + 1)
        elif "else" in schema:
            _validate(instance, schema["else"], root, path, depth + 1)


def _try_validate(
    instance: Any,
    schema: Any,
    root: Mapping[str, Any],
    path: str,
    depth: int,
) -> SweepError | None:
    try:
        _validate(instance, schema, root, path, depth)
    except SweepError as exc:
        return exc
    return None


def _expand_ref(
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    seen: tuple[str, ...],
) -> Mapping[str, Any]:
    if "$ref" not in schema:
        return schema
    ref = str(schema["$ref"])
    if ref in seen:
        _fail(path, f"cyclic $ref {ref}")
    target = _expand_ref(_resolve_ref(root, ref, path), root, path, seen + (ref,))
    extras = {
        key: value
        for key, value in schema.items()
        if key not in _ANNOTATIONS and key != "$ref"
    }
    if not extras:
        return target
    return {"allOf": [target, extras]}


def _check_type(instance: Any, expected: Any, path: str) -> None:
    types = expected if isinstance(expected, list) else [expected]
    if not any(_type_ok(instance, name) for name in types):
        _fail(path, f"expected type {expected}, got {_describe(instance)}")
    if (
        isinstance(instance, float)
        and not isinstance(instance, bool)
        and not math.isfinite(instance)
    ):
        _fail(path, "must be a finite number")


def _type_ok(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise SweepError(f"unsupported JSON Schema type {expected!r}")


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _check_numeric(instance: Any, schema: Mapping[str, Any], path: str) -> None:
    if isinstance(instance, bool) or not isinstance(instance, (int, float)):
        _fail(path, "must be a number")
    if isinstance(instance, float) and not math.isfinite(instance):
        _fail(path, "must be a finite number")
    if "minimum" in schema and instance < schema["minimum"]:
        _fail(path, f"must be >= {schema['minimum']}")
    if "maximum" in schema and instance > schema["maximum"]:
        _fail(path, f"must be <= {schema['maximum']}")
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        _fail(path, f"must be > {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        _fail(path, f"must be < {schema['exclusiveMaximum']}")


def _check_string(instance: str, schema: Mapping[str, Any], path: str) -> None:
    if "minLength" in schema and len(instance) < schema["minLength"]:
        _fail(path, f"must be at least {schema['minLength']} characters")
    if "maxLength" in schema and len(instance) > schema["maxLength"]:
        _fail(path, f"must be at most {schema['maxLength']} characters")
    if "pattern" in schema:
        try:
            matched = re.fullmatch(schema["pattern"], instance) is not None
        except re.error as exc:
            raise SweepError(f"{path}: invalid pattern") from exc
        if not matched:
            _fail(path, f"must match {schema['pattern']}")


def _check_array(
    instance: list[Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    depth: int,
) -> None:
    if "minItems" in schema and len(instance) < schema["minItems"]:
        _fail(path, f"must have at least {schema['minItems']} items")
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        _fail(path, f"must have at most {schema['maxItems']} items")
    if schema.get("uniqueItems"):
        seen: set[str] = set()
        for item in instance:
            key = _canon(item)
            if key in seen:
                _fail(path, "items must be unique")
            seen.add(key)
    if "items" in schema:
        for index, item in enumerate(instance):
            _validate(item, schema["items"], root, f"{path}[{index}]", depth + 1)


def _check_object(
    instance: dict[str, Any],
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    depth: int,
) -> None:
    properties = schema.get("properties", {})
    for key, value in instance.items():
        child = f"{path}.{key}"
        if key in properties:
            _validate(value, properties[key], root, child, depth + 1)
        elif "additionalProperties" in schema:
            extra = schema["additionalProperties"]
            if extra is False:
                _fail(child, "unexpected property")
            elif extra is not True:
                _validate(value, extra, root, child, depth + 1)
    for key in schema.get("required", ()):
        if key not in instance:
            _fail(f"{path}.{key}", "required property is missing")


def _check_semantics(doc: Mapping[str, Any]) -> None:
    dims = list(doc.get("dimensions") or [])
    strategy = doc.get("strategy", "grid")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, dim in enumerate(dims):
        where = f"$.dimensions[{index}]"
        name = dim["name"]
        path = dim.get("path", name)
        if name in seen_names:
            raise SweepError(f"{where}: duplicate dimension name {name!r}")
        if path in seen_paths:
            raise SweepError(f"{where}: duplicate dimension path {path!r}")
        seen_names.add(name)
        seen_paths.add(path)
        _check_dimension(dim, strategy, where)
    primaries = [
        obj["name"] for obj in doc.get("objectives") or [] if obj.get("primary")
    ]
    if len(primaries) > 1:
        raise SweepError("at most one objective may be primary")
    objectives = list(doc.get("objectives") or [])
    needs_budget = strategy == "random" or strategy in ADAPTIVE_STRATEGIES
    if needs_budget and dims and "max_trials" not in (doc.get("budget") or {}):
        raise SweepError(
            f"strategy {strategy} with dimensions requires budget.max_trials"
        )
    if strategy in ("tpe", "bayesian"):
        if not objectives:
            raise SweepError(f"strategy {strategy} requires an objective")
        if len(objectives) > 1 and not primaries:
            raise SweepError(
                f"strategy {strategy} with more than one objective requires a primary"
            )
    if strategy == "multi" and len(objectives) < 2:
        raise SweepError("strategy multi requires at least two objectives")


def _check_dimension(dim: Mapping[str, Any], strategy: str, where: str) -> None:
    kind = dim["type"]
    ranged = any(key in dim for key in ("low", "high", "step", "n"))
    if "values" in dim and ranged:
        raise SweepError(
            f"{where}: values cannot be combined with low, high, step, or n"
        )
    if "values" in dim:
        _check_values(dim, where)
    if kind == "categorical" and "values" not in dim:
        raise SweepError(f"{where}: categorical dimensions require values")
    if kind == "categorical" or "values" in dim:
        return
    if "low" not in dim or "high" not in dim:
        raise SweepError(f"{where}: {kind} requires values or low and high")
    low = dim["low"]
    high = dim["high"]
    _finite_number(low, f"{where}.low")
    _finite_number(high, f"{where}.high")
    if kind == "int":
        if not _int_like(low) or not _int_like(high):
            raise SweepError(f"{where}: int low and high must be integers")
        if "step" in dim and not _int_like(dim["step"]):
            raise SweepError(f"{where}: int step must be an integer")
    if float(high) < float(low):
        raise SweepError(f"{where}: high is less than low")
    if kind == "log_uniform" and (float(low) <= 0 or float(high) <= 0):
        raise SweepError(f"{where}: log_uniform bounds must be positive")
    if "step" in dim and "n" in dim:
        raise SweepError(f"{where}: set step or n, not both")
    if kind in ("float", "log_uniform") and "step" in dim:
        raise SweepError(f"{where}: {kind} uses n or values, not step")
    if (
        kind in ("float", "log_uniform")
        and "n" in dim
        and int(dim["n"]) > 1
        and float(high) == float(low)
    ):
        raise SweepError(f"{where}: high must be greater than low when n > 1")
    if strategy == "grid" and kind in ("float", "log_uniform") and "n" not in dim:
        raise SweepError(f"{where}: grid {kind} requires n or values")
    if kind == "int" and "n" in dim:
        _int_linspace(_as_int(low), _as_int(high), int(dim["n"]), where)


def _check_values(dim: Mapping[str, Any], where: str) -> None:
    kind = dim["type"]
    for index, value in enumerate(dim["values"]):
        spot = f"{where}.values[{index}]"
        if kind == "categorical":
            if isinstance(value, float):
                _finite_number(value, spot)
            continue
        if kind == "int":
            if not _int_like(value):
                raise SweepError(f"{spot}: int values must be integers")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SweepError(f"{spot}: {kind} values must be numbers")
        _finite_number(value, spot)
        if kind == "log_uniform" and float(value) <= 0:
            raise SweepError(f"{spot}: log_uniform values must be positive")


def _finite_number(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError(f"{where}: must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise SweepError(f"{where}: must be a finite number")


def _int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and abs(value) < 2**53
    )


def _as_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value)


def _build_spec(doc: Mapping[str, Any]) -> SweepSpec:
    early_raw = doc.get("early_stopping")
    early = None
    if early_raw is not None:
        early = {
            "metric": early_raw["metric"],
            "patience": int(early_raw.get("patience", 1)),
            "min_delta": float(early_raw.get("min_delta", 0.0)),
            "mode": early_raw.get("mode", "min"),
        }
    objectives = tuple(
        {
            "name": obj["name"],
            "direction": obj["direction"],
            "primary": bool(obj.get("primary", False)),
        }
        for obj in doc.get("objectives") or []
    )
    budget_raw = doc.get("budget") or {}
    budget: dict[str, Any] = {}
    if "max_trials" in budget_raw:
        budget["max_trials"] = int(budget_raw["max_trials"])
    if "max_gpu_hours" in budget_raw:
        budget["max_gpu_hours"] = float(budget_raw["max_gpu_hours"])
    estimate_raw = doc.get("estimate") or {}
    estimate = {
        "gpu_hours_per_trial": float(estimate_raw.get("gpu_hours_per_trial", 0.0)),
        "by_value": [dict(rule) for rule in estimate_raw.get("by_value") or []],
    }
    recipe = doc.get("recipe") or {}
    sampler_raw = doc.get("sampler") or {}
    sampler = {
        "startup_trials": int(sampler_raw.get("startup_trials", 10)),
        "gamma": float(sampler_raw.get("gamma", 0.25)),
        "candidates": int(sampler_raw.get("candidates", 64)),
        "prior_weight": float(sampler_raw.get("prior_weight", 1.0)),
    }
    return SweepSpec(
        schema=SCHEMA_ID,
        seed=int(doc.get("seed", 0)),
        strategy=str(doc.get("strategy", "grid")),
        parallelism=int(doc.get("parallelism", 1)),
        recipe=copy.deepcopy(dict(recipe)),
        dimensions=tuple(_build_dimension(raw) for raw in doc.get("dimensions") or []),
        constraints=tuple(
            copy.deepcopy(dict(item)) for item in doc.get("constraints") or []
        ),
        objectives=objectives,
        early_stopping=early,
        budget=budget,
        estimate=estimate,
        sampler=sampler,
    )


def _build_dimension(raw: Mapping[str, Any]) -> Dimension:
    kind = raw["type"]
    low = raw.get("low")
    high = raw.get("high")
    step = raw.get("step")
    if kind == "int":
        low = None if low is None else _as_int(low)
        high = None if high is None else _as_int(high)
        step = None if step is None else _as_int(step)
    values = tuple(raw["values"]) if "values" in raw else None
    when = None if "when" not in raw else copy.deepcopy(dict(raw["when"]))
    return Dimension(
        name=raw["name"],
        path=str(raw.get("path", raw["name"])),
        type=kind,
        values=values,
        low=low,
        high=high,
        step=step,
        n=None if "n" not in raw else int(raw["n"]),
        when=when,
    )


def _expand_spec(spec: SweepSpec) -> SweepPlan:
    if spec.strategy in ADAPTIVE_STRATEGIES:
        from .sweep_sampler import startup_rows

        rows = startup_rows(spec)
        plan = _plan_from_rows(spec, rows)
        return replace(plan, note=ADAPTIVE_PLAN_NOTE.format(strategy=spec.strategy))
    if spec.strategy == "random":
        rows = _expand_random(spec)
    elif spec.strategy == "grid":
        rows = _expand_grid(spec)
    else:
        raise SweepError(f"unknown strategy {spec.strategy!r}")
    return _plan_from_rows(spec, rows)


def _expand_grid(spec: SweepSpec) -> list[list[tuple[str, Any]]]:
    partials: list[list[tuple[str, Any]]] = [[]]
    for dim in spec.dimensions:
        levels = _grid_levels(dim)
        nxt: list[list[tuple[str, Any]]] = []
        for assigned in partials:
            view = _apply(spec.recipe, assigned)
            if dim.when is not None and not _matches(dim.when, view):
                nxt.append(assigned)
                continue
            for level in levels:
                nxt.append([*assigned, (dim.path, level)])
        if len(nxt) > MAX_TRIALS:
            raise SweepError(f"grid expanded to {len(nxt)} trials, above {MAX_TRIALS}")
        partials = nxt
    kept: list[list[tuple[str, Any]]] = []
    for assigned in partials:
        view = _apply(spec.recipe, assigned)
        if _allowed(view, spec.constraints):
            kept.append(assigned)
    return kept


def _expand_random(spec: SweepSpec) -> list[list[tuple[str, Any]]]:
    if not spec.dimensions:
        return [[]]
    count = int(spec.budget["max_trials"])
    if count == 0:
        return []
    rng = random.Random(sampler_seed(spec.seed))
    kept: list[list[tuple[str, Any]]] = []
    limit = max(_SAMPLE_MIN_ATTEMPTS, count * _SAMPLE_ATTEMPT_FACTOR)
    attempts = 0
    rejected = 0
    while len(kept) < count:
        if attempts >= limit:
            raise SweepError(
                f"could not sample {count} trials that satisfy constraints "
                f"after {limit} attempts ({rejected} rejected)"
            )
        attempts += 1
        assigned = _sample_one(spec, rng)
        view = _apply(spec.recipe, assigned)
        if not _allowed(view, spec.constraints):
            rejected += 1
            continue
        kept.append(assigned)
    return kept


def _sample_one(spec: SweepSpec, rng: random.Random) -> list[tuple[str, Any]]:
    assigned: list[tuple[str, Any]] = []
    for dim in spec.dimensions:
        view = _apply(spec.recipe, assigned)
        if dim.when is not None and not _matches(dim.when, view):
            continue
        assigned.append((dim.path, _sample_level(dim, rng)))
    return assigned


def _plan_from_rows(spec: SweepSpec, rows: list[list[tuple[str, Any]]]) -> SweepPlan:
    max_trials = spec.budget.get("max_trials")
    if max_trials is not None:
        rows = rows[: int(max_trials)]
    trials: list[Trial] = []
    total = 0.0
    limit = spec.budget.get("max_gpu_hours")
    for assigned in rows:
        recipe = _apply(spec.recipe, assigned)
        hours = _trial_hours(recipe, spec.estimate)
        if limit is not None and total + hours > float(limit) + 1e-9:
            break
        overrides = {path: copy.deepcopy(value) for path, value in assigned}
        sets = tuple(_format_set(path, value) for path, value in assigned)
        index = len(trials)
        trials.append(
            Trial(
                index=index,
                overrides=overrides,
                sets=sets,
                recipe=recipe,
                estimated_gpu_hours=hours,
                seed=trial_training_seed(spec.seed, index),
            )
        )
        total += hours
    return SweepPlan(
        schema=spec.schema,
        strategy=spec.strategy,
        seed=spec.seed,
        sampler_seed=sampler_seed(spec.seed),
        parallelism=spec.parallelism,
        trials=tuple(trials),
        estimated_gpu_hours=total,
        objectives=spec.objectives,
        early_stopping=spec.early_stopping,
        budget=dict(spec.budget),
        estimate_note=ESTIMATE_NOTE,
    )


def _grid_levels(dim: Dimension) -> list[Any]:
    if dim.values is not None:
        return list(dim.values)
    if dim.type == "int":
        return _int_levels(dim)
    if dim.type == "float":
        if dim.n is None or dim.low is None or dim.high is None:
            raise SweepError(f"dimension {dim.name}: grid float requires n")
        return _float_linspace(float(dim.low), float(dim.high), dim.n)
    if dim.type == "log_uniform":
        if dim.n is None or dim.low is None or dim.high is None:
            raise SweepError(f"dimension {dim.name}: grid log_uniform requires n")
        return _log_linspace(float(dim.low), float(dim.high), dim.n)
    raise SweepError(f"dimension {dim.name}: unknown type {dim.type}")


def _int_levels(dim: Dimension) -> list[int]:
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: int range requires low and high")
    low = int(dim.low)
    high = int(dim.high)
    if dim.n is not None:
        return _int_linspace(low, high, dim.n, f"dimension {dim.name}")
    step = 1 if dim.step is None else int(dim.step)
    if step <= 0:
        raise SweepError(f"dimension {dim.name}: step must be positive")
    count = ((high - low) // step) + 1
    if count > MAX_TRIALS:
        raise SweepError(
            f"dimension {dim.name} expands to {count} levels; "
            f"the maximum is {MAX_TRIALS}"
        )
    return list(range(low, high + 1, step))


def _sample_level(dim: Dimension, rng: random.Random) -> Any:
    if dim.values is not None:
        return rng.choice(dim.values)
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: nothing to sample")
    if dim.type == "int":
        if dim.n is not None:
            return rng.choice(
                _int_linspace(
                    int(dim.low), int(dim.high), dim.n, f"dimension {dim.name}"
                )
            )
        step = 1 if dim.step is None else int(dim.step)
        count = ((int(dim.high) - int(dim.low)) // step) + 1
        return int(dim.low) + step * rng.randrange(count)
    if dim.type == "float":
        if dim.n is not None:
            return rng.choice(_float_linspace(float(dim.low), float(dim.high), dim.n))
        return rng.uniform(float(dim.low), float(dim.high))
    if dim.type == "log_uniform":
        if dim.n is not None:
            return rng.choice(_log_linspace(float(dim.low), float(dim.high), dim.n))
        lo = math.log(float(dim.low))
        hi = math.log(float(dim.high))
        return math.exp(rng.uniform(lo, hi))
    raise SweepError(f"dimension {dim.name}: cannot sample type {dim.type}")


def _int_linspace(low: int, high: int, n: int, where: str) -> list[int]:
    if n == 1:
        return [low + (high - low) // 2]
    if n > (high - low + 1):
        raise SweepError(f"{where}: int n={n} does not fit in [{low}, {high}]")
    raw = [int(round(low + (high - low) * i / (n - 1))) for i in range(n)]
    raw[0] = low
    raw[-1] = high
    if any(raw[i] >= raw[i + 1] for i in range(n - 1)):
        raise SweepError(
            f"{where}: int n={n} does not yield {n} increasing points "
            f"in [{low}, {high}]; pass values or step"
        )
    return raw


def _float_linspace(low: float, high: float, n: int) -> list[float]:
    if n == 1:
        return [(low + high) / 2.0]
    return [low + (high - low) * i / (n - 1) for i in range(n)]


def _log_linspace(low: float, high: float, n: int) -> list[float]:
    if low <= 0 or high <= 0:
        raise SweepError("log_uniform bounds must be positive")
    if n == 1:
        return [math.sqrt(low * high)]
    log_low = math.log(low)
    span = math.log(high) - log_low
    return [math.exp(log_low + span * i / (n - 1)) for i in range(n)]


def _apply(
    base: Mapping[str, Any], assigned: Sequence[tuple[str, Any]]
) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for path, value in assigned:
        _assign(out, path, value)
    return out


def _assign(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(part == "" for part in parts):
        raise SweepError(f"invalid override path {path!r}")
    current = root
    for part in parts[:-1]:
        nxt = current.get(part)
        if nxt is None:
            nxt = {}
            current[part] = nxt
        elif not isinstance(nxt, dict):
            raise SweepError(
                f"cannot set {path}: {part!r} is not a mapping in the recipe"
            )
        current = nxt
    current[parts[-1]] = copy.deepcopy(value)


def _lookup(root: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _matches(condition: Mapping[str, Any], view: Mapping[str, Any]) -> bool:
    if "all" in condition:
        return all(_matches(item, view) for item in condition["all"])
    if "any" in condition:
        return any(_matches(item, view) for item in condition["any"])
    if "not" in condition:
        return not _matches(condition["not"], view)
    return _match_path(condition, view)


def _match_path(condition: Mapping[str, Any], view: Mapping[str, Any]) -> bool:
    found, value = _lookup(view, str(condition["path"]))
    ok = True
    saw = False
    if "exists" in condition:
        saw = True
        ok = ok and (found == bool(condition["exists"]))
    if "eq" in condition:
        saw = True
        ok = ok and found and _json_equal(value, condition["eq"])
    if "ne" in condition:
        saw = True
        ok = ok and found and not _json_equal(value, condition["ne"])
    if "in" in condition:
        saw = True
        ok = ok and found and any(_json_equal(value, item) for item in condition["in"])
    if "not_in" in condition:
        saw = True
        ok = (
            ok
            and found
            and all(not _json_equal(value, item) for item in condition["not_in"])
        )
    for op, predicate in (
        ("lt", lambda left, right: left < right),
        ("lte", lambda left, right: left <= right),
        ("gt", lambda left, right: left > right),
        ("gte", lambda left, right: left >= right),
    ):
        if op in condition:
            saw = True
            ok = ok and found and _number_cmp(value, condition[op], predicate)
    if not saw:
        raise SweepError(f"condition on {condition.get('path')!r} needs an operator")
    return ok


def _number_cmp(value: Any, bound: Any, predicate: Any) -> bool:
    if isinstance(value, bool) or isinstance(bound, bool):
        return False
    if not isinstance(value, (int, float)) or not isinstance(bound, (int, float)):
        return False
    return bool(predicate(value, bound))


def _allowed(view: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]) -> bool:
    for constraint in constraints:
        if "prune_when" in constraint and _matches(constraint["prune_when"], view):
            return False
        if "keep_when" in constraint and not _matches(constraint["keep_when"], view):
            return False
    return True


def _trial_hours(view: Mapping[str, Any], estimate: Mapping[str, Any]) -> float:
    for rule in estimate.get("by_value") or []:
        if _matches(rule["when"], view):
            return float(rule["gpu_hours"])
    return float(estimate.get("gpu_hours_per_trial", 0.0))


def _format_set(path: str, value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return f"{path}={rendered}"


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _json_equal(left: Any, right: Any) -> bool:
    return _canon(left) == _canon(right)


def _canon(value: Any) -> str:
    return json.dumps(
        _json_number(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _json_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and abs(value) < 2**53
    ):
        return int(value)
    if isinstance(value, list):
        return [_json_number(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SweepError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _parse_document(text: str) -> Any:
    stripped = text.lstrip()
    if not stripped:
        raise SweepError("empty sweep document")
    if stripped[0] in "{[":
        try:
            return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise SweepError(f"invalid JSON: {exc}") from exc
    return _YamlParser(_yaml_lines(text)).parse()


class _YamlParser:
    """Indentation parser for the sweep YAML subset (no PyYAML)."""

    def __init__(self, lines: list[tuple[int, int, str]]) -> None:
        self.lines = lines
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            raise SweepError("empty sweep document")
        if self.lines[0][2] == "---":
            self.index = 1
        if self.index >= len(self.lines):
            raise SweepError("empty sweep document")
        value = self._parse_node(0)
        if self.index < len(self.lines) and self.lines[self.index][2] == "...":
            self.index += 1
        if self.index < len(self.lines):
            lineno = self.lines[self.index][0]
            raise SweepError(f"line {lineno}: trailing content")
        return value

    def _parse_node(self, min_indent: int) -> Any:
        if self.index >= len(self.lines):
            raise SweepError("unexpected end of sweep document")
        lineno, indent, content = self.lines[self.index]
        if indent < min_indent:
            raise SweepError(f"line {lineno}: expected a nested value")
        if content.startswith("- ") or content == "-":
            return self._parse_list(indent)
        return self._parse_map(indent)

    def _parse_map(self, indent: int) -> dict[str, Any]:
        start = self.index
        result: dict[str, Any] = {}
        while self.index < len(self.lines):
            lineno, ind, content = self.lines[self.index]
            if ind != indent:
                break
            if content.startswith("- ") or content == "-":
                raise SweepError(f"line {lineno}: unexpected list item in a mapping")
            if not _looks_like_key(content):
                raise SweepError(f"line {lineno}: expected a key")
            key, rest = _split_key(content, lineno)
            if key in result:
                raise SweepError(f"line {lineno}: duplicate key {key!r}")
            self.index += 1
            result[key] = self._parse_value(rest, indent, lineno)
        if self.index == start:
            lineno = self.lines[self.index][0] if self.index < len(self.lines) else 0
            raise SweepError(f"line {lineno}: expected a mapping")
        return result

    def _parse_list(self, indent: int) -> list[Any]:
        start = self.index
        items: list[Any] = []
        while self.index < len(self.lines):
            lineno, ind, content = self.lines[self.index]
            if ind != indent or not (content.startswith("- ") or content == "-"):
                break
            body = content[1:].strip()
            if body == "":
                self.index += 1
                items.append(self._parse_node(indent + 1))
                continue
            if _looks_like_key(body):
                items.append(self._parse_dash_map(body, indent, lineno))
                continue
            self.index += 1
            items.append(_parse_flow_or_scalar(body, lineno))
        if self.index == start:
            lineno = self.lines[self.index][0] if self.index < len(self.lines) else 0
            raise SweepError(f"line {lineno}: expected a list")
        return items

    def _parse_dash_map(
        self, body: str, dash_indent: int, lineno: int
    ) -> dict[str, Any]:
        key, rest = _split_key(body, lineno)
        self.index += 1
        item: dict[str, Any] = {key: self._parse_value(rest, dash_indent, lineno)}
        cont_indent: int | None = None
        while self.index < len(self.lines):
            line_no, ind, content = self.lines[self.index]
            if ind <= dash_indent:
                break
            if cont_indent is None:
                cont_indent = ind
            if ind != cont_indent:
                break
            if (
                content.startswith("- ")
                or content == "-"
                or not _looks_like_key(content)
            ):
                break
            child, child_rest = _split_key(content, line_no)
            if child in item:
                raise SweepError(f"line {line_no}: duplicate key {child!r}")
            self.index += 1
            item[child] = self._parse_value(child_rest, cont_indent, line_no)
        return item

    def _parse_value(self, rest: str, key_indent: int, lineno: int) -> Any:
        if rest == "":
            if self.index >= len(self.lines) or self.lines[self.index][1] <= key_indent:
                return None
            return self._parse_node(key_indent + 1)
        if rest[0] in "| >":
            raise SweepError(f"line {lineno}: block scalars are not supported")
        return _parse_flow_or_scalar(rest, lineno)


def _yaml_lines(text: str) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw:
            raise SweepError(f"line {lineno}: tabs are not allowed in sweep YAML")
        code = _strip_comment(raw, lineno).rstrip()
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip(" "))
        lines.append((lineno, indent, code.strip()))
    return lines


def _strip_comment(line: str, lineno: int) -> str:
    out: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and quote == '"':
                if index + 1 >= len(line):
                    raise SweepError(f"line {lineno}: dangling escape")
                out.append(line[index + 1])
                index += 2
                continue
            if (
                quote == "'"
                and char == "'"
                and index + 1 < len(line)
                and line[index + 1] == "'"
            ):
                out.append("'")
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "#":
            break
        out.append(char)
        index += 1
    if quote is not None:
        raise SweepError(f"line {lineno}: unterminated string")
    return "".join(out)


def _looks_like_key(text: str) -> bool:
    if not text or text[0] in "'\"[{":
        return False
    colon = text.find(":")
    if colon <= 0:
        return False
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text[:colon]) is not None


def _split_key(text: str, lineno: int) -> tuple[str, str]:
    if text.startswith(("'", '"')):
        key, end = _parse_quoted(text, 0, lineno)
        rest = text[end:].lstrip()
        if not rest.startswith(":"):
            raise SweepError(f"line {lineno}: expected ':' after key")
        return str(key), rest[1:].strip()
    colon = text.find(":")
    if colon <= 0:
        raise SweepError(f"line {lineno}: expected a key")
    key = text[:colon].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SweepError(f"line {lineno}: invalid key {key!r}")
    return key, text[colon + 1 :].strip()


def _parse_flow_or_scalar(text: str, lineno: int) -> Any:
    if text[0] in "[{":
        value, end = _parse_flow_value(text, 0, lineno)
        if text[end:].strip():
            raise SweepError(f"line {lineno}: trailing {text[end:].strip()!r}")
        return value
    return _parse_scalar(text, lineno)


def _parse_flow_value(text: str, index: int, lineno: int) -> tuple[Any, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise SweepError(f"line {lineno}: unexpected end of flow value")
    char = text[index]
    if char == "[":
        return _parse_flow_seq(text, index, lineno)
    if char == "{":
        return _parse_flow_map(text, index, lineno)
    if char in "'\"":
        return _parse_quoted(text, index, lineno)
    end = index
    while end < len(text) and text[end] not in ",]}":
        end += 1
    raw = text[index:end].strip()
    if raw == "":
        raise SweepError(f"line {lineno}: missing flow value")
    return _parse_scalar(raw, lineno), end


def _parse_flow_seq(text: str, index: int, lineno: int) -> tuple[list[Any], int]:
    index += 1
    items: list[Any] = []
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "]":
            return items, index + 1
        value, index = _parse_flow_value(text, index, lineno)
        items.append(value)
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return items, index + 1
        raise SweepError(f"line {lineno}: expected ',' or ']' in a flow list")


def _parse_flow_map(text: str, index: int, lineno: int) -> tuple[dict[str, Any], int]:
    index += 1
    result: dict[str, Any] = {}
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            return result, index + 1
        if index < len(text) and text[index] in "'\"":
            key, index = _parse_quoted(text, index, lineno)
            key = str(key)
        else:
            end = index
            while end < len(text) and text[end] not in ":}":
                end += 1
            key = text[index:end].strip()
            index = end
            if not key:
                raise SweepError(f"line {lineno}: missing flow key")
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise SweepError(f"line {lineno}: expected ':' in a flow mapping")
        index += 1
        value, index = _parse_flow_value(text, index, lineno)
        if key in result:
            raise SweepError(f"line {lineno}: duplicate key {key!r}")
        result[key] = value
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return result, index + 1
        raise SweepError(f"line {lineno}: expected ',' or '}}' in a flow mapping")


def _parse_quoted(text: str, index: int, lineno: int) -> tuple[str, int]:
    quote = text[index]
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if quote == '"' and char == "\\":
            if index + 1 >= len(text):
                raise SweepError(f"line {lineno}: dangling escape")
            nxt = text[index + 1]
            mapping = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "/": "/"}
            if nxt not in mapping:
                raise SweepError(f"line {lineno}: unknown escape \\{nxt}")
            chars.append(mapping[nxt])
            index += 2
            continue
        if (
            quote == "'"
            and char == "'"
            and index + 1 < len(text)
            and text[index + 1] == "'"
        ):
            chars.append("'")
            index += 2
            continue
        if char == quote:
            return "".join(chars), index + 1
        chars.append(char)
        index += 1
    raise SweepError(f"line {lineno}: unterminated string")


_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(
    r"[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+[eE][+-]?\d+)|(?:\d+\.\d*[eE][+-]?\d+)|(?:\.\d+[eE][+-]?\d+))"
)


def _parse_scalar(text: str, lineno: int) -> Any:
    if text in ("null", "Null", "NULL", "~"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if text.startswith(("'", '"')):
        value, end = _parse_quoted(text, 0, lineno)
        if text[end:].strip():
            raise SweepError(f"line {lineno}: trailing {text[end:].strip()!r}")
        return value
    if text.startswith(("!", "&", "*", "|", ">", "%", "@", "`")):
        raise SweepError(f"line {lineno}: unsupported scalar {text!r}")
    if _INT_RE.fullmatch(text):
        return int(text)
    if _FLOAT_RE.fullmatch(text):
        return float(text)
    if ":" in text:
        raise SweepError(f"line {lineno}: unquoted ':' in scalar {text!r}")
    return text


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t":
        index += 1
    return index


__all__ = [
    "SCHEMA_ID",
    "STRATEGIES",
    "ADAPTIVE_STRATEGIES",
    "ADAPTIVE_PLAN_NOTE",
    "ESTIMATE_NOTE",
    "MAX_TRIALS",
    "SweepError",
    "Dimension",
    "SweepSpec",
    "Trial",
    "SweepPlan",
    "load_schema",
    "parse_sweep_text",
    "validate_sweep",
    "load_sweep",
    "expand",
    "format_plan",
    "plan_as_dict",
    "run_plan",
    "sampler_seed",
    "trial_training_seed",
    "derived_trial_seed",
]
