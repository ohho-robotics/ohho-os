"""ohho.bench/v1 result schema, validator, and version converter.

The JSON Schema at ``ohho/bench/schemas/bench.v1.json`` is the contract.
This module checks documents against the subset of draft 2020-12 that the
schema uses. ``jsonschema`` is not a dependency of the base install, so the
``ohho bench validate`` command runs without extras.

v1 objects allow additional properties. A breaking change is published as a
new schema id together with a converter registered in :func:`convert`.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

SCHEMA_ID = "ohho.bench/v1"

__all__ = [
    "SCHEMA_ID",
    "BenchConversionError",
    "BenchValidationError",
    "ValidationIssue",
    "convert",
    "load_schema",
    "validate",
    "validate_path",
]

_REF_PREFIX = "#/$defs/"
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)
_ANNOTATIONS = frozenset(
    {
        "$comment",
        "$defs",
        "$id",
        "$schema",
        "default",
        "description",
        "examples",
        "title",
    }
)
_KNOWN = _ANNOTATIONS | frozenset(
    {
        "$ref",
        "additionalProperties",
        "allOf",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
_ASSERTIONS = frozenset(
    {
        "$ref",
        "additionalProperties",
        "allOf",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
_MISSING = object()
_cached_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationIssue:
    """One schema failure and the JSON path that locates it."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class BenchValidationError(ValueError):
    """A document does not match the bench schema.

    ``issues`` lists every problem found. Each line of ``str(error)`` starts
    with a JSON path such as ``$.run.model_sha``.
    """

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        items = tuple(issues)
        if not items:
            raise ValueError("expected at least one validation issue")
        self.issues = items
        super().__init__("\n".join(str(item) for item in items))


class BenchConversionError(ValueError):
    """No converter is registered for this pair of schema ids."""


def load_schema() -> dict[str, Any]:
    """Return a copy of the packaged ohho.bench/v1 JSON Schema."""
    global _cached_schema
    if _cached_schema is None:
        raw = files("ohho.bench").joinpath("schemas/bench.v1.json").read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError("bench schema must be a JSON object")
        const = parsed.get("properties", {}).get("schema", {}).get("const")
        if const != SCHEMA_ID:
            raise RuntimeError(
                f"shipped schema id {const!r} does not match {SCHEMA_ID!r}"
            )
        _cached_schema = parsed
    return copy.deepcopy(_cached_schema)


def validate(instance: Any, schema: dict[str, Any] | None = None) -> None:
    """Raise :class:`BenchValidationError` when ``instance`` misses ``schema``.

    ``schema`` defaults to the packaged ohho.bench/v1 document. A missing
    required field is reported at that field's JSON path.
    """
    if schema is None:
        schema = load_schema()
    if not isinstance(schema, dict):
        raise BenchValidationError([ValidationIssue("#", "schema must be an object")])
    raw_defs = schema.get("$defs", {})
    defs = raw_defs if isinstance(raw_defs, dict) else {}
    audit: list[ValidationIssue] = []
    _audit(schema, "#", defs, audit, set())
    if audit:
        raise BenchValidationError(_dedupe(audit))
    issues: list[ValidationIssue] = []
    _validate(instance, schema, "$", defs, issues, ())
    if issues:
        raise BenchValidationError(_dedupe(issues))


def validate_path(path: str | Path) -> None:
    """Load a JSON file and validate it as ohho.bench/v1."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchValidationError(
            [ValidationIssue("$", f"file not found: {path}")]
        ) from exc
    except UnicodeDecodeError as exc:
        raise BenchValidationError(
            [ValidationIssue("$", f"invalid UTF-8: {exc.reason}")]
        ) from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BenchValidationError(
            [ValidationIssue("$", f"cannot read file: {detail}")]
        ) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchValidationError(
            [
                ValidationIssue(
                    "$",
                    f"invalid JSON: {exc.msg} (line {exc.lineno} column {exc.colno})",
                )
            ]
        ) from exc
    validate(document)


def convert(document: Any, to: str) -> dict[str, Any]:
    """Return ``document`` as schema id ``to``.

    v1 is additive: unknown fields are kept. A breaking change is a new schema
    id plus a converter registered here. The identity conversion deep-copies
    the document. Any other pair raises :class:`BenchConversionError`.
    """
    if not isinstance(to, str) or not to:
        raise BenchConversionError("conversion target must be a schema id")
    if not isinstance(document, dict):
        raise BenchConversionError("$: expected object")
    source = document.get("schema", _MISSING)
    if source is _MISSING:
        raise BenchConversionError("$.schema: required property is missing")
    if not isinstance(source, str):
        raise BenchConversionError(
            f"$.schema: expected string, got {_type_name(source)}"
        )
    if source == to:
        return copy.deepcopy(document)
    raise BenchConversionError(f"no converter from {source!r} to {to!r}")


def _dedupe(issues: Sequence[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str]] = set()
    unique: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.path, issue.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def _child(path: str, key: str) -> str:
    if _IDENT.fullmatch(key):
        return f"{path}.{key}"
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"{path}['{escaped}']"


def _index(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _is_json_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float) and math.isfinite(value):
        return value.is_integer()
    return False


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "non-finite number"
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _matches_type(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return _is_integer(value)
    if expected == "number":
        return _is_json_number(value)
    if expected == "null":
        return value is None
    return False


def _declared_types(schema: dict[str, Any]) -> list[str] | None:
    if "type" not in schema:
        return None
    spec = schema["type"]
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return [item for item in spec if isinstance(item, str)]
    return []


def _audit(
    schema: object,
    path: str,
    defs: dict[str, Any],
    issues: list[ValidationIssue],
    seen: set[int],
) -> None:
    if not isinstance(schema, dict):
        issues.append(ValidationIssue(path, "schema must be an object"))
        return
    marker = id(schema)
    if marker in seen:
        return
    seen.add(marker)
    for key in schema:
        if key not in _KNOWN:
            issues.append(ValidationIssue(path, f"unsupported schema keyword {key!r}"))
    if "type" in schema:
        _audit_type(schema["type"], f"{path}/type", issues)
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            issues.append(ValidationIssue(f"{path}/properties", "expected object"))
        else:
            for name, sub in props.items():
                if not isinstance(name, str):
                    issues.append(
                        ValidationIssue(
                            f"{path}/properties",
                            "property names must be strings",
                        )
                    )
                    continue
                _audit(sub, f"{path}/properties/{name}", defs, issues, seen)
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            issues.append(
                ValidationIssue(f"{path}/required", "expected an array of strings")
            )
        elif isinstance(schema.get("properties"), dict):
            props = schema["properties"]
            for item in required:
                if item not in props:
                    issues.append(
                        ValidationIssue(
                            path,
                            f"required property {item!r} is not in properties",
                        )
                    )
    if "items" in schema:
        items = schema["items"]
        if not isinstance(items, dict):
            issues.append(ValidationIssue(f"{path}/items", "expected a schema object"))
        else:
            _audit(items, f"{path}/items", defs, issues, seen)
    if "additionalProperties" in schema:
        extra = schema["additionalProperties"]
        if isinstance(extra, dict):
            _audit(extra, f"{path}/additionalProperties", defs, issues, seen)
        elif not isinstance(extra, bool):
            issues.append(
                ValidationIssue(
                    f"{path}/additionalProperties",
                    "expected a boolean or a schema",
                )
            )
    if "allOf" in schema:
        branches = schema["allOf"]
        if not isinstance(branches, list) or not branches:
            issues.append(
                ValidationIssue(f"{path}/allOf", "expected a non-empty array")
            )
        else:
            for index, branch in enumerate(branches):
                _audit(branch, f"{path}/allOf/{index}", defs, issues, seen)
    if "$ref" in schema:
        ref = schema["$ref"]
        if (
            not isinstance(ref, str)
            or not ref.startswith(_REF_PREFIX)
            or "/" in ref[len(_REF_PREFIX) :]
        ):
            issues.append(ValidationIssue(path, f"unsupported $ref {ref!r}"))
        elif ref[len(_REF_PREFIX) :] not in defs:
            issues.append(ValidationIssue(path, f"unresolved $ref {ref!r}"))
    if "enum" in schema and not isinstance(schema["enum"], list):
        issues.append(ValidationIssue(f"{path}/enum", "expected an array"))
    for bound in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if bound not in schema:
            continue
        value = schema[bound]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(ValidationIssue(f"{path}/{bound}", "expected a number"))
    for bound in ("minLength", "maxLength", "minItems", "maxItems"):
        if bound not in schema:
            continue
        value = schema[bound]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(
                ValidationIssue(f"{path}/{bound}", "expected a non-negative integer")
            )
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            issues.append(ValidationIssue(f"{path}/pattern", "expected a string"))
        else:
            try:
                re.compile(pattern)
            except re.error as exc:
                issues.append(ValidationIssue(path, f"invalid pattern: {exc}"))
    if "$defs" in schema:
        node = schema["$defs"]
        if not isinstance(node, dict):
            issues.append(ValidationIssue(f"{path}/$defs", "expected object"))
        else:
            for name, sub in node.items():
                _audit(sub, f"{path}/$defs/{name}", defs, issues, seen)


def _audit_type(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if isinstance(value, str):
        names = [value]
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        names = list(value)
    else:
        issues.append(
            ValidationIssue(path, "expected a type name or an array of type names")
        )
        return
    for name in names:
        if name not in _TYPES:
            issues.append(ValidationIssue(path, f"unsupported type {name!r}"))


def _validate(
    instance: Any,
    schema: dict[str, Any],
    path: str,
    defs: dict[str, Any],
    issues: list[ValidationIssue],
    stack: tuple[str, ...],
) -> None:
    if "allOf" in schema:
        branches = schema["allOf"]
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    _validate(instance, branch, path, defs, issues, stack)
                else:
                    issues.append(ValidationIssue(path, "schema must be an object"))
        rest = {
            key: value
            for key, value in schema.items()
            if key not in _ANNOTATIONS and key != "allOf"
        }
        if rest and any(key in _ASSERTIONS for key in rest):
            _validate(instance, rest, path, defs, issues, stack)
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        name = ""
        if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
            name = ref[len(_REF_PREFIX) :]
        if not name or "/" in name:
            issues.append(ValidationIssue(path, f"unsupported $ref {ref!r}"))
            return
        if name in stack:
            issues.append(ValidationIssue(path, f"cyclic $ref {ref!r}"))
            return
        target = defs.get(name)
        if not isinstance(target, dict):
            issues.append(ValidationIssue(path, f"unresolved $ref {ref!r}"))
            return
        _validate(instance, target, path, defs, issues, stack + (name,))
        rest = {
            key: value
            for key, value in schema.items()
            if key not in _ANNOTATIONS and key != "$ref"
        }
        if rest and any(key in _ASSERTIONS for key in rest):
            _validate(instance, rest, path, defs, issues, stack)
        return
    if "const" in schema and instance != schema["const"]:
        issues.append(ValidationIssue(path, f"expected {schema['const']!r}"))
        return
    if "enum" in schema and isinstance(schema["enum"], list):
        if instance not in schema["enum"]:
            choices = ", ".join(repr(item) for item in schema["enum"])
            issues.append(ValidationIssue(path, f"expected one of {choices}"))
            return
    declared = _declared_types(schema)
    if declared is not None and not any(
        _matches_type(instance, expected) for expected in declared
    ):
        label = " or ".join(declared) if declared else "a supported type"
        issues.append(
            ValidationIssue(path, f"expected {label}, got {_type_name(instance)}")
        )
        return
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            issues.append(
                ValidationIssue(path, f"expected length >= {schema['minLength']}")
            )
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            issues.append(
                ValidationIssue(path, f"expected length <= {schema['maxLength']}")
            )
        if "pattern" in schema and isinstance(schema["pattern"], str):
            try:
                matched = re.fullmatch(schema["pattern"], instance) is not None
            except re.error as exc:
                issues.append(ValidationIssue(path, f"invalid pattern: {exc}"))
                matched = True
            if not matched:
                issues.append(
                    ValidationIssue(path, f"does not match {schema['pattern']}")
                )
    if _is_json_number(instance):
        _check_bounds(instance, schema, path, issues)
    expects_object = (
        "properties" in schema
        or "required" in schema
        or (declared is not None and "object" in declared)
    )
    if expects_object:
        if not isinstance(instance, dict):
            if declared is None:
                issues.append(
                    ValidationIssue(
                        path, f"expected object, got {_type_name(instance)}"
                    )
                )
            return
        _validate_object(instance, schema, path, defs, issues, stack)
    expects_array = "items" in schema or (declared is not None and "array" in declared)
    if expects_array:
        if not isinstance(instance, list):
            if declared is None:
                issues.append(
                    ValidationIssue(path, f"expected array, got {_type_name(instance)}")
                )
            return
        _validate_array(instance, schema, path, defs, issues, stack)


def _check_bounds(
    instance: int | float,
    schema: dict[str, Any],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if "minimum" in schema and instance < schema["minimum"]:
        issues.append(ValidationIssue(path, f"expected >= {schema['minimum']}"))
    if "maximum" in schema and instance > schema["maximum"]:
        issues.append(ValidationIssue(path, f"expected <= {schema['maximum']}"))
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        issues.append(ValidationIssue(path, f"expected > {schema['exclusiveMinimum']}"))
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        issues.append(ValidationIssue(path, f"expected < {schema['exclusiveMaximum']}"))


def _validate_object(
    instance: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    defs: dict[str, Any],
    issues: list[ValidationIssue],
    stack: tuple[str, ...],
) -> None:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in instance:
                issues.append(
                    ValidationIssue(_child(path, key), "required property is missing")
                )
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        if not isinstance(key, str):
            continue
        child = _child(path, key)
        if key in properties and isinstance(properties[key], dict):
            _validate(value, properties[key], child, defs, issues, stack)
        elif additional is False:
            issues.append(ValidationIssue(child, "additional property is not allowed"))
        elif isinstance(additional, dict):
            _validate(value, additional, child, defs, issues, stack)


def _validate_array(
    instance: list[Any],
    schema: dict[str, Any],
    path: str,
    defs: dict[str, Any],
    issues: list[ValidationIssue],
    stack: tuple[str, ...],
) -> None:
    if "minItems" in schema and len(instance) < schema["minItems"]:
        issues.append(
            ValidationIssue(path, f"expected at least {schema['minItems']} items")
        )
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        issues.append(
            ValidationIssue(path, f"expected at most {schema['maxItems']} items")
        )
    items = schema.get("items")
    if not isinstance(items, dict):
        return
    for index, value in enumerate(instance):
        _validate(value, items, _index(path, index), defs, issues, stack)
