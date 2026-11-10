"""Validate a document against the subset of JSON Schema used by recipes.

Covers the draft 2020-12 keywords in ``ohho.recipe/v1``: types, const, enum,
required, properties, additionalProperties, items, numeric bounds, string
length and pattern, ``oneOf``, ``$ref``, and ``not``. Unknown keywords are
ignored so the published schema can carry descriptions.
"""

from __future__ import annotations

import re
from typing import Any


class SchemaError(ValueError):
    """The schema document itself is not usable."""


def validate_instance(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of errors. An empty list means the instance is valid."""
    return _validate(instance, schema, schema, "$")


def _validate(instance: Any, schema: Any, root: dict[str, Any], path: str) -> list[str]:
    schema = _deref(schema, root)
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: rejected by schema"]
    if not isinstance(schema, dict):
        raise SchemaError("schema node must be an object")

    if "type" in schema:
        expected = schema["type"]
        names = expected if isinstance(expected, list) else [expected]
        if not any(_is_type(instance, name) for name in names):
            joined = " or ".join(str(name) for name in names)
            return [f"{path}: expected {joined}"]

    errors: list[str] = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: expected one of {list(schema['enum'])!r}")
    if "pattern" in schema and isinstance(instance, str):
        if re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: does not match {schema['pattern']}")
    if "minLength" in schema and isinstance(instance, str):
        if len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: shorter than {schema['minLength']}")
    if "maxLength" in schema and isinstance(instance, str):
        if len(instance) > int(schema["maxLength"]):
            errors.append(f"{path}: longer than {schema['maxLength']}")
    errors.extend(_number_errors(instance, schema, path))
    if "not" in schema:
        inner = _validate(instance, schema["not"], root, path)
        if not inner:
            errors.append(f"{path}: must not satisfy the negated schema")
    if isinstance(instance, dict):
        errors.extend(_object_errors(instance, schema, root, path))
    if isinstance(instance, list) and "items" in schema:
        errors.extend(_array_errors(instance, schema, root, path))
    if "oneOf" in schema:
        errors.extend(_one_of_errors(instance, schema["oneOf"], root, path))
    return errors


def _number_errors(instance: Any, schema: dict[str, Any], path: str) -> list[str]:
    if isinstance(instance, bool) or not isinstance(instance, (int, float)):
        return []
    errors: list[str] = []
    if "minimum" in schema and instance < schema["minimum"]:
        errors.append(f"{path}: must be >= {schema['minimum']}")
    if "maximum" in schema and instance > schema["maximum"]:
        errors.append(f"{path}: must be <= {schema['maximum']}")
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        errors.append(f"{path}: must be > {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        errors.append(f"{path}: must be < {schema['exclusiveMaximum']}")
    return errors


def _object_errors(
    instance: dict[str, Any],
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaError("properties must be an object")
    for key in schema.get("required", []):
        if key not in instance:
            errors.append(f"{path}.{key}: is required")
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        child = f"{path}.{key}"
        if key in properties:
            errors.extend(_validate(value, properties[key], root, child))
            continue
        if additional is False:
            errors.append(f"{child}: is not allowed")
        elif isinstance(additional, dict):
            errors.extend(_validate(value, additional, root, child))
    return errors


def _array_errors(
    instance: list[Any],
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    if "minItems" in schema and len(instance) < int(schema["minItems"]):
        errors.append(f"{path}: must contain at least {schema['minItems']} items")
    if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
        errors.append(f"{path}: must contain at most {schema['maxItems']} items")
    for index, item in enumerate(instance):
        errors.extend(_validate(item, schema["items"], root, f"{path}[{index}]"))
    return errors


def _one_of_errors(
    instance: Any,
    branches: Any,
    root: dict[str, Any],
    path: str,
) -> list[str]:
    if not isinstance(branches, list) or not branches:
        raise SchemaError("oneOf must be a non-empty list")
    matched = 0
    for branch in branches:
        if not _validate(instance, branch, root, path):
            matched += 1
    if matched == 1:
        return []
    return [f"{path}: expected exactly one alternative, matched {matched}"]


def _deref(schema: Any, root: dict[str, Any], depth: int = 0) -> Any:
    if depth > 16:
        raise SchemaError("schema $ref is too deep")
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise SchemaError(f"unsupported $ref {ref!r}")
        node: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise SchemaError(f"unresolved $ref {ref}")
            node = node[part]
        return _deref(node, root, depth + 1)
    return schema


def _is_type(instance: Any, name: str) -> bool:
    if name == "object":
        return isinstance(instance, dict)
    if name == "array":
        return isinstance(instance, list)
    if name == "string":
        return isinstance(instance, str)
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "boolean":
        return isinstance(instance, bool)
    if name == "null":
        return instance is None
    raise SchemaError(f"unsupported type {name!r}")
