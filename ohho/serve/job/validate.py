"""Validate documents against the ohho.job/v1 JSON Schema.

The schema file is the only rule source. This module applies the keywords
that file uses. A successful result is a :class:`Job` whose
``schema_version`` is the schema ``const``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .errors import (
    JobSpecError,
    JobSpecInvalid,
    join_index,
    join_key,
    json_type_name,
    msg_additional,
    msg_const,
    msg_enum,
    msg_exclusive_maximum,
    msg_exclusive_minimum,
    msg_max_items,
    msg_max_length,
    msg_maximum,
    msg_min_items,
    msg_min_length,
    msg_minimum,
    msg_required,
    msg_type,
    msg_unique,
)
from .generated import SCHEMA_VERSION, Job
from .load import load_path, parse_text

# Keywords this validator reads. Anything else in schema.json is a bug.
APPLIED_KEYWORDS = frozenset(
    {
        "$ref",
        "type",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
IGNORED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$comment",
        "$defs",
        "title",
        "description",
    }
)
HANDLED_KEYWORDS = APPLIED_KEYWORDS | IGNORED_KEYWORDS


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the packaged ohho.job/v1 schema."""
    text = files("ohho.serve.job").joinpath("schema.json").read_text(encoding="utf-8")
    schema = json.loads(text)
    if not isinstance(schema, dict):
        raise ValueError("ohho.job/v1 schema must be a JSON object")
    return schema


def validate_data(data: Any) -> Job:
    """Validate a decoded document and record ``schema_version`` on the job."""
    errors = collect_errors(load_schema(), data)
    if errors:
        raise JobSpecInvalid(errors)
    job = Job.from_dict(data)
    if job.schema_version != SCHEMA_VERSION:
        raise JobSpecInvalid(
            [
                JobSpecError(
                    "$.schema_version",
                    "const",
                    f"value must be {json.dumps(SCHEMA_VERSION)}",
                )
            ]
        )
    return job


def validate_text(text: str, *, suffix: str = "") -> Job:
    """Parse YAML or JSON text and validate it."""
    return validate_data(parse_text(text, suffix=suffix))


def validate_path(path: str | Path) -> Job:
    """Read a ``.yaml`` or ``.json`` spec and validate it."""
    return validate_data(load_path(path))


def collect_errors(
    schema: Mapping[str, Any], instance: Any, path: str = "$"
) -> list[JobSpecError]:
    """Return stable errors for ``instance``. ``schema`` is the document root."""
    errors: list[JobSpecError] = []
    _walk(schema, schema, instance, path, errors)
    return errors


def _walk(
    root: Mapping[str, Any],
    schema: Mapping[str, Any],
    instance: Any,
    path: str,
    errors: list[JobSpecError],
) -> None:
    if "$ref" in schema:
        schema = _resolve(root, str(schema["$ref"]))
    expected = schema.get("type")
    if isinstance(expected, list):
        raise ValueError("ohho.job/v1 does not use union types")
    if expected is not None and not _type_ok(str(expected), instance):
        errors.append(
            JobSpecError(
                path,
                "type",
                msg_type(str(expected), json_type_name(instance)),
            )
        )
        return
    if "const" in schema and instance != schema["const"]:
        errors.append(JobSpecError(path, "const", msg_const(schema["const"])))
        return
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(JobSpecError(path, "enum", msg_enum(schema["enum"])))
        return
    if isinstance(instance, str):
        _check_string(schema, instance, path, errors)
    if _is_number(instance):
        _check_number(schema, instance, path, errors)
    if isinstance(instance, list):
        _check_array(root, schema, instance, path, errors)
    if isinstance(instance, dict):
        _check_object(root, schema, instance, path, errors)


def _resolve(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"unresolved $ref {ref}")
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"unresolved $ref {ref}")
    return node


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_string(
    schema: Mapping[str, Any],
    instance: str,
    path: str,
    errors: list[JobSpecError],
) -> None:
    if "minLength" in schema and len(instance) < int(schema["minLength"]):
        errors.append(
            JobSpecError(path, "min_length", msg_min_length(int(schema["minLength"])))
        )
    if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
        errors.append(
            JobSpecError(path, "max_length", msg_max_length(int(schema["maxLength"])))
        )


def _check_number(
    schema: Mapping[str, Any],
    instance: int | float,
    path: str,
    errors: list[JobSpecError],
) -> None:
    if "minimum" in schema and instance < schema["minimum"]:
        errors.append(JobSpecError(path, "minimum", msg_minimum(schema["minimum"])))
    if "maximum" in schema and instance > schema["maximum"]:
        errors.append(JobSpecError(path, "maximum", msg_maximum(schema["maximum"])))
    if "exclusiveMinimum" in schema and not instance > schema["exclusiveMinimum"]:
        errors.append(
            JobSpecError(
                path,
                "exclusive_minimum",
                msg_exclusive_minimum(schema["exclusiveMinimum"]),
            )
        )
    if "exclusiveMaximum" in schema and not instance < schema["exclusiveMaximum"]:
        errors.append(
            JobSpecError(
                path,
                "exclusive_maximum",
                msg_exclusive_maximum(schema["exclusiveMaximum"]),
            )
        )


def _check_array(
    root: Mapping[str, Any],
    schema: Mapping[str, Any],
    instance: list[Any],
    path: str,
    errors: list[JobSpecError],
) -> None:
    if "minItems" in schema and len(instance) < int(schema["minItems"]):
        errors.append(
            JobSpecError(path, "min_items", msg_min_items(int(schema["minItems"])))
        )
    if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
        errors.append(
            JobSpecError(path, "max_items", msg_max_items(int(schema["maxItems"])))
        )
    items = schema.get("items")
    seen: list[Any] = []
    for index, item in enumerate(instance):
        item_path = join_index(path, index)
        if schema.get("uniqueItems") is True:
            if any(item == prev for prev in seen):
                errors.append(JobSpecError(item_path, "unique_items", msg_unique()))
            else:
                seen.append(item)
        if isinstance(items, dict):
            _walk(root, items, item, item_path, errors)


def _check_object(
    root: Mapping[str, Any],
    schema: Mapping[str, Any],
    instance: dict[str, Any],
    path: str,
    errors: list[JobSpecError],
) -> None:
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("required must be an array")
    for name in required:
        if name not in instance:
            errors.append(
                JobSpecError(
                    join_key(path, str(name)), "required", msg_required(str(name))
                )
            )
    for name, sub in props.items():
        if name in instance and isinstance(sub, dict):
            _walk(root, sub, instance[name], join_key(path, str(name)), errors)
    additional = schema.get("additionalProperties", True)
    if additional is False:
        for name in instance:
            if name not in props:
                errors.append(
                    JobSpecError(
                        join_key(path, str(name)),
                        "additional_property",
                        msg_additional(str(name)),
                    )
                )
    elif isinstance(additional, dict):
        for name, value in instance.items():
            if name not in props:
                _walk(root, additional, value, join_key(path, str(name)), errors)
