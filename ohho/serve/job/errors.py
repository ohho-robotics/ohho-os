"""Stable validation messages for ohho.job/v1.

One error is one line: ``<path>: <code>: <message>``.

Split a line on the first two ``": "`` separators. The message may itself
contain ``": "``. Paths and codes do not.

Path grammar, starting at ``$``:

- ``.name`` for a key matching ``[A-Za-z_][A-Za-z0-9_]*``
- ``["..."]`` for any other key, with the key encoded by ``json.dumps``
  (``ensure_ascii=True``)
- ``[index]`` for an array index, decimal, no leading zeros
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def join_key(path: str, key: str) -> str:
    """Return the path of an object property."""
    if _IDENT.fullmatch(key):
        return f"{path}.{key}"
    encoded = json.dumps(key, ensure_ascii=True)
    return f"{path}[{encoded}]"


def join_index(path: str, index: int) -> str:
    """Return the path of an array element."""
    return f"{path}[{index}]"


def format_error(path: str, code: str, message: str) -> str:
    """Return the one-line form console and server both display."""
    text = message.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return f"{path}: {code}: {text.strip()}"


def msg_required(name: str) -> str:
    return f"missing required property {json.dumps(name, ensure_ascii=True)}"


def msg_type(expected: str, actual: str) -> str:
    return f"expected {expected}, got {actual}"


def msg_enum(values: Sequence[Any]) -> str:
    rendered = ", ".join(json.dumps(value, ensure_ascii=True) for value in values)
    return f"value must be one of {rendered}"


def msg_const(value: Any) -> str:
    return f"value must be {json.dumps(value, ensure_ascii=True)}"


def msg_additional(name: str) -> str:
    return f"unknown property {json.dumps(name, ensure_ascii=True)}"


def msg_minimum(bound: Any) -> str:
    return f"value must be >= {json.dumps(bound, ensure_ascii=True)}"


def msg_maximum(bound: Any) -> str:
    return f"value must be <= {json.dumps(bound, ensure_ascii=True)}"


def msg_exclusive_minimum(bound: Any) -> str:
    return f"value must be > {json.dumps(bound, ensure_ascii=True)}"


def msg_exclusive_maximum(bound: Any) -> str:
    return f"value must be < {json.dumps(bound, ensure_ascii=True)}"


def msg_min_length(bound: int) -> str:
    return f"string length must be >= {bound}"


def msg_max_length(bound: int) -> str:
    return f"string length must be <= {bound}"


def msg_min_items(bound: int) -> str:
    return f"array length must be >= {bound}"


def msg_max_items(bound: int) -> str:
    return f"array length must be <= {bound}"


def msg_unique() -> str:
    return "duplicate item"


def json_type_name(value: Any) -> str:
    """JSON Schema type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


@dataclass(frozen=True)
class JobSpecError:
    """One validation failure. ``str(error)`` is the stable line."""

    path: str
    code: str
    message: str

    def __str__(self) -> str:
        return format_error(self.path, self.code, self.message)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


class JobSpecInvalid(Exception):
    """The document was not accepted. ``errors`` is in display order."""

    def __init__(self, errors: Sequence[JobSpecError]) -> None:
        if not errors:
            raise ValueError("JobSpecInvalid requires at least one error")
        self.errors = tuple(errors)
        super().__init__("\n".join(str(error) for error in self.errors))
