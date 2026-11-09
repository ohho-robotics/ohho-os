"""YAML subset loader for training recipes.

The subset covers the documents this package ships: indentation-based
mappings and lists, scalars, ``[]``, ``{}``, and comments. Constructs outside
that subset raise :class:`UnsupportedYaml`. When PyYAML is installed, the
loader then retries with :func:`yaml.safe_load`.
"""

from __future__ import annotations

import re
from typing import Any

_INT = re.compile(r"-?\d+")
_FLOAT = re.compile(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+")


class YamlError(ValueError):
    """The text is not valid in the recipe YAML subset."""


class UnsupportedYaml(YamlError):
    """The text needs YAML features this subset does not parse."""


def loads(text: str) -> Any:
    """Parse a YAML document. Returns the top-level value."""
    try:
        return _loads_subset(text)
    except UnsupportedYaml as exc:
        return _loads_with_pyyaml(text, exc)


def _loads_with_pyyaml(text: str, reason: UnsupportedYaml) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise UnsupportedYaml(
            f"{reason}. Full YAML needs the yaml extra: pip install 'ohho-os[yaml]'"
        ) from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        raise YamlError(f"could not parse YAML: {exc}") from exc


def _loads_subset(text: str) -> Any:
    lines = _prepare(text)
    if not lines:
        raise YamlError("recipe YAML is empty")
    if lines[0][1] != 0:
        raise YamlError(f"line {lines[0][0]}: expected top-level content at column 0")
    value, index = _parse_node(lines, 0)
    if index != len(lines):
        raise YamlError(f"line {lines[index][0]}: unexpected content")
    return value


def _prepare(text: str) -> list[tuple[int, int, str]]:
    prepared: list[tuple[int, int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        cleaned = _strip_comment(raw).rstrip()
        if not cleaned.strip():
            continue
        indent_source = cleaned[: len(cleaned) - len(cleaned.lstrip(" \t"))]
        if "\t" in indent_source:
            raise YamlError(f"line {lineno}: tabs are not allowed for indent")
        indent = len(indent_source)
        prepared.append((lineno, indent, cleaned.strip()))
    return prepared


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def _parse_node(lines: list[tuple[int, int, str]], index: int) -> tuple[Any, int]:
    _lineno, _indent, text = lines[index]
    if text.startswith("- "):
        return _parse_list(lines, index, lines[index][1])
    return _parse_map(lines, index, lines[index][1])


def _parse_map(
    lines: list[tuple[int, int, str]], index: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        lineno, line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"line {lineno}: unexpected indent")
        if text.startswith("- "):
            raise YamlError(f"line {lineno}: expected a key")
        split = _split_kv(text)
        if split is None:
            raise YamlError(f"line {lineno}: expected 'key: value'")
        key, rest = split
        if key in result:
            raise YamlError(f"line {lineno}: duplicate key {key!r}")
        index += 1
        if rest == "":
            result[key], index = _nested_or_null(lines, index, indent)
        elif rest in {"|", ">", "|-", ">-", "|+", ">+"}:
            raise UnsupportedYaml(f"line {lineno}: multiline scalars are not supported")
        else:
            result[key] = _parse_scalar(rest, lineno)
    return result, index


def _nested_or_null(
    lines: list[tuple[int, int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index < len(lines) and lines[index][1] > indent:
        return _parse_node(lines, index)
    return None, index


def _parse_list(
    lines: list[tuple[int, int, str]], index: int, indent: int
) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        lineno, line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise YamlError(f"line {lineno}: unexpected indent")
        if not text.startswith("- "):
            break
        rest = text[2:].strip()
        index += 1
        if rest == "":
            child, index = _nested_or_null(lines, index, indent)
            items.append(child)
            continue
        split = _split_kv(rest)
        if split is None:
            items.append(_parse_scalar(rest, lineno))
            continue
        key, value_text = split
        item: dict[str, Any] = {}
        if value_text == "":
            item[key], index = _nested_or_null(lines, index, indent)
        else:
            item[key] = _parse_scalar(value_text, lineno)
        if (
            index < len(lines)
            and lines[index][1] > indent
            and not lines[index][2].startswith("- ")
        ):
            extra, index = _parse_map(lines, index, lines[index][1])
            overlap = set(extra) & set(item)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise YamlError(f"line {lineno}: duplicate key in list item: {names}")
            item.update(extra)
        items.append(item)
    if not items:
        raise YamlError("expected a list item")
    return items, index


def _split_kv(text: str) -> tuple[str, str] | None:
    if text[:1] in {'"', "'"}:
        quote = text[0]
        end = text.find(quote, 1)
        if end < 0:
            return None
        tail = text[end + 1 :].lstrip()
        if not tail.startswith(":"):
            return None
        return text[1:end], tail[1:].strip()
    quote: str | None = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == ":" and (index == len(text) - 1 or text[index + 1] == " "):
            key = text[:index].strip()
            if not key:
                return None
            return key, text[index + 1 :].strip()
    return None


def _parse_scalar(text: str, lineno: int) -> Any:
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return _unquote(text)
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text == "[]":
        return []
    if text == "{}":
        return {}
    if text[:1] in {"[", "{", ">", "|"}:
        raise UnsupportedYaml(f"line {lineno}: unsupported YAML syntax {text!r}")
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def _unquote(text: str) -> str:
    quote = text[0]
    inner = text[1:-1]
    if quote == "'":
        return inner.replace("''", "'")
    out: list[str] = []
    index = 0
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    while index < len(inner):
        if inner[index] == "\\" and index + 1 < len(inner):
            out.append(escapes.get(inner[index + 1], inner[index + 1]))
            index += 2
            continue
        out.append(inner[index])
        index += 1
    return "".join(out)
