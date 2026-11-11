"""Stdlib YAML subset for ohho.job/v1 specs.

Accepted: comments, space indentation, block mappings and sequences,
plain and quoted scalars, ``true`` / ``false`` / ``null`` / ``~``,
integers, floats, and one-line flow collections. A single leading
``---`` is ignored.

Rejected with a parse error: tabs in indentation, multiline ``|`` and
``>`` strings, anchors, aliases, tags, merge keys, and multiple documents.
Quote a scalar that would otherwise look like one of those forms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_INT = re.compile(r"-?(0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(
    r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)\Z"
    r"|-?(?:0|[1-9][0-9]*)\.[0-9]+\Z"
    r"|-?\.[0-9]+\Z"
)
_TRUE = {"true", "True", "TRUE"}
_FALSE = {"false", "False", "FALSE"}
_NULL = {"null", "Null", "NULL", "~"}


class YamlParseError(Exception):
    """The subset parser could not read a document."""

    def __init__(self, line: int, message: str) -> None:
        self.line = line
        self.message = message
        super().__init__(f"line {line}: {message}")


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    text: str


def load_yaml_subset(text: str) -> object:
    """Parse ``text`` as one document in the supported subset."""
    lines = _prepare(text)
    if not lines:
        raise YamlParseError(1, "document is empty")
    value, index = _parse_block(lines, 0, lines[0].indent)
    if index != len(lines):
        raise YamlParseError(lines[index].number, "unexpected content")
    return value


def _prepare(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for number, raw in enumerate(text.splitlines(), 1):
        indent_part = raw[: len(raw) - len(raw.lstrip(" \t"))]
        if "\t" in indent_part:
            raise YamlParseError(number, "tabs are not allowed in indentation")
        body = _strip_comment(raw).rstrip()
        if not body.strip():
            continue
        indent = len(body) - len(body.lstrip(" "))
        content = body[indent:]
        if content == "---":
            if lines:
                raise YamlParseError(number, "multiple documents are not supported")
            continue
        if content == "...":
            continue
        lines.append(_Line(number, indent, content))
    return lines


def _strip_comment(raw: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(raw):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#":
            return raw[:index]
    return raw


def _parse_block(lines: list[_Line], index: int, indent: int) -> tuple[object, int]:
    line = lines[index]
    if line.indent != indent:
        raise YamlParseError(line.number, "unexpected indentation")
    if _is_seq_item(line.text):
        return _parse_seq(lines, index, indent)
    if _mapping_sep(line.text) is not None:
        return _parse_map(lines, index, indent)
    if index == 0:
        return _parse_inline(line.text, line.number), index + 1
    raise YamlParseError(line.number, "expected a mapping or sequence")


def _parse_map(lines: list[_Line], index: int, indent: int) -> tuple[dict, int]:
    mapping: dict[str, object] = {}
    while (
        index < len(lines)
        and lines[index].indent == indent
        and not _is_seq_item(lines[index].text)
    ):
        line = lines[index]
        key, inline = _split_key(line.text, line.number)
        if key in mapping:
            raise YamlParseError(line.number, _duplicate(key))
        if inline is None:
            index += 1
            if index < len(lines) and lines[index].indent > indent:
                mapping[key], index = _parse_block(lines, index, lines[index].indent)
            else:
                mapping[key] = None
        else:
            mapping[key] = _parse_inline(inline, line.number)
            index += 1
            if index < len(lines) and lines[index].indent > indent:
                raise YamlParseError(lines[index].number, "unexpected indentation")
    return mapping, index


def _parse_seq(lines: list[_Line], index: int, indent: int) -> tuple[list, int]:
    items: list[object] = []
    while (
        index < len(lines)
        and lines[index].indent == indent
        and _is_seq_item(lines[index].text)
    ):
        line = lines[index]
        rest = _seq_rest(line.text)
        if rest == "":
            index += 1
            if index >= len(lines) or lines[index].indent <= indent:
                raise YamlParseError(line.number, "expected a nested value")
            value, index = _parse_block(lines, index, lines[index].indent)
            items.append(value)
            continue
        if _is_seq_item(rest):
            raise YamlParseError(
                line.number, "nested sequence on the same line is not supported"
            )
        if _mapping_sep(rest) is not None:
            value, index = _parse_seq_mapping(lines, index, _seq_key_indent(line))
            items.append(value)
            continue
        items.append(_parse_inline(rest, line.number))
        index += 1
    return items, index


def _parse_seq_mapping(
    lines: list[_Line], index: int, key_indent: int
) -> tuple[dict, int]:
    mapping: dict[str, object] = {}
    line = lines[index]
    key, inline = _split_key(_seq_rest(line.text), line.number)
    if inline is None:
        index += 1
        if index < len(lines) and lines[index].indent > key_indent:
            mapping[key], index = _parse_block(lines, index, lines[index].indent)
        else:
            mapping[key] = None
    else:
        mapping[key] = _parse_inline(inline, line.number)
        index += 1
    while (
        index < len(lines)
        and lines[index].indent == key_indent
        and not _is_seq_item(lines[index].text)
    ):
        line = lines[index]
        key, inline = _split_key(line.text, line.number)
        if key in mapping:
            raise YamlParseError(line.number, _duplicate(key))
        if inline is None:
            index += 1
            if index < len(lines) and lines[index].indent > key_indent:
                mapping[key], index = _parse_block(lines, index, lines[index].indent)
            else:
                mapping[key] = None
        else:
            mapping[key] = _parse_inline(inline, line.number)
            index += 1
            if index < len(lines) and lines[index].indent > key_indent:
                raise YamlParseError(lines[index].number, "unexpected indentation")
    return mapping, index


def _is_seq_item(text: str) -> bool:
    return text == "-" or text.startswith("- ")


def _seq_rest(text: str) -> str:
    return text[1:].strip()


def _seq_key_indent(line: _Line) -> int:
    cursor = 1
    while cursor < len(line.text) and line.text[cursor] == " ":
        cursor += 1
    return line.indent + cursor


def _mapping_sep(text: str) -> int | None:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == ":" and (index + 1 == len(text) or text[index + 1] == " "):
            return index
    return None


def _split_key(text: str, line: int) -> tuple[str, str | None]:
    sep = _mapping_sep(text)
    if sep is None:
        raise YamlParseError(line, "expected a mapping key")
    key = _parse_key(text[:sep].strip(), line)
    if key == "<<":
        raise YamlParseError(line, "merge keys are not supported")
    rest = text[sep + 1 :].strip()
    if rest == "":
        return key, None
    return key, rest


def _parse_key(token: str, line: int) -> str:
    if token == "":
        raise YamlParseError(line, "expected a mapping key")
    value = _parse_scalar(token, line)
    if not isinstance(value, str):
        raise YamlParseError(line, "mapping key must be a string")
    return value


def _duplicate(key: str) -> str:
    return f"duplicate key {json.dumps(key, ensure_ascii=True)}"


def _parse_inline(text: str, line: int) -> object:
    if text[:1] in "&*!":
        raise YamlParseError(line, "anchors, aliases and tags are not supported")
    if text in ("|", ">", "|-", ">-", "|+", ">+"):
        raise YamlParseError(line, "multiline strings are not supported")
    if text.startswith("[") or text.startswith("{"):
        value, pos = _flow_value(text, 0, line)
        if text[pos:].strip():
            raise YamlParseError(line, "trailing content after flow value")
        return value
    return _parse_scalar(text, line)


def _parse_scalar(text: str, line: int) -> object:
    if text.startswith('"'):
        value, pos = _scan_double(text, 0, line)
        if text[pos:].strip():
            raise YamlParseError(line, "trailing content after quoted string")
        return value
    if text.startswith("'"):
        value, pos = _scan_single(text, 0, line)
        if text[pos:].strip():
            raise YamlParseError(line, "trailing content after quoted string")
        return value
    if text in _NULL:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def _scan_double(text: str, index: int, line: int) -> tuple[str, int]:
    if index >= len(text) or text[index] != '"':
        raise YamlParseError(line, "expected a double-quoted string")
    index += 1
    chars: list[str] = []
    escapes = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "/": "/"}
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                raise YamlParseError(line, "unterminated escape")
            nxt = text[index + 1]
            if nxt not in escapes:
                raise YamlParseError(line, f"unsupported escape \\{nxt}")
            chars.append(escapes[nxt])
            index += 2
            continue
        if char == '"':
            return "".join(chars), index + 1
        chars.append(char)
        index += 1
    raise YamlParseError(line, "unterminated double-quoted string")


def _scan_single(text: str, index: int, line: int) -> tuple[str, int]:
    if index >= len(text) or text[index] != "'":
        raise YamlParseError(line, "expected a single-quoted string")
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "'":
            if index + 1 < len(text) and text[index + 1] == "'":
                chars.append("'")
                index += 2
                continue
            return "".join(chars), index + 1
        chars.append(char)
        index += 1
    raise YamlParseError(line, "unterminated single-quoted string")


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] == " ":
        index += 1
    return index


def _flow_value(text: str, index: int, line: int) -> tuple[object, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise YamlParseError(line, "expected a flow value")
    char = text[index]
    if char == "[":
        return _flow_seq(text, index, line)
    if char == "{":
        return _flow_map(text, index, line)
    if char == '"':
        return _scan_double(text, index, line)
    if char == "'":
        return _scan_single(text, index, line)
    if char in "&*!":
        raise YamlParseError(line, "anchors, aliases and tags are not supported")
    end = index
    while end < len(text) and text[end] not in ",]}":
        end += 1
    token = text[index:end].strip()
    if token == "":
        raise YamlParseError(line, "expected a flow value")
    return _parse_scalar(token, line), end


def _flow_seq(text: str, index: int, line: int) -> tuple[list, int]:
    index += 1
    items: list[object] = []
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "]":
        return items, index + 1
    while True:
        value, index = _flow_value(text, index, line)
        items.append(value)
        index = _skip_ws(text, index)
        if index >= len(text):
            raise YamlParseError(line, "unclosed flow sequence")
        if text[index] == ",":
            index = _skip_ws(text, index + 1)
            if index < len(text) and text[index] == "]":
                return items, index + 1
            continue
        if text[index] == "]":
            return items, index + 1
        raise YamlParseError(line, "expected ',' or ']' in flow sequence")


def _flow_map(text: str, index: int, line: int) -> tuple[dict, int]:
    index += 1
    mapping: dict[str, object] = {}
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "}":
        return mapping, index + 1
    while True:
        key, index = _flow_key(text, index, line)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise YamlParseError(line, "expected ':' in flow mapping")
        value, index = _flow_value(text, index + 1, line)
        if key in mapping:
            raise YamlParseError(line, _duplicate(key))
        mapping[key] = value
        index = _skip_ws(text, index)
        if index >= len(text):
            raise YamlParseError(line, "unclosed flow mapping")
        if text[index] == ",":
            index = _skip_ws(text, index + 1)
            if index < len(text) and text[index] == "}":
                return mapping, index + 1
            continue
        if text[index] == "}":
            return mapping, index + 1
        raise YamlParseError(line, "expected ',' or '}' in flow mapping")


def _flow_key(text: str, index: int, line: int) -> tuple[str, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise YamlParseError(line, "expected a flow key")
    if text[index] in ("'", '"'):
        if text[index] == '"':
            value, index = _scan_double(text, index, line)
        else:
            value, index = _scan_single(text, index, line)
        return value, index
    end = index
    while end < len(text) and text[end] not in ":{},[]":
        end += 1
    token = text[index:end].strip()
    if token == "":
        raise YamlParseError(line, "expected a flow key")
    key = _parse_scalar(token, line)
    if not isinstance(key, str):
        raise YamlParseError(line, "mapping key must be a string")
    return key, end
