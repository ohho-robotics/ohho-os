"""TOML read and surgical edit for Codex ``~/.codex/config.toml``.

Python 3.10 has no :mod:`tomllib`, and the base install cannot take a TOML
dependency. This module parses the subset Codex configs use (comments, quoted
and bare keys, strings, numbers, booleans, arrays, inline tables, dotted keys,
standard tables, and arrays of tables) and rewrites only the
``[mcp_servers.ohho]`` entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_BARE = re.compile(r"[A-Za-z0-9_-]")
_ATOM = re.compile(
    r"(?:"
    r"[+-]?0x[0-9A-Fa-f](?:_?[0-9A-Fa-f])*|"
    r"[+-]?0o[0-7](?:_?[0-7])*|"
    r"[+-]?0b[0-1](?:_?[0-1])*|"
    r"[+-]?(?:inf|nan)|"
    r"[+-]?(?:0|[1-9](?:_?[0-9])*)(?:\.[0-9](?:_?[0-9])*)?(?:[eE][+-]?[0-9](?:_?[0-9])*)?"
    r")"
)
_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})?)?"
)
_LOCAL_TIME = re.compile(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_ESCAPES = {
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    '"': '"',
    "\\": "\\",
}
_BOUNDARY = set(" \t\r\n#,]}")


class TomlError(ValueError):
    """The document is not TOML this editor can safely update."""


@dataclass
class _Span:
    start: int
    end: int
    kind: str
    name: str
    table: str = ""


@dataclass
class _Parsed:
    data: Dict[str, Any]
    headers: List[_Span] = field(default_factory=list)
    keys: List[_Span] = field(default_factory=list)


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.root: Dict[str, Any] = {}
        self.current: Dict[str, Any] = self.root
        self.current_path: Tuple[str, ...] = ()
        self.headers: List[_Span] = []
        self.keys: List[_Span] = []
        self.std_tables: set[Tuple[str, ...]] = set()
        self.array_tables: set[Tuple[str, ...]] = set()
        self.implicit: set[Tuple[str, ...]] = set()

    def parse(self) -> _Parsed:
        while True:
            self._skip_ws(newlines=True)
            if self._eof():
                break
            if self._peek() == "[":
                self._header()
            else:
                self._keyval()
        self._finish_header_ends()
        return _Parsed(self.root, self.headers, self.keys)

    def _eof(self) -> bool:
        return self.pos >= len(self.text)

    def _peek(self) -> str:
        if self._eof():
            return ""
        return self.text[self.pos]

    def _get(self) -> str:
        if self._eof():
            self._error("unexpected end of file")
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def _error(self, message: str) -> None:
        line = self.text.count("\n", 0, self.pos) + 1
        last = self.text.rfind("\n", 0, self.pos)
        col = self.pos - last
        raise TomlError(f"line {line}, column {col}: {message}")

    def _line_start(self) -> int:
        return self.text.rfind("\n", 0, self.pos) + 1

    def _skip_comment(self) -> None:
        if self._peek() != "#":
            return
        self.pos += 1
        while not self._eof() and self._peek() not in "\r\n":
            self.pos += 1

    def _skip_ws(self, *, newlines: bool) -> None:
        while not self._eof():
            ch = self._peek()
            if ch in " \t":
                self.pos += 1
                continue
            if ch in "\r\n":
                if not newlines:
                    break
                self.pos += 1
                continue
            if ch == "#":
                self._skip_comment()
                if newlines and not self._eof() and self._peek() in "\r\n":
                    if self._peek() == "\r":
                        self.pos += 1
                    if not self._eof() and self._peek() == "\n":
                        self.pos += 1
                continue
            break

    def _consume_eol(self) -> None:
        while not self._eof() and self._peek() in " \t":
            self.pos += 1
        if not self._eof() and self._peek() == "#":
            self._skip_comment()
        if self._eof():
            return
        if self._peek() == "\r":
            self.pos += 1
        if not self._eof() and self._peek() == "\n":
            self.pos += 1
            return
        if self._eof():
            return
        self._error("expected newline")

    def _expect(self, ch: str) -> None:
        if self._peek() != ch:
            self._error(f"expected {ch!r}")
        self.pos += 1

    def _boundary(self) -> None:
        if self._eof():
            return
        if self._peek() not in _BOUNDARY:
            self._error("invalid value")

    def _parse_bare(self) -> str:
        start = self.pos
        while not self._eof() and _BARE.match(self._peek()):
            self.pos += 1
        if self.pos == start:
            self._error("expected key")
        return self.text[start : self.pos]

    def _escape(self, *, multiline: bool) -> str:
        if self._eof():
            self._error("unterminated escape")
        ch = self._get()
        if ch in _ESCAPES:
            return _ESCAPES[ch]
        if ch == "u":
            return self._hex(4)
        if ch == "U":
            return self._hex(8)
        if multiline and ch in "\r\n":
            if ch == "\r" and self._peek() == "\n":
                self.pos += 1
            while not self._eof() and self._peek() in " \t\r\n":
                self.pos += 1
            return ""
        self._error(f"unknown escape \\{ch}")

    def _hex(self, count: int) -> str:
        raw = self.text[self.pos : self.pos + count]
        if len(raw) < count or any(c not in "0123456789abcdefABCDEF" for c in raw):
            self._error("invalid unicode escape")
        self.pos += count
        code = int(raw, 16)
        if 0xD800 <= code <= 0xDFFF or code > 0x10FFFF:
            self._error("invalid unicode escape")
        return chr(code)

    def _parse_string(self) -> str:
        if self.text.startswith('"""', self.pos) or self.text.startswith(
            "'''", self.pos
        ):
            return self._multiline_string()
        quote = self._peek()
        if quote not in "\"'":
            self._error("expected string")
        basic = quote == '"'
        self.pos += 1
        chars: List[str] = []
        while not self._eof():
            ch = self._get()
            if ch == quote:
                return "".join(chars)
            if ch in "\r\n":
                self._error("unterminated string")
            if basic and ch == "\\":
                chars.append(self._escape(multiline=False))
                continue
            chars.append(ch)
        self._error("unterminated string")

    def _multiline_string(self) -> str:
        basic = self.text.startswith('"""', self.pos)
        quote = '"""' if basic else "'''"
        self.pos += 3
        if self._peek() == "\r":
            self.pos += 1
        if self._peek() == "\n":
            self.pos += 1
        chars: List[str] = []
        while not self._eof():
            if self.text.startswith(quote, self.pos):
                self.pos += 3
                return "".join(chars)
            ch = self._get()
            if basic and ch == "\\":
                chars.append(self._escape(multiline=True))
                continue
            chars.append(ch)
        self._error("unterminated string")

    def _parse_key(self) -> List[str]:
        parts: List[str] = []
        while True:
            self._skip_ws(newlines=False)
            if self._peek() in "\"'":
                parts.append(self._parse_string())
            else:
                parts.append(self._parse_bare())
            self._skip_ws(newlines=False)
            if self._peek() == ".":
                self.pos += 1
                continue
            break
        if not parts:
            self._error("empty key")
        return parts

    def _match_bool(self) -> Optional[bool]:
        for word, value in (("true", True), ("false", False)):
            if not self.text.startswith(word, self.pos):
                continue
            after = self.pos + len(word)
            if after >= len(self.text) or self.text[after] in _BOUNDARY:
                self.pos = after
                return value
        return None

    def _parse_datetime_or_time(self) -> Optional[str]:
        match = _DATETIME.match(self.text, self.pos)
        if match is None and self._peek().isdigit():
            match = _LOCAL_TIME.match(self.text, self.pos)
        if match is None:
            return None
        end = match.end()
        if end < len(self.text) and self.text[end] not in _BOUNDARY:
            return None
        self.pos = end
        return match.group(0)

    def _parse_atom(self) -> Any:
        dated = self._parse_datetime_or_time()
        if dated is not None:
            return dated
        match = _ATOM.match(self.text, self.pos)
        if match is None:
            self._error("invalid value")
        token = match.group(0)
        self.pos = match.end()
        self._boundary()
        body = token[1:] if token[0] in "+-" else token
        lower = body.lower()
        if lower in {"inf", "nan"}:
            return float(token)
        if lower.startswith(("0x", "0o", "0b")):
            base = {"0x": 16, "0o": 8, "0b": 2}[lower[:2]]
            return int(body[:2] + body[2:].replace("_", ""), base)
        if "." in body or "e" in lower:
            return float(token.replace("_", ""))
        return int(token.replace("_", ""))

    def _parse_array(self) -> List[Any]:
        self._expect("[")
        items: List[Any] = []
        while True:
            self._skip_ws(newlines=True)
            if self._eof():
                self._error("unterminated array")
            if self._peek() == "]":
                self.pos += 1
                return items
            items.append(self._parse_value())
            self._skip_ws(newlines=True)
            if self._peek() == ",":
                self.pos += 1
                continue
            if self._peek() == "]":
                self.pos += 1
                return items
            self._error("expected ',' or ']'")

    def _parse_inline_table(self) -> Dict[str, Any]:
        self._expect("{")
        table: Dict[str, Any] = {}
        while True:
            self._skip_ws(newlines=True)
            if self._eof():
                self._error("unterminated inline table")
            if self._peek() == "}":
                self.pos += 1
                return table
            parts = self._parse_key()
            self._skip_ws(newlines=True)
            self._expect("=")
            self._skip_ws(newlines=True)
            value = self._parse_value()
            self._assign_into(table, self.current_path, parts, value)
            self._skip_ws(newlines=True)
            if self._peek() == ",":
                self.pos += 1
                continue
            if self._peek() == "}":
                self.pos += 1
                return table
            self._error("expected ',' or '}'")

    def _parse_value(self) -> Any:
        self._skip_ws(newlines=False)
        if self._eof():
            self._error("missing value")
        ch = self._peek()
        if ch in "\"'":
            return self._parse_string()
        if ch == "[":
            return self._parse_array()
        if ch == "{":
            return self._parse_inline_table()
        boolean = self._match_bool()
        if boolean is not None:
            return boolean
        return self._parse_atom()

    def _assign_into(
        self,
        table: Dict[str, Any],
        base: Tuple[str, ...],
        parts: List[str],
        value: Any,
    ) -> None:
        node = table
        for index, part in enumerate(parts[:-1]):
            path = base + tuple(parts[: index + 1])
            existing = node.get(part)
            if existing is None:
                node[part] = {}
                self.implicit.add(path)
            elif not isinstance(existing, dict):
                self._error(f"{'.'.join(path)} is not a table")
            node = node[part]
        last = parts[-1]
        if last in node:
            self._error(f"duplicate key {'.'.join(parts)}")
        node[last] = value

    def _open_table(self, parts: List[str], array: bool) -> Dict[str, Any]:
        if not parts:
            self._error("empty table name")
        node: Dict[str, Any] = self.root
        for index, part in enumerate(parts[:-1]):
            path = tuple(parts[: index + 1])
            existing = node.get(part)
            if existing is None:
                node[part] = {}
                self.implicit.add(path)
            elif not isinstance(existing, dict):
                self._error(f"{'.'.join(path)} is not a table")
            child = node[part]
            if not isinstance(child, dict):
                self._error(f"{'.'.join(path)} is not a table")
            node = child
        last = parts[-1]
        path = tuple(parts)
        if array:
            if path in self.std_tables:
                self._error(f"duplicate table [{'.'.join(parts)}]")
            existing = node.get(last)
            if existing is None:
                node[last] = []
            elif not isinstance(existing, list):
                self._error(f"{'.'.join(parts)} is not an array of tables")
            item: Dict[str, Any] = {}
            listing = node[last]
            if not isinstance(listing, list):
                self._error(f"{'.'.join(parts)} is not an array of tables")
            listing.append(item)
            self.array_tables.add(path)
            return item
        if path in self.std_tables or path in self.array_tables:
            self._error(f"duplicate table [{'.'.join(parts)}]")
        existing = node.get(last)
        if existing is None:
            created: Dict[str, Any] = {}
            node[last] = created
            self.std_tables.add(path)
            self.implicit.discard(path)
            return created
        if isinstance(existing, dict):
            self.std_tables.add(path)
            self.implicit.discard(path)
            return existing
        self._error(f"{'.'.join(parts)} is not a table")

    def _header(self) -> None:
        line_start = self._line_start()
        array = self.text.startswith("[[", self.pos)
        self.pos += 2 if array else 1
        parts = self._parse_key()
        self._skip_ws(newlines=False)
        self._expect("]")
        if array:
            self._skip_ws(newlines=False)
            self._expect("]")
        name = ".".join(parts)
        self.headers.append(_Span(line_start, -1, "header", name))
        self.current = self._open_table(parts, array)
        self.current_path = tuple(parts)
        self._consume_eol()

    def _keyval(self) -> None:
        line_start = self._line_start()
        parts = self._parse_key()
        self._skip_ws(newlines=True)
        self._expect("=")
        self._skip_ws(newlines=True)
        value = self._parse_value()
        self._consume_eol()
        self.keys.append(
            _Span(
                line_start,
                self.pos,
                "key",
                ".".join(parts),
                ".".join(self.current_path),
            )
        )
        self._assign_into(self.current, self.current_path, parts, value)

    def _finish_header_ends(self) -> None:
        for index, span in enumerate(self.headers):
            if index + 1 < len(self.headers):
                span.end = self.headers[index + 1].start
            else:
                span.end = len(self.text)


def parse_toml(text: str) -> Dict[str, Any]:
    """Parse ``text`` or raise :class:`TomlError`."""
    if text.startswith("\ufeff"):
        text = text[1:]
    return _Parser(text).parse().data


def toml_basic_string(value: str) -> str:
    """Return ``value`` as a TOML basic string."""
    chars = ['"']
    for ch in value:
        if ch == "\\":
            chars.append("\\\\")
        elif ch == '"':
            chars.append('\\"')
        elif ch == "\b":
            chars.append("\\b")
        elif ch == "\t":
            chars.append("\\t")
        elif ch == "\n":
            chars.append("\\n")
        elif ch == "\f":
            chars.append("\\f")
        elif ch == "\r":
            chars.append("\\r")
        elif ord(ch) < 0x20:
            chars.append(f"\\u{ord(ch):04x}")
        else:
            chars.append(ch)
    chars.append('"')
    return "".join(chars)


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _render_server(command: str, args: List[str], nl: str) -> str:
    rendered = ", ".join(toml_basic_string(arg) for arg in args)
    return (
        f"[mcp_servers.ohho]{nl}"
        f"command = {toml_basic_string(command)}{nl}"
        f"args = [{rendered}]{nl}"
    )


def _merge_spans(text: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: List[List[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        gap = text[merged[-1][1] : start]
        if gap.strip() == "":
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(item[0], item[1]) for item in merged]


def _append_block(text: str, block: str, nl: str) -> str:
    if not block.endswith(nl):
        block += nl
    if text.strip() == "":
        return block
    if not text.endswith("\n"):
        text += nl
    if not text.endswith("\n\n") and not text.endswith("\r\n\r\n"):
        text += nl
    return text + block


def _replace_span(text: str, start: int, end: int, block: str, nl: str) -> str:
    if block and not block.endswith(nl):
        block += nl
    if block and end < len(text) and not block.endswith(nl + nl):
        block += nl
    return text[:start] + block + text[end:]


def _ohho_regions(parsed: _Parsed) -> List[Tuple[int, int, str]]:
    regions: List[Tuple[int, int, str]] = []
    for span in parsed.headers:
        if span.name == "mcp_servers.ohho" or span.name.startswith("mcp_servers.ohho."):
            regions.append((span.start, span.end, "header"))
    for span in parsed.keys:
        if span.table == "mcp_servers" and (
            span.name == "ohho" or span.name.startswith("ohho.")
        ):
            regions.append((span.start, span.end, "key"))
    return regions


def edit_codex_server(text: str, *, command: str, args: List[str], remove: bool) -> str:
    """Insert, replace, or remove the ``[mcp_servers.ohho]`` table.

    Unrelated tables, keys, and comments stay in place. Raises
    :class:`TomlError` when ``text`` does not parse.
    """
    original = text
    if text.startswith("\ufeff"):
        text = text[1:]
    nl = _newline(text) if text else "\n"
    if text.strip() == "":
        if remove:
            return original
        return _render_server(command, args, "\n")
    parsed = _Parser(text).parse()
    regions = _ohho_regions(parsed)
    block = "" if remove else _render_server(command, args, nl)
    if not regions:
        if remove:
            return original
        new = _append_block(text, block, nl)
    else:
        header_regions = [
            (start, end) for start, end, kind in regions if kind == "header"
        ]
        key_regions = [(start, end) for start, end, kind in regions if kind == "key"]
        merged_headers = _merge_spans(text, header_regions)
        if header_regions and not key_regions and len(merged_headers) == 1:
            start, end = merged_headers[0]
            new = _replace_span(text, start, end, block, nl)
        else:
            new = text
            for start, end in reversed(
                _merge_spans(text, header_regions + key_regions)
            ):
                new = new[:start] + new[end:]
            if not remove:
                new = _append_block(new, block, nl)
    try:
        parse_toml(new)
    except TomlError as exc:
        raise RuntimeError(f"edited TOML did not parse: {exc}") from exc
    return new
