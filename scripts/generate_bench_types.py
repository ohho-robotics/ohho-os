#!/usr/bin/env python3
"""Generate Python and TypeScript types from ohho/bench/schemas/bench.v1.json.

The JSON Schema is the only source. This script writes:

    ohho/bench/generated/types.py
    ohho/bench/generated/bench.v1.ts

Usage:
    python scripts/generate_bench_types.py
    python scripts/generate_bench_types.py --check
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "ohho" / "bench" / "schemas" / "bench.v1.json"
PY_PATH = ROOT / "ohho" / "bench" / "generated" / "types.py"
TS_PATH = ROOT / "ohho" / "bench" / "generated" / "bench.v1.ts"
_REF_PREFIX = "#/$defs/"

_DEFS: dict[str, Any] = {}


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _load() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        _fail("schema root must be an object")
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict):
        _fail("$defs must be an object")
    global _DEFS
    _DEFS = defs
    return schema


def _schema_id(schema: dict[str, Any]) -> str:
    props = schema.get("properties")
    prop = props.get("schema") if isinstance(props, dict) else None
    const = prop.get("const") if isinstance(prop, dict) else None
    if not isinstance(const, str) or not const:
        _fail("root property schema.const must be a non-empty string")
    return const


def _follow(schema: dict[str, Any]) -> dict[str, Any]:
    seen: list[str] = []
    current = schema
    while "$ref" in current:
        ref = current["$ref"]
        if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
            _fail(f"unsupported $ref {ref!r}")
        if ref in seen:
            _fail(f"cyclic $ref {ref}")
        seen.append(ref)
        name = ref[len(_REF_PREFIX) :]
        target = _DEFS.get(name)
        if not isinstance(target, dict):
            _fail(f"unresolved $ref {ref}")
        current = target
    return current


def _py_literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    _fail(f"unsupported literal {value!r}")
    return ""


def _ts_literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return json.dumps(value)
    _fail(f"unsupported literal {value!r}")
    return ""


def _py_type(schema: dict[str, Any]) -> str:
    current = _follow(schema)
    if "const" in current:
        return f"Literal[{_py_literal(current['const'])}]"
    if "enum" in current:
        inner = ", ".join(_py_literal(item) for item in current["enum"])
        return f"Literal[{inner}]"
    kind = current.get("type")
    if kind == "string":
        return "str"
    if kind == "number":
        return "float"
    if kind == "integer":
        return "int"
    if kind == "boolean":
        return "bool"
    if kind == "array":
        items = current.get("items")
        if not isinstance(items, dict):
            _fail("array items must be a schema")
        return f"list[{_py_type(items)}]"
    if "allOf" in current or kind == "object" or "properties" in current:
        title = current.get("title")
        if isinstance(title, str):
            return title
        props = current.get("properties") or {}
        extra = current.get("additionalProperties", True)
        if not props and isinstance(extra, dict):
            return f"dict[str, {_py_type(extra)}]"
        _fail("object schema needs a title")
    _fail(f"unsupported schema fragment keys={sorted(current)}")
    return ""


def _ts_type(schema: dict[str, Any]) -> str:
    current = _follow(schema)
    if "const" in current:
        return _ts_literal(current["const"])
    if "enum" in current:
        return " | ".join(_ts_literal(item) for item in current["enum"])
    kind = current.get("type")
    if kind == "string":
        return "string"
    if kind in {"number", "integer"}:
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "array":
        items = current.get("items")
        if not isinstance(items, dict):
            _fail("array items must be a schema")
        inner = _ts_type(items)
        if inner.endswith("]") or "<" in inner:
            return f"Array<{inner}>"
        return f"{inner}[]"
    if "allOf" in current or kind == "object" or "properties" in current:
        title = current.get("title")
        if isinstance(title, str):
            return title
        props = current.get("properties") or {}
        extra = current.get("additionalProperties", True)
        if not props and isinstance(extra, dict):
            return f"Record<string, {_ts_type(extra)}>"
        _fail("object schema needs a title")
    _fail(f"unsupported schema fragment keys={sorted(current)}")
    return ""


def _merge(
    merged: list[tuple[str, dict[str, Any], bool]],
    index: dict[str, int],
    item: tuple[str, dict[str, Any], bool],
) -> None:
    name, sub, required = item
    if name in index:
        pos = index[name]
        _old_name, old_sub, old_required = merged[pos]
        if json.dumps(old_sub, sort_keys=True) != json.dumps(sub, sort_keys=True):
            _fail(f"field {name} is defined twice with different schemas")
        merged[pos] = (name, sub, old_required or required)
        return
    index[name] = len(merged)
    merged.append((name, sub, required))


def _own_fields(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    required = schema.get("required") or []
    required_set = set(required) if isinstance(required, list) else set()
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        _fail("properties must be an object")
    fields: list[tuple[str, dict[str, Any], bool]] = []
    for name, sub in props.items():
        if not isinstance(name, str) or not name.isidentifier():
            _fail(f"field {name!r} is not a Python identifier")
        if not isinstance(sub, dict):
            _fail(f"field {name} schema must be an object")
        fields.append((name, sub, name in required_set))
    return fields


def _flattened(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    current = _follow(schema)
    if "allOf" in current:
        merged: list[tuple[str, dict[str, Any], bool]] = []
        index: dict[str, int] = {}
        branches = current["allOf"]
        if not isinstance(branches, list):
            _fail("allOf must be an array")
        for branch in branches:
            if not isinstance(branch, dict):
                _fail("allOf branch must be a schema")
            for item in _flattened(branch):
                _merge(merged, index, item)
        for item in _own_fields(current):
            _merge(merged, index, item)
        return merged
    return _own_fields(current)


def _titles_of(schema: dict[str, Any]) -> set[str]:
    current = _follow(schema)
    title = current.get("title")
    object_like = (
        "allOf" in current or current.get("type") == "object" or "properties" in current
    )
    if isinstance(title, str) and object_like and current.get("type") != "array":
        if "const" in current or "enum" in current:
            return set()
        if current.get("type") in {"string", "number", "integer", "boolean"}:
            return set()
        return {title}
    if "allOf" in current:
        found: set[str] = set()
        for branch in current["allOf"]:
            if isinstance(branch, dict):
                found |= _titles_of(branch)
        return found
    if current.get("type") == "array" and isinstance(current.get("items"), dict):
        return _titles_of(current["items"])
    if current.get("type") == "object":
        found = set()
        props = current.get("properties") or {}
        if isinstance(props, dict):
            for sub in props.values():
                if isinstance(sub, dict):
                    found |= _titles_of(sub)
        extra = current.get("additionalProperties")
        if isinstance(extra, dict):
            found |= _titles_of(extra)
        return found
    return set()


def _additional(schema: dict[str, Any]) -> object:
    current = _follow(schema)
    if "allOf" in current:
        values: list[object] = []
        for branch in current["allOf"]:
            if isinstance(branch, dict):
                values.append(_additional(branch))
        if "additionalProperties" in current:
            values.append(current["additionalProperties"])
        if any(value is False for value in values):
            return False
        for value in values:
            if isinstance(value, dict):
                return value
        return True
    return current.get("additionalProperties", True)


def _phrase(additional: object) -> str:
    if additional is False:
        return "Additional properties are not allowed."
    if isinstance(additional, dict) and additional.get("type") == "string":
        return "Additional properties are allowed and must be strings."
    if isinstance(additional, dict) and additional.get("type") in {"number", "integer"}:
        return "Additional properties are allowed and must be numbers."
    return "Additional properties are allowed."


def _remember(named: dict[str, dict[str, Any]], schema: dict[str, Any]) -> None:
    title = schema.get("title")
    if not isinstance(title, str) or not title.isidentifier() or title.startswith("_"):
        _fail(f"bad title {title!r}")
    previous = named.get(title)
    if previous is not None and previous is not schema:
        _fail(f"duplicate title {title}")
    named[title] = schema


def _walk(schema: object, named: dict[str, dict[str, Any]]) -> None:
    if not isinstance(schema, dict):
        return
    if set(schema) == {"$ref"}:
        return
    if "allOf" in schema:
        branches = schema["allOf"]
        if isinstance(branches, list):
            for branch in branches:
                _walk(branch, named)
        if isinstance(schema.get("title"), str):
            _remember(named, schema)
        return
    if schema.get("type") == "object" or "properties" in schema:
        if isinstance(schema.get("title"), str):
            _remember(named, schema)
        props = schema.get("properties") or {}
        if isinstance(props, dict):
            for sub in props.values():
                _walk(sub, named)
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            _walk(extra, named)
        return
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        _walk(schema["items"], named)


def _ordered(named: dict[str, dict[str, Any]]) -> list[str]:
    deps = {name: _field_deps(schema) - {name} for name, schema in named.items()}
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            _fail(f"cycle in type {name}")
        if name not in deps:
            _fail(f"unknown type {name}")
        visiting.add(name)
        for dep in sorted(deps[name]):
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in named:
        visit(name)
    return ordered


def _field_deps(schema: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for _name, sub, _required in _flattened(schema):
        found |= _titles_of(sub)
    return found


def _wrap_words(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def _docstring(description: str, note: str) -> str:
    parts = [part.strip() for part in (description, note) if part and part.strip()]
    for part in parts:
        if '"""' in part:
            _fail("description contains a triple quote")
    if not parts:
        return ""
    wrapped = [_wrap_words(part, 76) for part in parts]
    if len(wrapped) == 1 and len(wrapped[0]) == 1 and len(wrapped[0][0]) <= 72:
        return f'    """{wrapped[0][0]}"""'
    lines = [f'    """{wrapped[0][0]}']
    lines.extend(f"    {line}" for line in wrapped[0][1:])
    for block in wrapped[1:]:
        lines.append("")
        lines.extend(f"    {line}" for line in block)
    lines.append('    """')
    return "\n".join(lines)


def _emit_class(
    name: str,
    fields: list[tuple[str, dict[str, Any], bool]],
    *,
    base: str | None,
    total_false: bool,
    doc: str,
) -> str:
    if base and total_false:
        header = f"class {name}({base}, total=False):"
    elif total_false:
        header = f"class {name}(TypedDict, total=False):"
    elif base:
        header = f"class {name}({base}):"
    else:
        header = f"class {name}(TypedDict):"
    lines = [header]
    if doc:
        lines.append(doc)
        if fields:
            # ruff format keeps a blank line between a class docstring and fields.
            lines.append("")
    if fields:
        for field_name, sub, _required in fields:
            lines.append(f"    {field_name}: {_py_type(sub)}")
    elif not doc:
        lines.append("    pass")
    return "\n".join(lines)


def _classes(name: str, schema: dict[str, Any]) -> list[str]:
    fields = _flattened(schema)
    required = [item for item in fields if item[2]]
    optional = [item for item in fields if not item[2]]
    description = schema.get("description")
    doc = _docstring(
        description if isinstance(description, str) else "",
        _phrase(_additional(schema)),
    )
    if required and optional:
        base = f"_{name}Required"
        return [
            _emit_class(base, required, base=None, total_false=False, doc=""),
            _emit_class(name, optional, base=base, total_false=True, doc=doc),
        ]
    if optional:
        return [_emit_class(name, optional, base=None, total_false=True, doc=doc)]
    return [_emit_class(name, required, base=None, total_false=False, doc=doc)]


def _render_python(
    schema_id: str, order: list[str], named: dict[str, dict[str, Any]]
) -> str:
    lines = [
        "# Generated from ohho/bench/schemas/bench.v1.json by scripts/generate_bench_types.py.",
        "# Do not edit. Regenerate with: python scripts/generate_bench_types.py",
        '"""Typed view of an ohho.bench/v1 benchmark result.',
        "",
        "Generated from ``ohho/bench/schemas/bench.v1.json``. Additional object fields",
        "are allowed. Validate with :func:`ohho.bench.schema.validate`.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Literal, TypedDict",
        "",
        f"BENCH_SCHEMA_ID: Literal[{json.dumps(schema_id)}] = {json.dumps(schema_id)}",
        "",
        "__all__ = [",
        '    "BENCH_SCHEMA_ID",',
    ]
    for name in order:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")
    lines.append("")
    chunks: list[str] = []
    for name in order:
        chunks.extend(_classes(name, named[name]))
    for index, chunk in enumerate(chunks):
        if index:
            lines.append("")
            lines.append("")
        lines.extend(chunk.splitlines())
    lines.append("")
    return "\n".join(lines)


def _ts_index(
    fields: list[tuple[str, dict[str, Any], bool]],
    additional: object,
) -> str | None:
    if not isinstance(additional, dict):
        return None
    if any(required is False for _name, _sub, required in fields):
        return None
    extra = _ts_type(additional)
    for _name, sub, _required in fields:
        if _ts_type(sub) != extra:
            return None
    return f"  [key: string]: {extra};"


def _render_typescript(
    schema_id: str,
    order: list[str],
    named: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "// Generated from ohho/bench/schemas/bench.v1.json by scripts/generate_bench_types.py.",
        "// Do not edit. Regenerate with: python scripts/generate_bench_types.py",
        "// Known fields are listed below. ohho.bench/v1 objects also allow additional fields.",
        "",
        "/** Schema id for an ohho.bench/v1 result. */",
        f"export const BENCH_SCHEMA_ID = {json.dumps(schema_id)} as const;",
        "",
    ]
    for name in order:
        schema = named[name]
        fields = _flattened(schema)
        description = schema.get("description")
        summary = description.strip() if isinstance(description, str) else ""
        note = _phrase(_additional(schema))
        lines.append("/**")
        for text in (summary, note):
            for line in _wrap_words(text, 76):
                lines.append(f" * {line}")
        lines.append(" */")
        lines.append(f"export interface {name} {{")
        for field_name, sub, required in fields:
            marker = "" if required else "?"
            lines.append(f"  {field_name}{marker}: {_ts_type(sub)};")
        index = _ts_index(fields, _additional(schema))
        if index is not None:
            lines.append(index)
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def render() -> tuple[str, str]:
    """Return ``(python, typescript)`` generated from the schema file."""
    schema = _load()
    schema_id = _schema_id(schema)
    named: dict[str, dict[str, Any]] = {}
    defs = schema.get("$defs", {})
    if isinstance(defs, dict):
        for sub in defs.values():
            _walk(sub, named)
    _walk(schema, named)
    if "BenchResult" not in named:
        _fail("root schema needs title BenchResult")
    order = _ordered(named)
    return _render_python(schema_id, order, named), _render_typescript(
        schema_id, order, named
    )


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _mismatch(path: Path, expected: str) -> str:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    if current == expected:
        return ""
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=_relative(path),
        tofile=f"{_relative(path)} (generated)",
    )
    return "".join(diff)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when generated files differ from the schema",
    )
    args = parser.parse_args(argv)
    python, typescript = render()
    targets = ((PY_PATH, python), (TS_PATH, typescript))
    if args.check:
        stale = False
        for path, text in targets:
            diff = _mismatch(path, text)
            if diff:
                stale = True
                print(diff, file=sys.stderr)
                print(
                    f"stale {_relative(path)} (run python scripts/generate_bench_types.py)",
                    file=sys.stderr,
                )
        return 1 if stale else 0
    for path, text in targets:
        _write(path, text)
        print(f"wrote {_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
