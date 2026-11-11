#!/usr/bin/env python3
"""Generate Python and TypeScript types from the ohho.job/v1 JSON Schema.

The schema ``ohho/serve/job/schema.json`` is the only source. This script
writes ``ohho/serve/job/generated.py`` and ``schemas/ts/job.v1.d.ts``.

Usage:
    python scripts/generate_job_types.py
    python scripts/generate_job_types.py --check
"""

from __future__ import annotations

import argparse
import json
import keyword
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_REL = Path("ohho/serve/job/schema.json")
PYTHON_REL = Path("ohho/serve/job/generated.py")
TYPESCRIPT_REL = Path("schemas/ts/job.v1.d.ts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if generated files differ from the schema",
    )
    args = parser.parse_args(argv)
    if args.check:
        stale = check()
        if stale:
            joined = ", ".join(stale)
            print(
                f"stale {joined} (run python scripts/generate_job_types.py)",
                file=sys.stderr,
            )
            return 1
        return 0
    written = write()
    print("wrote " + ", ".join(written))
    return 0


def check() -> list[str]:
    """Return repo-relative paths whose bytes do not match the schema."""
    root = repo_root()
    rendered = render(load_schema(root))
    stale: list[str] = []
    for rel, text in rendered.items():
        path = root / rel
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            stale.append(rel)
    return stale


def write() -> list[str]:
    """Write generated files with LF newlines. Return the relative paths."""
    root = repo_root()
    rendered = render(load_schema(root))
    for rel, text in rendered.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return list(rendered)


def repo_root() -> Path:
    return ROOT


def load_schema(root: Path | None = None) -> dict[str, Any]:
    path = (ROOT if root is None else root) / SCHEMA_REL
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("schema root must be an object")
    return schema


def render(schema: dict[str, Any]) -> dict[str, str]:
    """Return generated file text keyed by repo-relative path."""
    model = _model(schema)
    return {
        PYTHON_REL.as_posix(): _render_python(model),
        TYPESCRIPT_REL.as_posix(): _render_typescript(model),
    }


class _Alias:
    def __init__(self, name: str, kind: str, value: Any) -> None:
        self.name = name
        self.kind = kind
        self.value = value


class _Field:
    def __init__(
        self,
        name: str,
        annotation: str,
        ts_type: str,
        kind: str,
        required: bool,
        type_name: str | None,
    ) -> None:
        self.name = name
        self.annotation = annotation
        self.ts_type = ts_type
        self.kind = kind
        self.required = required
        self.type_name = type_name


class _Class:
    def __init__(self, name: str, description: str, fields: list[_Field]) -> None:
        self.name = name
        self.description = description
        self.fields = fields


class _Model:
    def __init__(
        self,
        schema_id: str,
        schema_version: str,
        aliases: list[_Alias],
        classes: list[_Class],
    ) -> None:
        self.schema_id = schema_id
        self.schema_version = schema_version
        self.aliases = aliases
        self.classes = classes


def _model(schema: dict[str, Any]) -> _Model:
    version = _schema_version(schema)
    aliases = _aliases(schema)
    classes = _classes(schema)
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError("schema $id must be a non-empty string")
    return _Model(schema_id, version, aliases, classes)


def _schema_version(schema: dict[str, Any]) -> str:
    required = schema.get("required")
    if not isinstance(required, list) or "schema_version" not in required:
        raise ValueError("schema_version must be required")
    prop = schema.get("properties", {}).get("schema_version")
    if not isinstance(prop, dict):
        raise ValueError("schema_version property is missing")
    value = prop.get("const")
    if not isinstance(value, str) or not value:
        raise ValueError("schema_version const must be a non-empty string")
    return value


def _aliases(schema: dict[str, Any]) -> list[_Alias]:
    found: list[_Alias] = []
    seen: dict[str, str] = {}
    for node in _nodes(schema):
        title = node.get("title")
        if not isinstance(title, str):
            continue
        if "enum" in node and "const" in node:
            raise ValueError(f"{title} has both enum and const")
        if "enum" in node:
            kind = "enum"
            value = node["enum"]
        elif "const" in node:
            kind = "const"
            value = node["const"]
        else:
            continue
        encoded = json.dumps(value, ensure_ascii=True)
        previous = seen.get(title)
        if previous is not None and previous != encoded:
            raise ValueError(f"title {title} is defined twice")
        if previous is None:
            seen[title] = encoded
            found.append(_Alias(title, kind, value))
    if not any(alias.kind == "const" for alias in found):
        raise ValueError("schema_version title alias is missing")
    return found


def _classes(schema: dict[str, Any]) -> list[_Class]:
    objects: list[tuple[str, dict[str, Any]]] = []
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict):
        raise ValueError("$defs must be an object")
    for name, node in defs.items():
        if not isinstance(node, dict) or node.get("type") != "object":
            raise ValueError(f"$defs.{name} must be an object schema")
        if node.get("title") != name:
            raise ValueError(f"$defs.{name} title must match its name")
        objects.append((name, node))
    if schema.get("type") != "object":
        raise ValueError("root schema must be an object")
    root_name = schema.get("title")
    if not isinstance(root_name, str) or not root_name:
        raise ValueError("root title is required")
    objects.append((root_name, schema))
    deps = {name: _direct_refs(node) - {name} for name, node in objects}
    order = _topo([name for name, _node in objects], deps)
    by_name = dict(objects)
    return [
        _Class(name, _description(by_name[name], name), _fields(by_name[name]))
        for name in order
    ]


def _fields(node: dict[str, Any]) -> list[_Field]:
    props = node.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("properties must be an object")
    required_list = node.get("required", [])
    if not isinstance(required_list, list):
        raise ValueError("required must be an array")
    required = {str(name) for name in required_list}
    names = [name for name in props if name in required]
    names.extend(name for name in props if name not in required)
    fields: list[_Field] = []
    for name in names:
        _check_identifier(str(name))
        sub = props[name]
        if not isinstance(sub, dict):
            raise ValueError(f"property {name} must be a schema object")
        annotation, ts_type, kind, type_name = _field_type(sub)
        fields.append(
            _Field(
                str(name),
                annotation,
                ts_type,
                kind,
                name in required,
                type_name,
            )
        )
    return fields


def _field_type(node: dict[str, Any]) -> tuple[str, str, str, str | None]:
    """Return Python annotation, TypeScript type, kind, and object class name."""
    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        return name, name, "object", name
    if isinstance(node.get("title"), str) and ("enum" in node or "const" in node):
        title = str(node["title"])
        return title, title, "scalar", None
    kind = node.get("type")
    if kind == "string":
        return "str", "string", "scalar", None
    if kind == "boolean":
        return "bool", "boolean", "scalar", None
    if kind == "integer":
        return "int", "number", "scalar", None
    if kind == "number":
        return "int | float", "number", "scalar", None
    if kind == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise ValueError("array items must be a schema object")
        inner, ts_inner, item_kind, type_name = _field_type(items)
        if item_kind == "object":
            return f"tuple[{inner}, ...]", f"{ts_inner}[]", "object_array", type_name
        return f"tuple[{inner}, ...]", f"{ts_inner}[]", "array", None
    raise ValueError(f"unsupported schema node: {node}")


def _check_identifier(name: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"property {name!r} is not a Python identifier")


def _description(node: dict[str, Any], name: str) -> str:
    text = node.get("description")
    if not isinstance(text, str) or not text.strip():
        return f"Generated {name}."
    cleaned = " ".join(text.split())
    if '"""' in cleaned or "\\" in cleaned:
        raise ValueError(f"description for {name} must be a plain line")
    return cleaned


def _direct_refs(node: dict[str, Any]) -> set[str]:
    return _ref_names(node.get("properties", {}))


def _ref_names(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            found.add(ref.rsplit("/", 1)[-1])
        for value in node.values():
            found |= _ref_names(value)
    elif isinstance(node, list):
        for value in node:
            found |= _ref_names(value)
    return found


def _nodes(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(_nodes(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_nodes(value))
    return found


def _topo(names: list[str], deps: dict[str, set[str]]) -> list[str]:
    remaining = list(names)
    ordered: list[str] = []
    while remaining:
        ready = [name for name in remaining if deps[name] <= set(ordered)]
        if not ready:
            raise ValueError("cyclic $ref: " + ", ".join(remaining))
        pick = ready[0]
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def _version_alias(model: _Model) -> str:
    matches = [
        alias.name
        for alias in model.aliases
        if alias.kind == "const" and alias.value == model.schema_version
    ]
    if len(matches) != 1:
        raise ValueError("schema_version const needs exactly one title")
    return matches[0]


def _render_python(model: _Model) -> str:
    version = json.dumps(model.schema_version, ensure_ascii=True)
    version_alias = _version_alias(model)
    lines = [
        '"""Generated from ohho/serve/job/schema.json. Do not edit.',
        "",
        f"Schema: {model.schema_version}",
        "Regenerate: python scripts/generate_job_types.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Mapping",
        "from dataclasses import dataclass",
        "from typing import Any, Literal",
        "",
    ]
    for alias in model.aliases:
        if alias.kind == "enum":
            lits = ", ".join(
                json.dumps(item, ensure_ascii=True) for item in alias.value
            )
            lines.append(f"{alias.name} = Literal[{lits}]")
        else:
            lines.append(
                f"{alias.name} = Literal[{json.dumps(alias.value, ensure_ascii=True)}]"
            )
    lines.append("")
    lines.append(f"SCHEMA_ID = {json.dumps(model.schema_id, ensure_ascii=True)}")
    lines.append(f"SCHEMA_VERSION: {version_alias} = {version}")
    for cls in model.classes:
        lines.append("")
        lines.append("")
        lines.extend(_render_python_class(cls))
    text = "\n".join(lines).rstrip() + "\n"
    return text


def _render_python_class(cls: _Class) -> list[str]:
    lines = [
        "@dataclass(frozen=True)",
        f"class {cls.name}:",
        f'    """{cls.description}"""',
        "",
    ]
    for field in cls.fields:
        if field.required:
            lines.append(f"    {field.name}: {field.annotation}")
        else:
            lines.append(f"    {field.name}: {field.annotation} | None = None")
    lines.append("")
    lines.extend(_render_from_dict(cls))
    lines.append("")
    lines.extend(_render_to_dict(cls))
    return lines


def _render_from_dict(cls: _Class) -> list[str]:
    signature = f"    def from_dict(cls, data: Mapping[str, Any]) -> {cls.name}:"
    lines = [
        "    @classmethod",
        signature,
        '        """Build this object from a validated mapping."""',
    ]
    for field in cls.fields:
        key = json.dumps(field.name, ensure_ascii=True)
        if field.kind == "object":
            assert field.type_name is not None
            call = f"{field.type_name}.from_dict(data[{key}])"
        elif field.kind == "object_array":
            assert field.type_name is not None
            call = f"tuple({field.type_name}.from_dict(item) for item in data[{key}])"
        elif field.kind == "array":
            call = f"tuple(data[{key}])"
        else:
            call = f"data[{key}]"
        if field.required:
            lines.append(f"        {field.name} = {call}")
        else:
            lines.append(f"        if {key} in data:")
            lines.append(f"            {field.name} = {call}")
            lines.append("        else:")
            lines.append(f"            {field.name} = None")
    lines.append("        return cls(")
    for field in cls.fields:
        lines.append(f"            {field.name}={field.name},")
    lines.append("        )")
    return lines


def _render_to_dict(cls: _Class) -> list[str]:
    lines = [
        "    def to_dict(self) -> dict[str, Any]:",
        '        """Return this object as a JSON-compatible dict."""',
        "        out: dict[str, Any] = {}",
    ]
    for field in cls.fields:
        key = json.dumps(field.name, ensure_ascii=True)
        if field.kind == "object":
            expr = f"self.{field.name}.to_dict()"
        elif field.kind == "object_array":
            expr = f"[item.to_dict() for item in self.{field.name}]"
        elif field.kind == "array":
            expr = f"list(self.{field.name})"
        else:
            expr = f"self.{field.name}"
        assignment = f"out[{key}] = {expr}"
        if field.required:
            lines.append(f"        {assignment}")
        else:
            lines.append(f"        if self.{field.name} is not None:")
            lines.append(f"            {assignment}")
    lines.append("        return out")
    return lines


def _render_typescript(model: _Model) -> str:
    version = json.dumps(model.schema_version, ensure_ascii=True)
    schema_id = json.dumps(model.schema_id, ensure_ascii=True)
    lines = [
        "/**",
        " * Generated from ohho/serve/job/schema.json. Do not edit.",
        " *",
        f" * Schema: {model.schema_version}",
        " * Regenerate: python scripts/generate_job_types.py",
        " */",
        "",
        f"export const JOB_SCHEMA_VERSION = {version} as const;",
        f"export const JOB_SCHEMA_ID = {schema_id} as const;",
        "",
    ]
    for alias in model.aliases:
        if alias.kind == "enum":
            union = " | ".join(
                json.dumps(item, ensure_ascii=True) for item in alias.value
            )
            lines.append(f"export type {alias.name} = {union};")
        else:
            literal = json.dumps(alias.value, ensure_ascii=True)
            lines.append(f"export type {alias.name} = {literal};")
    lines.append("")
    for cls in model.classes:
        lines.append(f"/** {cls.description} */")
        lines.append(f"export interface {cls.name} {{")
        for field in cls.fields:
            optional = "" if field.required else "?"
            lines.append(f"  {field.name}{optional}: {field.ts_type};")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
