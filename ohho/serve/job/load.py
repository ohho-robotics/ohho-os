"""Load an ohho.job/v1 document from YAML or JSON. Stdlib only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import JobSpecError, JobSpecInvalid
from .yaml_subset import YamlParseError, load_yaml_subset


def load_path(path: str | Path) -> Any:
    """Read ``path``. ``.json``, or a body starting with ``{`` or ``[``, is JSON."""
    file = Path(path)
    shown = json.dumps(str(file), ensure_ascii=True)
    if not file.is_file():
        raise JobSpecInvalid(
            [JobSpecError("$", "io", f"cannot read {shown}: not a file")]
        )
    try:
        text = file.read_text(encoding="utf-8-sig")
    except OSError as exc:
        reason = exc.strerror or "read failed"
        raise JobSpecInvalid(
            [JobSpecError("$", "io", f"cannot read {shown}: {reason}")]
        ) from exc
    return parse_text(text, suffix=file.suffix)


def parse_text(text: str, *, suffix: str = "") -> Any:
    """Parse one JSON document or one YAML-subset document."""
    stripped = text.lstrip("\ufeff").lstrip()
    if suffix.lower() == ".json" or stripped.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise JobSpecInvalid(
                [JobSpecError("$", "parse", f"line {exc.lineno}: {exc.msg}")]
            ) from exc
    try:
        return load_yaml_subset(text)
    except YamlParseError as exc:
        raise JobSpecInvalid(
            [JobSpecError("$", "parse", f"line {exc.line}: {exc.message}")]
        ) from exc
