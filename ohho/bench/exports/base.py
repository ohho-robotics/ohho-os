"""Shared types for optional result exporters.

Scheme names and vendor clients live in the adapter modules. This module
only parses a spec into a scheme and a target, and carries the job a bench
or optimize run wants to publish.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

# Key names that must not be copied into an external run or a result link.
_SECRET = re.compile(
    r"(^|_)(api[_-]?key|token|secret|password|credential|authorization)$",
    re.IGNORECASE,
)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class ExportError(Exception):
    """One exporter could not finish. The benchmark result is still valid."""


class ExportSpecError(ExportError):
    """The export spec is not ``scheme``, ``scheme:target``, or ``scheme=target``."""


@dataclass(frozen=True)
class ExportJob:
    """One bench or optimize run, with secret-looking fields already removed."""

    kind: str
    spec: Mapping[str, Any]
    manifest: Mapping[str, Any]
    ttl_minutes: float


@dataclass
class ExportContext:
    """Stand-ins a test can pass. A real run leaves every field unset.

    ``module`` and ``loader`` are for an adapter that imports an optional
    library. ``opener`` replaces ``urllib.request.urlopen``. ``sleeper``
    replaces ``time.sleep``. ``now`` returns seconds.
    """

    opener: Any = None
    sleeper: Callable[[float], None] | None = None
    module: Any = None
    loader: Callable[[], Any] | None = None
    now: Callable[[], float] | None = None


def parse_spec(spec: object) -> tuple[str, str]:
    """Split one ``--export`` value into a scheme and a target.

    ``wandb`` and ``wandb:entity/project`` use a colon. ``prometheus=<url>``
    uses an equals sign. The first colon or equals sign is the separator, so
    a URL in the target keeps its own colons.
    """
    if not isinstance(spec, str):
        raise ExportSpecError("export spec must be a string")
    text = spec.strip()
    if not text:
        raise ExportSpecError("export spec is empty")
    cut = len(text)
    for mark in (":", "="):
        found = text.find(mark)
        if found >= 0:
            cut = min(cut, found)
    scheme = text if cut == len(text) else text[:cut]
    target = "" if cut == len(text) else text[cut + 1 :]
    if not _SCHEME.fullmatch(scheme):
        raise ExportSpecError(
            f"export spec {spec!r} is not scheme, scheme:target, or scheme=target"
        )
    return scheme, target.strip()


def public_value(value: Any) -> Any:
    """Copy ``value`` without secret-looking keys or JSON nulls in objects."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET.search(str(key)):
                continue
            child = public_value(item)
            if child is None:
                continue
            cleaned[str(key)] = child
        return cleaned
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    return value


def check_ttl(ttl_minutes: object) -> float:
    """Return a finite, non-negative TTL in minutes.

    Raises:
        ExportError: the value cannot be used as a wait.
    """
    if isinstance(ttl_minutes, bool) or not isinstance(ttl_minutes, (int, float)):
        raise ExportError("export ttl must be a non-negative number of minutes")
    ttl = float(ttl_minutes)
    if not math.isfinite(ttl) or ttl < 0:
        raise ExportError("export ttl must be a non-negative number of minutes")
    return ttl
