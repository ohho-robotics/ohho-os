"""Load exporter adapters shipped next to this module.

The registry does not name a scheme. Each adapter module exposes
``exporter()``. Import errors propagate so a broken adapter is visible.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any


def discover_exporters() -> list[Any]:
    """Return one instance of each adapter that exposes ``exporter``."""
    import ohho.bench.exports as exports

    found: list[Any] = []
    seen: set[str] = set()
    prefix = exports.__name__
    for info in pkgutil.iter_modules(exports.__path__):
        if info.name in {"base", "metrics", "registry"}:
            continue
        module = importlib.import_module(f"{prefix}.{info.name}")
        factory = getattr(module, "exporter", None)
        if not callable(factory):
            continue
        plugin = factory()
        scheme = getattr(plugin, "scheme", None)
        export = getattr(plugin, "export", None)
        if not isinstance(scheme, str) or not scheme or not callable(export):
            continue
        if scheme in seen:
            continue
        seen.add(scheme)
        found.append(plugin)
    found.sort(key=lambda item: str(item.scheme))
    return found
