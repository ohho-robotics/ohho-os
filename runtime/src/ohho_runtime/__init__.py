"""OhhO onboard runtime (distribution ``ohho-runtime``).

This package is not the executor port in ``ohho.runtime``.
Service manifests live in :mod:`ohho_runtime.manifest`.
The supervisor lives in :mod:`ohho_runtime.supervisor`.

``LoopTimer``, ``LoopStats``, and ``loop_report`` load on first use so a
worker that only imports :mod:`ohho_runtime.heartbeat` does not import
``ohho``.

Structured logs live in :mod:`ohho_runtime.log`. Metrics live in
:mod:`ohho_runtime.metrics`. The operator CLI lives in
:mod:`ohho_runtime.cli` (console script ``ohho-runtime``).
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY = ("LoopStats", "LoopTimer", "loop_report")


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module("ohho_runtime.timing")
    for key in _LAZY:
        globals()[key] = getattr(module, key)
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["LoopStats", "LoopTimer", "__version__", "loop_report"]
