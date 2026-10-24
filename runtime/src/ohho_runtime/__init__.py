"""OhhO onboard runtime (distribution ``ohho-runtime``).

This package is not the executor port in ``ohho.runtime``.
Service manifests live in :mod:`ohho_runtime.manifest`.
The supervisor lives in :mod:`ohho_runtime.supervisor`.

``LoopTimer``, ``LoopStats``, and ``loop_report`` load on first use so a
worker that only imports :mod:`ohho_runtime.heartbeat` does not import
``ohho``.

Structured logs live in :mod:`ohho_runtime.log`. Metrics live in
:mod:`ohho_runtime.metrics`. The operator CLI lives in
:mod:`ohho_runtime.cli` (console script ``ohho-runtime``). Crash bundles
live in :mod:`ohho_runtime.crash`.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY = {
    "LoopStats": "ohho_runtime.timing",
    "LoopTimer": "ohho_runtime.timing",
    "loop_report": "ohho_runtime.timing",
    "CrashBundle": "ohho_runtime.crash",
    "write_crash_bundle": "ohho_runtime.crash",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CrashBundle",
    "LoopStats",
    "LoopTimer",
    "__version__",
    "loop_report",
    "write_crash_bundle",
]
