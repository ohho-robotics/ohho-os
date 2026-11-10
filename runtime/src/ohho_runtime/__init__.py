"""OhhO onboard runtime (distribution ``ohho-runtime``).

This package is not the executor port in ``ohho.runtime``.
Service manifests live in :mod:`ohho_runtime.manifest`.
The supervisor lives in :mod:`ohho_runtime.supervisor`.

``LoopTimer``, ``LoopStats``, and ``loop_report`` load on first use so a
worker that only imports :mod:`ohho_runtime.heartbeat` does not import
``ohho``. ``Deadline``, ``Heartbeat``, and ``wire_contract`` load the same
way. The wire contract is :mod:`ohho_runtime.deadline`. The socket lines
stay in :mod:`ohho_runtime.heartbeat`.

Structured logs live in :mod:`ohho_runtime.log`. Metrics live in
:mod:`ohho_runtime.metrics`. The operator CLI lives in
:mod:`ohho_runtime.cli` (console script ``ohho-runtime``). Crash bundles
live in :mod:`ohho_runtime.crash`. Safety-preserving degradation lives in
:mod:`ohho_runtime.degradation`.
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
    "Deadline": "ohho_runtime.deadline",
    "Heartbeat": "ohho_runtime.deadline",
    "HeartbeatContract": "ohho_runtime.deadline",
    "wire_contract": "ohho_runtime.deadline",
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
    "Deadline",
    "Heartbeat",
    "HeartbeatContract",
    "LoopStats",
    "LoopTimer",
    "__version__",
    "loop_report",
    "wire_contract",
    "write_crash_bundle",
]
