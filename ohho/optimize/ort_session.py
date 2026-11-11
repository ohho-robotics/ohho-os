"""ONNX Runtime sessions for a target profile's execution provider.

The caller has already checked the provider against the profile. This module
refuses a provider that is not installed, and refuses a session that silently
falls back to a different provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ohho.targets.providers import installed_execution_providers

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError

_INSTALL = (
    "onnxruntime is not installed. CPU and CUDA sessions need the optimize extra: "
    "pip install 'ohho-os[optimize]'."
)


def build_session(
    onnx_path: str | Path,
    *,
    provider: str,
    available: Sequence[str] | None = None,
    ort_module=None,
):
    """Open a session that uses ``provider`` and no other provider."""
    path = Path(onnx_path)
    if not path.is_file():
        raise OptimizeError(f"ONNX file does not exist: {path}")
    found = available if available is not None else installed_execution_providers()
    if found is None:
        raise OptimizeCapabilityError(_INSTALL)
    if provider not in found:
        listed = ", ".join(found) if found else "(none)"
        raise OptimizeCapabilityError(
            f"{provider} is not installed. Installed execution providers: {listed}."
        )
    ort = ort_module if ort_module is not None else _import_ort()
    session = ort.InferenceSession(str(path), providers=[provider])
    active = list(session.get_providers())
    if provider not in active:
        raise OptimizeCapabilityError(
            f"ONNX Runtime did not enable {provider}. "
            f"Active providers: {', '.join(active) or '(none)'}."
        )
    return session


def _import_ort():
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise OptimizeCapabilityError(_INSTALL) from exc
    return ort
