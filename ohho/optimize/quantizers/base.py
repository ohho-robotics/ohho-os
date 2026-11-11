"""Quantizer plugin contract.

A backend implements :meth:`supports`, :meth:`quantize`, and
:meth:`requirements`. ``quantize`` receives the loaded model and a
calibration job and returns one artifact. New backends register on the
``ohho.quantizers`` entry-point group.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantArtifact:
    """One file or directory written by a quantizer."""

    path: Path
    backend: str
    version: str
    role: str


@dataclass
class QuantCalibration:
    """Calibration payload and the directory the artifact is written into.

    ``examples`` is the GPTQ text or tensor list. ``data`` is a calibration
    directory. ``trt_module`` is a TensorRT stand-in for tests.
    """

    out_dir: Path
    precision: str
    data: str | None = None
    examples: Any = None
    batch: int | None = None
    cache: str | None = None
    workspace_bytes: int = 1
    max_batch: int = 1
    dynamic_batch: bool = False
    trt_module: Any = None


class Quantizer:
    """One quantization backend.

    ``requirements`` lists import names. The backend can run when any one of
    them imports. ``supports`` is the target-profile check and may be called
    with ``model=None`` before a checkpoint is loaded. ``quantize`` does the
    work and returns the artifact.
    """

    name = ""
    research_only = False

    def requirements(self) -> tuple[str, ...]:
        """Import names. Any one of them is enough."""
        return ()

    def supports(self, target, model) -> tuple[bool, str]:
        """Return whether this backend applies, and a short reason."""
        raise NotImplementedError

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        """Write one artifact for ``model`` using ``calib``."""
        raise NotImplementedError

    def version(self) -> str:
        """Installed distribution version, or ``unknown``."""
        names = self.requirements()
        module = names[0] if names else None
        return distribution_version(*names, module=module)


def distribution_version(*dists: str, module: str | None = None) -> str:
    """First installed distribution version, else the loaded module's ``__version__``."""
    for dist in dists:
        if not dist:
            continue
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    if module:
        loaded = sys.modules.get(module)
        if loaded is not None:
            text = getattr(loaded, "__version__", "")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return "unknown"
