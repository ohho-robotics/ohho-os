"""Errors for ``ohho optimize``."""

from __future__ import annotations


class OptimizeError(ValueError):
    """The optimize request is invalid for this model or target profile."""


class OptimizeCapabilityError(OptimizeError):
    """A requested backend is not available in this process.

    TensorRT engine builds, CUDA execution providers, and the weight
    quantizers fail this way on a CPU-only machine. The message names the
    missing piece. Nothing is partially applied.
    """
