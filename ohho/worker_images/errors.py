"""Errors for pinned serve worker images."""

from __future__ import annotations


class WorkerImageError(ValueError):
    """A worker image tag, stamp, matrix row, scan, or job result is invalid."""
