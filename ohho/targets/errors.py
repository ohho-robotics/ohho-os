"""Errors for hardware target profiles."""

from __future__ import annotations


class TargetProfileError(ValueError):
    """A target profile is missing, malformed, or not in the catalog."""


class TargetCombinationError(TargetProfileError):
    """A precision and execution provider are not valid on this target."""
