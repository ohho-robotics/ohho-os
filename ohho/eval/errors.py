"""Errors raised by the checkpoint eval runner."""

from __future__ import annotations


class EvalError(ValueError):
    """The eval could not start, or a Proof scenario file could not be read."""


class EvalValidationError(ValueError):
    """A report does not match the ohho.eval/v1 schema."""
