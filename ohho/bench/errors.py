"""Errors raised by the Serve bench harness."""

from __future__ import annotations


class BenchHarnessError(Exception):
    """The bench cannot start, or cannot finish a report.

    ``exit_code`` is 2 for usage, an unknown target, an unsupported input,
    or a missing optional dependency. Gate failures are not this exception:
    they are stored on the result and the command returns 1.
    """

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code
