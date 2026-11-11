"""Errors for ``ohho parity``."""

from __future__ import annotations

from typing import Any


class ParityError(ValueError):
    """The parity request cannot be scored.

    A missing file, a shape mismatch, or a threshold file that does not parse
    raises this. The CLI exits 2. This is separate from a scored fail.
    """


class ParityFailed(ParityError):
    """The candidate is outside the configured thresholds.

    ``report`` is the ``ohho.parity/v1`` document. ``ohho optimize`` must not
    publish the artifact unless ``--allow-parity-fail`` was set.
    """

    def __init__(self, report: dict[str, Any]) -> None:
        reasons = report.get("reasons") if isinstance(report, dict) else None
        if isinstance(reasons, list) and reasons:
            text = "; ".join(str(item) for item in reasons)
        else:
            text = "parity gate failed"
        super().__init__(text)
        self.report = report


class TaskFailed(ParityError):
    """The task-accuracy gate missed a tolerance.

    ``document`` is the ``ohho.bench/v1`` result. ``ohho optimize`` does not
    publish the artifact unless ``--allow-parity-fail`` was set. This is
    separate from a tensor-parity miss (:class:`ParityFailed`).
    """

    def __init__(self, document: dict[str, Any]) -> None:
        accuracy = document.get("accuracy") if isinstance(document, dict) else None
        reasons = accuracy.get("reasons") if isinstance(accuracy, dict) else None
        if isinstance(reasons, list) and reasons:
            text = "; ".join(str(item) for item in reasons)
        else:
            text = "task accuracy gate failed"
        super().__init__(text)
        self.document = document
