"""Numerical parity between a reference model and an optimized artifact.

Import this package by path (``import ohho.parity``). PyTorch and ONNX
Runtime are imported only when a comparison needs them.

The report schema is ``ohho.parity/v1``.
"""

from __future__ import annotations

from ohho.parity.capture import capture_parity_set
from ohho.parity.errors import ParityError, ParityFailed, TaskFailed
from ohho.parity.gate import evaluate
from ohho.parity.registry import ParitySetPublisher, attach_parity_set
from ohho.parity.report import SCHEMA_ID, schema_text
from ohho.parity.tasks import evaluate_tasks
from ohho.parity.thresholds import DEFAULT_MEAN_ABS

__all__ = [
    "DEFAULT_MEAN_ABS",
    "SCHEMA_ID",
    "ParityError",
    "ParityFailed",
    "ParitySetPublisher",
    "TaskFailed",
    "attach_parity_set",
    "capture_parity_set",
    "evaluate",
    "evaluate_tasks",
    "schema_text",
]
