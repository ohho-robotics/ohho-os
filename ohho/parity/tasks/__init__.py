"""Task-level accuracy on top of the tensor parity gate.

Evaluators are entry points in the ``ohho.parity.tasks`` group. The built-ins
score action error, closed-loop sim success, detection mAP, and token match.
The result is an ``ohho.bench/v1`` document.
"""

from __future__ import annotations

from ohho.parity.tasks.stage import evaluate_tasks, run_task_stage

__all__ = ["evaluate_tasks", "run_task_stage"]
