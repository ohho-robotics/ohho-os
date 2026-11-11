"""Validation bookkeeping shared by the imitation-learning trainers.

``step`` counts completed optimizer updates and starts at 1. A point is due on
every multiple of ``val_every`` and on the final requested step, so a short run
still reports a validation loss when ``val_every`` is larger than the step budget.
"""

from __future__ import annotations


def validation_due(step: int, train_steps: int, val_every: int) -> bool:
    """Return whether a validation event should be emitted after ``step``."""
    if val_every <= 0 or step <= 0:
        return False
    return step % val_every == 0 or step == train_steps


class ValTracker:
    """Patience counter over validation loss.

    ``patience`` of 0 never stops. An observation improves the best value only
    when it is lower by more than ``min_delta``.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.stale = 0

    def observe(self, loss: float) -> bool:
        """Record ``loss``. Return True when training should stop."""
        if self.best is None or loss < self.best - self.min_delta:
            self.best = loss
            self.stale = 0
            return False
        self.stale += 1
        return self.patience > 0 and self.stale >= self.patience
