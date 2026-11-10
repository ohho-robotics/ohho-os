"""ASHA and median early stopping for ``ohho.sweep/v1``.

A trial runner calls :meth:`EarlyStopping.report` with the intermediate
metric at a step. The sweep controller calls :meth:`EarlyStopping.decide`,
which returns ``continue`` or ``stop``. This module does not launch
training jobs. The controller that fans trials out across jobs is a
separate piece of work and is not in this package.

``rule`` ``asha`` is successive halving at ``rungs``. Once at least
``reduction_factor`` trials have reported a rung, the trials that are
strictly worse than the best ``floor(n / reduction_factor)`` are stopped.
A decision at a rung is frozen: a later peer does not revoke a continue.
Reporting every active trial at the rung before calling ``decide`` is the
synchronous bracket. Calling ``decide`` as each report arrives is the
asynchronous rule.

``rule`` ``median`` stops a trial that is strictly worse than the median
of the reports at that step, after ``grace_period``, and only once
``reduction_factor`` trials have reported the step. ``rule`` ``patience``
stops one trial after ``patience`` decision steps with no improvement of
``min_delta``.

A stop sets the trial status to ``pruned``. That is distinct from
``failed``. The last checkpoint path and the intermediate metrics stay on
the row. :class:`ohho.train.sweep_run.SweepRun` writes those fields into
the trial table and keeps them across resume.

:func:`simulate_asha_budget` runs the documented 32-trial synthetic
benchmark. The curves are :func:`synthetic_learning_curve`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sweep import SweepError, normalize_early_stopping

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_PRUNED = "pruned"
STATUS_FAILED = "failed"
ACTION_CONTINUE = "continue"
ACTION_STOP = "stop"
_TERMINAL = frozenset({STATUS_PRUNED, STATUS_FAILED, STATUS_COMPLETED})
_BENCHMARK_RUNGS = (1, 3, 9, 27, 81)


@dataclass(frozen=True)
class StopDecision:
    """Whether one trial should keep training.

    ``action`` is ``continue`` or ``stop``.
    """

    action: str
    trial: int
    step: int | None
    rule: str
    reason: str


def synthetic_learning_curve(trial: int, step: int) -> float:
    """Loss for the documented ASHA benchmark.

    The value is ``(trial + 1) / step``. It falls as ``step`` grows. A
    smaller trial index is better at every step, and two trials never tie.
    """

    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 0:
        raise SweepError("trial must be an integer >= 0")
    _require_step(step)
    return (trial + 1) / step


def simulate_asha_budget(
    *,
    trials: int = 32,
    rungs: Sequence[int] = _BENCHMARK_RUNGS,
    reduction_factor: int = 3,
    grace_period: int = 1,
) -> dict[str, Any]:
    """Run ASHA on synthetic learning curves and count trial-steps.

    Each trial stops at the rung where :meth:`EarlyStopping.decide` returns
    ``stop``, or reaches the last rung. ``steps_with_asha`` is the sum of
    those rungs. ``steps_without_asha`` is ``trials * max(rungs)``, the same
    sweep with no early stopping. The default arguments are the documented
    32-trial benchmark (rungs 1, 3, 9, 27, 81 and reduction factor 3).

    The loop reports every active trial at a rung before it decides that
    rung. A stop keeps the checkpoint path and the metrics recorded so far
    and marks the row ``pruned``.
    """

    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise SweepError("benchmark trials must be an integer >= 1")
    config = normalize_early_stopping(
        {
            "metric": "val_loss",
            "mode": "min",
            "rule": "asha",
            "rungs": list(rungs),
            "reduction_factor": reduction_factor,
            "grace_period": grace_period,
        }
    )
    stopper = EarlyStopping(config)
    rows = [_open_row(index) for index in range(trials)]
    active = list(range(trials))
    reached = dict.fromkeys(active, 0)
    rung_list = list(config["rungs"])
    for rung in rung_list:
        for index in active:
            stopper.report(
                rows,
                index,
                rung,
                synthetic_learning_curve(index, rung),
                checkpoint=f"checkpoints/{index}/{rung}",
            )
            reached[index] = rung
        still: list[int] = []
        for index in active:
            decision = stopper.decide(rows, index)
            if decision.action == ACTION_CONTINUE:
                still.append(index)
        active = still
        if not active:
            break
    pruned_rows = [row for row in rows if row["status"] == STATUS_PRUNED]
    finished = [row["index"] for row in rows if row["status"] == STATUS_RUNNING]
    steps_with = sum(reached.values())
    steps_without = trials * rung_list[-1]
    kept = all(row["checkpoint"] and row["metrics"] for row in pruned_rows)
    return {
        "trials": trials,
        "rungs": rung_list,
        "reduction_factor": config["reduction_factor"],
        "steps_with_asha": steps_with,
        "steps_without_asha": steps_without,
        "ratio": steps_with / steps_without,
        "pruned": len(pruned_rows),
        "finished": len(finished),
        "finished_trials": finished,
        "pruned_keep_checkpoint_and_metrics": kept,
        "failed": sum(row["status"] == STATUS_FAILED for row in rows),
    }


class EarlyStopping:
    """Decide continue or stop from intermediate metrics.

    Pass the trial rows the sweep run stores. :meth:`report` and
    :meth:`decide` update those rows in place. Job fan-out is not done here.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = normalize_early_stopping(config)

    def report(
        self,
        rows: Sequence[MutableRow],
        index: int,
        step: int,
        value: int | float,
        *,
        checkpoint: str | None = None,
    ) -> None:
        """Record ``value`` for ``index`` at ``step``.

        ``checkpoint`` replaces the trial's last checkpoint path. ``None``
        leaves the previous path in place. A later stop keeps that path and
        every metric recorded so far.
        """

        row = _find_row(rows, index)
        _ensure_early(row)
        step = _require_step(step)
        stored = _metric_value(value)
        if checkpoint is not None:
            _require_checkpoint(checkpoint)
        if row["status"] in _TERMINAL:
            raise SweepError(f"trial {index} is {row['status']}")
        metrics: list[dict[str, Any]] = row["metrics"]
        if metrics and step <= int(metrics[-1]["step"]):
            raise SweepError(
                f"trial {index} step {step} is not later than {metrics[-1]['step']}"
            )
        metrics.append({"step": step, "value": stored})
        if checkpoint is not None:
            row["checkpoint"] = checkpoint
        row["decision"] = None
        if row["status"] == STATUS_PENDING:
            row["status"] = STATUS_RUNNING

    def decide(self, rows: Sequence[MutableRow], index: int) -> StopDecision:
        """Return ``continue`` or ``stop`` for ``index`` from its latest report.

        The first decision at a step is kept. A trial that was told to
        continue is not stopped at that same step when more peers report.
        ``stop`` sets status to ``pruned`` and leaves the checkpoint and
        metrics in place.
        """

        row = _find_row(rows, index)
        _ensure_early(row)
        rule = str(self.config["rule"])
        status = row["status"]
        if status == STATUS_PRUNED:
            return _decision(
                index, _last_step(row), rule, ACTION_STOP, "trial is pruned"
            )
        if status == STATUS_FAILED:
            return _decision(index, _last_step(row), rule, ACTION_STOP, "trial failed")
        if status == STATUS_COMPLETED:
            return _decision(
                index, _last_step(row), rule, ACTION_STOP, "trial completed"
            )
        metrics: list[dict[str, Any]] = row["metrics"]
        if not metrics:
            return _decision(
                index, None, rule, ACTION_CONTINUE, "no intermediate metric yet"
            )
        step = int(metrics[-1]["step"])
        frozen = row["decision"]
        if (
            isinstance(frozen, dict)
            and frozen.get("step") == step
            and frozen.get("action") in (ACTION_CONTINUE, ACTION_STOP)
        ):
            action = str(frozen["action"])
            if action == ACTION_STOP and row["status"] not in _TERMINAL:
                row["status"] = STATUS_PRUNED
            return _decision(index, step, rule, action, "recorded decision")
        gate = _gate(self.config, step)
        if gate is not None:
            action, reason = ACTION_CONTINUE, gate
        elif rule == "patience":
            action, reason = _patience(row, self.config)
        elif rule == "median":
            action, reason = _median(rows, index, step, self.config)
        elif rule == "asha":
            action, reason = _asha(rows, index, step, self.config)
        else:
            raise SweepError(f"unknown early_stopping rule {rule!r}")
        row["decision"] = {"step": step, "action": action}
        if action == ACTION_STOP:
            row["status"] = STATUS_PRUNED
        return _decision(index, step, rule, action, reason)

    def mark_failed(self, rows: Sequence[MutableRow], index: int, error: str) -> None:
        """Mark ``index`` failed. The checkpoint and metrics stay.

        ``failed`` is not ``pruned``. A pruned, completed, or already failed
        trial is left unchanged and this raises.
        """

        row = _find_row(rows, index)
        _ensure_early(row)
        if not isinstance(error, str) or error.strip() == "":
            raise SweepError("failed trial requires an error string")
        if row["status"] in _TERMINAL:
            raise SweepError(f"trial {index} is {row['status']}")
        row["status"] = STATUS_FAILED
        row["error"] = error


MutableRow = dict[str, Any]


def _open_row(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "status": STATUS_PENDING,
        "checkpoint": None,
        "metrics": [],
        "decision": None,
        "error": None,
    }


def _decision(
    index: int, step: int | None, rule: str, action: str, reason: str
) -> StopDecision:
    return StopDecision(action=action, trial=index, step=step, rule=rule, reason=reason)


def _find_row(rows: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise SweepError("trial index must be an integer")
    for row in rows:
        if row.get("index") == index:
            if not isinstance(row, dict):
                raise SweepError(f"trial {index} must be an object")
            return row
    raise SweepError(f"trial {index} has not been proposed")


def _ensure_early(row: dict[str, Any]) -> None:
    row.setdefault("status", STATUS_PENDING)
    row.setdefault("checkpoint", None)
    row.setdefault("metrics", [])
    row.setdefault("decision", None)
    row.setdefault("error", None)
    if not isinstance(row["metrics"], list):
        raise SweepError(f"trial {row.get('index')} metrics must be an array")


def _require_step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise SweepError("step must be an integer >= 1")
    return step


def _require_checkpoint(checkpoint: str) -> None:
    if not isinstance(checkpoint, str) or checkpoint.strip() == "":
        raise SweepError("checkpoint must be a non-empty string")


def _metric_value(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError("intermediate metric must be a finite number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SweepError("intermediate metric must be a finite number")
        return float(value)
    return int(value)


def _last_step(row: Mapping[str, Any]) -> int | None:
    metrics = row.get("metrics") or []
    if not metrics:
        return None
    step = metrics[-1].get("step")
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    return None


def _gate(config: Mapping[str, Any], step: int) -> str | None:
    if step < int(config["grace_period"]):
        return "before grace period"
    rungs = list(config["rungs"])
    if rungs and step not in rungs:
        return "step is not a rung"
    if config["rule"] == "asha" and not rungs:
        return "step is not a rung"
    return None


def _patience(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, str]:
    best: float | None = None
    wait = 0
    for point in row["metrics"]:
        step = int(point["step"])
        if _gate(config, step) is not None:
            continue
        value = float(point["value"])
        if best is None or _improved(
            value, best, float(config["min_delta"]), str(config["mode"])
        ):
            best = value
            wait = 0
        else:
            wait += 1
    if wait >= int(config["patience"]):
        return ACTION_STOP, "patience exhausted"
    return ACTION_CONTINUE, "within patience"


def _improved(value: float, best: float, min_delta: float, mode: str) -> bool:
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


def _asha(
    rows: Sequence[Mapping[str, Any]],
    index: int,
    step: int,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    cohort = _cohort(rows, step)
    promote = len(cohort) // int(config["reduction_factor"])
    if promote < 1:
        return ACTION_CONTINUE, "not enough trials at this step"
    mode = str(config["mode"])
    ordered = sorted(cohort, key=lambda item: (_rank(item[1], mode), item[0]))
    cutoff = ordered[promote - 1][1]
    value = _cohort_value(cohort, index)
    if _worse(value, cutoff, mode):
        return ACTION_STOP, "outside the promotion cutoff"
    return ACTION_CONTINUE, "inside the promotion cutoff"


def _median(
    rows: Sequence[Mapping[str, Any]],
    index: int,
    step: int,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    cohort = _cohort(rows, step)
    if len(cohort) < int(config["reduction_factor"]):
        return ACTION_CONTINUE, "not enough trials at this step"
    midpoint = statistics.median([value for _, value in cohort])
    value = _cohort_value(cohort, index)
    if _worse(value, float(midpoint), str(config["mode"])):
        return ACTION_STOP, "worse than the median"
    return ACTION_CONTINUE, "at or better than the median"


def _cohort(rows: Sequence[Mapping[str, Any]], step: int) -> list[tuple[int, float]]:
    found: list[tuple[int, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        for point in row.get("metrics") or []:
            if not isinstance(point, Mapping):
                continue
            if point.get("step") == step:
                found.append((index, float(point["value"])))
                break
    return found


def _cohort_value(cohort: Sequence[tuple[int, float]], index: int) -> float:
    for trial, value in cohort:
        if trial == index:
            return value
    raise SweepError(f"trial {index} has no metric at this step")


def _rank(value: float, mode: str) -> float:
    if mode == "max":
        return -value
    return value


def _worse(value: float, cutoff: float, mode: str) -> bool:
    if mode == "max":
        return value < cutoff
    return value > cutoff


__all__ = [
    "ACTION_CONTINUE",
    "ACTION_STOP",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_PRUNED",
    "STATUS_RUNNING",
    "EarlyStopping",
    "StopDecision",
    "simulate_asha_budget",
    "synthetic_learning_curve",
]
