"""Policy action error on a held-out episode set.

Each policy is scored against the episode's ground-truth action. The gate
compares those two errors. Mean absolute error is the mean of
``|prediction - ground truth|`` over every joint and frame. L2 is the mean,
over frames, of the Euclidean norm of that frame's error. The last reported
axis is the joint axis.
"""

from __future__ import annotations

import math

from ohho.parity.errors import ParityError
from ohho.parity.tasks.tolerances import tolerance_for
from ohho.parity.tasks.types import Comparison, TaskContext, as_floats

_DIRECTION = "lower_is_better"


class ActionErrorEvaluator:
    """MAE and L2 of predicted actions against held-out ground truth."""

    name = "action_error"

    def run(self, context: TaskContext) -> list[Comparison]:
        if not context.episodes:
            raise ParityError("action error needs a held-out episode set.")
        reference = _errors(context.reference, context.episodes)
        optimized = _errors(context.optimized, context.episodes)
        if len(reference["per_joint"]) != len(optimized["per_joint"]):
            raise ParityError("reference and optimised actions have different widths.")
        samples = reference["frames"]
        joints = []
        for index, (left, right) in enumerate(
            zip(reference["per_joint"], optimized["per_joint"])
        ):
            joints.append(
                {
                    "index": index,
                    "reference": left,
                    "optimized": right,
                    "delta": right - left,
                }
            )
        seeds = tuple(
            episode.seed for episode in context.episodes if episode.seed is not None
        )
        seed_field = seeds if len(seeds) == len(context.episodes) else None
        return [
            Comparison(
                task=self.name,
                metric="action_mae",
                reference=reference["mae"],
                optimized=optimized["mae"],
                tolerance=tolerance_for(context.tolerances, "action_mae"),
                direction=_DIRECTION,
                unit="action",
                samples=samples,
                per_joint=tuple(joints),
                seeds=seed_field,
            ),
            Comparison(
                task=self.name,
                metric="action_l2",
                reference=reference["l2"],
                optimized=optimized["l2"],
                tolerance=tolerance_for(context.tolerances, "action_l2"),
                direction=_DIRECTION,
                unit="action",
                samples=samples,
                seeds=seed_field,
            ),
        ]


def _errors(policy, episodes) -> dict[str, object]:
    per_joint: list[float] | None = None
    totals: list[float] = []
    abs_sum = 0.0
    abs_count = 0
    l2_sum = 0.0
    frames = 0
    for episode in episodes:
        for frame in episode.frames:
            predicted = as_floats(policy(list(frame.observation)), "predicted action")
            truth = list(frame.action)
            if len(predicted) != len(truth) or not truth:
                raise ParityError(
                    "predicted action width does not match the ground-truth action."
                )
            if per_joint is None:
                per_joint = [0.0] * len(truth)
                totals = [0.0] * len(truth)
            squared = 0.0
            for index, (pred, target) in enumerate(zip(predicted, truth)):
                gap = abs(pred - target)
                totals[index] += gap
                abs_sum += gap
                abs_count += 1
                squared += (pred - target) ** 2
            l2_sum += math.sqrt(squared)
            frames += 1
    if per_joint is None or frames == 0 or abs_count == 0:
        raise ParityError("action error needs at least one frame.")
    return {
        "mae": abs_sum / abs_count,
        "l2": l2_sum / frames,
        "per_joint": [value / frames for value in totals],
        "frames": frames,
    }


evaluator = ActionErrorEvaluator()
