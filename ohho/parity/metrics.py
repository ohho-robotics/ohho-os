"""Numerical comparisons for one pair of output tensors.

The inputs are nested lists of finite numbers. No array library is imported.
Action-head metrics use the last axis as the joint axis and axis 0 as the
batch. Classifier metrics treat the last axis as logits.
"""

from __future__ import annotations

import math
from typing import Any

from ohho.parity.errors import ParityError


def shape_of(value: object) -> tuple[int, ...]:
    """Return the rectangular shape of a nested list. A number has shape ``()``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        raise ParityError(f"output value is {type(value).__name__}, expected a tensor")
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    first = shape_of(value[0])
    for item in value[1:]:
        if shape_of(item) != first:
            raise ParityError("output tensor is ragged")
    return (len(value), *first)


def flatten(value: object) -> list[float]:
    """Row-major finite floats. A non-finite value is an error."""
    if isinstance(value, list):
        flat: list[float] = []
        for item in value:
            flat.extend(flatten(item))
        return flat
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"output value is {type(value).__name__}, expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ParityError("output has a non-finite value")
    return [number]


def mean_abs_diff(reference: object, candidate: object) -> float:
    """Mean of ``|reference - candidate|`` over every element."""
    return _reduce(reference, candidate, "mean")


def max_abs_diff(reference: object, candidate: object) -> float:
    """Largest ``|reference - candidate|``."""
    return _reduce(reference, candidate, "max")


def rmse(reference: object, candidate: object) -> float:
    """Root mean square of the element-wise difference."""
    return _reduce(reference, candidate, "rmse")


def cosine_similarity(reference: object, candidate: object) -> float:
    """Cosine of the two flattened tensors. Two zero tensors score 1."""
    left, right = _pair(reference, candidate)
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lhs, rhs in zip(left, right):
        dot += lhs * rhs
        left_norm += lhs * lhs
        right_norm += rhs * rhs
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = dot / math.sqrt(left_norm * right_norm)
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


def tensor_metrics(reference: object, candidate: object) -> dict[str, float]:
    """Mean abs, max abs, RMSE, and cosine for one output."""
    return {
        "mean_abs": mean_abs_diff(reference, candidate),
        "max_abs": max_abs_diff(reference, candidate),
        "rmse": rmse(reference, candidate),
        "cosine": cosine_similarity(reference, candidate),
    }


def per_joint_error(reference: object, candidate: object) -> list[float]:
    """Mean absolute error of each joint. The last axis is the joint axis."""
    left_cols = _columns(reference)
    right_cols = _columns(candidate)
    if len(left_cols) != len(right_cols):
        raise ParityError("per-joint axes do not match")
    errors: list[float] = []
    for left, right in zip(left_cols, right_cols):
        if len(left) != len(right) or not left:
            raise ParityError("per-joint columns do not match")
        total = 0.0
        for lhs, rhs in zip(left, right):
            total += abs(lhs - rhs)
        errors.append(total / len(left))
    return errors


def action_chunk_error(reference: object, candidate: object) -> float:
    """Mean, over the batch, of the L2 norm of each action chunk.

    Axis 0 is the batch. The remaining axes are one chunk. A vector with no
    batch axis is a single chunk.
    """
    left_shape = shape_of(reference)
    right_shape = shape_of(candidate)
    if left_shape != right_shape:
        raise ParityError(
            f"action chunk shape {left_shape} != candidate shape {right_shape}"
        )
    if not left_shape or 0 in left_shape:
        raise ParityError("action chunk is empty")
    if not isinstance(reference, list) or not isinstance(candidate, list):
        raise ParityError("action chunk is not a tensor")
    if len(left_shape) == 1:
        rows_left = [reference]
        rows_right = [candidate]
    else:
        rows_left = reference
        rows_right = candidate
    total = 0.0
    for left_row, right_row in zip(rows_left, rows_right):
        left = flatten(left_row)
        right = flatten(right_row)
        square = 0.0
        for lhs, rhs in zip(left, right):
            delta = lhs - rhs
            square += delta * delta
        total += math.sqrt(square)
    return total / len(rows_left)


def top1_agreement(reference: object, candidate: object) -> float:
    """Fraction of rows whose last-axis argmax agrees. Ties take the low index."""
    left_rows = _logit_rows(reference)
    right_rows = _logit_rows(candidate)
    if len(left_rows) != len(right_rows) or not left_rows:
        raise ParityError("logit rows do not match")
    same = 0
    for left, right in zip(left_rows, right_rows):
        if len(left) != len(right) or not left:
            raise ParityError("logit rows do not match")
        if _argmax(left) == _argmax(right):
            same += 1
    return same / len(left_rows)


def logits_kl(reference: object, candidate: object) -> float:
    """Mean KL(softmax(reference) || softmax(candidate)), in nats."""
    left_rows = _logit_rows(reference)
    right_rows = _logit_rows(candidate)
    if len(left_rows) != len(right_rows) or not left_rows:
        raise ParityError("logit rows do not match")
    total = 0.0
    for left, right in zip(left_rows, right_rows):
        if len(left) != len(right) or not left:
            raise ParityError("logit rows do not match")
        total += _kl(_softmax(left), _softmax(right))
    return total / len(left_rows)


def _reduce(reference: object, candidate: object, kind: str) -> float:
    left, right = _pair(reference, candidate)
    if kind == "max":
        return max(abs(lhs - rhs) for lhs, rhs in zip(left, right))
    if kind == "mean":
        return sum(abs(lhs - rhs) for lhs, rhs in zip(left, right)) / len(left)
    square = 0.0
    for lhs, rhs in zip(left, right):
        delta = lhs - rhs
        square += delta * delta
    return math.sqrt(square / len(left))


def _pair(reference: object, candidate: object) -> tuple[list[float], list[float]]:
    left_shape = shape_of(reference)
    right_shape = shape_of(candidate)
    if left_shape != right_shape:
        raise ParityError(f"shape {left_shape} != candidate shape {right_shape}")
    if not left_shape or 0 in left_shape:
        raise ParityError("output tensor is empty")
    left = flatten(reference)
    right = flatten(candidate)
    if len(left) != len(right) or not left:
        raise ParityError("output tensors do not match")
    return left, right


def _columns(value: object) -> list[list[float]]:
    shape = shape_of(value)
    if not shape or 0 in shape:
        raise ParityError("output tensor is empty")
    if not isinstance(value, list):
        raise ParityError("output tensor is empty")
    joints = shape[-1]
    columns: list[list[float]] = [[] for _ in range(joints)]

    def walk(node: list[Any], depth: int) -> None:
        if depth == len(shape) - 1:
            for index, item in enumerate(node):
                columns[index].append(float(item))
            return
        for child in node:
            walk(child, depth + 1)

    walk(value, 0)
    return columns


def _logit_rows(value: object) -> list[list[float]]:
    shape = shape_of(value)
    if not shape or 0 in shape:
        raise ParityError("logit tensor is empty")
    if not isinstance(value, list):
        raise ParityError("logit tensor is empty")
    if len(shape) == 1:
        return [[float(item) for item in value]]
    rows: list[list[float]] = []

    def walk(node: list[Any], depth: int) -> None:
        if depth == len(shape) - 1:
            rows.append([float(item) for item in node])
            return
        for child in node:
            walk(child, depth + 1)

    walk(value, 0)
    return rows


def _argmax(row: list[float]) -> int:
    best = 0
    for index, value in enumerate(row):
        if value > row[best]:
            best = index
    return best


def _softmax(row: list[float]) -> list[float]:
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    if total == 0.0:
        raise ParityError("softmax of logits is zero")
    return [item / total for item in exps]


def _kl(reference: list[float], candidate: list[float]) -> float:
    total = 0.0
    for lhs, rhs in zip(reference, candidate):
        if lhs == 0.0:
            continue
        probability = max(lhs, 1e-12)
        other = max(rhs, 1e-12)
        total += probability * math.log(probability / other)
    return total
