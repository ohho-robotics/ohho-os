"""Top-1 and token-match rate for an LLM or VLM.

Each example is a dict with ``label`` and ``tokens``. The policy receives
that dict and returns one of:

- ``{"logits": [...], "tokens": [...]}``
- ``{"top1": 4, "tokens": [4, 5]}``
- a list of token ids (top-1 is the first id)

Top-1 is the fraction of examples whose predicted class equals ``label``.
Logit ties keep the lower index. Token match is the fraction of positions
that equal the example's ``tokens``. A length mismatch counts the extra or
missing positions as errors.
"""

from __future__ import annotations

from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.tasks.tolerances import tolerance_for
from ohho.parity.tasks.types import Comparison, TaskContext


class TokenMatchEvaluator:
    """Top-1 accuracy and token match rate against labelled examples."""

    name = "token_match"

    def run(self, context: TaskContext) -> list[Comparison]:
        examples = context.tokens
        if not examples:
            raise ParityError(
                "token match needs labelled examples (--task-tokens or tokens=)."
            )
        reference = _score(examples, context.reference)
        optimized = _score(examples, context.optimized)
        count = len(examples)
        return [
            Comparison(
                task=self.name,
                metric="top1",
                reference=reference["top1"],
                optimized=optimized["top1"],
                tolerance=tolerance_for(context.tolerances, "top1"),
                direction="higher_is_better",
                unit="fraction",
                samples=count,
            ),
            Comparison(
                task=self.name,
                metric="token_match",
                reference=reference["token_match"],
                optimized=optimized["token_match"],
                tolerance=tolerance_for(context.tolerances, "token_match"),
                direction="higher_is_better",
                unit="fraction",
                samples=count,
            ),
        ]


def _score(examples: tuple[dict[str, Any], ...], policy) -> dict[str, float]:
    correct = 0
    matched = 0
    total = 0
    for example in examples:
        if "label" not in example:
            raise ParityError("a token example needs a label.")
        label = _as_int(example["label"], "label")
        truth = example.get("tokens")
        if truth is None:
            expected = [label]
        else:
            if not isinstance(truth, list) or not truth:
                raise ParityError("tokens must be a non-empty list of ids.")
            expected = [_as_int(item, "token") for item in truth]
        top1, predicted = _parse(policy(dict(example)))
        correct += int(top1 == label)
        width = max(len(predicted), len(expected))
        total += width
        matched += sum(1 for left, right in zip(predicted, expected) if left == right)
    if total == 0:
        raise ParityError("token match needs at least one token.")
    return {"top1": correct / len(examples), "token_match": matched / total}


def _parse(raw: object) -> tuple[int, list[int]]:
    if isinstance(raw, list):
        tokens = [_as_int(item, "token") for item in raw]
        if not tokens:
            raise ParityError("token policy returned an empty list.")
        return tokens[0], tokens
    if not isinstance(raw, dict):
        raise ParityError("token policy must return a dict or a list of ids.")
    tokens_raw = raw.get("tokens")
    tokens = (
        [_as_int(item, "token") for item in tokens_raw]
        if isinstance(tokens_raw, list)
        else None
    )
    if "logits" in raw:
        logits = raw["logits"]
        if not isinstance(logits, list) or not logits:
            raise ParityError("logits must be a non-empty list.")
        top1 = _argmax(logits)
    elif "top1" in raw:
        top1 = _as_int(raw["top1"], "top1")
    elif tokens:
        top1 = tokens[0]
    else:
        raise ParityError("token policy returned no logits, top1, or tokens.")
    return top1, tokens if tokens is not None else [top1]


def _argmax(logits: list[object]) -> int:
    best_index = 0
    best_value: float | None = None
    for index, item in enumerate(logits):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ParityError("logits must be numbers.")
        value = float(item)
        if best_value is None or value > best_value:
            best_value = value
            best_index = index
    return best_index


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParityError(f"{label} must be an integer.")
    return value


evaluator = TokenMatchEvaluator()
