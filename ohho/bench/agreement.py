"""Top-1 action-token agreement.

``ohho.parity`` is used when that module is installed and exposes one of
the callables below. A checkout without it uses the local fraction of
matching positions. Extra tokens on the longer sequence count as misses.
"""

from __future__ import annotations

from collections.abc import Sequence

_PARITY_NAMES = (
    "top1_action_agreement",
    "action_token_agreement",
    "top1_agreement",
)


def top1_action_agreement(
    reference: Sequence[int],
    candidate: Sequence[int],
) -> float:
    """Return the fraction of positions where ``candidate`` matches ``reference``."""
    fn = _parity_function()
    if fn is not None:
        return float(fn(list(reference), list(candidate)))
    return local_top1_agreement(reference, candidate)


def local_top1_agreement(
    reference: Sequence[int],
    candidate: Sequence[int],
) -> float:
    """Fraction of equal positions. The longer length is the denominator."""
    width = max(len(reference), len(candidate))
    if width == 0:
        return 1.0
    matches = sum(1 for left, right in zip(reference, candidate) if left == right)
    return matches / width


def _parity_function() -> object | None:
    try:
        import ohho.parity as parity
    except ImportError:
        return None
    for name in _PARITY_NAMES:
        fn = getattr(parity, name, None)
        if callable(fn):
            return fn
    return None
