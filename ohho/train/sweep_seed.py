"""Deterministic seeds derived from an ``ohho.sweep/v1`` sweep seed.

The sweep document has one seed. Two integers come from it:

* the sampler seed, which is the sweep seed and initializes the proposal generator
* one training seed per trial index, mixed so it does not consume that generator

The training seed does not draw from the proposal generator, so recording it
or resuming a sweep does not change later proposals. The same sweep seed and
trial index always produce the same training seed.

``ohho.run/v1`` run manifests are a separate contract. :func:`derived_trial_seed`
reads the training seed already stored on a sweep trial so a manifest can
copy it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_SEED_VERSION = "ohho.sweep.seed/v1"
_SEED_MODULUS = 2**32


def sampler_seed(sweep_seed: int) -> int:
    """Proposal-generator seed derived from ``sweep_seed``.

    The derived value is the sweep seed. ``random``, ``tpe``, ``bayesian``,
    and ``multi`` draw from ``random.Random(sampler_seed(sweep_seed))``,
    which is ``random.Random(sweep_seed)``. ``grid`` order does not use it.
    Trial training seeds are a separate mix and do not advance this generator.
    """

    return _check_sweep_seed(sweep_seed)


def trial_training_seed(sweep_seed: int, trial_index: int) -> int:
    """Training seed for ``trial_index``, derived from ``sweep_seed``.

    The result is in ``0 .. 2**32 - 1``. It depends only on the sweep seed
    and the trial index.
    """

    return _mix(_check_sweep_seed(sweep_seed), "trial", _check_index(trial_index))


def derived_trial_seed(trial: Mapping[str, Any]) -> int:
    """Return the training seed stored on a sweep trial.

    ``trial`` is a trial-table row (or any mapping with an integer ``seed``).
    This does not define an ``ohho.run/v1`` manifest. A later manifest writer
    copies this integer onto the run record.
    """

    if not isinstance(trial, Mapping):
        _fail("trial seed record must be a mapping")
    seed = trial.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= _SEED_MODULUS
    ):
        _fail("trial seed must be an integer in [0, 2**32)")
    return seed


def _mix(sweep_seed: int, domain: str, trial_index: int | None = None) -> int:
    parts: list[Any] = [_SEED_VERSION, sweep_seed, domain]
    if trial_index is not None:
        parts.append(trial_index)
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _check_sweep_seed(sweep_seed: int) -> int:
    if (
        isinstance(sweep_seed, bool)
        or not isinstance(sweep_seed, int)
        or sweep_seed < 0
    ):
        _fail("sweep seed must be an integer >= 0")
    return sweep_seed


def _check_index(trial_index: int) -> int:
    if (
        isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or trial_index < 0
    ):
        _fail("trial index must be an integer >= 0")
    return trial_index


def _fail(message: str) -> None:
    from .sweep import SweepError

    raise SweepError(message)


__all__ = [
    "sampler_seed",
    "trial_training_seed",
    "derived_trial_seed",
]
