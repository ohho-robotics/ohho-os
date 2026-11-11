"""Sequential samplers for ``ohho.sweep/v1``.

``grid`` walks the expanded grid. ``random`` draws from ``seed``. ``tpe`` and
``bayesian`` are one built-in Tree-structured Parzen Estimator (Bergstra et
al.): after ``sampler.startup_trials`` random draws, candidates are drawn
from a Gaussian Parzen density on the better trials. The next trial is
sampled with probability proportional to the ratio of that density to the
density of the rest, so the step stays in the high-ratio region instead of
collapsing onto one previous trial. ``multi`` uses the same estimator with
a Pareto split and stores the non-dominated set as
``ohho.sweep.leaderboard/v1``.

The estimator is standard-library only. It does not import Optuna, NumPy,
or PyTorch. :meth:`SweepSampler.dump_state` is JSON. Loading that state and
asking again proposes the same next trials as the sampler that kept running.
Each proposed trial records the training seed derived from the sweep seed
and the trial index. That seed does not advance the proposal generator.

A batch returned by one :meth:`SweepSampler.ask` shares the model built from
trials :meth:`SweepSampler.tell` has already recorded.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sweep import (
    ADAPTIVE_STRATEGIES,
    Dimension,
    SweepError,
    SweepSpec,
    Trial,
    _allowed,
    _apply,
    _assert_refs,
    _expand_grid,
    _float_linspace,
    _format_set,
    _int_linspace,
    _json_equal,
    _log_linspace,
    _matches,
    _read_resource,
    _sample_one,
    _trial_hours,
    _validate,
    load_sweep,
    validate_sweep,
)
from .sweep_seed import sampler_seed, trial_training_seed

LEADERBOARD_SCHEMA_ID = "ohho.sweep.leaderboard/v1"
SAMPLER_SCHEMA_ID = "ohho.sweep.sampler/v1"
LEADERBOARD_FILENAME = "sweep.leaderboard.v1.json"
SAMPLER_FILENAME = "sweep.sampler.v1.json"
BUDGET_EXHAUSTED = "sweep budget is exhausted"
_DISCRETE_LIMIT = 24


@dataclass
class _DimModel:
    dim: Dimension
    discrete: list[Any] | None
    good_pairs: list[tuple[Any, float]]
    bad_pairs: list[tuple[Any, float]]
    good_comps: list[tuple[float, float, float]] | None
    bad_comps: list[tuple[float, float, float]] | None


def pareto_indices(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Indexes of non-dominated rows, ascending.

    Row A dominates row B when A is no worse on every objective and strictly
    better on at least one. ``maximize`` prefers a larger number.
    ``minimize`` prefers a smaller number. Equal values do not dominate.
    With no objectives, every row is on the front.
    """

    if not rows:
        return []
    ranks = _pareto_ranks(rows, objectives)
    return [index for index, rank in enumerate(ranks) if rank == 0]


def load_leaderboard_schema() -> dict[str, Any]:
    """Return a copy of the ``ohho.sweep.leaderboard/v1`` JSON Schema."""

    return copy.deepcopy(_schema(LEADERBOARD_FILENAME))


def load_sampler_schema() -> dict[str, Any]:
    """Return a copy of the ``ohho.sweep.sampler/v1`` JSON Schema."""

    return copy.deepcopy(_schema(SAMPLER_FILENAME))


class SweepSampler:
    """Propose trials for one validated sweep and record their objectives."""

    def __init__(self, spec: SweepSpec) -> None:
        if not isinstance(spec, SweepSpec):
            raise TypeError(
                "SweepSampler expects a SweepSpec; use SweepSampler.from_sweep"
            )
        self.spec = spec
        self._rng = random.Random(sampler_seed(spec.seed))
        self._cursor = 0
        self._hours = 0.0
        self._gpu_stop = False
        self._trials: list[dict[str, Any]] = []
        self._grid_cache: list[list[tuple[str, Any]]] | None = None

    @classmethod
    def from_sweep(cls, source: str | Path | Mapping[str, Any]) -> SweepSampler:
        """Validate ``source`` and start at the first proposal."""

        return cls(load_sweep(source))

    def ask(self, n: int = 1) -> tuple[Trial, ...]:
        """Propose up to ``n`` new trials.

        Raises :class:`ohho.train.sweep.SweepError` with
        :data:`BUDGET_EXHAUSTED` when the budget or the search space has no
        further trial. A shorter tuple means the budget ended mid-batch.
        """

        if isinstance(n, bool) or not isinstance(n, int):
            raise SweepError("ask count must be an integer")
        if n < 0:
            raise SweepError("ask count must be >= 0")
        if n == 0:
            return ()
        proposed: list[Trial] = []
        for _ in range(n):
            if self._no_more():
                break
            assigned = self._next_assignment()
            trial = self._make_trial(assigned)
            if self._over_budget_hours(trial):
                self._gpu_stop = True
                break
            self._remember(trial)
            proposed.append(trial)
        if not proposed:
            raise SweepError(BUDGET_EXHAUSTED)
        return tuple(proposed)

    def tell(self, index: int, values: Mapping[str, Any]) -> None:
        """Record objective values for a trial this sampler proposed."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise SweepError("trial index must be an integer")
        trial = self._trial(index)
        if trial is None:
            raise SweepError(f"trial {index} has not been proposed")
        if trial["values"] is not None:
            raise SweepError(f"trial {index} already has objective values")
        trial["values"] = _coerce_values(self.spec, values)

    def leaderboard(self) -> dict[str, Any]:
        """``ohho.sweep.leaderboard/v1`` document for told trials.

        ``pareto`` is the non-dominated set. ``trials`` lists every told
        trial and sets ``dominated`` when it is off that set. Pending
        proposals are omitted.
        """

        completed = [item for item in self._trials if item["values"] is not None]
        front = set(
            pareto_indices([item["values"] for item in completed], self.spec.objectives)
        )
        trials = [
            _leaderboard_row(item, dominated=index not in front)
            for index, item in enumerate(completed)
        ]
        pareto = [_leaderboard_row(completed[index]) for index in sorted(front)]
        return {
            "schema": LEADERBOARD_SCHEMA_ID,
            "strategy": self.spec.strategy,
            "seed": self.spec.seed,
            "objectives": [dict(obj) for obj in self.spec.objectives],
            "trials": trials,
            "pareto": pareto,
        }

    def dump_state(self) -> dict[str, Any]:
        """JSON-ready ``ohho.sweep.sampler/v1`` state, including the leaderboard."""

        board = self.leaderboard()
        _check_schema(board, _schema(LEADERBOARD_FILENAME))
        state = {
            "schema": SAMPLER_SCHEMA_ID,
            "spec": _spec_document(self.spec),
            "rng": _export_rng(self._rng),
            "cursor": self._cursor,
            "gpu_stop": self._gpu_stop,
            "trials": [_state_trial(item) for item in self._trials],
            "leaderboard": board,
        }
        _check_schema(state, _schema(SAMPLER_FILENAME))
        return state

    @classmethod
    def load_state(cls, state: Mapping[str, Any]) -> SweepSampler:
        """Restore a sampler from :meth:`dump_state`.

        The embedded leaderboard is checked, then derived again from the
        trial log. Later proposals use the restored generator and the told
        objective values.
        """

        if not isinstance(state, Mapping):
            raise SweepError("sampler state must be a mapping")
        _check_schema(dict(state), _schema(SAMPLER_FILENAME))
        if state.get("schema") != SAMPLER_SCHEMA_ID:
            raise SweepError(f"sampler state schema must be {SAMPLER_SCHEMA_ID!r}")
        try:
            spec = validate_sweep(state["spec"])
            rng_payload = state["rng"]
            cursor = state["cursor"]
            gpu_stop = state["gpu_stop"]
            raw_trials = state["trials"]
        except KeyError as exc:
            raise SweepError(f"sampler state is missing {exc.args[0]!r}") from exc
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise SweepError("sampler state cursor must be an integer >= 0")
        if not isinstance(gpu_stop, bool):
            raise SweepError("sampler state gpu_stop must be a boolean")
        if not isinstance(raw_trials, list):
            raise SweepError("sampler state trials must be an array")
        if cursor != len(raw_trials):
            raise SweepError("sampler state cursor does not match the trial log")
        sampler = cls(spec)
        sampler._rng = _import_rng(rng_payload)
        sampler._gpu_stop = gpu_stop
        for expected, raw in enumerate(raw_trials):
            restored = _restore_trial(spec, raw, expected)
            sampler._trials.append(restored)
            sampler._hours += float(restored["estimated_gpu_hours"])
        sampler._cursor = cursor
        return sampler

    def _trial(self, index: int) -> dict[str, Any] | None:
        for item in self._trials:
            if item["index"] == index:
                return item
        return None

    def _trial_cap(self) -> int | None:
        cap = 1 if not self.spec.dimensions else None
        budget = self.spec.budget.get("max_trials")
        if budget is None:
            return cap
        limit = int(budget)
        if cap is None:
            return limit
        return min(cap, limit)

    def _no_more(self) -> bool:
        if self._gpu_stop:
            return True
        cap = self._trial_cap()
        if cap is not None and self._cursor >= cap:
            return True
        if self.spec.strategy == "grid" and self.spec.dimensions:
            return self._cursor >= len(self._grid_rows())
        return False

    def _grid_rows(self) -> list[list[tuple[str, Any]]]:
        if self._grid_cache is None:
            rows = _expand_grid(self.spec)
            budget = self.spec.budget.get("max_trials")
            if budget is not None:
                rows = rows[: int(budget)]
            self._grid_cache = rows
        return self._grid_cache

    def _next_assignment(self) -> list[tuple[str, Any]]:
        if not self.spec.dimensions:
            return []
        if self.spec.strategy == "grid":
            return list(self._grid_rows()[self._cursor])
        if self.spec.strategy == "random" or not self._tpe_ready():
            return self._propose_random()
        return self._propose_tpe()

    def _tpe_ready(self) -> bool:
        if self.spec.strategy not in ADAPTIVE_STRATEGIES:
            return False
        told = sum(1 for item in self._trials if item["values"] is not None)
        startup = int(self.spec.sampler["startup_trials"])
        return told >= max(startup, 2)

    def _propose_random(self) -> list[tuple[str, Any]]:
        limit = 1000
        rejected = 0
        for _ in range(limit):
            assigned = _sample_one(self.spec, self._rng)
            view = _apply(self.spec.recipe, assigned)
            if _allowed(view, self.spec.constraints):
                return assigned
            rejected += 1
        raise SweepError(
            "could not sample a trial that satisfies constraints "
            f"after {limit} attempts ({rejected} rejected)"
        )

    def _propose_tpe(self) -> list[tuple[str, Any]]:
        completed = [item for item in self._trials if item["values"] is not None]
        rows = [item["values"] for item in completed]
        multi = self.spec.strategy == "multi"
        gamma = float(self.spec.sampler["gamma"])
        good, bad = _split_indices(rows, self.spec.objectives, gamma, multi=multi)
        losses = _weight_losses(rows, self.spec.objectives, multi=multi)
        prior = float(self.spec.sampler["prior_weight"])
        models = [
            _fit_model(dim, completed, good, bad, losses, len(completed), prior)
            for dim in self.spec.dimensions
        ]
        candidates = int(self.spec.sampler["candidates"])
        limit = max(1000, candidates * 50)
        pool: list[tuple[float, list[tuple[str, Any]]]] = []
        attempts = 0
        rejected = 0
        while len(pool) < candidates and attempts < limit:
            attempts += 1
            assigned = _draw_candidate(self.spec, self._rng, models, prior)
            view = _apply(self.spec.recipe, assigned)
            if not _allowed(view, self.spec.constraints):
                rejected += 1
                continue
            score = _score_assignment(assigned, models, prior)
            pool.append((score, assigned))
        if not pool:
            raise SweepError(
                "could not sample a trial that satisfies constraints "
                f"after {limit} attempts ({rejected} rejected)"
            )
        return _softmax_pick(self._rng, pool)

    def _make_trial(self, assigned: Sequence[tuple[str, Any]]) -> Trial:
        recipe = _apply(self.spec.recipe, assigned)
        hours = _trial_hours(recipe, self.spec.estimate)
        overrides = {path: copy.deepcopy(value) for path, value in assigned}
        sets = tuple(_format_set(path, value) for path, value in assigned)
        return Trial(
            index=self._cursor,
            overrides=overrides,
            sets=sets,
            recipe=recipe,
            estimated_gpu_hours=hours,
            seed=trial_training_seed(self.spec.seed, self._cursor),
        )

    def _over_budget_hours(self, trial: Trial) -> bool:
        limit = self.spec.budget.get("max_gpu_hours")
        if limit is None:
            return False
        return self._hours + float(trial.estimated_gpu_hours) > float(limit) + 1e-9

    def _remember(self, trial: Trial) -> None:
        self._trials.append(
            {
                "index": trial.index,
                "seed": trial.seed,
                "overrides": copy.deepcopy(trial.overrides),
                "sets": list(trial.sets),
                "estimated_gpu_hours": float(trial.estimated_gpu_hours),
                "values": None,
            }
        )
        self._hours += float(trial.estimated_gpu_hours)
        self._cursor += 1


def startup_rows(spec: SweepSpec) -> list[list[tuple[str, Any]]]:
    """Random startup assignments for an adaptive ``expand`` plan.

    No objectives have been recorded, so this is the same prefix a fresh
    :meth:`SweepSampler.ask` would propose. The GPU-hour cap can shorten it.
    """

    if spec.strategy not in ADAPTIVE_STRATEGIES:
        raise SweepError(f"strategy {spec.strategy} is not adaptive")
    count = 1 if not spec.dimensions else int(spec.sampler["startup_trials"])
    budget = spec.budget.get("max_trials")
    if budget is not None:
        count = min(count, int(budget))
    if count <= 0:
        return []
    sampler = SweepSampler(spec)
    try:
        trials = sampler.ask(count)
    except SweepError as exc:
        if str(exc) == BUDGET_EXHAUSTED:
            return []
        raise
    return [
        [(key, value) for key, value in trial.overrides.items()] for trial in trials
    ]


def _schema(filename: str) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] | None = getattr(_schema, "_cache", None)
    if cache is None:
        cache = {}
        _schema._cache = cache  # type: ignore[attr-defined]
    loaded = cache.get(filename)
    if loaded is None:
        loaded = json.loads(_read_resource("schemas", filename))
        _assert_refs(loaded, loaded, "$")
        cache[filename] = loaded
    return loaded


def _check_schema(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    _validate(dict(instance), schema, schema, "$", 0)


def _leaderboard_row(
    item: Mapping[str, Any], *, dominated: bool | None = None
) -> dict[str, Any]:
    row = {
        "trial": item["index"],
        "values": dict(item["values"]),
        "overrides": copy.deepcopy(item["overrides"]),
        "sets": list(item["sets"]),
    }
    if dominated is not None:
        row["dominated"] = dominated
    return row


def _state_trial(item: Mapping[str, Any]) -> dict[str, Any]:
    values = item["values"]
    return {
        "index": item["index"],
        "seed": item["seed"],
        "overrides": copy.deepcopy(item["overrides"]),
        "sets": list(item["sets"]),
        "estimated_gpu_hours": item["estimated_gpu_hours"],
        "values": None if values is None else dict(values),
    }


def _restore_trial(
    spec: SweepSpec, raw: Mapping[str, Any], expected: int
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SweepError("sampler trial must be an object")
    index = raw.get("index")
    if index != expected:
        raise SweepError("sampler trial indices must be 0..n-1 in order")
    values = raw.get("values")
    restored = None if values is None else _coerce_values(spec, values)
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SweepError("sampler trial seed must be an integer in [0, 2**32)")
    if seed != trial_training_seed(spec.seed, expected):
        raise SweepError(f"trial {expected} seed does not match the sweep seed")
    hours = raw.get("estimated_gpu_hours")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise SweepError("estimated_gpu_hours must be a number")
    if isinstance(hours, float) and not math.isfinite(hours):
        raise SweepError("estimated_gpu_hours must be a finite number")
    sets = raw.get("sets")
    if not isinstance(sets, list) or not all(isinstance(item, str) for item in sets):
        raise SweepError("sampler trial sets must be an array of strings")
    overrides = raw.get("overrides")
    if not isinstance(overrides, dict):
        raise SweepError("sampler trial overrides must be an object")
    return {
        "index": index,
        "seed": seed,
        "overrides": copy.deepcopy(overrides),
        "sets": list(sets),
        "estimated_gpu_hours": float(hours),
        "values": restored,
    }


def _coerce_values(spec: SweepSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise SweepError("objective values must be a mapping")
    expected = [obj["name"] for obj in spec.objectives]
    if set(values) != set(expected) or len(values) != len(expected):
        if not expected:
            raise SweepError("objective values must be empty")
        names = ", ".join(expected)
        raise SweepError(f"objective values must contain exactly {names}")
    return {name: _finite_objective(values[name], name) for name in expected}


def _finite_objective(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError(f"objective {name} must be a finite number")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SweepError(f"objective {name} must be a finite number")
        return float(value)
    return int(value)


def _export_rng(rng: random.Random) -> dict[str, Any]:
    version, state, gauss = rng.getstate()
    return {"version": version, "state": list(state), "gauss": gauss}


def _import_rng(payload: Mapping[str, Any]) -> random.Random:
    if not isinstance(payload, Mapping):
        raise SweepError("sampler state rng must be a mapping")
    try:
        version = payload["version"]
        state = payload["state"]
        gauss = payload["gauss"]
    except KeyError as exc:
        raise SweepError(f"sampler state rng is missing {exc.args[0]!r}") from exc
    if isinstance(version, bool) or not isinstance(version, int):
        raise SweepError("sampler state rng version must be an integer")
    if not isinstance(state, list) or not state:
        raise SweepError("sampler state rng state must be a non-empty array")
    ints: list[int] = []
    for item in state:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SweepError("sampler state rng state must be integers")
        ints.append(item)
    if gauss is not None and (
        isinstance(gauss, bool) or not isinstance(gauss, (int, float))
    ):
        raise SweepError("sampler state rng gauss must be a number or null")
    if isinstance(gauss, float) and not math.isfinite(gauss):
        raise SweepError("sampler state rng gauss must be a finite number")
    restored = random.Random()
    third = None if gauss is None else float(gauss)
    try:
        restored.setstate((version, tuple(ints), third))
    except (TypeError, ValueError) as exc:
        raise SweepError(f"sampler state has an invalid rng: {exc}") from exc
    return restored


def _spec_document(spec: SweepSpec) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": spec.schema,
        "seed": spec.seed,
        "strategy": spec.strategy,
        "parallelism": spec.parallelism,
        "recipe": copy.deepcopy(spec.recipe),
        "dimensions": [_dimension_document(dim) for dim in spec.dimensions],
        "constraints": [copy.deepcopy(item) for item in spec.constraints],
        "objectives": [dict(item) for item in spec.objectives],
        "budget": dict(spec.budget),
        "estimate": {
            "gpu_hours_per_trial": spec.estimate["gpu_hours_per_trial"],
            "by_value": [copy.deepcopy(rule) for rule in spec.estimate["by_value"]],
        },
        "sampler": dict(spec.sampler),
    }
    if spec.early_stopping is not None:
        document["early_stopping"] = copy.deepcopy(spec.early_stopping)
    return document


def _dimension_document(dim: Dimension) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": dim.name,
        "path": dim.path,
        "type": dim.type,
    }
    if dim.values is not None:
        document["values"] = list(dim.values)
    else:
        if dim.low is not None:
            document["low"] = dim.low
        if dim.high is not None:
            document["high"] = dim.high
        if dim.step is not None:
            document["step"] = dim.step
        if dim.n is not None:
            document["n"] = dim.n
    if dim.when is not None:
        document["when"] = copy.deepcopy(dict(dim.when))
    return document


def _pareto_ranks(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
) -> list[int]:
    remaining = set(range(len(rows)))
    ranks = [0] * len(rows)
    rank = 0
    while remaining:
        front: list[int] = []
        for index in sorted(remaining):
            dominated = False
            for other in remaining:
                if other != index and _dominates(rows[other], rows[index], objectives):
                    dominated = True
                    break
            if not dominated:
                front.append(index)
        if not front:
            for index in remaining:
                ranks[index] = rank
            break
        for index in front:
            ranks[index] = rank
            remaining.remove(index)
        rank += 1
    return ranks


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    objectives: Sequence[Mapping[str, Any]],
) -> bool:
    better = False
    for obj in objectives:
        name = str(obj["name"])
        lv = float(left[name])
        rv = float(right[name])
        if obj["direction"] == "maximize":
            if lv < rv:
                return False
            if lv > rv:
                better = True
        else:
            if lv > rv:
                return False
            if lv < rv:
                better = True
    return better


def _split_indices(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
    gamma: float,
    *,
    multi: bool,
) -> tuple[list[int], list[int]]:
    n = len(rows)
    if n == 0:
        return [], []
    if multi:
        ranks = _pareto_ranks(rows, objectives)
        front = [index for index, rank in enumerate(ranks) if rank == 0]
        if len(front) < n:
            target = _good_count(n, gamma)
            if len(front) >= target:
                good = list(front)
            else:
                order = sorted(range(n), key=lambda i: (ranks[i], i))
                good = order[:target]
            return _partition(n, good)
    losses = _objective_losses(rows, objectives, multi=multi)
    order = sorted(range(n), key=lambda i: (losses[i], i))
    return _partition(n, order[: _good_count(n, gamma)])


def _partition(n: int, good: Sequence[int]) -> tuple[list[int], list[int]]:
    chosen = set(good)
    return (
        sorted(chosen),
        [index for index in range(n) if index not in chosen],
    )


def _good_count(n: int, gamma: float) -> int:
    target = max(1, math.ceil(gamma * n - 1e-12))
    if n > 1:
        return min(target, n - 1)
    return 1


def _objective_losses(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
    *,
    multi: bool,
) -> list[float]:
    if multi:
        return _normalized_losses(rows, objectives)
    primary = _primary_objective(objectives)
    return [_scalar_loss(row, primary) for row in rows]


def _primary_objective(objectives: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    chosen = [obj for obj in objectives if obj.get("primary")]
    if len(chosen) == 1:
        return chosen[0]
    if len(objectives) == 1:
        return objectives[0]
    raise SweepError("tpe requires a primary objective")


def _scalar_loss(row: Mapping[str, Any], obj: Mapping[str, Any]) -> float:
    value = float(row[str(obj["name"])])
    if obj["direction"] == "maximize":
        return -value
    return value


def _normalized_losses(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
) -> list[float]:
    bounds: dict[str, tuple[float, float]] = {}
    for obj in objectives:
        name = str(obj["name"])
        values = [float(row[name]) for row in rows]
        bounds[name] = (min(values), max(values))
    losses: list[float] = []
    for row in rows:
        total = 0.0
        for obj in objectives:
            name = str(obj["name"])
            lo, hi = bounds[name]
            span = hi - lo
            unit = 0.0 if span == 0 else (float(row[name]) - lo) / span
            if obj["direction"] == "maximize":
                unit = 1.0 - unit
            total += unit
        losses.append(total)
    return losses


def _weight_losses(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Mapping[str, Any]],
    *,
    multi: bool,
) -> list[float]:
    if not multi:
        return _objective_losses(rows, objectives, multi=False)
    ranks = _pareto_ranks(rows, objectives)
    scaled = _normalized_losses(rows, objectives)
    return [float(ranks[i]) + 0.01 * scaled[i] for i in range(len(rows))]


def _fit_model(
    dim: Dimension,
    completed: Sequence[Mapping[str, Any]],
    good: Sequence[int],
    bad: Sequence[int],
    losses: Sequence[float],
    n_told: int,
    prior: float,
) -> _DimModel:
    good_pairs = _observed_pairs(dim, completed, good, losses)
    bad_pairs = _observed_pairs(dim, completed, bad, losses)
    discrete = _discrete_levels(dim)
    if discrete is not None:
        return _DimModel(dim, discrete, good_pairs, bad_pairs, None, None)
    return _DimModel(
        dim,
        None,
        good_pairs,
        bad_pairs,
        _components(dim, good_pairs, prior, tight=True, n_told=n_told),
        _components(dim, bad_pairs, prior, tight=False, n_told=n_told),
    )


def _observed_pairs(
    dim: Dimension,
    completed: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    losses: Sequence[float],
) -> list[tuple[Any, float]]:
    pairs: list[tuple[Any, float]] = []
    for index in indices:
        overrides = completed[index]["overrides"]
        if dim.path in overrides:
            pairs.append((overrides[dim.path], float(losses[index])))
    return pairs


def _discrete_levels(dim: Dimension) -> list[Any] | None:
    if dim.values is not None:
        return list(dim.values)
    if dim.low is None or dim.high is None:
        return None
    if dim.type == "int":
        if dim.n is not None:
            return list(
                _int_linspace(
                    int(dim.low),
                    int(dim.high),
                    dim.n,
                    f"dimension {dim.name}",
                )
            )
        step = 1 if dim.step is None else int(dim.step)
        count = ((int(dim.high) - int(dim.low)) // step) + 1
        if count <= _DISCRETE_LIMIT:
            return [int(dim.low) + step * i for i in range(count)]
        return None
    if dim.n is None:
        return None
    if dim.type == "float":
        return _float_linspace(float(dim.low), float(dim.high), dim.n)
    if dim.type == "log_uniform":
        return _log_linspace(float(dim.low), float(dim.high), dim.n)
    return None


def _components(
    dim: Dimension,
    pairs: Sequence[tuple[Any, float]],
    prior: float,
    *,
    tight: bool,
    n_told: int,
) -> list[tuple[float, float, float]]:
    bw = _scheduled_bandwidth(n_told, tight=tight)
    best = min((loss for _, loss in pairs), default=None)
    comps: list[tuple[float, float, float]] = []
    for value, loss in pairs:
        weight = 1.0
        if tight and best is not None and loss <= best + 1e-12:
            weight = 3.0
        comps.append((_to_unit(dim, value), bw, weight))
    prior_bw = 0.35 if tight else 0.55
    comps.append((0.5, prior_bw, prior))
    return comps


def _scheduled_bandwidth(n_told: int, *, tight: bool) -> float:
    """Global Parzen bandwidth. It shrinks as trials are told and stays wide
    enough that a new draw can leave the current best point.
    """

    step = max(0, n_told - 2)
    if tight:
        return max(0.04, 0.20 * (0.94**step))
    return max(0.10, 0.45 * (0.97**step))


def _draw_candidate(
    spec: SweepSpec,
    rng: random.Random,
    models: Sequence[_DimModel],
    prior: float,
) -> list[tuple[str, Any]]:
    by_path = {model.dim.path: model for model in models}
    assigned: list[tuple[str, Any]] = []
    for dim in spec.dimensions:
        view = _apply(spec.recipe, assigned)
        if dim.when is not None and not _matches(dim.when, view):
            continue
        assigned.append((dim.path, _draw_dim(by_path[dim.path], rng, prior)))
    return assigned


def _draw_dim(model: _DimModel, rng: random.Random, prior: float) -> Any:
    if model.discrete is not None:
        return _draw_discrete(rng, model.discrete, model.good_pairs, prior, boost=True)
    if model.good_comps is None:
        raise SweepError(f"dimension {model.dim.name}: nothing to sample")
    return _from_unit(model.dim, _sample_mixture(rng, model.good_comps))


def _draw_discrete(
    rng: random.Random,
    choices: Sequence[Any],
    pairs: Sequence[tuple[Any, float]],
    prior: float,
    *,
    boost: bool,
) -> Any:
    weights = _choice_weights(choices, pairs, prior, boost=boost)
    return choices[_weighted_index(rng, weights)]


def _choice_weights(
    choices: Sequence[Any],
    pairs: Sequence[tuple[Any, float]],
    prior: float,
    *,
    boost: bool,
) -> list[float]:
    best = min((loss for _, loss in pairs), default=None)
    weights: list[float] = []
    for choice in choices:
        weight = prior
        for value, loss in pairs:
            if not _json_equal(value, choice):
                continue
            extra = 3.0 if boost and best is not None and loss <= best + 1e-12 else 1.0
            weight += extra
        weights.append(weight)
    return weights


def _softmax_pick(
    rng: random.Random,
    pool: Sequence[tuple[float, list[tuple[str, Any]]]],
) -> list[tuple[str, Any]]:
    scores = [score for score, _ in pool]
    hi = max(scores)
    lo = min(scores)
    span = hi - lo
    if span < 1e-12:
        return pool[_weighted_index(rng, [1.0] * len(pool))][1]
    temp = 0.35 * span
    weights = [math.exp((score - hi) / temp) for score in scores]
    return pool[_weighted_index(rng, weights)][1]


def _score_assignment(
    assigned: Sequence[tuple[str, Any]],
    models: Sequence[_DimModel],
    prior: float,
) -> float:
    found = {path: value for path, value in assigned}
    total = 0.0
    for model in models:
        if model.dim.path not in found:
            continue
        total += _log_ratio(model, found[model.dim.path], prior)
    return total


def _log_ratio(model: _DimModel, value: Any, prior: float) -> float:
    if model.discrete is not None:
        good = _log_cat(value, model.good_pairs, model.discrete, prior, boost=True)
        bad = _log_cat(value, model.bad_pairs, model.discrete, prior, boost=False)
        return good - bad
    if model.good_comps is None or model.bad_comps is None:
        return 0.0
    unit = _to_unit(model.dim, value)
    return _log_mix(unit, model.good_comps) - _log_mix(unit, model.bad_comps)


def _log_cat(
    value: Any,
    pairs: Sequence[tuple[Any, float]],
    choices: Sequence[Any],
    prior: float,
    *,
    boost: bool,
) -> float:
    weights = _choice_weights(choices, pairs, prior, boost=boost)
    total = math.fsum(weights)
    matched = prior
    for choice, weight in zip(choices, weights):
        if _json_equal(choice, value):
            matched = weight
            break
    density = matched / total if total else 1.0
    return math.log(max(density, 1e-300))


def _log_mix(unit: float, comps: Sequence[tuple[float, float, float]]) -> float:
    return math.log(max(_mixture_density(unit, comps), 1e-300))


def _mixture_density(unit: float, comps: Sequence[tuple[float, float, float]]) -> float:
    total_w = math.fsum(weight for _, _, weight in comps)
    if total_w <= 0:
        return 1.0
    acc = 0.0
    for mu, bw, weight in comps:
        acc += weight * _reflected_gauss(unit, mu, bw)
    return acc / total_w


def _reflected_gauss(x: float, mu: float, bw: float) -> float:
    bw = max(bw, 1e-6)
    norm = 1.0 / (bw * math.sqrt(2.0 * math.pi))
    total = 0.0
    for center in (mu, -mu, 2.0 - mu):
        z = (x - center) / bw
        total += math.exp(-0.5 * z * z)
    return norm * total


def _sample_mixture(
    rng: random.Random, comps: Sequence[tuple[float, float, float]]
) -> float:
    index = _weighted_index(rng, [weight for _, _, weight in comps])
    mu, bw, _ = comps[index]
    return _fold(rng.gauss(mu, max(bw, 1e-6)))


def _weighted_index(rng: random.Random, weights: Sequence[float]) -> int:
    total = math.fsum(weights)
    if total <= 0:
        return 0
    pick = rng.random() * total
    acc = 0.0
    for index, weight in enumerate(weights):
        acc += weight
        if pick <= acc:
            return index
    return len(weights) - 1


def _fold(unit: float) -> float:
    if not math.isfinite(unit):
        return 0.5
    wrapped = math.fmod(unit, 2.0)
    if wrapped < 0.0:
        wrapped += 2.0
    if wrapped > 1.0:
        wrapped = 2.0 - wrapped
    return wrapped


def _span(dim: Dimension) -> tuple[float, float]:
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: nothing to sample")
    if dim.type == "log_uniform":
        return math.log(float(dim.low)), math.log(float(dim.high))
    return float(dim.low), float(dim.high)


def _to_unit(dim: Dimension, value: Any) -> float:
    lo, hi = _span(dim)
    span = hi - lo
    if span <= 0:
        return 0.5
    raw = math.log(float(value)) if dim.type == "log_uniform" else float(value)
    return min(1.0, max(0.0, (raw - lo) / span))


def _from_unit(dim: Dimension, unit: float) -> Any:
    unit = min(1.0, max(0.0, unit))
    lo, hi = _span(dim)
    raw = lo + unit * (hi - lo)
    if dim.type == "int":
        return _snap_int(dim, unit)
    if dim.type == "log_uniform":
        value = math.exp(raw)
        return min(float(dim.high), max(float(dim.low), value))
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: nothing to sample")
    return min(float(dim.high), max(float(dim.low), raw))


def _snap_int(dim: Dimension, unit: float) -> int:
    if dim.low is None or dim.high is None:
        raise SweepError(f"dimension {dim.name}: nothing to sample")
    low = int(dim.low)
    high = int(dim.high)
    step = 1 if dim.step is None else int(dim.step)
    if high <= low:
        return low
    slots = (high - low) // step
    bucket = int(round(unit * slots))
    bucket = min(slots, max(0, bucket))
    return low + bucket * step


__all__ = [
    "LEADERBOARD_SCHEMA_ID",
    "SAMPLER_SCHEMA_ID",
    "BUDGET_EXHAUSTED",
    "SweepSampler",
    "pareto_indices",
    "load_leaderboard_schema",
    "load_sampler_schema",
    "startup_rows",
]
