"""Grid, random, TPE, and multi-objective sweep samplers."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path

from ohho.cli import main
from ohho.train.sweep import (
    ADAPTIVE_PLAN_NOTE,
    SCHEMA_ID,
    STRATEGIES,
    SweepError,
    expand,
    load_schema,
    load_sweep,
    plan_as_dict,
    validate_sweep,
)
from ohho.train.sweep_sampler import (
    BUDGET_EXHAUSTED,
    LEADERBOARD_SCHEMA_ID,
    SAMPLER_SCHEMA_ID,
    SweepSampler,
    load_leaderboard_schema,
    load_sampler_schema,
    pareto_indices,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "ohho" / "train" / "schemas" / "examples"
_OPTIMUM = (0.23, 0.77)


def _quadratic(x: float, y: float) -> float:
    return (x - _OPTIMUM[0]) ** 2 + (y - _OPTIMUM[1]) ** 2


def _bench(strategy: str, *, candidates: int = 64) -> dict:
    return {
        "schema": SCHEMA_ID,
        "strategy": strategy,
        "seed": 0,
        "budget": {"max_trials": 40},
        "sampler": {
            "startup_trials": 10,
            "gamma": 0.25,
            "candidates": candidates,
            "prior_weight": 1.0,
        },
        "objectives": [{"name": "loss", "direction": "minimize", "primary": True}],
        "dimensions": [
            {"name": "x", "type": "float", "low": 0, "high": 1},
            {"name": "y", "type": "float", "low": 0, "high": 1},
        ],
    }


def _best_loss(strategy: str) -> float:
    sampler = SweepSampler.from_sweep(_bench(strategy))
    best = math.inf
    for _ in range(40):
        trial = sampler.ask()[0]
        x = float(trial.overrides["x"])
        y = float(trial.overrides["y"])
        loss = _quadratic(x, y)
        sampler.tell(trial.index, {"loss": loss})
        best = min(best, loss)
    return best


class TestStrategySchema(unittest.TestCase):
    def test_enum_matches_the_packaged_schema(self) -> None:
        schema = load_schema()
        strategy = schema["properties"]["strategy"]
        self.assertEqual(strategy["enum"], list(STRATEGIES))
        text = strategy["description"]
        self.assertIn("Bayesian", text)
        self.assertIn("Parzen", text)
        self.assertIn("Pareto", text)
        self.assertIn("multi-objective", text)
        self.assertIn("tpe", schema["description"])
        self.assertIn("multi", schema["description"])
        board = load_leaderboard_schema()
        self.assertEqual(board["title"], LEADERBOARD_SCHEMA_ID)
        self.assertEqual(board["properties"]["strategy"]["enum"], list(STRATEGIES))
        state = load_sampler_schema()
        self.assertEqual(state["title"], SAMPLER_SCHEMA_ID)
        for name in (
            "sweep.leaderboard.v1.json",
            "sweep.sampler.v1.json",
            "sweep.run.v1.json",
        ):
            packaged = (
                files("ohho.train")
                .joinpath("schemas", name)
                .read_text(encoding="utf-8")
            )
            on_disk = (_ROOT / "ohho" / "train" / "schemas" / name).read_text(
                encoding="utf-8"
            )
            self.assertEqual(packaged, on_disk)

    def test_adaptive_rules(self) -> None:
        with self.assertRaises(SweepError) as caught:
            validate_sweep({"schema": SCHEMA_ID, "strategy": "bayes"})
        self.assertIn("must be one of", str(caught.exception))
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "strategy": "tpe",
                    "budget": {"max_trials": 2},
                    "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
                }
            )
        self.assertIn("requires an objective", str(caught.exception))
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "strategy": "tpe",
                    "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
                    "objectives": [{"name": "loss", "direction": "minimize"}],
                }
            )
        self.assertIn("max_trials", str(caught.exception))
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "strategy": "bayesian",
                    "budget": {"max_trials": 2},
                    "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
                    "objectives": [
                        {"name": "a", "direction": "minimize"},
                        {"name": "b", "direction": "maximize"},
                    ],
                }
            )
        self.assertIn("requires a primary", str(caught.exception))
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "strategy": "multi",
                    "budget": {"max_trials": 2},
                    "objectives": [{"name": "a", "direction": "minimize"}],
                    "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
                }
            )
        self.assertIn("at least two objectives", str(caught.exception))
        with self.assertRaises(SweepError):
            validate_sweep({"schema": SCHEMA_ID, "sampler": {"gamma": 0}})
        with self.assertRaises(SweepError):
            validate_sweep({"schema": SCHEMA_ID, "sampler": {"candidates": 0}})
        with self.assertRaises(SweepError):
            validate_sweep({"schema": SCHEMA_ID, "sampler": {"extra": 1}})
        spec = validate_sweep(
            {
                "schema": SCHEMA_ID,
                "strategy": "tpe",
                "budget": {"max_trials": 2},
                "objectives": [{"name": "loss", "direction": "minimize"}],
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            }
        )
        self.assertEqual(spec.strategy, "tpe")
        self.assertEqual(spec.sampler["startup_trials"], 10)
        self.assertEqual(spec.sampler["candidates"], 64)

    def test_startup_plan_matches_random_prefix(self) -> None:
        doc = _bench("tpe")
        doc["sampler"] = {"startup_trials": 4, "candidates": 8}
        doc["budget"] = {"max_trials": 10}
        plan = expand(doc)
        self.assertEqual(len(plan.trials), 4)
        self.assertEqual(plan.note, ADAPTIVE_PLAN_NOTE.format(strategy="tpe"))
        self.assertIn("note", plan_as_dict(plan))
        random_doc = dict(doc)
        random_doc["strategy"] = "random"
        random_doc["budget"] = {"max_trials": 4}
        random_plan = expand(random_doc)
        self.assertEqual(
            [trial.overrides for trial in plan.trials],
            [trial.overrides for trial in random_plan.trials],
        )
        bayesian = dict(doc)
        bayesian["strategy"] = "bayesian"
        self.assertEqual(
            [trial.overrides for trial in expand(bayesian).trials],
            [trial.overrides for trial in plan.trials],
        )
        self.assertNotIn("note", plan_as_dict(expand({"schema": SCHEMA_ID})))
        empty = expand(
            {
                "schema": SCHEMA_ID,
                "strategy": "tpe",
                "budget": {"max_trials": 5},
                "sampler": {"startup_trials": 0},
                "objectives": [{"name": "loss", "direction": "minimize"}],
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            }
        )
        self.assertEqual(empty.trials, ())
        self.assertIn("startup batch", empty.note)


class TestSamplerAgreesWithExpand(unittest.TestCase):
    def test_grid_sampler_matches_expand(self) -> None:
        path = _EXAMPLES / "grid_2x3x4.json"
        plan = expand(path)
        sampler = SweepSampler(load_sweep(path))
        got = []
        with self.assertRaises(SweepError) as caught:
            while True:
                got.extend(sampler.ask(4))
        self.assertEqual(str(caught.exception), BUDGET_EXHAUSTED)
        self.assertEqual(
            [trial.overrides for trial in got],
            [trial.overrides for trial in plan.trials],
        )
        self.assertEqual(
            [trial.sets for trial in got], [trial.sets for trial in plan.trials]
        )

    def test_random_sampler_matches_expand(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "random",
            "seed": 1,
            "budget": {"max_trials": 5, "max_gpu_hours": 3},
            "estimate": {"gpu_hours_per_trial": 1},
            "dimensions": [
                {"name": "lr", "type": "log_uniform", "low": 1e-5, "high": 1e-2},
                {"name": "scale", "type": "float", "low": 0, "high": 1},
                {
                    "name": "choice",
                    "type": "categorical",
                    "values": ["a", "b", "c", "d", "e", "f"],
                },
            ],
        }
        plan = expand(doc)
        sampler = SweepSampler.from_sweep(doc)
        got = list(sampler.ask(5))
        self.assertEqual(
            [trial.overrides for trial in got],
            [trial.overrides for trial in plan.trials],
        )
        with self.assertRaises(SweepError) as caught:
            sampler.ask()
        self.assertEqual(str(caught.exception), BUDGET_EXHAUSTED)

    def test_gpu_cap_stops_on_the_same_prefix(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "budget": {"max_gpu_hours": 1.5},
            "estimate": {"gpu_hours_per_trial": 1},
            "dimensions": [
                {"name": "model", "type": "categorical", "values": ["a", "b", "c"]}
            ],
        }
        plan = expand(doc)
        sampler = SweepSampler.from_sweep(doc)
        got = list(sampler.ask(5))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].overrides, plan.trials[0].overrides)
        with self.assertRaises(SweepError) as caught:
            sampler.ask()
        self.assertEqual(str(caught.exception), BUDGET_EXHAUSTED)


class TestResume(unittest.TestCase):
    def test_json_state_resumes_every_strategy(self) -> None:
        cases = {
            "grid": {
                "schema": SCHEMA_ID,
                "strategy": "grid",
                "seed": 4,
                "dimensions": [
                    {"name": "a", "type": "categorical", "values": ["a", "b"]},
                    {"name": "b", "type": "int", "values": [1, 2, 3]},
                ],
            },
            "random": {
                "schema": SCHEMA_ID,
                "strategy": "random",
                "seed": 4,
                "budget": {"max_trials": 8},
                "dimensions": [
                    {"name": "x", "type": "float", "low": 0, "high": 1},
                    {"name": "n", "type": "int", "low": 0, "high": 5},
                ],
            },
            "tpe": {
                "schema": SCHEMA_ID,
                "strategy": "tpe",
                "seed": 4,
                "budget": {"max_trials": 8},
                "sampler": {"startup_trials": 2, "candidates": 8, "gamma": 0.25},
                "objectives": [
                    {"name": "loss", "direction": "minimize", "primary": True}
                ],
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            },
            "bayesian": {
                "schema": SCHEMA_ID,
                "strategy": "bayesian",
                "seed": 4,
                "budget": {"max_trials": 8},
                "sampler": {"startup_trials": 2, "candidates": 8, "gamma": 0.25},
                "objectives": [{"name": "loss", "direction": "minimize"}],
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            },
            "multi": {
                "schema": SCHEMA_ID,
                "strategy": "multi",
                "seed": 4,
                "budget": {"max_trials": 8},
                "sampler": {"startup_trials": 2, "candidates": 8, "gamma": 0.25},
                "objectives": [
                    {"name": "success_rate", "direction": "maximize"},
                    {"name": "latency_p95_ms", "direction": "minimize"},
                ],
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            },
        }
        for name, doc in cases.items():
            with self.subTest(strategy=name):
                self._resume(doc, before=3, after=3)

    def test_spec_round_trip_and_batch_asks(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "tpe",
            "seed": 3,
            "parallelism": 2,
            "recipe": {"task": "push", "optimizer": {"weight_decay": 0.0}},
            "dimensions": [
                {"name": "family", "type": "categorical", "values": ["act", "vla"]},
                {
                    "name": "rank",
                    "type": "int",
                    "values": [4, 8],
                    "when": {"path": "family", "eq": "vla"},
                },
                {
                    "name": "lr",
                    "path": "optimizer.lr",
                    "type": "log_uniform",
                    "low": 1e-4,
                    "high": 1e-2,
                },
            ],
            "constraints": [{"prune_when": {"path": "family", "eq": "act"}}],
            "objectives": [{"name": "loss", "direction": "minimize", "primary": True}],
            "early_stopping": {
                "metric": "loss",
                "patience": 2,
                "min_delta": 0.01,
                "mode": "min",
            },
            "budget": {"max_trials": 6, "max_gpu_hours": 10},
            "estimate": {
                "gpu_hours_per_trial": 0.1,
                "by_value": [
                    {"when": {"path": "family", "eq": "vla"}, "gpu_hours": 0.4}
                ],
            },
            "sampler": {
                "startup_trials": 2,
                "gamma": 0.3,
                "candidates": 8,
                "prior_weight": 0.5,
            },
        }
        spec = validate_sweep(doc)
        sampler = SweepSampler(spec)
        self.assertEqual(spec, validate_sweep(sampler.dump_state()["spec"]))
        fresh = SweepSampler.from_sweep(doc)
        self.assertEqual(fresh.ask(0), ())
        left = SweepSampler.from_sweep(doc)
        right = SweepSampler.from_sweep(doc)
        for _ in range(3):
            trial = left.ask()[0]
            other = right.ask()[0]
            self.assertEqual(trial.overrides, other.overrides)
            values = {"loss": float(trial.overrides["optimizer.lr"])}
            left.tell(trial.index, values)
            right.tell(other.index, values)
        self.assertEqual(
            [trial.overrides for trial in left.ask(2)],
            [right.ask()[0].overrides, right.ask()[0].overrides],
        )

    def test_bad_state_is_rejected(self) -> None:
        sampler = SweepSampler.from_sweep(
            {
                "schema": SCHEMA_ID,
                "strategy": "random",
                "seed": 1,
                "budget": {"max_trials": 2},
                "dimensions": [
                    {"name": "x", "type": "categorical", "values": ["a", "b"]}
                ],
            }
        )
        sampler.ask()
        state = sampler.dump_state()
        state["cursor"] = 5
        with self.assertRaises(SweepError) as caught:
            SweepSampler.load_state(state)
        self.assertIn("cursor", str(caught.exception))
        with self.assertRaises(SweepError):
            SweepSampler.load_state({"schema": "ohho.sweep.sampler/v0"})

    def _resume(self, doc: dict, *, before: int, after: int) -> None:
        live = SweepSampler.from_sweep(doc)
        self._play(live, before)
        restored = SweepSampler.load_state(json.loads(json.dumps(live.dump_state())))
        self.assertEqual(restored.leaderboard(), live.leaderboard())
        for _ in range(after):
            left = live.ask()[0]
            right = restored.ask()[0]
            self.assertEqual(left.overrides, right.overrides)
            self.assertEqual(left.sets, right.sets)
            values = self._values(doc, left.overrides)
            live.tell(left.index, values)
            restored.tell(right.index, values)

    def _play(self, sampler: SweepSampler, count: int) -> None:
        doc = validate_sweep(sampler.dump_state()["spec"])
        for _ in range(count):
            trial = sampler.ask()[0]
            sampler.tell(
                trial.index, self._values(doc_as_mapping(doc), trial.overrides)
            )

    def _values(self, doc: dict, overrides: dict) -> dict:
        names = [item["name"] for item in doc.get("objectives") or []]
        if not names:
            return {}
        x = float(next(iter(overrides.values())))
        if names == ["loss"]:
            return {"loss": (x - 0.2) ** 2}
        return {"success_rate": x, "latency_p95_ms": 1.0 - x}


def doc_as_mapping(spec) -> dict:
    return {
        "objectives": [dict(item) for item in spec.objectives],
    }


class TestParetoLeaderboard(unittest.TestCase):
    def test_dominance_rule(self) -> None:
        minimize = [{"name": "a", "direction": "minimize", "primary": False}]
        self.assertEqual(pareto_indices([{"a": 1}, {"a": 1}], minimize), [0, 1])
        self.assertEqual(pareto_indices([{"a": 1}, {"a": 2}], minimize), [0])
        maximize = [{"name": "a", "direction": "maximize", "primary": False}]
        self.assertEqual(pareto_indices([{"a": 1}, {"a": 2}], maximize), [1])

    def test_multi_objective_pareto_is_stored(self) -> None:
        values = [
            {"success_rate": 0.9, "latency_p95_ms": 10, "memory_mb": 100},
            {"success_rate": 0.95, "latency_p95_ms": 50, "memory_mb": 200},
            {"success_rate": 0.5, "latency_p95_ms": 40, "memory_mb": 300},
            {"success_rate": 0.92, "latency_p95_ms": 12, "memory_mb": 80},
            {"success_rate": 0.91, "latency_p95_ms": 30, "memory_mb": 150},
        ]
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "multi",
            "seed": 1,
            "budget": {"max_trials": 5},
            "objectives": [
                {"name": "success_rate", "direction": "maximize"},
                {"name": "latency_p95_ms", "direction": "minimize"},
                {"name": "memory_mb", "direction": "minimize"},
            ],
            "dimensions": [
                {"name": "tag", "type": "categorical", "values": ["a", "b"]}
            ],
        }
        sampler = SweepSampler.from_sweep(doc)
        for row in values:
            trial = sampler.ask()[0]
            sampler.tell(trial.index, row)
        board = sampler.leaderboard()
        self.assertEqual(board["schema"], LEADERBOARD_SCHEMA_ID)
        self.assertEqual(board["strategy"], "multi")
        self.assertEqual(board["seed"], 1)
        self.assertEqual(
            set(board),
            {"schema", "strategy", "seed", "objectives", "trials", "pareto"},
        )
        self.assertEqual(
            [row["name"] for row in board["objectives"]],
            ["success_rate", "latency_p95_ms", "memory_mb"],
        )
        self.assertEqual([row["trial"] for row in board["pareto"]], [0, 1, 3])
        self.assertEqual(
            [row["dominated"] for row in board["trials"]],
            [False, False, True, False, True],
        )
        self.assertEqual(board["pareto"][0]["values"], values[0])
        self.assertEqual(
            set(board["pareto"][0]), {"trial", "values", "overrides", "sets"}
        )
        self.assertEqual(
            set(board["trials"][2]),
            {"trial", "values", "overrides", "sets", "dominated"},
        )
        self.assertNotIn(2, [row["trial"] for row in board["pareto"]])
        self.assertNotIn(4, [row["trial"] for row in board["pareto"]])
        state = json.loads(json.dumps(sampler.dump_state()))
        self.assertEqual(state["schema"], SAMPLER_SCHEMA_ID)
        self.assertEqual(state["leaderboard"]["pareto"], board["pareto"])
        restored = SweepSampler.load_state(state)
        self.assertEqual(restored.leaderboard()["pareto"], board["pareto"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sampler.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["trial"] for row in loaded["leaderboard"]["pareto"]],
            [0, 1, 3],
        )


class TestTpeBeatsRandom(unittest.TestCase):
    def test_seeded_quadratic(self) -> None:
        tpe = _best_loss("tpe")
        random_best = _best_loss("random")
        self.assertLess(tpe, random_best, f"tpe {tpe} random {random_best}")
        self.assertLess(tpe, 1e-3)

    def test_bayesian_matches_tpe(self) -> None:
        self.assertEqual(_best_loss("bayesian"), _best_loss("tpe"))

    def test_draws_stay_in_bounds_and_honor_conditions(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "tpe",
            "seed": 5,
            "budget": {"max_trials": 8},
            "sampler": {"startup_trials": 2, "candidates": 8},
            "objectives": [{"name": "loss", "direction": "minimize", "primary": True}],
            "dimensions": [
                {"name": "family", "type": "categorical", "values": ["act", "vla"]},
                {
                    "name": "rank",
                    "type": "int",
                    "values": [4, 8],
                    "when": {"path": "family", "eq": "vla"},
                },
                {"name": "lr", "type": "log_uniform", "low": 1e-4, "high": 1e-2},
                {"name": "batch", "type": "int", "low": 1, "high": 40, "step": 1},
            ],
            "constraints": [{"prune_when": {"path": "family", "eq": "act"}}],
        }
        sampler = SweepSampler.from_sweep(doc)
        for index in range(8):
            trial = sampler.ask()[0]
            self.assertEqual(trial.overrides["family"], "vla")
            self.assertIn(trial.overrides["rank"], (4, 8))
            self.assertGreaterEqual(trial.overrides["lr"], 1e-4)
            self.assertLessEqual(trial.overrides["lr"], 1e-2)
            self.assertIsInstance(trial.overrides["batch"], int)
            self.assertGreaterEqual(trial.overrides["batch"], 1)
            self.assertLessEqual(trial.overrides["batch"], 40)
            self.assertIn("rank", trial.overrides)
            sampler.tell(trial.index, {"loss": 1.0 / (index + 1)})


class TestTellErrors(unittest.TestCase):
    def test_rejects_bad_reports(self) -> None:
        sampler = SweepSampler.from_sweep(_bench("tpe", candidates=4))
        trial = sampler.ask()[0]
        with self.assertRaises(SweepError):
            sampler.tell(trial.index, {"loss": math.nan})
        with self.assertRaises(SweepError):
            sampler.tell(trial.index, {"loss": True})
        with self.assertRaises(SweepError):
            sampler.tell(trial.index, {})
        with self.assertRaises(SweepError):
            sampler.tell(99, {"loss": 1})
        sampler.tell(trial.index, {"loss": 0.2})
        with self.assertRaises(SweepError) as caught:
            sampler.tell(trial.index, {"loss": 0.1})
        self.assertIn("already", str(caught.exception))
        with self.assertRaises(SweepError):
            sampler.ask(-1)


class TestCliStartupPlan(unittest.TestCase):
    def test_plan_prints_the_startup_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpe.json"
            path.write_text(json.dumps(_bench("tpe")), encoding="utf-8")
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["sweep", "plan", str(path)])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("strategy: tpe", text)
        self.assertIn("trials: 10", text)
        self.assertIn("startup batch", text)


if __name__ == "__main__":
    unittest.main()
