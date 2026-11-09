"""ohho.sweep/v1 schema, validator, and trial expander."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path

from ohho.cli import build_parser, main
from ohho.train.sweep import (
    ESTIMATE_NOTE,
    MAX_TRIALS,
    SCHEMA_ID,
    SweepError,
    expand,
    format_plan,
    load_schema,
    parse_sweep_text,
    plan_as_dict,
    validate_sweep,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "ohho" / "train" / "schemas" / "examples"


def _grid() -> dict:
    return json.loads((_EXAMPLES / "grid_2x3x4.json").read_text(encoding="utf-8"))


class TestSchemaPackage(unittest.TestCase):
    def test_schema_identity_and_refs(self) -> None:
        schema = load_schema()
        self.assertEqual(schema["title"], SCHEMA_ID)
        self.assertEqual(
            schema["$id"],
            "https://ohho-robotics.com/schemas/ohho.sweep.v1.json",
        )
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertIn("Semantic rules", schema["description"])

    def test_package_resource_matches_file(self) -> None:
        resource = files("ohho.train").joinpath("schemas", "sweep.v1.json")
        packaged = resource.read_text(encoding="utf-8")
        on_disk = (_ROOT / "ohho" / "train" / "schemas" / "sweep.v1.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(packaged, on_disk)
        self.assertIn("train/schemas/*.json", (_ROOT / "pyproject.toml").read_text())

    def test_examples_validate(self) -> None:
        names = (
            "grid_2x3x4.json",
            "conditional.json",
            "constrained.json",
            "tpe.json",
            "multi.json",
            "dataset_3x2.json",
        )
        for name in names:
            doc = json.loads((_EXAMPLES / name).read_text(encoding="utf-8"))
            spec = validate_sweep(doc)
            self.assertEqual(spec.schema, SCHEMA_ID)


class TestValidator(unittest.TestCase):
    def test_rejects_bad_documents(self) -> None:
        samples = [
            {},
            {"schema": "ohho.sweep/v0"},
            {"schema": SCHEMA_ID, "extra": 1},
            {"schema": SCHEMA_ID, "parallelism": 0},
            {"schema": SCHEMA_ID, "seed": -1},
            {"schema": SCHEMA_ID, "strategy": "bayes"},
            {
                "schema": SCHEMA_ID,
                "dimensions": [{"name": "model", "type": "categorical"}],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [{"name": "model", "type": "categorical", "values": []}],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "model",
                        "type": "categorical",
                        "values": ["a"],
                        "low": 1,
                    }
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "lr", "type": "log_uniform", "low": 0, "high": 1, "n": 2}
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "n", "type": "int", "low": 3, "high": 1, "step": 1}
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "n", "type": "int", "low": 0, "high": 1, "step": 0}
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "bad path", "type": "categorical", "values": ["a"]}
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "x",
                        "path": "has space",
                        "type": "categorical",
                        "values": ["a"],
                    }
                ],
            },
            {"schema": SCHEMA_ID, "constraints": [{}]},
            {
                "schema": SCHEMA_ID,
                "constraints": [
                    {
                        "prune_when": {"path": "x", "eq": 1},
                        "keep_when": {"path": "x", "eq": 1},
                    }
                ],
            },
            {
                "schema": SCHEMA_ID,
                "objectives": [
                    {"name": "a", "direction": "minimize", "primary": True},
                    {"name": "b", "direction": "maximize", "primary": True},
                ],
            },
            {
                "schema": SCHEMA_ID,
                "strategy": "random",
                "dimensions": [
                    {"name": "x", "type": "categorical", "values": ["a", "b"]}
                ],
            },
            {
                "schema": SCHEMA_ID,
                "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
            },
            {"schema": SCHEMA_ID, "early_stopping": {}},
            {"schema": SCHEMA_ID, "early_stopping": {"metric": "loss", "mode": "down"}},
            {"schema": SCHEMA_ID, "estimate": {"gpu_hours_per_trial": -1}},
            "not-a-mapping",
        ]
        for doc in samples:
            with self.subTest(doc=doc):
                with self.assertRaises(SweepError):
                    validate_sweep(doc)  # type: ignore[arg-type]

    def test_duplicate_name_and_path(self) -> None:
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "dimensions": [
                        {"name": "model", "type": "categorical", "values": ["a"]},
                        {"name": "model", "type": "categorical", "values": ["b"]},
                    ],
                }
            )
        self.assertIn("duplicate dimension name", str(caught.exception))
        with self.assertRaises(SweepError) as caught:
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "dimensions": [
                        {
                            "name": "a",
                            "path": "model",
                            "type": "categorical",
                            "values": ["a"],
                        },
                        {
                            "name": "b",
                            "path": "model",
                            "type": "categorical",
                            "values": ["b"],
                        },
                    ],
                }
            )
        self.assertIn("duplicate dimension path", str(caught.exception))

    def test_int_n_must_be_strictly_increasing(self) -> None:
        with self.assertRaises(SweepError):
            validate_sweep(
                {
                    "schema": SCHEMA_ID,
                    "dimensions": [
                        {"name": "n", "type": "int", "low": 0, "high": 1, "n": 3}
                    ],
                }
            )

    def test_defaults(self) -> None:
        spec = validate_sweep(
            {"schema": SCHEMA_ID, "early_stopping": {"metric": "loss"}}
        )
        self.assertEqual(spec.seed, 0)
        self.assertEqual(spec.strategy, "grid")
        self.assertEqual(spec.parallelism, 1)
        self.assertEqual(spec.recipe, {})
        self.assertEqual(spec.dimensions, ())
        self.assertEqual(
            spec.early_stopping,
            {"metric": "loss", "patience": 1, "min_delta": 0.0, "mode": "min"},
        )

    def test_duplicate_json_key(self) -> None:
        with self.assertRaises(SweepError):
            parse_sweep_text('{"schema": "ohho.sweep/v1", "schema": "ohho.sweep/v1"}')


class TestExpander(unittest.TestCase):
    def test_empty_dimensions_are_one_base_trial(self) -> None:
        for doc in (
            {"schema": SCHEMA_ID, "recipe": {"policy": "act"}},
            {"schema": SCHEMA_ID, "recipe": {"policy": "act"}, "dimensions": []},
        ):
            plan = expand(doc)
            self.assertEqual(len(plan.trials), 1)
            self.assertEqual(plan.trials[0].sets, ())
            self.assertEqual(plan.trials[0].overrides, {})
            self.assertEqual(plan.trials[0].recipe, {"policy": "act"})
            self.assertEqual(plan.estimated_gpu_hours, 0.0)

    def test_grid_2x3x4_resolves_twenty_four_recipes(self) -> None:
        plan = expand(_grid())
        self.assertEqual(len(plan.trials), 24)
        self.assertEqual(plan.parallelism, 4)
        self.assertEqual(plan.estimated_gpu_hours, 15.0)
        self.assertEqual(plan.estimate_note, ESTIMATE_NOTE)
        self.assertEqual(plan.objectives[0]["name"], "val_loss")
        self.assertEqual(plan.early_stopping["patience"], 5)
        models = ["small", "large"]
        batches = [8, 16, 32]
        lrs = [
            trial.overrides["optimizer.lr"]
            for trial in plan.trials
            if trial.overrides["model"] == "small"
            and trial.overrides["train.batch_size"] == 8
        ]
        self.assertEqual(len(lrs), 4)
        self.assertAlmostEqual(lrs[0], 1e-5)
        self.assertAlmostEqual(lrs[-1], 1e-3)
        self.assertEqual(lrs, sorted(lrs))
        ratios = [lrs[i + 1] / lrs[i] for i in range(3)]
        self.assertAlmostEqual(ratios[0], ratios[1])
        self.assertAlmostEqual(ratios[1], ratios[2])
        index = 0
        for model in models:
            for batch in batches:
                for lr in lrs:
                    trial = plan.trials[index]
                    self.assertEqual(trial.index, index)
                    self.assertEqual(trial.overrides["model"], model)
                    self.assertEqual(trial.overrides["train.batch_size"], batch)
                    self.assertAlmostEqual(trial.overrides["optimizer.lr"], lr)
                    self.assertEqual(
                        trial.recipe["optimizer"],
                        {"lr": trial.overrides["optimizer.lr"], "weight_decay": 0.0},
                    )
                    self.assertEqual(trial.recipe["task"], "push")
                    self.assertEqual(trial.recipe["model"], model)
                    self.assertEqual(trial.recipe["train"], {"batch_size": batch})
                    expected = 1.0 if model == "large" else 0.25
                    self.assertEqual(trial.estimated_gpu_hours, expected)
                    self.assertEqual(
                        trial.sets[0],
                        f"model={json.dumps(model)}",
                    )
                    index += 1
        self.assertEqual(index, 24)

    def test_grid_order_ignores_seed(self) -> None:
        left = expand(_grid())
        doc = _grid()
        doc["seed"] = 99
        right = expand(doc)
        self.assertEqual(
            [trial.sets for trial in left.trials],
            [trial.sets for trial in right.trials],
        )
        again = expand(_grid())
        self.assertEqual(plan_as_dict(left), plan_as_dict(again))

    def test_conditional_params_only_on_applicable_trials(self) -> None:
        plan = expand(
            json.loads((_EXAMPLES / "conditional.json").read_text(encoding="utf-8"))
        )
        self.assertEqual(len(plan.trials), 3)
        act = plan.trials[0]
        self.assertEqual(act.overrides, {"family": "act"})
        self.assertNotIn("model.lora.rank", act.overrides)
        self.assertNotIn("model", act.recipe)
        self.assertEqual(
            [trial.overrides["model.lora.rank"] for trial in plan.trials[1:]],
            [4, 8],
        )
        for trial in plan.trials[1:]:
            self.assertEqual(trial.overrides["family"], "vla")
            self.assertEqual(
                trial.recipe["model"]["lora"]["rank"],
                trial.overrides["model.lora.rank"],
            )
            self.assertIn("model.lora.rank=", " ".join(trial.sets))

    def test_when_reads_base_recipe(self) -> None:
        present = expand(
            {
                "schema": SCHEMA_ID,
                "recipe": {"family": "vla"},
                "dimensions": [
                    {
                        "name": "lora_rank",
                        "path": "model.lora.rank",
                        "type": "int",
                        "values": [4],
                        "when": {"path": "family", "eq": "vla"},
                    }
                ],
            }
        )
        absent = expand(
            {
                "schema": SCHEMA_ID,
                "recipe": {"family": "act"},
                "dimensions": [
                    {
                        "name": "lora_rank",
                        "path": "model.lora.rank",
                        "type": "int",
                        "values": [4],
                        "when": {"path": "family", "eq": "vla"},
                    }
                ],
            }
        )
        self.assertEqual(present.trials[0].overrides, {"model.lora.rank": 4})
        self.assertEqual(len(absent.trials), 1)
        self.assertEqual(absent.trials[0].overrides, {})
        self.assertNotIn("model", absent.trials[0].recipe)

    def test_constraints_prune_invalid_combinations(self) -> None:
        plan = expand(
            json.loads((_EXAMPLES / "constrained.json").read_text(encoding="utf-8"))
        )
        pairs = [
            (trial.overrides["precision"], trial.overrides["model"])
            for trial in plan.trials
        ]
        self.assertEqual(
            pairs, [("fp16", "small"), ("fp16", "large"), ("int8", "small")]
        )
        self.assertNotIn(("int8", "large"), pairs)

    def test_keep_when_and_not(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "model",
                        "type": "categorical",
                        "values": ["small", "large"],
                    }
                ],
                "constraints": [
                    {"keep_when": {"not": {"path": "model", "eq": "large"}}}
                ],
            }
        )
        self.assertEqual([trial.overrides["model"] for trial in plan.trials], ["small"])

    def test_constraint_on_nested_recipe_field(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "recipe": {"model": {"size": "large"}},
                "dimensions": [
                    {
                        "name": "precision",
                        "type": "categorical",
                        "values": ["fp16", "int8"],
                    }
                ],
                "constraints": [
                    {
                        "prune_when": {
                            "all": [
                                {"path": "model.size", "eq": "large"},
                                {"path": "precision", "eq": "int8"},
                            ]
                        }
                    }
                ],
            }
        )
        self.assertEqual(
            [trial.overrides["precision"] for trial in plan.trials],
            ["fp16"],
        )

    def test_constraints_can_remove_every_trial(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "model", "type": "categorical", "values": ["a"]}
                ],
                "constraints": [{"prune_when": {"path": "model", "eq": "a"}}],
            }
        )
        self.assertEqual(plan.trials, ())
        self.assertEqual(plan.estimated_gpu_hours, 0.0)

    def test_bool_equality_is_not_numeric(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "flag", "type": "categorical", "values": [True, False]},
                    {
                        "name": "extra",
                        "type": "categorical",
                        "values": ["on"],
                        "when": {"path": "flag", "eq": True},
                    },
                ],
            }
        )
        self.assertIn("extra", plan.trials[0].overrides)
        self.assertNotIn("extra", plan.trials[1].overrides)
        missed = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "flag", "type": "categorical", "values": [True]},
                    {
                        "name": "extra",
                        "type": "categorical",
                        "values": ["on"],
                        "when": {"path": "flag", "eq": 1},
                    },
                ],
            }
        )
        self.assertNotIn("extra", missed.trials[0].overrides)

    def test_ranges(self) -> None:
        ints = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "n", "type": "int", "low": 0, "high": 10, "step": 5}
                ],
            }
        )
        self.assertEqual([trial.overrides["n"] for trial in ints.trials], [0, 5, 10])
        floats = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "x", "type": "float", "low": 0, "high": 1, "n": 5}
                ],
            }
        )
        self.assertEqual(
            [trial.overrides["x"] for trial in floats.trials],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        one = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "lr",
                        "type": "log_uniform",
                        "low": 1e-4,
                        "high": 1e-2,
                        "n": 1,
                    }
                ],
            }
        )
        self.assertAlmostEqual(one.trials[0].overrides["lr"], 1e-3)
        midpoint = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "n", "type": "int", "low": 0, "high": 10, "n": 1}
                ],
            }
        )
        self.assertEqual(midpoint.trials[0].overrides["n"], 5)

    def test_cannot_override_through_a_scalar(self) -> None:
        with self.assertRaises(SweepError) as caught:
            expand(
                {
                    "schema": SCHEMA_ID,
                    "recipe": {"model": "flat"},
                    "dimensions": [
                        {
                            "name": "rank",
                            "path": "model.lora.rank",
                            "type": "int",
                            "values": [4],
                        }
                    ],
                }
            )
        self.assertIn("not a mapping", str(caught.exception))

    def test_budget_caps(self) -> None:
        doc = _grid()
        doc["budget"] = {"max_trials": 2}
        doc["estimate"] = {"gpu_hours_per_trial": 1}
        prefix = expand(doc)
        self.assertEqual(len(prefix.trials), 2)
        self.assertEqual([trial.index for trial in prefix.trials], [0, 1])
        self.assertEqual(prefix.trials[0].sets, expand(_grid()).trials[0].sets)
        doc["budget"] = {"max_gpu_hours": 2.5}
        hours = expand(doc)
        self.assertEqual(len(hours.trials), 2)
        self.assertEqual(hours.estimated_gpu_hours, 2.0)
        doc["budget"] = {"max_trials": 0}
        self.assertEqual(expand(doc).trials, ())
        blocked = expand(
            {
                "schema": SCHEMA_ID,
                "budget": {"max_gpu_hours": 0},
                "estimate": {"gpu_hours_per_trial": 1},
                "dimensions": [
                    {"name": "model", "type": "categorical", "values": ["a", "b"]}
                ],
            }
        )
        self.assertEqual(blocked.trials, ())
        free = expand(
            {
                "schema": SCHEMA_ID,
                "budget": {"max_gpu_hours": 0},
                "estimate": {"gpu_hours_per_trial": 0},
                "dimensions": [
                    {"name": "model", "type": "categorical", "values": ["a", "b"]}
                ],
            }
        )
        self.assertEqual(len(free.trials), 2)

    def test_estimate_first_match_wins(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "estimate": {
                    "gpu_hours_per_trial": 9,
                    "by_value": [
                        {"when": {"path": "model", "eq": "large"}, "gpu_hours": 4},
                        {
                            "when": {"path": "model", "in": ["large", "small"]},
                            "gpu_hours": 1,
                        },
                    ],
                },
                "dimensions": [
                    {
                        "name": "model",
                        "type": "categorical",
                        "values": ["large", "small"],
                    }
                ],
            }
        )
        self.assertEqual(
            [trial.estimated_gpu_hours for trial in plan.trials],
            [4.0, 1.0],
        )

    def test_random_seed_is_deterministic(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "random",
            "seed": 1,
            "budget": {"max_trials": 5},
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
        first = expand(doc)
        second = expand(doc)
        self.assertEqual(plan_as_dict(first), plan_as_dict(second))
        for trial in first.trials:
            self.assertGreaterEqual(trial.overrides["lr"], 1e-5)
            self.assertLessEqual(trial.overrides["lr"], 1e-2)
            self.assertGreaterEqual(trial.overrides["scale"], 0.0)
            self.assertLessEqual(trial.overrides["scale"], 1.0)
        doc["seed"] = 2
        other = expand(doc)
        self.assertNotEqual(
            [trial.overrides for trial in first.trials],
            [trial.overrides for trial in other.trials],
        )

    def test_random_rejects_impossible_constraints(self) -> None:
        with self.assertRaises(SweepError) as caught:
            expand(
                {
                    "schema": SCHEMA_ID,
                    "strategy": "random",
                    "seed": 0,
                    "budget": {"max_trials": 3},
                    "dimensions": [
                        {"name": "p", "type": "categorical", "values": ["a", "b"]}
                    ],
                    "constraints": [{"prune_when": {"path": "p", "in": ["a", "b"]}}],
                }
            )
        self.assertIn("could not sample", str(caught.exception))

    def test_random_without_dimensions_is_one_trial(self) -> None:
        plan = expand(
            {"schema": SCHEMA_ID, "strategy": "random", "recipe": {"policy": "act"}}
        )
        self.assertEqual(len(plan.trials), 1)
        self.assertEqual(plan.trials[0].recipe, {"policy": "act"})

    def test_missing_path_comparisons_are_false(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "model", "type": "categorical", "values": ["a"]}
                ],
                "constraints": [
                    {"prune_when": {"path": "missing", "eq": "a"}},
                    {"keep_when": {"path": "missing", "exists": False}},
                ],
            }
        )
        self.assertEqual(len(plan.trials), 1)

    def test_grid_cap(self) -> None:
        import ohho.train.sweep as sweep

        previous = sweep.MAX_TRIALS
        sweep.MAX_TRIALS = 3
        try:
            with self.assertRaises(SweepError):
                sweep.expand(
                    {
                        "schema": SCHEMA_ID,
                        "dimensions": [
                            {"name": "a", "type": "categorical", "values": ["a", "b"]},
                            {"name": "b", "type": "categorical", "values": ["a", "b"]},
                        ],
                    }
                )
            plan = sweep.expand(
                {
                    "schema": SCHEMA_ID,
                    "dimensions": [
                        {
                            "name": "family",
                            "type": "categorical",
                            "values": ["act", "vla"],
                        },
                        {
                            "name": "lora_rank",
                            "path": "model.lora.rank",
                            "type": "int",
                            "values": [4, 8],
                            "when": {"path": "family", "eq": "vla"},
                        },
                    ],
                }
            )
            self.assertEqual(len(plan.trials), 3)
        finally:
            sweep.MAX_TRIALS = previous
        self.assertGreater(MAX_TRIALS, 3)

    def test_numeric_condition(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [{"name": "batch", "type": "int", "values": [4, 8, 32]}],
                "constraints": [{"prune_when": {"path": "batch", "gte": 16}}],
            }
        )
        self.assertEqual([trial.overrides["batch"] for trial in plan.trials], [4, 8])


class TestYaml(unittest.TestCase):
    def test_yaml_grid_matches_json_grid(self) -> None:
        text = (_EXAMPLES / "grid_2x3x4.yaml").read_text(encoding="utf-8")
        parsed = parse_sweep_text(text)
        self.assertEqual(expand(parsed).trials[0].sets, expand(_grid()).trials[0].sets)
        self.assertEqual(
            plan_as_dict(load_and_expand_path("grid_2x3x4.yaml")),
            plan_as_dict(load_and_expand_path("grid_2x3x4.json")),
        )

    def test_flow_comment_bom_and_errors(self) -> None:
        doc = parse_sweep_text(
            "\ufeffschema: ohho.sweep/v1\n"
            "dimensions:\n"
            "  - name: flag\n"
            "    type: categorical\n"
            "    values: [true, false] # flags\n"
            "    when: {path: flag, eq: true}\n"
        )
        self.assertEqual(doc["dimensions"][0]["values"], [True, False])
        self.assertEqual(doc["dimensions"][0]["when"], {"path": "flag", "eq": True})
        with self.assertRaises(SweepError):
            parse_sweep_text("schema: ohho.sweep/v1\n\tseed: 1\n")
        with self.assertRaises(SweepError):
            parse_sweep_text("schema: ohho.sweep/v1\nschema: ohho.sweep/v1\n")
        with self.assertRaises(SweepError):
            parse_sweep_text("")
        quoted = parse_sweep_text('schema: "ohho.sweep/v1"\nseed: 2\n')
        self.assertEqual(quoted["schema"], SCHEMA_ID)
        self.assertEqual(quoted["seed"], 2)


def load_and_expand_path(name: str):
    return expand(_EXAMPLES / name)


class TestCli(unittest.TestCase):
    def test_parser(self) -> None:
        args = build_parser().parse_args(["sweep", "plan", "sweep.yaml", "--json"])
        self.assertEqual(args.command, "sweep")
        self.assertEqual(args.sweep_command, "plan")
        self.assertEqual(args.spec, "sweep.yaml")
        self.assertTrue(args.json)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["sweep"])

    def test_plan_text_and_json(self) -> None:
        path = _EXAMPLES / "grid_2x3x4.yaml"
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "plan", str(path)])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("trials: 24", text)
        self.assertIn("estimated_gpu_hours: 15", text)
        self.assertIn("estimate:", text)
        self.assertIn(ESTIMATE_NOTE, text)
        self.assertIn('model="small"', text)
        self.assertEqual(text.count("\ntrial: "), 24)
        self.assertIn('"task": "push"', text)

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "plan", str(_EXAMPLES / "grid_2x3x4.json"), "--json"])
        self.assertEqual(code, 0, err.getvalue())
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["trials"], 24)
        self.assertEqual(len(payload["trial_recipes"]), 24)
        self.assertIn("estimate", payload["estimate"])
        self.assertEqual(payload["trial_recipes"][0]["recipe"]["task"], "push")
        self.assertEqual(
            payload["trial_recipes"][0]["recipe"]["optimizer"]["weight_decay"],
            0.0,
        )

    def test_plan_errors(self) -> None:
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sweep", "plan", str(_EXAMPLES / "missing.yaml")])
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sweep", "plan", str(_EXAMPLES)])
        self.assertEqual(code, 1)
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "bad.json"
            invalid.write_text('{"schema": "ohho.sweep/v0"}', encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = main(["sweep", "plan", str(invalid)])
            self.assertEqual(code, 1)
            self.assertIn("error:", err.getvalue())

    def test_format_matches_expand(self) -> None:
        plan = expand(_grid())
        text = format_plan(plan)
        self.assertTrue(text.endswith("\n"))
        self.assertIn("trials: 24", text)
        self.assertAlmostEqual(
            math.fsum(trial.estimated_gpu_hours for trial in plan.trials), 15.0
        )


if __name__ == "__main__":
    unittest.main()
