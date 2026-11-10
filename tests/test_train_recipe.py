"""Tests for ohho.recipe/v1 and the ohho train command."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main as cli_main
from ohho.sim_loop import record_sim_episodes
from ohho.train.builtin_recipes import ACT
from ohho.train.recipe_registry import (
    RecipeRegistryError,
    TrainResult,
    ensure_loaded,
    family_ids,
    get_family,
    register,
    unregister,
)
from ohho.train.recipe_schema import (
    SCHEMA_ID,
    RecipeConfigError,
    apply_overrides,
    example_path,
    iter_examples,
    load_recipe_file,
    parse_override_value,
    schema_document,
    schema_path,
    validate_recipe,
)
from ohho.train.run_cli import (
    EXIT_DATASET,
    EXIT_INVALID,
    EXIT_OK,
    EXIT_TRAIN_FAILED,
    EXIT_UNAVAILABLE,
    EXIT_USAGE,
    STREAM,
    UNAVAILABLE_MESSAGE,
    execute,
)
from ohho.train.yaml_subset import UnsupportedYaml, YamlError, loads


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _run(path: str | Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _events_of(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


class TestYamlSubset(unittest.TestCase):
    def test_examples_parse(self) -> None:
        diffusion = loads(example_path("diffusion").read_text(encoding="utf-8"))
        self.assertEqual(diffusion["family"], "diffusion")
        self.assertEqual(diffusion["compression_hooks"], [])
        self.assertEqual(diffusion["eval_hooks"][0]["name"], "sim_success")
        self.assertEqual(diffusion["eval_hooks"][0]["params"]["episodes"], 5)
        smolvla = loads(example_path("smolvla").read_text(encoding="utf-8"))
        self.assertEqual(smolvla["dataset"]["version_id"], "dataset_version_example")
        self.assertNotIn("path", smolvla["dataset"])
        self.assertEqual(loads("lr: 1.0e-4\n")["lr"], 1.0e-4)

    def test_comments_bools_and_colons_in_values(self) -> None:
        loaded = loads(
            "\n".join(
                [
                    "name: tiny # comment",
                    "flag: true",
                    "word: yes",
                    "uri: local:act-init",
                    'quoted: "null"',
                ]
            )
        )
        self.assertEqual(loaded["name"], "tiny")
        self.assertIs(loaded["flag"], True)
        self.assertEqual(loaded["word"], "yes")
        self.assertEqual(loaded["uri"], "local:act-init")
        self.assertEqual(loaded["quoted"], "null")

    def test_duplicate_key_and_tabs_fail(self) -> None:
        with self.assertRaises(YamlError):
            loads("a: 1\na: 2\n")
        with self.assertRaises(YamlError):
            loads("\ta: 1\n")

    def test_unquoted_hash_in_tokens(self) -> None:
        doc = "\n".join(
            [
                "repo: https://example.com/model#rev",
                "dataset: datasets/run#2",
                "with_comment: https://example.com/model#rev # pinned commit",
            ]
        )
        loaded = loads(doc)
        self.assertEqual(loaded["repo"], "https://example.com/model#rev")
        self.assertEqual(loaded["dataset"], "datasets/run#2")
        self.assertEqual(loaded["with_comment"], "https://example.com/model#rev")

    def test_pyyaml_fallback_leading_document_separator_and_dash_only_items(
        self,
    ) -> None:
        doc_sep = "\n".join(
            [
                "---",
                "schema: ohho.recipe/v1",
                "family: act",
            ]
        )
        loaded_sep = loads(doc_sep)
        self.assertEqual(loaded_sep["schema"], "ohho.recipe/v1")
        self.assertEqual(loaded_sep["family"], "act")

        doc_dash = "\n".join(
            [
                "items:",
                "  -",
                "    name: first",
                "  -",
                "    name: second",
            ]
        )
        loaded_dash = loads(doc_dash)
        self.assertEqual(
            loaded_dash["items"],
            [{"name": "first"}, {"name": "second"}],
        )

    def test_pyyaml_fallback_missing_pyyaml(self) -> None:
        with patch.dict("sys.modules", {"yaml": None}):
            with self.assertRaises(UnsupportedYaml) as caught:
                loads("---\nfamily: act\n")
            self.assertIn("ohho-os[yaml]", str(caught.exception))


class TestRecipeSchema(unittest.TestCase):
    def test_schema_is_package_data(self) -> None:
        document = schema_document()
        self.assertEqual(document["title"], SCHEMA_ID)
        self.assertEqual(
            document["$id"],
            "https://ohho-robotics.com/schemas/ohho.recipe/v1.json",
        )
        self.assertEqual(
            document["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertTrue(schema_path().is_file())
        required = {
            "family",
            "base_weights",
            "dataset",
            "hyperparameters",
            "precision",
            "compression_hooks",
            "eval_hooks",
            "output",
        }
        self.assertTrue(required <= set(document["properties"]))
        root = Path(__file__).resolve().parents[1]
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("train/schema/*.json", project)
        self.assertIn("train/examples/*.yaml", project)
        self.assertIn('[project.entry-points."ohho.recipes"]', project)

    def test_each_family_example_validates(self) -> None:
        names = {path.stem for path in iter_examples()}
        self.assertEqual(names, {"act", "diffusion", "smolvla", "openvla", "ppo"})
        for path in iter_examples():
            with self.subTest(path=path.name):
                recipe = load_recipe_file(path)
                self.assertEqual(recipe["schema"], SCHEMA_ID)
                self.assertEqual(recipe["family"], path.stem)
                self.assertEqual(validate_recipe(recipe), [])

    def test_schema_rejects_bad_documents(self) -> None:
        recipe = load_recipe_file(example_path("act"))
        extra = dict(recipe)
        extra["note"] = "nope"
        self.assertTrue(any("note" in error for error in validate_recipe(extra)))
        both = json.loads(json.dumps(recipe))
        both["dataset"] = {"path": "datasets/sim-loop", "version_id": "dv"}
        errors = validate_recipe(both)
        self.assertIn("$.dataset: set exactly one of path or version_id", errors)
        neither = json.loads(json.dumps(recipe))
        neither["dataset"] = {}
        self.assertIn(
            "$.dataset: set exactly one of path or version_id",
            validate_recipe(neither),
        )
        bad_gamma = load_recipe_file(example_path("ppo"))
        bad_gamma["hyperparameters"]["gamma"] = 1.5
        self.assertTrue(any("gamma" in error for error in validate_recipe(bad_gamma)))
        flagged = json.loads(json.dumps(recipe))
        flagged["hyperparameters"]["batch_size"] = True
        self.assertTrue(validate_recipe(flagged))

    def test_json_recipe_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA_ID,
                        "family": "ppo",
                        "dataset": {"path": "datasets/sim-loop"},
                        "output": {"dir": "checkpoints/ppo"},
                    }
                ),
                encoding="utf-8",
            )
            recipe = load_recipe_file(path)
            self.assertEqual(recipe["family"], "ppo")
            self.assertEqual(recipe["precision"], "fp32")


class TestOverrides(unittest.TestCase):
    def test_dotted_override_replaces_nested_values(self) -> None:
        original = load_recipe_file(example_path("diffusion"))
        updated = apply_overrides(
            original,
            [
                "eval_hooks.0.params.episodes=9",
                "hyperparameters.lr=0.2",
                "dataset.path=null",
                "dataset.version_id=dv_123",
            ],
        )
        self.assertEqual(original["hyperparameters"]["lr"], 0.0001)
        self.assertEqual(updated["eval_hooks"][0]["params"]["episodes"], 9)
        self.assertEqual(updated["hyperparameters"]["lr"], 0.2)
        self.assertNotIn("path", updated["dataset"])
        self.assertEqual(updated["dataset"]["version_id"], "dv_123")
        self.assertEqual(validate_recipe(updated), [])

    def test_override_value_forms(self) -> None:
        self.assertIsNone(parse_override_value("null"))
        self.assertEqual(parse_override_value('"null"'), "null")
        self.assertEqual(parse_override_value("1e-4"), 1e-4)
        self.assertIs(parse_override_value("true"), True)
        self.assertEqual(parse_override_value('["a"]'), ["a"])
        with self.assertRaises(RecipeConfigError):
            apply_overrides({}, ["missing-equals"])
        with self.assertRaises(RecipeConfigError):
            apply_overrides(
                {"hyperparameters": {"lr": 0.1}}, ["hyperparameters.lr.extra=1"]
            )


class TestRegistry(unittest.TestCase):
    def test_builtins_and_duplicate_id(self) -> None:
        ensure_loaded()
        self.assertTrue(
            {"act", "diffusion", "smolvla", "openvla", "ppo"} <= set(family_ids())
        )
        self.assertTrue(get_family("act").available())
        for family in ("diffusion", "smolvla", "openvla", "ppo"):
            self.assertFalse(get_family(family).available())
        self.assertIs(get_family("act"), ACT)
        register(ACT, source="again")

        class Other:
            id = "act"

            def available(self) -> bool:
                return True

            def run(self, recipe, emit):
                del recipe, emit
                raise AssertionError("replacement act must not run")

        with self.assertRaises(RecipeRegistryError):
            register(Other(), source="test")
        self.assertIs(get_family("act"), ACT)

    def test_plugin_family_runs(self) -> None:
        class Custom:
            id = "custom_policy"

            def available(self) -> bool:
                return True

            def run(self, recipe, emit):
                out = Path(recipe["output"]["dir"])
                out.mkdir(parents=True, exist_ok=True)
                metrics = out / "metrics.json"
                metrics.write_text(
                    json.dumps({"final_loss": recipe["hyperparameters"]["lr"]}),
                    encoding="utf-8",
                )
                emit({"event": "train_start", "family": self.id, "device": "cpu"})
                return TrainResult(str(out), str(metrics), "cpu", False)

        register(Custom(), source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                recipe = Path(tmp) / "custom.yaml"
                out = Path(tmp) / "out"
                recipe.write_text(
                    "\n".join(
                        [
                            "schema: ohho.recipe/v1",
                            "family: custom_policy",
                            "dataset:",
                            "  path: datasets/sim-loop",
                            "hyperparameters:",
                            "  lr: 0.1",
                            "output:",
                            "  dir: checkpoints/custom",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                code, events, err = _run(
                    recipe, [f"output.dir={out}", "hyperparameters.lr=0.2"]
                )
                self.assertEqual(code, EXIT_OK, err)
                manifest = json.loads(
                    (out / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["family"], "custom_policy")
                self.assertEqual(manifest["status"], "succeeded")
                self.assertEqual(manifest["recipe"]["hyperparameters"]["lr"], 0.2)
                self.assertFalse(manifest["base_weights_loaded"])
                self.assertTrue(_events_of(events, "train_start"))
                self.assertEqual(_events_of(events, "done")[-1]["exit_code"], 0)
                self.assertTrue(all(event["stream"] == STREAM for event in events))
        finally:
            unregister("custom_policy")


class TestTrainCommand(unittest.TestCase):
    def test_usage_exit_is_argparse(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli_main(["train"])
        self.assertEqual(caught.exception.code, EXIT_USAGE)

    def test_unavailable_families_do_not_train(self) -> None:
        for family in ("diffusion", "smolvla", "openvla", "ppo"):
            with self.subTest(family=family):
                code, events, err = _run(example_path(family))
                self.assertEqual(code, EXIT_UNAVAILABLE, err)
                self.assertIn(UNAVAILABLE_MESSAGE, err)
                self.assertEqual(events[0]["message"], UNAVAILABLE_MESSAGE)
                self.assertEqual(events[0]["family"], family)
                self.assertEqual(events[0]["exit_code"], EXIT_UNAVAILABLE)
                self.assertFalse(_events_of(events, "train_start"))

    def test_cli_set_switches_family_before_dataset_check(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main(
                [
                    "train",
                    str(example_path("act")),
                    "--set",
                    "family=diffusion",
                    "--set",
                    "dataset.path=null",
                    "--set",
                    "dataset.version_id=not-resolved",
                ]
            )
        self.assertEqual(code, EXIT_UNAVAILABLE, stderr.getvalue())
        event = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(event["message"], UNAVAILABLE_MESSAGE)
        self.assertEqual(event["family"], "diffusion")

    def test_invalid_config_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            missing = Path(tmp) / "missing-dataset"
            cases = [
                (example_path("ppo"), ["hyperparameters.gamma=1.5"], "gamma"),
                (
                    example_path("act"),
                    ["precision=bf16", f"output.dir={out}"],
                    "precision",
                ),
                (
                    example_path("act"),
                    ["hyperparameters.horizon=8", f"output.dir={out}"],
                    "unknown hyperparameters",
                ),
                (
                    example_path("act"),
                    ["hyperparameters.mock=true", f"output.dir={out}"],
                    "fake trainer",
                ),
                (
                    example_path("act"),
                    [
                        "hyperparameters.d_model=30",
                        "hyperparameters.nhead=4",
                        f"output.dir={out}",
                    ],
                    "divisible",
                ),
            ]
            for path, overrides, needle in cases:
                with self.subTest(needle=needle):
                    code, events, err = _run(path, overrides)
                    self.assertEqual(code, EXIT_INVALID, err)
                    self.assertIn(needle, err)
                    self.assertFalse(_events_of(events, "train_start"))
                    self.assertFalse(out.exists())
            code, events, err = _run(
                example_path("act"),
                [f"dataset.path={missing}", f"output.dir={out}"],
            )
            self.assertEqual(code, EXIT_DATASET, err)
            self.assertIn("does not exist", err)
            self.assertFalse(_events_of(events, "train_start"))
            self.assertFalse(out.exists())
            code, events, err = _run(
                example_path("act"),
                [
                    "dataset.path=null",
                    "dataset.version_id=dv_123",
                    f"output.dir={out}",
                ],
            )
            self.assertEqual(code, EXIT_DATASET, err)
            self.assertIn("not resolved", err)
            self.assertFalse(_events_of(events, "train_start"))

    def test_parse_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("schema: [\n", encoding="utf-8")
            code, events, err = _run(bad)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertTrue(events)
            self.assertEqual(events[0]["event"], "error")
            missing = Path(tmp) / "nope.yaml"
            code, _events, err = _run(missing)
            self.assertEqual(code, EXIT_INVALID)
            self.assertIn("not found", err)


class _TorchStub:
    def manual_seed(self, seed: int) -> None:
        del seed


class TestActRun(unittest.TestCase):
    """ACT runs against a dataset recorded by the CPU sim-loop helper."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.dataset = record_sim_episodes(
            num_episodes=1,
            steps_per_episode=8,
            output_dir=str(root / "dataset"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _overrides(self, out: Path) -> list[str]:
        return [
            f"dataset.path={self.dataset}",
            f"output.dir={out}",
            "device=cpu",
            "precision=fp32",
            "hyperparameters.train_steps=2",
            "hyperparameters.batch_size=4",
            "hyperparameters.chunk_size=4",
            "hyperparameters.d_model=32",
            "hyperparameters.nhead=2",
            "hyperparameters.dim_feedforward=64",
            "hyperparameters.latent_dim=8",
            "hyperparameters.log_every=1",
        ]

    def test_missing_torch_does_not_fake_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with (
                patch(
                    "ohho.train.builtin_recipes._import_torch",
                    side_effect=ImportError("no torch"),
                ),
                patch(
                    "ohho.train.act.train_act",
                    side_effect=AssertionError("train_act must not be called"),
                ),
            ):
                code, events, err = _run(example_path("act"), self._overrides(out))
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("needs torch", err)
            self.assertFalse(_events_of(events, "train_start"))
            self.assertFalse((out / "run_manifest.json").exists())

    def test_trainer_failure_exits_after_train_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with (
                patch(
                    "ohho.train.builtin_recipes._import_torch",
                    return_value=_TorchStub(),
                ),
                patch(
                    "ohho.train.act.train_act",
                    side_effect=RuntimeError("boom"),
                ),
            ):
                code, events, err = _run(example_path("act"), self._overrides(out))
            self.assertEqual(code, EXIT_TRAIN_FAILED, err)
            self.assertIn("boom", err)
            self.assertTrue(_events_of(events, "train_start"))
            self.assertFalse(_events_of(events, "done"))
            self.assertFalse((out / "run_manifest.json").exists())

    @unittest.skipUnless(_has_torch(), "torch not installed — [train] extra")
    def test_tiny_act_recipe_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = ["train", str(example_path("act"))]
            for override in self._overrides(out):
                argv.extend(["--set", override])
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli_main(argv)
            self.assertEqual(code, EXIT_OK, stderr.getvalue() + stdout.getvalue())
            events = [
                json.loads(line)
                for line in stdout.getvalue().splitlines()
                if line.strip()
            ]
            self.assertTrue(all(event["stream"] == STREAM for event in events))
            self.assertEqual(
                _events_of(events, "base_weights")[0]["status"], "not_loaded"
            )
            self.assertLess(
                events.index(_events_of(events, "train_start")[0]),
                events.index(_events_of(events, "done")[0]),
            )
            self.assertFalse(_events_of(events, "error"))
            self.assertTrue((out / "policy.pt").is_file())
            self.assertGreater((out / "policy.pt").stat().st_size, 0)
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertIs(config["mock"], False)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertIsInstance(metrics["final_loss"], (int, float))
            self.assertTrue(metrics["loss_curve"])
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema"], "ohho.train.run_manifest/v0")
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["family"], "act")
            self.assertEqual(manifest["device"], "cpu")
            self.assertFalse(manifest["base_weights_loaded"])
            self.assertEqual(manifest["recipe"]["hyperparameters"]["train_steps"], 2)
            self.assertEqual(manifest["final_loss"], metrics["final_loss"])
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
