"""Tests for ``ohho run clone`` and ``ohho run diff``."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from ohho.cli import main as cli_main
from ohho.train.recipe_registry import TrainResult, register, unregister
from ohho.train.recipe_schema import validate_recipe
from ohho.train.run_cli import EXIT_OK, execute
from ohho.train.run_clone import (
    OUTPUT_DIR_NOTE,
    RECIPE_NAME,
    CloneResult,
    cloned_config,
    clone_run,
    diff_configs,
    diff_run_configs,
    format_clone_result,
    format_config_diff,
)
from ohho.train.run_manifest import (
    MANIFEST_NAME,
    SCHEMA_ID,
    BaseWeightsRecord,
    ClonedFrom,
    DatasetRecord,
    GitState,
    RunManifest,
    SeedRecord,
    schema_document,
    set_run_store,
    validate_run_manifest,
    write_run_manifest,
)


def _cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli_main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _config() -> dict[str, Any]:
    return {
        "schema": "ohho.recipe/v1",
        "name": "tiny-act-cpu",
        "family": "act",
        "dataset": {"version_id": "dv_placeholder"},
        "hyperparameters": {
            "train_steps": 4,
            "batch_size": 2,
            "lr": 0.0001,
            "seed": 1,
        },
        "precision": "fp32",
        "device": "cpu",
        "compression_hooks": [{"name": "quant", "params": {"rank": 4}}],
        "eval_hooks": [],
        "output": {"dir": "checkpoints/act"},
    }


def _manifest(config: dict[str, Any], recipe_path: str, **changes: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "status": "succeeded",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "family": config["family"],
        "device": config["device"],
        "precision": config.get("precision", "fp32"),
        "loss_scaling": False,
        "precision_backend": "fp32",
        "resolved_config": config,
        "seeds": SeedRecord(
            recipe=1,
            python=1,
            numpy=None,
            torch=None,
            pythonhashseed=None,
            extra={},
        ),
        "git": GitState(commit=None, dirty=False, root=None),
        "dataset": DatasetRecord(version_id="dv_placeholder", path=None, hash=None),
        "base_weights": BaseWeightsRecord(
            uri=None, revision=None, hash=None, loaded=False
        ),
        "container_image_digest": None,
        "library_versions": {"python": "3.12.0", "ohho": "1.2.0"},
        "gpu": None,
        "final_loss": 0.25,
        "precision_metrics": {
            "fp32": {"throughput_steps_per_s": 10.0, "peak_memory_bytes": 1024},
            "fp16": None,
            "bf16": None,
            "fp8": None,
        },
        "error": None,
        "recipe_path": recipe_path,
        "overrides": ["hyperparameters.seed=1"],
        "checkpoint_dir": "checkpoints/act",
        "metrics_path": None,
        "declared_hooks": {"compression_hooks": ["quant"], "eval_hooks": []},
    }
    fields.update(changes)
    return RunManifest(**fields)


def _write_run(
    directory: Path, config: dict[str, Any] | None = None, **changes: Any
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body = _config() if config is None else config
    write_run_manifest(
        directory / MANIFEST_NAME,
        _manifest(body, str(directory / "source-recipe.yaml"), **changes),
    )
    return directory


def _load(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))


class _RecordingFamily:
    id = "tiny_clone"

    def available(self) -> bool:
        return True

    def run(self, recipe: dict[str, Any], emit) -> TrainResult:
        out = Path(recipe["output"]["dir"])
        emit(
            {
                "event": "train_start",
                "family": self.id,
                "device": recipe.get("device") or "cpu",
            }
        )
        metrics = out / "metrics.json"
        metrics.write_text(json.dumps({"final_loss": 0.5}), encoding="utf-8")
        return TrainResult(str(out), str(metrics), "cpu", False)


class TestRunClone(unittest.TestCase):
    def tearDown(self) -> None:
        set_run_store(None)
        unregister("tiny_clone")

    def test_schema_keeps_cloned_from_optional(self) -> None:
        schema = schema_document()
        self.assertNotIn("cloned_from", schema["required"])
        self.assertIn("cloned_from", schema["properties"])
        plain = _manifest(_config(), "recipe.yaml")
        document = plain.to_dict()
        self.assertNotIn("cloned_from", document)
        self.assertEqual(validate_run_manifest(document), [])
        self.assertIsNone(RunManifest.from_dict(document).cloned_from)
        linked = _manifest(
            _config(),
            "recipe.yaml",
            cloned_from=ClonedFrom(run="/runs/a", manifest="/runs/a/run_manifest.json"),
        )
        stored = linked.to_dict()
        self.assertEqual(validate_run_manifest(stored), [])
        again = RunManifest.from_dict(stored)
        assert again.cloned_from is not None
        self.assertEqual(again.cloned_from.run, "/runs/a")
        self.assertEqual(again.cloned_from.manifest, "/runs/a/run_manifest.json")
        bad = dict(stored)
        bad["cloned_from"] = "runs/a"
        self.assertTrue(validate_run_manifest(bad))

    def test_clone_differs_only_in_edited_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_run(root / "source")
            before = (source / MANIFEST_NAME).read_bytes()
            dest = root / "tweaked"
            overrides = [
                "hyperparameters.lr=0.002",
                "hyperparameters.seed=9",
                "compression_hooks.0.params.rank=8",
                "name=null",
                "hyperparameters.flag=true",
            ]
            code, stdout, stderr = _cli(
                ["run", "clone", str(source), "--dest", str(dest)]
                + [part for item in overrides for part in ("--set", item)]
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual((source / MANIFEST_NAME).read_bytes(), before)
            source_doc = _load(source)
            cloned = _load(dest)
            self.assertEqual(validate_run_manifest(cloned), [])
            self.assertEqual(validate_recipe(cloned["resolved_config"]), [])
            recipe = json.loads((dest / RECIPE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(recipe, cloned["resolved_config"])
            self.assertNotIn("cloned_from", cloned["resolved_config"])
            self.assertEqual(Path(cloned["cloned_from"]["run"]), source.resolve())
            self.assertEqual(
                Path(cloned["cloned_from"]["manifest"]),
                (source / MANIFEST_NAME).resolve(),
            )
            self.assertEqual(cloned["overrides"], overrides)
            self.assertEqual(cloned["status"], source_doc["status"])
            self.assertEqual(cloned["final_loss"], source_doc["final_loss"])
            self.assertEqual(cloned["seeds"], source_doc["seeds"])
            diff = diff_run_configs(source, dest)
            self.assertEqual(
                [(change.path, change.op) for change in diff.changes],
                [
                    ("name", "remove"),
                    ("hyperparameters.lr", "replace"),
                    ("hyperparameters.seed", "replace"),
                    ("hyperparameters.flag", "add"),
                    ("compression_hooks.0.params.rank", "replace"),
                ],
            )
            by_path = {change.path: change for change in diff.changes}
            self.assertEqual(by_path["name"].before, "tiny-act-cpu")
            self.assertEqual(
                by_path["hyperparameters.lr"].before,
                source_doc["resolved_config"]["hyperparameters"]["lr"],
            )
            self.assertEqual(by_path["hyperparameters.lr"].after, 0.002)
            self.assertIsInstance(by_path["hyperparameters.lr"].after, float)
            self.assertEqual(by_path["hyperparameters.seed"].after, 9)
            self.assertIsInstance(by_path["hyperparameters.seed"].after, int)
            self.assertIs(by_path["hyperparameters.flag"].after, True)
            self.assertEqual(by_path["compression_hooks.0.params.rank"].before, 4)
            self.assertEqual(by_path["compression_hooks.0.params.rank"].after, 8)
            payload = diff.to_dict()
            self.assertFalse(payload["equal"])
            flag = next(
                item
                for item in payload["changes"]
                if item["path"] == "hyperparameters.flag"
            )
            self.assertEqual(
                flag, {"path": "hyperparameters.flag", "op": "add", "after": True}
            )
            removed = next(
                item for item in payload["changes"] if item["path"] == "name"
            )
            self.assertNotIn("after", removed)
            self.assertEqual(removed["before"], "tiny-act-cpu")
            text_code, text, text_err = _cli(["run", "diff", str(source), str(dest)])
            self.assertEqual(text_code, 0, text_err)
            self.assertEqual(text, format_config_diff(diff))
            self.assertIn("~ hyperparameters.lr: 0.0001 -> 0.002", text)
            self.assertIn('- name: "tiny-act-cpu"', text)
            self.assertIn("+ hyperparameters.flag: true", text)
            self.assertIn("~ compression_hooks.0.params.rank: 4 -> 8", text)
            self.assertNotIn("final_loss", text)
            self.assertNotIn("cloned_from", text)
            json_code, json_out, json_err = _cli(
                ["run", "diff", str(source), str(dest), "--json"]
            )
            self.assertEqual(json_code, 0, json_err)
            self.assertEqual(json.loads(json_out), payload)
            self.assertTrue(stdout.endswith(text))
            self.assertIn(f"cloned_from: {source.resolve()}", stdout)
            show_code, show_text, show_err = _cli(["run", "show", str(dest)])
            self.assertEqual(show_code, 0, show_err)
            self.assertNotIn("cloned_from:", show_text)
            show_json, show_body, show_json_err = _cli(
                ["run", "show", str(dest), "--json"]
            )
            self.assertEqual(show_json, 0, show_json_err)
            self.assertEqual(
                json.loads(show_body)["cloned_from"], cloned["cloned_from"]
            )

    def test_unchanged_clone_is_an_empty_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_run(root / "source")
            result = clone_run(source, [], dest=root / "copy")
            self.assertTrue(result.diff.equal)
            self.assertEqual(result.diff.changes, ())
            self.assertIsNone(result.note)
            cloned = _load(Path(result.dest))
            self.assertEqual(
                cloned["resolved_config"], _load(source)["resolved_config"]
            )
            self.assertEqual(cloned["overrides"], [])
            self.assertEqual(cloned["cloned_from"]["run"], str(source.resolve()))
            self.assertEqual(
                format_clone_result(result).splitlines()[-1], "equal: true"
            )

    def test_last_set_wins_and_typed_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_run(root / "source")
            result = clone_run(
                source / MANIFEST_NAME,
                [
                    "hyperparameters.lr=0.2",
                    "hyperparameters.lr=1e-3",
                    'name="001"',
                    'hyperparameters.extra={"rank": 2}',
                    'eval_hooks=[{"name": "score"}]',
                ],
                dest=root / "typed",
            )
            self.assertIsInstance(result.config["name"], str)
            self.assertEqual(result.config["name"], "001")
            self.assertIsInstance(result.config["hyperparameters"]["lr"], float)
            self.assertEqual(result.config["hyperparameters"]["lr"], float("1e-3"))
            self.assertEqual(result.config["hyperparameters"]["extra"], {"rank": 2})
            self.assertEqual(result.config["eval_hooks"], [{"name": "score"}])
            self.assertEqual(
                [change.path for change in result.diff.changes if "lr" in change.path],
                ["hyperparameters.lr"],
            )
            self.assertEqual(
                result.diff.to_dict()["left"]["manifest"],
                str((source / MANIFEST_NAME).resolve()),
            )

    def test_invalid_override_writes_nothing_and_keeps_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_run(root / "source")
            before = (source / MANIFEST_NAME).read_bytes()
            dest = root / "bad"
            cases = [
                ["hyperparameters.lr=-1"],
                ["hyperparameters.train_steps=2.5"],
                ["device=tpu"],
                ["family=null"],
                ["compression_hooks.5.name=nope"],
                ["not-a-pair"],
            ]
            for overrides in cases:
                code, _stdout, stderr = _cli(
                    [
                        "run",
                        "clone",
                        str(source),
                        "--dest",
                        str(dest),
                        "--set",
                        *overrides,
                    ]
                )
                self.assertEqual(code, 1, overrides)
                self.assertTrue(stderr.strip(), overrides)
                self.assertFalse(dest.exists(), overrides)
            self.assertEqual((source / MANIFEST_NAME).read_bytes(), before)
            missing, _stdout, stderr = _cli(["run", "clone", str(root / "absent")])
            self.assertEqual(missing, 1)
            self.assertIn("no run manifest", stderr)

    def test_dest_rules_and_output_dir_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run"
            config = _config()
            config["output"] = {"dir": str(source)}
            _write_run(source, config)
            with self.assertRaises(Exception) as overwritten:
                clone_run(source, [], dest=source)
            self.assertIn("source run", str(overwritten.exception))
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(Exception) as exists:
                clone_run(source, [], dest=occupied)
            self.assertIn("already exists", str(exists.exception))
            self.assertEqual(
                (occupied / "keep.txt").read_text(encoding="utf-8"), "keep"
            )
            result = clone_run(source, [])
            # Temp dirs are not spelled the same on every OS: macOS /var is a
            # symlink to /private/var, and Windows may use an 8.3 short name.
            # The clone path is the resolved source plus the sibling name.
            self.assertEqual(
                Path(result.dest),
                source.resolve().with_name(source.name + "-clone"),
            )
            self.assertEqual(result.note, OUTPUT_DIR_NOTE)
            self.assertIn(OUTPUT_DIR_NOTE, format_clone_result(result))
            self.assertTrue(result.diff.equal)
            saved: list[dict[str, Any]] = []
            set_run_store(saved.append)
            hooked = clone_run(source, ["device=cuda"], dest=root / "cuda")
            self.assertEqual(hooked.config["device"], "cuda")
            self.assertEqual(hooked.manifest.device, "cuda")
            self.assertEqual(hooked.manifest.family, "act")
            self.assertEqual(saved[-1]["cloned_from"]["run"], str(source.resolve()))
            self.assertEqual(saved[-1]["device"], "cuda")

    def test_diff_ignores_execution_fields_and_is_directional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = _write_run(root / "left")
            right = _write_run(
                root / "right",
                final_loss=0.9,
                checkpoint_dir=str(root / "other"),
                status="succeeded",
            )
            same = diff_run_configs(left, right)
            self.assertTrue(same.equal)
            self.assertIn("equal: true", format_config_diff(same))
            code, text, err = _cli(["run", "diff", str(left), str(right)])
            self.assertEqual(code, 0, err)
            self.assertEqual(text, format_config_diff(same))
            forward = diff_configs(
                {"flag": True, "keep": 1},
                {"flag": 1, "added": "x"},
                left_run="a",
                right_run="b",
            )
            backward = diff_configs(
                {"flag": 1, "added": "x"},
                {"flag": True, "keep": 1},
                left_run="b",
                right_run="a",
            )
            self.assertEqual(
                [(change.path, change.op) for change in forward.changes],
                [("flag", "replace"), ("keep", "remove"), ("added", "add")],
            )
            self.assertIs(forward.changes[0].before, True)
            self.assertEqual(forward.changes[0].after, 1)
            self.assertNotIsInstance(forward.changes[0].after, bool)
            self.assertEqual(backward.changes[0].before, forward.changes[0].after)
            self.assertIs(backward.changes[0].after, True)
            self.assertEqual(backward.changes[1].op, "remove")
            self.assertEqual(
                [change["path"] for change in forward.to_dict()["changes"]],
                [change.path for change in forward.changes],
            )

    def test_usage_and_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as clone_usage,
            ):
                cli_main(["run", "clone"])
            self.assertEqual(clone_usage.exception.code, 2)
            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as diff_usage,
            ):
                cli_main(["run", "diff", str(root)])
            self.assertEqual(diff_usage.exception.code, 2)
            bad = root / "broken"
            bad.mkdir()
            (bad / MANIFEST_NAME).write_text("{", encoding="utf-8")
            code, _stdout, stderr = _cli(["run", "diff", str(bad), str(bad)])
            self.assertEqual(code, 1)
            self.assertIn("could not read", stderr)
            preview = _config()
            with self.assertRaises(Exception) as rejected:
                cloned_config(preview, ["hyperparameters.batch_size=true"])
            self.assertIn("ohho.recipe/v1", str(rejected.exception))

    def test_trained_run_clone_differs_only_in_the_edit(self) -> None:
        register(_RecordingFamily(), source="test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "name: tiny-act-cpu",
                        "family: tiny_clone",
                        "dataset:",
                        "  version_id: dv_clone",
                        "hyperparameters:",
                        "  train_steps: 2",
                        "  batch_size: 2",
                        "  lr: 0.0001",
                        "  seed: 1",
                        "precision: fp32",
                        "device: cpu",
                        "output:",
                        "  dir: placeholder",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = execute(str(recipe), [f"output.dir={out}"])
            self.assertEqual(code, EXIT_OK, stderr.getvalue())
            before = (out / MANIFEST_NAME).read_bytes()
            dest = root / "cloned"
            clone_code, clone_out, clone_err = _cli(
                [
                    "run",
                    "clone",
                    str(out),
                    "--set",
                    "hyperparameters.lr=0.002",
                    "--dest",
                    str(dest),
                ]
            )
            self.assertEqual(clone_code, 0, clone_err)
            self.assertEqual((out / MANIFEST_NAME).read_bytes(), before)
            diff = diff_run_configs(out, dest)
            self.assertEqual(
                [change.path for change in diff.changes], ["hyperparameters.lr"]
            )
            self.assertEqual(diff.changes[0].op, "replace")
            self.assertEqual(diff.changes[0].after, 0.002)
            self.assertIsInstance(diff.changes[0].after, float)
            cloned = _load(dest)
            self.assertEqual(Path(cloned["cloned_from"]["run"]), out.resolve())
            self.assertEqual(
                Path(cloned["cloned_from"]["manifest"]),
                (out / MANIFEST_NAME).resolve(),
            )
            self.assertEqual(cloned["overrides"], ["hyperparameters.lr=0.002"])
            self.assertIn(OUTPUT_DIR_NOTE, clone_out)
            self.assertIsInstance(
                clone_run(out / MANIFEST_NAME, [], dest=root / "same"), CloneResult
            )


if __name__ == "__main__":
    unittest.main()
