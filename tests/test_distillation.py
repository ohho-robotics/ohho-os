"""Knowledge distillation: tiny CPU teacher, student, cache, and lineage."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.train.compression.distillation import (
    DistillationSettings,
    combine_loss,
    dataset_version_of,
    load_tiny_teacher,
    save_tiny_teacher,
    set_model_registry,
    teacher_cache_key,
)
from ohho.train.recipe_registry import register, unregister
from ohho.train.recipe_schema import apply_overrides, load_recipe_file, validate_recipe
from ohho.train.run_cli import EXIT_INVALID, EXIT_OK, EXIT_UNAVAILABLE, execute
from ohho.train.run_manifest import (
    RunManifest,
    format_run_text,
    validate_run_manifest,
)

try:
    import torch
except ImportError:  # pragma: no cover - the suite skips the torch cases
    torch = None  # type: ignore[assignment]

_FRAMES = (
    ([0.0, 1.0], [0.0, 0.0]),
    ([1.0, 0.0], [0.5, -0.5]),
    ([0.5, 0.5], [1.0, 0.0]),
    ([-1.0, 0.5], [0.0, 1.0]),
)


def _run(path: Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _quote(path: Path) -> str:
    return json.dumps(path.as_posix())


def _write_dataset(root: Path, frames=_FRAMES) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    state_dim = len(frames[0][0])
    action_dim = len(frames[0][1])
    info = {
        "codebase_version": "2.0",
        "robot_type": "sim",
        "repo_id": "tiny-distill",
        "fps": 10,
        "total_episodes": 1,
        "total_frames": len(frames),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
        "tasks": ["distill"],
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["distill"], "length": len(frames)})
        + "\n",
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "distill"}) + "\n",
        encoding="utf-8",
    )
    lines = []
    for index, (state, action) in enumerate(frames):
        lines.append(
            json.dumps(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": index / 10,
                    "frame_index": index,
                    "episode_index": 0,
                    "task_index": 0,
                    "next.done": index == len(frames) - 1,
                }
            )
        )
    (data / "episode_000000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _recipe(
    path: Path,
    *,
    family: str,
    dataset: Path,
    output: Path,
    cache: Path,
    teacher: str,
    version: str | None = "teacher-v1",
) -> None:
    teacher_lines = ["  teacher:"]
    if teacher == "registry":
        teacher_lines.extend(
            [
                "    registry:",
                "      model: smolvla",
                "      version: mv_test",
            ]
        )
    else:
        teacher_lines.append(f"    checkpoint: {_quote(Path(teacher))}")
        if version is not None:
            teacher_lines.append(f"    version: {version}")
    lines = [
        "schema: ohho.recipe/v1",
        f"family: {family}",
        "dataset:",
        f"  path: {_quote(dataset)}",
        "hyperparameters:",
        "  train_steps: 2",
        "  batch_size: 2",
        "  lr: 0.01",
        "  d_model: 4",
        "  nhead: 2",
        "  seed: 0",
        "precision: fp32",
        "device: cpu",
        "distillation:",
        *teacher_lines,
        "  loss: action_regression",
        "  mix_weight: 0.5",
        "  temperature: 2",
        "  feature_matching: false",
        f"  cache_dir: {_quote(cache)}",
        "output:",
        f"  dir: {_quote(output)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


class TestDistillationSchema(unittest.TestCase):
    def test_checkpoint_and_registry_forms_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            recipe = root / "recipe.yaml"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=root / "out",
                cache=root / "cache",
                teacher=str(root / "teacher"),
            )
            loaded = load_recipe_file(recipe)
            self.assertEqual(validate_recipe(loaded), [])
            block = loaded["distillation"]
            self.assertEqual(block["loss"], "action_regression")
            self.assertEqual(block["mix_weight"], 0.5)
            self.assertEqual(block["temperature"], 2)
            self.assertIs(block["feature_matching"], False)
            self.assertEqual(block["feature_weight"], 1.0)
            self.assertEqual(block["teacher"]["version"], "teacher-v1")

            minimal = root / "minimal.yaml"
            minimal.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "family: act",
                        "dataset:",
                        f"  path: {_quote(dataset)}",
                        "distillation:",
                        "  teacher:",
                        f"    checkpoint: {_quote(root / 'teacher')}",
                        "output:",
                        f"  dir: {_quote(root / 'out')}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            filled = load_recipe_file(minimal)
            self.assertEqual(filled["distillation"]["loss"], "action_regression")
            self.assertEqual(filled["distillation"]["temperature"], 1.0)
            self.assertEqual(filled["distillation"]["mix_weight"], 0.5)

            registry = dict(loaded)
            registry["distillation"] = {
                "teacher": {"registry": {"model": "smolvla", "version": "mv_test"}}
            }
            self.assertEqual(validate_recipe(registry), [])

    def test_schema_rejects_a_bad_teacher_or_knob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            recipe = root / "recipe.yaml"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=root / "out",
                cache=root / "cache",
                teacher=str(root / "teacher"),
            )
            loaded = load_recipe_file(recipe)
            both = apply_overrides(
                loaded,
                ['distillation.teacher.registry={"model":"smolvla","version":"mv"}'],
            )
            self.assertTrue(validate_recipe(both))
            neither = json.loads(json.dumps(loaded))
            neither["distillation"]["teacher"] = {"version": "only"}
            self.assertTrue(validate_recipe(neither))
            cold = apply_overrides(loaded, ["distillation.temperature=0"])
            self.assertTrue(validate_recipe(cold))
            heavy = apply_overrides(loaded, ["distillation.mix_weight=1.5"])
            self.assertTrue(validate_recipe(heavy))
            named = apply_overrides(loaded, ["distillation.loss=mse"])
            self.assertTrue(validate_recipe(named))

    def test_cache_key_uses_both_versions(self) -> None:
        same = teacher_cache_key("teacher-v1", "sha256:abc")
        self.assertEqual(same, teacher_cache_key("teacher-v1", "sha256:abc"))
        self.assertEqual(len(same), 64)
        self.assertNotEqual(same, teacher_cache_key("teacher-v2", "sha256:abc"))
        self.assertNotEqual(same, teacher_cache_key("teacher-v1", "sha256:abd"))

    def test_non_il_student_is_refused_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            _write_dataset(dataset)
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="ppo",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(root / "missing"),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("imitation-learning", err)
            self.assertIn("ppo", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
            self.assertFalse((out / "run_manifest.json").exists())

    def test_unavailable_il_family_exits_before_the_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            recipe = root / "recipe.yaml"
            _recipe(
                recipe,
                family="smolvla",
                dataset=dataset,
                output=root / "out",
                cache=root / "cache",
                teacher=str(root / "missing"),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_UNAVAILABLE, err)
            self.assertIn("not available", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )

    def test_missing_checkpoint_does_not_write_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            _write_dataset(dataset)
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(root / "missing"),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("not found", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
            self.assertFalse((out / "run_manifest.json").exists())

    def test_registry_without_a_client_names_the_model_version(self) -> None:
        set_model_registry(None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            _write_dataset(dataset)
            recipe = root / "recipe.yaml"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=root / "out",
                cache=root / "cache",
                teacher="registry",
            )
            code, _events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("no model registry client", err)
            self.assertIn("smolvla@mv_test", err)
            self.assertIn("does not download", err)


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestDistillationLoss(unittest.TestCase):
    def test_mix_weight_blends_supervised_and_teacher_mse(self) -> None:
        student = torch.tensor([[1.0, 3.0]])
        teacher = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[1.0, 3.0]])
        features = torch.zeros(1, 2)
        pure = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(mix_weight=0.0),
        )
        distilled = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(mix_weight=1.0),
        )
        mixed = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(mix_weight=0.25),
        )
        self.assertAlmostEqual(float(pure.total), 0.0, places=5)
        self.assertAlmostEqual(float(distilled.total), 2.0, places=5)
        self.assertAlmostEqual(float(mixed.total), 0.5, places=5)

    def test_temperature_changes_kl_and_not_action_regression(self) -> None:
        student = torch.tensor([[0.0, 0.0]])
        teacher = torch.tensor([[0.0, 4.0]])
        target = torch.zeros(1, 2)
        features = torch.zeros(1, 2)
        cool = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(loss="kl", mix_weight=1.0, temperature=1.0),
        )
        hot = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(loss="kl", mix_weight=1.0, temperature=8.0),
        )
        self.assertGreater(float(cool.distill), 0.0)
        self.assertNotAlmostEqual(float(cool.distill), float(hot.distill), places=4)
        # T^2 rescales the loss. The softened distributions themselves get closer.
        self.assertGreater(float(cool.distill) / 1.0, float(hot.distill) / 64.0)
        low = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(temperature=1.0, mix_weight=1.0),
        )
        high = combine_loss(
            student,
            teacher,
            target,
            features,
            features,
            _settings(temperature=8.0, mix_weight=1.0),
        )
        self.assertAlmostEqual(float(low.distill), float(high.distill), places=5)

    def test_feature_matching_adds_its_weight(self) -> None:
        student = torch.tensor([[1.0, 3.0]])
        teacher = torch.tensor([[1.0, 3.0]])
        target = torch.tensor([[1.0, 3.0]])
        student_features = torch.tensor([[1.0, 0.0]])
        teacher_features = torch.zeros(1, 2)
        parts = combine_loss(
            student,
            teacher,
            target,
            student_features,
            teacher_features,
            _settings(mix_weight=0.0, feature_matching=True, feature_weight=2.0),
        )
        self.assertAlmostEqual(float(parts.supervised), 0.0, places=5)
        self.assertAlmostEqual(float(parts.feature), 0.5, places=5)
        self.assertAlmostEqual(float(parts.total), 1.0, places=5)


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestDistillationRun(unittest.TestCase):
    def tearDown(self) -> None:
        set_model_registry(None)
        unregister("tiny_il")
        unregister("tiny_rl")

    def test_cache_is_reused_and_lineage_records_the_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = save_tiny_teacher(
                root / "teacher", state_dim=2, action_dim=2, hidden=4, seed=1
            )
            cache = root / "cache"
            recipe = root / "recipe.yaml"
            out1 = root / "out1"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out1,
                cache=cache,
                teacher=str(teacher),
            )
            calls = {"n": 0}
            real = load_tiny_teacher

            def spy(directory, device="cpu"):
                calls["n"] += 1
                return real(directory, device=device)

            with patch("ohho.train.compression.distillation.load_tiny_teacher", spy):
                code, _events, err = _run(recipe)
                self.assertEqual(code, EXIT_OK, err)
                self.assertEqual(calls["n"], 1)
                metrics = json.loads(
                    (out1 / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertFalse(metrics["distillation"]["cache_reused"])
                self.assertEqual(metrics["distillation"]["teacher_forward_passes"], 1)
                self.assertTrue(metrics["final_loss"] == metrics["final_loss"])
                manifest = json.loads(
                    (out1 / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(validate_run_manifest(manifest), [])
                teacher_line = manifest["lineage"]["teacher"]
                self.assertEqual(teacher_line["version"], "teacher-v1")
                self.assertEqual(teacher_line["source"], "checkpoint")
                self.assertIsNone(teacher_line["model"])
                self.assertEqual(Path(teacher_line["checkpoint"]), teacher.resolve())
                self.assertIn(
                    "lineage.teacher.version: teacher-v1", format_run_text(manifest)
                )
                loaded = RunManifest.from_dict(manifest)
                self.assertEqual(
                    loaded.to_dict()["lineage"]["teacher"]["version"], "teacher-v1"
                )
                bare = json.loads(json.dumps(manifest))
                del bare["lineage"]
                self.assertEqual(validate_run_manifest(bare), [])
                self.assertIsNone(RunManifest.from_dict(bare).lineage)
                self.assertNotIn("lineage", RunManifest.from_dict(bare).to_dict())
                extra = json.loads(json.dumps(manifest))
                extra["lineage"]["pruned_from"] = "later-method"
                self.assertEqual(validate_run_manifest(extra), [])
                files = list(cache.glob("*.json"))
                self.assertEqual(len(files), 1)
                stored = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(stored["teacher_version"], "teacher-v1")
                self.assertTrue(stored["dataset_version"].startswith("sha256:"))
                self.assertEqual(
                    files[0].stem,
                    teacher_cache_key(
                        stored["teacher_version"], stored["dataset_version"]
                    ),
                )
                self.assertEqual(
                    stored["dataset_version"],
                    dataset_version_of(load_recipe_file(recipe)),
                )

                out2 = root / "out2"
                code, _events, err = _run(recipe, [f"output.dir={_quote(out2)}"])
                self.assertEqual(code, EXIT_OK, err)
                self.assertEqual(calls["n"], 1)
                metrics2 = json.loads(
                    (out2 / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertTrue(metrics2["distillation"]["cache_reused"])
                self.assertEqual(metrics2["distillation"]["teacher_forward_passes"], 0)
                self.assertEqual(len(list(cache.glob("*.json"))), 1)

                frame = dataset / "data" / "chunk-000" / "episode_000000.jsonl"
                frame.write_text(
                    frame.read_text(encoding="utf-8").replace("0.5", "0.25", 1),
                    encoding="utf-8",
                )
                out3 = root / "out3"
                code, _events, err = _run(recipe, [f"output.dir={_quote(out3)}"])
                self.assertEqual(code, EXIT_OK, err)
                self.assertEqual(calls["n"], 2)
                metrics3 = json.loads(
                    (out3 / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertFalse(metrics3["distillation"]["cache_reused"])
                self.assertEqual(len(list(cache.glob("*.json"))), 2)

                out4 = root / "out4"
                code, _events, err = _run(
                    recipe,
                    [
                        f"output.dir={_quote(out4)}",
                        "distillation.teacher.version=teacher-v2",
                    ],
                )
                self.assertEqual(code, EXIT_OK, err)
                self.assertEqual(calls["n"], 3)
                manifest4 = json.loads(
                    (out4 / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    manifest4["lineage"]["teacher"]["version"], "teacher-v2"
                )
                self.assertEqual(len(list(cache.glob("*.json"))), 3)

    def test_omitted_version_is_the_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = save_tiny_teacher(
                root / "teacher", state_dim=2, action_dim=2, hidden=4, seed=2
            )
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(teacher),
                version=None,
            )
            code, _events, err = _run(recipe)
            self.assertEqual(code, EXIT_OK, err)
            digest = hashlib.sha256((teacher / "policy.pt").read_bytes()).hexdigest()
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["lineage"]["teacher"]["version"], "sha256:" + digest
            )

    def test_registry_client_supplies_the_lineage_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = save_tiny_teacher(
                root / "teacher", state_dim=2, action_dim=2, hidden=4, seed=3
            )

            class _Registry:
                def __init__(self) -> None:
                    self.calls: list[tuple[str, str]] = []

                def resolve(self, model: str, version: str) -> dict[str, str]:
                    self.calls.append((model, version))
                    return {"checkpoint": str(teacher), "version": "canonical-9"}

            client = _Registry()
            set_model_registry(client)
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher="registry",
            )
            code, _events, err = _run(recipe)
            self.assertEqual(code, EXIT_OK, err)
            self.assertEqual(set(client.calls), {("smolvla", "mv_test")})
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["lineage"]["teacher"]["version"], "canonical-9")
            self.assertEqual(manifest["lineage"]["teacher"]["source"], "registry")
            self.assertEqual(manifest["lineage"]["teacher"]["model"], "smolvla")
            self.assertEqual(manifest["family"], "act")

    def test_full_model_checkpoint_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = root / "teacher"
            teacher.mkdir()
            (teacher / "policy.pt").write_bytes(b"not-a-tiny-teacher")
            (teacher / "config.json").write_text(
                json.dumps({"policy": "smolvla"}) + "\n", encoding="utf-8"
            )
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(teacher),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("SmolVLA", err)
            self.assertIn("GPU", err)
            self.assertIn("No fallback teacher is loaded", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
            self.assertFalse((out / "run_manifest.json").exists())

    def test_mismatched_dims_and_feature_width_fail_before_train_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            wide = save_tiny_teacher(
                root / "wide", state_dim=3, action_dim=2, hidden=4, seed=4
            )
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(wide),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("do not match", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )

            matched = save_tiny_teacher(
                root / "matched", state_dim=2, action_dim=2, hidden=8, seed=5
            )
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=out,
                cache=root / "cache-features",
                teacher=str(matched),
            )
            code, events, err = _run(recipe, ["distillation.feature_matching=true"])
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("feature matching", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
            self.assertFalse((out / "run_manifest.json").exists())

    def test_corrupt_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = save_tiny_teacher(
                root / "teacher", state_dim=2, action_dim=2, hidden=4, seed=6
            )
            cache = root / "cache"
            recipe = root / "recipe.yaml"
            _recipe(
                recipe,
                family="act",
                dataset=dataset,
                output=root / "out1",
                cache=cache,
                teacher=str(teacher),
            )
            code, _events, err = _run(recipe)
            self.assertEqual(code, EXIT_OK, err)
            stored = next(cache.glob("*.json"))
            stored.write_text("{", encoding="utf-8")
            code, _events, err = _run(recipe, [f"output.dir={_quote(root / 'out2')}"])
            self.assertEqual(code, EXIT_OK, err)
            metrics = json.loads(
                (root / "out2" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertFalse(metrics["distillation"]["cache_reused"])
            self.assertEqual(metrics["distillation"]["teacher_forward_passes"], 1)
            reloaded = json.loads(stored.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["schema"], "ohho.teacher-cache/v1")

    def test_plugin_il_student_uses_the_distillation_runner(self) -> None:
        class _Il:
            id = "tiny_il"

            def available(self) -> bool:
                return True

            def metadata(self) -> dict[str, bool]:
                return {"imitation": True}

            def run(self, recipe, emit) -> None:
                del recipe, emit
                raise AssertionError("distillation must not call the family trainer")

        class _Rl:
            id = "tiny_rl"

            def available(self) -> bool:
                return True

            def metadata(self) -> dict[str, bool]:
                return {"imitation": False}

            def run(self, recipe, emit) -> None:
                del recipe, emit
                raise AssertionError("a non-IL student must be refused")

        register(_Il(), source="test")
        register(_Rl(), source="test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            teacher = save_tiny_teacher(
                root / "teacher", state_dim=2, action_dim=2, hidden=4, seed=7
            )
            recipe = root / "recipe.yaml"
            out = root / "out"
            _recipe(
                recipe,
                family="tiny_il",
                dataset=dataset,
                output=out,
                cache=root / "cache",
                teacher=str(teacher),
            )
            code, _events, err = _run(recipe)
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["family"], "tiny_il")
            self.assertEqual(manifest["lineage"]["teacher"]["version"], "teacher-v1")
            _recipe(
                recipe,
                family="tiny_rl",
                dataset=dataset,
                output=root / "rl",
                cache=root / "cache",
                teacher=str(teacher),
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("tiny_rl", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )


def _settings(**overrides: object) -> DistillationSettings:
    values: dict[str, object] = {
        "loss": "action_regression",
        "mix_weight": 0.5,
        "temperature": 1.0,
        "feature_matching": False,
        "feature_weight": 1.0,
        "cache_dir": None,
    }
    values.update(overrides)
    return DistillationSettings(**values)  # type: ignore[arg-type]
