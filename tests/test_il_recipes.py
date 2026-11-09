"""OHH-508: ACT and Diffusion Policy recipes, validation, and serve layout."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from ohho.serve import _is_act_checkpoint
from ohho.train.builtin_recipes import ACT, DIFFUSION
from ohho.train.il.data import classify_dataset_path, materialize_hub_dataset
from ohho.train.il.params import documented_parameter_lines, resolve_hyperparameters
from ohho.train.il.progress import ValTracker, validation_due
from ohho.train.recipe_registry import ensure_loaded, get_family
from ohho.train.recipe_schema import (
    DatasetUnavailable,
    RecipeConfigError,
    example_path,
    load_recipe_file,
)
from ohho.train.run_cli import (
    EXIT_DATASET,
    EXIT_INVALID,
    EXIT_OK,
    STREAM,
    execute,
)

_IL = Path(__file__).resolve().parents[1] / "ohho" / "train" / "examples" / "il"
_DOCS = Path(__file__).resolve().parents[1] / "docs" / "train" / "il-recipes.md"


def _run(path: str | Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    from contextlib import redirect_stderr, redirect_stdout

    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _events(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _torch_skip_reason() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return (
            "skipped: torch is not installed, so the tiny CPU ACT and Diffusion "
            "Policy recipes cannot train (lerobot/torch: pip install "
            "'ohho-os[train]'; lerobot is only required to download an hf:// dataset)"
        )
    return None


def _http_skip_reason() -> str | None:
    missing = []
    for name in ("torch", "fastapi", "httpx"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return None
    listed = ", ".join(missing)
    return (
        f"skipped: serve HTTP load needs {listed} (pip install 'ohho-os[train,serve]')"
    )


class _TorchStub:
    def manual_seed(self, seed: int) -> None:
        del seed


class TestParameterValidation(unittest.TestCase):
    def test_defaults_and_shipped_examples(self) -> None:
        act = resolve_hyperparameters("act", {})
        self.assertEqual(act["train_steps"], 200)
        self.assertEqual(act["batch_size"], 8)
        self.assertEqual(act["lr"], 0.0001)
        self.assertEqual(act["chunk_size"], 10)
        self.assertEqual(act["vision_backbone"], "none")
        self.assertEqual(act["val_every"], 50)
        self.assertNotIn("num_epochs", act)
        self.assertNotIn("seed", act)
        diffusion = resolve_hyperparameters("diffusion", {})
        self.assertEqual(diffusion["chunk_size"], 16)
        self.assertEqual(diffusion["horizon"], 16)
        self.assertEqual(diffusion["diffusion_steps"], 100)
        self.assertEqual(diffusion["scheduler"], "ddpm")
        self.assertEqual(diffusion["num_inference_steps"], 10)
        self.assertEqual(diffusion["vision_backbone"], "none")
        clamped = resolve_hyperparameters("diffusion", {"diffusion_steps": 4})
        self.assertEqual(clamped["num_inference_steps"], 4)
        shipped = load_recipe_file(example_path("diffusion"))
        resolved = resolve_hyperparameters("diffusion", shipped["hyperparameters"])
        self.assertEqual(resolved["train_steps"], 200)
        self.assertEqual(resolved["chunk_size"], 16)
        act_shipped = load_recipe_file(example_path("act"))
        act_resolved = resolve_hyperparameters("act", act_shipped["hyperparameters"])
        self.assertEqual(act_resolved["train_steps"], 200)
        self.assertEqual(act_resolved["chunk_size"], 10)
        self.assertEqual(act_resolved["seed"], 0)

    def test_ranges_reject_before_training_values(self) -> None:
        cases = [
            ("act", {"chunk_size": 2000}, "chunk_size"),
            ("act", {"lr": 2}, "lr"),
            ("act", {"vision_backbone": "vit"}, "vision_backbone"),
            ("act", {"batch_size": True}, "batch_size"),
            ("act", {"d_model": 30, "nhead": 4}, "divisible"),
            ("act", {"mock": True}, "fake trainer"),
            ("act", {"horizon": 8}, "unknown hyperparameters"),
            ("diffusion", {"scheduler": "euler"}, "scheduler"),
            ("diffusion", {"diffusion_steps": 5000}, "diffusion_steps"),
            ("diffusion", {"horizon": 4, "chunk_size": 8}, "must match"),
            (
                "diffusion",
                {"diffusion_steps": 4, "num_inference_steps": 5},
                "num_inference_steps",
            ),
        ]
        for family, raw, needle in cases:
            with self.subTest(family=family, needle=needle):
                with self.assertRaises(RecipeConfigError) as caught:
                    resolve_hyperparameters(family, raw)
                self.assertIn(needle, str(caught.exception))

    def test_resnet_name_is_in_range_but_not_trainable(self) -> None:
        resolved = resolve_hyperparameters("act", {"vision_backbone": "resnet18"})
        self.assertEqual(resolved["vision_backbone"], "resnet18")

    def test_epochs_become_steps_until_train_steps_is_set(self) -> None:
        from_epochs = resolve_hyperparameters("diffusion", {"num_epochs": 3})
        self.assertEqual(from_epochs["train_steps"], 6)
        both = resolve_hyperparameters("act", {"num_epochs": 9, "train_steps": 4})
        self.assertEqual(both["train_steps"], 4)


class TestDatasetRef(unittest.TestCase):
    def test_hub_and_local_paths(self) -> None:
        hub = classify_dataset_path("hf://lerobot/pusht")
        self.assertEqual(hub.kind, "hub")
        self.assertEqual(hub.repo_id, "lerobot/pusht")
        bare = classify_dataset_path("lerobot/pusht")
        self.assertEqual((bare.kind, bare.repo_id), ("hub", "lerobot/pusht"))
        with self.assertRaises(DatasetUnavailable) as caught:
            classify_dataset_path("hf://not a repo")
        self.assertIn("invalid HF Hub", str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing-dataset"
            with self.assertRaises(DatasetUnavailable) as missing_caught:
                classify_dataset_path(str(missing))
            self.assertIn("does not exist", str(missing_caught.exception))
            local = root / "demos"
            local.mkdir()
            found = classify_dataset_path(str(local))
            self.assertEqual(found.kind, "local")
            self.assertEqual(found.path, local)
            nested = root / "lerobot" / "pusht"
            nested.mkdir(parents=True)
            with _chdir(root):
                self.assertEqual(classify_dataset_path("lerobot/pusht").kind, "local")
                with self.assertRaises(DatasetUnavailable) as placeholder:
                    classify_dataset_path("datasets/sim-loop")
                self.assertIn("does not exist", str(placeholder.exception))

    def test_missing_lerobot_explains_the_hub_download(self) -> None:
        with patch(
            "ohho.train.il.data._import_lerobot_dataset",
            side_effect=ImportError("no lerobot"),
        ):
            with self.assertRaises(DatasetUnavailable) as caught:
                materialize_hub_dataset("lerobot/pusht")
        self.assertIn("lerobot/pusht", str(caught.exception))
        self.assertIn("pip install lerobot", str(caught.exception))


class TestRegistryAndDocs(unittest.TestCase):
    def test_families_are_registered_without_importing_torch(self) -> None:
        ensure_loaded()
        with patch(
            "ohho.train.builtin_recipes._import_torch",
            side_effect=AssertionError("torch should not be imported"),
        ):
            self.assertIs(get_family("act"), ACT)
            self.assertIs(get_family("diffusion"), DIFFUSION)
            self.assertTrue(ACT.available())
            self.assertTrue(DIFFUSION.available())
            self.assertTrue(callable(ACT.run))
            self.assertTrue(callable(DIFFUSION.run))

    def test_docs_list_every_parameter_default_and_range(self) -> None:
        text = _DOCS.read_text(encoding="utf-8")
        for line in documented_parameter_lines():
            self.assertIn(line, text)
        self.assertIn("ohho.train.il.diffusion_policy.DiffusionPolicyModel", text)
        self.assertIn("policy.pt", text)
        self.assertIn("config.json", text)
        self.assertIn("weights_only", text)

    def test_shipped_il_configs_validate(self) -> None:
        names = sorted(path.name for path in _IL.glob("*.yaml"))
        self.assertEqual(
            names,
            [
                "act-gpu.yaml",
                "act-tiny-cpu.yaml",
                "diffusion-gpu.yaml",
                "diffusion-tiny-cpu.yaml",
            ],
        )
        for path in sorted(_IL.glob("*.yaml")):
            recipe = load_recipe_file(path)
            self.assertEqual(recipe["schema"], "ohho.recipe/v1")
            self.assertIn(recipe["family"], {"act", "diffusion"})
        tiny = load_recipe_file(_IL / "act-tiny-cpu.yaml")
        self.assertEqual(tiny["device"], "cpu")
        self.assertEqual(tiny["hyperparameters"]["vision_backbone"], "none")
        gpu = load_recipe_file(_IL / "diffusion-gpu.yaml")
        self.assertEqual(gpu["device"], "cuda")
        self.assertTrue(str(gpu["dataset"]["path"]).startswith("hf://"))
        self.assertEqual(gpu["hyperparameters"]["scheduler"], "ddim")

    def test_validation_schedule_and_patience(self) -> None:
        self.assertFalse(validation_due(1, 3, 2))
        self.assertTrue(validation_due(2, 3, 2))
        self.assertTrue(validation_due(3, 3, 2))
        patience = ValTracker(patience=1, min_delta=1e9)
        self.assertFalse(patience.observe(1.0))
        self.assertTrue(patience.observe(0.9))
        disabled = ValTracker(patience=0, min_delta=0.0)
        self.assertFalse(disabled.observe(1.0))
        self.assertFalse(disabled.observe(2.0))
        streak = ValTracker(patience=2, min_delta=0.0)
        self.assertFalse(streak.observe(1.0))
        self.assertFalse(streak.observe(0.5))
        self.assertFalse(streak.observe(0.5))
        self.assertTrue(streak.observe(0.5))


class TestPreflight(unittest.TestCase):
    def test_bad_values_do_not_start_training(self) -> None:
        cases = [
            (
                example_path("act"),
                ["hyperparameters.chunk_size=2000"],
                "chunk_size",
            ),
            (
                example_path("act"),
                ["hyperparameters.lr=2"],
                "lr",
            ),
            (
                example_path("act"),
                ["hyperparameters.vision_backbone=vit"],
                "vision_backbone",
            ),
            (
                example_path("diffusion"),
                ["hyperparameters.scheduler=euler"],
                "scheduler",
            ),
            (
                _IL / "act-gpu.yaml",
                [],
                "not trainable",
            ),
            (
                _IL / "diffusion-gpu.yaml",
                [],
                "not trainable",
            ),
        ]
        for path, overrides, needle in cases:
            with self.subTest(path=Path(path).name, needle=needle):
                with patch(
                    "ohho.train.il.data.materialize_hub_dataset",
                    side_effect=AssertionError("hub download"),
                ):
                    code, events, err = _run(path, overrides)
                self.assertEqual(code, EXIT_INVALID, err)
                self.assertIn(needle, err)
                self.assertFalse(_events(events, "train_start"))
                self.assertEqual([event["event"] for event in events], ["error"])

    def test_hub_id_is_resolved_before_train_start_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with (
                patch(
                    "ohho.train.builtin_recipes._import_torch",
                    return_value=_TorchStub(),
                ),
                patch(
                    "ohho.train.il.data.materialize_hub_dataset",
                    side_effect=DatasetUnavailable(
                        "HF Hub dataset 'lerobot/pusht' needs lerobot to download"
                    ),
                ) as fetch,
            ):
                code, events, err = _run(
                    example_path("diffusion"),
                    [
                        "dataset.path=hf://lerobot/pusht",
                        f"output.dir={out}",
                        "hyperparameters.train_steps=1",
                        "device=cpu",
                        "hyperparameters.vision_backbone=none",
                    ],
                )
            self.assertEqual(code, EXIT_DATASET, err)
            self.assertIn("lerobot/pusht", err)
            self.assertEqual([event["event"] for event in events], ["error"])
            fetch.assert_called_once_with("lerobot/pusht")

    def test_diffusion_loader_requests_weights_only(self) -> None:
        import ohho.train.il.diffusion_policy as mod

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "config.json").write_text(
                json.dumps(
                    {
                        "policy": "diffusion",
                        "state_dim": 4,
                        "action_dim": 4,
                        "chunk_size": 2,
                        "diffusion_steps": 4,
                        "d_model": 8,
                        "scheduler": "ddpm",
                        "num_inference_steps": 2,
                    }
                ),
                encoding="utf-8",
            )
            (folder / "policy.pt").write_bytes(b"weights")
            mock_torch = MagicMock()
            mock_model = MagicMock()
            with (
                patch.object(mod, "TORCH_AVAILABLE", True),
                patch.object(mod, "torch", mock_torch, create=True),
                patch.object(mod, "TinyDiffusionPolicy", return_value=mock_model),
            ):
                model, config = mod.load_diffusion_checkpoint(str(folder), device="cpu")
            mock_torch.load.assert_called_once()
            self.assertEqual(mock_torch.load.call_args.kwargs.get("weights_only"), True)
            self.assertIs(model, mock_model)
            self.assertEqual(config["policy"], "diffusion")

    def test_serve_auto_detects_only_the_act_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            act = root / "act"
            diffusion = root / "diffusion"
            act.mkdir()
            diffusion.mkdir()
            (act / "config.json").write_text(
                json.dumps({"policy": "act"}), encoding="utf-8"
            )
            (diffusion / "config.json").write_text(
                json.dumps(
                    {
                        "policy": "diffusion",
                        "model_type": "diffusion",
                        "model_family": "diffusion",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_is_act_checkpoint(str(act)))
            self.assertFalse(_is_act_checkpoint(str(diffusion)))


class TestTinyCpuTrain(unittest.TestCase):
    """Both recipes train on a recorded LeRobot dataset when torch is installed."""

    dataset = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip = _torch_skip_reason()
        cls._tmp = None
        if cls._skip:
            return
        from ohho.sim_loop import record_sim_episodes

        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = record_sim_episodes(
            num_episodes=1,
            steps_per_episode=8,
            output_dir=str(Path(cls._tmp.name) / "dataset"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmp is not None:
            cls._tmp.cleanup()

    def setUp(self) -> None:
        if self._skip:
            self.skipTest(self._skip)

    def test_validation_loss_every_n_steps_and_serve_layout(self) -> None:
        specs = (
            ("act", _IL / "act-tiny-cpu.yaml", "act"),
            ("diffusion", _IL / "diffusion-tiny-cpu.yaml", "diffusion"),
        )
        for family, recipe, policy in specs:
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "out"
                    code, events, err = _run(
                        recipe,
                        [
                            f"dataset.path={self.dataset}",
                            f"output.dir={out}",
                            "device=cpu",
                            "hyperparameters.train_steps=6",
                            "hyperparameters.val_every=2",
                            "hyperparameters.early_stop_patience=0",
                            "hyperparameters.vision_backbone=none",
                        ],
                    )
                    self.assertEqual(code, EXIT_OK, err + "\n" + _dump(events))
                    self.assertTrue(all(event["stream"] == STREAM for event in events))
                    self.assertEqual(
                        [event["step"] for event in _events(events, "validation")],
                        [2, 4, 6],
                    )
                    for event in _events(events, "validation"):
                        self.assertIsInstance(event["validation_loss"], float)
                        self.assertEqual(event["family"], family)
                    self.assertLess(
                        events.index(_events(events, "train_start")[0]),
                        events.index(_events(events, "validation")[0]),
                    )
                    self.assertFalse(_events(events, "error"))
                    self.assertTrue((out / "policy.pt").is_file())
                    self.assertGreater((out / "policy.pt").stat().st_size, 0)
                    self.assertTrue((out / "metrics.json").is_file())
                    config = json.loads(
                        (out / "config.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(config["policy"], policy)
                    self.assertIs(config["mock"], False)
                    metrics = json.loads(
                        (out / "metrics.json").read_text(encoding="utf-8")
                    )
                    self.assertIsInstance(metrics["final_loss"], float)
                    self.assertEqual(
                        metrics["validation_loss"], events_validation_last(events)
                    )
                    self._assert_serve_loads(family, out, config)

    def test_early_stop_uses_validation_loss(self) -> None:
        for family, recipe in (
            ("act", _IL / "act-tiny-cpu.yaml"),
            ("diffusion", _IL / "diffusion-tiny-cpu.yaml"),
        ):
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / "out"
                    code, events, err = _run(
                        recipe,
                        [
                            f"dataset.path={self.dataset}",
                            f"output.dir={out}",
                            "device=cpu",
                            "hyperparameters.train_steps=20",
                            "hyperparameters.val_every=1",
                            "hyperparameters.early_stop_patience=1",
                            "hyperparameters.early_stop_min_delta=1e9",
                            "hyperparameters.vision_backbone=none",
                        ],
                    )
                    self.assertEqual(code, EXIT_OK, err)
                    self.assertEqual(
                        [event["step"] for event in _events(events, "validation")],
                        [1, 2],
                    )
                    stopped = _events(events, "early_stop")
                    self.assertEqual(len(stopped), 1)
                    self.assertEqual(stopped[0]["step"], 2)
                    self.assertEqual(stopped[0]["family"], family)
                    metrics = json.loads(
                        (out / "metrics.json").read_text(encoding="utf-8")
                    )
                    self.assertTrue(metrics["stopped_early"])
                    self.assertEqual(metrics["completed_steps"], 2)
                    self.assertTrue((out / "policy.pt").is_file())

    def test_both_diffusion_schedulers_sample(self) -> None:
        import torch
        from ohho.train.il.diffusion_policy import TinyDiffusionPolicy

        model = TinyDiffusionPolicy(
            state_dim=4,
            action_dim=4,
            chunk_size=2,
            diffusion_steps=4,
            d_model=16,
            scheduler="ddpm",
            num_inference_steps=4,
        )
        calls = {"n": 0}
        original = model.forward

        def _wrapped(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        model.forward = _wrapped  # type: ignore[method-assign]
        action = model.select_action([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(action), 4)
        self.assertEqual(calls["n"], 4)
        self.assertTrue(all(math.isfinite(item) for item in action))
        calls["n"] = 0
        model.scheduler = "ddim"
        model.num_inference_steps = 2
        action = model.select_action([0.0, 0.1, 0.0, -0.1])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(action), 4)
        self.assertTrue(all(math.isfinite(item) for item in action))
        del torch

    def _assert_serve_loads(self, family: str, out: Path, config: dict) -> None:
        state = [0.0] * int(config["state_dim"])
        action_dim = int(config["action_dim"])
        if family == "act":
            from ohho.serve.act import ACTModel

            model = ACTModel(device="cpu")
            model.load_model(str(out))
            self.assertIsNotNone(model.policy)
            predicted = model.predict_action(instruction="reach", state=state)
        else:
            from ohho.train.il.diffusion_policy import DiffusionPolicyModel

            model = DiffusionPolicyModel(device="cpu")
            model.load_model(str(out))
            self.assertIsNotNone(model.policy)
            predicted = model.predict_action(instruction="reach", state=state)
        self.assertFalse(predicted["mock"])
        self.assertEqual(len(predicted["vector"]), action_dim)
        reason = _http_skip_reason()
        if reason:
            return
        self._assert_http(family, out, state, action_dim)

    def _assert_http(
        self, family: str, out: Path, state: list[float], action_dim: int
    ) -> None:
        from fastapi.testclient import TestClient
        from ohho.serve import build_app

        if family == "act":
            app = build_app(model_path=str(out), device="cpu", auto_load=True)
        else:
            app = build_app(
                model_class="ohho.train.il.diffusion_policy.DiffusionPolicyModel",
                model_path=str(out),
                device="cpu",
                auto_load=True,
            )
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["model_loaded"])
            response = client.post(
                "/predict",
                json={"instruction": "reach", "config": {"state": state}},
            )
            self.assertEqual(response.status_code, 200, response.text)
            vector = response.json()["action"]["vector"]
            self.assertEqual(len(vector), action_dim)


def events_validation_last(events: list[dict]) -> float:
    return float(_events(events, "validation")[-1]["validation_loss"])


def _dump(events: list[dict]) -> str:
    return "\n".join(json.dumps(event) for event in events)


if __name__ == "__main__":
    unittest.main()
