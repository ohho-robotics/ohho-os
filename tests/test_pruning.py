"""Magnitude, structured, and 2:4 pruning. CPU tensors only."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ohho.train.compression.pruning import (
    NM_SCHEMA,
    PruningCapabilityError,
    PruningSession,
    build_tensorrt_sparse_engine,
    cubic_sparsity,
    load_2to4_tensor,
    normalize_pruning,
    optimizer_steps,
    pending_report,
)
from ohho.train.recipe_schema import (
    RecipeConfigError,
    apply_overrides,
    example_path,
    load_recipe_file,
    validate_recipe,
)
from ohho.train.run_cli import EXIT_INVALID, EXIT_OK, EXIT_TRAIN_FAILED, execute
from ohho.train.run_manifest import set_run_store, validate_run_manifest

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - the suite skips when torch is absent
    torch = None
    nn = None


def _run(path: str | Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


class TestScheduleMath(unittest.TestCase):
    def test_cubic_endpoints_and_midpoint(self) -> None:
        self.assertEqual(cubic_sparsity(0, 0, 10, 0.0, 0.8), 0.0)
        self.assertEqual(cubic_sparsity(10, 0, 10, 0.0, 0.8), 0.8)
        self.assertAlmostEqual(cubic_sparsity(5, 0, 10, 0.0, 0.8), 0.7)
        self.assertEqual(cubic_sparsity(-1, 0, 10, 0.0, 0.8), 0.0)
        self.assertEqual(cubic_sparsity(3, 4, 4, 0.0, 0.5), 0.0)
        self.assertEqual(cubic_sparsity(4, 4, 4, 0.0, 0.5), 0.5)

    def test_sweep_fields_and_rejects(self) -> None:
        recipe = load_recipe_file(example_path("act"))
        updated = apply_overrides(
            recipe,
            [
                "pruning.method=unstructured",
                "pruning.schedule=one_shot",
                "pruning.target_sparsity=0.25",
                'pruning.include=["encoder.*"]',
                'pruning.exclude=["action_head"]',
            ],
        )
        self.assertEqual(validate_recipe(updated), [])
        norm = normalize_pruning(updated["pruning"])
        self.assertEqual(norm["target_sparsity"], 0.25)
        self.assertEqual(norm["schedule"], "one_shot")
        self.assertEqual(norm["include"], ["encoder.*"])
        self.assertEqual(norm["exclude"], ["action_head"])
        self.assertEqual(validate_recipe(updated), [])

        gradual = apply_overrides(
            updated,
            ["pruning.schedule=gradual", "pruning.target_sparsity=0.4"],
        )
        filled = normalize_pruning(gradual["pruning"])
        self.assertEqual(filled["start_step"], 0)
        self.assertNotIn("end_step", filled)
        again = normalize_pruning(filled)
        self.assertEqual(again, filled)

        bad_method = apply_overrides(recipe, ["pruning.method=dropout"])
        bad_method["pruning"]["schedule"] = "one_shot"
        bad_method["pruning"]["target_sparsity"] = 0.5
        self.assertTrue(validate_recipe(bad_method))

        extra = json.loads(json.dumps(updated))
        extra["pruning"]["note"] = "no"
        self.assertTrue(validate_recipe(extra))

    def test_nm_target_and_iterative_budget(self) -> None:
        with self.assertRaises(RecipeConfigError) as raised:
            normalize_pruning(
                {
                    "method": "nm_2_4",
                    "schedule": "one_shot",
                    "target_sparsity": 0.8,
                }
            )
        self.assertIn("0.5", str(raised.exception))
        normalize_pruning(
            {"method": "nm_2_4", "schedule": "one_shot", "target_sparsity": 0.5}
        )
        with self.assertRaises(RecipeConfigError) as budget:
            normalize_pruning(
                {
                    "method": "unstructured",
                    "schedule": "iterative",
                    "target_sparsity": 0.5,
                    "iterations": 100,
                },
                train_steps=2,
            )
        self.assertIn("fine-tune", str(budget.exception))
        planned = normalize_pruning(
            {
                "method": "unstructured",
                "schedule": "iterative",
                "target_sparsity": 0.5,
                "iterations": 2,
                "finetune_steps": 3,
            },
            train_steps=10,
        )
        self.assertEqual(optimizer_steps(planned, 10), 6)
        pending = pending_report(planned)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertIsNone(pending["achieved_sparsity"])
        self.assertEqual(pending["iterations"], 2)

    def test_cli_rejects_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, events, err = _run(
                example_path("act"),
                [
                    "pruning.method=nm_2_4",
                    "pruning.schedule=one_shot",
                    "pruning.target_sparsity=0.8",
                    f"output.dir={out}",
                ],
            )
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("0.5", err)
            self.assertFalse(events and events[0].get("event") == "train_start")
            self.assertFalse(out.exists())
            code, events, err = _run(
                example_path("act"),
                [
                    "pruning.method=unstructured",
                    "pruning.schedule=iterative",
                    "pruning.target_sparsity=0.5",
                    "pruning.iterations=100",
                    "hyperparameters.train_steps=2",
                    f"output.dir={out}",
                ],
            )
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("fine-tune", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
            self.assertFalse((out / "run_manifest.json").exists())


class TestTensorRTExport(unittest.TestCase):
    def test_capability_error_and_builder_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PruningCapabilityError) as raised:
                build_tensorrt_sparse_engine(root)
            text = str(raised.exception)
            self.assertIn("TensorRT", text)
            self.assertIn("CUDA", text)
            self.assertIn("No engine was written", text)

            seen: list[Path] = []

            def builder(path: Path) -> str:
                seen.append(path)
                return "engine.plan"

            self.assertEqual(
                build_tensorrt_sparse_engine(root, builder=builder), "engine.plan"
            )
            self.assertEqual(seen, [root])

            partial = root / "nm24"
            partial.mkdir()
            (partial / "manifest.json").write_text(
                json.dumps({"schema": NM_SCHEMA, "complete": False}),
                encoding="utf-8",
            )
            with self.assertRaises(PruningCapabilityError) as partial_error:
                build_tensorrt_sparse_engine(
                    root,
                    cuda_available=True,
                    tensorrt_available=True,
                )
            self.assertIn("complete", str(partial_error.exception))
            (partial / "manifest.json").write_text(
                json.dumps({"schema": NM_SCHEMA, "complete": True}),
                encoding="utf-8",
            )
            with self.assertRaises(PruningCapabilityError) as follow_up:
                build_tensorrt_sparse_engine(
                    root,
                    cuda_available=True,
                    tensorrt_available=True,
                )
            self.assertIn("follow-up", str(follow_up.exception))
            self.assertIn("No engine was written", str(follow_up.exception))


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestPruningSession(unittest.TestCase):
    def test_unstructured_one_shot_and_filters(self) -> None:
        torch.manual_seed(0)

        class Two(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = nn.Linear(16, 16)
                self.right = nn.Linear(16, 16)

        model = Two()
        with torch.no_grad():
            model.left.weight.copy_(torch.linspace(0.1, 2.0, 256).view(16, 16))
            model.right.weight.copy_(torch.linspace(3.0, 4.0, 256).view(16, 16))
            model.left.bias.fill_(1)
            model.right.bias.fill_(1)
        right_before = model.right.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(
                model,
                {
                    "method": "unstructured",
                    "schedule": "one_shot",
                    "target_sparsity": 0.5,
                    "exclude": ["right"],
                },
                train_steps=1,
                export_dir=tmp,
            )
            report = session.finish()
        self.assertEqual(int((model.left.weight == 0).sum()), 128)
        self.assertTrue(torch.equal(model.right.weight, right_before))
        self.assertEqual(report["layers"][0]["name"], "left")
        self.assertEqual(report["layers"][0]["sparsity"], 0.5)
        self.assertEqual(report["params_before"], report["params_after"])
        self.assertAlmostEqual(report["achieved_sparsity"], 0.5)
        self.assertTrue(torch.equal(model.left.bias, torch.ones(16)))

    def test_gradual_and_iterative_magnitude(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(32, 32)
        with torch.no_grad():
            model.weight.copy_(torch.linspace(0.2, 1.5, 1024).view(32, 32))
        gradual = {
            "method": "unstructured",
            "schedule": "gradual",
            "target_sparsity": 0.5,
            "end_step": 4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(model, gradual, train_steps=5, export_dir=tmp)
            counts = []
            for step in range(session.optimizer_steps):
                session.before_step(step)
                session.after_step(step)
                counts.append(int((model.weight == 0).sum()))
            report = session.finish()
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], 512)
        self.assertEqual(report["achieved_sparsity"], 0.5)

        fresh = nn.Linear(32, 32)
        with torch.no_grad():
            fresh.weight.copy_(torch.linspace(0.2, 1.5, 1024).view(32, 32))
        iterative = {
            "method": "unstructured",
            "schedule": "iterative",
            "target_sparsity": 0.5,
            "iterations": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(fresh, iterative, train_steps=4, export_dir=tmp)
            session.before_step(0)
            self.assertEqual(int((fresh.weight == 0).sum()), 256)
            session.before_step(2)
            self.assertEqual(int((fresh.weight == 0).sum()), 512)
            report = session.finish()
        self.assertEqual(report["achieved_sparsity"], 0.5)
        self.assertEqual(report["iterations"], 2)

    def test_structured_channel_shrinks_the_file(self) -> None:
        torch.manual_seed(0)

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(128, 256)
                self.fc2 = nn.Linear(256, 128)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(value)))

        model = MLP()
        before_params = sum(int(param.numel()) for param in model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(
                model,
                {
                    "method": "structured_channel",
                    "schedule": "one_shot",
                    "target_sparsity": 0.5,
                },
                train_steps=1,
                export_dir=tmp,
            )
            report = session.finish()
            before = Path(tmp) / "pruning_export" / "before.pt"
            after = Path(tmp) / "pruning_export" / "after.pt"
            self.assertLess(after.stat().st_size, before.stat().st_size)
            self.assertEqual(report["export_bytes_before"], before.stat().st_size)
            self.assertEqual(report["export_bytes_after"], after.stat().st_size)
        self.assertEqual(model.fc1.out_features, 128)
        self.assertEqual(model.fc2.in_features, 128)
        self.assertEqual(model.fc2.out_features, 128)
        self.assertLess(report["params_after"], before_params)
        self.assertLess(report["params_after"], report["params_before"])
        self.assertEqual(report["achieved_sparsity"], 0.5)
        output = model(torch.randn(2, 128))
        self.assertEqual(tuple(output.shape), (2, 128))
        self.assertTrue(torch.isfinite(output).all())

    def test_structured_head_keeps_the_residual_width(self) -> None:
        torch.manual_seed(0)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )
        before = sum(int(param.numel()) for param in layer.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(
                layer,
                {
                    "method": "structured_head",
                    "schedule": "one_shot",
                    "target_sparsity": 0.5,
                },
                train_steps=1,
                export_dir=tmp,
            )
            report = session.finish()
            self.assertLess(
                (Path(tmp) / "pruning_export" / "after.pt").stat().st_size,
                (Path(tmp) / "pruning_export" / "before.pt").stat().st_size,
            )
        self.assertLess(report["params_after"], before)
        self.assertEqual(layer.self_attn.num_heads, 2)
        self.assertEqual(layer.self_attn.embed_dim, 32)
        layer.eval()
        output = layer(torch.randn(2, 5, 32))
        self.assertEqual(tuple(output.shape), (2, 5, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_conv_channels_keep_the_spatial_size(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
                self.conv2 = nn.Conv2d(8, 4, 3, padding=1)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.conv2(torch.relu(self.conv1(value)))

        model = Net()
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(
                model,
                {
                    "method": "structured_channel",
                    "schedule": "one_shot",
                    "target_sparsity": 0.5,
                },
                train_steps=1,
                export_dir=tmp,
            )
            report = session.finish()
        self.assertEqual(model.conv1.out_channels, 4)
        self.assertEqual(model.conv2.in_channels, 4)
        self.assertEqual(model.conv2.out_channels, 4)
        self.assertLess(report["params_after"], report["params_before"])
        output = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 4, 8, 8))

    def test_two_of_four_export_roundtrip(self) -> None:
        torch.manual_seed(0)
        layer = nn.Linear(8, 16)
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight))
        with tempfile.TemporaryDirectory() as tmp:
            session = PruningSession(
                layer,
                {
                    "method": "nm_2_4",
                    "schedule": "one_shot",
                    "target_sparsity": 0.5,
                },
                train_steps=1,
                export_dir=tmp,
            )
            report = session.finish()
            manifest = json.loads(
                (Path(tmp) / "pruning_export" / "nm24" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            rebuilt = load_2to4_tensor(tmp, "weight")
        grouped = layer.weight.detach().reshape(-1, 4)
        self.assertTrue(torch.all(grouped.ne(0).sum(dim=-1) == 2))
        self.assertEqual(report["export_format"], NM_SCHEMA)
        self.assertIn("No engine was written", report["tensorrt_error"])
        self.assertIsNone(report["tensorrt_engine"])
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["schema"], NM_SCHEMA)
        self.assertTrue(torch.allclose(rebuilt, layer.weight.detach().cpu()))
        self.assertAlmostEqual(report["achieved_sparsity"], 0.5, places=5)

    def test_no_matching_layer_is_an_error(self) -> None:
        model = nn.Linear(8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RecipeConfigError) as raised:
                PruningSession(
                    model,
                    {
                        "method": "unstructured",
                        "schedule": "one_shot",
                        "target_sparsity": 0.5,
                        "include": ["missing"],
                    },
                    train_steps=1,
                    export_dir=tmp,
                )
        self.assertIn("matched no layers", str(raised.exception))


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestPruningRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from ohho.sim_loop import record_sim_episodes

        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = record_sim_episodes(
            num_episodes=1,
            steps_per_episode=6,
            output_dir=str(Path(cls._tmp.name) / "dataset"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _overrides(
        self, out: Path, method: str, schedule: str = "one_shot"
    ) -> list[str]:
        return [
            f"dataset.path={self.dataset}",
            f"output.dir={out}",
            "device=cpu",
            "precision=fp32",
            "hyperparameters.train_steps=2",
            "hyperparameters.batch_size=4",
            "hyperparameters.chunk_size=2",
            "hyperparameters.d_model=32",
            "hyperparameters.nhead=4",
            "hyperparameters.dim_feedforward=64",
            "hyperparameters.latent_dim=8",
            "hyperparameters.log_every=1",
            f"pruning.method={method}",
            f"pruning.schedule={schedule}",
            "pruning.target_sparsity=0.5",
        ]

    def test_unstructured_sparsity_lands_in_the_manifest(self) -> None:
        seen: list[dict] = []

        def save(document: dict) -> None:
            seen.append(json.loads(json.dumps(document)))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            set_run_store(save)
            try:
                code, _events, err = _run(
                    example_path("act"), self._overrides(out, "unstructured")
                )
            finally:
                set_run_store(None)
            self.assertEqual(code, EXIT_OK, err)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertEqual(manifest["pruning"]["method"], "unstructured")
            self.assertEqual(manifest["pruning"]["schedule"], "one_shot")
            self.assertGreater(len(manifest["pruning"]["layers"]), 0)
            for layer in manifest["pruning"]["layers"]:
                self.assertAlmostEqual(layer["sparsity"], 0.5, places=2)
            self.assertAlmostEqual(
                manifest["pruning"]["achieved_sparsity"], 0.5, places=2
            )
            self.assertEqual(manifest["pruning"], metrics["pruning"])
            self.assertEqual(
                manifest["resolved_config"]["pruning"]["target_sparsity"], 0.5
            )
            self.assertEqual(seen[0]["status"], "running")
            self.assertIsNone(seen[0]["pruning"]["achieved_sparsity"])
            self.assertEqual(seen[-1]["status"], "succeeded")
            older = dict(manifest)
            del older["pruning"]
            self.assertEqual(validate_run_manifest(older), [])

    def test_structured_channel_export_is_smaller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, _events, err = _run(
                example_path("act"), self._overrides(out, "structured_channel")
            )
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_run_manifest(manifest), [])
            pruning = manifest["pruning"]
            self.assertLess(pruning["params_after"], pruning["params_before"])
            self.assertLess(
                pruning["export_bytes_after"], pruning["export_bytes_before"]
            )
            before = (out / "pruning_export" / "before.pt").stat().st_size
            after = (out / "pruning_export" / "after.pt").stat().st_size
            policy = (out / "policy.pt").stat().st_size
            self.assertEqual(pruning["export_bytes_before"], before)
            self.assertEqual(pruning["export_bytes_after"], after)
            self.assertLess(after, before)
            self.assertLess(policy, before)
            names = [layer["name"] for layer in pruning["layers"]]
            self.assertTrue(any(name.endswith("linear1") for name in names))

    def test_two_of_four_records_the_engine_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, _events, err = _run(
                example_path("act"), self._overrides(out, "nm_2_4")
            )
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_run_manifest(manifest), [])
            pruning = manifest["pruning"]
            self.assertEqual(pruning["export_format"], NM_SCHEMA)
            self.assertIn("No engine was written", pruning["tensorrt_error"])
            self.assertIsNone(pruning["tensorrt_engine"])
            export = json.loads(
                (out / "pruning_export" / "nm24" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(export["schema"], NM_SCHEMA)
            self.assertTrue(export["tensors"])

    def test_unknown_include_fails_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            overrides = self._overrides(out, "unstructured")
            overrides.append('pruning.include=["does_not_exist"]')
            code, events, err = _run(example_path("act"), overrides)
            self.assertEqual(code, EXIT_TRAIN_FAILED, err)
            self.assertIn("matched no layers", err)
            self.assertTrue(
                any(event.get("event") == "train_start" for event in events)
            )
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertIsNone(manifest["pruning"]["achieved_sparsity"])
            self.assertIn("matched no layers", manifest["error"])


if __name__ == "__main__":
    unittest.main()
