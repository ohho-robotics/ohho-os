"""Quantisation-aware training: recipe block, fake quant, Q/DQ ONNX."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.compression.qat import (
    QatCapabilityError,
    QatError,
    build_tensorrt_engine,
    compare_ptq_qat,
    export_qdq_onnx,
    insert_fake_quant,
    prepare_qat_recipe,
    run_calibration,
    run_finetune,
    settings_from_recipe,
    spec_from_mapping,
    unwrap_fake_quant,
)
from ohho.train.recipe_registry import TrainResult, register, unregister
from ohho.train.recipe_schema import (
    apply_overrides,
    example_path,
    load_recipe_file,
    validate_recipe,
)
from ohho.train.run_cli import EXIT_DATASET, EXIT_INVALID, EXIT_OK, execute
from ohho.train.run_manifest import (
    capture_run,
    format_run_text,
    manifest_from_capture,
    validate_run_manifest,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def _run(path: Path, overrides: list[str] | None = None) -> tuple[int, list[dict], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = []
    text = stdout.getvalue().strip()
    if text:
        events = [json.loads(line) for line in text.splitlines()]
    return code, events, stderr.getvalue()


class TestQatRecipeAndManifest(unittest.TestCase):
    def tearDown(self) -> None:
        unregister("qat_record")

    def test_schema_accepts_a_sweep_of_plain_fields(self) -> None:
        loaded = load_recipe_file(example_path("act"))
        updated = apply_overrides(
            loaded,
            [
                "qat.bit_width=4",
                "qat.granularity=per_tensor",
                "qat.calibration_steps=2",
                "qat.finetune_steps=7",
                "qat.scope=weight_and_activation",
                "qat.symmetric=false",
                "qat.activation_granularity=per_channel",
                "qat.activation_symmetric=true",
            ],
        )
        self.assertEqual(validate_recipe(updated), [])
        spec = spec_from_mapping(updated["qat"])
        self.assertEqual(spec.bit_width, 4)
        self.assertEqual(spec.granularity, "per_tensor")
        self.assertEqual(spec.activation_granularity, "per_channel")
        self.assertEqual(spec.calibration_steps, 2)
        self.assertEqual(spec.finetune_steps, 7)
        self.assertEqual(spec.scope, "weight_and_activation")
        self.assertFalse(spec.symmetric)
        self.assertTrue(spec.activation_symmetric)
        self.assertTrue(spec.enabled)
        self.assertFalse(spec.export_onnx)

    def test_schema_rejects_unknown_width_and_fields(self) -> None:
        loaded = load_recipe_file(example_path("act"))
        wide = apply_overrides(loaded, ["qat.bit_width=3"])
        self.assertTrue(any("bit_width" in error for error in validate_recipe(wide)))
        extra = apply_overrides(loaded, ["qat.note=nope"])
        self.assertTrue(any("note" in error for error in validate_recipe(extra)))
        self.assertTrue(spec_from_mapping({"bit_width": 8}).enabled)

    def test_activation_fields_follow_the_weight_fields(self) -> None:
        spec = spec_from_mapping({"granularity": "per_tensor", "symmetric": False})
        self.assertEqual(spec.activation_granularity, "per_tensor")
        self.assertFalse(spec.activation_symmetric)
        self.assertEqual(spec.scope, "weight_only")
        self.assertEqual(spec.calibration_steps, 0)

    def test_weight_and_activation_needs_a_warm_up(self) -> None:
        with self.assertRaises(QatError) as caught:
            prepare_qat_recipe(
                {"scope": "weight_and_activation", "calibration_steps": 0},
                family_id="act",
            )
        self.assertIn("calibration_steps", str(caught.exception))
        self.assertIn("No fallback", str(caught.exception))

    def test_act_refuses_onnx_export_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, events, err = _run(
                example_path("act"),
                ["qat.export_onnx=true", f"output.dir={out}"],
            )
        self.assertEqual(code, EXIT_INVALID, err)
        self.assertIn("export_onnx", err)
        self.assertIn("No fallback", err)
        self.assertFalse(any(event.get("event") == "train_start" for event in events))
        self.assertFalse(out.exists())

    def test_weight_only_with_no_warm_up_reaches_the_dataset_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-dataset"
            code, events, err = _run(
                example_path("act"),
                [
                    "qat.scope=weight_only",
                    "qat.calibration_steps=0",
                    "qat.export_onnx=false",
                    f"dataset.path={missing}",
                ],
            )
        self.assertEqual(code, EXIT_DATASET, err)
        self.assertIn("does not exist", err)
        self.assertFalse(any(event.get("event") == "train_start" for event in events))

    def test_manifest_records_resolved_settings_and_null(self) -> None:
        class _Family:
            id = "qat_record"

            def available(self) -> bool:
                return True

            def run(self, recipe, emit):
                del recipe
                out = Path(self.directory)
                out.mkdir(parents=True, exist_ok=True)
                emit({"event": "train_start", "family": self.id, "device": "cpu"})
                metrics = out / "metrics.json"
                metrics.write_text(
                    json.dumps(
                        {
                            "final_loss": 0.1,
                            "throughput_steps_per_s": 1.0,
                            "peak_memory_bytes": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                return TrainResult(str(out), str(metrics), "cpu", False)

        family = _Family()
        register(family, source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = root / "recipe.yaml"
                recipe.write_text(
                    "\n".join(
                        [
                            "schema: ohho.recipe/v1",
                            "family: qat_record",
                            "dataset:",
                            "  version_id: dv_qat",
                            "output:",
                            "  dir: placeholder",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                out = root / "out"
                family.directory = out
                code, _events, err = _run(
                    recipe,
                    [
                        f"output.dir={out}",
                        "qat.bit_width=4",
                        "qat.granularity=per_tensor",
                        "qat.calibration_steps=3",
                        "qat.finetune_steps=5",
                    ],
                )
                self.assertEqual(code, EXIT_OK, err)
                manifest = json.loads(
                    (out / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(validate_run_manifest(manifest), [])
                self.assertEqual(manifest["qat"]["bit_width"], 4)
                self.assertEqual(manifest["qat"]["granularity"], "per_tensor")
                self.assertEqual(
                    manifest["qat"]["activation_granularity"], "per_tensor"
                )
                self.assertEqual(manifest["qat"]["calibration_steps"], 3)
                self.assertEqual(manifest["qat"]["finetune_steps"], 5)
                self.assertEqual(manifest["qat"]["scope"], "weight_only")
                self.assertTrue(manifest["qat"]["symmetric"])
                self.assertIn("qat:", format_run_text(manifest))
                bare = dict(manifest)
                del bare["qat"]
                self.assertEqual(validate_run_manifest(bare), [])
                self.assertIsNone(settings_from_recipe({}))
        finally:
            unregister("qat_record")

    def test_capture_without_a_block_is_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recipe_path = Path(tmp) / "recipe.yaml"
            recipe_path.write_text("schema: ohho.recipe/v1\n", encoding="utf-8")
            recipe = {
                "schema": "ohho.recipe/v1",
                "family": "act",
                "precision": "fp32",
                "dataset": {"version_id": "dv"},
                "output": {"dir": tmp},
            }
            capture = capture_run(
                recipe=recipe,
                recipe_path=str(recipe_path),
                overrides=[],
                device="cpu",
                started_at="2026-01-01T00:00:00Z",
            )
            document = manifest_from_capture(
                capture,
                status="running",
                finished_at=None,
                error=None,
                checkpoint_dir=None,
                metrics_path=None,
                final_loss=None,
                base_weights_loaded=False,
            ).to_dict()
        self.assertIsNone(document["qat"])
        self.assertEqual(validate_run_manifest(document), [])


@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestFakeQuant(unittest.TestCase):
    def test_wrappers_cover_width_granularity_and_symmetry(self) -> None:
        import torch

        from ohho.compression.qat import TinyActLike

        cases = (
            (8, "per_tensor", True, "weight_only", -128, 127),
            (8, "per_channel", False, "weight_and_activation", 0, 255),
            (4, "per_channel", True, "weight_only", -8, 7),
            (4, "per_tensor", False, "weight_and_activation", 0, 15),
        )
        for bit_width, granularity, symmetric, scope, qmin, qmax in cases:
            with self.subTest(
                bit_width=bit_width, granularity=granularity, scope=scope
            ):
                model = TinyActLike()
                spec = spec_from_mapping(
                    {
                        "bit_width": bit_width,
                        "granularity": granularity,
                        "symmetric": symmetric,
                        "scope": scope,
                        "calibration_steps": 1,
                    }
                )
                insert_fake_quant(model, spec)
                wrapper = model.state_proj
                self.assertEqual(type(wrapper).__name__, "FakeQuantLinear")
                self.assertEqual(type(wrapper.weight_fq).__name__, "FakeQuantize")
                observer = wrapper.weight_fq.activation_post_process
                self.assertEqual(observer.quant_min, qmin)
                self.assertEqual(observer.quant_max, qmax)
                if scope == "weight_only":
                    self.assertIsNone(wrapper.act_fq)
                else:
                    self.assertIsNotNone(wrapper.act_fq)
                sample = torch.randn(2, 4)
                run_calibration(model, 1, lambda: model(sample))
                output = model(sample)
                self.assertEqual(tuple(output.shape), (2, 4))
                self.assertTrue(torch.isfinite(output).all())
                if symmetric:
                    self.assertTrue(torch.all(wrapper.weight_fq.zero_point == 0))
                unwrap_fake_quant(model)
                self.assertIsInstance(model.state_proj, torch.nn.Linear)

    def test_per_channel_scales_follow_the_row_magnitude(self) -> None:
        import torch

        from ohho.compression.qat import TinyActLike

        model = TinyActLike()
        spec = spec_from_mapping(
            {"scope": "weight_only", "granularity": "per_channel", "symmetric": True}
        )
        insert_fake_quant(model, spec)
        with torch.no_grad():
            model.state_proj.linear.weight.zero_()
            model.state_proj.linear.weight[0].fill_(3.0)
            model.state_proj.linear.weight[1].fill_(0.05)
        run_calibration(model, 0, lambda: None)
        scale = model.state_proj.weight_fq.scale.detach().flatten()
        self.assertEqual(int(scale.numel()), int(model.state_proj.linear.out_features))
        self.assertGreater(float(scale[0]), float(scale[1]))

    def test_finetune_updates_master_weights(self) -> None:
        import torch

        from ohho.compression.qat import TinyActLike

        torch.manual_seed(0)
        model = TinyActLike()
        before = model.state_proj.weight.detach().clone()
        spec = spec_from_mapping(
            {
                "scope": "weight_only",
                "bit_width": 8,
                "granularity": "per_tensor",
                "calibration_steps": 1,
                "finetune_steps": 5,
            }
        )
        inputs = torch.randn(8, 4)
        target = torch.randn(8, 4)
        insert_fake_quant(model, spec)
        run_calibration(model, 1, lambda: model(inputs))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        def loss_once() -> torch.Tensor:
            return torch.nn.functional.mse_loss(model(inputs), target)

        last = run_finetune(model, 5, optimizer, loss_once)
        unwrap_fake_quant(model)
        self.assertIsNotNone(last)
        self.assertTrue(math.isfinite(float(last)))
        self.assertFalse(torch.allclose(before, model.state_proj.weight))

    def test_act_transformer_is_not_a_qdq_chain(self) -> None:
        import torch

        from ohho.train.act import TinyACTPolicy

        model = TinyACTPolicy(
            state_dim=4,
            action_dim=4,
            chunk_size=2,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            latent_dim=4,
        )
        spec = spec_from_mapping(
            {"scope": "weight_only", "calibration_steps": 0, "export_onnx": True}
        )
        insert_fake_quant(model, spec)
        run_calibration(model, 0, lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(QatError) as caught:
                export_qdq_onnx(model, torch.randn(1, 4), Path(tmp) / "qdq.onnx")
        self.assertIn("TinyActLike", str(caught.exception))
        self.assertIn("No ONNX file was written", str(caught.exception))

    def test_compare_ptq_and_qat_reports_cpu_losses_only(self) -> None:
        spec = spec_from_mapping(
            {
                "scope": "weight_only",
                "bit_width": 8,
                "granularity": "per_tensor",
                "calibration_steps": 2,
                "finetune_steps": 3,
            }
        )
        report = compare_ptq_qat(spec, batch=4, eval_batch=2, d_model=4)
        self.assertEqual(
            report["method"], "shared_calibration_then_optional_fake_quant_finetune"
        )
        self.assertEqual(report["ptq_finetune_steps"], 0)
        self.assertEqual(report["qat_finetune_steps"], 3)
        self.assertTrue(math.isfinite(report["ptq_eval_loss"]))
        self.assertTrue(math.isfinite(report["qat_eval_loss"]))
        self.assertFalse(report["gpu_measured"])
        self.assertFalse(report["tensorrt_measured"])


class TestTensorRTCapability(unittest.TestCase):
    def test_missing_tensorrt_is_a_capability_error(self) -> None:
        if importlib.util.find_spec("tensorrt") is not None:
            self.skipTest("tensorrt is installed")
        with self.assertRaises(QatCapabilityError) as caught:
            build_tensorrt_engine("policy.qdq.onnx")
        text = str(caught.exception)
        self.assertIn("TensorRT", text)
        self.assertIn("No fallback", text)

    def test_missing_onnx_is_a_capability_error(self) -> None:
        from ohho.compression import qat as qat_module

        with patch.dict("sys.modules", {"onnx": None}):
            with self.assertRaises(QatCapabilityError) as caught:
                qat_module._import_onnx()
        self.assertIn("onnx", str(caught.exception))
        self.assertIn("No fallback", str(caught.exception))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ONNX and _HAS_ORT, "onnx runtime is not installed"
)
class TestQdqExport(unittest.TestCase):
    def test_tiny_act_like_matches_onnx_runtime(self) -> None:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        from ohho.compression.qat import TinyActLike

        cases = (
            {
                "bit_width": 8,
                "granularity": "per_channel",
                "symmetric": True,
                "scope": "weight_only",
                "calibration_steps": 1,
                "finetune_steps": 1,
            },
            {
                "bit_width": 8,
                "granularity": "per_tensor",
                "symmetric": False,
                "scope": "weight_and_activation",
                "activation_granularity": "per_tensor",
                "calibration_steps": 2,
                "finetune_steps": 1,
            },
            {
                "bit_width": 4,
                "granularity": "per_channel",
                "symmetric": True,
                "scope": "weight_only",
                "calibration_steps": 1,
                "finetune_steps": 1,
            },
            {
                "bit_width": 4,
                "granularity": "per_tensor",
                "symmetric": False,
                "scope": "weight_and_activation",
                "activation_granularity": "per_channel",
                "activation_symmetric": False,
                "calibration_steps": 2,
                "finetune_steps": 1,
            },
            {
                "bit_width": 8,
                "granularity": "per_channel",
                "symmetric": False,
                "scope": "weight_and_activation",
                "activation_granularity": "per_tensor",
                "activation_symmetric": True,
                "calibration_steps": 2,
                "finetune_steps": 0,
            },
            {
                "bit_width": 4,
                "granularity": "per_channel",
                "symmetric": True,
                "scope": "weight_and_activation",
                "activation_granularity": "per_channel",
                "activation_symmetric": False,
                "calibration_steps": 2,
                "finetune_steps": 1,
            },
        )
        for fields in cases:
            label = (
                fields["bit_width"],
                fields["granularity"],
                fields["symmetric"],
                fields["scope"],
            )
            with self.subTest(label=label):
                torch.manual_seed(0)
                model = TinyActLike()
                spec = spec_from_mapping(fields)
                inputs = torch.randn(3, 4)
                target = torch.randn(3, 4)
                insert_fake_quant(model, spec)
                run_calibration(model, spec.calibration_steps, lambda: model(inputs))
                if spec.finetune_steps:
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

                    def loss_once() -> torch.Tensor:
                        return torch.nn.functional.mse_loss(model(inputs), target)

                    run_finetune(model, spec.finetune_steps, optimizer, loss_once)
                model.eval()
                with torch.no_grad():
                    reference = model(inputs).detach().cpu().numpy()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "qdq.onnx"
                    export_qdq_onnx(model, inputs, path)
                    document = onnx.load(path)
                    ops = [node.op_type for node in document.graph.node]
                    self.assertIn("QuantizeLinear", ops)
                    self.assertIn("DequantizeLinear", ops)
                    self.assertEqual(
                        ops.count("QuantizeLinear"), ops.count("DequantizeLinear")
                    )
                    session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                    output = session.run(None, {"state": inputs.numpy()})[0]
                self.assertLessEqual(
                    float(np.max(np.abs(output - reference))),
                    1e-4,
                )


@unittest.skipUnless(_HAS_TORCH, "torch is not installed")
class TestActRecipeQat(unittest.TestCase):
    def test_train_runs_qat_and_records_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset)
            out = root / "out"
            recipe = root / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "schema": "ohho.recipe/v1",
                        "family": "act",
                        "dataset": {"path": str(dataset)},
                        "hyperparameters": {
                            "train_steps": 1,
                            "batch_size": 2,
                            "lr": 0.01,
                            "chunk_size": 2,
                            "d_model": 8,
                            "nhead": 2,
                            "dim_feedforward": 16,
                            "latent_dim": 4,
                            "log_every": 1,
                            "seed": 0,
                        },
                        "precision": "fp32",
                        "device": "cpu",
                        "qat": {
                            "bit_width": 4,
                            "scope": "weight_only",
                            "granularity": "per_tensor",
                            "symmetric": True,
                            "calibration_steps": 1,
                            "finetune_steps": 1,
                            "export_onnx": False,
                        },
                        "output": {"dir": str(out)},
                    }
                ),
                encoding="utf-8",
            )
            code, events, err = _run(recipe)
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads(
                (out / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertEqual(manifest["qat"]["bit_width"], 4)
            self.assertEqual(manifest["qat"]["finetune_steps"], 1)
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["qat"]["ran"])
            self.assertEqual(metrics["qat"]["bit_width"], 4)
            self.assertIsNotNone(metrics["qat"]["finetune_loss"])
            self.assertTrue((out / "policy.pt").is_file())
            self.assertTrue(
                any(event.get("event") == "train_start" for event in events)
            )
            from ohho.train.act import load_act_checkpoint

            model, config = load_act_checkpoint(str(out), device="cpu")
            self.assertIsNotNone(model)
            self.assertEqual(config["qat"]["bit_width"], 4)
            action = model.select_action([0.1, 0.1, 0.1, 0.1])
            self.assertEqual(len(action), 4)


def _write_dataset(root: Path, frames: int = 4, dim: int = 4) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "2.0",
                "total_episodes": 1,
                "total_frames": frames,
                "features": {
                    "observation.state": {"shape": [dim]},
                    "action": {"shape": [dim]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": frames}) + "\n",
        encoding="utf-8",
    )
    lines = []
    for index in range(frames):
        lines.append(
            json.dumps(
                {
                    "observation.state": [0.1 * (index + 1)] * dim,
                    "action": [0.05 * (index + 1)] * dim,
                    "timestamp": index * 0.1,
                    "frame_index": index,
                    "episode_index": 0,
                    "task_index": 0,
                    "next.done": index == frames - 1,
                }
            )
        )
    (data / "episode_000000.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    unittest.main()
