"""Unit tests for the CPU sim loop (Record -> Validate -> Train -> Serve -> Eval)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main as cli_main
from ohho.data.reader import DatasetReader
from ohho.sim_loop import (
    compute_percentiles,
    format_summary,
    record_sim_episodes,
    run_sim_loop,
    validate_dataset,
    validate_lerobot_dataset,
    validate_schema,
)


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _has_lerobot() -> bool:
    return importlib.util.find_spec("lerobot") is not None


class TestSimLoop(unittest.TestCase):
    def test_record_sim_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            saved_path = record_sim_episodes(
                robot_id="omnibot",
                num_episodes=5,
                steps_per_episode=10,
                fps=10.0,
                output_dir=dataset_dir,
            )
            self.assertEqual(saved_path, dataset_dir)

            # Check meta files
            meta = Path(saved_path) / "meta"
            self.assertTrue((meta / "info.json").exists())
            self.assertTrue((meta / "tasks.jsonl").exists())
            self.assertTrue((meta / "episodes.jsonl").exists())
            self.assertTrue((meta / "episodes_stats.jsonl").exists())

            # Check data files
            data_dir = Path(saved_path) / "data" / "chunk-000"
            for ep in range(5):
                pq = data_dir / f"episode_{ep:06d}.parquet"
                jl = data_dir / f"episode_{ep:06d}.jsonl"
                self.assertTrue(pq.exists() or jl.exists())

            reader = DatasetReader(saved_path)
            self.assertEqual(reader.episode_count, 5)
            self.assertEqual(reader.frame_count, 50)
            self.assertEqual(reader.state_dim, 9)
            self.assertEqual(reader.action_dim, 9)

            with open(meta / "episodes_stats.jsonl", encoding="utf-8") as f:
                stat_rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(stat_rows), 5)
            for row in stat_rows:
                for key in ("observation.state", "action"):
                    feat = row["stats"][key]
                    self.assertEqual(feat["count"], [10])
                    for stat_key in ("min", "max", "mean", "std"):
                        val = feat[stat_key]
                        self.assertEqual(len(val), 9)
                        self.assertFalse(any(isinstance(x, list) for x in val))

    def test_validate_schema_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=5,
                steps_per_episode=10,
                output_dir=dataset_dir,
            )
            self.assertTrue(validate_schema(dataset_dir))

            val_info = validate_dataset(dataset_dir, mock=True)
            self.assertTrue(val_info["schema_valid"])
            self.assertEqual(val_info["total_episodes"], 5)
            self.assertEqual(val_info["total_frames"], 50)

    def test_validate_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Empty directory
            with self.assertRaises(FileNotFoundError):
                validate_schema(tmp)

            # Corrupted info.json
            meta = Path(tmp) / "meta"
            meta.mkdir(parents=True)
            with open(meta / "info.json", "w", encoding="utf-8") as f:
                json.dump({"total_episodes": 0, "total_frames": 0}, f)
            with open(meta / "tasks.jsonl", "w", encoding="utf-8") as f:
                f.write('{"task": "test"}\n')
            with open(meta / "episodes.jsonl", "w", encoding="utf-8") as f:
                f.write('{"episode_index": 0}\n')

            with self.assertRaises(ValueError):
                validate_schema(tmp)

    def test_compute_percentiles(self):
        # Empty list
        self.assertEqual(compute_percentiles([]), (0.0, 0.0))

        # Single element
        self.assertEqual(compute_percentiles([42.0]), (42.0, 42.0))

        # Known range 1..100
        vals = list(range(1, 101))
        p50, p95 = compute_percentiles([float(x) for x in vals])
        self.assertAlmostEqual(p50, 50.5, delta=1.0)
        self.assertAlmostEqual(p95, 95.05, delta=1.5)

    def test_format_summary(self):
        metrics_mock = {
            "episodes_recorded": 5,
            "total_frames": 100,
            "schema_valid": True,
            "lerobot_dataset_loaded": False,
            "mock": True,
            "model_name": "Tiny ACT (Action Chunking Transformer)",
            "train_steps": 200,
            "initial_loss": 0.8421,
            "final_loss": 0.0412,
            "eval_steps_taken": 50,
            "eval_errors": 0,
            "latency_p50_ms": 2.15,
            "latency_p95_ms": 4.50,
            "loss_curve": [(0, 0.8421), (100, 0.2011), (200, 0.0412)],
        }

        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
            tmp_path = tmp.name

        try:
            os.environ["GITHUB_STEP_SUMMARY"] = tmp_path
            summary = format_summary(metrics_mock)

            self.assertIn("OhhO OS Nightly Sim Loop Summary", summary)
            self.assertIn("Sim Episodes Recorded", summary)
            self.assertIn("5 (100 total frames)", summary)
            self.assertIn("initial=0.8421 -> final=0.0412", summary)
            self.assertIn("50 steps (0 errors)", summary)
            self.assertIn("2.15 ms", summary)
            self.assertIn("4.50 ms", summary)
            self.assertIn("Train Loss Curve", summary)
            self.assertIn("Skipped (mock mode)", summary)

            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content.strip(), summary.strip())
        finally:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Non-mock summary: when lerobot_dataset_loaded is False and mock is False, status is Fail
        metrics_non_mock = dict(metrics_mock)
        metrics_non_mock["mock"] = False
        summary_fail = format_summary(metrics_non_mock)
        self.assertIn("| **LeRobotDataset Loading** | Fail |", summary_fail)

        # Pass case: when lerobot_dataset_loaded is True
        metrics_pass = dict(metrics_mock)
        metrics_pass["lerobot_dataset_loaded"] = True
        summary_pass = format_summary(metrics_pass)
        self.assertIn("| **LeRobotDataset Loading** | Pass |", summary_pass)

    def test_validation_strictness_non_mock_lerobot_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=2,
                steps_per_episode=5,
                output_dir=dataset_dir,
            )

            with patch.dict(
                sys.modules,
                {
                    "lerobot": None,
                    "lerobot.datasets": None,
                    "lerobot.datasets.lerobot_dataset": None,
                    "lerobot.common.datasets.lerobot_dataset": None,
                },
            ):
                # In non-mock mode, missing lerobot must raise RuntimeError
                with self.assertRaises(RuntimeError) as cm:
                    validate_lerobot_dataset(dataset_dir, mock=False)
                self.assertIn("lerobot is not installed", str(cm.exception))

                with self.assertRaises(RuntimeError) as cm_ds:
                    validate_dataset(dataset_dir, mock=False)
                self.assertIn("lerobot is not installed", str(cm_ds.exception))

                # In mock mode, missing lerobot must return False without raising
                self.assertFalse(validate_lerobot_dataset(dataset_dir, mock=True))
                val_info = validate_dataset(dataset_dir, mock=True)
                self.assertFalse(val_info["lerobot_dataset_loaded"])
                self.assertTrue(val_info["schema_valid"])

    def test_validation_strictness_non_mock_loader_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=2,
                steps_per_episode=5,
                output_dir=dataset_dir,
            )

            class RaisingDataset:
                def __init__(self, *args, **kwargs):
                    raise ValueError("corrupted parquet data chunk")

            fake_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
            fake_mod.LeRobotDataset = RaisingDataset
            fake_pkg = types.ModuleType("lerobot.datasets")
            fake_pkg.LeRobotDataset = RaisingDataset

            with patch.dict(
                sys.modules,
                {
                    "lerobot": types.ModuleType("lerobot"),
                    "lerobot.datasets": fake_pkg,
                    "lerobot.datasets.lerobot_dataset": fake_mod,
                    "lerobot.common.datasets.lerobot_dataset": fake_mod,
                },
            ):
                # In non-mock mode, loader error must raise RuntimeError
                with self.assertRaises(RuntimeError) as cm:
                    validate_lerobot_dataset(dataset_dir, mock=False)
                self.assertIn("corrupted parquet data chunk", str(cm.exception))

                with self.assertRaises(RuntimeError) as cm_ds:
                    validate_dataset(dataset_dir, mock=False)
                self.assertIn("corrupted parquet data chunk", str(cm_ds.exception))

                # In mock mode, loader error must return False without raising
                self.assertFalse(validate_lerobot_dataset(dataset_dir, mock=True))
                val_info = validate_dataset(dataset_dir, mock=True)
                self.assertFalse(val_info["lerobot_dataset_loaded"])

    def test_validation_strictness_non_mock_zero_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=2,
                steps_per_episode=5,
                output_dir=dataset_dir,
            )

            class EmptyDataset:
                def __init__(self, *args, **kwargs):
                    pass

                def __len__(self):
                    return 0

            fake_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
            fake_mod.LeRobotDataset = EmptyDataset
            fake_pkg = types.ModuleType("lerobot.datasets")
            fake_pkg.LeRobotDataset = EmptyDataset

            with patch.dict(
                sys.modules,
                {
                    "lerobot": types.ModuleType("lerobot"),
                    "lerobot.datasets": fake_pkg,
                    "lerobot.datasets.lerobot_dataset": fake_mod,
                    "lerobot.common.datasets.lerobot_dataset": fake_mod,
                },
            ):
                with self.assertRaises(RuntimeError) as cm:
                    validate_lerobot_dataset(dataset_dir, mock=False)
                self.assertIn("0 frames", str(cm.exception))

                self.assertFalse(validate_lerobot_dataset(dataset_dir, mock=True))

    def test_validate_lerobot_dataset_uses_info_repo_and_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=1,
                steps_per_episode=3,
                output_dir=dataset_dir,
            )

            captured: dict = {}

            class ValidDataset:
                def __init__(self, *args, **kwargs):
                    captured.update(kwargs)

                def __len__(self):
                    return 3

            fake_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
            fake_mod.LeRobotDataset = ValidDataset
            fake_pkg = types.ModuleType("lerobot.datasets")
            fake_pkg.LeRobotDataset = ValidDataset

            with patch.dict(
                sys.modules,
                {
                    "lerobot": types.ModuleType("lerobot"),
                    "lerobot.datasets": fake_pkg,
                    "lerobot.datasets.lerobot_dataset": fake_mod,
                    "lerobot.common.datasets.lerobot_dataset": fake_mod,
                },
            ):
                prev = os.environ.pop("HF_HUB_OFFLINE", None)
                try:
                    self.assertTrue(validate_lerobot_dataset(dataset_dir, mock=False))
                finally:
                    if prev is not None:
                        os.environ["HF_HUB_OFFLINE"] = prev

            self.assertEqual(captured.get("repo_id"), "local/sim_loop")
            self.assertEqual(captured.get("revision"), "v2.1")
            self.assertEqual(captured.get("root"), dataset_dir)
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), prev)

    def test_validation_strictness_non_mock_loader_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = os.path.join(tmp, "dataset")
            record_sim_episodes(
                robot_id="omnibot",
                num_episodes=2,
                steps_per_episode=5,
                output_dir=dataset_dir,
            )

            class ValidDataset:
                def __init__(self, *args, **kwargs):
                    pass

                def __len__(self):
                    return 10

            fake_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
            fake_mod.LeRobotDataset = ValidDataset
            fake_pkg = types.ModuleType("lerobot.datasets")
            fake_pkg.LeRobotDataset = ValidDataset

            with patch.dict(
                sys.modules,
                {
                    "lerobot": types.ModuleType("lerobot"),
                    "lerobot.datasets": fake_pkg,
                    "lerobot.datasets.lerobot_dataset": fake_mod,
                    "lerobot.common.datasets.lerobot_dataset": fake_mod,
                },
            ):
                self.assertTrue(validate_lerobot_dataset(dataset_dir, mock=False))
                val_info = validate_dataset(dataset_dir, mock=False)
                self.assertTrue(val_info["schema_valid"])
                self.assertTrue(val_info["lerobot_dataset_loaded"])

    def test_validation_strictness_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Missing files should raise FileNotFoundError in both mock and non-mock
            with self.assertRaises(FileNotFoundError):
                validate_dataset(tmp, mock=False)
            with self.assertRaises(FileNotFoundError):
                validate_dataset(tmp, mock=True)

    def test_cli_non_mock_exits_nonzero_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Running CLI in non-mock mode without lerobot must exit non-zero
            with patch.dict(
                sys.modules,
                {
                    "lerobot": None,
                    "lerobot.datasets": None,
                    "lerobot.datasets.lerobot_dataset": None,
                    "lerobot.common.datasets.lerobot_dataset": None,
                },
            ):
                rc = cli_main(
                    [
                        "sim-loop",
                        "--episodes",
                        "2",
                        "--steps-per-episode",
                        "2",
                        "--eval-steps",
                        "2",
                        "--output-dir",
                        tmp,
                    ]
                )
                self.assertNotEqual(rc, 0)

    @unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
    def test_mock_sim_loop_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = run_sim_loop(
                robot_id="omnibot",
                episodes=5,
                steps_per_episode=10,
                train_steps=100,
                eval_steps=50,
                output_dir=tmp,
                mock=True,
            )

            self.assertEqual(results["episodes_recorded"], 5)
            self.assertEqual(results["total_frames"], 50)
            self.assertTrue(results["schema_valid"])
            self.assertEqual(results["eval_steps_taken"], 50)
            self.assertEqual(results["eval_errors"], 0)
            self.assertGreaterEqual(results["latency_p50_ms"], 0.0)
            self.assertGreaterEqual(results["latency_p95_ms"], 0.0)
            self.assertIn("loss_curve", results)
            self.assertTrue(len(results["loss_curve"]) > 0)

            # Check output files
            ckpt_dir = Path(tmp) / "checkpoint"
            self.assertTrue((ckpt_dir / "config.json").exists())
            self.assertTrue((ckpt_dir / "metrics.json").exists())

    @unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
    def test_cli_sim_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli_main(
                [
                    "sim-loop",
                    "--mock",
                    "--episodes",
                    "5",
                    "--steps-per-episode",
                    "5",
                    "--eval-steps",
                    "50",
                    "--output-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)

    def test_finetune_act_train_steps_forwarding(self):
        """OHH-86: finetune(policy='act') must forward train_steps without TypeError."""
        from ohho.train import finetune

        # Test fallback path when _lerobot_train raises ImportError
        with (
            patch("ohho.train._lerobot_train", side_effect=ImportError("no lerobot")),
            patch("ohho.train.act.train_act", return_value="dummy_ckpt") as mock_act,
        ):
            res = finetune(
                "fake_dataset",
                policy="act",
                train_steps=42,
                mock=False,
                extra_arg="custom",
            )
            self.assertEqual(res, "dummy_ckpt")
            mock_act.assert_called_once()
            _, kwargs = mock_act.call_args
            self.assertEqual(kwargs["train_steps"], 42)
            self.assertEqual(kwargs["extra_arg"], "custom")
            self.assertFalse(kwargs["mock"])

        # Test default train_steps calculation (num_epochs * 2) in fallback
        with (
            patch("ohho.train._lerobot_train", side_effect=ImportError("no lerobot")),
            patch("ohho.train.act.train_act", return_value="dummy_ckpt") as mock_act,
        ):
            finetune("fake_dataset", policy="act", num_epochs=15, mock=False)
            _, kwargs = mock_act.call_args
            self.assertEqual(kwargs["train_steps"], 30)

        # Test mock=True path with explicit train_steps
        with patch("ohho.train.act.train_act", return_value="dummy_ckpt") as mock_act:
            finetune("fake_dataset", policy="act", train_steps=75, mock=True)
            _, kwargs = mock_act.call_args
            self.assertEqual(kwargs["train_steps"], 75)
            self.assertTrue(kwargs["mock"])

    @unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
    def test_serve_act_forwards_device(self):
        """OHH-86: build_app and /load_model must forward device to ACTModel."""
        from fastapi.testclient import TestClient
        from ohho.serve import build_app

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "act_ckpt"
            ckpt_dir.mkdir()
            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"policy": "act", "state_dim": 9, "action_dim": 9}, f)

            with patch("ohho.serve.resolve_device", return_value="cuda:0"):
                # Test auto_load path forwards device
                app = build_app(
                    model_class="act",
                    model_path=str(ckpt_dir),
                    device="cuda:0",
                    auto_load=True,
                )
                self.assertIsNotNone(app.state.model)
                self.assertEqual(app.state.model.device, "cuda:0")

                # Test /load_model path forwards device
                app2 = build_app(
                    model_class="act",
                    device="cuda:0",
                    auto_load=False,
                )
                client = TestClient(app2)
                r = client.post("/load_model", params={"model_path": str(ckpt_dir)})
                self.assertEqual(r.status_code, 200)
                self.assertIsNotNone(app2.state.model)
                self.assertEqual(app2.state.model.device, "cuda:0")

    @unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
    def test_serve_act_auto_detect_positive_marker_only(self):
        """OHH-86: only auto-detect ACT on positive ACT marker, not generic HF config.json."""
        from fastapi.testclient import TestClient
        from ohho.serve import _is_act_checkpoint, build_app

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. ACT checkpoint directory with positive marker
            act_dir = tmp_path / "act_checkpoint"
            act_dir.mkdir()
            with open(act_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"policy": "act", "state_dim": 9, "action_dim": 9}, f)
            (act_dir / "policy.pt").touch()

            self.assertTrue(_is_act_checkpoint(str(act_dir)))
            self.assertTrue(_is_act_checkpoint(str(act_dir / "policy.pt")))

            from ohho.serve.act import ACTModel

            with patch.object(ACTModel, "load_model") as mock_load:
                app_act = build_app(model_path=str(act_dir), auto_load=True)
                self.assertEqual(type(app_act.state.model).__name__, "ACTModel")
                mock_load.assert_called_once()

                client_act = TestClient(build_app(auto_load=False))
                r = client_act.post("/load_model", params={"model_path": str(act_dir)})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["model"], "ACTModel")
                self.assertEqual(mock_load.call_count, 2)

            # 2. Generic Hugging Face checkpoint directory with config.json
            hf_dir = tmp_path / "hf_checkpoint"
            hf_dir.mkdir()
            with open(hf_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": "openvla",
                        "architectures": ["OpenVLAForActionPrediction"],
                    },
                    f,
                )
            (hf_dir / "model.safetensors").touch()

            self.assertFalse(_is_act_checkpoint(str(hf_dir)))
            self.assertFalse(_is_act_checkpoint("openvla/openvla-7b"))

            # When loading HF checkpoint without model_class, it must NOT resolve to ACTModel
            with patch("ohho.serve._resolve_model_class") as mock_resolve:
                mock_cls = unittest.mock.MagicMock()
                mock_resolve.return_value = mock_cls
                build_app(model_path=str(hf_dir), auto_load=True)
                mock_resolve.assert_called_with("")
                self.assertNotEqual(mock_resolve.call_args[0][0], "act")

                client = TestClient(build_app(auto_load=False))
                client.post("/load_model", params={"model_path": str(hf_dir)})
                # /load_model also retains default model_class ("") instead of "act"
                self.assertEqual(mock_resolve.call_args[0][0], "")

    def test_train_lazy_exports(self):
        """OHH-86: ohho.train must lazily export train_act, TinyACTPolicy, load_act_checkpoint."""
        import ohho.train

        for name in ("train_act", "TinyACTPolicy", "load_act_checkpoint"):
            self.assertIn(name, ohho.train.__all__)
            obj = getattr(ohho.train, name)
            self.assertIsNotNone(obj)

        with self.assertRaises(AttributeError):
            getattr(ohho.train, "non_existent_export_symbol")

    def test_load_act_checkpoint_enforces_weights_only(self):
        """OHH-86: load_act_checkpoint must deserialize policy.pt with weights_only=True."""
        import ohho.train.act as act_module
        from ohho.train.act import load_act_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "act_ckpt"
            ckpt_dir.mkdir()
            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"policy": "act", "state_dim": 9, "action_dim": 9}, f)
            (ckpt_dir / "policy.pt").write_bytes(b"dummy_weights")

            mock_torch = unittest.mock.MagicMock()
            mock_model = unittest.mock.MagicMock()
            mock_policy_cls = unittest.mock.MagicMock(return_value=mock_model)

            with (
                patch.object(act_module, "TORCH_AVAILABLE", True),
                patch.object(act_module, "torch", mock_torch, create=True),
                patch.object(act_module, "TinyACTPolicy", mock_policy_cls),
            ):
                model, config = load_act_checkpoint(str(ckpt_dir), device="cpu")
                mock_torch.load.assert_called_once()
                _args, kwargs = mock_torch.load.call_args
                self.assertEqual(kwargs.get("weights_only"), True)
                self.assertEqual(model, mock_model)

    @unittest.skipUnless(_has_torch(), "torch not installed — [train] extra")
    def test_act_checkpoint_save_and_load_roundtrip_weights_only(self):
        """OHH-86: saved TinyACTPolicy state_dict loads safely with weights_only=True."""
        import torch
        from ohho.train.act import TinyACTPolicy, load_act_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "real_ckpt"
            ckpt_dir.mkdir()
            model = TinyACTPolicy(state_dim=9, action_dim=9, chunk_size=5, d_model=32)
            torch.save(model.state_dict(), ckpt_dir / "policy.pt")
            with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "policy": "act",
                        "state_dim": 9,
                        "action_dim": 9,
                        "chunk_size": 5,
                        "d_model": 32,
                    },
                    f,
                )

            loaded_model, loaded_config = load_act_checkpoint(
                str(ckpt_dir), device="cpu"
            )
            self.assertIsNotNone(loaded_model)
            self.assertEqual(loaded_config["chunk_size"], 5)


if __name__ == "__main__":
    unittest.main()
