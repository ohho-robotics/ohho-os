import json
import os
import tempfile
import unittest

from ohho.adapters.sim import SimTransport
from ohho.data import Recorder
from ohho.data.reader import DatasetReader
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime


def _bot(robot_id="omnibot"):
    spec = get_spec(robot_id)
    tp = SimTransport(spec)
    tp.connect()
    return Robot(spec, tp, NativeRuntime())


class TestRecorder(unittest.TestCase):
    def test_start_and_stop_episode(self):
        bot = _bot("omnibot")
        rec = Recorder(bot, repo_id="local/test", fps=10.0)
        rec.start_episode(task="pick up the cup")
        bot.drive(vx=0.1)
        rec.capture_frame()
        bot.drive(vx=0.0, w=0.2)
        rec.capture_frame()
        ep = rec.stop_episode()
        self.assertEqual(ep.length, 2)
        self.assertEqual(ep.task, "pick up the cup")

    def test_capture_frame_state(self):
        bot = _bot("omnibot")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="test")
        bot.drive(vx=0.1)
        bot.transport.step(0.1)  # type: ignore[attr-defined]
        frame = rec.capture_frame()
        # 9-D state: 6 arm joints + 3 base velocity
        self.assertEqual(len(frame.observation_state), 9)
        self.assertAlmostEqual(frame.observation_state[6], 0.1, places=2)
        rec.stop_episode()

    def test_action_captured_from_drive(self):
        bot = _bot("sim")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="drive")
        bot.drive(vx=0.15, w=0.3)
        frame = rec.capture_frame()
        # sim has no arm → 3-D action
        self.assertEqual(len(frame.action), 3)
        self.assertAlmostEqual(frame.action[0], 0.15)
        self.assertAlmostEqual(frame.action[2], 0.3)
        rec.stop_episode()

    def test_action_captured_from_move_joints(self):
        bot = _bot("omnibot")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="arm")
        bot.move_joints([0.5, -0.3, 0.0, 0.0, 0.0, 0.2])
        frame = rec.capture_frame()
        # 9-D action: 6 arm + 3 base
        self.assertEqual(len(frame.action), 9)
        self.assertAlmostEqual(frame.action[0], 0.5)
        self.assertAlmostEqual(frame.action[5], 0.2)
        rec.stop_episode()

    def test_next_done_on_last_frame(self):
        bot = _bot("sim")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="test")
        rec.capture_frame()
        rec.capture_frame()
        ep = rec.stop_episode()
        self.assertFalse(ep.frames[0].next_done)
        self.assertTrue(ep.frames[-1].next_done)

    def test_discard_episode(self):
        bot = _bot("sim")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="discard me")
        rec.capture_frame()
        rec.discard_episode()
        self.assertEqual(len(rec.episodes), 0)

    def test_double_start_raises(self):
        bot = _bot("sim")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="a")
        with self.assertRaises(RuntimeError):
            rec.start_episode(task="b")
        rec.discard_episode()

    def test_state_dim_base_only(self):
        bot = _bot("unitree-go2")
        rec = Recorder(bot, fps=10.0)
        rec.start_episode(task="walk")
        frame = rec.capture_frame()
        # Go2 has no arm → 3-D state
        self.assertEqual(len(frame.observation_state), 3)
        rec.stop_episode()


class TestWriterReader(unittest.TestCase):
    def test_save_and_read_jsonl(self):
        bot = _bot("omnibot")
        rec = Recorder(bot, repo_id="local/test", fps=10.0)
        rec.start_episode(task="pick up the cup")
        for _ in range(5):
            bot.drive(vx=0.1)
            bot.transport.step(0.1)  # type: ignore[attr-defined]
            rec.capture_frame()
        rec.stop_episode()
        rec.start_episode(task="pick up the cup")
        for _ in range(3):
            bot.drive(vx=0.05, w=0.1)
            bot.transport.step(0.1)  # type: ignore[attr-defined]
            rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            # meta files exist
            self.assertTrue(os.path.exists(os.path.join(path, "meta", "info.json")))
            self.assertTrue(os.path.exists(os.path.join(path, "meta", "tasks.jsonl")))
            self.assertTrue(
                os.path.exists(os.path.join(path, "meta", "episodes.jsonl"))
            )
            # data file exists (jsonl or parquet)
            data_file = os.path.join(path, "data", "chunk-000", "episode_000000.jsonl")
            parquet_file = os.path.join(
                path, "data", "chunk-000", "episode_000000.parquet"
            )
            self.assertTrue(os.path.exists(data_file) or os.path.exists(parquet_file))

            # reader
            reader = DatasetReader(path)
            self.assertEqual(reader.episode_count, 2)
            self.assertEqual(reader.frame_count, 8)
            self.assertEqual(reader.state_dim, 9)
            self.assertEqual(reader.action_dim, 9)
            self.assertEqual(reader.task_text(0), "pick up the cup")

            frames = reader.load_episode(0)
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].frame_index, 0)
            self.assertTrue(frames[-1].next_done)

    def test_info_json_contents(self):
        bot = _bot("sim")
        rec = Recorder(bot, repo_id="local/sim", fps=30.0)
        rec.start_episode(task="drive")
        bot.drive(vx=0.1)
        rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            with open(os.path.join(path, "meta", "info.json")) as f:
                info = json.load(f)
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["fps"], 30.0)
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 1)
            # sim has no arm → 3-D
            self.assertEqual(info["features"]["observation.state"]["shape"], [3])
            self.assertEqual(info["features"]["action"]["shape"], [3])

    def test_all_frames_and_stats(self):
        bot = _bot("sim")
        rec = Recorder(bot, repo_id="local/sim", fps=10.0)
        rec.start_episode(task="a")
        for _ in range(3):
            bot.drive(vx=0.1)
            rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            reader = DatasetReader(path)
            all_frames = reader.all_frames()
            self.assertEqual(len(all_frames), 3)
            stats = reader.stats()
            self.assertIn("observation.state", stats)
            self.assertIn("action", stats)

    def test_episodes_stats_vector_shapes(self):
        bot = _bot("omnibot")
        rec = Recorder(bot, repo_id="local/test", fps=10.0)
        rec.start_episode(task="move")
        for _ in range(4):
            bot.drive(vx=0.1)
            rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            path = rec.save(tmp)
            reader = DatasetReader(path)
            state_dim = reader.state_dim
            action_dim = reader.action_dim
            stats_path = os.path.join(path, "meta", "episodes_stats.jsonl")
            with open(stats_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 1)
            ep_stats = rows[0]["stats"]
            self.assertEqual(ep_stats["observation.state"]["count"], [4])
            for key, dim in (
                ("observation.state", state_dim),
                ("action", action_dim),
            ):
                feat = ep_stats[key]
                for stat_key in ("min", "max", "mean", "std"):
                    val = feat[stat_key]
                    self.assertIsInstance(val, list)
                    self.assertEqual(len(val), dim)
                    self.assertFalse(any(isinstance(x, list) for x in val))


def _has_fastapi() -> bool:
    import importlib.util

    return importlib.util.find_spec("fastapi") is not None


def _has_lerobot() -> bool:
    import importlib.util

    return importlib.util.find_spec("lerobot") is not None


@unittest.skipUnless(_has_lerobot(), "lerobot not installed — [train] extra")
class TestFeatureStatsLayout(unittest.TestCase):
    """``_feature_stats`` must match lerobot 0.3.3 ``compute_episode_stats`` shapes."""

    def test_vector_and_scalar_shapes(self):
        import numpy as np

        from lerobot.datasets.compute_stats import compute_episode_stats
        from ohho.data.writer import _feature_stats

        vector_rows = [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]
        vector = _feature_stats(vector_rows)
        for key in ("min", "max", "mean", "std"):
            self.assertEqual(len(vector[key]), 3)

        scalar_rows = [[1.0], [3.0], [2.0]]
        scalar = _feature_stats(scalar_rows)
        self.assertEqual(scalar["min"], [1.0])
        self.assertEqual(scalar["max"], [3.0])
        self.assertEqual(scalar["count"], [3])
        for key in ("min", "max", "mean", "std"):
            self.assertEqual(len(scalar[key]), 1)

        features = {
            "observation.state": {"dtype": "float32", "shape": (3,)},
            "action": {"dtype": "float32", "shape": (1,)},
        }
        episode_data = {
            "observation.state": np.array(vector_rows, dtype=np.float32),
            "action": np.array([r[0] for r in scalar_rows], dtype=np.float32),
        }
        ref = compute_episode_stats(episode_data, features)
        for key, rows in (
            ("observation.state", vector_rows),
            ("action", scalar_rows),
        ):
            ours = _feature_stats(rows if key == "observation.state" else scalar_rows)
            for stat_key in ("min", "max", "mean", "std", "count"):
                ref_arr = ref[key][stat_key]
                ours_val = ours[stat_key]
                self.assertEqual(list(ref_arr.reshape(-1)), ours_val)


@unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
class TestEndToEndRecordTrainServe(unittest.TestCase):
    """The M3 acceptance test: record → mock train → mock serve on sim."""

    def test_record_train_serve_loop(self):
        # 1. Record
        bot = _bot("omnibot")
        rec = Recorder(bot, repo_id="local/demo", fps=10.0)
        rec.start_episode(task="pick up the cup")
        for _ in range(10):
            bot.drive(vx=0.1)
            bot.transport.step(0.1)  # type: ignore[attr-defined]
            rec.capture_frame()
        rec.stop_episode()

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = rec.save(tmp)

            # 2. Mock train (no GPU needed)
            from ohho.train import finetune

            ckpt_path = finetune(
                dataset=dataset_path,
                policy="smolvla",
                device="cpu",
                mock=True,
                output_dir=os.path.join(tmp, "checkpoints"),
            )
            self.assertTrue(os.path.exists(os.path.join(ckpt_path, "checkpoint.json")))

            # 3. Mock serve (no GPU needed)
            from ohho.serve import build_app
            from fastapi.testclient import TestClient

            app = build_app(
                model_path=ckpt_path,
                device="cpu",
                mock_model=True,
                auto_load=True,
            )
            client = TestClient(app)

            # health
            r = client.get("/health")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "ok")

            # load model
            r = client.post("/load_model", params={"model_path": ckpt_path})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "loaded")

            # predict
            r = client.post(
                "/predict",
                json={"instruction": "pick up the cup", "image_base64": ""},
            )
            self.assertEqual(r.status_code, 200)
            self.assertIn("vector", r.json()["action"])
            self.assertEqual(len(r.json()["action"]["vector"]), 9)


if __name__ == "__main__":
    unittest.main()
