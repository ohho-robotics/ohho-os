"""OHH-128: Go2 base-velocity serve contract and PolicyRunner."""

from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

from ohho.adapters.sim import SimTransport
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.serve.runner import PolicyRunner, _SimpleResponse
from ohho.serve import BASE_ACTION_SCHEMA_VERSION, build_app


def _has_fastapi() -> bool:
    return importlib.util.find_spec("fastapi") is not None


def _go2_sim():
    spec = get_spec("unitree-go2")
    tp = SimTransport(spec)
    tp.connect()
    bot = Robot(spec, tp, NativeRuntime())
    return bot, tp


@unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
class TestGo2ServeContract(unittest.TestCase):
    def test_predict_base_schema_for_go2_mock(self):
        from fastapi.testclient import TestClient

        app = build_app(
            robot_id="unitree-go2",
            mock_model=True,
            model_class="go2",
            auto_load=True,
        )
        client = TestClient(app)
        r = client.post(
            "/predict",
            json={
                "instruction": "go forward",
                "config": {"state": [0.0, 0.0, 0.0], "odom": {"x": 0.0}},
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["schema_version"], BASE_ACTION_SCHEMA_VERSION)
        self.assertIn("base", body["action"])
        base = body["action"]["base"]
        self.assertIn("vx", base)
        self.assertIn("vy", base)
        self.assertIn("wz", base)
        self.assertGreater(base["vx"], 0.0)

    def test_load_model_go2_class_and_mock(self):
        from fastapi.testclient import TestClient

        app = build_app(device="cpu")
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            cfg.write_text('{"policy": "go2_mock", "mock": true}', encoding="utf-8")
            r = client.post(
                "/load_model",
                params={
                    "model_path": tmp,
                    "model_class": "go2",
                    "mock": True,
                    "robot_id": "unitree-go2",
                },
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "loaded")
        self.assertEqual(data["model"], "Go2PolicyModel")
        self.assertEqual(data["schema_version"], BASE_ACTION_SCHEMA_VERSION)


@unittest.skipUnless(_has_fastapi(), "fastapi not installed — [serve] extra")
class TestPolicyRunner(unittest.TestCase):
    def test_go_forward_one_meter_in_sim(self):
        from fastapi.testclient import TestClient

        bot, tp = _go2_sim()
        app = build_app(
            robot_id="unitree-go2",
            mock_model=True,
            model_class="go2",
            auto_load=True,
        )
        client = TestClient(app)

        def post(url, payload, timeout_s):
            del url, timeout_s
            resp = client.post("/predict", json=payload)
            return _SimpleResponse(resp.status_code, resp.content)

        runner = PolicyRunner(
            bot,
            "http://test",
            hz=10.0,
            post=post,
            sleep=lambda _: None,
            step=lambda: tp.step(0.1),
        )
        result = runner.run("go forward", goal_x=1.0)
        self.assertEqual(result.stopped_reason, "goal_reached")
        self.assertGreaterEqual(result.last_odom_x, 0.99)
        bot.disconnect()

    def test_slow_predict_stops_runner(self):
        bot, _ = _go2_sim()

        def slow_post(url, payload, timeout_s):
            del url, payload, timeout_s
            time.sleep(0.25)
            return _SimpleResponse(200, b'{"action":{"base":{"vx":0.1,"vy":0,"wz":0}}}')

        runner = PolicyRunner(
            bot,
            "http://test",
            post=slow_post,
            sleep=lambda _: None,
            predict_timeout_s=0.2,
        )
        result = runner.run("go forward", max_steps=5)
        self.assertEqual(result.stopped_reason, "predict_fault")
        bot.disconnect()

    def test_dead_server_stops_within_300ms(self):
        bot, _ = _go2_sim()

        def dead_post(url, payload, timeout_s):
            del url, payload, timeout_s
            raise ConnectionError("server unreachable")

        runner = PolicyRunner(
            bot,
            "http://127.0.0.1:9",
            post=dead_post,
            sleep=lambda _: None,
        )
        t0 = time.monotonic()
        runner.run("go forward", max_steps=100)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self.assertLess(elapsed_ms, 300.0)
        bot.disconnect()


if __name__ == "__main__":
    unittest.main()
