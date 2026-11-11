"""Task-accuracy gate (OHH-470).

CPU policies are deterministic and do not need torch. The ACT FP16 case rounds
through IEEE binary16. A torch module test skips when torch is absent. GPU
float16 execution is not run.
"""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ohho.bench.schema import validate, validate_path
from ohho.cli import main
from ohho.optimize.cache import LocalBuildCache
from ohho.optimize.engine import OptimizeBackends, _applied_flags, _gate_and_publish
from ohho.optimize.plan import OptimizeRequest, build_plan
from ohho.optimize.toolchain import HostGpu
from ohho.parity.errors import ParityError, TaskFailed
from ohho.parity.gate import decide
from ohho.parity.metrics import tensor_metrics
from ohho.parity.report import build_report
from ohho.parity.tasks.bench_report import build_bench_document
from ohho.parity.tasks.episodes import fixed_seeds, synthetic_action, synthetic_episodes
from ohho.parity.tasks.registry import load_evaluators, select_evaluators
from ohho.parity.tasks.sim_success import pursuit_action
from ohho.parity.tasks.stage import evaluate_tasks
from ohho.parity.tasks.types import Comparison
from ohho.parity.thresholds import resolve_thresholds

ROOT = Path(__file__).resolve().parents[1]


def _binary16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def _act_fp32(observation: list[float]) -> list[float]:
    if len(observation) >= 6:
        return pursuit_action(observation)
    return synthetic_action(observation)


def _act_fp16(observation: list[float]) -> list[float]:
    rounded = [_binary16(value) for value in observation]
    return [_binary16(value) for value in _act_fp32(rounded)]


def _broken(observation: list[float]) -> list[float]:
    return [value + 0.04 for value in _act_fp32(observation)]


def _comparison(metric: str, document: dict) -> dict:
    for item in document["accuracy"]["comparisons"]:
        if item["metric"] == metric and item["task"] in {
            "action_error",
            "sim_success",
            "detection_map",
            "token_match",
        }:
            return item
    raise AssertionError(metric)


def _tensor_report() -> dict:
    limits = resolve_thresholds("act", "x86-cpu", "FP32")
    return build_report(
        result="pass",
        reasons=[],
        reference={"ref": "reference-act", "kind": "act", "family": "act"},
        candidate={"path": "model.onnx", "format": "onnx"},
        target="x86-cpu",
        precision="FP32",
        thresholds=limits.to_json(),
        inputs={
            "source": "random",
            "path": None,
            "seed": 0,
            "n_requested": 4,
            "n": 4,
            "indices": None,
            "sha256": "ab" * 32,
            "dtype": "float32",
        },
        outputs=[
            {
                "name": "actions",
                "shape": [4, 4],
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "rmse": 0.0,
                "cosine": 1.0,
                "per_joint": [0.0, 0.0, 0.0, 0.0],
                "action_chunk": 0.0,
                "top1": None,
                "kl": None,
            }
        ],
    )


def _gate_kwargs(plan, cache_root: Path) -> dict:
    return {
        "target": {
            "id": plan.profile.id,
            "arch": plan.profile.arch,
            "accelerator": plan.profile.accelerator,
        },
        "flags": _applied_flags(plan, None),
        "versions": dict.fromkeys(
            ("torch", "onnx", "onnxruntime", "tensorrt", "cuda", "driver", "jetpack")
        ),
        "model_sha256": "0" * 64,
        "calibration_cache_sha256": None,
        "host": HostGpu(None, None),
        "duration": 0.0,
        "cache_key": "0" * 64,
        "local": LocalBuildCache(cache_root),
        "registry": None,
    }


class TestEvaluators(unittest.TestCase):
    def test_builtins_load_without_reinstall(self) -> None:
        found = load_evaluators()
        self.assertEqual(
            list(found),
            ["action_error", "sim_success", "detection_map", "token_match"],
        )
        selected = select_evaluators("sim_success,action_error")
        self.assertEqual(
            [item.name for item in selected], ["sim_success", "action_error"]
        )

    def test_unknown_evaluator_is_a_request_error(self) -> None:
        with self.assertRaises(ParityError):
            select_evaluators("not-a-task")

    def test_action_mae_and_l2_match_on_held_out_episodes(self) -> None:
        document = evaluate_tasks(
            _act_fp32,
            _act_fp32,
            tasks="action_error",
            seeds=fixed_seeds(4),
        )
        mae = _comparison("action_mae", document)
        l2 = _comparison("action_l2", document)
        self.assertEqual(mae["reference"], 0.0)
        self.assertEqual(mae["optimized"], 0.0)
        self.assertEqual(mae["delta"], 0.0)
        self.assertEqual(mae["result"], "pass")
        self.assertEqual(len(mae["per_joint"]), 4)
        self.assertEqual(l2["result"], "pass")
        self.assertEqual(mae["samples"], 16)
        self.assertEqual(document["accuracy"]["result"], "pass")

    def test_sim_success_on_20_fixed_seeds(self) -> None:
        document = evaluate_tasks(
            _act_fp32,
            _act_fp32,
            tasks="sim_success",
            seeds=fixed_seeds(20),
        )
        row = _comparison("sim_success", document)
        self.assertEqual(row["seeds"], list(range(20)))
        self.assertEqual(row["samples"], 20)
        self.assertEqual(row["reference"], 1.0)
        self.assertEqual(row["optimized"], 1.0)
        self.assertEqual(row["result"], "pass")
        validate(document)

    def test_act_fp16_vs_fp32_mae_and_sim_success(self) -> None:
        document = evaluate_tasks(
            _act_fp32,
            _act_fp16,
            tasks="action_error,sim_success",
            seeds=fixed_seeds(20),
            model="reference-act",
            family="act",
        )
        self.assertEqual(document["schema"], "ohho.bench/v1")
        self.assertEqual(document["accuracy"]["result"], "pass")
        mae = _comparison("action_mae", document)
        sim = _comparison("sim_success", document)
        self.assertLess(mae["delta"], 0.05)
        self.assertEqual(mae["samples"], 80)
        self.assertEqual(sim["seeds"], list(range(20)))
        self.assertEqual(sim["reference"], 1.0)
        self.assertEqual(sim["optimized"], 1.0)
        validate(document)

    def test_detection_map_perfect_and_shifted(self) -> None:
        images = (
            {
                "image_id": "a",
                "ground_truth": [
                    {"class_id": 0, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}
                ],
            },
            {
                "image_id": "b",
                "ground_truth": [
                    {"class_id": 1, "x": 2.0, "y": 2.0, "w": 4.0, "h": 4.0}
                ],
            },
        )

        def perfect(sample: dict) -> list[dict]:
            return [{**box, "score": 0.9} for box in sample["ground_truth"]]

        def shifted(sample: dict) -> list[dict]:
            return [
                {**box, "x": box["x"] + 30.0, "score": 0.9}
                for box in sample["ground_truth"]
            ]

        passed = evaluate_tasks(
            perfect,
            perfect,
            tasks="detection_map",
            detections=images,
            seeds=(0,),
        )
        self.assertEqual(_comparison("map", passed)["reference"], 1.0)
        self.assertEqual(_comparison("map", passed)["optimized"], 1.0)
        failed = evaluate_tasks(
            perfect,
            shifted,
            tasks="detection_map",
            detections=images,
            seeds=(0,),
        )
        row = _comparison("map", failed)
        self.assertEqual(row["reference"], 1.0)
        self.assertEqual(row["optimized"], 0.0)
        self.assertEqual(row["result"], "fail")
        self.assertEqual(failed["accuracy"]["result"], "fail")
        validate(passed)
        validate(failed)

    def test_token_top1_and_match(self) -> None:
        examples = (
            {"label": 1, "tokens": [1, 2, 3]},
            {"label": 4, "tokens": [4, 5]},
        )

        def good(sample: dict) -> dict:
            logits = [0.0] * 6
            logits[sample["label"]] = 3.0
            return {"logits": logits, "tokens": list(sample["tokens"])}

        def bad(sample: dict) -> dict:
            del sample
            return {"logits": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0], "tokens": [0, 0]}

        passed = evaluate_tasks(
            good, good, tasks="token_match", tokens=examples, seeds=(0,)
        )
        self.assertEqual(_comparison("top1", passed)["optimized"], 1.0)
        self.assertEqual(_comparison("token_match", passed)["optimized"], 1.0)
        failed = evaluate_tasks(
            good, bad, tasks="token_match", tokens=examples, seeds=(0,)
        )
        self.assertEqual(failed["accuracy"]["result"], "fail")
        self.assertLess(_comparison("top1", failed)["optimized"], 1.0)
        validate(passed)
        validate(failed)

    def test_broken_quant_fails_task_gate_and_passes_loose_tensor_gate(self) -> None:
        seeds = fixed_seeds(20)
        episodes = synthetic_episodes(seeds)
        reference_batch = []
        broken_batch = []
        for episode in episodes:
            for frame in episode.frames:
                observation = list(frame.observation)
                reference_batch.append(_act_fp32(observation))
                broken_batch.append(_broken(observation))
        stats = tensor_metrics(reference_batch, broken_batch)
        tensor_result, tensor_reasons = decide(
            [
                {
                    "name": "actions",
                    "mean_abs": stats["mean_abs"],
                    "max_abs": stats["max_abs"],
                    "rmse": stats["rmse"],
                    "cosine": stats["cosine"],
                }
            ],
            resolve_thresholds("act", None, None, mean_abs=0.2),
        )
        self.assertEqual(tensor_result, "pass", tensor_reasons)
        self.assertLess(stats["mean_abs"], 0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tolerances.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            "broken-quant": {
                                "action_mae": 0.01,
                                "action_l2": 0.01,
                                "sim_success": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            document = evaluate_tasks(
                _act_fp32,
                _broken,
                tasks="action_error,sim_success",
                seeds=seeds,
                tolerances=str(path),
                model="broken-quant",
            )
        self.assertEqual(document["accuracy"]["result"], "fail")
        self.assertEqual(_comparison("action_mae", document)["result"], "fail")
        self.assertGreater(_comparison("action_mae", document)["delta"], 0.01)
        self.assertEqual(_comparison("sim_success", document)["reference"], 1.0)
        self.assertEqual(_comparison("sim_success", document)["optimized"], 0.0)
        self.assertEqual(_comparison("sim_success", document)["result"], "fail")
        validate(document)

    def test_shipped_examples_and_additive_accuracy_schema(self) -> None:
        schema = json.loads(
            (ROOT / "ohho" / "bench" / "schemas" / "bench.v1.json").read_text(
                encoding="utf-8"
            )
        )
        accuracy = schema["$defs"]["accuracy"]
        self.assertEqual(accuracy["required"], ["primary"])
        self.assertIn("comparisons", accuracy["properties"])
        self.assertNotIn("comparisons", accuracy["required"])
        for name in ("vision.json", "vla.json", "llm.json"):
            validate_path(ROOT / "ohho" / "bench" / "examples" / name)
        completed = subprocess.run(
            [sys.executable, "scripts/generate_bench_types.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class TestWiring(unittest.TestCase):
    def test_cli_task_failure_exits_1_after_a_passing_tensor_gate(self) -> None:
        now = datetime.now(timezone.utc)
        accuracy = build_bench_document(
            [
                Comparison(
                    task="action_error",
                    metric="action_mae",
                    reference=0.0,
                    optimized=0.04,
                    tolerance=0.01,
                    direction="lower_is_better",
                    unit="action",
                    samples=4,
                )
            ],
            model="reference",
            candidate="broken",
            family="act",
            manifest_ref="ohho.parity/v1",
            latencies_ms=[1.0],
            duration_s=0.01,
            started_at=now,
            finished_at=now,
        )
        args = Namespace(
            reference="reference-act",
            candidate="candidate.pt",
            inputs="random",
            n=4,
            seed=0,
            family=None,
            target=None,
            precision=None,
            mean_abs=None,
            thresholds=None,
            record_dir=None,
            tasks="action_error",
            task_seeds=4,
            task_tolerances=None,
            task_episodes=None,
            task_detections=None,
            task_tokens=None,
            json=True,
            out=None,
            accuracy_out=None,
        )
        out = io.StringIO()
        with patch("ohho.parity.gate.evaluate", return_value=_tensor_report()):
            with patch("ohho.parity.tasks.stage.evaluate_cli", return_value=accuracy):
                from ohho.parity.command import run_parity

                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    code = run_parity(args)
        self.assertEqual(code, 1)
        body = json.loads(out.getvalue())
        self.assertEqual(body["parity"]["result"], "pass")
        self.assertEqual(body["accuracy"]["accuracy"]["result"], "fail")

    def test_help_lists_tasks(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["parity", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--tasks", out.getvalue())

    def test_optimize_skips_tasks_unless_asked(self) -> None:
        report = _tensor_report()
        request = OptimizeRequest(
            model="reference-act",
            target="x86-cpu",
            precision="fp32",
            out_dir="unused",
        )
        plan = build_plan(request)
        loaded = SimpleNamespace(
            ref="reference-act", kind="act", submodule=None, source="registry"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cache_tmp = tempfile.TemporaryDirectory()
            self.addCleanup(cache_tmp.cleanup)
            cache_root = Path(cache_tmp.name)
            with patch("ohho.parity.gate.compare_export", return_value=report):
                with patch(
                    "ohho.parity.tasks.stage.run_task_stage",
                    side_effect=AssertionError("tasks should not run"),
                ):
                    document = _gate_and_publish(
                        request,
                        plan,
                        loaded,
                        out,
                        [],
                        [],
                        OptimizeBackends(),
                        **_gate_kwargs(plan, cache_root),
                    )
            self.assertEqual(document["parity"]["result"], "pass")
            self.assertFalse((out / "accuracy.json").exists())
            self.assertTrue((out / "manifest.json").is_file())

    def test_optimize_task_failure_discards_a_passing_tensor_export(self) -> None:
        report = _tensor_report()
        request = OptimizeRequest(
            model="reference-act",
            target="x86-cpu",
            precision="fp32",
            out_dir="unused",
            tasks="action_error",
        )
        plan = build_plan(request)
        loaded = SimpleNamespace(
            ref="reference-act", kind="act", submodule=None, source="registry"
        )
        failing = {"accuracy": {"result": "fail", "reasons": ["action_mae missed"]}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cache_tmp = tempfile.TemporaryDirectory()
            self.addCleanup(cache_tmp.cleanup)
            cache_root = Path(cache_tmp.name)
            artifact = out / "model.onnx"
            artifact.write_bytes(b"onnx")
            outputs = [{"role": "onnx", "path": "model.onnx"}]
            with patch("ohho.parity.gate.compare_export", return_value=dict(report)):
                with patch(
                    "ohho.parity.tasks.stage.run_task_stage",
                    return_value=failing,
                ):
                    with self.assertRaises(TaskFailed):
                        _gate_and_publish(
                            request,
                            plan,
                            loaded,
                            out,
                            outputs,
                            [],
                            OptimizeBackends(),
                            **_gate_kwargs(plan, cache_root),
                        )
            self.assertFalse(artifact.exists())
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse((out / "accuracy.json").exists())


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is not installed; ACT FP16 module path skipped (CPU binary16 test covers the metrics)",
)
class TestTorchAct(unittest.TestCase):
    def test_fp16_weight_roundtrip_matches_fp32_on_20_seeds(self) -> None:
        import torch

        torch.manual_seed(0)

        class TinyAct(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.readout = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    weight = torch.zeros(4, 4)
                    weight[0, 0] = 0.5
                    weight[1, 1] = -0.25
                    self.readout.weight.copy_(weight)
                    self.readout.bias.copy_(torch.tensor([0.0, 0.0, 0.1, 0.2]))

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                if obs.shape[-1] >= 6:
                    goal_x = obs[-3]
                    goal_y = obs[-2]
                    distance = torch.sqrt(goal_x * goal_x + goal_y * goal_y).clamp_min(
                        1e-12
                    )
                    scale = obs.new_tensor(0.5) / distance
                    zero = torch.zeros_like(goal_x)
                    return torch.stack((goal_x * scale, goal_y * scale, zero, zero))
                return self.readout(obs[:4])

        fp32 = TinyAct().eval()
        fp16 = TinyAct().eval()
        fp16.load_state_dict(fp32.state_dict())
        with torch.no_grad():
            for parameter in fp16.parameters():
                parameter.copy_(parameter.detach().half().float())

        def predict(module: torch.nn.Module, round_input: bool):
            def run(observation: list[float]) -> list[float]:
                values = (
                    [_binary16(value) for value in observation]
                    if round_input
                    else list(observation)
                )
                tensor = torch.tensor(values, dtype=torch.float32)
                with torch.no_grad():
                    output = module(tensor)
                return [float(value) for value in output]

            return run

        document = evaluate_tasks(
            predict(fp32, False),
            predict(fp16, True),
            tasks="action_error,sim_success",
            seeds=fixed_seeds(20),
            model="reference-act",
            family="act",
        )
        self.assertEqual(document["accuracy"]["result"], "pass")
        sim = _comparison("sim_success", document)
        self.assertEqual(sim["seeds"], list(range(20)))
        self.assertEqual(sim["reference"], 1.0)
        self.assertEqual(sim["optimized"], 1.0)
        validate(document)


if __name__ == "__main__":
    unittest.main()
