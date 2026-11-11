"""Numerical parity gate (OHH-430).

Cases that execute a module or an ONNX file skip when torch or the optimize
extra is absent. Metric, dataset, threshold, and manifest checks always run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.data.reader import DatasetReader
from ohho.optimize.manifest import build_manifest, manifest_problems
from ohho.optimize.models import ExportInput
from ohho.optimize.plan import OptimizeRequest
from ohho.parity import DEFAULT_MEAN_ABS, SCHEMA_ID, schema_text
from ohho.parity.errors import ParityError, ParityFailed
from ohho.parity.gate import decide, evaluate
from ohho.parity.inputs import (
    batches_sha256,
    load_batches,
    load_observation_frames,
)
from ohho.parity.metrics import (
    action_chunk_error,
    cosine_similarity,
    logits_kl,
    max_abs_diff,
    mean_abs_diff,
    per_joint_error,
    rmse,
    top1_agreement,
)
from ohho.parity.report import build_report
from ohho.parity.thresholds import resolve_thresholds

ROOT = Path(__file__).resolve().parents[1]
_HEAVY = {"torch", "onnx", "onnxruntime", "tensorrt", "numpy"}


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _has_export_stack() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "onnx", "onnxruntime")
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _write_dataset(root: Path, frames: list[list[float]]) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": 1,
                "total_frames": len(frames),
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(frames[0])],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": len(frames)}) + "\n",
        encoding="utf-8",
    )
    lines = []
    for index, state in enumerate(frames):
        lines.append(
            json.dumps(
                {
                    "observation.state": state,
                    "action": [0.0, 0.0],
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


def _sample_report(*, allow: bool = False, result: str = "pass") -> dict:
    reasons = [] if result == "pass" else ["output: mean abs diff 0.2 > 0.05"]
    return build_report(
        result=result,
        reasons=reasons,
        reference={"ref": "reference-act", "kind": "act", "family": "act"},
        candidate={"path": "model.onnx", "format": "onnx"},
        target="x86-cpu",
        precision="FP32",
        thresholds=resolve_thresholds("act", "x86-cpu", "FP32").to_json(),
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
                "shape": [4, 3, 4],
                "mean_abs": 0.0 if result == "pass" else 0.2,
                "max_abs": 0.0 if result == "pass" else 0.2,
                "rmse": 0.0 if result == "pass" else 0.2,
                "cosine": 1.0,
                "per_joint": [0.0, 0.0, 0.0, 0.0],
                "action_chunk": 0.0 if result == "pass" else 0.4,
                "top1": None,
                "kl": None,
            }
        ],
        allow_parity_fail=allow,
    )


def _manifest(report: dict, *, allow: bool) -> dict:
    flag = {
        "onnx": True,
        "tensorrt": False,
        "quant": None,
        "ort_ep": "cpu",
        "workspace_gb": None,
        "workspace_bytes": 0,
        "max_batch": 1,
        "dynamic_batch": False,
        "calib_batch": None,
        "calib_max_images": None,
        "quantization": None,
        "opset": 17,
        "submodule": None,
        "allow_parity_fail": allow,
    }
    inputs = {
        "model": "reference-act",
        "target": "x86-cpu",
        "precision": "fp32",
        "onnx": False,
        "tensorrt": False,
        "quant": None,
        "ort_ep": "cpu",
        "workspace_gb": None,
        "max_batch": 1,
        "dynamic_batch": False,
        "calib_data": None,
        "calib_cache": None,
        "calib_batch": None,
        "calib_max_images": None,
        "submodule": None,
        "allow_parity_fail": allow,
    }
    return build_manifest(
        model={
            "ref": "reference-act",
            "kind": "act",
            "submodule": None,
            "source": "registry",
        },
        target={"id": "x86-cpu", "arch": "x86_64", "accelerator": "none"},
        precision="FP32",
        execution_provider="CPUExecutionProvider",
        flags=flag,
        inputs=inputs,
        outputs=[],
        session_providers=["CPUExecutionProvider"],
        parity=report,
        created="2026-01-01T00:00:00+00:00",
    )


class TestMetrics(unittest.TestCase):
    def test_action_and_base_metrics(self) -> None:
        reference = [[[0.0, 0.0], [0.0, 0.0]]]
        candidate = [[[3.0, 4.0], [0.0, 0.0]]]
        self.assertEqual(per_joint_error(reference, candidate), [1.5, 2.0])
        self.assertEqual(action_chunk_error(reference, candidate), 5.0)
        self.assertEqual(mean_abs_diff(reference, candidate), 1.75)
        self.assertEqual(max_abs_diff(reference, candidate), 4.0)
        self.assertEqual(rmse(reference, candidate), 2.5)
        self.assertEqual(cosine_similarity(reference, candidate), 0.0)
        self.assertEqual(cosine_similarity(reference, reference), 1.0)

    def test_top1_and_kl(self) -> None:
        reference = [[0.0, 1.0], [3.0, 0.0]]
        flipped = [[1.0, 0.0], [3.0, 0.0]]
        self.assertEqual(top1_agreement(reference, reference), 1.0)
        self.assertEqual(top1_agreement(reference, flipped), 0.5)
        self.assertAlmostEqual(logits_kl(reference, reference), 0.0, places=6)
        self.assertGreater(logits_kl(reference, flipped), 0.0)

    def test_mean_abs_limit_is_inclusive(self) -> None:
        limits = resolve_thresholds("module", None, None)
        self.assertEqual(limits.mean_abs, DEFAULT_MEAN_ABS)
        self.assertEqual(DEFAULT_MEAN_ABS, 0.05)

        def row(mean: float) -> dict:
            return {
                "name": "output",
                "mean_abs": mean,
                "max_abs": mean,
                "rmse": mean,
                "cosine": 1.0,
                "per_joint": None,
                "action_chunk": None,
                "top1": None,
                "kl": None,
            }

        result, reasons = decide([row(0.05)], limits)
        self.assertEqual(result, "pass")
        self.assertEqual(reasons, [])
        result, reasons = decide([row(0.05 + 1e-6)], limits)
        self.assertEqual(result, "fail")
        self.assertIn("mean abs", reasons[0])

    def test_per_joint_limit(self) -> None:
        limits = resolve_thresholds("act", None, None)
        limits = type(limits)(
            **{**limits.to_json(), "per_joint": 0.1}  # type: ignore[arg-type]
        )
        result, reasons = decide(
            [
                {
                    "name": "actions",
                    "mean_abs": 0.0,
                    "max_abs": 0.0,
                    "rmse": 0.0,
                    "cosine": 1.0,
                    "per_joint": [0.2],
                    "action_chunk": 0.0,
                    "top1": None,
                    "kl": None,
                }
            ],
            limits,
        )
        self.assertEqual(result, "fail")
        self.assertIn("joint 0", reasons[0])


class TestThresholds(unittest.TestCase):
    def test_family_and_target_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "limits.json"
            path.write_text(
                json.dumps(
                    {
                        "default": {"mean_abs": 0.05},
                        "families": {
                            "act": {"mean_abs": 0.02},
                            "classifier": {
                                "mean_abs": 1.0,
                                "top1_min": 0.0,
                                "kl_max": 10.0,
                            },
                        },
                        "targets": {
                            "jetson-orin-nx": {
                                "mean_abs": 0.08,
                                "precisions": {"INT8": {"mean_abs": 0.12}},
                                "families": {"act": {"mean_abs": 0.03}},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            act = resolve_thresholds("act", None, None, path=str(path))
            self.assertEqual(act.mean_abs, 0.02)
            host = resolve_thresholds("module", "jetson-orin-nx", None, path=str(path))
            self.assertEqual(host.mean_abs, 0.08)
            int8 = resolve_thresholds(
                "module", "jetson-orin-nx", "int8", path=str(path)
            )
            self.assertEqual(int8.mean_abs, 0.12)
            both = resolve_thresholds("act", "jetson-orin-nx", "INT8", path=str(path))
            self.assertEqual(both.mean_abs, 0.03)
            forced = resolve_thresholds(
                "act", "jetson-orin-nx", "INT8", mean_abs=1.5, path=str(path)
            )
            self.assertEqual(forced.mean_abs, 1.5)
            clf = resolve_thresholds("classifier", None, None)
            self.assertEqual(clf.top1_min, 0.99)
            self.assertEqual(clf.kl_max, 0.05)
            loosened = resolve_thresholds("classifier", None, None, path=str(path))
            self.assertEqual(loosened.top1_min, 0.0)
            self.assertEqual(loosened.kl_max, 10.0)

    def test_unknown_threshold_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "limits.json"
            path.write_text(json.dumps({"default": {"nope": 1}}), encoding="utf-8")
            with self.assertRaises(ParityError):
                resolve_thresholds("act", None, None, path=str(path))


class TestInputs(unittest.TestCase):
    def test_random_seed_is_stable(self) -> None:
        specs = (ExportInput("input", (1, 4)),)
        first = load_batches("random", specs, n=8, seed=5)
        second = load_batches("random", specs, n=8, seed=5)
        other = load_batches("random", specs, n=8, seed=6)
        self.assertEqual(batches_sha256(first.batches), batches_sha256(second.batches))
        self.assertNotEqual(
            batches_sha256(first.batches), batches_sha256(other.batches)
        )
        self.assertEqual(len(first.batches["input"]), 8)
        self.assertEqual(len(first.batches["input"][0]), 4)
        self.assertIsNone(first.indices)
        self.assertEqual(first.seed, 5)
        self.assertEqual(first.source, "random")

    def test_dataset_frames_are_used(self) -> None:
        frames = [
            [0.25, 0.5, -0.25, 1.0],
            [0.0, 1.0, 0.5, -1.0],
            [-0.5, 0.25, 0.75, 0.125],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            _write_dataset(root, frames)
            specs = (ExportInput("state", (1, 4)),)
            bundle = load_batches(str(root), specs, n=256, seed=3)
            self.assertEqual(bundle.source, "dataset")
            self.assertEqual(bundle.n, 3)
            self.assertEqual(bundle.n_requested, 256)
            self.assertEqual(bundle.indices, [0, 1, 2])
            self.assertEqual(bundle.batches["state"], frames)
            self.assertEqual(bundle.seed, 3)
            prefixed = load_batches(f"dataset:{root}", specs, n=256, seed=3)
            self.assertEqual(prefixed.batches["state"], frames)
            subsampled = load_batches(str(root), specs, n=2, seed=0)
            self.assertEqual(len(subsampled.indices), 2)
            self.assertEqual(
                subsampled.batches["state"],
                [frames[index] for index in subsampled.indices],
            )
            again = load_batches(str(root), specs, n=2, seed=0)
            self.assertEqual(subsampled.indices, again.indices)

    def test_recorded_keyword_uses_sim_loop_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = load_observation_frames(
                "recorded", n=4, seed=1, record_dir=str(Path(tmp) / "episode")
            )
            self.assertEqual(source.source, "recorded")
            self.assertIsNotNone(source.path)
            assert source.path is not None
            self.assertTrue((Path(source.path) / "meta" / "info.json").is_file())
            expected = [
                [float(value) for value in frame.observation_state]
                for frame in DatasetReader(source.path).all_frames()
            ]
            self.assertGreater(len(expected), 0)
            self.assertEqual(
                source.frames, [expected[index] for index in source.indices]
            )


class TestReportAndManifest(unittest.TestCase):
    def test_schema_id_and_package_data(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("parity/*.json", text)
        document = json.loads(schema_text())
        self.assertEqual(document["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertEqual(SCHEMA_ID, "ohho.parity/v1")

    def test_modules_do_not_import_heavy_packages_at_import(self) -> None:
        package = ROOT / "ohho" / "parity"
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    self.assertNotIn(name, _HEAVY, path.name)

    def test_manifest_records_the_allow_flag(self) -> None:
        passed = _manifest(_sample_report(allow=False), allow=False)
        self.assertEqual(manifest_problems(passed), [])
        self.assertFalse(passed["flags"]["allow_parity_fail"])
        self.assertFalse(passed["inputs"]["allow_parity_fail"])
        self.assertEqual(passed["parity"]["result"], "pass")
        allowed = _manifest(_sample_report(allow=True, result="fail"), allow=True)
        self.assertEqual(manifest_problems(allowed), [])
        self.assertTrue(allowed["flags"]["allow_parity_fail"])
        self.assertTrue(allowed["inputs"]["allow_parity_fail"])
        self.assertTrue(allowed["parity"]["allow_parity_fail"])
        self.assertEqual(allowed["parity"]["result"], "fail")
        broken = dict(passed)
        broken.pop("parity")
        problems = manifest_problems(broken)
        self.assertTrue(any("parity" in item for item in problems))

    def test_engine_candidate_is_rejected_without_running_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.engine"
            path.write_bytes(b"not-an-engine")
            with self.assertRaises(ParityError) as caught:
                evaluate("reference-act", str(path), inputs="random", n=1)
        self.assertIn("TensorRT", str(caught.exception))

    def test_missing_candidate_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.onnx"
            with self.assertRaises(ParityError):
                evaluate("reference-act", str(missing), inputs="random", n=1)

    def test_defaults_and_bad_sample_count(self) -> None:
        request = OptimizeRequest(
            model="reference-act",
            target="x86-cpu",
            precision="fp32",
            out_dir="out",
        )
        self.assertEqual(request.parity_n, 256)
        self.assertEqual(request.parity_seed, 0)
        self.assertEqual(request.parity_inputs, "random")
        self.assertFalse(request.allow_parity_fail)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, text, _err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-act",
                    "--target",
                    "x86-cpu",
                    "--precision",
                    "fp32",
                    "--out",
                    str(out),
                    "--parity-n",
                    "0",
                    "--json",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("parity-n", json.loads(text)["error"])
        self.assertFalse(out.exists())

    def test_help_and_doctor(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["parity", "--help"])
        self.assertEqual(caught.exception.code, 0)
        text = out.getvalue()
        for flag in ("--reference", "--candidate", "--inputs", "--n", "--mean-abs"):
            self.assertIn(flag, text)
        code, doctor, _err = _run(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("parity", doctor)


@unittest.skipUnless(_has_torch(), "torch is not installed")
class TestParityModels(unittest.TestCase):
    def test_identical_model_passes_and_records_seed(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = _save_linear(Path(tmp), shift=0.0)
            report = evaluate(str(ref), str(cand), inputs="random", seed=0)
        self.assertEqual(report["result"], "pass")
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["inputs"]["source"], "random")
        self.assertEqual(report["inputs"]["seed"], 0)
        self.assertEqual(report["inputs"]["n_requested"], 256)
        self.assertEqual(report["inputs"]["n"], 256)
        self.assertEqual(len(report["inputs"]["sha256"]), 64)
        item = report["outputs"][0]
        for key in ("mean_abs", "max_abs", "rmse", "cosine"):
            self.assertIn(key, item)
        self.assertLess(item["mean_abs"], 1e-6)
        self.assertIsNone(item["per_joint"])
        del torch

    def test_perturbed_model_fails_until_the_limit_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, cand = _save_linear(root, shift=1.0)
            failed = evaluate(str(ref), str(cand), inputs="random", n=16, seed=1)
            allowed = evaluate(
                str(ref),
                str(cand),
                inputs="random",
                n=16,
                seed=1,
                mean_abs=10,
            )
            code, text, err = _run(
                [
                    "parity",
                    "--reference",
                    str(ref),
                    "--candidate",
                    str(cand),
                    "--inputs",
                    "random",
                    "--n",
                    "8",
                    "--seed",
                    "2",
                    "--json",
                ]
            )
            overridden = _run(
                [
                    "parity",
                    "--reference",
                    str(ref),
                    "--candidate",
                    str(cand),
                    "--inputs",
                    "random",
                    "--n",
                    "8",
                    "--seed",
                    "2",
                    "--mean-abs",
                    "10",
                    "--json",
                ]
            )
        self.assertEqual(failed["result"], "fail")
        self.assertGreater(failed["outputs"][0]["mean_abs"], 0.05)
        self.assertTrue(any("mean abs" in reason for reason in failed["reasons"]))
        self.assertEqual(allowed["result"], "pass")
        self.assertEqual(allowed["thresholds"]["mean_abs"], 10)
        self.assertGreater(allowed["outputs"][0]["mean_abs"], 0.05)
        self.assertEqual(code, 1, err + text)
        self.assertEqual(json.loads(text)["result"], "fail")
        self.assertEqual(overridden[0], 0, overridden[2] + overridden[1])
        self.assertEqual(json.loads(overridden[1])["result"], "pass")

    def test_action_head_reports_joint_and_chunk_error(self) -> None:
        import torch

        from ohho.optimize.models import REFERENCE_ACT, build_reference_act

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "kind": "act",
                        "inputs": [
                            {
                                "name": "state",
                                "shape": [1, int(REFERENCE_ACT["state_dim"])],
                            }
                        ],
                        "outputs": ["actions"],
                    }
                ),
                encoding="utf-8",
            )
            clean = build_reference_act()
            sample = torch.zeros(1, int(REFERENCE_ACT["state_dim"]))
            torch.jit.trace(clean, sample).save(str(root / "ref.ts"))
            shifted = build_reference_act()
            with torch.no_grad():
                shifted.action_head.bias.add_(1.0)
            torch.jit.trace(shifted, sample).save(str(root / "cand.ts"))
            report = evaluate(
                str(root / "ref.ts"),
                str(root / "cand.ts"),
                inputs="random",
                n=4,
                seed=0,
                family="act",
            )
        self.assertEqual(report["reference"]["family"], "act")
        self.assertEqual(report["result"], "fail")
        item = report["outputs"][0]
        self.assertEqual(item["name"], "actions")
        self.assertEqual(len(item["per_joint"]), int(REFERENCE_ACT["action_dim"]))
        self.assertGreater(item["action_chunk"], 0.05)

    def test_classifier_reports_top1_and_kl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, cand = _save_classifier(root / "same", flip=False)
            passed = evaluate(str(ref), str(cand), inputs="random", n=8, seed=0)
            ref2, dirty = _save_classifier(root / "flipped", flip=True)
            failed = evaluate(str(ref2), str(dirty), inputs="random", n=8, seed=0)
            limits = root / "limits.json"
            limits.write_text(
                json.dumps(
                    {
                        "families": {
                            "classifier": {
                                "mean_abs": 100,
                                "top1_min": 0,
                                "kl_max": 100,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            allowed = evaluate(
                str(ref2),
                str(dirty),
                inputs="random",
                n=8,
                seed=0,
                thresholds=str(limits),
            )
        self.assertEqual(passed["reference"]["family"], "classifier")
        self.assertEqual(passed["result"], "pass")
        self.assertEqual(passed["outputs"][0]["top1"], 1.0)
        self.assertAlmostEqual(passed["outputs"][0]["kl"], 0.0, places=6)
        self.assertEqual(failed["result"], "fail")
        self.assertEqual(failed["outputs"][0]["top1"], 0.0)
        self.assertGreater(failed["outputs"][0]["kl"], 0.05)
        self.assertEqual(allowed["result"], "pass")

    def test_dataset_frames_reach_the_model(self) -> None:
        frames = [
            [0.25, 0.5, -0.25, 1.0],
            [0.0, 1.0, 0.5, -1.0],
            [-0.5, 0.25, 0.75, 0.125],
        ]
        specs = (ExportInput("input", (1, 4)),)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            _write_dataset(dataset, frames)
            ref, cand = _save_linear(root, shift=0.0)
            report = evaluate(str(ref), str(cand), inputs=str(dataset), n=256, seed=9)
            bundle = load_batches(str(dataset), specs, n=256, seed=9)
            random_bundle = load_batches("random", specs, n=3, seed=9)
        self.assertEqual(report["result"], "pass")
        self.assertEqual(report["inputs"]["source"], "dataset")
        self.assertEqual(report["inputs"]["seed"], 9)
        self.assertEqual(report["inputs"]["n"], 3)
        self.assertEqual(report["inputs"]["indices"], [0, 1, 2])
        self.assertEqual(report["inputs"]["sha256"], batches_sha256(bundle.batches))
        self.assertEqual(bundle.batches["input"], frames)
        self.assertNotEqual(
            report["inputs"]["sha256"], batches_sha256(random_bundle.batches)
        )


@unittest.skipUnless(_has_export_stack(), "optimize extra (torch, onnx, onnxruntime)")
class TestOptimizeParityGate(unittest.TestCase):
    def test_passing_optimize_records_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, text, err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-act",
                    "--target",
                    "x86-cpu",
                    "--precision",
                    "fp32",
                    "--ort-ep",
                    "cpu",
                    "--parity-n",
                    "4",
                    "--parity-seed",
                    "0",
                    "--out",
                    str(out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            document = json.loads(text)
            self.assertEqual(document["parity"]["result"], "pass")
            self.assertEqual(document["parity"]["schema"], SCHEMA_ID)
            self.assertEqual(document["parity"]["inputs"]["seed"], 0)
            self.assertEqual(document["parity"]["inputs"]["source"], "random")
            self.assertLess(document["parity"]["outputs"][0]["mean_abs"], 0.05)
            self.assertIn("max_abs", document["parity"]["outputs"][0])
            self.assertFalse(document["flags"]["allow_parity_fail"])
            self.assertFalse(document["parity"]["allow_parity_fail"])
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "parity.json").is_file())
            self.assertTrue((out / "model.onnx").is_file())

    def test_failing_parity_exits_nonzero_and_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("ohho.optimize.engine.optimize", _wrap_optimize(_perturb_act)):
                code, text, err = _run(
                    [
                        "optimize",
                        "--model",
                        "reference-act",
                        "--target",
                        "x86-cpu",
                        "--precision",
                        "fp32",
                        "--ort-ep",
                        "cpu",
                        "--parity-n",
                        "4",
                        "--out",
                        str(out),
                        "--json",
                    ]
                )
            self.assertEqual(code, 1, err + text)
            body = json.loads(text)
            self.assertFalse(body["ok"])
            self.assertEqual(body["parity"]["result"], "fail")
            self.assertGreater(body["parity"]["outputs"][0]["mean_abs"], 0.05)
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse((out / "parity.json").exists())
            self.assertFalse((out / "model.onnx").exists())

    def test_allow_parity_fail_publishes_and_records_the_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("ohho.optimize.engine.optimize", _wrap_optimize(_perturb_act)):
                code, text, err = _run(
                    [
                        "optimize",
                        "--model",
                        "reference-act",
                        "--target",
                        "x86-cpu",
                        "--precision",
                        "fp32",
                        "--ort-ep",
                        "cpu",
                        "--parity-n",
                        "4",
                        "--allow-parity-fail",
                        "--out",
                        str(out),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, err + text)
            document = json.loads(text)
            self.assertEqual(document["parity"]["result"], "fail")
            self.assertTrue(document["parity"]["allow_parity_fail"])
            self.assertTrue(document["flags"]["allow_parity_fail"])
            self.assertTrue(document["inputs"]["allow_parity_fail"])
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "model.onnx").is_file())
            digest = hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest()
            recorded = next(
                item for item in document["outputs"] if item["role"] == "onnx"
            )
            self.assertEqual(recorded["sha256"], digest)

    def test_optimize_api_refuses_without_writing_the_manifest(self) -> None:
        from ohho.optimize.engine import OptimizeBackends
        from ohho.optimize.engine import optimize as real_optimize

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with self.assertRaises(ParityFailed):
                real_optimize(
                    OptimizeRequest(
                        model="reference-act",
                        target="x86-cpu",
                        precision="fp32",
                        out_dir=str(out),
                        ort_ep="cpu",
                        parity_n=4,
                    ),
                    backends=OptimizeBackends(before_parity=_perturb_act),
                )
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse((out / "model.onnx").exists())


def _save_linear(
    root: Path,
    *,
    shift: float,
    kind: str | None = None,
    features: int = 4,
    outputs: int = 2,
) -> tuple[Path, Path]:
    import torch

    root.mkdir(parents=True, exist_ok=True)
    ref_dir = root / "ref"
    cand_dir = root / "cand"
    ref_dir.mkdir()
    cand_dir.mkdir()
    torch.manual_seed(0)
    module = torch.nn.Linear(features, outputs).eval()
    torch.save(module, ref_dir / "model.pt")
    torch.manual_seed(0)
    other = torch.nn.Linear(features, outputs).eval()
    if shift:
        with torch.no_grad():
            other.bias.add_(shift)
    torch.save(other, cand_dir / "model.pt")
    if kind:
        name = "logits" if kind == "classifier" else "output"
        payload = json.dumps(
            {
                "kind": kind,
                "inputs": [{"name": "input", "shape": [1, features]}],
                "outputs": [name],
            }
        )
        (ref_dir / "model.json").write_text(payload, encoding="utf-8")
        (cand_dir / "model.json").write_text(payload, encoding="utf-8")
    return ref_dir / "model.pt", cand_dir / "model.pt"


def _save_classifier(root: Path, *, flip: bool) -> tuple[Path, Path]:
    """Two classifiers. The flipped candidate's class 0 logit is shifted by 20.

    Weights are zero and the reference bias prefers class 1, so top-1 agreement
    of the flipped pair is 0 for every input.
    """
    import torch

    root.mkdir(parents=True, exist_ok=True)

    def build(flipped: bool) -> torch.nn.Linear:
        module = torch.nn.Linear(3, 4)
        with torch.no_grad():
            module.weight.zero_()
            bias = torch.tensor([20.0 if flipped else 0.0, 1.0, 0.0, 0.0])
            module.bias.copy_(bias)
        module.eval()
        return module

    ref_dir = root / "ref"
    cand_dir = root / "cand"
    ref_dir.mkdir()
    cand_dir.mkdir()
    torch.save(build(False), ref_dir / "model.pt")
    torch.save(build(flip), cand_dir / "model.pt")
    payload = json.dumps(
        {
            "kind": "classifier",
            "inputs": [{"name": "input", "shape": [1, 3]}],
            "outputs": ["logits"],
        }
    )
    (ref_dir / "model.json").write_text(payload, encoding="utf-8")
    (cand_dir / "model.json").write_text(payload, encoding="utf-8")
    return ref_dir / "model.pt", cand_dir / "model.pt"


def _perturb_act(out: Path) -> None:
    import torch

    from ohho.optimize.exporters import export_onnx
    from ohho.optimize.models import load_model

    loaded = load_model("reference-act")
    with torch.no_grad():
        loaded.module.action_head.bias.add_(1.0)
    export_onnx(loaded, out / "model.onnx")


def _wrap_optimize(hook):
    from ohho.optimize.engine import OptimizeBackends
    from ohho.optimize.engine import optimize as real_optimize

    def wrapped(request, backends=None):
        standins = backends or OptimizeBackends()
        standins.before_parity = hook
        return real_optimize(request, backends=standins)

    return wrapped


if __name__ == "__main__":
    unittest.main()
