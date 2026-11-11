"""Quantizer plugins (OHH-468).

CPU tests mock AWQ, GPTQ, bitsandbytes, and TensorRT. ORT static INT8 runs
for real when onnxruntime is installed and skips when it is not. The
SmolVLA INT4 AWQ load test skips unless a CUDA device and the awq package
are both present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ohho.optimize.errors import OptimizeCapabilityError, OptimizeError
from ohho.optimize.manifest import schema_text
from ohho.optimize.models import REFERENCE_ACT, LoadedModel
from ohho.optimize.plan import build_plan
from ohho.optimize.publish import run_parity
from ohho.optimize.quantizers import get_quantizer, list_quantizers, reset_quantizers
from ohho.optimize.quantizers.base import QuantArtifact, QuantCalibration, Quantizer
from ohho.optimize.quantizers.fp8 import ada_or_hopper
from ohho.optimize.quantizers.load import load_quantized
from ohho.targets.schema import get_profile

ROOT = Path(__file__).resolve().parents[1]


def _request(**kwargs):
    from ohho.optimize.plan import OptimizeRequest

    base = {
        "model": "reference-act",
        "target": "x86-cpu",
        "precision": "fp32",
        "out_dir": "out",
    }
    base.update(kwargs)
    return OptimizeRequest(**base)


def _int4_profile(directory: Path) -> Path:
    path = directory / "int4-test.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ohho.target_profile.v1",
                "id": "int4-test",
                "arch": "x86_64",
                "accelerator": "dgpu",
                "precisions": ["FP32", "INT4"],
                "execution_providers": [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
                "tensorrt": None,
                "memory_budget_mib": 8192,
                "workspace_size_mib": 0,
                "slos_p95_ms": {
                    "act": 10,
                    "smolvla": 20,
                    "diffusion": 20,
                    "openvla": 20,
                },
                "summary": "Test profile that lists INT4 on a discrete GPU.",
            }
        ),
        encoding="utf-8",
    )
    return path


def _hf_dir(root: Path) -> Path:
    model = root / "hf"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def _ort_stack() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "onnx", "onnxruntime")
    )


def _fake_tensorrt():
    """Load FakeTensorRT from the optimize tests without a tests package."""
    cached = getattr(_fake_tensorrt, "cls", None)
    if cached is not None:
        return cached
    path = Path(__file__).with_name("test_optimize.py")
    spec = importlib.util.spec_from_file_location("_ohho_test_optimize_fake", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load FakeTensorRT.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _fake_tensorrt.cls = module.FakeTensorRT
    return module.FakeTensorRT


def _awq_gpu() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    if (
        importlib.util.find_spec("awq") is None
        and importlib.util.find_spec("autoawq") is None
    ):
        return False
    import torch

    return bool(torch.cuda.is_available())


class _Writer(Quantizer):
    """Stand-in that writes one artifact and does not import a quant package."""

    def __init__(self, name: str) -> None:
        self.name = name

    def requirements(self) -> tuple[str, ...]:
        return ()

    def supports(self, target, model) -> tuple[bool, str]:
        del target, model
        return True, self.name

    def quantize(self, model, calib: QuantCalibration) -> QuantArtifact:
        del model
        out = Path(calib.out_dir)
        if self.name == "ort-int8":
            path = out / "model.int8.onnx"
            path.write_bytes(b"int8")
        elif self.name == "fp8":
            path = out / "model.engine"
            path.write_bytes(b"fp8")
        elif self.name == "bitsandbytes":
            path = out / "bitsandbytes.pt"
            path.write_bytes(b"bnb")
        else:
            path = out / self.name
            path.mkdir()
            (path / "weights.bin").write_bytes(b"q")
        return QuantArtifact(path, self.name, "9.9.9", self.name)

    def version(self) -> str:
        return "9.9.9"


class TestRegistry(unittest.TestCase):
    def tearDown(self) -> None:
        reset_quantizers()

    def test_builtins_and_entry_points(self) -> None:
        reset_quantizers()
        names = list_quantizers()
        self.assertEqual(names, ["awq", "bitsandbytes", "fp8", "gptq", "ort-int8"])
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[project.entry-points."ohho.quantizers"]', text)
        for name in names:
            self.assertIn(f"{name} = ", text)
        awq = get_quantizer("awq")
        self.assertEqual(awq.requirements(), ("awq", "autoawq"))
        bnb = get_quantizer("bitsandbytes")
        self.assertTrue(bnb.research_only)
        self.assertEqual(bnb.requirements(), ("bitsandbytes",))
        self.assertEqual(get_quantizer("ort-int8").requirements(), ("onnxruntime",))
        self.assertEqual(get_quantizer("fp8").requirements(), ("tensorrt",))
        self.assertIn("auto_gptq", get_quantizer("gptq").requirements())

    def test_manifest_schema_records_backend_name_and_version(self) -> None:
        document = json.loads(schema_text())
        item = document["properties"]["outputs"]["items"]
        self.assertIn("backend", item["required"])
        self.assertIn("backend_version", item["required"])
        self.assertIn("ort-int8", item["properties"]["role"]["enum"])

    def test_parity_hook_is_a_no_op_without_ohho_parity(self) -> None:
        self.assertIsNone(run_parity({"outputs": []}, "."))


class TestPrecisionDispatch(unittest.TestCase):
    def test_int4_aliases_select_the_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = _int4_profile(Path(tmp))
            awq = build_plan(
                _request(
                    target=str(profile),
                    precision="int4-awq",
                    out_dir=str(Path(tmp) / "out"),
                )
            )
            gptq = build_plan(
                _request(
                    target=str(profile),
                    precision="int4-gptq",
                    out_dir=str(Path(tmp) / "out"),
                )
            )
            bnb = build_plan(
                _request(
                    target=str(profile),
                    precision="int4-bnb",
                    out_dir=str(Path(tmp) / "out"),
                )
            )
            nf4 = build_plan(
                _request(
                    target=str(profile),
                    precision="nf4",
                    out_dir=str(Path(tmp) / "out"),
                )
            )
        self.assertEqual((awq.precision, awq.quant), ("INT4", "awq"))
        self.assertEqual(gptq.quant, "gptq")
        self.assertEqual(bnb.quant, "bitsandbytes")
        self.assertEqual(nf4.quant, "bitsandbytes")
        self.assertFalse(awq.export_onnx)

    def test_int4_awq_still_accepts_the_quant_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = _int4_profile(Path(tmp))
            plan = build_plan(
                _request(
                    target=str(profile),
                    precision="int4",
                    quant="awq",
                    out_dir=str(Path(tmp) / "out"),
                )
            )
        self.assertEqual(plan.quant, "awq")

    def test_conflicting_precision_and_quant_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = _int4_profile(Path(tmp))
            with self.assertRaises(OptimizeError) as caught:
                build_plan(
                    _request(
                        target=str(profile),
                        precision="int4-awq",
                        quant="gptq",
                        out_dir=str(Path(tmp) / "out"),
                    )
                )
        self.assertIn("conflicts", str(caught.exception))

    def test_fp8_plan_on_hopper_and_ada(self) -> None:
        for profile_id in ("cloud-h100", "cloud-l4"):
            plan = build_plan(_request(target=profile_id, precision="fp8"))
            self.assertEqual(plan.precision, "FP8")
            self.assertEqual(plan.quant, "fp8")
            self.assertTrue(plan.build_tensorrt)
            self.assertTrue(ada_or_hopper(get_profile(profile_id)))

    def test_fp8_on_a_non_fp8_gpu_is_rejected_before_work(self) -> None:
        from ohho.optimize.engine import optimize

        for profile_id in ("cloud-a10g", "x86-cuda", "jetson-orin-nano", "rpi5-cpu"):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out"
                with patch("ohho.optimize.engine.load_model") as load:
                    with self.assertRaises(OptimizeError) as caught:
                        optimize(
                            _request(
                                target=profile_id,
                                precision="fp8",
                                out_dir=str(out),
                            )
                        )
                text = str(caught.exception)
                self.assertIn("FP8", text)
                self.assertIn("Hopper", text)
                if profile_id == "jetson-orin-nano":
                    self.assertIn("Orin", text)
                load.assert_not_called()
                self.assertFalse(out.exists())


class TestPublishHook(unittest.TestCase):
    def test_each_backend_output_goes_through_run_parity(self) -> None:
        from ohho.optimize.engine import optimize

        seen: list[str] = []

        def _spy(document, out_dir) -> None:
            del out_dir
            seen.append(document["outputs"][0]["backend"])
            self.assertEqual(document["outputs"][0]["backend_version"], "9.9.9")

        def _session(*args, **kwargs):
            del args, kwargs
            return SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        def _passing_report(*args, **kwargs):
            del args, kwargs
            from ohho.parity.sets import unscored_parity_set
            from ohho.parity.thresholds import resolve_thresholds

            return {
                "schema": "ohho.parity/v1",
                "result": "pass",
                "parity_set": unscored_parity_set(),
                "reasons": [],
                "reference": {
                    "ref": "publish-hook-test",
                    "kind": "unknown",
                    "family": "unknown",
                },
                "candidate": {"path": None, "format": None},
                "target": "publish-hook-test",
                "precision": "FP32",
                "thresholds": resolve_thresholds(
                    "unknown", "publish-hook-test", "FP32"
                ).to_json(),
                "inputs": {
                    "source": "random",
                    "path": None,
                    "seed": 0,
                    "n_requested": 1,
                    "n": 0,
                    "indices": None,
                    "sha256": None,
                    "dtype": "float32",
                },
                "outputs": [
                    {
                        "name": "output",
                        "shape": [1],
                        "mean_abs": 0.0,
                        "max_abs": 0.0,
                        "rmse": 0.0,
                        "cosine": 1.0,
                        "per_joint": None,
                        "action_chunk": None,
                        "top1": None,
                        "kl": None,
                    }
                ],
                "unchecked": [],
            }

        cases = (
            ("int4-awq", "awq", "int4"),
            ("int4-gptq", "gptq", "int4"),
            ("int4-bnb", "bitsandbytes", "int4"),
            ("fp8", "fp8", "fp8"),
            ("int8", "ort-int8", "cpu"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _int4_profile(root)
            model = _hf_dir(root)
            for precision, backend, kind in cases:
                if kind == "int4":
                    target = str(profile)
                    standins = None
                elif kind == "fp8":
                    target = "cloud-h100"
                    from ohho.optimize.engine import OptimizeBackends

                    standins = OptimizeBackends(trt_module=object(), cuda=False)
                else:
                    target = "x86-cpu"
                    from ohho.optimize.engine import OptimizeBackends

                    standins = OptimizeBackends(
                        available_providers=("CPUExecutionProvider",)
                    )
                with (
                    patch(
                        "ohho.optimize.quantizers.get_quantizer",
                        side_effect=_Writer,
                    ),
                    patch(
                        "ohho.optimize.engine.probe_quant",
                        return_value=(True, backend),
                    ),
                    patch("ohho.optimize.engine.build_session", _session),
                    patch(
                        "ohho.parity.gate.compare_export",
                        side_effect=_passing_report,
                    ),
                    patch("ohho.optimize.publish.run_parity", _spy),
                ):
                    kwargs = {"no_cache": True}
                    if standins is not None:
                        kwargs["backends"] = standins
                    document = optimize(
                        _request(
                            model=str(model),
                            target=target,
                            precision=precision,
                            out_dir=str(root / backend),
                            allow_parity_fail=True,
                        ),
                        **kwargs,
                    )
                self.assertEqual(document["outputs"][0]["backend"], backend)
        self.assertEqual(seen, ["awq", "gptq", "bitsandbytes", "fp8", "ort-int8"])


class TestSmolVLADispatch(unittest.TestCase):
    def test_int4_awq_quantizes_the_lm_backbone(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        import torch.nn as nn

        from ohho.optimize.engine import optimize
        from ohho.optimize.exporters import select_submodule

        class _Text(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(4, 4)

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.text_model = _Text()

        class _Vlm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Inner()

        class _Expert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vlm = _Vlm()

        class _Core(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vlm_with_expert = _Expert()

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Core()

        policy = _Policy()
        backbone = select_submodule(policy, "lm")
        self.assertIs(backbone, policy.model.vlm_with_expert.vlm.model.text_model)
        loaded = LoadedModel(
            ref="smolvla",
            kind="smolvla",
            source="file",
            module=policy,
            path=None,
            onnx_path=None,
            submodule="lm",
            inputs=(),
            outputs=(),
        )

        calls: list[object] = []

        class _LM:
            def quantize(self, quant_config) -> None:
                calls.append(quant_config)

            def save_quantized(self, dest: str) -> None:
                path = Path(dest)
                path.mkdir(parents=True, exist_ok=True)
                (path / "weights.bin").write_bytes(b"awq")

        class _Auto:
            @staticmethod
            def from_pretrained(path: str):
                config = json.loads(
                    (Path(path) / "config.json").read_text(encoding="utf-8")
                )
                calls.append(config)
                return _LM()

        fake = types.ModuleType("awq")
        fake.AutoAWQForCausalLM = _Auto
        fake.__version__ = "0.2.6"
        parity: list[dict] = []

        def _spy(document, out_dir) -> None:
            del out_dir
            parity.append(document)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _int4_profile(root)
            with (
                patch("ohho.optimize.engine.load_model", return_value=loaded),
                patch("ohho.optimize.engine.probe_quant", return_value=(True, "awq")),
                patch.dict(sys.modules, {"awq": fake}),
                patch("ohho.optimize.publish.run_parity", _spy),
            ):
                document = optimize(
                    _request(
                        model="smolvla",
                        target=str(profile),
                        precision="int4-awq",
                        submodule="lm",
                        out_dir=str(root / "out"),
                        allow_parity_fail=True,
                    ),
                    no_cache=True,
                )
        self.assertEqual(calls[0]["ohho_backbone"], "lm")
        self.assertEqual(calls[0]["model_type"], "smollm")
        self.assertEqual(calls[1]["w_bit"], 4)
        output = document["outputs"][0]
        self.assertEqual(output["backend"], "awq")
        self.assertEqual(output["backend_version"], "0.2.6")
        self.assertEqual(output["role"], "awq")
        self.assertEqual(parity[0]["outputs"][0]["backend"], "awq")

    @unittest.skipUnless(
        _awq_gpu(),
        "GPU follow-up: SmolVLA INT4 AWQ was not run (needs CUDA and awq)",
    )
    def test_smolvla_lm_int4_awq_loads_in_awq(self) -> None:
        """Load the AWQ artifact with AutoAWQ. Skipped without a GPU."""
        directory = os.environ.get("OHHO_SMOLVLA_DIR", "").strip()
        if not directory:
            self.skipTest("OHHO_SMOLVLA_DIR is not set")
        from ohho.optimize.engine import optimize

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = _int4_profile(root)
            document = optimize(
                _request(
                    model=directory,
                    target=str(profile),
                    precision="int4-awq",
                    submodule="lm",
                    out_dir=str(root / "out"),
                    allow_parity_fail=True,
                ),
                no_cache=True,
            )
            artifact = root / "out" / document["outputs"][0]["path"]
            loaded = load_quantized(artifact, "awq")
        self.assertIsNotNone(loaded)


class TestGptqAndBitsAndBytes(unittest.TestCase):
    def test_gptq_plugin_refuses_to_download_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = _hf_dir(root)
            loaded = LoadedModel(
                ref=str(model),
                kind="llm",
                source="file",
                module=None,
                path=model,
                onnx_path=None,
                submodule=None,
                inputs=(),
                outputs=(),
            )
            calib = QuantCalibration(out_dir=root / "out", precision="INT4")
            with (
                patch.dict(sys.modules, {"auto_gptq": types.ModuleType("auto_gptq")}),
                self.assertRaises(OptimizeError) as caught,
            ):
                get_quantizer("gptq").quantize(loaded, calib)
        self.assertIn("examples", str(caught.exception))

    def test_bitsandbytes_int4_uses_nf4(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
        import torch.nn as nn

        seen: list[str] = []

        class _Linear4bit(nn.Linear):
            def __init__(self, in_features, out_features, bias=True, quant_type="fp4"):
                super().__init__(int(in_features), int(out_features), bias=bool(bias))
                self.quant_type = quant_type
                seen.append(quant_type)

        fake = types.ModuleType("bitsandbytes")
        fake.nn = SimpleNamespace(Linear4bit=_Linear4bit, Linear8bitLt=nn.Linear)
        fake.__version__ = "0.43.0"
        module = nn.Sequential(nn.Linear(3, 2))
        loaded = LoadedModel(
            ref="tiny",
            kind="module",
            source="file",
            module=module,
            path=None,
            onnx_path=None,
            submodule=None,
            inputs=(),
            outputs=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            calib = QuantCalibration(out_dir=Path(tmp), precision="INT4")
            with patch.dict(sys.modules, {"bitsandbytes": fake}):
                artifact = get_quantizer("bitsandbytes").quantize(loaded, calib)
            self.assertEqual(seen, ["nf4"])
            self.assertTrue(artifact.path.is_file())
            self.assertEqual(artifact.backend, "bitsandbytes")
        self.assertTrue(get_quantizer("bitsandbytes").research_only)


class TestFp8Flag(unittest.TestCase):
    def test_builder_flag_is_set_and_a_non_fp8_gpu_is_refused(self) -> None:
        from ohho.optimize.tensorrt import build_engine

        FakeTensorRT = _fake_tensorrt()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx = root / "model.onnx"
            onnx.write_bytes(b"onnx")
            ok = FakeTensorRT(input_shape=(-1, 4), invoke_calibrator=False)
            built = build_engine(
                onnx,
                root / "model.engine",
                precision="FP8",
                workspace_bytes=1024,
                max_batch=1,
                dynamic_batch=False,
                trt_module=ok,
            )
            self.assertEqual(ok.last_config.flags, ["fp8"])
            self.assertTrue(built.is_file())
            refused = FakeTensorRT(
                input_shape=(-1, 4), invoke_calibrator=False, fast_fp8=False
            )
            with self.assertRaises(OptimizeCapabilityError) as caught:
                build_engine(
                    onnx,
                    root / "refused.engine",
                    precision="FP8",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    trt_module=refused,
                )
        self.assertIn("Ada or Hopper", str(caught.exception))


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") is not None,
    "onnxruntime is not installed",
)
class TestOrtStaticInt8(unittest.TestCase):
    def test_x86_cpu_and_rpi5(self) -> None:
        if not _ort_stack():
            self.skipTest("onnxruntime is installed but torch or onnx is not")
        from ohho.optimize.engine import optimize

        parity: list[str] = []

        def _spy(document, out_dir) -> None:
            del out_dir
            parity.append(document["outputs"][0]["backend"])

        for profile_id in ("x86-cpu", "rpi5-cpu"):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out"
                with patch("ohho.optimize.publish.run_parity", _spy):
                    document = optimize(
                        _request(
                            model="reference-act",
                            target=profile_id,
                            precision="int8",
                            out_dir=str(out),
                            allow_parity_fail=True,
                        ),
                        no_cache=True,
                    )
                self.assertEqual(document["precision"], "INT8")
                self.assertEqual(document["flags"]["quant"], "ort-int8")
                item = document["outputs"][0]
                self.assertEqual(item["role"], "ort-int8")
                self.assertEqual(item["backend"], "ort-int8")
                self.assertTrue(item["backend_version"])
                artifact = out / item["path"]
                self.assertIn(b"QuantizeLinear", artifact.read_bytes())
                session = load_quantized(artifact, "ort-int8")
                self.assertIn("CPUExecutionProvider", session.get_providers())
                self.assertEqual(
                    session.get_inputs()[0].name,
                    "state",
                )
                self.assertEqual(
                    document["session_providers"][0], "CPUExecutionProvider"
                )
        self.assertEqual(parity, ["ort-int8", "ort-int8"])


@unittest.skipUnless(_ort_stack(), "optimize extra (torch, onnx, onnxruntime)")
class TestFp8EngineDispatch(unittest.TestCase):
    def test_fp8_backend_sets_the_flag_on_a_fake_builder(self) -> None:
        from ohho.optimize.engine import OptimizeBackends, optimize

        FakeTensorRT = _fake_tensorrt()

        trt = FakeTensorRT(
            input_shape=(-1, int(REFERENCE_ACT["state_dim"])),
            invoke_calibrator=False,
        )
        seen: list[dict] = []

        def _spy(document, out_dir) -> None:
            del out_dir
            seen.append(document)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("ohho.optimize.publish.run_parity", _spy):
                document = optimize(
                    _request(
                        model="reference-act",
                        target="cloud-h100",
                        precision="fp8",
                        out_dir=str(out),
                        allow_parity_fail=True,
                    ),
                    backends=OptimizeBackends(trt_module=trt, cuda=False),
                    no_cache=True,
                )
            self.assertIn("fp8", trt.last_config.flags)
            self.assertEqual(document["outputs"][0]["backend"], "fp8")
            self.assertEqual(document["outputs"][0]["role"], "fp8")
            self.assertTrue((out / "model.engine").is_file())
            self.assertEqual(seen[0]["outputs"][0]["backend"], "fp8")


if __name__ == "__main__":
    unittest.main()
