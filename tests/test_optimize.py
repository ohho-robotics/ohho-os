"""ONNX, TensorRT, ORT, and weight-quant engine (OHH-429, OHH-467)."""

from __future__ import annotations

import array
import ast
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ohho.cli import main
from ohho.optimize import (
    OptimizeBackends,
    OptimizeCapabilityError,
    OptimizeError,
    OptimizeRequest,
    build_engine,
    build_plan,
    build_session,
    export_onnx,
    export_torchscript,
    load_calibration_source,
    load_model,
    make_entropy_calibrator,
    optimize,
    probe_quant,
    quantize,
    register_exporter,
    schema_text,
)
from ohho.optimize.calibrator import (
    EmptyBatchSource,
    cache_sidecar_path,
    entropy_scale,
    preprocessing_config,
    preprocessing_config_from_dir,
)
from ohho.optimize.qdq import insert_qdq, quantization_mode
from ohho.optimize.exporters import (
    describe_export,
    exporter_names,
    select_submodule,
    unregister_exporter,
)
from ohho.optimize.manifest import SCHEMA_ID, manifest_problems
from ohho.optimize.models import (
    REFERENCE_ACT,
    REFERENCE_MLP,
    LoadedModel,
    build_reference_act,
    build_reference_mlp,
)
from ohho.optimize.quant import replace_linear

ROOT = Path(__file__).resolve().parents[1]
_HEAVY = {"torch", "onnx", "onnxruntime", "tensorrt", "numpy", "awq", "autoawq"}


def _has_export_stack() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "onnx", "onnxruntime")
    )


def _live_tensorrt() -> bool:
    if (
        importlib.util.find_spec("tensorrt") is None
        or importlib.util.find_spec("torch") is None
    ):
        return False
    import torch

    return bool(torch.cuda.is_available())


def _bare_model() -> LoadedModel:
    return LoadedModel(
        ref="reference-act",
        kind="act",
        source="registry",
        module=None,
        path=None,
        onnx_path=None,
        submodule=None,
        inputs=(),
        outputs=(),
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _request(**kwargs) -> OptimizeRequest:
    base = {
        "model": "reference-act",
        "target": "x86-cpu",
        "precision": "fp32",
        "out_dir": "out",
    }
    base.update(kwargs)
    return OptimizeRequest(**base)


def _write_bins(root: Path, shape: tuple[int, ...], count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calib.json").write_text(
        json.dumps({"shape": list(shape), "dtype": "float32"}),
        encoding="utf-8",
    )
    width = 1
    for dim in shape:
        width *= dim
    for index in range(count):
        values = array.array("f", [0.01 * (index + 1)] * width)
        (root / f"{index:03d}.bin").write_bytes(values.tobytes())


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


class _CalibratorBase:
    def __init__(self) -> None:
        return None


class _Logger:
    WARNING = 1

    def __init__(self, level: int) -> None:
        self.level = level


class _Parser:
    num_errors = 0

    def __init__(self, network, logger) -> None:
        self.network = network
        self.logger = logger

    def parse(self, data: bytes) -> bool:
        self.data = data
        return True

    def get_error(self, index: int) -> str:
        return f"error {index}"


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.name = "input"
        self.shape = shape


class _Network:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.num_inputs = 1
        self._tensor = _Tensor(shape)

    def get_input(self, index: int) -> _Tensor:
        del index
        return self._tensor


class _Profile:
    def __init__(self) -> None:
        self.shapes: list[tuple] = []

    def set_shape(self, name, low, opt, high) -> None:
        self.shapes.append((name, tuple(low), tuple(opt), tuple(high)))


class _Config:
    def __init__(self) -> None:
        self.flags: list[object] = []
        self.pool = None
        self.int8_calibrator = None
        self.profiles: list[_Profile] = []

    def set_memory_pool_limit(self, pool, nbytes: int) -> None:
        self.pool = (pool, nbytes)

    def set_flag(self, flag) -> None:
        self.flags.append(flag)

    def add_optimization_profile(self, profile: _Profile) -> None:
        self.profiles.append(profile)


class _Builder:
    def __init__(self, api: FakeTensorRT) -> None:
        self.api = api
        self.platform_has_fast_fp16 = api.fast_fp16
        self.platform_has_fast_int8 = api.fast_int8
        self.platform_has_fp8 = api.fast_fp8
        self.network = _Network(api.input_shape)
        self.config = _Config()

    def create_network(self, flags: int) -> _Network:
        self.flags = flags
        return self.network

    def create_builder_config(self) -> _Config:
        return self.config

    def create_optimization_profile(self) -> _Profile:
        return _Profile()

    def build_serialized_network(self, network, config) -> bytes:
        del network
        self.api.last_config = config
        calibrator = config.int8_calibrator
        if calibrator is not None and self.api.invoke_calibrator:
            cached = calibrator.read_calibration_cache()
            self.api.last_cache_read = cached
            if cached:
                self.api.cache_reuses += 1
            else:
                self.api.calibration_runs += 1
                batch = calibrator.get_batch(["input"])
                self.api.last_batch = batch
                calibrator.write_calibration_cache(b"cache-bytes")
        return b"engine-bytes"


class FakeTensorRT:
    """Enough of the TensorRT builder API to prove the calibrator is attached."""

    def __init__(
        self,
        *,
        fast_fp16: bool = True,
        fast_int8: bool = True,
        fast_fp8: bool | None = None,
        invoke_calibrator: bool = True,
        input_shape: tuple[int, ...] = (-1, 4),
    ) -> None:
        self.fast_fp16 = fast_fp16
        self.fast_int8 = fast_int8
        self.fast_fp8 = fast_fp8
        self.invoke_calibrator = invoke_calibrator
        self.input_shape = input_shape
        self.last_config = None
        self.last_batch = None
        self.last_cache_read = None
        self.last_builder = None
        self.last_parser = None
        self.cache_reuses = 0
        self.calibration_runs = 0
        self.Logger = _Logger
        self.NetworkDefinitionCreationFlag = SimpleNamespace(EXPLICIT_BATCH=0)
        self.MemoryPoolType = SimpleNamespace(WORKSPACE="workspace")
        self.BuilderFlag = SimpleNamespace(FP16="fp16", INT8="int8", FP8="fp8")
        self.IInt8EntropyCalibrator2 = _CalibratorBase

    def OnnxParser(self, network, logger) -> _Parser:
        parser = _Parser(network, logger)
        self.last_parser = parser
        return parser

    def Builder(self, logger) -> _Builder:
        del logger
        builder = _Builder(self)
        self.last_builder = builder
        return builder


class TestImportAndSchema(unittest.TestCase):
    def test_modules_do_not_import_heavy_packages_at_import(self) -> None:
        package = ROOT / "ohho" / "optimize"
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    self.assertNotIn(name, _HEAVY, path.name)

    def test_schema_is_package_data(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("optimize/*.json", text)
        self.assertIn("optimize = ", text)
        document = json.loads(schema_text())
        self.assertEqual(
            document["$id"],
            "https://ohho-robotics.com/schemas/optimize-manifest.v1.json",
        )
        self.assertEqual(document["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertIn("sha256", document["properties"]["outputs"]["items"]["required"])
        self.assertIn("calib_max_images", document["properties"]["flags"]["required"])
        self.assertIn("quantization", document["properties"]["flags"]["required"])
        self.assertIn("calib_max_images", document["properties"]["inputs"]["required"])
        self.assertIn("model_sha256", document["required"])
        self.assertIn("calibration_cache_sha256", document["required"])
        self.assertIn("build_duration_s", document["required"])
        self.assertIn("cuda", document["properties"]["versions"]["required"])
        self.assertIn("driver", document["properties"]["versions"]["required"])
        self.assertIn("jetpack", document["properties"]["versions"]["required"])

    def test_help_lists_the_engine_flags(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["optimize", "--help"])
        self.assertEqual(caught.exception.code, 0)
        text = out.getvalue()
        for flag in (
            "--model",
            "--target",
            "--precision",
            "--onnx",
            "--tensorrt",
            "--calib-data",
            "--calib-batch",
            "--calib-cache",
            "--calib-max-images",
            "--ort-ep",
            "--quant",
            "--workspace-gb",
            "--max-batch",
            "--dynamic-batch",
            "--out",
            "--json",
            "--submodule",
            "--explain",
            "--no-cache",
        ):
            self.assertIn(flag, text)

    def test_doctor_mentions_optimize(self) -> None:
        code, out, _err = _run(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("optimize", out)


class TestPlanFailFast(unittest.TestCase):
    def test_fp16_on_a_cpu_profile_is_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="rpi5-cpu", precision="fp16"))
        self.assertIn("FP16", str(caught.exception))
        self.assertIn("rpi5-cpu", str(caught.exception))

    def test_fp8_on_orin_is_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(
                _request(target="jetson-orin-nano", precision="fp8", tensorrt=True)
            )
        self.assertIn("FP8", str(caught.exception))
        self.assertIn("Orin", str(caught.exception))

    def test_int8_tensorrt_on_a_cpu_profile_is_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(
                _request(
                    target="rpi5-cpu",
                    precision="int8",
                    tensorrt=True,
                    calib_data="calib",
                )
            )
        self.assertIn("INT8 TensorRT", str(caught.exception))

    def test_cuda_provider_on_a_cpu_profile_is_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="rpi5-cpu", ort_ep="cuda"))
        self.assertIn("CUDAExecutionProvider", str(caught.exception))

    def test_unknown_target(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="not-a-profile"))
        self.assertIn("Unknown target profile", str(caught.exception))

    def test_workspace_above_the_memory_budget(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="jetson-orin-nano", workspace_gb=5))
        self.assertIn("memory budget", str(caught.exception))

    def test_fp16_without_tensorrt_fails_fast(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="x86-cuda", precision="fp16", ort_ep="cuda"))
        self.assertIn("--tensorrt", str(caught.exception))

    def test_int8_on_cpu_selects_ort_static(self) -> None:
        plan = build_plan(_request(target="x86-cpu", precision="int8"))
        self.assertEqual(plan.precision, "INT8")
        self.assertEqual(plan.quant, "ort-int8")
        self.assertTrue(plan.build_session)
        self.assertFalse(plan.build_tensorrt)
        rpi = build_plan(_request(target="rpi5-cpu", precision="int8"))
        self.assertEqual(rpi.quant, "ort-int8")

    def test_int8_on_a_gpu_profile_without_a_quantizer_fails_fast(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="x86-cuda", precision="int8"))
        self.assertIn("--quant", str(caught.exception))

    def test_int8_tensorrt_requires_calibration_data(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="x86-cuda", precision="int8", tensorrt=True))
        text = str(caught.exception)
        self.assertIn("--calib-data", text)
        self.assertIn("Q/DQ", text)
        self.assertIn("no Q/DQ", text)

    def test_int8_qdq_onnx_does_not_need_calibration_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"QuantizeLinear\nDequantizeLinear\n")
            plan = build_plan(
                _request(
                    model=str(onnx),
                    target="x86-cuda",
                    precision="int8",
                    tensorrt=True,
                    out_dir=str(Path(tmp) / "out"),
                )
            )
        self.assertEqual(plan.precision, "INT8")
        self.assertTrue(plan.build_tensorrt)
        self.assertIsNone(plan.calib_batch)

    def test_plain_onnx_without_qdq_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"fake-onnx")
            with self.assertRaises(OptimizeError) as caught:
                build_plan(
                    _request(
                        model=str(onnx),
                        target="x86-cuda",
                        precision="int8",
                        tensorrt=True,
                        out_dir=str(Path(tmp) / "out"),
                    )
                )
        self.assertIn("no Q/DQ", str(caught.exception))

    def test_calib_max_images_on_fp32_is_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(calib_max_images=4))
        self.assertIn("INT8", str(caught.exception))

    def test_calib_max_images_must_cover_the_batch(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(
                _request(
                    target="x86-cuda",
                    precision="int8",
                    tensorrt=True,
                    calib_data="calib",
                    calib_batch=8,
                    calib_max_images=4,
                )
            )
        self.assertIn("--calib-max-images", str(caught.exception))

    def test_calibration_flags_on_fp32_are_rejected(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(calib_data="calib"))
        self.assertIn("INT8", str(caught.exception))

    def test_tensorrt_ep_conflicts_with_cpu_session(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(
                _request(
                    target="x86-cuda",
                    precision="fp16",
                    tensorrt=True,
                    ort_ep="cpu",
                )
            )
        self.assertIn("--ort-ep", str(caught.exception))

    def test_int4_needs_a_profile_that_lists_it(self) -> None:
        with self.assertRaises(OptimizeError) as caught:
            build_plan(_request(target="x86-cuda", precision="int4", quant="awq"))
        self.assertIn("INT4", str(caught.exception))

    def test_int4_plan_passes_on_a_profile_that_lists_it(self) -> None:
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
        self.assertEqual(plan.precision, "INT4")
        self.assertEqual(plan.quant, "awq")
        self.assertFalse(plan.export_onnx)

    def test_bad_tensorrt_version_is_rejected(self) -> None:
        with patch("ohho.optimize.plan.installed_trt_version", return_value="7.0.0"):
            with self.assertRaises(OptimizeError) as caught:
                build_plan(_request(target="x86-cuda", precision="fp16", tensorrt=True))
        self.assertIn("TensorRT 7.0.0", str(caught.exception))

    def test_fp16_engine_without_tensorrt_fails_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OptimizeCapabilityError) as caught:
                optimize(
                    _request(
                        target="x86-cuda",
                        precision="fp16",
                        tensorrt=True,
                        out_dir=str(Path(tmp) / "out"),
                    )
                )
        self.assertIn("TensorRT", str(caught.exception))
        self.assertIn("GPU follow-up", str(caught.exception))


class TestQuantCapability(unittest.TestCase):
    def test_missing_packages_fail_fast(self) -> None:
        awq_ok, awq_reason = probe_quant("awq")
        gptq_ok, gptq_reason = probe_quant("gptq")
        bnb_ok, bnb_reason = probe_quant("bitsandbytes")
        self.assertFalse(awq_ok)
        self.assertIn("AWQ", awq_reason)
        self.assertFalse(gptq_ok)
        self.assertIn("GPTQ", gptq_reason)
        self.assertFalse(bnb_ok)
        self.assertIn("bitsandbytes", bnb_reason)
        with self.assertRaises(OptimizeCapabilityError):
            quantize(_bare_model(), "awq", "out", "INT4")

    def test_fp8_without_cuda_is_a_gpu_follow_up(self) -> None:
        ok, reason = probe_quant("fp8")
        if importlib.util.find_spec("torch") is None:
            self.assertFalse(ok)
            return
        import torch

        if torch.cuda.is_available():
            self.assertTrue(ok)
            return
        self.assertFalse(ok)
        self.assertIn("CUDA", reason)

    def test_awq_calls_the_library_for_a_local_directory(self) -> None:
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
                calls.append(path)
                return _LM()

        fake = types.ModuleType("awq")
        fake.AutoAWQForCausalLM = _Auto
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "hf"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            loaded = load_model(str(model))
            with (
                patch("ohho.optimize.quant.probe_quant", return_value=(True, "awq")),
                patch.dict(sys.modules, {"awq": fake}),
            ):
                written = quantize(loaded, "awq", root / "out", "INT4")
            self.assertTrue((written / "weights.bin").is_file())
            self.assertEqual(calls[0], str(model))
            self.assertEqual(calls[1]["w_bit"], 4)

    def test_gptq_refuses_to_download_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "hf"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            loaded = load_model(str(model))
            with (
                patch(
                    "ohho.optimize.quant.probe_quant", return_value=(True, "auto_gptq")
                ),
                self.assertRaises(OptimizeError) as caught,
            ):
                quantize(loaded, "gptq", Path(tmp) / "out", "INT4")
        self.assertIn("examples", str(caught.exception))

    def test_awq_without_config_json_is_rejected(self) -> None:
        with (
            patch("ohho.optimize.quant.probe_quant", return_value=(True, "awq")),
            self.assertRaises(OptimizeError) as caught,
        ):
            quantize(_bare_model(), "awq", "out", "INT4")
        self.assertIn("config.json", str(caught.exception))


class TestCalibratorAndTensorRT(unittest.TestCase):
    def test_cache_round_trip_and_empty_source(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.bin"
            cache.write_bytes(b"cached")
            calibrator = make_entropy_calibrator(
                EmptyBatchSource(),
                batch_size=2,
                cache_path=cache,
                trt_module=trt,
            )
            self.assertEqual(calibrator.get_batch_size(), 2)
            self.assertEqual(calibrator.read_calibration_cache(), b"cached")
            self.assertIsNone(calibrator.get_batch(["input"]))
            calibrator.write_calibration_cache(b"next")
            self.assertEqual(cache.read_bytes(), b"next")

    def test_directory_source_groups_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root, (4,), 4)
            source = load_calibration_source(root, batch_size=2, sample_shape=(4,))
            self.assertEqual(source.nbatches, 2)
            first = source.batch(0)
            self.assertEqual(len(first), 8)
            with self.assertRaises(OptimizeError):
                load_calibration_source(root, batch_size=2, sample_shape=(8,))

    def test_int8_builder_attaches_the_calibrator(self) -> None:
        trt = FakeTensorRT(input_shape=(-1, 4))
        trt.__version__ = "8.6.1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root / "calib", (4,), 4)
            cache = root / "int8.cache"
            source = load_calibration_source(
                root / "calib", batch_size=2, sample_shape=(4,)
            )
            calibrator = make_entropy_calibrator(
                source,
                batch_size=2,
                cache_path=cache,
                trt_module=trt,
            )
            onnx = root / "model.onnx"
            onnx.write_bytes(b"fake-onnx")
            engine = build_engine(
                onnx,
                root / "model.engine",
                precision="INT8",
                workspace_bytes=256 * (1 << 20),
                max_batch=4,
                dynamic_batch=True,
                calibrator=calibrator,
                trt_module=trt,
            )
            config = trt.last_config
            self.assertIs(config.int8_calibrator, calibrator)
            self.assertEqual(quantization_mode(trt), "implicit")
            self.assertIn("int8", config.flags)
            self.assertEqual(config.pool, ("workspace", 256 * (1 << 20)))
            self.assertEqual(config.profiles[0].shapes[0][1][0], 1)
            self.assertEqual(config.profiles[0].shapes[0][3][0], 4)
            self.assertIsNotNone(trt.last_batch)
            self.assertEqual(cache.read_bytes(), b"cache-bytes")
            self.assertEqual(engine.read_bytes(), b"engine-bytes")

    def test_int8_without_a_calibrator_is_rejected(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False)
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"fake-onnx")
            with self.assertRaises(OptimizeError) as caught:
                build_engine(
                    onnx,
                    Path(tmp) / "model.engine",
                    precision="INT8",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    calibrator=None,
                    trt_module=trt,
                )
        self.assertIn("calibrator", str(caught.exception))
        self.assertIn("Q/DQ", str(caught.exception))

    def test_versions_choose_implicit_or_explicit_qdq(self) -> None:
        self.assertEqual(
            quantization_mode(SimpleNamespace(__version__="8.6.1")), "implicit"
        )
        self.assertEqual(
            quantization_mode(SimpleNamespace(__version__="9.3.0.post12")), "implicit"
        )
        self.assertEqual(quantization_mode(SimpleNamespace()), "implicit")
        self.assertEqual(
            quantization_mode(SimpleNamespace(__version__="10.0.1")), "explicit"
        )
        self.assertEqual(
            quantization_mode(SimpleNamespace(__version__="10.9.0.34")), "explicit"
        )

    def test_tensorrt_10_uses_explicit_qdq_and_skips_the_calibrator(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False, input_shape=(-1, 4))
        trt.__version__ = "10.2.0"
        trt.NetworkDefinitionCreationFlag = SimpleNamespace(
            EXPLICIT_BATCH=0, STRONGLY_TYPED=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"QuantizeLinear DequantizeLinear")
            with self.assertLogs("ohho.optimize", level="INFO") as logs:
                build_engine(
                    onnx,
                    Path(tmp) / "model.engine",
                    precision="INT8",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    calibrator=None,
                    trt_module=trt,
                )
        self.assertIsNone(trt.last_config.int8_calibrator)
        self.assertNotIn("int8", trt.last_config.flags)
        self.assertEqual(trt.last_builder.flags, 1 << 1)
        self.assertIsNotNone(trt.last_parser)
        self.assertIn(b"QuantizeLinear", trt.last_parser.data)
        self.assertTrue(any("explicit Q/DQ" in line for line in logs.output))

    def test_explicit_int8_without_calibration_or_qdq_is_refused(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False)
        trt.__version__ = "10.1.0"
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"not-quantized")
            with self.assertRaises(OptimizeError) as caught:
                build_engine(
                    onnx,
                    Path(tmp) / "model.engine",
                    precision="INT8",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    calibrator=None,
                    trt_module=trt,
                )
        text = str(caught.exception)
        self.assertIn("Q/DQ", text)
        self.assertIn("--calib-data", text)

    def test_cache_written_reused_and_invalidated(self) -> None:
        trt = FakeTensorRT(input_shape=(-1, 4))
        trt.__version__ = "8.6.1"
        sha = "a" * 64
        other = "b" * 64
        preprocessing = preprocessing_config(shape=(4,), max_images=4)
        changed = preprocessing_config(shape=(4,), max_images=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root / "calib", (4,), 4)
            cache = root / "int8.cache"
            source = load_calibration_source(
                root / "calib", batch_size=2, sample_shape=(4,), max_images=4
            )
            onnx = root / "model.onnx"
            onnx.write_bytes(b"fake-onnx")

            def _build(model_sha: str, prep: dict) -> None:
                calibrator = make_entropy_calibrator(
                    source,
                    batch_size=2,
                    cache_path=cache,
                    trt_module=trt,
                    model_sha256=model_sha,
                    preprocessing=prep,
                )
                build_engine(
                    onnx,
                    root / "model.engine",
                    precision="INT8",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    calibrator=calibrator,
                    trt_module=trt,
                )

            with self.assertLogs("ohho.optimize", level="INFO") as first:
                _build(sha, preprocessing)
            sidecar = json.loads(cache_sidecar_path(cache).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["model_sha256"], sha)
            self.assertEqual(sidecar["preprocessing"]["max_images"], 4)
            self.assertEqual(sidecar["preprocessing"]["shape"], [4])
            self.assertEqual(sidecar["quantization"], "implicit")
            self.assertEqual(cache.read_bytes(), b"cache-bytes")
            self.assertTrue(
                any("Wrote INT8 calibration cache" in line for line in first.output)
            )
            self.assertEqual(trt.calibration_runs, 1)
            self.assertEqual(trt.cache_reuses, 0)

            trt.last_batch = None
            with self.assertLogs("ohho.optimize", level="INFO") as second:
                _build(sha, preprocessing)
            self.assertTrue(
                any("Reusing INT8 calibration cache" in line for line in second.output)
            )
            self.assertIsNone(trt.last_batch)
            self.assertEqual(trt.calibration_runs, 1)
            self.assertEqual(trt.cache_reuses, 1)

            with self.assertLogs("ohho.optimize", level="INFO") as third:
                _build(other, preprocessing)
            self.assertTrue(
                any("model sha256 changed" in line for line in third.output)
            )
            self.assertEqual(trt.calibration_runs, 2)

            with self.assertLogs("ohho.optimize", level="INFO") as fourth:
                _build(other, changed)
            self.assertTrue(
                any("preprocessing config changed" in line for line in fourth.output)
            )
            self.assertEqual(trt.calibration_runs, 3)

    def test_max_images_limits_samples_and_preprocessing_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root, (4,), 6)
            source = load_calibration_source(
                root, batch_size=2, sample_shape=(4,), max_images=4
            )
            self.assertEqual(source.nbatches, 2)
            with self.assertRaises(OptimizeError) as caught:
                load_calibration_source(
                    root, batch_size=4, sample_shape=(4,), max_images=2
                )
            self.assertIn("--calib-max-images", str(caught.exception))
            meta = json.loads((root / "calib.json").read_text(encoding="utf-8"))
            meta["preprocessing"] = {
                "mean": [0.5],
                "std": [0.25],
                "resize": [8, 8],
                "layout": "nchw",
            }
            (root / "calib.json").write_text(json.dumps(meta), encoding="utf-8")
            prep = preprocessing_config_from_dir(
                root, shape=(4,), max_images=4, submodule="vision"
            )
            self.assertEqual(prep["mean"], [0.5])
            self.assertEqual(prep["std"], [0.25])
            self.assertEqual(prep["resize"], [8, 8])
            self.assertEqual(prep["layout"], "nchw")
            self.assertEqual(prep["submodule"], "vision")
            meta["preprocessing"] = {"crop": 1}
            (root / "calib.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(OptimizeError) as caught:
                load_calibration_source(root, batch_size=2, sample_shape=(4,))
            self.assertIn("preprocessing", str(caught.exception))

    def test_entropy_scale_is_positive_and_bounded(self) -> None:
        zeros = array.array("f", [0.0, 0.0, 0.0])
        self.assertEqual(entropy_scale(zeros), 1.0)
        values = array.array("f", [1.0, -1.0, 0.5, -0.25, 0.0])
        scale = entropy_scale(values)
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, (1.0 / 127.0) + 1e-12)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None, "onnx package is not installed"
    )
    def test_insert_qdq_adds_quantize_and_dequantize(self) -> None:
        import onnx
        from onnx import TensorProto, helper

        weight = helper.make_tensor(
            "W", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0]
        )
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["input", "W"], ["output"])],
            "gemm",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            [weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        rewritten = insert_qdq(model.SerializeToString(), 0.02)
        parsed = onnx.load_from_string(rewritten)
        ops = {node.op_type for node in parsed.graph.node}
        self.assertIn("QuantizeLinear", ops)
        self.assertIn("DequantizeLinear", ops)
        self.assertIn("Gemm", ops)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None, "onnx package is not installed"
    )
    def test_tensorrt_10_inserts_qdq_and_reuses_the_scale_cache(self) -> None:
        from onnx import TensorProto, helper

        weight = helper.make_tensor(
            "W", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0]
        )
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["input", "W"], ["output"])],
            "gemm",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
            [weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        trt = FakeTensorRT(invoke_calibrator=False, input_shape=(-1, 2))
        trt.__version__ = "10.2.0"
        trt.NetworkDefinitionCreationFlag = SimpleNamespace(
            EXPLICIT_BATCH=0, STRONGLY_TYPED=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx = root / "gemm.onnx"
            onnx.write_bytes(model.SerializeToString())
            _write_bins(root / "calib", (2,), 2)
            cache = root / "int8.cache"
            request = _request(
                model=str(onnx),
                target="x86-cuda",
                precision="int8",
                tensorrt=True,
                calib_data=str(root / "calib"),
                calib_batch=1,
                calib_max_images=2,
                calib_cache=str(cache),
                out_dir=str(root / "out1"),
            )
            with self.assertLogs("ohho.optimize", level="INFO") as first:
                optimize(
                    request,
                    backends=OptimizeBackends(trt_module=trt, cuda=False),
                    no_cache=True,
                )
            self.assertIsNone(trt.last_config.int8_calibrator)
            self.assertIsNotNone(trt.last_parser)
            self.assertIn(b"QuantizeLinear", trt.last_parser.data)
            self.assertIn(b"DequantizeLinear", trt.last_parser.data)
            sidecar = json.loads(cache_sidecar_path(cache).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["quantization"], "explicit")
            self.assertIsInstance(sidecar["activation_scale"], float)
            self.assertTrue(
                any("Wrote INT8 calibration cache" in line for line in first.output)
            )
            self.assertTrue(any("explicit Q/DQ" in line for line in first.output))
            second = _request(
                model=str(onnx),
                target="x86-cuda",
                precision="int8",
                tensorrt=True,
                calib_data=str(root / "calib"),
                calib_batch=1,
                calib_max_images=2,
                calib_cache=str(cache),
                out_dir=str(root / "out2"),
            )
            with self.assertLogs("ohho.optimize", level="INFO") as reused:
                optimize(
                    second,
                    backends=OptimizeBackends(trt_module=trt, cuda=False),
                    no_cache=True,
                )
            self.assertTrue(
                any("Reusing INT8 calibration cache" in line for line in reused.output)
            )
            self.assertFalse(
                any("Wrote INT8 calibration cache" in line for line in reused.output)
            )
            self.assertIsNone(trt.last_config.int8_calibrator)

    def test_qdq_onnx_engine_does_not_attach_a_calibrator(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False, input_shape=(-1, 4))
        trt.__version__ = "10.2.0"
        trt.NetworkDefinitionCreationFlag = SimpleNamespace(
            EXPLICIT_BATCH=0, STRONGLY_TYPED=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx = root / "model.onnx"
            onnx.write_bytes(b"QuantizeLinear DequantizeLinear")
            document = optimize(
                _request(
                    model=str(onnx),
                    target="x86-cuda",
                    precision="int8",
                    tensorrt=True,
                    out_dir=str(root / "out"),
                ),
                backends=OptimizeBackends(trt_module=trt, cuda=False),
                no_cache=True,
            )
        self.assertIsNone(trt.last_config.int8_calibrator)
        self.assertEqual(document["flags"]["quantization"], "explicit")
        self.assertIsNone(document["flags"]["calib_max_images"])
        self.assertIsNone(document["inputs"]["calib_max_images"])
        roles = {item["role"] for item in document["outputs"]}
        self.assertEqual(roles, {"onnx", "engine"})

    def test_cli_int8_without_calibration_or_qdq_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _text, err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-act",
                    "--target",
                    "x86-cuda",
                    "--precision",
                    "int8",
                    "--tensorrt",
                    "--out",
                    str(Path(tmp) / "out"),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("--calib-data", err)
        self.assertIn("Q/DQ", err)

    def test_fp16_flag_and_missing_fast_fp16(self) -> None:
        trt = FakeTensorRT(invoke_calibrator=False, input_shape=(-1, 4))
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"fake-onnx")
            build_engine(
                onnx,
                Path(tmp) / "model.engine",
                precision="FP16",
                workspace_bytes=1024,
                max_batch=2,
                dynamic_batch=False,
                trt_module=trt,
            )
            self.assertIn("fp16", trt.last_config.flags)
            self.assertIsNone(trt.last_config.int8_calibrator)
            self.assertEqual(trt.last_config.profiles[0].shapes[0][1][0], 2)
            self.assertEqual(trt.last_config.profiles[0].shapes[0][3][0], 2)
        slow = FakeTensorRT(fast_fp16=False, invoke_calibrator=False)
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"fake-onnx")
            with self.assertRaises(OptimizeCapabilityError) as caught:
                build_engine(
                    onnx,
                    Path(tmp) / "model.engine",
                    precision="FP16",
                    workspace_bytes=1024,
                    max_batch=1,
                    dynamic_batch=False,
                    trt_module=slow,
                )
        self.assertIn("FP16", str(caught.exception))

    def test_session_rejects_a_missing_or_fallback_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx = Path(tmp) / "model.onnx"
            onnx.write_bytes(b"unused")
            with self.assertRaises(OptimizeCapabilityError) as caught:
                build_session(
                    onnx,
                    provider="CUDAExecutionProvider",
                    available=("CPUExecutionProvider",),
                )
            self.assertIn("CUDAExecutionProvider", str(caught.exception))

            class _Session:
                def __init__(self) -> None:
                    self.providers = ["CPUExecutionProvider"]

                def get_providers(self) -> list[str]:
                    return self.providers

            class _Ort:
                def InferenceSession(self, path, providers):
                    del path
                    self.providers = providers
                    return _Session()

            ort = _Ort()
            with self.assertRaises(OptimizeCapabilityError) as caught:
                build_session(
                    onnx,
                    provider="CUDAExecutionProvider",
                    available=("CUDAExecutionProvider", "CPUExecutionProvider"),
                    ort_module=ort,
                )
            self.assertEqual(ort.providers, ["CUDAExecutionProvider"])
            self.assertIn("did not enable", str(caught.exception))


class TestCpuExport(unittest.TestCase):
    def test_reference_act_matches_pytorch(self) -> None:
        self._assert_matches(
            "reference-act", "state", (1, int(REFERENCE_ACT["state_dim"]))
        )

    def test_reference_mlp_matches_pytorch(self) -> None:
        self._assert_matches("reference-mlp", "obs", (1, 6))

    def test_dynamic_batch_matches(self) -> None:
        import torch

        loaded = load_model("reference-mlp")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            self.assertEqual(export_onnx(loaded, path), "rl_mlp")
            sample = torch.randn(2, 6)
            self._compare(loaded.module, path, "obs", sample)

    def test_encoder_submodule_matches(self) -> None:
        import torch

        loaded = load_model("reference-act", submodule="encoder")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.onnx"
            self.assertEqual(export_onnx(loaded, path), "torch.onnx")
            sample = torch.randn(2, int(REFERENCE_ACT["state_dim"]))
            self._compare(loaded.module.encoder, path, "state", sample)

    def test_torchscript_matches(self) -> None:
        import torch

        loaded = load_model("reference-mlp")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ts"
            export_torchscript(loaded, path)
            scripted = torch.jit.load(str(path), map_location="cpu")
            sample = torch.randn(1, 6)
            with torch.no_grad():
                ref = loaded.module(sample)
                got = scripted(sample)
            self.assertTrue(torch.allclose(ref, got, rtol=1e-4, atol=1e-4))

    def test_actor_state_dict_reloads(self) -> None:
        import torch

        mlp = build_reference_mlp()
        state = {
            "model_state_dict": {
                f"actor.{key}": value for key, value in mlp.actor.state_dict().items()
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            torch.save(state, path)
            loaded = load_model(str(path))
            sample = torch.randn(1, 6)
            with torch.no_grad():
                ref = mlp.actor(sample)
                got = loaded.module(sample)
            self.assertTrue(torch.allclose(ref, got, rtol=1e-5, atol=1e-5))
            onnx = Path(tmp) / "actor.onnx"
            export_onnx(loaded, onnx)
            self._compare(loaded.module, onnx, "obs", sample)

    def test_vision_role_selects_backbone(self) -> None:
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(4, 2)

            def forward(self, features):
                return self.backbone(features)

        selected = select_submodule(_Net(), "vision")
        self.assertIsInstance(selected, nn.Linear)

    def test_custom_exporter_runs_before_the_generic_one(self) -> None:
        class _Boom:
            name = "boom"

            def matches(self, spec) -> bool:
                return spec.kind == "boom"

            def export_onnx(self, spec, path, *, opset, dynamic_batch) -> None:
                del spec, opset, dynamic_batch
                Path(path).write_bytes(b"boom")

        register_exporter(_Boom())
        try:
            loaded = load_model("reference-mlp")
            replaced = type(loaded)(
                ref=loaded.ref,
                kind="boom",
                source=loaded.source,
                module=loaded.module,
                path=loaded.path,
                onnx_path=loaded.onnx_path,
                submodule=loaded.submodule,
                inputs=loaded.inputs,
                outputs=loaded.outputs,
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.onnx"
                self.assertEqual(export_onnx(replaced, path), "boom")
                self.assertEqual(path.read_bytes(), b"boom")
            self.assertLess(
                exporter_names().index("boom"), exporter_names().index("torch.onnx")
            )
        finally:
            unregister_exporter("boom")

    def test_replace_linear_walks_children(self) -> None:
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(3, 2)
                self.inner = nn.Sequential(nn.Linear(2, 1))

        seen: list[tuple[int, int]] = []

        def _factory(in_features: int, out_features: int, bias: bool):
            del bias
            seen.append((in_features, out_features))
            return nn.Linear(in_features, out_features)

        count = replace_linear(_Block(), _factory)
        self.assertEqual(count, 2)
        self.assertEqual(seen, [(3, 2), (2, 1)])

    def test_cli_fp32_manifest_matches_pytorch(self) -> None:
        import numpy as np
        import torch

        fresh = build_reference_act()
        sample = torch.randn(2, int(REFERENCE_ACT["state_dim"]))
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
                    "--out",
                    str(out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            document = json.loads(text)
            self.assertEqual(document["schema"], SCHEMA_ID)
            self.assertEqual(manifest_problems(document), [])
            schema = json.loads(schema_text())
            self.assertEqual(set(schema["required"]), set(document))
            onnx = out / "model.onnx"
            recorded = next(
                item for item in document["outputs"] if item["role"] == "onnx"
            )
            digest = hashlib.sha256(onnx.read_bytes()).hexdigest()
            self.assertEqual(recorded["sha256"], digest)
            self.assertEqual(recorded["bytes"], onnx.stat().st_size)
            self.assertEqual(document["session_providers"][0], "CPUExecutionProvider")
            self.assertIsNone(document["versions"]["tensorrt"])
            self.assertTrue(document["versions"]["torch"])
            feed = {"state": sample.numpy()}
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(onnx), providers=["CPUExecutionProvider"]
            )
            got = session.run(None, feed)[0]
            with torch.no_grad():
                ref = fresh(sample).numpy()
            self.assertTrue(np.allclose(got, ref, rtol=1e-4, atol=1e-4))

    def test_cli_rejects_an_illegal_combination_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, text, _err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-act",
                    "--target",
                    "jetson-orin-nano",
                    "--precision",
                    "fp8",
                    "--tensorrt",
                    "--out",
                    str(Path(tmp) / "out"),
                    "--json",
                ]
            )
        self.assertEqual(code, 2)
        document = json.loads(text)
        self.assertFalse(document["ok"])
        self.assertIn("FP8", document["error"])

    def _assert_matches(
        self, model_id: str, input_name: str, shape: tuple[int, ...]
    ) -> None:
        import torch

        loaded = load_model(model_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            export_onnx(loaded, path)
            sample = torch.randn(*shape)
            self._compare(loaded.module, path, input_name, sample)

    def _compare(self, module, path: Path, input_name: str, sample) -> None:
        import numpy as np
        import onnxruntime as ort
        import torch

        module.eval()
        with torch.no_grad():
            ref = module(sample).detach().cpu().numpy()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(None, {input_name: sample.detach().cpu().numpy()})[0]
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(
            np.allclose(got, ref, rtol=1e-4, atol=1e-4),
            f"max abs {np.max(np.abs(got - ref))}",
        )


class TestOptimizeInt8Wiring(unittest.TestCase):
    def test_engine_attaches_the_calibrator_for_a_real_onnx_graph(self) -> None:
        trt = FakeTensorRT(input_shape=(-1, int(REFERENCE_ACT["state_dim"])))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root / "calib", (int(REFERENCE_ACT["state_dim"]),), 4)
            cache = root / "cache.bin"
            document = optimize(
                _request(
                    target="x86-cuda",
                    precision="int8",
                    tensorrt=True,
                    calib_data=str(root / "calib"),
                    calib_batch=2,
                    calib_cache=str(cache),
                    max_batch=2,
                    dynamic_batch=True,
                    out_dir=str(root / "out"),
                ),
                backends=OptimizeBackends(trt_module=trt, cuda=False),
                no_cache=True,
            )
            self.assertEqual(trt.last_config.int8_calibrator.get_batch_size(), 2)
            self.assertIn("int8", trt.last_config.flags)
            self.assertEqual(cache.read_bytes(), b"cache-bytes")
            roles = {item["role"] for item in document["outputs"]}
            self.assertEqual(roles, {"onnx", "engine"})
            self.assertEqual(
                document["execution_provider"], "TensorrtExecutionProvider"
            )
            self.assertEqual(document["flags"]["calib_batch"], 2)
            self.assertEqual(document["flags"]["quantization"], "implicit")
            self.assertIsNone(document["flags"]["calib_max_images"])
            self.assertEqual(document["session_providers"], [])


@unittest.skipUnless(_has_export_stack(), "optimize extra (torch, onnx, onnxruntime)")
class TestCpuExportNeedsStack(TestCpuExport):
    """The numerical cases above are skipped when the extra is absent.

    The methods live on TestCpuExport so a machine without the extra still
    collects the class, and this subclass applies the skip.
    """


# Rebind: the methods were defined on TestCpuExport without a skip, which would
# fail on a bare install. Move them onto the skipped subclass and drop the
# unskipped class from the loader's run by clearing its tests.
def _move_export_tests() -> None:
    for name in list(TestCpuExport.__dict__):
        if name.startswith("test_"):
            setattr(TestCpuExportNeedsStack, name, getattr(TestCpuExport, name))
            delattr(TestCpuExport, name)


_move_export_tests()


@unittest.skipUnless(_has_export_stack(), "optimize extra (torch, onnx, onnxruntime)")
class TestOptimizeInt8WiringNeedsStack(TestOptimizeInt8Wiring):
    pass


def _move_int8_test() -> None:
    for name in list(TestOptimizeInt8Wiring.__dict__):
        if name.startswith("test_"):
            setattr(
                TestOptimizeInt8WiringNeedsStack,
                name,
                getattr(TestOptimizeInt8Wiring, name),
            )
            delattr(TestOptimizeInt8Wiring, name)


_move_int8_test()


class TestLiveTensorRT(unittest.TestCase):
    @unittest.skipUnless(
        _live_tensorrt(),
        "GPU follow-up: TensorRT engine build needs TensorRT and a CUDA device",
    )
    def test_live_fp16_engine(self) -> None:
        """gpu marker: skipped unless TensorRT and CUDA are both present."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code, text, err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-mlp",
                    "--target",
                    "x86-cuda",
                    "--precision",
                    "fp16",
                    "--tensorrt",
                    "--out",
                    str(out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            self.assertTrue((out / "model.engine").is_file())

    @unittest.skipUnless(
        _live_tensorrt(),
        "GPU follow-up: SmolVLA vision-encoder INT8 cache reuse was not run "
        "(needs TensorRT and a CUDA device)",
    )
    def test_smolvla_vision_encoder_int8_cache_reuse(self) -> None:
        """GPU follow-up. Not run in CPU CI.

        Builds the SmolVLA vision encoder at INT8, writes a calibration cache,
        then builds again and reuses that cache.
        """
        checkpoint = os.environ.get("OHHO_SMOLVLA_CHECKPOINT", "").strip()
        if not checkpoint or not Path(checkpoint).exists():
            self.skipTest("Set OHHO_SMOLVLA_CHECKPOINT to a local SmolVLA checkpoint.")
        if not _has_export_stack():
            self.skipTest("optimize extra is not installed")
        loaded = load_model(checkpoint, submodule="vision")
        shape_text = os.environ.get("OHHO_SMOLVLA_SHAPE", "").strip()
        if shape_text:
            shape = tuple(int(part) for part in shape_text.split(",") if part)
        else:
            spec = describe_export(loaded)
            shape = tuple(spec.inputs[0].shape[1:])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib = root / "calib"
            _write_bins(calib, shape, 2)
            cache = root / "vision.cache"
            common = [
                "optimize",
                "--model",
                checkpoint,
                "--submodule",
                "vision",
                "--target",
                "x86-cuda",
                "--precision",
                "int8",
                "--tensorrt",
                "--calib-data",
                str(calib),
                "--calib-batch",
                "1",
                "--calib-max-images",
                "2",
                "--calib-cache",
                str(cache),
            ]
            with self.assertLogs("ohho.optimize", level="INFO") as first:
                code, text, err = _run([*common, "--out", str(root / "out1"), "--json"])
            self.assertEqual(code, 0, err + text)
            self.assertTrue(cache.is_file())
            self.assertTrue(cache_sidecar_path(cache).is_file())
            self.assertTrue(
                any("Wrote INT8 calibration cache" in line for line in first.output)
            )
            with self.assertLogs("ohho.optimize", level="INFO") as second:
                code, text, err = _run([*common, "--out", str(root / "out2"), "--json"])
            self.assertEqual(code, 0, err + text)
            self.assertTrue(
                any("Reusing INT8 calibration cache" in line for line in second.output)
            )

    @unittest.skipUnless(
        _live_tensorrt(),
        "GPU follow-up: INT8 parity gate was not run "
        "(needs TensorRT and a CUDA device)",
    )
    def test_int8_parity_gate(self) -> None:
        """GPU follow-up. Not run in CPU CI.

        Builds reference-mlp INT8 and calls ``ohho.parity`` when that module
        is present (OHH-430). This file does not define the parity gate.
        """
        if not _has_export_stack():
            self.skipTest("optimize extra is not installed")
        if importlib.util.find_spec("ohho.parity") is None:
            self.skipTest("ohho.parity is not available")
        import ohho.parity as parity

        gate = None
        for name in ("check_parity", "parity_gate", "run_parity_gate", "gate"):
            candidate = getattr(parity, name, None)
            if callable(candidate):
                gate = candidate
                break
        if gate is None:
            self.fail("ohho.parity does not expose a parity gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bins(root / "calib", (int(REFERENCE_MLP["obs_dim"]),), 2)
            out = root / "out"
            code, text, err = _run(
                [
                    "optimize",
                    "--model",
                    "reference-mlp",
                    "--target",
                    "x86-cuda",
                    "--precision",
                    "int8",
                    "--tensorrt",
                    "--calib-data",
                    str(root / "calib"),
                    "--calib-batch",
                    "1",
                    "--calib-max-images",
                    "2",
                    "--calib-cache",
                    str(root / "int8.cache"),
                    "--out",
                    str(out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, err + text)
            result = gate(
                model="reference-mlp",
                onnx=str(out / "model.onnx"),
                engine=str(out / "model.engine"),
                precision="int8",
            )
        self.assertTrue(bool(getattr(result, "ok", result)))


if __name__ == "__main__":
    unittest.main()
