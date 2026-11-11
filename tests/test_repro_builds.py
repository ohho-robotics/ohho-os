"""Reproducible optimize builds (OHH-469).

CPU tests inject the toolchain and a TensorRT stand-in. They do not import
torch, onnxruntime, tensorrt, or git, and they do not require a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.optimize.cache import cache_key
from ohho.optimize.engine import OptimizeBackends, optimize
from ohho.optimize.errors import OptimizeError
from ohho.optimize.manifest import SCHEMA_ID, manifest_problems
from ohho.optimize.plan import OptimizeRequest
from ohho.optimize.toolchain import (
    HostGpu,
    Toolchain,
    detect_driver,
    detect_jetpack,
    jetpack_from_tegra_release,
    require_engine_host,
)


def _load_optimize_helpers():
    """Load the TensorRT stand-in from the sibling test module.

    ``discover -s tests`` does not make this directory a package, and a
    site-packages ``tests`` package would shadow it.
    """
    path = Path(__file__).with_name("test_optimize.py")
    spec = importlib.util.spec_from_file_location("ohho_test_optimize_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_optimize_helpers()
FakeTensorRT = _HELPERS.FakeTensorRT
_write_bins = _HELPERS._write_bins

_TEGRA = (
    "# R36 (release), REVISION: 3.0, GCID: 1, BOARD: generic, "
    "EABI: aarch64, DATE: Mon Jan 1 00:00:00 UTC 2024\n"
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
        "target": "x86-cuda",
        "precision": "fp16",
        "out_dir": "out",
        "tensorrt": True,
    }
    base.update(kwargs)
    return OptimizeRequest(**base)


def _toolchain(tensorrt: str = "8.6.1") -> Toolchain:
    return Toolchain(
        torch=None,
        onnx=None,
        onnxruntime=None,
        tensorrt=tensorrt,
        cuda="12.4",
        driver="550.54.14",
        jetpack=None,
    )


def _host(architecture: str = "8.9") -> HostGpu:
    return HostGpu("test-gpu", architecture)


def _backends(
    trt: FakeTensorRT | None = None,
    *,
    tensorrt: str = "8.6.1",
    architecture: str = "8.9",
) -> OptimizeBackends:
    return OptimizeBackends(
        trt_module=trt if trt is not None else FakeTensorRT(),
        cuda=False,
        toolchain=_toolchain(tensorrt),
        host=_host(architecture),
    )


class _Registry:
    """In-memory stand-in for the console registry hook."""

    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    def find(self, key: str) -> str | None:
        return self.items.get(key)

    def publish(self, key: str, directory: str) -> None:
        self.items[key] = directory


class TestCacheKeyAndProbes(unittest.TestCase):
    def test_key_changes_when_a_flag_or_tensorrt_version_changes(self) -> None:
        target = {"id": "x86-cuda", "arch": "x86_64", "accelerator": "dgpu"}
        flags = {"max_batch": 1, "dynamic_batch": False, "tensorrt": True}
        versions = _toolchain().as_dict()
        first = cache_key(
            model_sha256="a" * 64,
            target=target,
            precision="FP16",
            flags=flags,
            versions=versions,
        )
        again = cache_key(
            model_sha256="a" * 64,
            target=target,
            precision="FP16",
            flags=dict(flags),
            versions=dict(versions),
        )
        self.assertEqual(first, again)
        changed_flag = cache_key(
            model_sha256="a" * 64,
            target=target,
            precision="FP16",
            flags={**flags, "max_batch": 2},
            versions=versions,
        )
        changed_trt = cache_key(
            model_sha256="a" * 64,
            target=target,
            precision="FP16",
            flags=flags,
            versions={**versions, "tensorrt": "8.6.2"},
        )
        self.assertNotEqual(first, changed_flag)
        self.assertNotEqual(first, changed_trt)

    def test_jetpack_release_and_missing_file(self) -> None:
        self.assertEqual(jetpack_from_tegra_release(_TEGRA), "R36.3.0")
        self.assertIsNone(jetpack_from_tegra_release("no release here"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nv_tegra_release"
            path.write_text(_TEGRA, encoding="utf-8")
            self.assertEqual(detect_jetpack(path), "R36.3.0")
            self.assertIsNone(detect_jetpack(Path(tmp) / "missing"))

    def test_driver_query_is_trimmed(self) -> None:
        self.assertEqual(detect_driver(query=lambda: " 550.54.14\n"), "550.54.14")
        self.assertIsNone(detect_driver(query=lambda: "  "))
        self.assertIsNone(detect_driver(query=lambda: None))


class TestReproBuilds(unittest.TestCase):
    def test_identical_inputs_hit_and_a_flag_or_version_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            cache = root / "cache"
            calls: list[str] = []
            loads: list[str] = []

            def _spy(*args, **kwargs):
                from ohho.optimize.tensorrt import build_engine

                calls.append(str(kwargs.get("precision")))
                return build_engine(*args, **kwargs)

            def _load(*args, **kwargs):
                from ohho.optimize.models import load_model

                loads.append(args[0])
                return load_model(*args, **kwargs)

            def _build(out_name: str, **kwargs):
                trt_version = kwargs.pop("tensorrt_version", "8.6.1")
                return optimize(
                    _request(model=str(model), out_dir=str(root / out_name), **kwargs),
                    backends=_backends(tensorrt=trt_version),
                    cache_dir=cache,
                )

            with (
                patch("ohho.optimize.engine.build_engine", _spy),
                patch("ohho.optimize.engine.load_model", _load),
            ):
                first = _build("out1")
                second = _build("out2")
                flagged = _build("out3", max_batch=2)
                versioned = _build("out4", tensorrt_version="8.6.2")
                model.write_bytes(b"different-onnx-bytes")
                changed_model = _build("out5")
                stored_text = (
                    cache / first["cache"]["key"] / "manifest.json"
                ).read_text(encoding="utf-8")

        self.assertEqual(calls, ["FP16", "FP16", "FP16", "FP16"])
        self.assertEqual(len(loads), 4)
        self.assertFalse(first["cache"]["hit"])
        self.assertEqual(first["cache"]["source"], "miss")
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["cache"]["source"], "local")
        self.assertEqual(second["cache"]["key"], first["cache"]["key"])
        self.assertEqual(second["build_duration_s"], first["build_duration_s"])
        self.assertGreaterEqual(first["build_duration_s"], 0)
        self.assertEqual(
            first["model_sha256"], hashlib.sha256(b"fake-onnx-bytes").hexdigest()
        )
        self.assertIsNone(first["calibration_cache_sha256"])
        self.assertEqual(first["precision"], "FP16")
        self.assertEqual(first["target"]["id"], "x86-cuda")
        self.assertEqual(first["versions"]["tensorrt"], "8.6.1")
        self.assertEqual(first["versions"]["cuda"], "12.4")
        self.assertEqual(first["versions"]["driver"], "550.54.14")
        self.assertIsNone(first["versions"]["jetpack"])
        self.assertEqual(first["host"]["gpu_name"], "test-gpu")
        self.assertEqual(first["host"]["gpu_architecture"], "8.9")
        self.assertEqual(first["schema"], SCHEMA_ID)
        self.assertEqual(manifest_problems(first), [])
        self.assertEqual(manifest_problems(second), [])
        self.assertFalse(flagged["cache"]["hit"])
        self.assertNotEqual(flagged["cache"]["key"], first["cache"]["key"])
        self.assertEqual(flagged["flags"]["max_batch"], 2)
        self.assertFalse(versioned["cache"]["hit"])
        self.assertEqual(versioned["versions"]["tensorrt"], "8.6.2")
        self.assertNotEqual(versioned["cache"]["key"], first["cache"]["key"])
        self.assertFalse(changed_model["cache"]["hit"])
        self.assertEqual(
            changed_model["model_sha256"],
            hashlib.sha256(b"different-onnx-bytes").hexdigest(),
        )
        stored = json.loads(stored_text)
        self.assertFalse(stored["cache"]["hit"])
        self.assertEqual(stored["cache"]["source"], "miss")

    def test_dynamic_batch_flag_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            calls: list[int] = []

            def _spy(*args, **kwargs):
                from ohho.optimize.tensorrt import build_engine

                calls.append(1)
                return build_engine(*args, **kwargs)

            with patch("ohho.optimize.engine.build_engine", _spy):
                optimize(
                    _request(model=str(model), out_dir=str(root / "a")),
                    backends=_backends(),
                    cache_dir=root / "cache",
                )
                second = optimize(
                    _request(
                        model=str(model),
                        out_dir=str(root / "b"),
                        dynamic_batch=True,
                    ),
                    backends=_backends(),
                    cache_dir=root / "cache",
                )
        self.assertEqual(calls, [1, 1])
        self.assertFalse(second["cache"]["hit"])
        self.assertTrue(second["flags"]["dynamic_batch"])

    def test_no_cache_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            calls: list[int] = []

            def _spy(*args, **kwargs):
                from ohho.optimize.tensorrt import build_engine

                calls.append(1)
                return build_engine(*args, **kwargs)

            with patch("ohho.optimize.engine.build_engine", _spy):
                optimize(
                    _request(model=str(model), out_dir=str(root / "a")),
                    backends=_backends(),
                    cache_dir=root / "cache",
                    no_cache=True,
                )
                second = optimize(
                    _request(model=str(model), out_dir=str(root / "b")),
                    backends=_backends(),
                    cache_dir=root / "cache",
                    no_cache=True,
                )
                third = optimize(
                    _request(model=str(model), out_dir=str(root / "c")),
                    backends=_backends(),
                    cache_dir=root / "cache",
                )
        self.assertEqual(len(calls), 2)
        self.assertFalse(second["cache"]["hit"])
        self.assertTrue(third["cache"]["hit"])

    def test_int8_records_the_calibration_cache_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            _write_bins(root / "calib", (4,), 2)
            cache = root / "int8.cache"
            trt = FakeTensorRT(input_shape=(-1, 4))
            trt.__version__ = "8.6.1"
            document = optimize(
                _request(
                    model=str(model),
                    precision="int8",
                    calib_data=str(root / "calib"),
                    calib_batch=1,
                    calib_cache=str(cache),
                    out_dir=str(root / "out"),
                ),
                backends=_backends(trt),
                cache_dir=root / "builds",
                no_cache=True,
            )
            digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        self.assertEqual(document["calibration_cache_sha256"], digest)
        self.assertEqual(document["flags"]["quantization"], "implicit")
        self.assertEqual(manifest_problems(document), [])

    def test_registry_hit_skips_the_build(self) -> None:
        registry = _Registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            calls: list[int] = []

            def _spy(*args, **kwargs):
                from ohho.optimize.tensorrt import build_engine

                calls.append(1)
                return build_engine(*args, **kwargs)

            with patch("ohho.optimize.engine.build_engine", _spy):
                optimize(
                    _request(model=str(model), out_dir=str(root / "a")),
                    backends=_backends(),
                    cache_dir=root / "local-a",
                    registry=registry,
                )
                second = optimize(
                    _request(model=str(model), out_dir=str(root / "b")),
                    backends=_backends(),
                    cache_dir=root / "local-b",
                    registry=registry,
                )
        self.assertEqual(calls, [1])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["cache"]["source"], "registry")
        self.assertEqual(len(registry.items), 1)

    def test_mismatched_host_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            calls: list[int] = []

            def _spy(*args, **kwargs):
                from ohho.optimize.tensorrt import build_engine

                calls.append(1)
                return build_engine(*args, **kwargs)

            with patch("ohho.optimize.engine.build_engine", _spy):
                first = optimize(
                    _request(model=str(model), out_dir=str(root / "out")),
                    backends=_backends(architecture="8.9"),
                    cache_dir=root / "cache",
                )
                with self.assertRaises(OptimizeError) as caught:
                    optimize(
                        _request(model=str(model), out_dir=str(root / "other")),
                        backends=_backends(architecture="9.0"),
                        cache_dir=root / "cache",
                    )
            self.assertEqual(calls, [1])
            self.assertIn("8.9", str(caught.exception))
            self.assertIn("9.0", str(caught.exception))
            self.assertIn("GPU architecture", str(caught.exception))
            matched = require_engine_host(
                root / "out" / "model.engine",
                toolchain=_toolchain(),
                host=_host("8.9"),
            )
            self.assertEqual(matched["host"]["gpu_architecture"], "8.9")
            with self.assertRaises(OptimizeError) as arch:
                require_engine_host(
                    root / "out",
                    toolchain=_toolchain(),
                    host=_host("9.0"),
                )
            self.assertIn("GPU architecture", str(arch.exception))
            with self.assertRaises(OptimizeError) as trt:
                require_engine_host(
                    root / "out" / "manifest.json",
                    toolchain=_toolchain("9.1.0"),
                    host=_host("8.9"),
                )
            self.assertIn("TensorRT", str(trt.exception))
            self.assertIn("8.6.1", str(trt.exception))
            self.assertIn("9.1.0", str(trt.exception))
            self.assertEqual(first["versions"]["tensorrt"], "8.6.1")

    def test_explain_prints_the_manifest_and_no_cache_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.onnx"
            model.write_bytes(b"fake-onnx-bytes")
            calls: list[int] = []

            def _spy(*args, **kwargs):
                calls.append(1)
                destination = Path(args[1])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"engine-bytes")
                return destination

            fixed = _toolchain()
            host = _host()
            with (
                patch("ohho.optimize.engine.resolve_toolchain", return_value=fixed),
                patch("ohho.optimize.engine.resolve_host", return_value=host),
                patch(
                    "ohho.optimize.engine.probe_tensorrt",
                    return_value=(True, "8.6.1"),
                ),
                patch(
                    "ohho.optimize.engine.cuda_device_available",
                    return_value=True,
                ),
                patch("ohho.optimize.engine.build_engine", _spy),
                patch.dict(os.environ, {"OHHO_OPTIMIZE_CACHE": str(root / "cache")}),
            ):
                common = [
                    "optimize",
                    "--model",
                    str(model),
                    "--target",
                    "x86-cuda",
                    "--precision",
                    "fp16",
                    "--tensorrt",
                ]
                code, text, err = _run([*common, "--out", str(root / "out1"), "--json"])
                self.assertEqual(code, 0, err + text)
                code, text, err = _run([*common, "--out", str(root / "out2"), "--json"])
                self.assertEqual(code, 0, err + text)
                hit = json.loads(text)
                code, explained, err = _run(
                    ["optimize", "--explain", str(root / "out2")]
                )
                self.assertEqual(code, 0, err)
                code, from_engine, err = _run(
                    ["optimize", "--explain", str(root / "out2" / "model.engine")]
                )
                self.assertEqual(code, 0, err)
                code, _text, err = _run(
                    ["optimize", "--explain", str(root / "missing")]
                )
                code_rebuild, _text, err_rebuild = _run(
                    [
                        *common,
                        "--out",
                        str(root / "out3"),
                        "--no-cache",
                        "--json",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("manifest.json", err)
        self.assertEqual(code_rebuild, 0, err_rebuild)
        self.assertEqual(calls, [1, 1])
        self.assertTrue(hit["cache"]["hit"])
        printed = json.loads(explained)
        from_file = json.loads(from_engine)
        self.assertEqual(printed, from_file)
        self.assertEqual(printed["schema"], SCHEMA_ID)
        self.assertEqual(printed["model_sha256"], hit["model_sha256"])
        self.assertEqual(printed["target"]["id"], "x86-cuda")
        self.assertEqual(printed["precision"], "FP16")
        self.assertEqual(printed["versions"]["tensorrt"], "8.6.1")
        self.assertEqual(printed["host"]["gpu_architecture"], "8.9")
        self.assertIn("max_batch", printed["flags"])
        self.assertTrue(printed["cache"]["hit"])
