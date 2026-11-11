"""Mixed-precision recipe field, device check, and run-manifest metrics."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.train.precision import (
    PRECISIONS,
    DeviceCapabilities,
    PrecisionUnsupported,
    fp8_autocast,
    probe_capabilities,
    require_precision,
    smoke_train,
)
from ohho.train.recipe_registry import (
    TrainResult,
    family_metadata,
    register,
    unregister,
)
from ohho.train.recipe_schema import (
    SCHEMA_ID,
    RecipeConfigError,
    apply_overrides,
    example_path,
    load_recipe_file,
    schema_document,
    validate_recipe,
)
from ohho.train.run_cli import EXIT_INVALID, EXIT_OK, execute
from ohho.train.run_manifest import schema_document as run_schema_document
from ohho.train.run_manifest import validate_run_manifest


def _cpu(*, bf16: bool) -> DeviceCapabilities:
    return DeviceCapabilities(
        device="cpu",
        kind="cpu",
        bf16=bf16,
        fp16=False,
        fp8=False,
        available=True,
        name="cpu",
        detail="",
    )


def _gpu(
    *,
    bf16: bool,
    fp16: bool,
    fp8: bool,
    name: str,
    detail: str,
) -> DeviceCapabilities:
    return DeviceCapabilities(
        device="cuda",
        kind="cuda",
        bf16=bf16,
        fp16=fp16,
        fp8=fp8,
        available=True,
        name=name,
        detail=detail,
    )


class _Cuda:
    def __init__(self, cc: tuple[int, int], name: str, bf16: bool = True) -> None:
        self._cc = cc
        self._name = name
        self._bf16 = bf16

    def is_available(self) -> bool:
        return True

    def get_device_capability(self, index: int = 0) -> tuple[int, int]:
        del index
        return self._cc

    def get_device_name(self, index: int = 0) -> str:
        del index
        return self._name

    def is_bf16_supported(self, including_emulation: bool = False) -> bool:
        del including_emulation
        return self._bf16


class _Torch:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda = cuda


def _run(path: Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _recipe(directory: Path, family: str, precision: str, device: str = "cpu") -> Path:
    path = directory / "recipe.yaml"
    path.write_text(
        "\n".join(
            [
                "schema: ohho.recipe/v1",
                f"family: {family}",
                "dataset:",
                "  version_id: dv_precision",
                f"precision: {precision}",
                f"device: {device}",
                "output:",
                "  dir: placeholder",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


class TestPrecisionSchema(unittest.TestCase):
    def test_recipe_and_run_schemas_list_the_same_precisions(self) -> None:
        recipe = schema_document()
        run = run_schema_document()
        self.assertEqual(recipe["properties"]["precision"]["enum"], list(PRECISIONS))
        self.assertEqual(run["properties"]["precision"]["enum"], list(PRECISIONS))
        self.assertEqual(
            recipe["title"],
            SCHEMA_ID,
        )
        loaded = load_recipe_file(example_path("act"))
        for precision in PRECISIONS:
            updated = apply_overrides(loaded, [f"precision={precision}"])
            self.assertEqual(validate_recipe(updated), [], precision)
        rejected = apply_overrides(loaded, ["precision=int8"])
        errors = validate_recipe(rejected)
        self.assertTrue(any("precision" in error for error in errors))


class TestCapabilityCheck(unittest.TestCase):
    def test_cpu_accepts_fp32_and_refuses_the_rest_when_bf16_is_absent(self) -> None:
        caps = _cpu(bf16=False)
        plan = require_precision("fp32", caps)
        self.assertEqual(plan.precision, "fp32")
        self.assertFalse(plan.loss_scaling)
        self.assertEqual(plan.backend, "fp32")
        for precision in ("fp16", "bf16", "fp8"):
            with self.subTest(precision=precision):
                text = require_message(precision, caps)
                self.assertIn(precision, text)
                self.assertIn("not supported", text)
                self.assertIn("No fallback is applied", text)
                self.assertIn("cpu", text)
        self.assertIn("loss scaling", require_message("fp16", caps))
        self.assertIn("Transformer Engine", require_message("fp8", caps))

    def test_cpu_accepts_bf16_when_the_device_reports_it(self) -> None:
        plan = require_precision("bf16", _cpu(bf16=True))
        self.assertEqual(plan.backend, "torch.amp")
        self.assertFalse(plan.loss_scaling)
        with self.assertRaises(PrecisionUnsupported):
            require_precision("fp16", _cpu(bf16=True))

    def test_gpu_without_bf16(self) -> None:
        caps = _gpu(
            bf16=False,
            fp16=True,
            fp8=False,
            name="GTX 1080",
            detail="compute capability 6.1",
        )
        fp16 = require_precision("fp16", caps)
        self.assertTrue(fp16.loss_scaling)
        self.assertEqual(fp16.backend, "torch.amp")
        self.assertEqual(require_precision("fp32", caps).backend, "fp32")
        for precision in ("bf16", "fp8"):
            with self.subTest(precision=precision):
                text = require_message(precision, caps)
                self.assertIn("GTX 1080", text)
                self.assertIn("6.1", text)
                self.assertIn("No fallback is applied", text)

    def test_gpu_without_fp8(self) -> None:
        caps = _gpu(
            bf16=True,
            fp16=True,
            fp8=False,
            name="A10",
            detail="compute capability 8.6",
        )
        bf16 = require_precision("bf16", caps)
        self.assertEqual(bf16.backend, "torch.amp")
        self.assertFalse(bf16.loss_scaling)
        self.assertTrue(require_precision("fp16", caps).loss_scaling)
        text = require_message("fp8", caps)
        self.assertIn("fp8", text)
        self.assertIn("Transformer Engine", text)
        self.assertIn("A10", text)
        self.assertIn("No fallback is applied", text)

    def test_probe_reads_mocked_cuda_capabilities(self) -> None:
        pascal = probe_capabilities(
            "cuda",
            torch_module=_Torch(_Cuda((6, 1), "GTX 1080")),
            fp8_importable=False,
        )
        self.assertFalse(pascal.bf16)
        self.assertTrue(pascal.fp16)
        self.assertFalse(pascal.fp8)
        ampere = probe_capabilities(
            "cuda",
            torch_module=_Torch(_Cuda((8, 6), "A10", bf16=True)),
            fp8_importable=False,
        )
        self.assertTrue(ampere.bf16)
        self.assertFalse(ampere.fp8)
        self.assertIn("8.6", ampere.detail)
        no_package = probe_capabilities(
            "cuda",
            torch_module=_Torch(_Cuda((9, 0), "H100", bf16=True)),
            fp8_importable=False,
        )
        self.assertFalse(no_package.fp8)
        self.assertIn("not installed", no_package.detail)
        hopper = probe_capabilities(
            "cuda",
            torch_module=_Torch(_Cuda((9, 0), "H100", bf16=True)),
            fp8_importable=True,
        )
        self.assertTrue(hopper.fp8)
        plan = require_precision("fp8", hopper)
        self.assertEqual(plan.backend, "transformer_engine")
        self.assertFalse(plan.loss_scaling)


def require_message(precision: str, caps: DeviceCapabilities) -> str:
    try:
        require_precision(precision, caps)
    except RecipeConfigError as exc:
        return str(exc)
    raise AssertionError(f"{precision} was accepted")


class _RefuseFamily:
    id = "prec_plugin"

    def __init__(self) -> None:
        self.ran = False

    def available(self) -> bool:
        return True

    def run(self, recipe, emit):
        del recipe, emit
        self.ran = True
        raise AssertionError("precision check must refuse before run")


class _Fp32Family:
    id = "fp32_only"

    def available(self) -> bool:
        return True

    def metadata(self) -> dict[str, list[str]]:
        return {"precisions": ["fp32"]}

    def run(self, recipe, emit):
        del recipe, emit
        raise AssertionError("family allowlist must refuse before run")


class _RecordingFamily:
    id = "prec_record"

    def available(self) -> bool:
        return True

    def run(self, recipe, emit):
        out = Path(recipe["output"]["dir"])
        out.mkdir(parents=True, exist_ok=True)
        emit(
            {
                "event": "train_start",
                "family": self.id,
                "device": recipe.get("device", "cpu"),
            }
        )
        metrics = out / "metrics.json"
        metrics.write_text(
            json.dumps(
                {
                    "final_loss": 0.125,
                    "precision": recipe["precision"],
                    "throughput_steps_per_s": 12.5,
                    "peak_memory_bytes": 4096,
                }
            ),
            encoding="utf-8",
        )
        return TrainResult(
            str(out), str(metrics), str(recipe.get("device", "cpu")), False
        )


class TestRunnerAndManifest(unittest.TestCase):
    def tearDown(self) -> None:
        unregister("prec_plugin")
        unregister("fp32_only")
        unregister("prec_record")

    def test_runner_refuses_mocked_devices_before_training(self) -> None:
        cases = (
            (
                "bf16",
                _gpu(
                    bf16=False,
                    fp16=True,
                    fp8=False,
                    name="GTX 1080",
                    detail="compute capability 6.1",
                ),
                "bf16",
            ),
            (
                "fp8",
                _gpu(
                    bf16=True,
                    fp16=True,
                    fp8=False,
                    name="A10",
                    detail="compute capability 8.6",
                ),
                "Transformer Engine",
            ),
            ("fp16", _cpu(bf16=True), "loss scaling"),
        )
        family = _RefuseFamily()
        register(family, source="test")
        try:
            for precision, caps, needle in cases:
                with self.subTest(precision=precision):
                    family.ran = False
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        recipe = _recipe(root, "prec_plugin", precision, device="cpu")
                        out = root / "out"
                        with patch(
                            "ohho.train.precision.probe_capabilities",
                            return_value=caps,
                        ):
                            code, events, err = _run(recipe, [f"output.dir={out}"])
                        self.assertEqual(code, EXIT_INVALID, err)
                        self.assertIn(needle, err)
                        self.assertIn("No fallback is applied", err)
                        self.assertFalse(family.ran)
                        self.assertFalse(
                            any(event.get("event") == "train_start" for event in events)
                        )
                        self.assertFalse((out / "run_manifest.json").exists())
        finally:
            unregister("prec_plugin")

    def test_family_metadata_lists_precisions_and_can_narrow_them(self) -> None:
        act = family_metadata("act")
        self.assertEqual(act["precisions"], list(PRECISIONS))
        self.assertTrue(act["available"])
        smolvla = family_metadata("smolvla")
        self.assertEqual(smolvla["precisions"], list(PRECISIONS))
        self.assertFalse(smolvla["available"])
        register(_Fp32Family(), source="test")
        try:
            self.assertEqual(family_metadata("fp32_only")["precisions"], ["fp32"])
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = _recipe(root, "fp32_only", "bf16")
                code, events, err = _run(recipe, [f"output.dir={root / 'out'}"])
            self.assertEqual(code, EXIT_INVALID, err)
            self.assertIn("fp32_only", err)
            self.assertIn("not available", err)
            self.assertFalse(
                any(event.get("event") == "train_start" for event in events)
            )
        finally:
            unregister("fp32_only")

    def test_manifest_records_precision_throughput_and_peak_memory(self) -> None:
        register(_RecordingFamily(), source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = _recipe(root, "prec_record", "fp32")
                out = root / "out"
                code, events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                manifest = json.loads(
                    (out / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(validate_run_manifest(manifest), [])
                self.assertEqual(manifest["precision"], "fp32")
                self.assertIs(manifest["loss_scaling"], False)
                self.assertEqual(manifest["precision_backend"], "fp32")
                self.assertEqual(
                    manifest["precision_metrics"]["fp32"],
                    {"throughput_steps_per_s": 12.5, "peak_memory_bytes": 4096},
                )
                self.assertIsNone(manifest["precision_metrics"]["fp16"])
                self.assertIsNone(manifest["precision_metrics"]["bf16"])
                self.assertIsNone(manifest["precision_metrics"]["fp8"])
                self.assertEqual(events_precision(events), "fp32")
                broken = dict(manifest)
                broken["loss_scaling"] = True
                self.assertTrue(validate_run_manifest(broken))
        finally:
            unregister("prec_record")

    def test_fp16_manifest_records_loss_scaling_on_a_mocked_gpu(self) -> None:
        caps = _gpu(
            bf16=True,
            fp16=True,
            fp8=False,
            name="A10",
            detail="compute capability 8.6",
        )
        register(_RecordingFamily(), source="test")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = _recipe(root, "prec_record", "fp16", device="cuda")
                out = root / "out"
                with (
                    patch(
                        "ohho.train.precision.resolve_training_device",
                        return_value="cuda",
                    ),
                    patch(
                        "ohho.train.precision.probe_capabilities",
                        return_value=caps,
                    ),
                ):
                    code, _events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                manifest = json.loads(
                    (out / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(validate_run_manifest(manifest), [])
                self.assertEqual(manifest["precision"], "fp16")
                self.assertIs(manifest["loss_scaling"], True)
                self.assertEqual(manifest["precision_backend"], "torch.amp")
                self.assertEqual(
                    manifest["precision_metrics"]["fp16"]["throughput_steps_per_s"],
                    12.5,
                )
                self.assertEqual(
                    manifest["precision_metrics"]["fp16"]["peak_memory_bytes"],
                    4096,
                )
                self.assertIsNone(manifest["precision_metrics"]["fp32"])
        finally:
            unregister("prec_record")


def events_precision(events: list[dict]) -> str:
    for event in events:
        if event.get("event") == "train_start":
            return str(event.get("precision"))
    raise AssertionError("no train_start")


class TestFp8Import(unittest.TestCase):
    def test_fp8_autocast_names_transformer_engine_when_it_is_missing(self) -> None:
        if importlib.util.find_spec("transformer_engine") is not None:
            self.skipTest("transformer_engine is installed")
        with self.assertRaises(PrecisionUnsupported) as caught:
            with fp8_autocast():
                pass
        self.assertIn("Transformer Engine", str(caught.exception))
        self.assertIn("No fallback is applied", str(caught.exception))


class TestCpuSmoke(unittest.TestCase):
    def test_fp32_cpu_smoke(self) -> None:
        torch = _torch_module()
        if torch is None:
            self.skipTest("torch is not installed")
        stats = smoke_train("fp32", device="cpu", steps=2)
        self.assertEqual(stats["precision"], "fp32")
        self.assertFalse(stats["loss_scaling"])
        self.assertEqual(stats["backend"], "fp32")
        self.assertGreater(stats["throughput_steps_per_s"], 0)
        self.assertIsInstance(stats["peak_memory_bytes"], int)
        self.assertGreaterEqual(stats["peak_memory_bytes"], 0)
        self.assertIsInstance(stats["final_loss"], float)

    def test_bf16_cpu_smoke(self) -> None:
        torch = _torch_module()
        if torch is None:
            self.skipTest("torch is not installed")
        caps = probe_capabilities("cpu", torch_module=torch, try_bf16=True)
        if not caps.bf16:
            self.skipTest("CPU bfloat16 autocast is not supported")
        stats = smoke_train("bf16", device="cpu", steps=2)
        self.assertEqual(stats["precision"], "bf16")
        self.assertFalse(stats["loss_scaling"])
        self.assertEqual(stats["backend"], "torch.amp")
        self.assertGreater(stats["throughput_steps_per_s"], 0)
        self.assertIsInstance(stats["peak_memory_bytes"], int)


def _torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


if __name__ == "__main__":
    unittest.main()
