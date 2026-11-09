"""Hardware target profiles (OHH-427)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.resources import files
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ohho.cli import build_parser, main
from ohho.targets import (
    BUILTIN_IDS,
    MODEL_CLASSES,
    SCHEMA_ID,
    MachineProbe,
    catalog_document,
    check_combination,
    detect_target,
    get_profile,
    list_profiles,
    load_profile,
    probe_machine,
)
from ohho.targets.detect import (
    ProbeSources,
    _default_apple,
    _default_jetson,
    _default_model,
    _default_nvidia_smi,
    _default_pi,
    _default_torch,
    normalize_arch,
)
from ohho.targets.errors import TargetCombinationError, TargetProfileError
from ohho.targets.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    TRT_PROVIDER,
    onnx_providers,
    provider_preference,
    workstation_provider_order,
)
from ohho.targets.schema import (
    ACCELERATORS,
    ARCHES,
    PRECISIONS,
    profile_from_dict,
    profile_from_json,
    profile_from_yaml,
    profile_to_dict,
    schema_text,
)
from ohho.train import SUPPORTED_POLICIES


def _slos(value: int = 10) -> dict[str, int]:
    return {name: value for name in MODEL_CLASSES}


def _cpu_doc(**updates: object) -> dict:
    document = {
        "schema": SCHEMA_ID,
        "id": "custom-cpu",
        "arch": "x86_64",
        "accelerator": "none",
        "precisions": ["FP32"],
        "execution_providers": [CPU_PROVIDER],
        "tensorrt": None,
        "memory_budget_mib": 1024,
        "workspace_size_mib": 0,
        "slos_p95_ms": _slos(),
        "summary": "Custom CPU target.",
    }
    document.update(updates)
    return document


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class TargetCatalogTest(unittest.TestCase):
    def test_builtin_ids_and_package_files(self) -> None:
        self.assertEqual(list_profiles(), list(BUILTIN_IDS))
        root = files("ohho.targets").joinpath("profiles")
        names = sorted(
            item.name for item in root.iterdir() if str(item.name).endswith(".json")
        )
        self.assertEqual(
            names, sorted(f"{profile_id}.json" for profile_id in BUILTIN_IDS)
        )

    def test_pyproject_ships_the_json(self) -> None:
        text = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pyproject.toml")
            .read_text(encoding="utf-8")
        )
        self.assertIn("targets/profiles/*.json", text)
        self.assertIn("targets/*.json", text)

    def test_each_file_round_trips(self) -> None:
        root = files("ohho.targets").joinpath("profiles")
        for profile_id in BUILTIN_IDS:
            with self.subTest(profile_id=profile_id):
                path = root.joinpath(f"{profile_id}.json")
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                profile = get_profile(profile_id)
                self.assertEqual(raw, profile_to_dict(profile))
                self.assertEqual(profile_from_dict(raw), profile)
                self.assertEqual(profile.schema, SCHEMA_ID)
                self.assertIn("FP32", profile.precisions)
                self.assertEqual(
                    tuple(profile.slos_p95_ms),
                    MODEL_CLASSES,
                )
                for name in MODEL_CLASSES:
                    self.assertGreater(profile.slos_p95_ms[name], 0)

    def test_model_classes_match_training_policies(self) -> None:
        self.assertEqual(set(MODEL_CLASSES), set(SUPPORTED_POLICIES))

    def test_provider_order_matches_the_ported_helper(self) -> None:
        orin = provider_preference("orin_gpu")
        cloud = provider_preference("dgpu", prefer_tensorrt=True)
        cpu = provider_preference("none")
        for profile_id in ("jetson-orin-nano", "jetson-orin-nx", "jetson-agx-orin"):
            self.assertEqual(get_profile(profile_id).execution_providers, orin)
        for profile_id in ("cloud-l4", "cloud-a10g", "cloud-h100"):
            self.assertEqual(get_profile(profile_id).execution_providers, cloud)
        for profile_id in ("rpi5-cpu", "x86-cpu"):
            self.assertEqual(get_profile(profile_id).execution_providers, cpu)
        self.assertEqual(
            get_profile("x86-cuda").execution_providers,
            workstation_provider_order(),
        )

    def test_precision_and_budget_shape(self) -> None:
        for profile_id in ("jetson-orin-nano", "jetson-orin-nx", "jetson-agx-orin"):
            profile = get_profile(profile_id)
            self.assertEqual(profile.arch, "aarch64")
            self.assertEqual(profile.accelerator, "orin_gpu")
            self.assertNotIn("FP8", profile.precisions)
            self.assertNotIn("INT4", profile.precisions)
            self.assertIsNotNone(profile.tensorrt)
        for profile_id in ("rpi5-cpu", "x86-cpu"):
            profile = get_profile(profile_id)
            self.assertEqual(profile.accelerator, "none")
            self.assertIsNone(profile.tensorrt)
            self.assertEqual(profile.workspace_size_mib, 0)
            self.assertEqual(profile.execution_providers, (CPU_PROVIDER,))
        self.assertIn("FP8", get_profile("cloud-l4").precisions)
        self.assertIn("FP8", get_profile("cloud-h100").precisions)
        self.assertNotIn("FP8", get_profile("cloud-a10g").precisions)
        self.assertNotIn("FP8", get_profile("x86-cuda").precisions)
        self.assertEqual(get_profile("rpi5-cpu").arch, "aarch64")
        self.assertEqual(get_profile("x86-cpu").arch, "x86_64")
        self.assertGreater(
            get_profile("cloud-h100").memory_budget_mib,
            get_profile("cloud-l4").memory_budget_mib,
        )
        self.assertGreater(
            get_profile("cloud-l4").memory_budget_mib,
            get_profile("jetson-orin-nano").memory_budget_mib,
        )
        self.assertLess(
            get_profile("cloud-h100").slos_p95_ms["act"],
            get_profile("jetson-orin-nano").slos_p95_ms["act"],
        )
        self.assertLess(
            get_profile("cloud-h100").slos_p95_ms["openvla"],
            get_profile("rpi5-cpu").slos_p95_ms["openvla"],
        )

    def test_tensorrt_windows(self) -> None:
        nano = get_profile("jetson-orin-nano")
        assert nano.tensorrt is not None
        self.assertTrue(nano.tensorrt.allows("8.5.2"))
        self.assertTrue(nano.tensorrt.allows("10.16.2"))
        self.assertFalse(nano.tensorrt.allows("8.5.1"))
        self.assertFalse(nano.tensorrt.allows("11.0.0"))
        self.assertFalse(nano.tensorrt.allows("11.4.1"))
        agx = get_profile("jetson-agx-orin")
        assert agx.tensorrt is not None
        self.assertTrue(agx.tensorrt.allows("10.16.2"))
        self.assertFalse(agx.tensorrt.allows("11.0.0"))
        self.assertFalse(agx.tensorrt.allows("11.2.0"))
        self.assertFalse(agx.tensorrt.allows("11.4.0"))
        self.assertTrue(agx.tensorrt.allows("11.4.1"))
        self.assertFalse(agx.tensorrt.allows("12.0.0"))
        h100 = get_profile("cloud-h100")
        assert h100.tensorrt is not None
        self.assertFalse(h100.tensorrt.allows("8.5.2"))
        self.assertTrue(h100.tensorrt.allows("8.6.0"))
        self.assertTrue(h100.tensorrt.allows("11.4.1"))

    def test_unknown_profile(self) -> None:
        with self.assertRaises(TargetProfileError) as caught:
            get_profile("not-a-target")
        self.assertIn("Unknown target profile", str(caught.exception))
        self.assertIn("rpi5-cpu", str(caught.exception))

    def test_schema_document_matches_the_python_enums(self) -> None:
        document = json.loads(schema_text())
        properties = document["properties"]
        self.assertEqual(document["properties"]["schema"]["const"], SCHEMA_ID)
        self.assertEqual(properties["arch"]["enum"], list(ARCHES))
        self.assertEqual(properties["accelerator"]["enum"], list(ACCELERATORS))
        self.assertEqual(properties["precisions"]["items"]["enum"], list(PRECISIONS))
        self.assertEqual(
            properties["execution_providers"]["items"]["enum"],
            [TRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER],
        )
        self.assertEqual(
            properties["slos_p95_ms"]["required"],
            list(MODEL_CLASSES),
        )
        self.assertEqual(
            set(document["required"]), set(profile_to_dict(get_profile("x86-cpu")))
        )

    def test_catalog_document(self) -> None:
        catalog = catalog_document()
        self.assertEqual(catalog["schema"], SCHEMA_ID)
        self.assertEqual(
            [item["id"] for item in catalog["profiles"]],
            list(BUILTIN_IDS),
        )


class TargetRulesTest(unittest.TestCase):
    def test_int8_tensorrt_on_cpu_targets_is_rejected(self) -> None:
        for profile_id in ("rpi5-cpu", "x86-cpu"):
            with self.subTest(profile_id=profile_id):
                with self.assertRaises(TargetCombinationError) as caught:
                    check_combination(profile_id, "INT8", "tensorrt")
                message = str(caught.exception)
                self.assertIn("INT8 TensorRT", message)
                self.assertIn(profile_id, message)
                self.assertIn("no TensorRT", message)
        accepted = check_combination("rpi5-cpu", "int8", "cpu")
        self.assertEqual(accepted, ("INT8", CPU_PROVIDER))

    def test_fp8_on_orin_is_rejected(self) -> None:
        for profile_id in ("jetson-orin-nano", "jetson-orin-nx", "jetson-agx-orin"):
            with self.subTest(profile_id=profile_id):
                with self.assertRaises(TargetCombinationError) as caught:
                    check_combination(profile_id, "fp8", "TensorRT")
                message = str(caught.exception)
                self.assertIn("FP8 is not valid", message)
                self.assertIn(profile_id, message)
                self.assertIn("Ada or Hopper", message)

    def test_fp8_on_ampere_discrete_gpus_is_rejected(self) -> None:
        for profile_id in ("cloud-a10g", "x86-cuda"):
            with self.subTest(profile_id=profile_id):
                with self.assertRaises(TargetCombinationError) as caught:
                    check_combination(profile_id, "FP8", "cuda")
                self.assertIn("FP8 is not valid", str(caught.exception))
                self.assertIn(profile_id, str(caught.exception))

    def test_supported_combinations_pass(self) -> None:
        self.assertEqual(
            check_combination("jetson-orin-nano", "FP16", "trt"),
            ("FP16", TRT_PROVIDER),
        )
        self.assertEqual(
            check_combination(
                "jetson-orin-nano",
                "FP16",
                "tensorrt",
                tensorrt_version="10.16.2",
            ),
            ("FP16", TRT_PROVIDER),
        )
        self.assertEqual(
            check_combination(
                "jetson-agx-orin",
                "INT8",
                "tensorrt",
                tensorrt_version="11.4.1",
            ),
            ("INT8", TRT_PROVIDER),
        )
        self.assertEqual(
            check_combination(
                "cloud-h100",
                "FP8",
                "tensorrt",
                tensorrt_version="11.4.1",
            ),
            ("FP8", TRT_PROVIDER),
        )
        self.assertEqual(
            check_combination("cloud-l4", "FP8", "cuda"),
            ("FP8", CUDA_PROVIDER),
        )
        self.assertEqual(
            check_combination("x86-cuda", "INT8", "tensorrt"),
            ("INT8", TRT_PROVIDER),
        )

    def test_version_outside_the_window_is_rejected(self) -> None:
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination(
                "jetson-agx-orin",
                "FP16",
                "trt",
                tensorrt_version="11.2.0",
            )
        self.assertIn("excluding", str(caught.exception))
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination(
                "cloud-l4",
                "FP16",
                "tensorrt",
                tensorrt_version="8.5.2",
            )
        self.assertIn("8.6.0", str(caught.exception))
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination(
                "cloud-h100",
                "FP8",
                "tensorrt",
                tensorrt_version="latest",
            )
        self.assertIn("dotted triple", str(caught.exception))
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination(
                "x86-cpu",
                "FP32",
                "cpu",
                tensorrt_version="10.16.2",
            )
        self.assertIn(
            "only when the execution provider is TensorRT", str(caught.exception)
        )

    def test_unknown_precision_and_provider(self) -> None:
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination("x86-cpu", "BF16", "cpu")
        self.assertIn("Unknown precision", str(caught.exception))
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination("x86-cpu", "FP32", "openvino")
        self.assertIn("Unknown execution provider", str(caught.exception))

    def test_int4_is_not_on_the_built_in_profiles(self) -> None:
        with self.assertRaises(TargetCombinationError) as caught:
            check_combination("cloud-h100", "INT4", "tensorrt")
        self.assertIn("INT4 is not valid", str(caught.exception))
        self.assertIn("cloud-h100", str(caught.exception))

    def test_profile_document_rejects_invalid_shapes(self) -> None:
        orin = _cpu_doc(
            id="custom-orin",
            arch="aarch64",
            accelerator="orin_gpu",
            precisions=["FP32", "FP16", "INT8", "FP8"],
            execution_providers=list(provider_preference("orin_gpu")),
            tensorrt={
                "min_version": "8.5.2",
                "before_version": "11.0.0",
                "excluded": [],
            },
            workspace_size_mib=256,
            summary="Custom Orin target.",
        )
        with self.assertRaises(TargetProfileError) as caught:
            profile_from_dict(orin)
        self.assertIn("FP8 is not valid", str(caught.exception))
        self.assertIn("custom-orin", str(caught.exception))

        with self.assertRaises(TargetProfileError) as caught:
            profile_from_dict(
                _cpu_doc(
                    precisions=["FP32", "FP8"],
                    summary="CPU profile claiming FP8.",
                )
            )
        self.assertIn("FP8", str(caught.exception))

        with self.assertRaises(TargetProfileError):
            profile_from_dict(_cpu_doc(memory_budget_mib=0))
        with self.assertRaises(TargetProfileError):
            profile_from_dict(_cpu_doc(workspace_size_mib=32))
        with self.assertRaises(TargetProfileError):
            profile_from_dict(_cpu_doc(arch="riscv64"))
        with self.assertRaises(TargetProfileError):
            profile_from_dict({**_cpu_doc(), "extra": True})

        npu = profile_from_dict(
            _cpu_doc(
                id="custom-npu",
                arch="aarch64",
                accelerator="npu",
                precisions=["FP32", "INT8"],
                summary="NPU placeholder.",
            )
        )
        self.assertEqual(npu.accelerator, "npu")
        self.assertIsNone(npu.tensorrt)
        with self.assertRaises(TargetProfileError):
            profile_from_dict(
                _cpu_doc(
                    id="custom-npu",
                    arch="aarch64",
                    accelerator="npu",
                    execution_providers=[CUDA_PROVIDER, CPU_PROVIDER],
                    summary="NPU placeholder with CUDA.",
                )
            )

    def test_json_and_yaml_loaders(self) -> None:
        with self.assertRaises(TargetProfileError) as caught:
            profile_from_json("{")
        self.assertIn("not valid", str(caught.exception))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom-cpu.json"
            path.write_text(json.dumps(_cpu_doc()), encoding="utf-8")
            self.assertEqual(load_profile(path).id, "custom-cpu")
            other = Path(directory) / "notes.txt"
            other.write_text("{}", encoding="utf-8")
            with self.assertRaises(TargetProfileError):
                load_profile(other)
        yaml_text = "\n".join(
            [
                "schema: ohho.target_profile.v1",
                "id: example-npu",
                "arch: aarch64",
                "accelerator: npu",
                "precisions: [FP32, INT8]",
                "execution_providers: [CPUExecutionProvider]",
                "tensorrt: null",
                "memory_budget_mib: 1024",
                "workspace_size_mib: 0",
                "slos_p95_ms: {act: 10, smolvla: 20, diffusion: 15, openvla: 40}",
                "summary: NPU placeholder for a future compiler target.",
            ]
        )
        try:
            import yaml  # noqa: F401
        except ImportError:
            with self.assertRaises(TargetProfileError) as caught:
                profile_from_yaml(yaml_text)
            self.assertIn("ohho-os[yaml]", str(caught.exception))
        else:
            loaded = profile_from_yaml(yaml_text)
            self.assertEqual(loaded.id, "example-npu")
            self.assertEqual(loaded.accelerator, "npu")


class ProviderOrderTest(unittest.TestCase):
    def test_preference_and_filter(self) -> None:
        self.assertEqual(
            provider_preference("dgpu", prefer_tensorrt=True),
            (TRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER),
        )
        self.assertEqual(
            provider_preference("orin_gpu", prefer_tensorrt=True),
            provider_preference("orin_gpu"),
        )
        self.assertEqual(
            provider_preference("none", prefer_tensorrt=True),
            (CPU_PROVIDER,),
        )
        self.assertEqual(provider_preference("npu"), (CPU_PROVIDER,))
        with self.assertRaises(TargetProfileError):
            provider_preference("tpu")
        self.assertEqual(onnx_providers("orin_gpu"), [CPU_PROVIDER])
        self.assertEqual(
            onnx_providers(
                "orin_gpu",
                available=[CPU_PROVIDER, CUDA_PROVIDER],
            ),
            [CUDA_PROVIDER, CPU_PROVIDER],
        )
        self.assertEqual(onnx_providers("dgpu", available=[]), [CPU_PROVIDER])
        with patch(
            "ohho.targets.providers.installed_execution_providers",
            return_value=None,
        ):
            self.assertEqual(onnx_providers("dgpu", probe=True), [CPU_PROVIDER])
        with patch(
            "ohho.targets.providers.installed_execution_providers",
            return_value=(CPU_PROVIDER,),
        ):
            self.assertEqual(
                onnx_providers("dgpu", prefer_tensorrt=True, probe=True),
                [CPU_PROVIDER],
            )


class DetectTargetTest(unittest.TestCase):
    def test_cpu_only_probe_selects_x86_cpu_and_reports_gaps(self) -> None:
        found = detect_target(MachineProbe())
        self.assertEqual(found.profile_id, "x86-cpu")
        self.assertEqual(found.arch, "x86_64")
        self.assertEqual(found.cuda.status, "missing")
        self.assertEqual(found.tensorrt.status, "missing")
        text = found.text()
        self.assertIn("CUDA is missing", text)
        self.assertIn("TensorRT is missing", text)
        self.assertIn("detected: x86-cpu", text)
        payload = found.to_dict()
        self.assertEqual(payload["cuda"]["status"], "missing")
        self.assertEqual(payload["tensorrt"]["status"], "missing")
        self.assertEqual(payload["profile"]["id"], "x86-cpu")
        self.assertIsNone(payload["nvidia_gpu"])

    def test_board_and_gpu_selection(self) -> None:
        nano = detect_target(
            MachineProbe(
                system="Linux",
                machine="aarch64",
                arch="aarch64",
                device_tree_model="NVIDIA Jetson Orin Nano Developer Kit",
                jetson=True,
            )
        )
        self.assertEqual(nano.profile_id, "jetson-orin-nano")
        self.assertEqual(nano.cuda.status, "present")
        self.assertEqual(nano.tensorrt.status, "missing")
        self.assertTrue(any("TensorRT is missing" in note for note in nano.notes))

        pi5 = detect_target(
            MachineProbe(
                system="Linux",
                machine="aarch64",
                arch="aarch64",
                device_tree_model="Raspberry Pi 5 Model B Rev 1.1",
                raspberry_pi=True,
            )
        )
        self.assertEqual(pi5.profile_id, "rpi5-cpu")
        self.assertEqual(pi5.cuda.status, "missing")
        self.assertEqual(pi5.tensorrt.status, "missing")

        pi4 = detect_target(
            MachineProbe(
                system="Linux",
                machine="aarch64",
                arch="aarch64",
                device_tree_model="Raspberry Pi 4 Model B Rev 1.1",
                raspberry_pi=True,
            )
        )
        self.assertIsNone(pi4.profile_id)
        self.assertTrue(any("not a Pi 5" in note for note in pi4.notes))

        apple = detect_target(
            MachineProbe(
                system="Darwin",
                machine="arm64",
                arch="aarch64",
                apple_silicon=True,
            )
        )
        self.assertIsNone(apple.profile_id)
        self.assertEqual(apple.cuda.status, "missing")
        self.assertEqual(apple.tensorrt.status, "missing")
        self.assertTrue(any("Apple Silicon" in note for note in apple.notes))

        unknown = detect_target(
            MachineProbe(
                system="Linux",
                machine="aarch64",
                arch="aarch64",
                jetson=True,
            )
        )
        self.assertIsNone(unknown.profile_id)
        self.assertEqual(unknown.cuda.status, "present")
        self.assertTrue(any("module" in note for note in unknown.notes))

        self.assertEqual(
            detect_target(MachineProbe(nvidia_gpu_name="NVIDIA L4")).profile_id,
            "cloud-l4",
        )
        self.assertEqual(
            detect_target(MachineProbe(nvidia_gpu_name="NVIDIA A10G")).profile_id,
            "cloud-a10g",
        )
        self.assertEqual(
            detect_target(
                MachineProbe(nvidia_gpu_name="NVIDIA H100 80GB HBM3")
            ).profile_id,
            "cloud-h100",
        )
        rtx = detect_target(MachineProbe(nvidia_gpu_name="NVIDIA GeForce RTX 4070"))
        self.assertEqual(rtx.profile_id, "x86-cuda")
        self.assertFalse(any("no cloud profile" in note for note in rtx.notes))
        other = detect_target(MachineProbe(nvidia_gpu_name="NVIDIA A100-SXM4-80GB"))
        self.assertEqual(other.profile_id, "x86-cuda")
        self.assertTrue(any("no cloud profile" in note for note in other.notes))
        self.assertEqual(
            detect_target(MachineProbe(nvidia_gpu_name="NVIDIA L40S")).profile_id,
            "x86-cuda",
        )
        unnamed = detect_target(
            MachineProbe(
                torch_device="cuda",
                torch_detail="torch.cuda.is_available() is true",
            )
        )
        self.assertEqual(unnamed.profile_id, "x86-cuda")
        self.assertEqual(unnamed.cuda.status, "present")
        self.assertTrue(any("GPU name was not read" in note for note in unnamed.notes))
        stray = detect_target(MachineProbe(trtexec=True))
        self.assertEqual(stray.profile_id, "x86-cpu")
        self.assertEqual(stray.tensorrt.status, "present")

    def test_override_keeps_the_hardware_report(self) -> None:
        found = detect_target(MachineProbe(target_override="cloud-h100"))
        self.assertEqual(found.profile_id, "cloud-h100")
        self.assertEqual(found.cuda.status, "missing")
        self.assertEqual(found.tensorrt.status, "missing")
        self.assertTrue(any("OHHO_TARGET" in note for note in found.notes))
        self.assertTrue(any("lists CUDA" in note for note in found.notes))
        with self.assertRaises(TargetProfileError):
            detect_target(MachineProbe(target_override="nope"))

    def test_arch_names(self) -> None:
        self.assertEqual(normalize_arch("AMD64"), "x86_64")
        self.assertEqual(normalize_arch("arm64"), "aarch64")
        self.assertEqual(normalize_arch("i686"), "other")
        found = detect_target(MachineProbe(machine="i686", arch="other"))
        self.assertIsNone(found.profile_id)

    def test_injected_probe_strips_device_tree_nuls(self) -> None:
        sources = ProbeSources(
            system=lambda: "Linux",
            machine=lambda: "aarch64",
            env=lambda key: "",
            device_tree_model=lambda: "NVIDIA Jetson Orin Nano Developer Kit\x00",
            jetson=lambda: False,
            raspberry_pi=lambda: False,
            apple_silicon=lambda: False,
            torch_probe=lambda: ("cpu", "torch is not installed"),
            nvidia_smi=lambda: ("", 0, "nvidia-smi was not found"),
            ort_providers=lambda: None,
            tensorrt_version=lambda: "",
            trtexec=lambda: False,
        )
        probe = probe_machine(sources)
        self.assertNotIn("\x00", probe.device_tree_model)
        self.assertEqual(detect_target(probe).profile_id, "jetson-orin-nano")

    def test_default_probes_reuse_profile_helpers(self) -> None:
        with patch("ohho.profiles._is_jetson", return_value=True):
            self.assertTrue(_default_jetson())
        with (
            patch("ohho.profiles._is_jetson", return_value=False),
            patch(
                "ohho.targets.detect._default_model",
                return_value="NVIDIA Jetson Orin NX",
            ),
        ):
            self.assertTrue(_default_jetson())
        with (
            patch("ohho.profiles._is_jetson", return_value=False),
            patch("ohho.targets.detect._default_model", return_value=""),
        ):
            self.assertFalse(_default_jetson())
        with patch("ohho.profiles._is_raspberry_pi", return_value=True):
            self.assertTrue(_default_pi())
        with patch("ohho.profiles._is_apple_silicon", return_value=True):
            self.assertTrue(_default_apple())

    def test_torch_probe_uses_resolve_device(self) -> None:
        fake = types.ModuleType("torch")
        with (
            patch.dict(sys.modules, {"torch": fake}),
            patch("ohho.targets.detect.resolve_device", return_value="cpu") as resolve,
        ):
            device, detail = _default_torch()
        self.assertEqual(device, "cpu")
        self.assertIn("false", detail)
        resolve.assert_called_once_with("auto")
        with patch.dict(sys.modules, {"torch": None}):
            device, detail = _default_torch()
        self.assertEqual(device, "cpu")
        self.assertIn("not installed", detail)

    def test_nvidia_smi_query_is_read_only(self) -> None:
        with (
            patch("ohho.targets.detect.shutil.which", return_value=None),
            patch("ohho.targets.detect.subprocess.run") as run,
        ):
            name, memory, detail = _default_nvidia_smi()
        self.assertEqual((name, memory), ("", 0))
        self.assertIn("not found", detail)
        run.assert_not_called()
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA L4, 23034\n",
            stderr="",
        )
        with (
            patch("ohho.targets.detect.shutil.which", return_value="nvidia-smi"),
            patch("ohho.targets.detect.subprocess.run", return_value=completed) as run,
        ):
            name, memory, detail = _default_nvidia_smi()
        self.assertEqual(name, "NVIDIA L4")
        self.assertEqual(memory, 23034)
        self.assertIn("NVIDIA L4", detail)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "nvidia-smi")
        self.assertIn("--query-gpu=name,memory.total", argv)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(run.call_args.kwargs.get("timeout"), 5)

    def test_device_tree_read_does_not_raise(self) -> None:
        self.assertIsInstance(_default_model(), str)

    def test_live_probe_does_not_arm_hardware(self) -> None:
        before = os.environ.get("OHHO_ARM_HARDWARE")
        probe_machine()
        self.assertEqual(os.environ.get("OHHO_ARM_HARDWARE"), before)

    def test_live_detect_names_cuda_and_tensorrt(self) -> None:
        previous = os.environ.pop("OHHO_TARGET", None)
        try:
            found = detect_target()
        finally:
            if previous is not None:
                os.environ["OHHO_TARGET"] = previous
        text = found.text()
        self.assertIn("CUDA", text)
        self.assertIn("TensorRT", text)
        self.assertIn(found.cuda.status, text)
        self.assertIn(found.tensorrt.status, text)
        if found.arch == "x86_64" and found.cuda.status == "missing":
            self.assertEqual(found.profile_id, "x86-cpu")
            self.assertEqual(found.tensorrt.status, "missing")
            self.assertIn("CUDA is missing", found.cuda.detail)
            self.assertIn("TensorRT is missing", found.tensorrt.detail)


class TargetCliTest(unittest.TestCase):
    def test_parser_accepts_list_and_detect(self) -> None:
        listed = build_parser().parse_args(["targets", "list", "--json"])
        self.assertEqual(listed.command, "targets")
        self.assertEqual(listed.action, "list")
        self.assertTrue(listed.json)
        detected = build_parser().parse_args(["targets", "detect"])
        self.assertEqual(detected.action, "detect")
        with patch("sys.stdout", StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["targets", "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_list_json_and_check_exit_codes(self) -> None:
        code, out, err = _run(["targets", "list", "--json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(
            [item["id"] for item in payload["profiles"]],
            list(BUILTIN_IDS),
        )
        code, out, err = _run(
            [
                "targets",
                "check",
                "--profile",
                "rpi5-cpu",
                "--precision",
                "INT8",
                "--provider",
                "tensorrt",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("INT8 TensorRT", err)
        self.assertIn("rpi5-cpu", err)
        code, out, err = _run(
            [
                "targets",
                "check",
                "--profile",
                "jetson-orin-nano",
                "--precision",
                "FP8",
                "--provider",
                "tensorrt",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("FP8 is not valid", err)
        self.assertIn("jetson-orin-nano", err)
        code, out, err = _run(
            [
                "targets",
                "check",
                "--profile",
                "cloud-h100",
                "--precision",
                "FP8",
                "--provider",
                "tensorrt",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("ok: FP8 on cloud-h100", out)
        code, out, err = _run(["targets", "show", "missing"])
        self.assertEqual(code, 2)
        self.assertIn("Unknown target profile", err)

    def test_detect_command_reports_a_cpu_probe(self) -> None:
        found = detect_target(MachineProbe())
        with patch("ohho.targets.command.detect_target", return_value=found):
            code, out, err = _run(["targets", "detect"])
            self.assertEqual(code, 0, err)
            self.assertIn("detected: x86-cpu", out)
            self.assertIn("cuda: missing", out)
            self.assertIn("tensorrt: missing", out)
            self.assertIn("CUDA is missing", out)
            self.assertIn("TensorRT is missing", out)
            code, out, err = _run(["targets", "detect", "--json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["profile_id"], "x86-cpu")
        self.assertEqual(payload["cuda"]["status"], "missing")
        self.assertEqual(payload["tensorrt"]["status"], "missing")

    def test_live_detect_command(self) -> None:
        previous = os.environ.pop("OHHO_TARGET", None)
        try:
            code, out, err = _run(["targets", "detect"])
        finally:
            if previous is not None:
                os.environ["OHHO_TARGET"] = previous
        self.assertEqual(code, 0, err)
        self.assertIn("CUDA", out)
        self.assertIn("TensorRT", out)
        if "arch: x86_64" in out and "cuda: missing" in out:
            self.assertIn("detected: x86-cpu", out)
            self.assertIn("tensorrt: missing", out)


if __name__ == "__main__":
    unittest.main()
