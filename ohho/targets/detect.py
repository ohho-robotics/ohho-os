"""Read-only detection of an optimisation target for this machine.

Board checks reuse :func:`ohho.profiles._is_jetson`,
:func:`ohho.profiles._is_raspberry_pi` and
:func:`ohho.profiles._is_apple_silicon`. The device string reuses
:func:`ohho.hardware.resolve_device`. GPU name and memory come from a
``nvidia-smi`` query. Nothing here writes a file, opens a robot, or sets
``OHHO_ARM_HARDWARE``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.hardware import resolve_device
from ohho.targets.providers import (
    CUDA_PROVIDER,
    TRT_PROVIDER,
    installed_execution_providers,
)
from ohho.targets.schema import TargetProfile, get_profile, profile_to_dict

DETECTION_SCHEMA = "ohho.target_detection.v1"

_ARCH = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def normalize_arch(machine: str) -> str:
    """Map a ``platform.machine()`` value onto ``aarch64``, ``x86_64`` or ``other``."""
    return _ARCH.get(machine.lower(), "other")


def match_gpu_profile(name: str) -> str | None:
    """Return a built-in id for a discrete GPU name, or ``None``."""
    if re.search(r"\bH100\b", name, re.IGNORECASE):
        return "cloud-h100"
    if re.search(r"\bA10G\b", name, re.IGNORECASE):
        return "cloud-a10g"
    if re.search(r"\bL4\b", name, re.IGNORECASE):
        return "cloud-l4"
    if re.search(r"\bRTX\b", name, re.IGNORECASE):
        return "x86-cuda"
    return None


@dataclass(frozen=True)
class ComponentStatus:
    """Whether CUDA or TensorRT was observed, with a short reason."""

    status: str
    detail: str


@dataclass(frozen=True)
class MachineProbe:
    """Facts collected from the local machine. Defaults describe a CPU-only x86_64 host."""

    system: str = "Linux"
    machine: str = "x86_64"
    arch: str = "x86_64"
    device_tree_model: str = ""
    jetson: bool = False
    raspberry_pi: bool = False
    apple_silicon: bool = False
    torch_device: str = "cpu"
    torch_detail: str = "torch is not installed"
    nvidia_gpu_name: str = ""
    nvidia_memory_mib: int = 0
    nvidia_smi_detail: str = "nvidia-smi was not found"
    ort_providers: tuple[str, ...] | None = None
    tensorrt_version: str = ""
    trtexec: bool = False
    target_override: str = ""


@dataclass(frozen=True)
class Detection:
    """The selected target plus the CUDA and TensorRT probe."""

    profile_id: str | None
    arch: str
    platform: str
    accelerator: str | None
    cuda: ComponentStatus
    tensorrt: ComponentStatus
    ort_providers: tuple[str, ...] | None
    notes: tuple[str, ...]
    profile: TargetProfile | None
    nvidia_gpu_name: str = ""
    nvidia_memory_mib: int = 0

    def text(self) -> str:
        """Multi-line report for ``ohho targets detect``."""
        lines = [
            f"detected: {self.profile_id or 'none'}",
            f"  arch: {self.arch}",
            f"  platform: {self.platform}",
            f"  accelerator: {self.accelerator or 'unknown'}",
            f"  cuda: {self.cuda.status} - {self.cuda.detail}",
            f"  tensorrt: {self.tensorrt.status} - {self.tensorrt.detail}",
        ]
        if self.ort_providers is None:
            lines.append("  onnxruntime: not installed")
        else:
            joined = ", ".join(self.ort_providers) or "(none)"
            lines.append(f"  onnxruntime providers: {joined}")
        if self.profile is not None:
            lines.append(
                "  profile providers: " + ", ".join(self.profile.execution_providers)
            )
        if self.nvidia_gpu_name:
            lines.append(
                f"  gpu: {self.nvidia_gpu_name} ({self.nvidia_memory_mib} MiB)"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON report for ``ohho targets detect --json``."""
        gpu: dict[str, Any] | None
        if self.nvidia_gpu_name:
            gpu = {
                "name": self.nvidia_gpu_name,
                "memory_mib": self.nvidia_memory_mib,
            }
        else:
            gpu = None
        profile = None if self.profile is None else profile_to_dict(self.profile)
        return {
            "schema": DETECTION_SCHEMA,
            "profile_id": self.profile_id,
            "arch": self.arch,
            "platform": self.platform,
            "accelerator": self.accelerator,
            "cuda": {"status": self.cuda.status, "detail": self.cuda.detail},
            "tensorrt": {
                "status": self.tensorrt.status,
                "detail": self.tensorrt.detail,
            },
            "onnxruntime": {
                "installed": self.ort_providers is not None,
                "providers": list(self.ort_providers or ()),
            },
            "nvidia_gpu": gpu,
            "notes": list(self.notes),
            "profile": profile,
        }


def _env(key: str) -> str:
    return os.environ.get(key, "")


def _default_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(encoding="utf-8")
    except OSError:
        return ""


def _default_jetson() -> bool:
    from ohho.profiles import _is_jetson

    if _is_jetson():
        return True
    model = _default_model().lower()
    return "jetson" in model or "tegra" in model


def _default_pi() -> bool:
    from ohho.profiles import _is_raspberry_pi

    return _is_raspberry_pi()


def _default_apple() -> bool:
    from ohho.profiles import _is_apple_silicon

    return _is_apple_silicon()


def _default_torch() -> tuple[str, str]:
    import importlib

    try:
        importlib.import_module("torch")
    except Exception:
        return "cpu", "torch is not installed"
    device = resolve_device("auto")
    if device == "cuda":
        return "cuda", "torch.cuda.is_available() is true"
    if device == "mps":
        return "mps", "torch MPS is available"
    return "cpu", "torch.cuda.is_available() is false"


def _default_nvidia_smi() -> tuple[str, int, str]:
    if shutil.which("nvidia-smi") is None:
        return "", 0, "nvidia-smi was not found"
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return "", 0, "nvidia-smi could not be run"
    if out.returncode != 0 or not out.stdout.strip():
        return "", 0, "nvidia-smi reported no GPU"
    parts = [item.strip() for item in out.stdout.splitlines()[0].split(",")]
    if len(parts) < 2 or not parts[0]:
        return "", 0, "nvidia-smi output could not be parsed"
    try:
        memory = int(float(parts[1]))
    except ValueError:
        return parts[0], 0, f"nvidia-smi reported {parts[0]}"
    return parts[0], memory, f"nvidia-smi reported {parts[0]}"


def _default_tensorrt() -> str:
    try:
        import tensorrt as trt
    except Exception:
        return ""
    version = getattr(trt, "__version__", "")
    return str(version or "")


def _default_trtexec() -> bool:
    return shutil.which("trtexec") is not None


@dataclass(frozen=True)
class ProbeSources:
    """Callables used by :func:`probe_machine`. Tests pass fakes; defaults only read."""

    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    env: Callable[[str], str] = _env
    device_tree_model: Callable[[], str] = _default_model
    jetson: Callable[[], bool] = _default_jetson
    raspberry_pi: Callable[[], bool] = _default_pi
    apple_silicon: Callable[[], bool] = _default_apple
    torch_probe: Callable[[], tuple[str, str]] = _default_torch
    nvidia_smi: Callable[[], tuple[str, int, str]] = _default_nvidia_smi
    ort_providers: Callable[[], tuple[str, ...] | None] = installed_execution_providers
    tensorrt_version: Callable[[], str] = _default_tensorrt
    trtexec: Callable[[], bool] = _default_trtexec


def probe_machine(sources: ProbeSources | None = None) -> MachineProbe:
    """Read this machine. The default sources do not write or open a robot."""
    src = sources or ProbeSources()
    machine = src.machine()
    name, memory, smi_detail = src.nvidia_smi()
    torch_device, torch_detail = src.torch_probe()
    return MachineProbe(
        system=src.system(),
        machine=machine,
        arch=normalize_arch(machine),
        device_tree_model=src.device_tree_model().replace("\x00", "").strip(),
        jetson=bool(src.jetson()),
        raspberry_pi=bool(src.raspberry_pi()),
        apple_silicon=bool(src.apple_silicon()),
        torch_device=torch_device,
        torch_detail=torch_detail,
        nvidia_gpu_name=name.strip(),
        nvidia_memory_mib=memory,
        nvidia_smi_detail=smi_detail,
        ort_providers=src.ort_providers(),
        tensorrt_version=src.tensorrt_version().strip(),
        trtexec=bool(src.trtexec()),
        target_override=src.env("OHHO_TARGET").strip(),
    )


def _model_text(probe: MachineProbe) -> str:
    return probe.device_tree_model.replace("\x00", "").strip()


def _jetson_board(probe: MachineProbe, model: str) -> bool:
    lowered = model.lower()
    return probe.jetson or "jetson" in lowered or "tegra" in lowered


def _jetson_profile(model: str) -> str | None:
    lowered = model.lower()
    if "orin" in lowered and "nano" in lowered:
        return "jetson-orin-nano"
    if "orin" in lowered and re.search(r"\bnx\b", lowered):
        return "jetson-orin-nx"
    if "orin" in lowered and "agx" in lowered:
        return "jetson-agx-orin"
    return None


def _cuda_seen(probe: MachineProbe) -> bool:
    if probe.torch_device == "cuda" or probe.nvidia_gpu_name:
        return True
    return CUDA_PROVIDER in (probe.ort_providers or ())


def _cuda_status(probe: MachineProbe, jetson_board: bool) -> ComponentStatus:
    reasons: list[str] = []
    if probe.torch_device == "cuda" and probe.torch_detail:
        reasons.append(probe.torch_detail)
    if probe.nvidia_gpu_name:
        detail = probe.nvidia_smi_detail
        if not detail or "not found" in detail:
            detail = f"GPU {probe.nvidia_gpu_name} was reported"
        reasons.append(detail)
    if CUDA_PROVIDER in (probe.ort_providers or ()):
        reasons.append("ONNX Runtime lists CUDAExecutionProvider")
    if jetson_board:
        reasons.append("Jetson board detected")
    if reasons:
        return ComponentStatus(
            "present", "CUDA is present (" + "; ".join(reasons) + ")."
        )
    parts = [part for part in (probe.torch_detail, probe.nvidia_smi_detail) if part]
    if probe.ort_providers is None:
        parts.append("onnxruntime is not installed")
    else:
        parts.append("ONNX Runtime has no CUDAExecutionProvider")
    return ComponentStatus("missing", "CUDA is missing (" + "; ".join(parts) + ").")


def _tensorrt_status(probe: MachineProbe) -> ComponentStatus:
    version = probe.tensorrt_version.strip()
    reasons: list[str] = []
    if version:
        reasons.append(f"TensorRT {version} is installed")
    if probe.trtexec:
        reasons.append("trtexec was found")
    if TRT_PROVIDER in (probe.ort_providers or ()):
        reasons.append("ONNX Runtime lists TensorrtExecutionProvider")
    if reasons:
        return ComponentStatus(
            "present", "TensorRT is present (" + "; ".join(reasons) + ")."
        )
    parts = [
        "the tensorrt package is not installed",
        "trtexec was not found",
    ]
    if probe.ort_providers is None:
        parts.append("onnxruntime is not installed")
    else:
        parts.append("ONNX Runtime has no TensorrtExecutionProvider")
    return ComponentStatus("missing", "TensorRT is missing (" + "; ".join(parts) + ").")


def _x86_profile(probe: MachineProbe) -> tuple[str, list[str]]:
    notes: list[str] = []
    if probe.nvidia_gpu_name:
        matched = match_gpu_profile(probe.nvidia_gpu_name)
        if matched is not None:
            return matched, notes
        notes.append(
            f"Discrete GPU {probe.nvidia_gpu_name!r} has no cloud profile. "
            "Using x86-cuda."
        )
        return "x86-cuda", notes
    if _cuda_seen(probe):
        notes.append("CUDA is present, but the GPU name was not read. Using x86-cuda.")
        return "x86-cuda", notes
    return "x86-cpu", notes


def _select(
    probe: MachineProbe, model: str, jetson_board: bool
) -> tuple[str | None, list[str]]:
    notes: list[str] = []
    override = probe.target_override.strip()
    if override:
        notes.append(f"OHHO_TARGET selected {override}. Hardware probes are unchanged.")
        return override, notes
    if jetson_board:
        chosen = _jetson_profile(model)
        if chosen is not None:
            return chosen, notes
        notes.append(
            "Jetson board detected, but the module was not Orin Nano, Orin NX "
            "or AGX Orin. No target profile was selected."
        )
        return None, notes
    lowered = model.lower()
    if "raspberry pi 5" in lowered:
        return "rpi5-cpu", notes
    if probe.raspberry_pi or "raspberry pi" in lowered:
        notes.append(
            "Raspberry Pi detected, but it is not a Pi 5. rpi5-cpu was not selected."
        )
        return None, notes
    if probe.apple_silicon:
        notes.append(
            "Apple Silicon has no built-in optimisation target. "
            "CUDA and TensorRT are not used on this machine."
        )
        return None, notes
    if probe.arch == "x86_64":
        chosen, extra = _x86_profile(probe)
        notes.extend(extra)
        return chosen, notes
    if probe.arch == "aarch64":
        notes.append(
            "aarch64 CPU detected with no Orin or Pi 5 board id. "
            "No target profile was selected."
        )
        return None, notes
    notes.append(f"Architecture {probe.machine!r} has no built-in target profile.")
    return None, notes


def _from_probe(probe: MachineProbe) -> Detection:
    model = _model_text(probe)
    jetson_board = _jetson_board(probe, model)
    cuda = _cuda_status(probe, jetson_board)
    tensorrt = _tensorrt_status(probe)
    profile_id, notes = _select(probe, model, jetson_board)
    profile: TargetProfile | None = None
    if profile_id is not None:
        profile = get_profile(profile_id)
        if TRT_PROVIDER in profile.execution_providers and tensorrt.status == "missing":
            notes.append(
                f"{profile.id} lists TensorRT, but TensorRT is missing on this machine."
            )
        if CUDA_PROVIDER in profile.execution_providers and cuda.status == "missing":
            notes.append(
                f"{profile.id} lists CUDA, but CUDA is missing on this machine."
            )
    return Detection(
        profile_id=None if profile is None else profile.id,
        arch=probe.arch,
        platform=f"{probe.system} {probe.machine}",
        accelerator=None if profile is None else profile.accelerator,
        cuda=cuda,
        tensorrt=tensorrt,
        ort_providers=probe.ort_providers,
        notes=tuple(notes),
        profile=profile,
        nvidia_gpu_name=probe.nvidia_gpu_name,
        nvidia_memory_mib=probe.nvidia_memory_mib,
    )


def detect_target(probe: MachineProbe | None = None) -> Detection:
    """Select a built-in target for ``probe``, or read this machine when omitted."""
    return _from_probe(probe if probe is not None else probe_machine())
