"""Toolchain versions and the host GPU recorded on an optimize manifest.

Probes stay inside the functions that need them. A missing package, a missing
``nvidia-smi``, or a machine with no CUDA device leaves that field null.
JetPack is recorded only when ``/etc/nv_tegra_release`` is present.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ohho.optimize.errors import OptimizeError
from ohho.optimize.manifest import manifest_beside, package_versions
from ohho.optimize.qdq import trt_version_text

_TEGRA_PATH = Path("/etc/nv_tegra_release")
_TEGRA_RELEASE = re.compile(
    r"R(\d+)\s*\(release\),\s*REVISION:\s*([0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)
_ENGINE_ROLES = frozenset({"engine", "fp8"})


@dataclass(frozen=True)
class Toolchain:
    """Versions that identify one build. Null means that piece was not detectable."""

    torch: str | None
    onnx: str | None
    onnxruntime: str | None
    tensorrt: str | None
    cuda: str | None
    driver: str | None
    jetpack: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Manifest ``versions`` object."""
        return {
            "torch": self.torch,
            "onnx": self.onnx,
            "onnxruntime": self.onnxruntime,
            "tensorrt": self.tensorrt,
            "cuda": self.cuda,
            "driver": self.driver,
            "jetpack": self.jetpack,
        }


@dataclass(frozen=True)
class HostGpu:
    """GPU visible to this process. Both fields are null when none is visible.

    ``gpu_architecture`` is the compute capability ``major.minor`` (for example
    ``8.9``). It is the value compared when an engine is loaded.
    """

    gpu_name: str | None
    gpu_architecture: str | None

    def as_dict(self) -> dict[str, str | None]:
        """Manifest ``host`` object."""
        return {
            "gpu_name": self.gpu_name,
            "gpu_architecture": self.gpu_architecture,
        }


def detect_toolchain() -> Toolchain:
    """Package versions plus CUDA, the driver, and JetPack when those are visible."""
    found = package_versions()
    tensorrt = _dotted(found.get("tensorrt")) or found.get("tensorrt")
    return Toolchain(
        torch=found.get("torch"),
        onnx=found.get("onnx"),
        onnxruntime=found.get("onnxruntime"),
        tensorrt=tensorrt,
        cuda=detect_cuda(),
        driver=detect_driver(),
        jetpack=detect_jetpack(),
    )


def detect_cuda() -> str | None:
    """``torch.version.cuda`` when PyTorch imports, else null."""
    try:
        import torch
    except Exception:
        return None
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if not cuda:
        return None
    return _clean(str(cuda))


def detect_driver(query: Callable[[], str | None] | None = None) -> str | None:
    """Driver version from ``nvidia-smi``, or null when the tool is absent.

    ``query`` replaces the ``nvidia-smi`` call in tests.
    """
    if query is not None:
        return _clean(query())
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    return _clean(lines[0])


def detect_host_gpu() -> HostGpu:
    """Compute capability ``major.minor`` and the device name, or a pair of nulls."""
    try:
        import torch
    except Exception:
        return HostGpu(None, None)
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        return HostGpu(None, None)
    if not available:
        return HostGpu(None, None)
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = _clean(str(torch.cuda.get_device_name(0)))
    except Exception:
        return HostGpu(None, None)
    return HostGpu(name, f"{int(major)}.{int(minor)}")


def detect_jetpack(path: Path | None = None) -> str | None:
    """L4T release from ``/etc/nv_tegra_release``, or null when that file is absent.

    That file is how a device image exposes its JetPack release. Other hosts
    record null.
    """
    candidate = _TEGRA_PATH if path is None else path
    try:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return jetpack_from_tegra_release(text)


def jetpack_from_tegra_release(text: str) -> str | None:
    """Return ``R<release>.<revision>`` from an ``nv_tegra_release`` body."""
    match = _TEGRA_RELEASE.search(text)
    if match is None:
        return None
    return f"R{match.group(1)}.{match.group(2)}"


def resolve_toolchain(standins: object) -> Toolchain:
    """Prefer an injected toolchain. A versioned TensorRT stand-in fills that field."""
    injected = getattr(standins, "toolchain", None)
    if injected is not None:
        return injected
    detected = detect_toolchain()
    module = getattr(standins, "trt_module", None)
    if module is None:
        return detected
    text = trt_version_text(module)
    if not text:
        return detected
    return replace(detected, tensorrt=text)


def resolve_host(standins: object) -> HostGpu:
    """Prefer an injected host GPU, otherwise detect one."""
    injected = getattr(standins, "host", None)
    if injected is not None:
        return injected
    return detect_host_gpu()


def injected_tensorrt_version(standins: object) -> str | None:
    """Version for the target-profile gate, or ``None`` to probe the install.

    An injected toolchain or a versioned TensorRT stand-in is checked instead
    of whatever TensorRT happens to be installed in the test process.
    """
    injected = getattr(standins, "toolchain", None)
    if injected is not None:
        return injected.tensorrt or ""
    module = getattr(standins, "trt_module", None)
    if module is None:
        return None
    text = trt_version_text(module)
    if text:
        return text
    return None


def artifact_has_engine(document: dict) -> bool:
    """True when the manifest lists a TensorRT engine output."""
    outputs = document.get("outputs")
    if not isinstance(outputs, list):
        return False
    return any(
        isinstance(item, dict) and item.get("role") in _ENGINE_ROLES for item in outputs
    )


def require_engine_host(
    artifact: str | Path,
    *,
    toolchain: Toolchain | None = None,
    host: HostGpu | None = None,
) -> dict:
    """Refuse an engine whose GPU architecture or TensorRT version differs.

    ``artifact`` is a build directory, an engine file in that directory, or
    ``manifest.json``. A matching host returns the manifest. ONNX-only builds
    are not engines and are refused by this function.
    """
    _path, document = manifest_beside(artifact)
    if not artifact_has_engine(document):
        raise OptimizeError(f"{artifact} has no TensorRT engine in its manifest.")
    current_host = host if host is not None else detect_host_gpu()
    current_trt = (
        toolchain.tensorrt if toolchain is not None else detect_toolchain().tensorrt
    )
    recorded_arch = document["host"]["gpu_architecture"]
    recorded_trt = document["versions"]["tensorrt"]
    if recorded_arch != current_host.gpu_architecture:
        raise OptimizeError(
            f"Refusing engine {artifact}: manifest GPU architecture is "
            f"{_show(recorded_arch)}, this host is "
            f"{_show(current_host.gpu_architecture)}."
        )
    if recorded_trt != current_trt:
        raise OptimizeError(
            f"Refusing engine {artifact}: manifest TensorRT version is "
            f"{_show(recorded_trt)}, this host has {_show(current_trt)}."
        )
    return document


def _dotted(raw: str | None) -> str | None:
    if not raw:
        return None
    return trt_version_text(SimpleNamespace(__version__=raw)) or None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _show(value: str | None) -> str:
    if value:
        return value
    return "unknown"
