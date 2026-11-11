"""ONNX Runtime execution-provider order for a hardware target.

Ported from the learning_engine hardware helper. An Orin target tries
TensorRT, then CUDA, then CPU. A discrete GPU tries CUDA, then CPU, and
promotes TensorRT only when the caller asks. A CPU target, and the NPU
placeholder, stay on the CPU provider.

``onnx_providers`` filters that order through the providers installed in
this process. A stored target profile keeps its own order: cloud GPUs ask
for TensorRT first (``prefer_tensorrt=True``), and ``x86-cuda`` keeps CUDA
first with TensorRT available next.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from ohho.targets.errors import TargetProfileError

TRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
PROVIDER_NAMES = (TRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER)

_PREFERENCE: dict[str, tuple[str, ...]] = {
    "orin_gpu": (TRT_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER),
    "dgpu": (CUDA_PROVIDER, CPU_PROVIDER),
    "npu": (CPU_PROVIDER,),
    "none": (CPU_PROVIDER,),
}


def provider_preference(
    accelerator: str, *, prefer_tensorrt: bool = False
) -> tuple[str, ...]:
    """Return the execution-provider order for an accelerator class.

    ``prefer_tensorrt`` moves TensorRT to the front for ``orin_gpu`` and
    ``dgpu``. It does not invent a TensorRT provider for ``none`` or ``npu``.
    """
    try:
        preferred = _PREFERENCE[accelerator]
    except KeyError:
        known = ", ".join(sorted(_PREFERENCE))
        raise TargetProfileError(
            f"Unknown accelerator {accelerator!r}. Expected one of: {known}."
        ) from None
    if prefer_tensorrt and accelerator in {"orin_gpu", "dgpu"}:
        return (TRT_PROVIDER, *[item for item in preferred if item != TRT_PROVIDER])
    return preferred


def workstation_provider_order() -> tuple[str, ...]:
    """CUDA, then TensorRT, then CPU.

    CUDA stays first, matching the discrete-GPU preference. TensorRT is
    second so a benchmark can select it without promoting it ahead of CUDA.
    """
    cuda, cpu = provider_preference("dgpu")
    return (cuda, TRT_PROVIDER, cpu)


def installed_execution_providers() -> tuple[str, ...] | None:
    """Providers reported by the installed onnxruntime, or ``None``.

    The import is read-only. A missing or broken install returns ``None``.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return None
    try:
        found = ort.get_available_providers()
    except Exception:
        return None
    if not isinstance(found, (list, tuple)):
        return None
    return tuple(str(item) for item in found)


def filter_providers(
    preferred: Sequence[str], available: Collection[str] | None
) -> list[str]:
    """Keep ``preferred`` entries that are installed, else CPU.

    ``available is None`` means onnxruntime is not installed, which matches
    the ported helper: the only honest session provider is CPU.
    """
    if available is None:
        return [CPU_PROVIDER]
    have = set(available)
    picked = [item for item in preferred if item in have]
    return picked or [CPU_PROVIDER]


def onnx_providers(
    accelerator: str,
    *,
    prefer_tensorrt: bool = False,
    available: Collection[str] | None = None,
    probe: bool = False,
) -> list[str]:
    """Filter the ported provider order by what this process can run.

    Pass ``available`` to avoid a live probe. Set ``probe=True`` to ask
    onnxruntime. With neither, a missing install is assumed and the result
    is the CPU provider.
    """
    preferred = provider_preference(accelerator, prefer_tensorrt=prefer_tensorrt)
    if probe and available is None:
        available = installed_execution_providers()
    return filter_providers(preferred, available)
