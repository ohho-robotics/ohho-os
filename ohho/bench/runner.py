"""Stage runners for the bench harness.

A runner is five calls, in order: preprocess, h2d, inference, d2h,
postprocess. The harness times each call. Another package can implement the
same methods and pass the object to :func:`ohho.bench.execute`. This module
does not import that package.

ONNX Runtime and NumPy are imported only when an ONNX file is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from ohho.bench.errors import BenchHarnessError
from ohho.targets.providers import filter_providers, installed_execution_providers

STAGE_NAMES: tuple[str, ...] = (
    "preprocess",
    "h2d",
    "inference",
    "d2h",
    "postprocess",
)

_FLOAT_DTYPES = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
}


class StageRunner(Protocol):
    """One timed inference. Each method is one stage.

    ``batch`` is the requested batch size. Later stages receive the value
    returned by the previous stage. The harness does not interpret it.
    """

    def preprocess(self, batch: int) -> Any:
        """Build the host input for ``batch``."""

    def h2d(self, prepared: Any) -> Any:
        """Move the input toward the device. On CPU this is a host copy."""

    def inference(self, device_input: Any) -> Any:
        """Run the model."""

    def d2h(self, device_output: Any) -> Any:
        """Move the output back to the host."""

    def postprocess(self, host_output: Any) -> Any:
        """Read the host output so the copy is not skipped."""


def resolve_artifact(artifact: str) -> Path:
    """Return an existing file, or explain why this reference cannot run.

    A missing path with no suffix and no separator is treated as a registry
    id. Registry ids are not yet supported. A missing filesystem path is an
    ordinary "not found" error.
    """
    path = Path(artifact)
    if path.is_file():
        return path
    if path.exists():
        raise BenchHarnessError(f"artifact is not a file: {artifact}")
    looks_like_path = (
        path.suffix != ""
        or "/" in artifact
        or "\\" in artifact
        or artifact.startswith(".")
    )
    if looks_like_path:
        raise BenchHarnessError(f"artifact not found: {artifact}")
    raise BenchHarnessError(
        f"registry id {artifact!r} is not yet supported; pass a path to an ONNX file"
    )


def session_providers(preferred: Sequence[str]) -> tuple[tuple[str, ...], str]:
    """Pick installed ONNX Runtime providers, preferring ``preferred`` order.

    The note is empty when the first preference is installed. A missing
    onnxruntime install raises :class:`BenchHarnessError` and does not
    silently switch to another backend.
    """
    available = installed_execution_providers()
    if available is None:
        raise BenchHarnessError(
            "Running an ONNX artifact needs onnxruntime, which is not installed. "
            "Install it with: pip install onnxruntime"
        )
    chosen = tuple(filter_providers(preferred, available))
    note = ""
    if preferred and chosen and chosen[0] != preferred[0]:
        note = (
            f"{preferred[0]} is not installed in this process; "
            f"the session is using {chosen[0]}."
        )
    return chosen, note


class OnnxArtifactRunner:
    """Random-input ONNX Runtime session split into the five bench stages.

    Host-to-device and device-to-host are explicit NumPy copies. On CPU the
    runtime's own copy stays inside ``inference``. CUDA events, when the
    harness selects them, therefore time those copies plus ``session.run``.
    """

    def __init__(
        self,
        path: Path,
        providers: Sequence[str],
        *,
        provider_note: str = "",
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise BenchHarnessError(
                "Running an ONNX artifact needs numpy, which is not installed. "
                "Install it with: pip install numpy"
            ) from exc
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BenchHarnessError(
                "Running an ONNX artifact needs onnxruntime, which is not installed. "
                "Install it with: pip install onnxruntime"
            ) from exc
        self._np = np
        self.providers = tuple(providers)
        self.provider_note = provider_note
        try:
            self._session = ort.InferenceSession(str(path), providers=list(providers))
        except Exception as exc:
            raise BenchHarnessError(f"cannot load ONNX artifact {path}: {exc}") from exc
        inputs = self._session.get_inputs()
        if not inputs:
            raise BenchHarnessError(f"ONNX artifact {path} has no inputs")
        self._inputs = inputs

    def preprocess(self, batch: int) -> dict[str, Any]:
        feeds: dict[str, Any] = {}
        for meta in self._inputs:
            dtype = _FLOAT_DTYPES.get(str(meta.type))
            if dtype is None:
                raise BenchHarnessError(
                    f"ONNX input {meta.name!r} has type {meta.type!r}; "
                    "only float tensors are supported"
                )
            shape = _concrete_shape(meta.name, meta.shape, batch)
            array = self._np.random.standard_normal(shape).astype(dtype)
            feeds[meta.name] = array
        return feeds

    def h2d(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return {
            name: self._np.ascontiguousarray(value) for name, value in prepared.items()
        }

    def inference(self, device_input: dict[str, Any]) -> list[Any]:
        return self._session.run(None, device_input)

    def d2h(self, device_output: list[Any]) -> list[Any]:
        return [self._np.array(value, copy=True) for value in device_output]

    def postprocess(self, host_output: list[Any]) -> float:
        if not host_output:
            return 0.0
        array = host_output[0]
        size = int(getattr(array, "size", 0))
        if size <= 0:
            return 0.0
        return float(array.reshape(-1)[0])


def _concrete_shape(
    name: str, raw_shape: Sequence[object], batch: int
) -> tuple[int, ...]:
    shape: list[int] = []
    for index, dim in enumerate(raw_shape):
        if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
            if index == 0 and dim != batch:
                raise BenchHarnessError(
                    f"ONNX input {name!r} has a fixed batch of {dim}, "
                    f"which does not match --batch {batch}"
                )
            shape.append(dim)
            continue
        if index == 0:
            shape.append(batch)
            continue
        shape.append(1)
    if not shape:
        shape.append(batch)
    return tuple(shape)
