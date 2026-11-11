"""TensorRT INT8 entropy calibrator.

The builder attaches the object this module returns to
``config.int8_calibrator`` before ``build_serialized_network``. Batch
decoding is a separate :class:`BatchSource`. A later change can add an
image-folder source without touching that attach point.

Samples are raw little-endian float32. A directory holds ``calib.json``
(``shape`` without the batch axis, ``dtype`` ``float32``) and one ``.bin``
file per sample. ``.npy`` files are read when NumPy is installed.
"""

from __future__ import annotations

import array
import ctypes
import json
from pathlib import Path
from typing import Protocol

from ohho.optimize.errors import OptimizeError

_CALIB_KEYS = frozenset({"shape", "dtype"})


class BatchSource(Protocol):
    """One batch of float32 samples at a time."""

    nbatches: int

    def batch(self, index: int) -> array.array:
        """Return a flat float32 array of ``batch_size * product(shape)``."""


class EmptyBatchSource:
    """No samples. Used when a calibration cache is reused as-is."""

    nbatches = 0

    def batch(self, index: int) -> array.array:
        raise IndexError(index)


class DirectoryBatchSource:
    """Group on-disk samples into fixed-size batches."""

    def __init__(
        self, data_dir: str | Path, batch_size: int, sample_shape: tuple[int, ...]
    ):
        if batch_size < 1:
            raise OptimizeError("--calib-batch must be at least 1.")
        self.batch_size = batch_size
        self.sample_shape = tuple(int(dim) for dim in sample_shape)
        self._count = _product(self.sample_shape)
        root = Path(data_dir)
        if not root.is_dir():
            raise OptimizeError(f"Calibration directory does not exist: {root}")
        self._files = _sample_files(root, self.sample_shape)
        if len(self._files) < batch_size:
            raise OptimizeError(
                f"Calibration directory has {len(self._files)} samples; "
                f"--calib-batch is {batch_size}."
            )
        self.nbatches = len(self._files) // batch_size

    def batch(self, index: int) -> array.array:
        if index < 0 or index >= self.nbatches:
            raise IndexError(index)
        start = index * self.batch_size
        flat = array.array("f")
        for path in self._files[start : start + self.batch_size]:
            flat.extend(_read_sample(path, self._count))
        return flat


def load_calibration_source(
    data_dir: str | Path,
    *,
    batch_size: int,
    sample_shape: tuple[int, ...],
) -> DirectoryBatchSource:
    """Read ``data_dir`` into a batch source for the entropy calibrator."""
    return DirectoryBatchSource(data_dir, batch_size, sample_shape)


def make_entropy_calibrator(
    source: BatchSource,
    *,
    batch_size: int,
    cache_path: str | Path | None,
    trt_module: object,
    copy_to_device=None,
):
    """Return an ``IInt8EntropyCalibrator2`` bound to ``source``.

    ``copy_to_device`` turns a host ``array.array`` into a device pointer
    and keeps the device buffer alive on the list it is given. When it is
    omitted, ``get_batch`` returns a host pointer. Tests use the host
    pointer. A real TensorRT build passes a CUDA copy.
    """
    if batch_size < 1:
        raise OptimizeError("--calib-batch must be at least 1.")
    base = getattr(trt_module, "IInt8EntropyCalibrator2", None)
    if not isinstance(base, type):
        raise OptimizeError(
            "TensorRT did not provide IInt8EntropyCalibrator2. "
            "INT8 engine builds need a TensorRT install with the entropy calibrator."
        )

    class EntropyCalibrator(base):
        """Feeds ``source`` and reads or writes ``cache_path``."""

        def __init__(self) -> None:
            super().__init__()
            self._source = source
            self._batch_size = batch_size
            self._cache = None if cache_path is None else Path(cache_path)
            self._cursor = 0
            self._buffers: list[ctypes.Array] = []
            self._device_owner: list[object] = []
            self._copy = copy_to_device

        def get_batch_size(self) -> int:
            return self._batch_size

        def get_batch(self, names) -> list[int] | None:
            try:
                flat = self._source.batch(self._cursor)
            except IndexError:
                return None
            self._cursor += 1
            try:
                width = len(names)
            except TypeError:
                width = 1
            if width != 1:
                raise OptimizeError(
                    "The entropy calibrator feeds one input tensor. "
                    "Multi-input calibration is a separate batch source."
                )
            buffer = (ctypes.c_float * len(flat)).from_buffer_copy(flat.tobytes())
            self._buffers.append(buffer)
            if self._copy is not None:
                pointer = int(self._copy(flat, self._device_owner))
            else:
                pointer = ctypes.addressof(buffer)
            return [pointer]

        def read_calibration_cache(self):
            if self._cache is None or not self._cache.is_file():
                return None
            data = self._cache.read_bytes()
            return data or None

        def write_calibration_cache(self, cache) -> None:
            if self._cache is None:
                return
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_bytes(bytes(cache))

    return EntropyCalibrator()


def cuda_copy(flat: array.array, owner: list[object]) -> int:
    """Copy ``flat`` to a CUDA tensor and return its device pointer."""
    try:
        import torch
    except Exception as exc:
        raise OptimizeError(
            "INT8 calibration needs PyTorch to copy batches onto the CUDA device."
        ) from exc
    if not torch.cuda.is_available():
        raise OptimizeError(
            "INT8 calibration needs a CUDA device to copy batches. GPU follow-up."
        )
    tensor = torch.tensor(flat.tolist(), dtype=torch.float32, device="cuda")
    owner.append(tensor)
    return int(tensor.data_ptr())


def _product(shape: tuple[int, ...]) -> int:
    if not shape:
        raise OptimizeError("Calibration shape must name at least one axis.")
    count = 1
    for dim in shape:
        if dim < 1:
            raise OptimizeError(f"Calibration shape {shape} must be positive.")
        count *= dim
    return count


def _sample_files(root: Path, sample_shape: tuple[int, ...]) -> list[Path]:
    meta_path = root / "calib.json"
    if meta_path.is_file():
        declared = _read_meta(meta_path)
        if declared != sample_shape:
            raise OptimizeError(
                f"Calibration shape {declared} does not match the model input "
                f"{sample_shape}."
            )
        files = sorted(path for path in root.glob("*.bin") if path.is_file())
        if not files:
            raise OptimizeError(f"{root} has calib.json but no .bin samples.")
        return files
    npy = sorted(path for path in root.glob("*.npy") if path.is_file())
    if npy:
        return npy
    raise OptimizeError(
        f"{root} needs calib.json plus .bin samples, or .npy files. "
        "Each sample is float32 in the model input shape, without the batch axis."
    )


def _read_meta(path: Path) -> tuple[int, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OptimizeError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OptimizeError(f"{path.name} must be an object.")
    extra = set(data) - _CALIB_KEYS
    if extra:
        raise OptimizeError(
            f"{path.name} has unknown fields: {', '.join(sorted(extra))}."
        )
    if data.get("dtype") != "float32":
        raise OptimizeError(f"{path.name} dtype must be float32.")
    shape = data.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(type(d) is int for d in shape)
    ):
        raise OptimizeError(f"{path.name} shape must be a list of positive integers.")
    return tuple(shape)


def _read_sample(path: Path, count: int) -> array.array:
    if path.suffix.lower() == ".npy":
        return _read_npy(path, count)
    data = path.read_bytes()
    if len(data) != count * 4:
        raise OptimizeError(
            f"{path.name} has {len(data)} bytes; expected {count * 4} for {count} float32 values."
        )
    values = array.array("f")
    values.frombytes(data)
    return values


def _read_npy(path: Path, count: int) -> array.array:
    try:
        import numpy as np
    except Exception as exc:
        raise OptimizeError(
            "Reading .npy calibration files needs NumPy. "
            "Use calib.json and .bin samples, or pip install 'ohho-os[optimize]'."
        ) from exc
    array_np = np.load(path)
    if array_np.dtype != np.float32:
        raise OptimizeError(f"{path.name} must be float32.")
    flat = array_np.reshape(-1)
    if int(flat.shape[0]) != count:
        raise OptimizeError(
            f"{path.name} has {flat.shape[0]} values; expected {count}."
        )
    values = array.array("f")
    values.frombytes(flat.tobytes())
    return values
