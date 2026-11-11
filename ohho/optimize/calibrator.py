"""TensorRT INT8 entropy calibrator.

The builder attaches the object this module returns to
``config.int8_calibrator`` before ``build_serialized_network`` when TensorRT
still uses implicit calibration. Batch decoding is a separate
:class:`BatchSource`. ``--calib-max-images`` limits how many samples are read.

Samples are raw little-endian float32. A directory holds ``calib.json``
(``shape`` without the batch axis, ``dtype`` ``float32``, optional
``preprocessing``) and one ``.bin`` file per sample. ``.npy`` files are read
when NumPy is installed.

A calibration cache is reused only when the sidecar next to it records the
same model sha256 and preprocessing config. Either change invalidates the
cache and the build calibrates again.
"""

from __future__ import annotations

import array
import ctypes
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Protocol

from ohho.optimize.errors import OptimizeError

logger = logging.getLogger("ohho.optimize")

_CALIB_KEYS = frozenset({"shape", "dtype", "preprocessing"})
_PREPROCESS_FIELDS = frozenset({"mean", "std", "resize", "layout"})
_LAYOUTS = frozenset({"nchw", "nhwc"})
_SIDECAR_SUFFIX = ".meta.json"


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
        self,
        data_dir: str | Path,
        batch_size: int,
        sample_shape: tuple[int, ...],
        max_images: int | None = None,
    ):
        if batch_size < 1:
            raise OptimizeError("--calib-batch must be at least 1.")
        if max_images is not None and max_images < 1:
            raise OptimizeError("--calib-max-images must be at least 1.")
        self.batch_size = batch_size
        self.sample_shape = tuple(int(dim) for dim in sample_shape)
        self.max_images = max_images
        self._count = _product(self.sample_shape)
        root = Path(data_dir)
        if not root.is_dir():
            raise OptimizeError(f"Calibration directory does not exist: {root}")
        files = _sample_files(root, self.sample_shape)
        available = len(files)
        if max_images is not None:
            files = files[:max_images]
        self._files = files
        if len(self._files) < batch_size:
            if max_images is not None and available >= batch_size:
                raise OptimizeError(
                    f"--calib-max-images {max_images} leaves {len(self._files)} "
                    f"samples; --calib-batch is {batch_size}."
                )
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


def declared_calib_shape(data_dir: str | Path) -> tuple[int, ...]:
    """Return the sample shape declared by ``calib.json``."""
    path = Path(data_dir) / "calib.json"
    if not path.is_file():
        raise OptimizeError(
            f"{data_dir} needs calib.json to supply the calibration shape."
        )
    return _read_meta(path)


def load_calibration_source(
    data_dir: str | Path,
    *,
    batch_size: int,
    sample_shape: tuple[int, ...],
    max_images: int | None = None,
) -> DirectoryBatchSource:
    """Read ``data_dir`` into a batch source for the entropy calibrator.

    ``max_images`` keeps the first N samples (``--calib-max-images``). ``None``
    reads every sample in the directory.
    """
    return DirectoryBatchSource(
        data_dir, batch_size, sample_shape, max_images=max_images
    )


def make_entropy_calibrator(
    source: BatchSource,
    *,
    batch_size: int,
    cache_path: str | Path | None,
    trt_module: object,
    copy_to_device=None,
    model_sha256: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    quantization: str = "implicit",
):
    """Return an ``IInt8EntropyCalibrator2`` bound to ``source``.

    ``copy_to_device`` turns a host ``array.array`` into a device pointer
    and keeps the device buffer alive on the list it is given. When it is
    omitted, ``get_batch`` returns a host pointer. Tests use the host
    pointer. A real TensorRT build passes a CUDA copy.

    When ``model_sha256`` is set, ``cache_path`` is reused only if the sidecar
    records that digest and the same preprocessing config.
    """
    if batch_size < 1:
        raise OptimizeError("--calib-batch must be at least 1.")
    identity = None
    if model_sha256 is not None:
        if cache_path is None:
            raise OptimizeError(
                "An INT8 calibration cache path is required to record the model sha256."
            )
        identity = CalibrationCache(
            cache_path,
            model_sha256=model_sha256,
            preprocessing=preprocessing or {},
            quantization=quantization,
        )
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
            self._identity = identity
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
            if self._identity is not None:
                return self._identity.read()
            if self._cache is None or not self._cache.is_file():
                return None
            data = self._cache.read_bytes()
            return data or None

        def write_calibration_cache(self, cache) -> None:
            payload = bytes(cache)
            if self._identity is not None:
                self._identity.write(payload)
                return
            if self._cache is None:
                return
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_bytes(payload)

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


def cache_sidecar_path(cache_path: str | Path) -> Path:
    """JSON sidecar stored next to a calibration cache."""
    path = Path(cache_path)
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def canonical_preprocessing(raw: dict[str, Any]) -> dict[str, Any]:
    """Stable preprocessing document stored in the cache sidecar."""
    if not isinstance(raw, dict):
        raise OptimizeError("Calibration preprocessing must be an object.")
    shape = raw.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or not shape
        or not all(type(dim) is int and dim >= 1 for dim in shape)
    ):
        raise OptimizeError(
            "Calibration preprocessing shape must be a list of positive integers."
        )
    dtype = raw.get("dtype", "float32")
    if dtype != "float32":
        raise OptimizeError("Calibration preprocessing dtype must be float32.")
    max_images = raw.get("max_images")
    if max_images is not None and (type(max_images) is not int or max_images < 1):
        raise OptimizeError("--calib-max-images must be at least 1.")
    submodule = raw.get("submodule")
    if submodule is not None and not isinstance(submodule, str):
        raise OptimizeError("Calibration preprocessing submodule must be a string.")
    document: dict[str, Any] = {
        "dtype": "float32",
        "max_images": max_images,
        "shape": [int(dim) for dim in shape],
        "submodule": submodule,
    }
    for key in ("mean", "std"):
        if key not in raw or raw[key] is None:
            continue
        document[key] = _number_list(raw[key], f"preprocessing.{key}")
    if raw.get("resize") is not None:
        resize = raw["resize"]
        if (
            not isinstance(resize, (list, tuple))
            or len(resize) != 2
            or not all(type(dim) is int and dim >= 1 for dim in resize)
        ):
            raise OptimizeError("preprocessing.resize must be [height, width].")
        document["resize"] = [int(dim) for dim in resize]
    if raw.get("layout") is not None:
        layout = raw["layout"]
        if layout not in _LAYOUTS:
            raise OptimizeError("preprocessing.layout must be nchw or nhwc.")
        document["layout"] = layout
    return document


def preprocessing_config(
    *,
    shape: tuple[int, ...],
    dtype: str = "float32",
    max_images: int | None = None,
    submodule: str | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    resize: list[int] | None = None,
    layout: str | None = None,
) -> dict[str, Any]:
    """Build the preprocessing document recorded beside a calibration cache."""
    raw: dict[str, Any] = {
        "dtype": dtype,
        "shape": list(shape),
        "max_images": max_images,
        "submodule": submodule,
    }
    if mean is not None:
        raw["mean"] = mean
    if std is not None:
        raw["std"] = std
    if resize is not None:
        raw["resize"] = resize
    if layout is not None:
        raw["layout"] = layout
    return canonical_preprocessing(raw)


def preprocessing_config_from_dir(
    data_dir: str | Path | None,
    *,
    shape: tuple[int, ...],
    max_images: int | None,
    submodule: str | None,
) -> dict[str, Any]:
    """Merge ``calib.json`` preprocessing with the model shape and the CLI cap."""
    mean = None
    std = None
    resize = None
    layout = None
    dtype = "float32"
    if data_dir is not None:
        meta_path = Path(data_dir) / "calib.json"
        if meta_path.is_file():
            data = _load_calib_json(meta_path)
            dtype = str(data.get("dtype") or "float32")
            extra = data.get("preprocessing") or {}
            mean = extra.get("mean")
            std = extra.get("std")
            resize = extra.get("resize")
            layout = extra.get("layout")
    return preprocessing_config(
        shape=shape,
        dtype=dtype,
        max_images=max_images,
        submodule=submodule,
        mean=None if mean is None else [float(item) for item in mean],
        std=None if std is None else [float(item) for item in std],
        resize=None if resize is None else [int(item) for item in resize],
        layout=layout,
    )


class CalibrationCache:
    """TensorRT cache plus a sidecar of model sha256 and preprocessing.

    ``read`` returns the cache bytes when both keys match. A different model
    digest or preprocessing config logs an invalidation and returns ``None``
    so the build calibrates again. ``write`` replaces the cache and the sidecar
    together.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        model_sha256: str,
        preprocessing: dict[str, Any],
        quantization: str,
    ):
        self.path = Path(path)
        self.sidecar = cache_sidecar_path(self.path)
        self.model_sha256 = _check_sha256(model_sha256)
        self.preprocessing = canonical_preprocessing(preprocessing)
        if quantization not in {"implicit", "explicit"}:
            raise OptimizeError(
                f"Calibration quantization must be implicit or explicit, not {quantization!r}."
            )
        self.quantization = quantization
        self.reason = ""

    def evaluate(self) -> str:
        """Return ``missing``, ``reusable``, or ``invalid``. Sets ``reason``."""
        self.reason = ""
        if not self.path.is_file():
            return "missing"
        if self.path.stat().st_size < 1:
            self.reason = "the cache file is empty"
            return "invalid"
        if not self.sidecar.is_file():
            self.reason = (
                "no sidecar recording the model sha256 and preprocessing config"
            )
            return "invalid"
        try:
            data = json.loads(self.sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.reason = "the sidecar is not valid JSON"
            return "invalid"
        if not isinstance(data, dict):
            self.reason = "the sidecar is not an object"
            return "invalid"
        if data.get("model_sha256") != self.model_sha256:
            self.reason = "model sha256 changed"
            return "invalid"
        try:
            stored = canonical_preprocessing(data.get("preprocessing") or {})
        except OptimizeError:
            self.reason = "preprocessing config changed"
            return "invalid"
        if stored != self.preprocessing:
            self.reason = "preprocessing config changed"
            return "invalid"
        if data.get("quantization") != self.quantization:
            self.reason = "quantization mode changed"
            return "invalid"
        return "reusable"

    def read(self) -> bytes | None:
        """Return cache bytes, logging reuse or invalidation."""
        status = self.evaluate()
        if status == "reusable":
            logger.info(
                "Reusing INT8 calibration cache %s (model sha256 %s, preprocessing unchanged).",
                self.path,
                self.model_sha256,
            )
            return self.path.read_bytes()
        if status == "invalid":
            logger.info(
                "INT8 calibration cache %s invalidated: %s.",
                self.path,
                self.reason,
            )
        return None

    def write(self, cache: bytes, *, activation_scale: float | None = None) -> None:
        """Write the cache and the sha256 / preprocessing sidecar."""
        if not cache:
            raise OptimizeError("Refusing to write an empty INT8 calibration cache.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(cache)
        document: dict[str, Any] = {
            "model_sha256": self.model_sha256,
            "preprocessing": self.preprocessing,
            "quantization": self.quantization,
        }
        if activation_scale is not None:
            document["activation_scale"] = float(activation_scale)
        self.sidecar.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Wrote INT8 calibration cache %s (model sha256 %s).",
            self.path,
            self.model_sha256,
        )

    def activation_scale(self) -> float | None:
        """Scale stored for an explicit Q/DQ rebuild, if the sidecar has one."""
        if not self.sidecar.is_file():
            return None
        try:
            data = json.loads(self.sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        value = data.get("activation_scale")
        if type(value) in (int, float) and not isinstance(value, bool):
            return float(value)
        return None


def read_all_samples(source: BatchSource) -> array.array:
    """Concatenate every batch from ``source`` into one float32 array."""
    flat = array.array("f")
    for index in range(source.nbatches):
        flat.extend(source.batch(index))
    return flat


def entropy_scale(values: array.array) -> float:
    """Symmetric INT8 scale from an entropy (KL) histogram of ``values``.

    The threshold is the absolute cutoff that minimises KL divergence between
    the sample histogram and an 8-bit quantization of it. The scale is that
    threshold divided by 127. An all-zero sample uses scale ``1`` so a Q/DQ
    node still has a positive scale.
    """
    if not isinstance(values, array.array) or values.typecode != "f":
        raise OptimizeError("Entropy calibration expects a float32 array.")
    if len(values) < 1:
        raise OptimizeError("Calibration batches are empty.")
    amax = 0.0
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise OptimizeError("Calibration samples must be finite float32 values.")
        magnitude = abs(number)
        if magnitude > amax:
            amax = magnitude
    if amax == 0.0:
        return 1.0
    bins = 2048
    counts = [0] * bins
    for value in values:
        index = int(abs(float(value)) / amax * bins)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    threshold = _kl_threshold(counts, amax)
    if threshold <= 0.0 or not math.isfinite(threshold):
        return 1.0
    return threshold / 127.0


def sha256_path(path: Path) -> str:
    """sha256 of a file, or of a directory's sorted relative paths and bytes."""
    digest = hashlib.sha256()
    if path.is_file():
        _hash_file(digest, path)
        return digest.hexdigest()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            _hash_file(digest, item)
        return digest.hexdigest()
    raise OptimizeError(f"Cannot hash model path {path}.")


def _kl_threshold(counts: list[int], amax: float) -> float:
    """Absolute threshold whose 128-bin quantization has the smallest KL."""
    bins = len(counts)
    quant_bins = 128
    if bins <= quant_bins:
        return amax
    best_kl = float("inf")
    best_index = bins - 1
    for end in range(quant_bins, bins):
        reference = counts[:end]
        reference[-1] += sum(counts[end:])
        total = sum(reference)
        if total < 1:
            continue
        step = end / quant_bins
        expanded = [0.0] * end
        for quant in range(quant_bins):
            begin = int(quant * step)
            stop = end if quant == quant_bins - 1 else int((quant + 1) * step)
            if stop <= begin:
                stop = begin + 1
            share = sum(reference[begin:stop]) / (stop - begin)
            for offset in range(begin, min(stop, end)):
                expanded[offset] = share
        kl = 0.0
        for observed, approx in zip(reference, expanded):
            if observed <= 0:
                continue
            probability = observed / total
            approx_p = max(approx / total, 1e-12)
            kl += probability * math.log(probability / approx_p)
        if kl < best_kl:
            best_kl = kl
            best_index = end - 1
    return amax * (best_index + 1) / bins


def _hash_file(digest, path: Path) -> None:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return
            digest.update(chunk)


def _check_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OptimizeError("Model sha256 must be 64 lowercase hex characters.")
    return value


def _number_list(value: object, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise OptimizeError(f"{label} must be a non-empty list of numbers.")
    numbers: list[float] = []
    for item in value:
        if type(item) not in (int, float) or isinstance(item, bool):
            raise OptimizeError(f"{label} must be a non-empty list of numbers.")
        number = float(item)
        if not math.isfinite(number):
            raise OptimizeError(f"{label} must be finite.")
        numbers.append(round(number, 8))
    return numbers


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
    data = _load_calib_json(path)
    shape = data.get("shape")
    return tuple(shape)


def _load_calib_json(path: Path) -> dict[str, Any]:
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
        or not all(type(dim) is int and dim >= 1 for dim in shape)
    ):
        raise OptimizeError(f"{path.name} shape must be a list of positive integers.")
    _validate_preprocess(data.get("preprocessing"), path.name)
    return data


def _validate_preprocess(raw: object, filename: str) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise OptimizeError(f"{filename} preprocessing must be an object.")
    extra = set(raw) - _PREPROCESS_FIELDS
    if extra:
        raise OptimizeError(
            f"{filename} preprocessing has unknown fields: {', '.join(sorted(extra))}."
        )
    for key in ("mean", "std"):
        if key in raw:
            _number_list(raw[key], f"{filename} preprocessing.{key}")
    if "resize" in raw:
        resize = raw["resize"]
        if (
            not isinstance(resize, list)
            or len(resize) != 2
            or not all(type(dim) is int and dim >= 1 for dim in resize)
        ):
            raise OptimizeError(
                f"{filename} preprocessing.resize must be [height, width]."
            )
    if "layout" in raw and raw["layout"] not in _LAYOUTS:
        raise OptimizeError(f"{filename} preprocessing.layout must be nchw or nhwc.")


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
