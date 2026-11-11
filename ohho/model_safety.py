"""Safe loading of untrusted model files (OHH-485).

Prefer ``.safetensors`` and ``.onnx``. A ``.pt`` / ``.bin`` pickle is never
unpickled with the stdlib unpickler. Before ``torch.load(..., weights_only=True)``
the pickle opcodes are scanned with ``pickletools`` and refused when they use a
dangerous global (``os``, ``subprocess``, ``builtins.exec`` / ``eval``, and the
Windows ``nt`` aliases) or any ``GLOBAL`` / ``REDUCE`` outside the weights
allowlist.

Managed workers (``OHHO_MANAGED_WORKER=1``) refuse a ``.pt`` / ``.bin`` file
unless that scan passes and ``weights_only=True`` succeeds. There is no
unrestricted pickle fallback.

``prepare_model_load`` is the hook serve, train, and optimize call before
reading a user checkpoint. This tree does not edit ``ohho.optimize`` (OHH-429);
that package should call ``prepare_model_load`` before it opens a file.

The scanner does not call ``pickle.loads``, ``pickle.Unpickler``, or
``torch.jit.load``.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickletools

MANAGED_WORKER_ENV = "OHHO_MANAGED_WORKER"

_MAX_PICKLE_BYTES = 32 * 1024 * 1024
_MAX_OPCODES = 2_000_000
_MAX_STACK = 100_000
_MAX_MARK = 64
_MAX_MEMO = 1_000_000
_MAX_STRING = 1_000_000
_MAX_EMBEDDED_BYTES = 1_000_000
_MAX_ZIP_MEMBERS = 100_000
_MAX_SAFETENSORS_HEADER = 100_000_000

_PICKLE_SUFFIXES = frozenset({".pt", ".pth", ".bin", ".pkl", ".pickle"})
_WEIGHT_SUFFIXES = _PICKLE_SUFFIXES | frozenset({".safetensors", ".onnx"})

_DANGEROUS_MODULES = frozenset(
    {
        "os",
        "nt",
        "posix",
        "subprocess",
        "sys",
        "builtins",
        "builtin",
        "__builtin__",
        "commands",
        "pty",
        "shutil",
        "socket",
        "ctypes",
        "multiprocessing",
        "importlib",
        "pickle",
        "code",
        "codeop",
        "runpy",
        "webbrowser",
    }
)
_DANGEROUS_NAMES = frozenset(
    {
        "eval",
        "exec",
        "execfile",
        "system",
        "popen",
        "compile",
        "__import__",
    }
)

_STORAGE_NAMES = frozenset(
    {
        "FloatStorage",
        "DoubleStorage",
        "HalfStorage",
        "LongStorage",
        "IntStorage",
        "ShortStorage",
        "CharStorage",
        "ByteStorage",
        "BoolStorage",
        "BFloat16Storage",
        "ComplexFloatStorage",
        "ComplexDoubleStorage",
        "QUInt8Storage",
        "QInt8Storage",
        "QInt32Storage",
        "QUInt4x2Storage",
        "QUInt2x4Storage",
        "UntypedStorage",
        "TypedStorage",
        "_TypedStorage",
    }
)

_REBUILDERS = frozenset(
    {
        ("collections", "OrderedDict"),
        ("torch", "Size"),
        ("torch", "Tensor"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._tensor", "_rebuild_from_type_v2"),
    }
)


def _allowed_globals() -> frozenset[tuple[str, str]]:
    pairs = set(_REBUILDERS)
    for name in _STORAGE_NAMES:
        pairs.add(("torch", name))
        pairs.add(("torch.storage", name))
    return frozenset(pairs)


ALLOWED_GLOBALS = _allowed_globals()

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


class UnsafeModelFile(ValueError):
    """A model file failed the safe-load checks. The message is the reason."""


class ModelRuntimeUnavailable(RuntimeError):
    """The optional runtime for this format (torch or onnx) is not installed."""


class _NotAPickle(Exception):
    """The bytes are not a complete pickle stream."""


class _Rejected(Exception):
    def __init__(self, reason: str, *, serious: bool) -> None:
        super().__init__(reason)
        self.serious = serious


@dataclass
class _Node:
    kind: str
    text: str | None = None
    number: int | None = None
    module: str | None = None
    name: str | None = None
    elems: tuple[_Node, ...] = ()


@dataclass
class _State:
    stack: list[_Node]
    metastack: list[list[_Node]]
    memo: dict[int, _Node]
    proto: int | None = None
    serious: bool = False
    stopped: bool = False
    opcodes: int = 0

    def push(self, node: _Node) -> None:
        if len(self.stack) > _MAX_STACK:
            raise _Rejected("pickle stack exceeds the scan limit", serious=True)
        self.stack.append(node)

    def pop(self) -> _Node:
        if not self.stack:
            raise _Rejected("pickle stack underflow", serious=self.serious)
        return self.stack.pop()

    def mark(self) -> None:
        if len(self.metastack) > _MAX_MARK:
            raise _Rejected("pickle mark stack exceeds the scan limit", serious=True)
        self.metastack.append(self.stack)
        self.stack = []

    def pop_mark(self) -> list[_Node]:
        if not self.metastack:
            raise _Rejected("pickle MARK underflow", serious=self.serious)
        items = self.stack
        self.stack = self.metastack.pop()
        return items


@dataclass(frozen=True)
class TensorBlob:
    """One tensor from a safetensors file. ``data`` is the raw little-endian bytes."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass
class LoadedWeights:
    """Weights read by :func:`load_weights`.

    ``format`` is ``safetensors``, ``onnx``, or ``torch``. ``state_dict`` is set
    for a weights-only torch checkpoint. ``onnx_model`` is the parsed ONNX model
    when the ``onnx`` package is installed. ``device`` is the device actually
    used; a missing GPU becomes ``cpu`` and ``device_note`` says why.
    """

    format: str
    source: Path
    tensors: tuple[TensorBlob, ...] = ()
    state_dict: Any = None
    onnx_model: Any = None
    device: str = "cpu"
    device_note: str = ""


def managed_worker_mode(explicit: bool | None = None) -> bool:
    """True when this process is a managed cloud worker.

    ``explicit`` overrides the environment. Otherwise
    ``OHHO_MANAGED_WORKER`` is true for ``1``, ``true``, ``yes``, or ``on``.
    """
    if explicit is not None:
        return explicit
    return os.environ.get(MANAGED_WORKER_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def scan_pickle_bytes(data: bytes) -> None:
    """Scan a pickle payload. Raise :class:`UnsafeModelFile` when it is unsafe.

    Raise :class:`_NotAPickle` when the bytes are not a pickle. This function
    does not execute the pickle.
    """
    if len(data) > _MAX_PICKLE_BYTES:
        raise UnsafeModelFile(
            "pickle payload exceeds the scan limit; use a .safetensors file"
        )
    try:
        _walk_pickle(data)
    except _Rejected as exc:
        if exc.serious:
            raise UnsafeModelFile(str(exc)) from None
        raise _NotAPickle(str(exc)) from None


def ensure_torch_weights_allowed(
    path: str | Path, *, managed: bool | None = None
) -> None:
    """Scan a ``.pt`` / ``.bin`` file before ``torch.load(weights_only=True)``.

    A zip checkpoint is scanned via its ``.pkl`` members only. Bytes that are
    not a pickle are ignored outside managed mode so a caller can still attempt
    ``weights_only``. Managed mode refuses them. A dangerous pickle is always
    refused, and this function never unpickles it.
    """
    file = Path(path)
    if not file.is_file():
        raise UnsafeModelFile(f"model file not found: {file}")
    is_managed = managed_worker_mode(managed)
    with file.open("rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"PK":
        _scan_zip_checkpoint(file, managed=is_managed)
        return
    claimed = len(magic) >= 2 and magic[0] == 0x80 and magic[1] <= 5
    size = file.stat().st_size
    if size > _MAX_PICKLE_BYTES:
        raise UnsafeModelFile(
            "refusing a large raw .pt/.bin pickle; use .safetensors or .onnx"
        )
    try:
        scan_pickle_bytes(file.read_bytes())
    except _NotAPickle:
        if claimed or is_managed:
            raise UnsafeModelFile(
                "refusing this .pt/.bin file because it is not a scannable "
                "weights-only checkpoint"
            ) from None
        return
    except UnsafeModelFile:
        raise


def prepare_model_load(path: str | Path, *, managed: bool | None = None) -> None:
    """Refuse unsafe local weight files. Missing paths (hub ids) are ignored.

    Serve and train call this before opening a checkpoint. Optimize should call
    it too; that wiring is a follow-up while OHH-429 owns ``ohho.optimize``.
    """
    candidate = Path(path).expanduser()
    try:
        exists = candidate.exists()
    except OSError as exc:
        raise UnsafeModelFile(f"model path cannot be read: {exc}") from exc
    if not exists:
        return
    for file in _iter_weight_files(candidate):
        suffix = file.suffix.lower()
        if suffix in _PICKLE_SUFFIXES:
            ensure_torch_weights_allowed(file, managed=managed)
        elif suffix == ".safetensors":
            _preflight_safetensors(file)
        elif suffix == ".onnx":
            _reject_disguised_pickle(file, label="ONNX")


def guard_runtime_checkpoint(
    path: str | Path, *, kind: str, managed: bool | None = None
) -> None:
    """Check a locomotion checkpoint before the policy runtime loads it.

    ONNX bytes that are actually a pickle are refused. TorchScript ``.pt``
    files stay on ``torch.jit.load`` outside managed mode, because they are
    not weights-only state dicts. Managed workers refuse them.
    """
    file = Path(path)
    if kind == "onnx":
        _reject_disguised_pickle(file, label="ONNX")
        return
    if kind == "torchscript" and managed_worker_mode(managed):
        raise UnsafeModelFile(
            "managed workers refuse TorchScript .pt pickle archives; "
            "use .onnx or a weights-only state dict "
            "(safetensors, or torch.load with weights_only=True after the opcode scan)"
        )


def load_weights(
    path: str | Path, *, device: str = "cpu", managed: bool | None = None
) -> LoadedWeights:
    """Load safetensors, ONNX, or a weights-only torch checkpoint.

    ``.safetensors`` is parsed with the standard library. ``.onnx`` is parsed
    with the optional ``onnx`` package and is not executed. ``.pt`` / ``.bin``
    go through :func:`ensure_torch_weights_allowed` and then
    ``torch.load(..., weights_only=True)`` only.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        source = _prefer_weight_file(source)
    if not source.is_file():
        raise UnsafeModelFile(f"model file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".safetensors":
        try:
            loaded = _read_safetensors(source, read_data=True)
        except UnsafeModelFile:
            reason = _pickle_refusal(source)
            if reason:
                raise UnsafeModelFile(reason) from None
            raise
        used, note = _resolve_load_device(device)
        loaded.device = used
        loaded.device_note = note
        return loaded
    if suffix == ".onnx":
        return load_onnx(source, device=device)
    if suffix in _PICKLE_SUFFIXES:
        state, used, note = load_torch_state_dict(
            source, device=device, managed=managed
        )
        return LoadedWeights(
            format="torch",
            source=source,
            state_dict=state,
            device=used,
            device_note=note,
        )
    raise UnsafeModelFile(
        f"unsupported model file '{suffix or source.name}'; prefer .safetensors or .onnx"
    )


def load_torch_state_dict(
    path: str | Path, *, device: str = "cpu", managed: bool | None = None
) -> tuple[Any, str, str]:
    """Return ``(state_dict, device, note)`` from ``torch.load(weights_only=True)``.

    Managed mode turns a ``weights_only`` failure into :class:`UnsafeModelFile`
    and does not retry without that flag.
    """
    file = Path(path)
    ensure_torch_weights_allowed(file, managed=managed)
    try:
        import torch
    except ImportError as exc:
        raise ModelRuntimeUnavailable(
            "torch is not installed; cannot load a weights-only checkpoint. "
            "Prefer a .safetensors file, or pip install 'ohho-os[train]'."
        ) from exc
    used, note = _resolve_load_device(device)
    try:
        state = torch.load(file, map_location=used, weights_only=True)
    except Exception as exc:
        if managed_worker_mode(managed):
            raise UnsafeModelFile(
                "managed workers refuse this .pt/.bin file because "
                "torch.load(weights_only=True) failed: " + _short_error(exc)
            ) from exc
        raise
    return state, used, note


def load_onnx(path: str | Path, *, device: str = "cpu") -> LoadedWeights:
    """Parse an ONNX file. A pickle disguised as ``.onnx`` is refused."""
    file = Path(path)
    _reject_disguised_pickle(file, label="ONNX")
    used, note = _resolve_load_device(device)
    try:
        import onnx
    except ImportError as exc:
        raise ModelRuntimeUnavailable(
            "onnx is not installed; pip install 'ohho-os[humanoid-sim]'"
        ) from exc
    model = onnx.load(str(file))
    return LoadedWeights(
        format="onnx",
        source=file,
        onnx_model=model,
        device=used,
        device_note=note,
    )


def _short_error(exc: BaseException) -> str:
    line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    line = line.replace("\n", " ")
    if len(line) > 200:
        return line[:200] + "..."
    return line


def _resolve_load_device(device: str) -> tuple[str, str]:
    """Use ``cpu`` when the requested GPU runtime is not available."""
    if device in ("", "auto", "cpu"):
        return "cpu", ""
    try:
        import torch
    except ImportError:
        return "cpu", f"{device} is unavailable; loaded on cpu"
    if device.startswith("cuda"):
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return "cpu", f"{device} is unavailable; loaded on cpu"
        return device, ""
    if device == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is None or not mps.is_available():
            return "cpu", "mps is unavailable; loaded on cpu"
        return device, ""
    return "cpu", f"{device} is not attached by this loader; loaded on cpu"


def _iter_weight_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            return [path]
        return []
    if not path.is_dir():
        return []
    found: list[Path] = []
    for child in path.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() in _WEIGHT_SUFFIXES:
            found.append(child)
    return found


def _prefer_weight_file(root: Path) -> Path:
    files = _iter_weight_files(root)
    if not files:
        raise UnsafeModelFile(
            "no safetensors, ONNX, or weights file in the model directory"
        )
    order = (".safetensors", ".onnx", ".pt", ".pth", ".bin", ".pkl", ".pickle")
    by_suffix: dict[str, list[Path]] = {}
    for file in files:
        by_suffix.setdefault(file.suffix.lower(), []).append(file)
    for suffix in order:
        group = sorted(by_suffix.get(suffix, []), key=lambda item: item.name)
        if not group:
            continue
        for preferred in (f"model{suffix}", f"policy{suffix}"):
            for file in group:
                if file.name == preferred:
                    return file
        return group[0]
    raise UnsafeModelFile(
        "no safetensors, ONNX, or weights file in the model directory"
    )


def _preflight_safetensors(path: Path) -> None:
    try:
        _read_safetensors(path, read_data=False)
    except UnsafeModelFile:
        reason = _pickle_refusal(path)
        if reason:
            raise UnsafeModelFile(reason) from None
        return


def _read_safetensors(path: Path, *, read_data: bool) -> LoadedWeights:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise UnsafeModelFile("safetensors file is truncated")
        header_len = int.from_bytes(prefix, "little")
        if header_len <= 0 or header_len > _MAX_SAFETENSORS_HEADER:
            raise UnsafeModelFile("safetensors header length is invalid")
        if 8 + header_len > file_size:
            raise UnsafeModelFile("safetensors header is truncated")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise UnsafeModelFile("safetensors header is truncated")
        header = _parse_header(header_bytes)
        data_len = file_size - 8 - header_len
        metas = _tensor_metas(header, data_len)
        blob = b""
        if read_data and data_len:
            blob = handle.read(data_len)
            if len(blob) != data_len:
                raise UnsafeModelFile("safetensors data is truncated")
    tensors: list[TensorBlob] = []
    for name, dtype, shape, start, end in metas:
        data = blob[start:end] if read_data else b""
        tensors.append(TensorBlob(name=name, dtype=dtype, shape=shape, data=data))
    return LoadedWeights(format="safetensors", source=path, tensors=tuple(tensors))


def _parse_header(header_bytes: bytes) -> dict[str, Any]:
    def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        out: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str):
                raise UnsafeModelFile("safetensors header keys must be strings")
            if key in seen and key != "__metadata__":
                raise UnsafeModelFile(f"duplicate safetensors tensor {key}")
            seen.add(key)
            out[key] = value
        return out

    try:
        header = json.loads(header_bytes.decode("utf-8"), object_pairs_hook=_pairs)
    except UnsafeModelFile:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsafeModelFile("safetensors header is not UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise UnsafeModelFile("safetensors header must be a JSON object")
    return header


def _tensor_metas(
    header: dict[str, Any], data_len: int
) -> list[tuple[str, str, tuple[int, ...], int, int]]:
    metas: list[tuple[str, str, tuple[int, ...], int, int]] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise UnsafeModelFile(f"safetensors tensor {name} is not an object")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise UnsafeModelFile(f"safetensors tensor {name} has an unknown dtype")
        if not isinstance(shape, list) or not isinstance(offsets, list):
            raise UnsafeModelFile(f"safetensors tensor {name} has an invalid layout")
        if len(offsets) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) for item in offsets
        ):
            raise UnsafeModelFile(f"safetensors tensor {name} has invalid data_offsets")
        dims: list[int] = []
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise UnsafeModelFile(f"safetensors tensor {name} has an invalid shape")
            dims.append(dim)
        start, end = offsets
        if start < 0 or end < start or end > data_len:
            raise UnsafeModelFile(f"safetensors tensor {name} offsets are out of range")
        expect = _DTYPE_BYTES[dtype]
        for dim in dims:
            expect *= dim
            if expect > data_len and dim != 0:
                raise UnsafeModelFile(
                    f"safetensors tensor {name} is larger than the file"
                )
        if end - start != expect:
            raise UnsafeModelFile(
                f"safetensors tensor {name} size does not match dtype and shape"
            )
        metas.append((name, dtype, tuple(dims), start, end))
    return metas


def _reject_disguised_pickle(path: Path, *, label: str) -> None:
    reason = _pickle_refusal(path)
    if reason:
        raise UnsafeModelFile(f"{label} file: {reason}")


def _pickle_refusal(path: Path) -> str | None:
    """Return a refusal reason when ``path`` is a pickle, else None."""
    try:
        with path.open("rb") as handle:
            magic = handle.read(8)
    except OSError:
        return None
    if magic[:2] == b"PK":
        return _zip_pickle_refusal(path)
    if not (len(magic) >= 2 and magic[0] == 0x80 and magic[1] <= 5):
        return None
    size = path.stat().st_size
    if size > _MAX_PICKLE_BYTES:
        return "pickle payload exceeds the scan limit; use a .safetensors file"
    try:
        scan_pickle_bytes(path.read_bytes())
    except _NotAPickle:
        return None
    except UnsafeModelFile as exc:
        return str(exc)
    return "file is a pickle payload; refusing"


def _zip_pickle_refusal(path: Path) -> str | None:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return None
    with archive:
        for info in archive.infolist():
            if not info.filename.lower().endswith((".pkl", ".pickle")):
                continue
            if info.file_size > _MAX_PICKLE_BYTES:
                return "pickle payload exceeds the scan limit; use a .safetensors file"
            try:
                scan_pickle_bytes(archive.read(info))
            except _NotAPickle:
                return "file is a pickle payload; refusing"
            except UnsafeModelFile as exc:
                return str(exc)
            return "file is a pickle payload; refusing"
    return None


def _scan_zip_checkpoint(path: Path, *, managed: bool) -> None:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        if managed:
            raise UnsafeModelFile(
                "refusing this .pt/.bin file because it is not a scannable "
                "weights-only checkpoint"
            ) from None
        return
    with archive:
        infos = archive.infolist()
        if len(infos) > _MAX_ZIP_MEMBERS:
            raise UnsafeModelFile("checkpoint archive has too many members")
        payloads: list[bytes] = []
        for info in infos:
            if info.flag_bits & 0x1:
                raise UnsafeModelFile("refusing an encrypted checkpoint archive")
            if not _zip_member_ok(info.filename):
                raise UnsafeModelFile(
                    f"refusing checkpoint archive member {info.filename!r}"
                )
            if not info.filename.lower().endswith((".pkl", ".pickle")):
                continue
            if info.file_size > _MAX_PICKLE_BYTES:
                raise UnsafeModelFile(
                    "pickle member exceeds the scan limit; use a .safetensors file"
                )
            payload = archive.read(info)
            if len(payload) > _MAX_PICKLE_BYTES:
                raise UnsafeModelFile(
                    "pickle member exceeds the scan limit; use a .safetensors file"
                )
            payloads.append(payload)
        if not payloads:
            if managed:
                raise UnsafeModelFile(
                    "refusing this archive because it has no weights pickle to scan; "
                    "use .safetensors or .onnx"
                )
            return
        for payload in payloads:
            try:
                scan_pickle_bytes(payload)
            except _NotAPickle:
                raise UnsafeModelFile(
                    "refusing a checkpoint pickle that could not be scanned"
                ) from None


def _zip_member_ok(name: str) -> bool:
    if not name or name.startswith(("/", "\\")):
        return False
    parts = Path(name).parts
    if ".." in parts:
        return False
    if parts and parts[0].endswith(":"):
        return False
    return True


def _walk_pickle(data: bytes) -> None:
    state = _State(stack=[], metastack=[], memo={})
    try:
        for opcode, arg, _pos in pickletools.genops(data):
            if state.stopped:
                raise _Rejected("trailing opcodes after STOP", serious=True)
            state.opcodes += 1
            if state.opcodes > _MAX_OPCODES:
                raise _Rejected("pickle exceeds the opcode scan limit", serious=True)
            handler = _HANDLERS.get(opcode.name)
            if handler is None:
                raise _Rejected(
                    f"refusing unknown pickle opcode {opcode.name}",
                    serious=state.serious or state.proto is not None,
                )
            handler(state, arg)
    except _Rejected:
        raise
    except (ValueError, IndexError, KeyError, EOFError, TypeError) as exc:
        raise _Rejected(
            f"unreadable pickle ({exc.__class__.__name__})",
            serious=state.serious,
        ) from exc
    if not state.stopped:
        raise _Rejected("pickle did not end with STOP", serious=state.serious)


def _split_global(arg: object) -> tuple[str, str]:
    if not isinstance(arg, str) or " " not in arg:
        raise _Rejected("refusing a malformed GLOBAL operand", serious=True)
    module, name = arg.split(" ", 1)
    if not module or not name:
        raise _Rejected("refusing a malformed GLOBAL operand", serious=True)
    return module, name


def _is_dangerous(module: str, name: str) -> bool:
    root = module.split(".", 1)[0]
    if root in _DANGEROUS_MODULES or module in _DANGEROUS_MODULES:
        return True
    return name in _DANGEROUS_NAMES


def _make_global(module: str, name: str) -> _Node:
    qual = f"{module}.{name}"
    if _is_dangerous(module, name):
        raise _Rejected(f"refusing dangerous global {qual}", serious=True)
    if (module, name) not in ALLOWED_GLOBALS:
        raise _Rejected(
            f"refusing global {qual}; it is outside the weights allowlist",
            serious=True,
        )
    return _Node(kind="global", module=module, name=name)


def _require_global(node: _Node) -> tuple[str, str]:
    if node.kind != "global" or not node.module or not node.name:
        raise _Rejected(
            "refusing a call through a value that is not an allowlisted global",
            serious=True,
        )
    if (node.module, node.name) not in ALLOWED_GLOBALS:
        raise _Rejected(
            f"refusing call of {node.module}.{node.name}; it is outside the weights allowlist",
            serious=True,
        )
    return node.module, node.name


def _reduced(module: str, name: str) -> _Node:
    if (module, name) == ("collections", "OrderedDict"):
        return _Node(kind="dict")
    return _Node(kind="object")


def _string(text: str) -> _Node:
    if len(text) > _MAX_STRING:
        raise _Rejected("pickle string exceeds the scan limit", serious=True)
    return _Node(kind="str", text=text)


def _literal(arg: object) -> _Node:
    if arg is None:
        return _Node(kind="none")
    if isinstance(arg, bool):
        return _Node(kind="bool")
    if isinstance(arg, int):
        return _Node(kind="int", number=arg)
    if isinstance(arg, float):
        return _Node(kind="float")
    if isinstance(arg, str):
        return _string(arg)
    if isinstance(arg, (bytes, bytearray)):
        if len(arg) > _MAX_EMBEDDED_BYTES:
            raise _Rejected("embedded pickle bytes exceed the scan limit", serious=True)
        return _Node(kind="bytes")
    raise _Rejected("unrecognized pickle literal", serious=True)


def _memo_put(state: _State, index: object) -> None:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index > _MAX_MEMO
    ):
        raise _Rejected("refusing pickle memo index", serious=True)
    if not state.stack:
        raise _Rejected("PUT on an empty stack", serious=state.serious)
    if len(state.memo) > _MAX_MEMO:
        raise _Rejected("pickle memo exceeds the scan limit", serious=True)
    state.memo[index] = state.stack[-1]


def _memo_get(state: _State, index: object) -> None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise _Rejected("refusing pickle memo index", serious=True)
    try:
        node = state.memo[index]
    except KeyError:
        raise _Rejected("pickle memo index is missing", serious=state.serious) from None
    state.push(node)


def _on_proto(state: _State, arg: object) -> None:
    if isinstance(arg, bool) or not isinstance(arg, int) or arg < 0 or arg > 5:
        raise _Rejected(f"refusing pickle protocol {arg}", serious=True)
    state.proto = arg
    state.serious = True


def _on_frame(state: _State, arg: object) -> None:
    del state, arg


def _on_stop(state: _State, arg: object) -> None:
    del arg
    if not state.stack:
        raise _Rejected("STOP with an empty stack", serious=state.serious)
    state.pop()
    state.stopped = True


def _on_mark(state: _State, arg: object) -> None:
    del arg
    state.mark()


def _on_pop(state: _State, arg: object) -> None:
    del arg
    if state.stack:
        state.pop()
    else:
        state.pop_mark()


def _on_pop_mark(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()


def _on_dup(state: _State, arg: object) -> None:
    del arg
    if not state.stack:
        raise _Rejected("DUP on an empty stack", serious=state.serious)
    state.push(state.stack[-1])


def _on_none(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="none"))


def _on_true(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="bool"))


def _on_false(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="bool"))


def _on_literal(state: _State, arg: object) -> None:
    state.push(_literal(arg))


def _on_empty_list(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="list"))


def _on_empty_dict(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="dict"))


def _on_empty_tuple(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="tuple", elems=()))


def _on_empty_set(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="set"))


def _on_tuple(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="tuple", elems=tuple(state.pop_mark())))


def _on_tuple1(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="tuple", elems=(state.pop(),)))


def _on_tuple2(state: _State, arg: object) -> None:
    del arg
    second = state.pop()
    first = state.pop()
    state.push(_Node(kind="tuple", elems=(first, second)))


def _on_tuple3(state: _State, arg: object) -> None:
    del arg
    third = state.pop()
    second = state.pop()
    first = state.pop()
    state.push(_Node(kind="tuple", elems=(first, second, third)))


def _on_list(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    state.push(_Node(kind="list"))


def _on_dict(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    state.push(_Node(kind="dict"))


def _on_frozenset(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    state.push(_Node(kind="object"))


def _container_top(state: _State, kinds: set[str], opcode: str) -> None:
    if not state.stack or state.stack[-1].kind not in kinds:
        raise _Rejected(
            f"refusing {opcode} on an unexpected object",
            serious=state.serious,
        )


def _on_append(state: _State, arg: object) -> None:
    del arg
    state.pop()
    _container_top(state, {"list", "object"}, "APPEND")


def _on_appends(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    _container_top(state, {"list", "object"}, "APPENDS")


def _on_setitem(state: _State, arg: object) -> None:
    del arg
    state.pop()
    state.pop()
    _container_top(state, {"dict", "object"}, "SETITEM")


def _on_setitems(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    _container_top(state, {"dict", "object"}, "SETITEMS")


def _on_additems(state: _State, arg: object) -> None:
    del arg
    state.pop_mark()
    _container_top(state, {"set", "object"}, "ADDITEMS")


def _on_put(state: _State, arg: object) -> None:
    _memo_put(state, arg)


def _on_get(state: _State, arg: object) -> None:
    _memo_get(state, arg)


def _on_memoize(state: _State, arg: object) -> None:
    del arg
    _memo_put(state, len(state.memo))


def _on_global(state: _State, arg: object) -> None:
    state.serious = True
    module, name = _split_global(arg)
    state.push(_make_global(module, name))


def _on_stack_global(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    name = state.pop()
    module = state.pop()
    if (
        name.kind != "str"
        or module.kind != "str"
        or name.text is None
        or module.text is None
    ):
        raise _Rejected("refusing STACK_GLOBAL with a non-literal name", serious=True)
    state.push(_make_global(module.text, name.text))


def _on_reduce(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    args = state.pop()
    if not state.stack:
        raise _Rejected("REDUCE missing callable", serious=True)
    module, name = _require_global(state.stack[-1])
    if args.kind != "tuple":
        raise _Rejected("refusing REDUCE whose argument is not a tuple", serious=True)
    state.stack[-1] = _reduced(module, name)


def _on_build(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    state.pop()
    _container_top(state, {"dict", "list", "set", "object", "global"}, "BUILD")


def _on_inst(state: _State, arg: object) -> None:
    state.serious = True
    module, name = _split_global(arg)
    _make_global(module, name)
    state.pop_mark()
    state.push(_reduced(module, name))


def _on_obj(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    args = state.pop_mark()
    if not args:
        raise _Rejected("OBJ missing class", serious=True)
    module, name = _require_global(args[0])
    state.push(_reduced(module, name))


def _on_newobj(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    args = state.pop()
    module, name = _require_global(state.pop())
    if args.kind != "tuple":
        raise _Rejected("NEWOBJ arguments are not a tuple", serious=True)
    state.push(_reduced(module, name))


def _on_newobj_ex(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    state.pop()
    args = state.pop()
    module, name = _require_global(state.pop())
    if args.kind != "tuple":
        raise _Rejected("NEWOBJ_EX arguments are not a tuple", serious=True)
    state.push(_reduced(module, name))


def _on_ext(state: _State, arg: object) -> None:
    del state, arg
    raise _Rejected(
        "refusing pickle extension opcode; the extension registry is outside the weights allowlist",
        serious=True,
    )


def _on_persid(state: _State, arg: object) -> None:
    del state, arg
    raise _Rejected("refusing persistent id opcode PERSID", serious=True)


def _on_binpersid(state: _State, arg: object) -> None:
    del arg
    state.serious = True
    pid = state.pop()
    if pid.kind != "tuple" or not pid.elems:
        raise _Rejected(
            "refusing persistent id that is not a torch storage record",
            serious=True,
        )
    kind = pid.elems[0]
    if kind.kind != "str" or kind.text != "storage":
        raise _Rejected(
            "refusing persistent id; only torch storage ids are allowed",
            serious=True,
        )
    if len(pid.elems) < 2:
        raise _Rejected("refusing persistent id with no storage type", serious=True)
    storage = pid.elems[1]
    if storage.kind != "global" or not storage.module or not storage.name:
        raise _Rejected(
            "refusing persistent id with an unknown storage type", serious=True
        )
    if (
        storage.name not in _STORAGE_NAMES
        or (storage.module, storage.name) not in ALLOWED_GLOBALS
    ):
        raise _Rejected(
            f"refusing persistent id for {storage.module}.{storage.name}",
            serious=True,
        )
    for extra in pid.elems[2:]:
        if extra.kind not in {"str", "int", "bool", "none"}:
            raise _Rejected(
                "refusing persistent id with a non-literal field",
                serious=True,
            )
    state.push(_Node(kind="object"))


def _on_next_buffer(state: _State, arg: object) -> None:
    del arg
    state.push(_Node(kind="bytes"))


def _on_readonly_buffer(state: _State, arg: object) -> None:
    del arg
    if not state.stack or state.stack[-1].kind not in {"bytes", "object"}:
        raise _Rejected(
            "READONLY_BUFFER on an unexpected object", serious=state.serious
        )


_HANDLERS: dict[str, Any] = {
    "PROTO": _on_proto,
    "FRAME": _on_frame,
    "STOP": _on_stop,
    "MARK": _on_mark,
    "POP": _on_pop,
    "POP_MARK": _on_pop_mark,
    "DUP": _on_dup,
    "NONE": _on_none,
    "NEWTRUE": _on_true,
    "NEWFALSE": _on_false,
    "INT": _on_literal,
    "BININT": _on_literal,
    "BININT1": _on_literal,
    "BININT2": _on_literal,
    "LONG": _on_literal,
    "LONG1": _on_literal,
    "LONG4": _on_literal,
    "FLOAT": _on_literal,
    "BINFLOAT": _on_literal,
    "STRING": _on_literal,
    "BINSTRING": _on_literal,
    "SHORT_BINSTRING": _on_literal,
    "BINBYTES": _on_literal,
    "SHORT_BINBYTES": _on_literal,
    "BINBYTES8": _on_literal,
    "BYTEARRAY8": _on_literal,
    "UNICODE": _on_literal,
    "SHORT_BINUNICODE": _on_literal,
    "BINUNICODE": _on_literal,
    "BINUNICODE8": _on_literal,
    "EMPTY_LIST": _on_empty_list,
    "EMPTY_DICT": _on_empty_dict,
    "EMPTY_TUPLE": _on_empty_tuple,
    "EMPTY_SET": _on_empty_set,
    "TUPLE": _on_tuple,
    "TUPLE1": _on_tuple1,
    "TUPLE2": _on_tuple2,
    "TUPLE3": _on_tuple3,
    "LIST": _on_list,
    "DICT": _on_dict,
    "FROZENSET": _on_frozenset,
    "APPEND": _on_append,
    "APPENDS": _on_appends,
    "SETITEM": _on_setitem,
    "SETITEMS": _on_setitems,
    "ADDITEMS": _on_additems,
    "PUT": _on_put,
    "BINPUT": _on_put,
    "LONG_BINPUT": _on_put,
    "GET": _on_get,
    "BINGET": _on_get,
    "LONG_BINGET": _on_get,
    "MEMOIZE": _on_memoize,
    "GLOBAL": _on_global,
    "STACK_GLOBAL": _on_stack_global,
    "REDUCE": _on_reduce,
    "BUILD": _on_build,
    "INST": _on_inst,
    "OBJ": _on_obj,
    "NEWOBJ": _on_newobj,
    "NEWOBJ_EX": _on_newobj_ex,
    "EXT1": _on_ext,
    "EXT2": _on_ext,
    "EXT4": _on_ext,
    "PERSID": _on_persid,
    "BINPERSID": _on_binpersid,
    "NEXT_BUFFER": _on_next_buffer,
    "READONLY_BUFFER": _on_readonly_buffer,
}


__all__ = [
    "ALLOWED_GLOBALS",
    "MANAGED_WORKER_ENV",
    "LoadedWeights",
    "ModelRuntimeUnavailable",
    "TensorBlob",
    "UnsafeModelFile",
    "ensure_torch_weights_allowed",
    "guard_runtime_checkpoint",
    "load_onnx",
    "load_torch_state_dict",
    "load_weights",
    "managed_worker_mode",
    "prepare_model_load",
    "scan_pickle_bytes",
]
