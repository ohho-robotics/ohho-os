"""Mixed-precision training for ``ohho train``.

Recipes name one of ``fp32``, ``fp16``, ``bf16``, or ``fp8``. The runner
checks that choice against the device before a family starts, and refuses
it with :class:`ohho.train.recipe_schema.RecipeConfigError` when the device
cannot run it. Nothing in this module rewrites the request to another
precision.

``fp16`` trains under ``torch.amp`` autocast with ``GradScaler`` loss
scaling, and only on CUDA. ``bf16`` uses autocast and no loss scaler.
``fp8`` uses Transformer Engine on a CUDA GPU of compute capability 8.9 or
newer. ``fp32`` is the default and runs on every device this package can
select.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .recipe_schema import RecipeConfigError

PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "fp8")
_FP16_CC = (5, 3)
_BF16_CC = (8, 0)
_FP8_CC = (8, 9)

_FP16_NEED = (
    "fp16 training needs a CUDA GPU and runs under torch.amp autocast "
    "with GradScaler loss scaling."
)
_BF16_NEED = (
    "bf16 training needs torch.amp autocast on a device that supports bfloat16."
)
_FP8_NEED = (
    "fp8 training needs Transformer Engine (transformer_engine) on a CUDA GPU "
    "with compute capability 8.9 or newer."
)
_NEEDS = {"fp16": _FP16_NEED, "bf16": _BF16_NEED, "fp8": _FP8_NEED}


class PrecisionUnsupported(RecipeConfigError):
    """The requested precision cannot run on this device.

    Raised before a trainer starts. The run does not fall back to another
    precision.
    """


@dataclass(frozen=True)
class DeviceCapabilities:
    """What one compute device can train.

    ``fp16`` is CUDA autocast with GradScaler. ``bf16`` is autocast on a
    device whose build can run bfloat16. ``fp8`` is Transformer Engine on
    CUDA compute capability 8.9 or newer. ``available`` is false when that
    device cannot be selected.
    """

    device: str
    kind: str
    bf16: bool
    fp16: bool
    fp8: bool
    available: bool = True
    name: str = ""
    detail: str = ""


@dataclass(frozen=True)
class PrecisionPlan:
    """Precision resolved for one run.

    ``loss_scaling`` is true only for fp16, where the step uses
    ``torch.amp.GradScaler``. ``backend`` is ``fp32``, ``torch.amp``, or
    ``transformer_engine``.
    """

    precision: str
    device: str
    loss_scaling: bool
    backend: str
    capabilities: DeviceCapabilities

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description of the plan, without the device probe."""
        return {
            "precision": self.precision,
            "device": self.device,
            "loss_scaling": self.loss_scaling,
            "backend": self.backend,
        }


def backend_for(precision: str) -> tuple[bool, str]:
    """Return ``(loss_scaling, backend)`` for a known precision name."""
    if precision == "fp16":
        return True, "torch.amp"
    if precision == "bf16":
        return False, "torch.amp"
    if precision == "fp8":
        return False, "transformer_engine"
    return False, "fp32"


def blank_precision_metrics(
    precision: str,
    throughput: int | float | None,
    peak: int | None,
) -> dict[str, Any]:
    """Metrics map with a slot for every precision.

    Only ``precision`` holds throughput and peak memory. The other slots
    are null because this run did not use them.
    """
    metrics: dict[str, Any] = {name: None for name in PRECISIONS}
    metrics[precision] = {
        "throughput_steps_per_s": throughput,
        "peak_memory_bytes": peak,
    }
    return metrics


def supported_precisions(family: Any) -> tuple[str, ...]:
    """Precisions a family accepts.

    Families that do not declare ``metadata()`` accept every shared
    precision. A non-empty ``metadata()["precisions"]`` list replaces that
    default so a plugin can opt out. The runner still checks the device.
    """
    meta = getattr(family, "metadata", None)
    if not callable(meta):
        return PRECISIONS
    supplied = meta()
    if not isinstance(supplied, dict):
        return PRECISIONS
    raw = supplied.get("precisions")
    if not isinstance(raw, (list, tuple)) or not raw:
        return PRECISIONS
    names = tuple(str(item) for item in raw)
    return names or PRECISIONS


def resolve_training_device(requested: str) -> str:
    """Resolve a recipe device to ``cpu``, ``cuda``, ``cuda:N``, or ``mps``.

    ``auto`` uses :func:`ohho.hardware.resolve_device`. CUDA and MPS need
    torch. A missing device raises :class:`RecipeConfigError` before
    training.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        from ohho.hardware import resolve_device

        resolved = resolve_device("auto")
        if resolved in {"cpu", "cuda", "mps"} or (
            isinstance(resolved, str) and resolved.startswith("cuda:")
        ):
            return resolved
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        torch = _require_torch()
        try:
            ok = bool(torch.cuda.is_available())
        except Exception:
            ok = False
        if not ok:
            raise RecipeConfigError("device 'cuda' is not available on this machine")
        return requested
    if requested == "mps":
        torch = _require_torch()
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        try:
            ok = mps is not None and bool(mps.is_available())
        except Exception:
            ok = False
        if not ok:
            raise RecipeConfigError("device 'mps' is not available on this machine")
        return "mps"
    raise RecipeConfigError(f"device {requested!r} is not available in this build")


def probe_capabilities(
    device: str,
    *,
    torch_module: Any | None = None,
    fp8_importable: bool | None = None,
    try_bf16: bool = True,
) -> DeviceCapabilities:
    """Probe ``device``.

    Pass ``torch_module`` and ``fp8_importable`` from tests so the check
    does not need a GPU. ``try_bf16`` skips the CPU autocast experiment
    when the run is not asking for bf16.
    """
    kind, index = _device_kind(device)
    torch = torch_module if torch_module is not None else _try_import_torch()
    if kind == "cpu":
        return _cpu_capabilities(torch, try_bf16=try_bf16)
    if kind == "mps":
        return _mps_capabilities(device, torch, try_bf16=try_bf16)
    if kind == "cuda":
        return _cuda_capabilities(
            device,
            index if index is not None else 0,
            torch,
            fp8_importable=fp8_importable,
        )
    return DeviceCapabilities(
        device=device,
        kind=kind,
        bf16=False,
        fp16=False,
        fp8=False,
        available=False,
        name="",
        detail="unrecognised device",
    )


def capabilities_for(
    device: str,
    precision: str,
    *,
    torch_module: Any | None = None,
    fp8_importable: bool | None = None,
) -> DeviceCapabilities:
    """Probe only the facts ``precision`` needs."""
    if fp8_importable is None and precision != "fp8":
        fp8_importable = False
    return probe_capabilities(
        device,
        torch_module=torch_module,
        fp8_importable=fp8_importable,
        try_bf16=(precision == "bf16"),
    )


def require_precision(
    precision: str, capabilities: DeviceCapabilities
) -> PrecisionPlan:
    """Return a plan, or raise when ``capabilities`` cannot run ``precision``.

    fp32 is accepted on every device. fp16, bf16, and fp8 are refused with
    a message that names the precision, the device, and the requirement.
    The message says that no fallback is applied.
    """
    if precision not in PRECISIONS:
        listed = ", ".join(PRECISIONS)
        raise RecipeConfigError(
            f"precision {precision!r} is not one of {listed}. No fallback is applied."
        )
    loss_scaling, backend = backend_for(precision)
    if precision == "fp32":
        return PrecisionPlan(
            precision="fp32",
            device=capabilities.device,
            loss_scaling=False,
            backend=backend,
            capabilities=capabilities,
        )
    if not capabilities.available:
        raise RecipeConfigError(
            f"device {capabilities.device!r} is not available on this machine"
        )
    supported = {
        "fp16": capabilities.fp16,
        "bf16": capabilities.bf16,
        "fp8": capabilities.fp8,
    }[precision]
    if not supported:
        raise PrecisionUnsupported(_unsupported(precision, capabilities))
    return PrecisionPlan(
        precision=precision,
        device=capabilities.device,
        loss_scaling=loss_scaling,
        backend=backend,
        capabilities=capabilities,
    )


def prepare_precision(
    recipe: Mapping[str, Any], *, family: Any | None = None
) -> PrecisionPlan:
    """Check a recipe before the family runs.

    When ``family`` is set, its registry metadata is applied first. The
    device is then resolved and probed. An unsupported precision raises
    before ``train_start``.
    """
    precision = _recipe_precision(recipe)
    if family is not None and precision not in supported_precisions(family):
        family_id = str(getattr(family, "id", ""))
        allowed = ", ".join(supported_precisions(family))
        raise RecipeConfigError(
            f"precision {precision!r} is not available for family {family_id!r}. "
            f"This family accepts {allowed}."
        )
    device = resolve_training_device(_recipe_device(recipe))
    return require_precision(precision, capabilities_for(device, precision))


def smoke_train(precision: str, device: str = "cpu", steps: int = 2) -> dict[str, Any]:
    """Train a tiny linear layer and return throughput and peak memory.

    The capability check runs first, so fp16 and fp8 raise on a device that
    cannot run them. bf16 runs only when autocast works on ``device``.
    """
    torch = _require_torch()
    resolved = device
    if device == "auto":
        resolved = resolve_training_device(device)
    plan = require_precision(
        precision, capabilities_for(resolved, precision, torch_module=torch)
    )
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    model.to(plan.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.ones(8, 4, device=plan.device)
    target = torch.ones(8, 4, device=plan.device)
    session = PrecisionSession(plan, torch_module=torch)
    session.begin()
    loss_value = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        with session.autocast():
            predicted = model(inputs)
            loss = torch.nn.functional.mse_loss(predicted, target)
        session.backward(loss)
        session.optimizer_step(optimizer)
        loss_value = float(loss.detach().float().item())
    stats = session.end(steps)
    stats["final_loss"] = loss_value
    return stats


class PrecisionSession:
    """Autocast, loss scaling, throughput, and peak memory for one loop.

    fp32 steps are ordinary PyTorch. fp16 scales the loss with
    ``GradScaler`` inside CUDA autocast. bf16 uses autocast and an unscaled
    backward. fp8 enters Transformer Engine's ``fp8_autocast`` with a
    delayed-scaling recipe. Master weights stay fp32 in all four modes.
    """

    def __init__(self, plan: PrecisionPlan, torch_module: Any | None = None) -> None:
        self.plan = plan
        self._torch = torch_module
        self._scaler: Any = None
        self._peak: int | None = None
        self._t0 = 0.0

    def begin(self) -> None:
        """Reset peak-memory counters and start the step timer."""
        torch = self._torch_mod()
        if self.plan.loss_scaling:
            self._scaler = _grad_scaler(torch, self.plan.capabilities.kind)
        if self.plan.capabilities.kind == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._t0 = time.perf_counter()
        self._sample()

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Cast the forward to this plan's precision."""
        if self.plan.precision == "fp32":
            yield
            return
        if self.plan.precision == "fp8":
            with fp8_autocast():
                yield
            return
        torch = self._torch_mod()
        dtype = torch.float16 if self.plan.precision == "fp16" else torch.bfloat16
        with torch.autocast(device_type=self.plan.capabilities.kind, dtype=dtype):
            yield

    def backward(self, loss: Any) -> None:
        """Backpropagate ``loss``, scaling it when the plan is fp16."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        self._sample()

    def optimizer_step(self, optimizer: Any) -> None:
        """Step ``optimizer``. fp16 steps through the GradScaler."""
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def end(self, steps: int) -> dict[str, Any]:
        """Stop the timer and return throughput and peak memory."""
        elapsed = time.perf_counter() - self._t0
        self._sample()
        throughput: float | None = None
        if steps > 0 and elapsed > 0:
            throughput = steps / elapsed
        return {
            "precision": self.plan.precision,
            "loss_scaling": self.plan.loss_scaling,
            "backend": self.plan.backend,
            "device": self.plan.device,
            "throughput_steps_per_s": throughput,
            "peak_memory_bytes": self._peak,
        }

    def _sample(self) -> None:
        reading = sample_memory(self._torch_mod(), self.plan.capabilities.kind)
        if reading is None:
            return
        if self._peak is None or reading > self._peak:
            self._peak = reading

    def _torch_mod(self) -> Any:
        if self._torch is not None:
            return self._torch
        self._torch = _require_torch()
        return self._torch


@contextmanager
def fp8_autocast() -> Iterator[None]:
    """Enter Transformer Engine FP8 autocast with delayed scaling.

    Import happens here, not at module import. A missing package raises
    :class:`PrecisionUnsupported`.
    """
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
    except ImportError as exc:
        raise PrecisionUnsupported(
            "precision 'fp8' needs Transformer Engine (transformer_engine) "
            "on a CUDA GPU with compute capability 8.9 or newer. "
            "No fallback is applied."
        ) from exc
    try:
        recipe = DelayedScaling(fp8_format=Format.HYBRID)
    except TypeError:
        recipe = DelayedScaling()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        yield


def sample_memory(torch: Any, kind: str) -> int | None:
    """Peak bytes visible for ``kind``.

    CUDA uses ``max_memory_allocated`` after :meth:`PrecisionSession.begin`
    resets the counter, so the figure includes weights resident at the
    start of the loop and activations allocated after that. CPU and MPS
    use the process resident set sampled during the loop. That number
    includes the interpreter, not only tensors.
    """
    if kind == "cuda":
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None
    if kind == "mps":
        current = _mps_allocated(torch)
        if current is not None:
            return current
    return _process_rss_bytes()


def normalize_throughput(value: Any) -> int | float | None:
    """A non-negative finite step rate, or null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    if value < 0:
        return None
    return value


def normalize_peak(value: Any) -> int | None:
    """A non-negative byte count, or null."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if (
        isinstance(value, float)
        and value >= 0
        and value == value
        and value.is_integer()
    ):
        if value in (float("inf"), float("-inf")):
            return None
        return int(value)
    return None


def _recipe_precision(recipe: Mapping[str, Any]) -> str:
    precision = recipe.get("precision", "fp32")
    if not isinstance(precision, str) or not precision:
        return "fp32"
    return precision


def _recipe_device(recipe: Mapping[str, Any]) -> str:
    device = recipe.get("device", "cpu")
    if not isinstance(device, str) or not device.strip():
        return "cpu"
    return device


def _unsupported(precision: str, capabilities: DeviceCapabilities) -> str:
    label = capabilities.device
    if capabilities.name and capabilities.name not in {capabilities.device, ""}:
        label = f"{capabilities.device} ({capabilities.name})"
    detail = ""
    if capabilities.detail:
        detail = f" {capabilities.detail}."
    need = _NEEDS.get(precision, "This precision is not supported here.")
    return (
        f"precision {precision!r} is not supported on device {label!r}.{detail} "
        f"{need} No fallback is applied."
    )


def _device_kind(device: str) -> tuple[str, int | None]:
    if device == "cpu":
        return "cpu", None
    if device == "mps":
        return "mps", None
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return "cuda", int(suffix)
        return "cuda", 0
    return device, None


def _cpu_capabilities(torch: Any | None, *, try_bf16: bool) -> DeviceCapabilities:
    bf16 = bool(try_bf16 and torch is not None and _bf16_autocast_ok(torch, "cpu"))
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


def _mps_capabilities(
    device: str, torch: Any | None, *, try_bf16: bool
) -> DeviceCapabilities:
    available = False
    if torch is not None:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        try:
            available = mps is not None and bool(mps.is_available())
        except Exception:
            available = False
    bf16 = bool(
        try_bf16 and available and torch is not None and _bf16_autocast_ok(torch, "mps")
    )
    return DeviceCapabilities(
        device=device,
        kind="mps",
        bf16=bf16,
        fp16=False,
        fp8=False,
        available=available,
        name="mps",
        detail="",
    )


def _cuda_capabilities(
    device: str,
    index: int,
    torch: Any | None,
    *,
    fp8_importable: bool | None,
) -> DeviceCapabilities:
    if torch is None:
        return DeviceCapabilities(
            device=device,
            kind="cuda",
            bf16=False,
            fp16=False,
            fp8=False,
            available=False,
            name="",
            detail="CUDA is not available",
        )
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if not available:
        return DeviceCapabilities(
            device=device,
            kind="cuda",
            bf16=False,
            fp16=False,
            fp8=False,
            available=False,
            name="",
            detail="CUDA is not available",
        )
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        major, minor = (0, 0)
    cc = (int(major), int(minor))
    try:
        name = str(torch.cuda.get_device_name(index)).strip()
    except Exception:
        name = ""
    bf16 = cc >= _BF16_CC and _cuda_bf16_flag(torch)
    fp16 = cc >= _FP16_CC
    te_ok = _fp8_package(fp8_importable)
    fp8 = cc >= _FP8_CC and te_ok
    if cc < _FP8_CC:
        detail = f"compute capability {cc[0]}.{cc[1]}"
    elif not te_ok:
        detail = (
            f"compute capability {cc[0]}.{cc[1]}; transformer_engine is not installed"
        )
    else:
        detail = f"compute capability {cc[0]}.{cc[1]} with transformer_engine"
    return DeviceCapabilities(
        device=device,
        kind="cuda",
        bf16=bf16,
        fp16=fp16,
        fp8=fp8,
        available=True,
        name=name,
        detail=detail,
    )


def _cuda_bf16_flag(torch: Any) -> bool:
    fn = getattr(getattr(torch, "cuda", None), "is_bf16_supported", None)
    if not callable(fn):
        return True
    try:
        return bool(fn(including_emulation=False))
    except TypeError:
        try:
            return bool(fn())
        except Exception:
            return True
    except Exception:
        return True


def _fp8_package(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    try:
        return importlib.util.find_spec("transformer_engine.pytorch") is not None
    except Exception:
        return False


def _bf16_autocast_ok(torch: Any, device_type: str) -> bool:
    """True when a linear step runs under bfloat16 autocast.

    The global RNG is restored so the probe does not change the training
    seed applied afterwards.
    """
    autocast = getattr(torch, "autocast", None)
    if autocast is None or not hasattr(torch, "bfloat16"):
        return False
    cpu_state = None
    cuda_state = None
    try:
        cpu_state = torch.get_rng_state()
    except Exception:
        cpu_state = None
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is not None and bool(cuda.is_available()):
            cuda_state = cuda.get_rng_state_all()
    except Exception:
        cuda_state = None
    try:
        model = torch.nn.Linear(4, 4)
        inputs = torch.ones(2, 4)
        if device_type != "cpu":
            model = model.to(device_type)
            inputs = inputs.to(device_type)
        target = torch.zeros(2, 4, device=inputs.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        with autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        return True
    except Exception:
        return False
    finally:
        if cpu_state is not None:
            try:
                torch.set_rng_state(cpu_state)
            except Exception:
                pass
        if cuda_state is not None and cuda is not None:
            try:
                cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass


def _grad_scaler(torch: Any, device_kind: str) -> Any:
    if device_kind != "cuda":
        raise PrecisionUnsupported(
            "precision 'fp16' is not supported off CUDA. "
            "fp16 training needs a CUDA GPU and runs under torch.amp autocast "
            "with GradScaler loss scaling. No fallback is applied."
        )
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_cls is not None:
        try:
            return scaler_cls("cuda")
        except TypeError:
            return scaler_cls()
    cuda_amp = getattr(getattr(torch, "cuda", None), "amp", None)
    scaler_cls = getattr(cuda_amp, "GradScaler", None) if cuda_amp is not None else None
    if scaler_cls is None:
        raise PrecisionUnsupported(
            "precision 'fp16' needs torch.amp.GradScaler for loss scaling. "
            "No fallback is applied."
        )
    return scaler_cls()


def _mps_allocated(torch: Any) -> int | None:
    mps = getattr(torch, "mps", None)
    fn = getattr(mps, "current_allocated_memory", None) if mps is not None else None
    if not callable(fn):
        return None
    try:
        return int(fn())
    except Exception:
        return None


def _process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_working_set()
    if sys.platform.startswith("linux"):
        return _linux_rss()
    try:
        import resource
    except ImportError:
        return None
    try:
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError, AttributeError):
        return None
    if usage < 0:
        return None
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _linux_rss() -> int | None:
    try:
        text = Path("/proc/self/statm").read_text(encoding="utf-8")
    except OSError:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        pages = int(parts[1])
        page = int(sysconf_page())
    except (OSError, ValueError):
        return None
    if pages < 0 or page <= 0:
        return None
    return pages * page


def sysconf_page() -> int:
    """Memory page size, or 4096 when the host does not report one."""
    try:
        import os

        value = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return 4096
    if isinstance(value, int) and value > 0:
        return value
    return 4096


def _windows_working_set() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        info = psapi.GetProcessMemoryInfo
        info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        info.restype = wintypes.BOOL
        if not info(get_process(), ctypes.byref(counters), counters.cb):
            return None
        size = int(counters.WorkingSetSize)
    except (OSError, AttributeError, ValueError):
        return None
    if size < 0:
        return None
    return size


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _try_import_torch()
    if torch is None:
        raise RecipeConfigError(
            "that device needs torch. Install it with: pip install 'ohho-os[train]'"
        )
    return torch
