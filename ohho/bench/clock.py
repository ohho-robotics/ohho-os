"""Clocks for per-stage bench timing.

CPU stages use :func:`time.perf_counter`. A CUDA device uses
``torch.cuda.Event`` for h2d, inference, and d2h. Torch is imported only
when a CUDA clock is constructed. :class:`VirtualClock` moves only when a
test runner calls :meth:`advance`, so a scripted slowdown is exact.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ohho.bench.runner import STAGE_NAMES


def _stamp(runner: Any, batch: int, now: Callable[[], float]) -> dict[str, float]:
    marks = [now()]
    prepared = runner.preprocess(batch)
    marks.append(now())
    device_in = runner.h2d(prepared)
    marks.append(now())
    device_out = runner.inference(device_in)
    marks.append(now())
    host_out = runner.d2h(device_out)
    marks.append(now())
    runner.postprocess(host_out)
    marks.append(now())
    return {
        name: max(0.0, marks[index + 1] - marks[index])
        for index, name in enumerate(STAGE_NAMES)
    }


class PerfClock:
    """Host clock. Stage durations are consecutive ``perf_counter`` spans."""

    kind = "perf_counter"

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, target: float) -> None:
        """Sleep until ``perf_counter`` reaches ``target``. A past target returns."""
        delay = float(target) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def measure(self, runner: Any, batch: int) -> dict[str, float]:
        return _stamp(runner, batch, self.now)


class VirtualClock:
    """Clock that advances only through :meth:`advance`.

    Durations are exact for a runner that advances this same instance inside
    each stage. ``kind`` is ``virtual`` so a scripted run is not labeled as
    a wall-clock measurement.
    """

    kind = "virtual"

    def __init__(self) -> None:
        self._seconds = 0.0

    def now(self) -> float:
        return self._seconds

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("virtual clock cannot move backwards")
        self._seconds += float(seconds)

    def sleep_until(self, target: float) -> None:
        """Advance to ``target`` without sleeping. A past target returns."""
        delay = float(target) - self._seconds
        if delay > 0:
            self.advance(delay)

    def measure(self, runner: Any, batch: int) -> dict[str, float]:
        return _stamp(runner, batch, self.now)


class CudaEventClock:
    """CUDA events for the device stages, ``perf_counter`` for the host stages.

    ``cuda`` is the ``torch.cuda`` module, or a test double with
    ``Event(enable_timing=True)`` and ``elapsed_time`` in milliseconds.
    The published end-to-end sample is still the sum of the five stages.
    """

    kind = "cuda_event"

    def __init__(self, cuda: Any | None = None) -> None:
        if cuda is None:
            cuda = _load_torch_cuda()
        self._cuda = cuda

    def measure(self, runner: Any, batch: int) -> dict[str, float]:
        prepared, preprocess = _cpu_span(lambda: runner.preprocess(batch))
        device_in, h2d = self._gpu_span(lambda: runner.h2d(prepared))
        device_out, inference = self._gpu_span(lambda: runner.inference(device_in))
        host_out, d2h = self._gpu_span(lambda: runner.d2h(device_out))
        _host, postprocess = _cpu_span(lambda: runner.postprocess(host_out))
        return {
            "preprocess": preprocess,
            "h2d": h2d,
            "inference": inference,
            "d2h": d2h,
            "postprocess": postprocess,
        }

    def _gpu_span(self, fn: Callable[[], Any]) -> tuple[Any, float]:
        start = self._cuda.Event(enable_timing=True)
        end = self._cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        return value, max(0.0, float(start.elapsed_time(end)) / 1000.0)


def _cpu_span(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, max(0.0, time.perf_counter() - started)


def _load_torch_cuda() -> Any:
    from ohho.bench.errors import BenchHarnessError

    try:
        import torch
    except ImportError as exc:
        raise BenchHarnessError(
            "CUDA event timing needs torch, which is not installed."
        ) from exc
    if not torch.cuda.is_available():
        raise BenchHarnessError(
            "CUDA event timing needs a CUDA device, and this process has none."
        )
    return torch.cuda


def cuda_device_available() -> bool:
    """True when PyTorch imports and reports a CUDA device. Failures are false."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def clock_for_providers(providers: Sequence[str]) -> PerfClock | CudaEventClock:
    """Use CUDA events only when a GPU provider is selected and CUDA is up.

    Otherwise the CPU clock is used. The caller records that fallback on the
    result so a GPU target measured on CPU is visible.
    """
    names = " ".join(providers)
    wants_gpu = "CUDA" in names or "Tensorrt" in names or "TensorRT" in names
    if wants_gpu and cuda_device_available():
        return CudaEventClock()
    return PerfClock()
