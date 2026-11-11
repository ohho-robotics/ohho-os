"""LLM serving sweep: batching, prefix cache, prefill, and speculation.

``ohho bench llm --sweep`` measures concurrency and the server options the
selected backend can change. Options the backend cannot change are recorded
as skips with a reason. The CPU path is a deterministic stand-in. It does
not launch vLLM, SGLang, TensorRT-LLM, or llama.cpp, and it does not
measure a GPU.

The result is ``ohho.bench/v1``. ``llm.pareto`` is the non-dominated set of
p95 time-to-first-token against tokens/s, compared inside each workload.
``--target-p95-ttft`` fills ``llm.recommendation`` with the shared-prefix
cell at or under that TTFT that has the highest tokens/s.
"""

from __future__ import annotations

import hashlib
import math
import platform
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ohho.bench.cost import CostOverrides, attach_cost, require_price
from ohho.bench.environment import git_sha
from ohho.bench.errors import BenchHarnessError
from ohho.bench.schema import SCHEMA_ID, validate
from ohho.serve.backends.errors import ServingBackendError
from ohho.serve.backends.sweep_support import (
    SWEEP_AXES,
    canonical_backend,
    support_for,
)
from ohho.serve.loadgen import LoadGenError, LoadSpec, latency_stats, run_load

LLM_BLOCK_ID = "ohho.bench.llm/v1"
WORKLOAD_NAMES = ("short", "long-context", "shared-prefix")
_KNOBS = (
    "max_batch_tokens",
    "prefix_cache",
    "chunked_prefill",
    "speculative_decoding",
)
_RECOMMENDATION_WORKLOAD = "shared-prefix"

SHARED_SYSTEM_PROMPT = (
    "You are the runtime assistant. Obey the command exactly. "
    "Reply with one action and no explanation. "
    "Keep motion inside the safety caps. "
    "The workspace is clear of people. "
    "Prefer the shortest action that meets the command."
)
SHORT_COMMANDS = ("stop", "move forward", "report status", "turn left")
# A 1x1 PNG. The long-context workload attaches it as an image part so a
# vision model receives an image. The stand-in scores image_tokens, not pixels.
TINY_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
LONG_CONTEXT_PROMPT = "Describe the scene and answer the command. " + " ".join(
    f"Object {index} is at station {index % 17}." for index in range(180)
)

STANDIN_NOTE = (
    "CPU stand-in with a virtual clock. Prefix-cache, batch, and "
    "chunked-prefill effects are a deterministic model, not a GPU "
    "measurement. This command does not launch an engine."
)
LIVE_NOTE = (
    "Measured against a running server. Server launch settings were not "
    "changed. This command does not start an engine."
)
_ENGINE_REQUIRED = (
    "{backend} is not measured from this command without --endpoint. "
    "A CPU sweep uses --backend stand-in. Starting the engine would be a "
    "one-GPU run, which this command does not do."
)
_RELAUNCH = (
    "{option} is set when the server process starts. This sweep measures "
    "one running server and does not relaunch it."
)


@dataclass(frozen=True)
class Workload:
    """One prompt mix. Token counts are the nominal sizes the stand-in scores."""

    name: str
    prompts: tuple[str, ...]
    system_prompt: str
    image_url: str
    prompt_tokens: int
    image_tokens: int
    prefill_ms: float
    prefix_fraction: float


WORKLOADS: dict[str, Workload] = {
    "short": Workload(
        name="short",
        prompts=SHORT_COMMANDS,
        system_prompt="",
        image_url="",
        prompt_tokens=16,
        image_tokens=0,
        prefill_ms=8.0,
        prefix_fraction=0.0,
    ),
    "long-context": Workload(
        name="long-context",
        prompts=(LONG_CONTEXT_PROMPT,),
        system_prompt=SHARED_SYSTEM_PROMPT,
        image_url=TINY_IMAGE_URL,
        prompt_tokens=2048,
        image_tokens=256,
        prefill_ms=120.0,
        prefix_fraction=0.18,
    ),
    "shared-prefix": Workload(
        name="shared-prefix",
        prompts=SHORT_COMMANDS,
        system_prompt=SHARED_SYSTEM_PROMPT,
        image_url="",
        prompt_tokens=512,
        image_tokens=0,
        prefill_ms=55.0,
        prefix_fraction=0.75,
    ),
}


@dataclass(frozen=True)
class SweepConfig:
    """One ``ohho bench llm --sweep`` run."""

    backend: str = "stand-in"
    model: str = "stand-in"
    endpoint: str = ""
    precision: str = "fp16"
    concurrencies: tuple[int, ...] = (1, 2, 4, 8)
    max_batch_tokens: tuple[int, ...] = (512, 2048)
    prefix_cache: tuple[bool, ...] = (False, True)
    chunked_prefill: tuple[bool, ...] = (False, True)
    speculative: tuple[bool, ...] = (False, True)
    workloads: tuple[str, ...] = WORKLOAD_NAMES
    max_tokens: int = 8
    requests: int = 4
    timeout_s: float = 30.0
    target_p95_ttft_ms: float | None = None
    speculative_model: str = ""
    speculative_tokens: int = 4
    launcher: str = ""
    backend_version: str = ""
    chart: str = ""
    hardware_target: str = ""

    def __post_init__(self) -> None:
        try:
            backend = canonical_backend(self.backend)
        except ServingBackendError as exc:
            raise BenchHarnessError(str(exc)) from exc
        model = self.model.strip() if isinstance(self.model, str) else ""
        if backend == "stand-in" and not model:
            model = "stand-in"
        if not model:
            raise BenchHarnessError("--model is required")
        precision = self.precision.strip() if isinstance(self.precision, str) else ""
        if not precision:
            raise BenchHarnessError("--precision is required")
        endpoint = self.endpoint.strip() if isinstance(self.endpoint, str) else ""
        if backend == "stand-in" and endpoint:
            raise BenchHarnessError("--backend stand-in does not use --endpoint")
        concurrencies = _unique_ints(self.concurrencies, "--concurrency")
        batches = _unique_ints(self.max_batch_tokens, "--max-batch-tokens")
        prefix = _unique_bools(self.prefix_cache, "--prefix-cache")
        chunked = _unique_bools(self.chunked_prefill, "--chunked-prefill")
        speculative = _unique_bools(self.speculative, "--speculative")
        workloads = _workloads(self.workloads)
        _check_count(self.max_tokens, "--max-tokens", minimum=1)
        _check_count(self.requests, "--requests", minimum=1)
        _check_count(self.speculative_tokens, "--speculative-tokens", minimum=1)
        if self.timeout_s <= 0:
            raise BenchHarnessError("--timeout must be positive")
        target = self.target_p95_ttft_ms
        if target is not None and (not math.isfinite(target) or target < 0):
            raise BenchHarnessError(
                "--target-p95-ttft must be a non-negative number of milliseconds"
            )
        version = self.backend_version.strip() if self.backend_version else ""
        if not version:
            version = "stand-in" if backend == "stand-in" else "unknown"
        chart = self.chart.strip() if isinstance(self.chart, str) else ""
        hardware = (
            self.hardware_target.strip()
            if isinstance(self.hardware_target, str)
            else ""
        )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "concurrencies", concurrencies)
        object.__setattr__(self, "max_batch_tokens", batches)
        object.__setattr__(self, "prefix_cache", prefix)
        object.__setattr__(self, "chunked_prefill", chunked)
        object.__setattr__(self, "speculative", speculative)
        object.__setattr__(self, "workloads", workloads)
        object.__setattr__(self, "backend_version", version)
        object.__setattr__(self, "chart", chart)
        object.__setattr__(self, "hardware_target", hardware)
        object.__setattr__(
            self,
            "speculative_model",
            self.speculative_model.strip()
            if isinstance(self.speculative_model, str)
            else "",
        )


@dataclass(frozen=True)
class SweepCell:
    """One measured combination. ``None`` means that axis was not varied."""

    workload: str
    concurrency: int
    max_batch_tokens: int | None = None
    prefix_cache: bool | None = None
    chunked_prefill: bool | None = None
    speculative_decoding: bool | None = None


Measurer = Callable[[SweepCell, SweepConfig], dict[str, Any]]


def execute_sweep(
    config: SweepConfig,
    *,
    measurer: Measurer | None = None,
    cost: CostOverrides | None = None,
) -> dict[str, Any]:
    """Run the sweep and return a validated ``ohho.bench/v1`` document.

    Cost is filled from the measured token throughput. Passing ``cost``
    changes the price inputs and does not add cells.
    """
    if config.hardware_target:
        require_price(config.hardware_target, cost)
    cells, skipped = expand_grid(config)
    if not cells:
        raise BenchHarnessError("the sweep has no cells to measure")
    measure = measurer if measurer is not None else _default_measurer(config)
    started = datetime.now(timezone.utc)
    points: list[dict[str, Any]] = []
    try:
        for cell in cells:
            points.append(measure(cell, config))
    except LoadGenError as exc:
        raise BenchHarnessError(str(exc), exit_code=1) from exc
    points.sort(key=_point_sort_key)
    pareto = pareto_table(points)
    _mark_pareto(points, pareto)
    recommendations = None
    recommendation = None
    if config.target_p95_ttft_ms is not None:
        recommendations = [
            recommend(points, config.target_p95_ttft_ms, workload)
            for workload in _present_workloads(points)
        ]
        recommendation = _headline(recommendations, points)
    primary = _primary_point(points, recommendation)
    document = _document(
        config,
        points,
        pareto,
        skipped,
        primary,
        recommendation,
        recommendations,
        started=started,
    )
    attach_cost(document, cost, profile_id=config.hardware_target)
    validate(document)
    return document


def expand_grid(config: SweepConfig) -> tuple[list[SweepCell], list[dict[str, str]]]:
    """Cartesian product of supported axes, plus a skip record for the rest."""
    support = support_for(
        config.backend,
        speculative_model=config.speculative_model,
        launcher=config.launcher,
    )
    live = bool(config.endpoint) and config.backend != "stand-in"
    skipped: list[dict[str, str]] = []
    varied: dict[str, Sequence[Any]] = {"concurrency": config.concurrencies}
    requested = {
        "max_batch_tokens": config.max_batch_tokens,
        "prefix_cache": config.prefix_cache,
        "chunked_prefill": config.chunked_prefill,
        "speculative_decoding": config.speculative,
    }
    for name in SWEEP_AXES:
        if name == "concurrency":
            continue
        axis = support.axis(name)
        if not axis.supported:
            skipped.append({"option": name, "reason": axis.reason})
            continue
        if live:
            skipped.append({"option": name, "reason": _RELAUNCH.format(option=name)})
            continue
        varied[name] = requested[name]
    cells: list[SweepCell] = []
    batches: Sequence[Any] = varied.get("max_batch_tokens", (None,))
    prefixes: Sequence[Any] = varied.get("prefix_cache", (None,))
    chunked_values: Sequence[Any] = varied.get("chunked_prefill", (None,))
    speculative_values: Sequence[Any] = varied.get("speculative_decoding", (None,))
    for workload in config.workloads:
        for concurrency in config.concurrencies:
            for batch in batches:
                for prefix in prefixes:
                    for chunked in chunked_values:
                        for speculative in speculative_values:
                            cells.append(
                                SweepCell(
                                    workload=workload,
                                    concurrency=concurrency,
                                    max_batch_tokens=batch,
                                    prefix_cache=prefix,
                                    chunked_prefill=chunked,
                                    speculative_decoding=speculative,
                                )
                            )
    return cells, skipped


def recommend(
    points: Sequence[Mapping[str, Any]],
    target_p95_ttft_ms: float,
    workload: str,
) -> dict[str, Any]:
    """Highest tokens/s at or under ``target_p95_ttft_ms`` for one workload.

    An equal p95 meets the target. When several cells share that tokens/s,
    the lower TTFT wins, then prefix cache on, then a larger batch.
    """
    pool = [point for point in points if point["workload"] == workload]
    meeting = [
        point
        for point in pool
        if float(point["ttft"]["p95"]) <= target_p95_ttft_ms + 1e-9
    ]
    shown = _format_ms(target_p95_ttft_ms)
    if not meeting:
        return {
            "target_p95_ttft_ms": target_p95_ttft_ms,
            "met": False,
            "workload": workload,
            "reason": (f"no {workload} cell has p95 TTFT <= {shown} ms"),
        }
    best = max(meeting, key=_throughput_rank)
    row: dict[str, Any] = {
        "target_p95_ttft_ms": target_p95_ttft_ms,
        "met": True,
        "workload": workload,
        "reason": (
            f"highest tokens/s among {workload} cells with p95 TTFT <= {shown} ms"
        ),
        "concurrency": best["concurrency"],
        "ttft_p95_ms": best["ttft"]["p95"],
        "tokens_per_s": best["tokens_per_s"],
    }
    for key in _KNOBS:
        if key in best:
            row[key] = best[key]
    return row


def pareto_table(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated cells inside each workload.

    A cell is dominated when another cell of the same workload has p95 TTFT
    no higher and tokens/s no lower, and is strictly better on one of them.
    """
    rows = [_pareto_row(point) for point in points]
    kept: list[dict[str, Any]] = []
    for row in rows:
        group = [other for other in rows if other["workload"] == row["workload"]]
        if any(_dominates(other, row) for other in group):
            continue
        kept.append(row)
    kept.sort(
        key=lambda row: (
            _workload_index(row["workload"]),
            row["ttft_p95_ms"],
            -row["tokens_per_s"],
        )
    )
    return kept


def render_pareto_svg(document: Mapping[str, Any]) -> str:
    """Scatter p95 TTFT against tokens/s. Pareto cells are drawn larger."""
    llm = document["llm"]
    points = llm["points"]
    if not points:
        raise BenchHarnessError("the sweep has no points to chart")
    width, height = 720, 440
    left, right, top, bottom = 72, 24, 48, 56
    xs = [float(point["tokens_per_s"]) for point in points]
    ys = [float(point["ttft"]["p95"]) for point in points]
    x0, x1 = _span(min(xs), max(xs))
    y0, y1 = _span(min(ys), max(ys))

    def px(value: float) -> float:
        return left + (value - x0) / (x1 - x0) * (width - left - right)

    def py(value: float) -> float:
        return top + (y1 - value) / (y1 - y0) * (height - top - bottom)

    circles: list[str] = []
    for point in points:
        color = "#0f6e56" if point.get("pareto") else "#98a2b3"
        radius = 5 if point.get("pareto") else 3
        circles.append(
            f'<circle cx="{px(float(point["tokens_per_s"])):.2f}" '
            f'cy="{py(float(point["ttft"]["p95"])):.2f}" r="{radius}" '
            f'fill="{color}"/>'
        )
    backend = _xml(str(llm.get("backend", "")))
    title = "p95 TTFT against tokens/s"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f"<title>{_xml(title)}</title>\n"
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="16">'
        f"{_xml(title)}</text>\n"
        f'<text x="{width - 16}" y="28" text-anchor="end" '
        f'font-family="sans-serif" font-size="12" fill="#475467">{backend}</text>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
        f'stroke="#344054"/>\n'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#344054"/>\n'
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" '
        f'font-family="sans-serif" font-size="11" fill="#475467">'
        f"{y1:.1f}</text>\n"
        f'<text x="{left - 8}" y="{height - bottom}" text-anchor="end" '
        f'font-family="sans-serif" font-size="11" fill="#475467">'
        f"{y0:.1f}</text>\n"
        f'<text x="{left}" y="{height - 16}" font-family="sans-serif" '
        f'font-size="12" fill="#344054">tokens/s</text>\n'
        f'<text x="16" y="{top + 80}" font-family="sans-serif" font-size="12" '
        f'fill="#344054" transform="rotate(-90 16 {top + 80})">p95 TTFT (ms)</text>\n'
        f'<circle cx="{left}" cy="{height - 28}" r="5" fill="#0f6e56"/>\n'
        f'<text x="{left + 12}" y="{height - 24}" font-family="sans-serif" '
        f'font-size="11" fill="#344054">pareto</text>\n'
        f'<circle cx="{left + 80}" cy="{height - 28}" r="3" fill="#98a2b3"/>\n'
        f'<text x="{left + 90}" y="{height - 24}" font-family="sans-serif" '
        f'font-size="11" fill="#344054">other</text>\n'
        + "\n".join(circles)
        + "\n</svg>\n"
    )


def measure_standin(cell: SweepCell, config: SweepConfig) -> dict[str, Any]:
    """Deterministic CPU timings for one cell. No process is started."""
    workload = WORKLOADS[cell.workload]
    ttft_center, tpot_center = _centers(workload, cell)
    ttft_ms = _spread(ttft_center, config.requests)
    tpot_ms = _spread(tpot_center, config.requests)
    gaps: list[float] = []
    e2e_ms: list[float] = []
    gaps_per_request = max(config.max_tokens - 1, 0)
    for index in range(config.requests):
        gaps.extend([tpot_ms[index]] * gaps_per_request)
        e2e_ms.append(ttft_ms[index] + tpot_ms[index] * gaps_per_request)
    mean_e2e_s = (sum(e2e_ms) / len(e2e_ms)) / 1000.0
    per_request = config.max_tokens / mean_e2e_s
    occupancy = cell.concurrency / (1.0 + 0.12 * (cell.concurrency - 1))
    tokens_per_s = per_request * occupancy
    output_tokens = config.requests * config.max_tokens
    duration_s = output_tokens / tokens_per_s
    return _point(
        cell,
        config,
        workload,
        ttft_ms=ttft_ms,
        gap_ms=gaps,
        e2e_ms=e2e_ms,
        tokens_per_s=tokens_per_s,
        tokens_per_request_s=per_request,
        requests_ok=config.requests,
        output_tokens=output_tokens,
        duration_s=duration_s,
    )


def measure_http(cell: SweepCell, config: SweepConfig) -> dict[str, Any]:
    """Time one cell against an already running OpenAI-compatible server."""
    workload = WORKLOADS[cell.workload]
    spec = LoadSpec(
        endpoint_url=config.endpoint,
        model=config.model,
        prompt=workload.prompts[0],
        prompts=workload.prompts,
        system_prompt=workload.system_prompt,
        image_url=workload.image_url,
        max_tokens=config.max_tokens,
        concurrency=cell.concurrency,
        requests=config.requests,
        timeout_s=config.timeout_s,
    )
    result = run_load(spec)
    ok = [sample for sample in result.samples if sample.ok]
    ttft_ms = [sample.ttft_s * 1000.0 for sample in ok]
    gap_ms = [gap * 1000.0 for sample in ok for gap in sample.inter_token_s]
    e2e_ms = [sample.e2e_s * 1000.0 for sample in ok]
    output_tokens = sum(sample.output_tokens for sample in ok)
    wall = result.wall_s if result.wall_s > 0 else 1e-9
    per_request = [
        sample.output_tokens / sample.e2e_s
        for sample in ok
        if sample.e2e_s > 0 and sample.output_tokens
    ]
    tokens_per_request_s = sum(per_request) / len(per_request) if per_request else 0.0
    return _point(
        cell,
        config,
        workload,
        ttft_ms=ttft_ms,
        gap_ms=gap_ms,
        e2e_ms=e2e_ms,
        tokens_per_s=output_tokens / wall,
        tokens_per_request_s=tokens_per_request_s,
        requests_ok=len(ok),
        output_tokens=output_tokens,
        duration_s=wall,
    )


def _default_measurer(config: SweepConfig) -> Measurer:
    if config.backend == "stand-in":
        return measure_standin
    if not config.endpoint:
        raise BenchHarnessError(_ENGINE_REQUIRED.format(backend=config.backend))
    return measure_http


def _centers(workload: Workload, cell: SweepCell) -> tuple[float, float]:
    """Return stand-in (TTFT ms, TPOT ms) before the per-request spread.

    A shared system prompt drops most of the prefill when prefix caching is
    on. Short commands share nothing, so that fraction is zero. A larger
    token budget delays the first token a little and speeds later tokens.
    Chunked prefill cuts long-prompt TTFT and adds a small decode cost.
    """
    batch = cell.max_batch_tokens if cell.max_batch_tokens is not None else 512
    prefill = workload.prefill_ms
    if cell.prefix_cache:
        prefill *= 1.0 - workload.prefix_fraction
    delay = 0.004 * batch
    efficiency = 1.0 + 0.35 * math.log2(max(batch, 256) / 256.0)
    if cell.chunked_prefill and workload.prompt_tokens >= 256:
        prefill *= 0.62
        delay *= 0.5
        efficiency *= 0.93
    elif cell.chunked_prefill:
        delay *= 0.85
    queue = prefill * 0.16 * (cell.concurrency - 1)
    ttft = max(prefill + delay + queue, 0.05)
    tpot = max((3.1 / efficiency) * (1.0 + 0.045 * (cell.concurrency - 1)), 0.05)
    return ttft, tpot


def _spread(center: float, count: int) -> list[float]:
    if count <= 1:
        return [center]
    step = 0.08 / (count - 1)
    return [center * (0.96 + step * index) for index in range(count)]


def _point(
    cell: SweepCell,
    config: SweepConfig,
    workload: Workload,
    *,
    ttft_ms: list[float],
    gap_ms: list[float],
    e2e_ms: list[float],
    tokens_per_s: float,
    tokens_per_request_s: float,
    requests_ok: int,
    output_tokens: int,
    duration_s: float,
) -> dict[str, Any]:
    point: dict[str, Any] = {
        "workload": cell.workload,
        "concurrency": cell.concurrency,
        "prompt_tokens": workload.prompt_tokens,
        "image_tokens": workload.image_tokens,
        "precision": config.precision,
        "prompt": workload.prompts[0],
        "ttft": latency_stats(ttft_ms, "ms"),
        "tpot": latency_stats(gap_ms, "ms"),
        "inter_token": latency_stats(gap_ms, "ms"),
        "e2e": latency_stats(e2e_ms, "ms"),
        "tokens_per_s": tokens_per_s,
        "tokens_per_request_s": tokens_per_request_s,
        "requests_per_s": requests_ok / duration_s if duration_s > 0 else 0.0,
        "requests": config.requests,
        "requests_ok": requests_ok,
        "output_tokens": output_tokens,
        "duration_s": duration_s,
    }
    if cell.max_batch_tokens is not None:
        point["max_batch_tokens"] = cell.max_batch_tokens
    if cell.prefix_cache is not None:
        point["prefix_cache"] = cell.prefix_cache
    if cell.chunked_prefill is not None:
        point["chunked_prefill"] = cell.chunked_prefill
    if cell.speculative_decoding is not None:
        point["speculative_decoding"] = cell.speculative_decoding
    return point


def _pareto_row(point: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "workload": point["workload"],
        "concurrency": point["concurrency"],
        "ttft_p95_ms": point["ttft"]["p95"],
        "tokens_per_s": point["tokens_per_s"],
        "prompt_tokens": point["prompt_tokens"],
        "image_tokens": point["image_tokens"],
    }
    for key in _KNOBS:
        if key in point:
            row[key] = point[key]
    return row


def _mark_pareto(
    points: list[dict[str, Any]], pareto: Sequence[Mapping[str, Any]]
) -> None:
    chosen = {_identity(row) for row in pareto}
    for point in points:
        point["pareto"] = _identity(_pareto_row(point)) in chosen


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["workload"],
        row["concurrency"],
        row.get("max_batch_tokens"),
        row.get("prefix_cache"),
        row.get("chunked_prefill"),
        row.get("speculative_decoding"),
        row["ttft_p95_ms"],
        row["tokens_per_s"],
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    better_ttft = left["ttft_p95_ms"] <= right["ttft_p95_ms"] + 1e-9
    better_rate = left["tokens_per_s"] >= right["tokens_per_s"] - 1e-9
    strict = (
        left["ttft_p95_ms"] < right["ttft_p95_ms"] - 1e-9
        or left["tokens_per_s"] > right["tokens_per_s"] + 1e-9
    )
    return better_ttft and better_rate and strict


def _throughput_rank(point: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        point["tokens_per_s"],
        -float(point["ttft"]["p95"]),
        1 if point.get("prefix_cache") else 0,
        point.get("max_batch_tokens") or 0,
        1 if point.get("chunked_prefill") else 0,
        point["concurrency"],
    )


def _headline(
    recommendations: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    preferred = _RECOMMENDATION_WORKLOAD
    if not any(point["workload"] == preferred for point in points):
        preferred = str(recommendations[0]["workload"])
    for row in recommendations:
        if row["workload"] == preferred:
            return dict(row)
    return dict(recommendations[0])


def _primary_point(
    points: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if recommendation and recommendation.get("met"):
        for point in points:
            if _same_cell(point, recommendation):
                return point
    return _baseline(points)


def _same_cell(point: Mapping[str, Any], recommendation: Mapping[str, Any]) -> bool:
    if point["workload"] != recommendation["workload"]:
        return False
    if point["concurrency"] != recommendation.get("concurrency"):
        return False
    for key in _KNOBS:
        if key in recommendation and point.get(key) != recommendation[key]:
            return False
    return True


def _baseline(points: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    workload = _RECOMMENDATION_WORKLOAD
    if not any(point["workload"] == workload for point in points):
        workload = str(points[0]["workload"])
    pool = [point for point in points if point["workload"] == workload]

    def key(point: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            point["concurrency"],
            0 if point.get("prefix_cache") is False else 1,
            point.get("max_batch_tokens") or 0,
            0 if point.get("chunked_prefill") is False else 1,
        )

    return min(pool, key=key)


def _document(
    config: SweepConfig,
    points: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    primary: Mapping[str, Any],
    recommendation: Mapping[str, Any] | None,
    recommendations: list[dict[str, Any]] | None,
    *,
    started: datetime,
) -> dict[str, Any]:
    duration = sum(float(point["duration_s"]) for point in points)
    finished = started + timedelta(seconds=duration)
    output_tokens = int(primary["output_tokens"])
    attempted = int(primary["requests"])
    ok = int(primary["requests_ok"])
    completion = ok / attempted if attempted else 0.0
    standin = config.backend == "stand-in"
    timing = "virtual" if standin else "perf_counter"
    note = STANDIN_NOTE if standin else LIVE_NOTE
    model_sha = _sha(config.model)
    artifact = "\n".join(
        [
            config.backend,
            config.backend_version,
            config.precision,
            config.model,
            "llm-sweep",
        ]
    )
    versions = {
        "python": platform.python_version(),
        "ohho": _ohho_version(),
        config.backend: config.backend_version,
    }
    sweep: dict[str, Any] = {
        "workloads": list(config.workloads),
        "cells": len(points),
        "timing_source": timing,
        "engine_launched": False,
        "note": note,
        "primary": "recommendation"
        if recommendation and recommendation.get("met")
        else "baseline",
        "concurrencies": list(config.concurrencies),
    }
    if any("max_batch_tokens" in point for point in points):
        sweep["max_batch_tokens"] = list(config.max_batch_tokens)
    if any("prefix_cache" in point for point in points):
        sweep["prefix_cache"] = list(config.prefix_cache)
    if any("chunked_prefill" in point for point in points):
        sweep["chunked_prefill"] = list(config.chunked_prefill)
    if any("speculative_decoding" in point for point in points):
        sweep["speculative_decoding"] = list(config.speculative)
    if config.chart:
        sweep["chart"] = config.chart
    llm: dict[str, Any] = {
        "schema": LLM_BLOCK_ID,
        "backend": config.backend,
        "backend_version": config.backend_version,
        "precision": config.precision,
        "model": config.model,
        "model_sha_of": "identifier",
        "prompt": primary["prompt"],
        "workload": primary["workload"],
        "concurrency": primary["concurrency"],
        "ttft": primary["ttft"],
        "inter_token": primary["inter_token"],
        "tpot": primary["tpot"],
        "e2e": primary["e2e"],
        "tokens_per_s": primary["tokens_per_s"],
        "tokens_per_request_s": primary["tokens_per_request_s"],
        "requests_per_s": primary["requests_per_s"],
        "prompt_tokens": primary["prompt_tokens"],
        "image_tokens": primary["image_tokens"],
        "output_tokens": output_tokens,
        "requests": attempted,
        "requests_ok": ok,
        "points": points,
        "sweep": sweep,
        "pareto": pareto,
        "skipped": skipped,
    }
    if config.endpoint:
        llm["endpoint_url"] = config.endpoint
    if recommendation is not None:
        llm["recommendation"] = recommendation
    if recommendations is not None:
        llm["recommendations"] = recommendations
    return {
        "schema": SCHEMA_ID,
        "run": {
            "id": f"sweep-{config.backend}-{uuid.uuid4().hex[:8]}",
            "model_sha": model_sha,
            "artifact_sha": _sha(artifact),
            "artifact_sha_of": "serving-setup",
            "target": "llm",
            "manifest_ref": f"llm-sweep://{config.backend}",
            "host": _host(),
            "started_at": _stamp(started),
            "finished_at": _stamp(finished),
        },
        "latency": {
            **primary["e2e"],
            "stages": [
                {"name": "ttft", **primary["ttft"]},
                {"name": "inter_token", **primary["inter_token"]},
            ],
        },
        "throughput": {
            "value": primary["tokens_per_s"],
            "unit": "tokens/s",
            "samples": output_tokens,
            "batch_size": primary["concurrency"],
            "duration_s": duration,
        },
        "accuracy": {
            "primary": {
                "name": "request_completion_rate",
                "value": completion,
                "unit": "fraction",
            },
            "metrics": [
                {
                    "name": "requests_per_s",
                    "value": primary["requests_per_s"],
                    "unit": "requests/s",
                },
                {
                    "name": "tokens_per_s",
                    "value": primary["tokens_per_s"],
                    "unit": "tokens/s",
                },
            ],
            "samples": attempted,
        },
        "telemetry": {
            "duration_s": duration,
            "samples": sum(point["requests"] for point in points),
        },
        "environment": {
            "versions": versions,
            "timing_source": timing,
            "timing_note": note,
            "git_sha": git_sha(),
            **(
                {"target_profile": config.hardware_target}
                if config.hardware_target
                else {}
            ),
        },
        "llm": llm,
    }


def _present_workloads(points: Sequence[Mapping[str, Any]]) -> list[str]:
    found = []
    for name in WORKLOAD_NAMES:
        if any(point["workload"] == name for point in points):
            found.append(name)
    return found


def _point_sort_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    def flag(key: str) -> int:
        value = point.get(key)
        if value is False:
            return 0
        if value is True:
            return 1
        return 2

    return (
        _workload_index(str(point["workload"])),
        point["concurrency"],
        point.get("max_batch_tokens") or 0,
        flag("prefix_cache"),
        flag("chunked_prefill"),
        flag("speculative_decoding"),
    )


def _workload_index(name: str) -> int:
    try:
        return WORKLOAD_NAMES.index(name)
    except ValueError:
        return len(WORKLOAD_NAMES)


def _format_ms(value: float) -> str:
    return format(value, "g")


def _span(low: float, high: float) -> tuple[float, float]:
    if high - low < 1e-9:
        return low - 1.0, high + 1.0
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _host() -> dict[str, str]:
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "name": platform.node() or "localhost",
        "platform": platform.system().lower() or "unknown",
        "arch": platform.machine() or "unknown",
        "cpu": cpu,
        "accelerator": "none",
    }


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ohho_version() -> str:
    from ohho import __version__

    return __version__


def _unique_ints(values: Sequence[int], flag: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BenchHarnessError(f"{flag} needs a list of positive integers")
    seen: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BenchHarnessError(f"{flag} values must be integers >= 1")
        if value not in seen:
            seen.append(value)
    if not seen:
        raise BenchHarnessError(f"{flag} needs at least one integer")
    return tuple(seen)


def _unique_bools(values: Sequence[bool], flag: str) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BenchHarnessError(f"{flag} needs on or off")
    seen: list[bool] = []
    for value in values:
        if not isinstance(value, bool):
            raise BenchHarnessError(f"{flag} values must be on or off")
        if value not in seen:
            seen.append(value)
    if not seen:
        raise BenchHarnessError(f"{flag} needs at least one of on or off")
    return tuple(seen)


def _workloads(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BenchHarnessError("--workloads needs a list of workload names")
    seen: list[str] = []
    known = ", ".join(WORKLOAD_NAMES)
    for value in values:
        if not isinstance(value, str) or value not in WORKLOADS:
            raise BenchHarnessError(f"--workloads must be one of {known}")
        if value not in seen:
            seen.append(value)
    if not seen:
        raise BenchHarnessError("--workloads needs at least one workload")
    return tuple(seen)


def _check_count(value: int, flag: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchHarnessError(f"{flag} must be an integer >= {minimum}")
