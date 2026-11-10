"""Run a benchmark and shape it as ohho.bench/v1.

The measurement and the gates are separate. :func:`run_benchmark` times the
runner. :func:`finalize_result` adds a baseline comparison, gates, and an
optional parity document. :func:`execute` does both, and checks gate
expressions before the first iteration.

``run.target`` is the workload class (``inference`` unless the caller sets
another). ``--target`` on the command line is the hardware profile id and is
stored at ``environment.target_profile``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohho.bench.clock import clock_for_providers
from ohho.bench.compare import compare_results
from ohho.bench.cost import CostOverrides, attach_cost, require_price
from ohho.bench.environment import (
    MemoryTrace,
    collect_environment,
    gpu_telemetry,
    host_block,
)
from ohho.bench.errors import BenchHarnessError
from ohho.bench.gates import check_gate_expressions, evaluate_gates
from ohho.bench.runner import (
    STAGE_NAMES,
    OnnxArtifactRunner,
    resolve_artifact,
    session_providers,
)
from ohho.bench.schema import SCHEMA_ID, BenchValidationError, validate
from ohho.bench.stats import summarize
from ohho.targets.errors import TargetProfileError
from ohho.targets.schema import get_profile


@dataclass(frozen=True)
class BenchConfig:
    """What to measure. Gates and the baseline live on :func:`execute`."""

    artifact: str
    target: str
    warmup: int = 10
    iters: int = 50
    batches: tuple[int, ...] = (1,)
    inputs: str = "random"
    workload: str = "inference"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, str) or not self.artifact.strip():
            raise BenchHarnessError("--artifact is required")
        if not isinstance(self.target, str) or not self.target.strip():
            raise BenchHarnessError("--target is required")
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or self.warmup < 0
        ):
            raise BenchHarnessError("--warmup must be a non-negative integer")
        if (
            isinstance(self.iters, bool)
            or not isinstance(self.iters, int)
            or self.iters < 1
        ):
            raise BenchHarnessError("--iters must be a positive integer")
        if not self.batches or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in self.batches
        ):
            raise BenchHarnessError("--batch must list positive integers")
        inputs = self.inputs.strip() if isinstance(self.inputs, str) else ""
        if inputs != "random":
            raise BenchHarnessError(
                "dataset inputs are not yet supported "
                f"(got {self.inputs!r}); use --inputs random"
            )
        workload = self.workload.strip() if isinstance(self.workload, str) else ""
        if not workload:
            raise BenchHarnessError("--workload must be a non-empty string")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "target", self.target.strip())
        object.__setattr__(self, "batches", tuple(self.batches))


def primary_batch_size(batches: Sequence[int]) -> int:
    """Batch size 1 when it was requested, otherwise the first entry."""
    if 1 in batches:
        return 1
    return batches[0]


def run_benchmark(
    config: BenchConfig,
    *,
    runner: Any = None,
    clock: Any = None,
    cost: CostOverrides | None = None,
) -> dict[str, Any]:
    """Time ``config`` and return a schema-valid ohho.bench/v1 document.

    Pass ``runner`` to measure a custom five-stage object instead of an ONNX
    session. Pass ``clock`` to force :class:`~ohho.bench.clock.PerfClock`,
    :class:`~ohho.bench.clock.CudaEventClock`, or a
    :class:`~ohho.bench.clock.VirtualClock`.
    """
    try:
        profile = get_profile(config.target)
    except TargetProfileError as exc:
        raise BenchHarnessError(str(exc)) from exc
    require_price(profile.id, cost)
    path = resolve_artifact(config.artifact)
    if runner is None:
        if path.suffix.lower() != ".onnx":
            suffix = path.suffix or "(none)"
            raise BenchHarnessError(
                f"artifact type {suffix} is not yet supported; pass an .onnx file"
            )
        chosen, note = session_providers(profile.execution_providers)
        runner = OnnxArtifactRunner(path, chosen, provider_note=note)
    providers = tuple(getattr(runner, "providers", ("CPUExecutionProvider",)))
    if not providers:
        providers = ("CPUExecutionProvider",)
    if clock is None:
        clock = clock_for_providers(providers)
    trace = MemoryTrace()
    trace.start()
    started = _stamp()
    try:
        reports = [
            _run_batch(runner, clock, batch, config.warmup, config.iters)
            for batch in config.batches
        ]
        memory = trace.snapshot()
    finally:
        trace.close()
    finished = _stamp()
    primary_size = primary_batch_size(config.batches)
    primary = next(item for item in reports if item["batch_size"] == primary_size)
    for item in reports:
        item["primary"] = item is primary
    note = _timing_note(
        providers,
        clock,
        str(getattr(runner, "provider_note", "") or ""),
    )
    environment, gpu = collect_environment(
        profile,
        timing_source=str(getattr(clock, "kind", "perf_counter")),
        execution_provider=providers[0],
        execution_providers=providers,
        timing_note=note,
    )
    digest = _file_sha256(path)
    telemetry: dict[str, Any] = {
        "duration_s": primary["throughput"]["duration_s"],
        "samples": primary["throughput"]["samples"],
    }
    telemetry.update(memory.as_telemetry())
    telemetry.update(gpu_telemetry(gpu))
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "run": {
            "id": f"bench-{uuid.uuid4().hex[:12]}",
            "model_sha": digest,
            "artifact_sha": digest,
            "target": config.workload,
            "manifest_ref": config.artifact,
            "host": host_block(profile),
            "started_at": started,
            "finished_at": finished,
        },
        "latency": primary["latency"],
        "throughput": primary["throughput"],
        "accuracy": {
            "primary": {"name": "not_measured", "value": 0.0, "unit": "1"},
            "samples": 0,
        },
        "telemetry": telemetry,
        "environment": environment,
        "batches": reports,
    }
    attach_cost(document, cost, profile_id=profile.id)
    _require_valid(document)
    return document


def finalize_result(
    document: dict[str, Any],
    *,
    baseline_path: str | None = None,
    slos: Sequence[str] = (),
    max_regressions: Sequence[str] = (),
    parity_path: str | None = None,
) -> dict[str, Any]:
    """Copy ``document`` and attach comparison, gates, and parity."""
    result = copy.deepcopy(document)
    if baseline_path:
        baseline = _load_json(baseline_path, "baseline")
        if not isinstance(baseline, dict):
            raise BenchHarnessError("baseline must be a JSON object")
        try:
            validate(baseline)
        except BenchValidationError as exc:
            raise BenchHarnessError(f"baseline is not ohho.bench/v1:\n{exc}") from exc
        result["comparison"] = compare_results(
            result, baseline, baseline_ref=baseline_path
        )
    if slos or max_regressions:
        result["gates"] = evaluate_gates(result, slos, max_regressions)
    if parity_path:
        result["parity"] = _load_json(parity_path, "parity file")
    _require_valid(result)
    return result


def execute(
    config: BenchConfig,
    *,
    runner: Any = None,
    clock: Any = None,
    baseline_path: str | None = None,
    slos: Sequence[str] = (),
    max_regressions: Sequence[str] = (),
    parity_path: str | None = None,
    cost: CostOverrides | None = None,
) -> dict[str, Any]:
    """Measure, then apply baseline, gates, parity, and the cost estimate.

    Gate expressions are checked before the runner is called. Cost uses the
    measured throughput and does not add iterations.
    """
    check_gate_expressions(slos, max_regressions, has_baseline=bool(baseline_path))
    document = run_benchmark(config, runner=runner, clock=clock, cost=cost)
    return finalize_result(
        document,
        baseline_path=baseline_path,
        slos=slos,
        max_regressions=max_regressions,
        parity_path=parity_path,
    )


def _run_batch(runner: Any, clock: Any, batch: int, warmup: int, iters: int) -> dict:
    cold = clock.measure(runner, batch)
    for _ in range(warmup):
        clock.measure(runner, batch)
    stage_samples = {name: [] for name in STAGE_NAMES}
    totals: list[float] = []
    for _ in range(iters):
        sample = clock.measure(runner, batch)
        for name in STAGE_NAMES:
            stage_samples[name].append(float(sample[name]))
        totals.append(sum(float(sample[name]) for name in STAGE_NAMES))
    latency = _latency_report(
        totals,
        stage_samples,
        sum(float(cold[name]) for name in STAGE_NAMES),
    )
    return {
        "batch_size": batch,
        "latency": latency,
        "throughput": {
            "value": latency["achievable_hz"],
            "unit": "Hz",
            "samples": iters,
            "batch_size": batch,
            "duration_s": sum(totals),
        },
    }


def _latency_report(
    totals_s: list[float],
    stage_samples: dict[str, list[float]],
    cold_s: float,
) -> dict[str, Any]:
    stats = summarize([item * 1000.0 for item in totals_s])
    stages = []
    for name in STAGE_NAMES:
        block = summarize([item * 1000.0 for item in stage_samples[name]])
        block["name"] = name
        block["unit"] = "ms"
        block["samples"] = len(stage_samples[name])
        stages.append(block)
    mean_s = stats["mean"] / 1000.0
    stats["unit"] = "ms"
    stats["samples"] = len(totals_s)
    stats["stages"] = stages
    stats["cold_start_ms"] = cold_s * 1000.0
    stats["achievable_hz"] = 0.0 if mean_s == 0 else 1.0 / mean_s
    return stats


def _timing_note(providers: tuple[str, ...], clock: Any, provider_note: str) -> str:
    parts = [provider_note] if provider_note else []
    names = " ".join(providers)
    wants_gpu = "CUDA" in names or "Tensorrt" in names or "TensorRT" in names
    if wants_gpu and getattr(clock, "kind", "") != "cuda_event":
        parts.append(
            "CUDA events were not used because this process has no CUDA device. "
            "Stages were timed with perf_counter."
        )
    return " ".join(parts)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: str, label: str) -> Any:
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BenchHarnessError(f"{label} not found: {path}") from exc
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BenchHarnessError(f"cannot read {label}: {detail}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchHarnessError(
            f"{label} is not JSON: {exc.msg} (line {exc.lineno})"
        ) from exc


def _require_valid(document: dict[str, Any]) -> None:
    try:
        validate(document)
    except BenchValidationError as exc:
        raise BenchHarnessError(
            f"bench result does not match ohho.bench/v1:\n{exc}",
            exit_code=1,
        ) from exc
