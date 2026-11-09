"""Run the optional task gate after tensor parity.

:func:`evaluate_tasks` scores callables and returns an ``ohho.bench/v1``
document. It does not change the tensor report. A tolerance miss is
``accuracy.result == "fail"`` and does not raise. A set that cannot be
scored raises :class:`ohho.parity.errors.ParityError`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.tasks.bench_report import build_bench_document
from ohho.parity.tasks.episodes import (
    episodes_from_path,
    fixed_seeds,
    synthetic_episodes,
)
from ohho.parity.tasks.registry import select_evaluators
from ohho.parity.tasks.tolerances import resolve_tolerances
from ohho.parity.tasks.types import Episode, Policy, TaskContext, as_floats


def evaluate_tasks(
    reference: Policy,
    optimized: Policy,
    *,
    tasks: str = "all",
    episodes: Sequence[Episode] | None = None,
    episode_path: str | None = None,
    seeds: Sequence[int] | None = None,
    tolerances: str | None = None,
    detections: Sequence[dict[str, Any]] | None = None,
    tokens: Sequence[dict[str, Any]] | None = None,
    model: str = "reference",
    candidate: str = "optimized",
    family: str | None = None,
    sim_steps: int = 8,
    robot_id: str = "sim",
    manifest_ref: str = "ohho.parity/v1",
) -> dict[str, Any]:
    """Score ``optimized`` against ``reference`` and validate the bench document."""
    chosen = select_evaluators(tasks)
    if not chosen:
        raise ParityError("the task gate needs at least one evaluator.")
    seed_list = tuple(seeds) if seeds is not None else fixed_seeds()
    if not seed_list:
        raise ParityError("the task gate needs at least one seed.")
    held_out: tuple[Episode, ...]
    if episodes is not None:
        held_out = tuple(episodes)
    elif episode_path:
        held_out = episodes_from_path(episode_path)
    else:
        held_out = synthetic_episodes(seed_list)
    context = TaskContext(
        reference=reference,
        optimized=optimized,
        episodes=held_out,
        seeds=seed_list,
        tolerances=resolve_tolerances(model, family, tolerances),
        detections=tuple(detections) if detections is not None else None,
        tokens=tuple(tokens) if tokens is not None else None,
        sim_steps=sim_steps,
        robot_id=robot_id,
        model=model,
        family=family,
    )
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    rows = []
    for evaluator in chosen:
        rows.extend(evaluator.run(context))
    duration_s = time.perf_counter() - t0
    finished = datetime.now(timezone.utc)
    if not rows:
        raise ParityError("the task gate produced no metrics.")
    return build_bench_document(
        rows,
        model=model,
        candidate=candidate,
        family=family,
        manifest_ref=manifest_ref,
        latencies_ms=list(context.latencies_ms),
        duration_s=duration_s,
        started_at=started,
        finished_at=finished,
    )


def evaluate_cli(args: Any) -> dict[str, Any]:
    """Score the ``ohho parity`` reference and candidate when ``--tasks`` is set."""
    from ohho.parity.runners import (
        prepare_reference,
        resolve_candidate,
        run_candidate,
        run_prepared,
    )

    prepared = prepare_reference(args.reference, None)
    path, fmt = resolve_candidate(args.candidate)
    reference_policy = _policy(
        lambda batches: run_prepared(prepared, batches),
        prepared.inputs,
    )
    candidate_policy = _policy(
        lambda batches: run_candidate(
            path, fmt, batches, prepared.inputs, prepared.outputs
        ),
        prepared.inputs,
    )
    return evaluate_tasks(
        reference_policy,
        candidate_policy,
        tasks=args.tasks,
        episode_path=args.task_episodes,
        seeds=fixed_seeds(args.task_seeds),
        tolerances=args.task_tolerances,
        detections=_optional_rows(args.task_detections, "images"),
        tokens=_optional_rows(args.task_tokens, "examples"),
        model=args.reference,
        candidate=args.candidate,
        family=args.family or prepared.kind,
        manifest_ref="ohho.parity/v1",
    )


def run_task_stage(
    request: Any, out_dir: str | Path, parity_report: dict[str, Any]
) -> dict[str, Any]:
    """Score the optimize reference and the runnable candidate.

    TensorRT engines are not executed. The ONNX or TorchScript file selected
    by the tensor gate is the optimised policy.
    """
    from ohho.parity.runners import (
        prepare_reference,
        resolve_candidate,
        run_candidate,
        run_prepared,
    )

    out = Path(out_dir)
    prepared = prepare_reference(request.model, request.submodule)
    shown = (parity_report.get("candidate") or {}).get("path")
    if not isinstance(shown, str) or not shown:
        raise ParityError("task accuracy needs a runnable candidate.")
    candidate_path, fmt = resolve_candidate(str(out / shown))
    reference_policy = _policy(
        lambda batches: run_prepared(prepared, batches),
        prepared.inputs,
    )
    candidate_policy = _policy(
        lambda batches: run_candidate(
            candidate_path,
            fmt,
            batches,
            prepared.inputs,
            prepared.outputs,
        ),
        prepared.inputs,
    )
    return evaluate_tasks(
        reference_policy,
        candidate_policy,
        tasks=request.tasks,
        episode_path=request.task_episodes,
        seeds=fixed_seeds(request.task_seeds),
        tolerances=request.task_tolerances,
        detections=_optional_rows(request.task_detections, "images"),
        tokens=_optional_rows(request.task_tokens, "examples"),
        model=request.model,
        candidate=shown,
        family=request.parity_family or prepared.kind,
        manifest_ref=str(out / "manifest.json"),
    )


def _policy(execute, inputs) -> Policy:
    if not inputs:
        raise ParityError("the model has no inputs to score.")
    spec = inputs[0]
    width = int(spec.shape[-1])
    name = spec.name

    def predict(observation: object) -> list[float]:
        if isinstance(observation, dict):
            raise ParityError(
                "this model policy returns an action vector. "
                "Detection and token tasks need --task-detections or --task-tokens "
                "filled with predictions, or a callable policy."
            )
        row = as_floats(list(observation), "observation")  # type: ignore[arg-type]
        if len(row) < width:
            row = row + [0.0] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        outputs = execute({name: [row]})
        return _first_action(outputs)

    return predict


def _first_action(outputs: dict[str, Any]) -> list[float]:
    if not outputs:
        raise ParityError("the model returned no action.")
    value = next(iter(outputs.values()))
    if isinstance(value, list) and value and isinstance(value[0], list):
        row = value[0]
        if row and isinstance(row[0], list):
            row = row[0]
        return as_floats(row, "action")
    if isinstance(value, list):
        return as_floats(value, "action")
    raise ParityError("the model action is not a vector.")


def _optional_rows(path: str | None, key: str) -> list[dict[str, Any]] | None:
    if not path:
        return None
    import json

    file = Path(path)
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"could not read {file}: {exc}") from exc
    rows = document.get(key) if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ParityError(f"{file} must contain a non-empty {key} list.")
    if not all(isinstance(item, dict) for item in rows):
        raise ParityError(f"{file} {key} must be a list of objects.")
    return rows
