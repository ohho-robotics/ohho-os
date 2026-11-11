"""Compare a reference model to a candidate and decide pass or fail.

The default fail is a mean absolute difference above 0.05. Action families
also report per-joint error and action-chunk L2. Classifier and LLM families
report top-1 agreement and logit KL, and those two are gated by the built-in
family limits unless a threshold file replaces them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.inputs import batches_sha256, load_batches, parse_input_source
from ohho.parity.metrics import (
    action_chunk_error,
    logits_kl,
    per_joint_error,
    shape_of,
    tensor_metrics,
    top1_agreement,
)
from ohho.parity.report import build_report
from ohho.parity.runners import (
    prepare_reference,
    resolve_candidate,
    run_candidate,
    run_prepared,
)
from ohho.parity.thresholds import Thresholds, resolve_thresholds

_ACTION_FAMILIES = frozenset(
    {"act", "rl_mlp", "diffusion", "smolvla", "openvla", "policy"}
)
_LOGIT_FAMILIES = frozenset({"classifier", "llm", "vlm"})
_RUNNABLE_ROLES = ("onnx", "ort-int8", "torchscript")
_UNCHECKED_REASON = {
    "engine": "TensorRT engine execution is a GPU follow-up and was not compared.",
    "awq": "AWQ weights were not executed.",
    "gptq": "GPTQ weights were not executed.",
    "bitsandbytes": "bitsandbytes weights were not executed.",
    "nf4": "NF4 weights were not executed.",
    "fp8": "FP8 weights were not executed.",
}
_NOT_RUNNABLE = (
    "no runnable ONNX or TorchScript output; TensorRT engines and weight-quant "
    "artifacts are not executed in this process"
)


def evaluate(
    reference: str,
    candidate: str,
    *,
    inputs: str = "random",
    n: int = 256,
    seed: int = 0,
    family: str | None = None,
    target: str | None = None,
    precision: str | None = None,
    mean_abs: float | None = None,
    thresholds: str | None = None,
    record_dir: str | None = None,
    submodule: str | None = None,
    display_path: str | None = None,
    unchecked: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score ``candidate`` against ``reference``. A numerical fail is ``result: fail``.

    The returned document is ``ohho.parity/v1``. This function does not raise
    for a threshold miss. It raises :class:`ParityError` when the pair cannot
    be scored.
    """
    if n < 1:
        raise ParityError("--n must be at least 1.")
    if type(seed) is not int:
        raise ParityError("--seed must be an integer.")
    path, fmt = resolve_candidate(candidate)
    prepared = prepare_reference(reference, submodule)
    if not prepared.inputs:
        raise ParityError(f"{prepared.ref} has no input shapes.")
    effective = family or prepared.kind
    limits = resolve_thresholds(
        effective,
        target,
        precision,
        mean_abs=mean_abs,
        path=thresholds,
    )
    bundle = load_batches(
        inputs, prepared.inputs, n=n, seed=seed, record_dir=record_dir
    )
    reference_out = run_prepared(prepared, bundle.batches)
    candidate_out = run_candidate(
        path, fmt, bundle.batches, prepared.inputs, prepared.outputs
    )
    rows = _score(effective, _align(reference_out, candidate_out))
    result, reasons = decide(rows, limits)
    shown = display_path if display_path is not None else str(path)
    return build_report(
        result=result,
        reasons=reasons,
        reference={"ref": prepared.ref, "kind": prepared.kind, "family": effective},
        candidate={"path": shown, "format": fmt},
        target=target,
        precision=precision,
        thresholds=limits.to_json(),
        inputs={
            "source": bundle.source,
            "path": bundle.path,
            "seed": bundle.seed,
            "n_requested": bundle.n_requested,
            "n": bundle.n,
            "indices": bundle.indices,
            "sha256": batches_sha256(bundle.batches),
            "dtype": "float32",
        },
        outputs=rows,
        unchecked=unchecked,
    )


def compare_export(
    out_dir: str | Path,
    outputs: list[dict[str, Any]],
    *,
    reference: str,
    inputs: str,
    n: int,
    seed: int,
    family: str | None,
    target: str,
    precision: str,
    mean_abs: float | None,
    thresholds: str | None,
    submodule: str | None,
    record_dir: str | None,
) -> dict[str, Any]:
    """Score the runnable file in an optimize output directory.

    An engine or a weight-quant directory is listed under ``unchecked``. When
    nothing in the directory can be executed, the report fails closed.
    """
    out = Path(out_dir)
    candidate = select_runnable(out, outputs)
    unchecked = unchecked_entries(out, outputs, candidate)
    if candidate is None:
        return _not_runnable(
            reference=reference,
            inputs=inputs,
            n=n,
            seed=seed,
            family=family,
            target=target,
            precision=precision,
            mean_abs=mean_abs,
            thresholds=thresholds,
            unchecked=unchecked,
        )
    try:
        shown = candidate.resolve().relative_to(out.resolve()).as_posix()
    except ValueError:
        shown = candidate.name
    return evaluate(
        reference,
        str(candidate),
        inputs=inputs,
        n=n,
        seed=seed,
        family=family,
        target=target,
        precision=precision,
        mean_abs=mean_abs,
        thresholds=thresholds,
        record_dir=record_dir,
        submodule=submodule,
        display_path=shown,
        unchecked=unchecked,
    )


def select_runnable(out_dir: Path, outputs: list[dict[str, Any]]) -> Path | None:
    """Prefer the ONNX graph, then TorchScript. Engines are not runnable here."""
    by_role: dict[str, Path] = {}
    for item in outputs:
        role = str(item.get("role", ""))
        by_role.setdefault(role, out_dir / str(item.get("path", "")))
    for role in _RUNNABLE_ROLES:
        path = by_role.get(role)
        if path is not None and path.is_file():
            return path
    for name in ("model.onnx", "model.ts"):
        path = out_dir / name
        if path.is_file():
            return path
    return None


def unchecked_entries(
    out_dir: Path, outputs: list[dict[str, Any]], compared: Path | None
) -> list[dict[str, str]]:
    """Outputs that were written and not numerically compared."""
    compared_resolved = compared.resolve() if compared is not None else None
    rows: list[dict[str, str]] = []
    for item in outputs:
        path = out_dir / str(item.get("path", ""))
        if (
            compared_resolved is not None
            and path.is_file()
            and path.resolve() == compared_resolved
        ):
            continue
        role = str(item.get("role", ""))
        rows.append(
            {
                "role": role,
                "path": str(item.get("path", "")),
                "reason": _UNCHECKED_REASON.get(
                    role, "This output was not executed by the parity gate."
                ),
            }
        )
    return rows


def decide(outputs: list[dict[str, Any]], limits: Thresholds) -> tuple[str, list[str]]:
    """Return ``pass`` or ``fail`` and the reasons a limit was missed.

    A value equal to the limit passes. Only metrics that were computed, and
    limits that were set, are gated.
    """
    reasons: list[str] = []
    for item in outputs:
        name = str(item["name"])
        _over_max(reasons, name, "mean abs diff", item["mean_abs"], limits.mean_abs)
        if limits.max_abs is not None:
            _over_max(reasons, name, "max abs diff", item["max_abs"], limits.max_abs)
        if limits.rmse is not None:
            _over_max(reasons, name, "rmse", item["rmse"], limits.rmse)
        if limits.cosine_min is not None and item["cosine"] < limits.cosine_min:
            reasons.append(
                f"{name}: cosine {_fmt(item['cosine'])} < {_fmt(limits.cosine_min)}"
            )
        joints = item.get("per_joint")
        if limits.per_joint is not None and isinstance(joints, list):
            for index, value in enumerate(joints):
                if value > limits.per_joint:
                    reasons.append(
                        f"{name}: joint {index} error {_fmt(value)} > {_fmt(limits.per_joint)}"
                    )
        chunk = item.get("action_chunk")
        if (
            limits.action_chunk is not None
            and chunk is not None
            and chunk > limits.action_chunk
        ):
            reasons.append(
                f"{name}: action chunk error {_fmt(chunk)} > {_fmt(limits.action_chunk)}"
            )
        top1 = item.get("top1")
        if limits.top1_min is not None and top1 is not None and top1 < limits.top1_min:
            reasons.append(
                f"{name}: top-1 agreement {_fmt(top1)} < {_fmt(limits.top1_min)}"
            )
        kl = item.get("kl")
        if limits.kl_max is not None and kl is not None and kl > limits.kl_max:
            reasons.append(f"{name}: logits KL {_fmt(kl)} > {_fmt(limits.kl_max)}")
    if reasons:
        return "fail", reasons
    return "pass", []


def _score(family: str, pairs: list[tuple[str, Any, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, reference, candidate in pairs:
        try:
            stats = tensor_metrics(reference, candidate)
        except ParityError as exc:
            raise ParityError(f"{name}: {exc}") from exc
        action = _wants_action(family, name)
        logits = _wants_logits(family, name)
        per_joint = None
        chunk = None
        top1 = None
        kl = None
        if action:
            try:
                per_joint = per_joint_error(reference, candidate)
                chunk = action_chunk_error(reference, candidate)
            except ParityError as exc:
                raise ParityError(f"{name}: {exc}") from exc
        if logits:
            try:
                top1 = top1_agreement(reference, candidate)
                kl = logits_kl(reference, candidate)
            except ParityError as exc:
                raise ParityError(f"{name}: {exc}") from exc
        rows.append(
            {
                "name": name,
                "shape": list(shape_of(reference)),
                "mean_abs": stats["mean_abs"],
                "max_abs": stats["max_abs"],
                "rmse": stats["rmse"],
                "cosine": stats["cosine"],
                "per_joint": per_joint,
                "action_chunk": chunk,
                "top1": top1,
                "kl": kl,
            }
        )
    return rows


def _align(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    if not reference or not candidate:
        raise ParityError("the model returned no outputs.")
    if set(reference) == set(candidate):
        return [(name, reference[name], candidate[name]) for name in reference]
    if len(reference) == len(candidate):
        return [
            (name, reference[name], candidate[other])
            for name, other in zip(reference, candidate)
        ]
    raise ParityError(
        "Output names do not match. "
        f"Reference: {', '.join(reference)}. "
        f"Candidate: {', '.join(candidate)}."
    )


def _wants_action(family: str, name: str) -> bool:
    if "action" in name.lower():
        return True
    return family in _ACTION_FAMILIES


def _wants_logits(family: str, name: str) -> bool:
    if "logit" in name.lower():
        return True
    return family in _LOGIT_FAMILIES


def _over_max(
    reasons: list[str], name: str, label: str, value: float, limit: float
) -> None:
    if value > limit:
        reasons.append(f"{name}: {label} {_fmt(value)} > {_fmt(limit)}")


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _not_runnable(
    *,
    reference: str,
    inputs: str,
    n: int,
    seed: int,
    family: str | None,
    target: str,
    precision: str,
    mean_abs: float | None,
    thresholds: str | None,
    unchecked: list[dict[str, str]],
) -> dict[str, Any]:
    kind, path = parse_input_source(inputs)
    if kind == "random":
        source = "random"
        source_path = None
    elif kind == "recorded":
        source = "recorded"
        source_path = path
    else:
        source = "dataset"
        source_path = path
    effective = family or "unknown"
    limits = resolve_thresholds(
        effective, target, precision, mean_abs=mean_abs, path=thresholds
    )
    return build_report(
        result="fail",
        reasons=[_NOT_RUNNABLE],
        reference={"ref": reference, "kind": "unknown", "family": effective},
        candidate={"path": None, "format": None},
        target=target,
        precision=precision,
        thresholds=limits.to_json(),
        inputs={
            "source": source,
            "path": source_path,
            "seed": seed,
            "n_requested": n,
            "n": 0,
            "indices": None,
            "sha256": None,
            "dtype": "float32",
        },
        outputs=[],
        unchecked=unchecked,
    )
