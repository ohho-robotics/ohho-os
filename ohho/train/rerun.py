"""Re-run a training run from its ``ohho.run/v1`` manifest and compare metrics.

``ohho run rerun <id>`` reads ``run_manifest.json``, trains
``resolved_config`` again into a new directory, and compares the final
metrics with the original. The original directory is not modified.

``<id>`` is the run directory, or the path of ``run_manifest.json`` itself.
The same strings ``ohho run show`` accepts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .run_manifest import (
    MANIFEST_NAME,
    resolve_manifest_path,
    validate_run_manifest,
)
from .variance import CONSOLE_FOLLOWUP, comparison_hint

_COMPARED = (
    "final_loss",
    "validation_loss",
    "loss_curve",
    "validation_curve",
    "completed_steps",
    "train_steps",
    "stopped_early",
)


class RerunError(ValueError):
    """The requested run cannot be re-run."""


@dataclass
class RerunResult:
    """Comparison of one re-run against the manifest it came from."""

    run_id: str
    original_dir: str
    rerun_dir: str
    family: str
    deterministic: bool
    train_exit: int
    exact: bool
    within_noise: bool
    original_final_loss: Any
    rerun_final_loss: Any
    abs_delta: float | None
    expected_variance: float | None
    tolerance: float | None
    error: str | None
    compared: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON document printed by ``ohho run rerun --json``."""
        return {
            "schema": "ohho.run.rerun/v1",
            "id": self.run_id,
            "original": self.original_dir,
            "rerun": self.rerun_dir,
            "family": self.family,
            "deterministic": self.deterministic,
            "metric": "final_loss",
            "original_final_loss": self.original_final_loss,
            "rerun_final_loss": self.rerun_final_loss,
            "abs_delta": self.abs_delta,
            "exact": self.exact,
            "expected_variance": self.expected_variance,
            "tolerance": self.tolerance,
            "within_noise": self.within_noise,
            "train_exit": self.train_exit,
            "error": self.error,
            "compared": self.compared,
            "note": self.note,
            "console_followup": CONSOLE_FOLLOWUP,
        }

    def text(self) -> str:
        """Human report. The same fields are in :meth:`to_dict`."""
        lines = [
            "ohho.run.rerun/v1",
            f"id: {self.run_id}",
            f"original: {self.original_dir}",
            f"rerun: {self.rerun_dir}",
            f"family: {self.family}",
            f"deterministic: {_format(self.deterministic)}",
            "metric: final_loss",
            f"original_final_loss: {_format(self.original_final_loss)}",
            f"rerun_final_loss: {_format(self.rerun_final_loss)}",
            f"abs_delta: {_format(self.abs_delta)}",
            f"exact: {_format(self.exact)}",
            f"expected_variance: {_format(self.expected_variance)}",
            f"tolerance: {_format(self.tolerance)}",
            f"within_noise: {_format(self.within_noise)}",
            f"train_exit: {self.train_exit}",
            f"error: {_format(self.error)}",
            CONSOLE_FOLLOWUP,
        ]
        return "\n".join(lines) + "\n"


def rerun_run(
    run_id: str | Path, *, output_dir: str | Path | None = None
) -> RerunResult:
    """Re-run ``run_id`` from its manifest and compare final metrics.

    Raises :class:`RerunError` when the manifest is missing, invalid, or not
    a succeeded run, or when ``output_dir`` cannot be used. A trainer failure
    is a result with a non-zero ``train_exit``, not an exception.
    """
    manifest_path = resolve_manifest_path(run_id)
    if manifest_path is None:
        raise RerunError(f"no run manifest in {Path(run_id).expanduser()}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RerunError(f"could not read run manifest: {exc}") from exc
    errors = validate_run_manifest(document)
    if errors:
        raise RerunError("run manifest is not valid ohho.run/v1: " + "; ".join(errors))
    if document.get("status") != "succeeded":
        raise RerunError(
            f"only a succeeded run can be re-run (status is {document.get('status')!r})"
        )
    recipe = _recipe_from_manifest(document)
    original_dir = _original_dir(document, manifest_path)
    rerun_dir = _choose_output(original_dir, output_dir)
    if rerun_dir.resolve() == original_dir.resolve():
        raise RerunError(
            "rerun output must be a different directory from the original run"
        )
    if rerun_dir.exists() and any(rerun_dir.iterdir()):
        raise RerunError(f"rerun output is not empty: {rerun_dir}")
    rerun_dir.mkdir(parents=True, exist_ok=True)
    recipe["output"] = {"dir": str(rerun_dir)}
    recipe_path = rerun_dir / "recipe.resolved.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    code, events, stderr = _execute_captured(recipe_path)
    rerun_manifest = _load_optional(rerun_dir / MANIFEST_NAME)
    original_metrics = _read_metrics(document.get("metrics_path"))
    rerun_metrics_path = None
    if isinstance(rerun_manifest, dict):
        rerun_metrics_path = rerun_manifest.get("metrics_path")
    rerun_metrics = _read_metrics(rerun_metrics_path)
    if rerun_metrics is None:
        rerun_metrics = _read_metrics(rerun_dir / "metrics.json")
    original_loss = _loss(document, original_metrics)
    rerun_loss = _loss(
        rerun_manifest if isinstance(rerun_manifest, dict) else {}, rerun_metrics
    )
    deterministic = recipe.get("deterministic") is True
    family = str(document.get("family") or recipe.get("family") or "")
    hint = comparison_hint(
        family,
        original_loss if _is_number(original_loss) else None,
        rerun_loss if _is_number(rerun_loss) else None,
        deterministic=deterministic,
    )
    compared, metrics_exact = _compare_metrics(original_metrics, rerun_metrics)
    if not compared and _is_number(original_loss) and _is_number(rerun_loss):
        metrics_exact = bool(hint["exact"])
    error = _error_message(events, stderr, code)
    return RerunResult(
        run_id=str(manifest_path),
        original_dir=str(original_dir),
        rerun_dir=str(rerun_dir),
        family=family,
        deterministic=deterministic,
        train_exit=code,
        exact=code == 0 and metrics_exact,
        within_noise=bool(hint["within_noise"]),
        original_final_loss=original_loss,
        rerun_final_loss=rerun_loss,
        abs_delta=hint["abs_delta"] if isinstance(hint["abs_delta"], float) else None,
        expected_variance=_optional_float(hint["expected_variance"]),
        tolerance=_optional_float(hint["tolerance"]),
        error=error,
        compared=compared,
        note=str(hint["note"]),
    )


def rerun_cli(
    run_id: str,
    *,
    output_dir: str | None = None,
    as_json: bool = False,
) -> int:
    """CLI body for ``ohho run rerun``.

    Exit ``0`` when the re-run finishes and the final metrics match exactly.
    Exit ``1`` when the manifest cannot be re-run or the metrics differ.
    A trainer failure returns that trainer's exit code.
    """
    try:
        result = rerun_run(run_id, output_dir=output_dir)
    except RerunError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if as_json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(result.text())
    if result.train_exit != 0:
        return result.train_exit
    if not result.exact:
        return 1
    return 0


def default_rerun_directory(original: Path) -> Path:
    """Sibling directory that does not exist yet."""
    parent = original.parent
    stem = original.name + "-rerun"
    candidate = parent / stem
    index = 2
    while candidate.exists():
        candidate = parent / f"{stem}-{index}"
        index += 1
        if index > 1000:
            raise RerunError(f"could not choose a rerun directory next to {original}")
    return candidate


def _execute_captured(recipe_path: Path) -> tuple[int, list[dict[str, Any]], str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from .run_cli import execute

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(recipe_path))
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _recipe_from_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    raw = document.get("resolved_config")
    if not isinstance(raw, dict):
        raise RerunError("manifest resolved_config is not an object")
    recipe = json.loads(json.dumps(raw))
    dataset = document.get("dataset")
    recorded = dataset.get("path") if isinstance(dataset, dict) else None
    if isinstance(recorded, str) and recorded.strip() and Path(recorded).exists():
        recipe["dataset"] = {"path": recorded}
    if not isinstance(recipe.get("dataset"), dict):
        raise RerunError("manifest has no dataset to re-run")
    if "schema" not in recipe:
        recipe["schema"] = "ohho.recipe/v1"
    return recipe


def _original_dir(document: Mapping[str, Any], manifest_path: Path) -> Path:
    raw = document.get("checkpoint_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return manifest_path.parent.resolve()


def _choose_output(original: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return default_rerun_directory(original)
    return Path(output_dir).expanduser().resolve()


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(document, dict):
        return document
    return None


def _read_metrics(path: Any) -> dict[str, Any] | None:
    if not isinstance(path, (str, Path)):
        return None
    return _load_optional(Path(path))


def _loss(manifest: Mapping[str, Any], metrics: dict[str, Any] | None) -> Any:
    if isinstance(metrics, dict) and "final_loss" in metrics:
        return metrics["final_loss"]
    return manifest.get("final_loss")


def _compare_metrics(
    original: dict[str, Any] | None, rerun: dict[str, Any] | None
) -> tuple[dict[str, Any], bool]:
    if not isinstance(original, dict) or not isinstance(rerun, dict):
        return {}, False
    compared: dict[str, Any] = {}
    exact = True
    saw = False
    for key in _COMPARED:
        if key not in original and key not in rerun:
            continue
        saw = True
        if key not in original or key not in rerun:
            compared[key] = {"exact": False}
            exact = False
            continue
        same = _values_equal(original[key], rerun[key])
        entry: dict[str, Any] = {"exact": same}
        if _is_number(original[key]) and _is_number(rerun[key]):
            entry["abs_delta"] = abs(float(original[key]) - float(rerun[key]))
        compared[key] = entry
        if not same:
            exact = False
    if not saw:
        return {}, False
    return compared, exact


def _error_message(events: list[dict[str, Any]], stderr: str, code: int) -> str | None:
    if code == 0:
        return None
    for event in events:
        if event.get("event") == "error" and isinstance(event.get("message"), str):
            return event["message"]
    text = stderr.strip()
    if text:
        return text.splitlines()[-1]
    return f"re-run exited {code}"


def _values_equal(left: Any, right: Any) -> bool:
    return _normalise(left) == _normalise(right)


def _normalise(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional_float(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None


def _format(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)
