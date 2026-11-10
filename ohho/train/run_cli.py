"""``ohho train`` implementation.

Stdout is a stream of JSON objects, one per line. Stderr repeats the human
readable error. The process exit code matches ``exit_code`` on the error or
done event. Argparse usage errors exit 2 and do not write a JSON line.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .recipe_registry import (
    RecipeRegistryError,
    TrainResult,
    ensure_loaded,
    get_family,
    load_error,
)
from .schema_validate import SchemaError
from .recipe_schema import (
    DatasetUnavailable,
    FamilyUnavailable,
    RecipeConfigError,
    RecipeError,
    TrainFailed,
    load_recipe_file,
)

STREAM = "ohho.train.progress/v1"
MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA = "ohho.train.run_manifest/v0"

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_DATASET = 4
EXIT_TRAIN_FAILED = 5

UNAVAILABLE_MESSAGE = "family not available in this build"


def execute(recipe: str, overrides: Sequence[str] | None = None) -> int:
    """Validate ``recipe``, run its family, and return a process exit code."""
    family_id = ""
    try:
        loaded = load_recipe_file(recipe, overrides)
        family_id = str(loaded.get("family", ""))
        ensure_loaded()
        family = get_family(family_id)
        if family is None or not family.available():
            return _unavailable(family_id)
        started = _now()
        result = family.run(loaded, _emit)
        manifest_path, final_loss = _write_manifest(
            recipe_path=recipe,
            overrides=list(overrides or []),
            recipe=loaded,
            result=result,
            started_at=started,
        )
        _emit_success(result, manifest_path, final_loss)
        return EXIT_OK
    except RecipeConfigError as exc:
        return _fail(EXIT_INVALID, exc.summary, errors=exc.errors, family=family_id)
    except DatasetUnavailable as exc:
        return _fail(EXIT_DATASET, str(exc), family=family_id)
    except FamilyUnavailable as exc:
        return _unavailable(exc.family or family_id)
    except TrainFailed as exc:
        return _fail(EXIT_TRAIN_FAILED, str(exc), family=family_id)
    except RecipeRegistryError as exc:
        return _fail(EXIT_INVALID, str(exc), family=family_id)
    except SchemaError as exc:
        return _fail(
            EXIT_INVALID, f"recipe schema is unusable: {exc}", family=family_id
        )
    except RecipeError as exc:
        return _fail(EXIT_INVALID, str(exc), family=family_id)
    except (KeyboardInterrupt, SystemExit, BrokenPipeError):
        raise
    except Exception as exc:
        return _fail(
            EXIT_TRAIN_FAILED,
            f"training failed: {type(exc).__name__}: {exc}",
            family=family_id,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``python -m ohho.train.run_cli``."""
    parser = argparse.ArgumentParser(prog="ohho train")
    parser.add_argument("recipe", help="path to a recipe file (.yaml or .json)")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="k=v",
        help="override a dotted field before validation (repeatable)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return execute(args.recipe, args.set)


def _unavailable(family_id: str) -> int:
    message = UNAVAILABLE_MESSAGE
    detail = load_error(family_id) if family_id else ""
    if detail:
        message = f"{message} ({detail})"
    return _fail(EXIT_UNAVAILABLE, message, family=family_id)


def _emit_success(result: TrainResult, manifest_path: Path, final_loss: Any) -> None:
    metrics_path = Path(result.metrics_path)
    _emit(
        {
            "event": "metrics",
            "path": str(metrics_path.resolve()),
            "final_loss": final_loss,
        }
    )
    _emit(
        {
            "event": "checkpoint",
            "path": str(Path(result.checkpoint_dir).resolve()),
        }
    )
    _emit({"event": "manifest", "path": str(manifest_path.resolve())})
    _emit(
        {
            "event": "done",
            "exit_code": EXIT_OK,
            "checkpoint_dir": str(Path(result.checkpoint_dir).resolve()),
            "manifest": str(manifest_path.resolve()),
        }
    )


def _write_manifest(
    *,
    recipe_path: str,
    overrides: list[str],
    recipe: dict[str, Any],
    result: TrainResult,
    started_at: str,
) -> tuple[Path, Any]:
    checkpoint_dir = Path(result.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise TrainFailed(
            f"trainer did not write a checkpoint directory: {checkpoint_dir}"
        )
    metrics_path = Path(result.metrics_path)
    if not metrics_path.is_file():
        raise TrainFailed(f"trainer did not write a metrics file: {metrics_path}")
    final_loss = None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainFailed(f"could not read metrics {metrics_path}: {exc}") from exc
    if isinstance(metrics, dict):
        final_loss = metrics.get("final_loss")
    declared = {
        "compression_hooks": _hook_names(recipe.get("compression_hooks")),
        "eval_hooks": _hook_names(recipe.get("eval_hooks")),
    }
    document = {
        "schema": MANIFEST_SCHEMA,
        "status": "succeeded",
        "family": recipe.get("family"),
        "device": result.device,
        "recipe_path": str(Path(recipe_path).resolve()),
        "overrides": overrides,
        "recipe": recipe,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "final_loss": final_loss,
        "base_weights_loaded": result.base_weights_loaded,
        "declared_hooks": declared,
        "started_at": started_at,
        "finished_at": _now(),
    }
    path = checkpoint_dir / MANIFEST_NAME
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path, final_loss


def _hook_names(hooks: Any) -> list[str]:
    if not isinstance(hooks, list):
        return []
    names: list[str] = []
    for hook in hooks:
        if isinstance(hook, dict) and isinstance(hook.get("name"), str):
            names.append(hook["name"])
    return names


def _fail(code: int, message: str, **fields: Any) -> int:
    payload: dict[str, Any] = {"event": "error", "exit_code": code, "message": message}
    for key, value in fields.items():
        if value in (None, ""):
            continue
        payload[key] = value
    _emit(payload)
    print(message, file=sys.stderr, flush=True)
    errors = fields.get("errors")
    if isinstance(errors, list):
        for line in errors:
            if line != message:
                print(line, file=sys.stderr, flush=True)
    return code


def _emit(fields: dict[str, Any]) -> None:
    body = dict(fields)
    body.pop("stream", None)
    payload = {"stream": STREAM, **body}
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
