"""``ohho train`` implementation.

Stdout is a stream of JSON objects, one per line. Stderr repeats the human
readable error. The process exit code matches ``exit_code`` on the error or
done event. Argparse usage errors exit 2 and do not write a JSON line.

A schema-valid family that emits ``train_start`` gets an ``ohho.run/v1``
manifest in its output directory. The file is finalised on success, failure,
and interrupt. Preflight failures before ``train_start`` do not write one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .recipe_registry import (
    RecipeRegistryError,
    TrainResult,
    ensure_loaded,
    get_family,
    load_error,
)
from .recipe_schema import (
    DatasetUnavailable,
    FamilyUnavailable,
    RecipeConfigError,
    RecipeError,
    TrainFailed,
    load_recipe_file,
)
from ..metrics import clear_last_metrics_run, wandb_url_for
from .run_manifest import (
    MANIFEST_NAME,
    RunCapture,
    capture_run,
    manifest_from_capture,
    utc_now,
    write_run_manifest,
)
from .schema_validate import SchemaError

STREAM = "ohho.train.progress/v1"

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_DATASET = 4
EXIT_TRAIN_FAILED = 5

UNAVAILABLE_MESSAGE = "family not available in this build"


class _RunSession:
    """Writes the run manifest at start and finalises it at the end."""

    def __init__(self) -> None:
        self.started = False
        self.final = False
        self.path: Path | None = None
        self.output_dir: Path | None = None
        self.capture: RunCapture | None = None
        self.recipe_path = ""
        self.overrides: list[str] = []
        self.recipe: dict[str, Any] = {}

    def bind(
        self, *, recipe_path: str, overrides: list[str], recipe: dict[str, Any]
    ) -> None:
        self.recipe_path = recipe_path
        self.overrides = overrides
        self.recipe = recipe
        self.output_dir = Path(str(recipe["output"]["dir"])).expanduser().resolve()

    def begin(self, device: str) -> None:
        if self.started or self.output_dir is None:
            return
        capture = capture_run(
            recipe=self.recipe,
            recipe_path=self.recipe_path,
            overrides=self.overrides,
            device=device,
            started_at=utc_now(),
        )
        manifest = manifest_from_capture(
            capture,
            status="running",
            finished_at=None,
            error=None,
            checkpoint_dir=str(self.output_dir),
            metrics_path=None,
            final_loss=None,
            base_weights_loaded=False,
            wandb_url=None,
        )
        path = self.output_dir / MANIFEST_NAME
        write_run_manifest(path, manifest)
        self.capture = capture
        self.path = path
        self.started = True

    def succeed(self, result: TrainResult) -> tuple[Path, Any]:
        checkpoint_dir = Path(result.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_dir():
            raise TrainFailed(
                f"trainer did not write a checkpoint directory: {checkpoint_dir}"
            )
        metrics_path = Path(result.metrics_path).expanduser()
        if not metrics_path.is_file():
            raise TrainFailed(f"trainer did not write a metrics file: {metrics_path}")
        final_loss = _read_final_loss(metrics_path)
        if self.capture is None:
            self.begin(result.device)
        if self.capture is None:
            raise TrainFailed("run manifest was not opened")
        manifest = manifest_from_capture(
            self.capture,
            status="succeeded",
            finished_at=utc_now(),
            error=None,
            checkpoint_dir=str(checkpoint_dir.resolve()),
            metrics_path=str(metrics_path.resolve()),
            final_loss=final_loss,
            base_weights_loaded=result.base_weights_loaded,
            device=result.device,
            wandb_url=wandb_url_for(checkpoint_dir) or wandb_url_for(self.output_dir),
        )
        path = write_run_manifest(checkpoint_dir / MANIFEST_NAME, manifest)
        if self.path is not None and self.path.resolve() != path.resolve():
            write_run_manifest(self.path, manifest)
        self.path = path
        self.final = True
        return path, final_loss

    def fail(self, status: str, error: str) -> None:
        if not self.started or self.final or self.capture is None or self.path is None:
            return
        try:
            manifest = manifest_from_capture(
                self.capture,
                status=status,
                finished_at=utc_now(),
                error=error,
                checkpoint_dir=str(self.output_dir) if self.output_dir else None,
                metrics_path=None,
                final_loss=None,
                base_weights_loaded=False,
                wandb_url=wandb_url_for(self.output_dir),
            )
            write_run_manifest(self.path, manifest)
            self.final = True
            _emit(
                {
                    "event": "manifest",
                    "path": str(self.path.resolve()),
                    "status": status,
                }
            )
        except Exception as exc:
            print(
                f"could not finalise run manifest: {exc}",
                file=sys.stderr,
                flush=True,
            )


def execute(recipe: str, overrides: Sequence[str] | None = None) -> int:
    """Validate ``recipe``, run its family, and return a process exit code."""
    clear_last_metrics_run()
    family_id = ""
    run = _RunSession()
    try:
        loaded = load_recipe_file(recipe, overrides)
        family_id = str(loaded.get("family", ""))
        ensure_loaded()
        family = get_family(family_id)
        if family is None or not family.available():
            return _unavailable(family_id)
        run.bind(recipe_path=recipe, overrides=list(overrides or []), recipe=loaded)

        def emit(event: dict[str, Any]) -> None:
            if event.get("event") == "train_start" and not run.started:
                device = event.get("device")
                run.begin(device if isinstance(device, str) else "")
                _emit(event)
                if run.path is not None:
                    _emit(
                        {
                            "event": "manifest",
                            "path": str(run.path.resolve()),
                            "status": "running",
                        }
                    )
                return
            _emit(event)

        result = family.run(loaded, emit)
        if not run.started:
            run.begin(result.device)
        manifest_path, final_loss = run.succeed(result)
        _emit_success(result, manifest_path, final_loss)
        return EXIT_OK
    except RecipeConfigError as exc:
        run.fail("failed", exc.summary)
        return _fail(EXIT_INVALID, exc.summary, errors=exc.errors, family=family_id)
    except DatasetUnavailable as exc:
        run.fail("failed", str(exc))
        return _fail(EXIT_DATASET, str(exc), family=family_id)
    except FamilyUnavailable as exc:
        run.fail("failed", str(exc))
        return _unavailable(exc.family or family_id)
    except TrainFailed as exc:
        run.fail("failed", str(exc))
        return _fail(EXIT_TRAIN_FAILED, str(exc), family=family_id)
    except RecipeRegistryError as exc:
        run.fail("failed", str(exc))
        return _fail(EXIT_INVALID, str(exc), family=family_id)
    except SchemaError as exc:
        run.fail("failed", f"recipe schema is unusable: {exc}")
        return _fail(
            EXIT_INVALID, f"recipe schema is unusable: {exc}", family=family_id
        )
    except RecipeError as exc:
        run.fail("failed", str(exc))
        return _fail(EXIT_INVALID, str(exc), family=family_id)
    except (KeyboardInterrupt, SystemExit, BrokenPipeError):
        run.fail("interrupted", "interrupted")
        raise
    except Exception as exc:
        message = f"training failed: {type(exc).__name__}: {exc}"
        run.fail("failed", message)
        return _fail(EXIT_TRAIN_FAILED, message, family=family_id)


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
    _emit(
        {
            "event": "manifest",
            "path": str(manifest_path.resolve()),
            "status": "succeeded",
        }
    )
    _emit(
        {
            "event": "done",
            "exit_code": EXIT_OK,
            "checkpoint_dir": str(Path(result.checkpoint_dir).resolve()),
            "manifest": str(manifest_path.resolve()),
        }
    )


def _read_final_loss(metrics_path: Path) -> Any:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainFailed(f"could not read metrics {metrics_path}: {exc}") from exc
    if isinstance(metrics, dict):
        return metrics.get("final_loss")
    return None


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


if __name__ == "__main__":
    raise SystemExit(main())
