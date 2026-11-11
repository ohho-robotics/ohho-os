"""Image stamp written into a worker image at build time."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import WorkerVariant, get_variant

STAMP_SCHEMA = "ohho.worker_image_stamp.v1"
STACKS = ("platform", "full")


def stamp_document(row: WorkerVariant, stack: str) -> dict[str, Any]:
    """JSON object stored at ``/etc/ohho/worker-image.json``."""
    if stack not in STACKS:
        raise WorkerImageError("OHHO_WORKER_STACK must be full or platform.")
    return {
        "schema": STAMP_SCHEMA,
        "tag": row.tag,
        "variant": row.variant,
        "version": row.version,
        "base_image": row.base_image,
        "stack": stack,
        "require_stack": stack == "full",
        "torch": row.torch,
        "ort": row.ort,
        "tensorrt": row.tensorrt,
        "cuda": row.cuda,
        "jetpack": row.jetpack,
        "l4t": row.l4t,
    }


def write_image_stamp(
    path: str | Path, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Write the stamp after checking the image environment against the matrix.

    The image build calls this. A pin in the environment that disagrees with
    the matrix fails the build.
    """
    env = os.environ if environ is None else environ
    variant = env.get("OHHO_WORKER_VARIANT", "").strip()
    row = get_variant(variant)
    image = env.get("OHHO_WORKER_IMAGE", "").strip()
    if image != row.tag:
        raise WorkerImageError(
            f"OHHO_WORKER_IMAGE is {image or 'unset'}, pin is {row.tag}."
        )
    stack = env.get("OHHO_WORKER_STACK", "").strip() or "platform"
    _expect(env, "OHHO_WORKER_TORCH", row.torch)
    _expect(env, "OHHO_WORKER_ORT", row.ort)
    _expect(env, "OHHO_WORKER_TENSORRT", row.tensorrt)
    _expect(env, "OHHO_WORKER_CUDA", row.cuda)
    _expect(env, "OHHO_WORKER_JETPACK", row.jetpack)
    _expect(env, "OHHO_WORKER_L4T", row.l4t)
    base = env.get("OHHO_WORKER_BASE", "").strip()
    if base and base != row.base_image:
        raise WorkerImageError(f"OHHO_WORKER_BASE is {base}, pin is {row.base_image}.")
    document = stamp_document(row, stack)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def read_stamp(path: str | Path) -> dict[str, Any]:
    """Load a stamp and refuse one that disagrees with the matrix."""
    target = Path(path)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerImageError(f"Worker image stamp {target} is unreadable.") from exc
    if not isinstance(document, dict) or document.get("schema") != STAMP_SCHEMA:
        raise WorkerImageError(f"Worker image stamp {target} is not {STAMP_SCHEMA}.")
    row = get_variant(str(document.get("variant", "")))
    if document.get("tag") != row.tag or document.get("version") != row.version:
        raise WorkerImageError(f"Worker image stamp {target} does not match {row.tag}.")
    for key, pin in (
        ("torch", row.torch),
        ("ort", row.ort),
        ("tensorrt", row.tensorrt),
        ("cuda", row.cuda),
        ("jetpack", row.jetpack),
        ("l4t", row.l4t),
        ("base_image", row.base_image),
    ):
        if document.get(key) != pin:
            raise WorkerImageError(
                f"Worker image stamp {key} does not match the matrix."
            )
    stack = document.get("stack")
    if stack not in STACKS or document.get("require_stack") != (stack == "full"):
        raise WorkerImageError("Worker image stamp stack is invalid.")
    return document


def _expect(env: Mapping[str, str], name: str, pin: str | None) -> None:
    if name not in env:
        return
    raw = env.get(name, "").strip()
    got = raw or None
    if got != pin:
        raise WorkerImageError(f"{name} is {raw or 'unset'}, pin is {pin or 'none'}.")
