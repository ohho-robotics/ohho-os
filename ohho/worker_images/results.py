"""Job result records. Every result names the worker image digest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import parse_image_ref, variant_by_tag

RESULT_SCHEMA = "ohho.worker_job_result.v1"
STATUSES = ("verified", "succeeded", "failed")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_job_result(
    *,
    job_id: str,
    status: str,
    image: str,
    image_digest: str,
) -> dict[str, Any]:
    """Return a job result that includes the image digest.

    ``image`` must be a pinned tag. ``image_digest`` is ``sha256:`` plus 64
    hex characters, the value ``docker image inspect`` reports as ``Id``
    before a registry push, or the registry digest after one.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise WorkerImageError("job_id is required")
    if status not in STATUSES:
        raise WorkerImageError(f"job status {status!r} is not supported")
    parsed = parse_image_ref(image)
    row = variant_by_tag(parsed.tag)
    digest = image_digest.strip()
    if _DIGEST.fullmatch(digest) is None:
        raise WorkerImageError("image_digest must be sha256 and 64 hex characters")
    if parsed.digest and parsed.digest != digest:
        raise WorkerImageError("image_digest does not match the image reference")
    pins = row.pins()
    return {
        "schema": RESULT_SCHEMA,
        "job_id": job_id.strip(),
        "status": status,
        "image": row.tag,
        "image_digest": digest,
        "variant": row.variant,
        "version": row.version,
        "base_image": row.base_image,
        "torch": pins["torch"],
        "ort": pins["ort"],
        "tensorrt": pins["tensorrt"],
        "cuda": pins["cuda"],
        "jetpack": pins["jetpack"],
        "l4t": pins["l4t"],
    }


def write_job_result(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write ``build_job_result`` JSON to ``path`` and return the document."""
    document = build_job_result(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document
