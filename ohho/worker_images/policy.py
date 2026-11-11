"""Console allowlist for serve worker images.

The published matrix is the default allowlist. ``OHHO_WORKER_ALLOWLIST``
narrows it. A tag outside the matrix is never allowed, and a tag the console
omits is refused. The agent calls :func:`refusal_for_environment` before it
runs.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import (
    ImageRef,
    parse_image_ref,
    variant_by_tag,
    variants,
)

ALLOWLIST_SCHEMA = "ohho.worker_allowlist.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class AllowEntry:
    """One image the console permits."""

    tag: str
    digest: str | None = None


@dataclass(frozen=True)
class AllowList:
    """Tags the console allows. Digests are optional."""

    entries: tuple[AllowEntry, ...]

    def entry_for(self, tag: str) -> AllowEntry | None:
        """Return the entry for ``tag``, or ``None`` when the console omits it."""
        for entry in self.entries:
            if entry.tag == tag:
                return entry
        return None


def default_allowlist() -> AllowList:
    """Every pinned tag, with no digest attached."""
    return AllowList(tuple(AllowEntry(row.tag) for row in variants()))


def refusal_for_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """Return a refusal reason, or ``None`` when this process may run.

    An empty ``OHHO_WORKER_IMAGE`` is an in-process run and is allowed.
    A set image must be a pinned tag the console allowlist names.
    """
    env = os.environ if environ is None else environ
    image = env.get("OHHO_WORKER_IMAGE", "").strip()
    if not image:
        return None
    digest = env.get("OHHO_WORKER_IMAGE_DIGEST", "").strip() or None
    try:
        allow = load_allowlist(env.get("OHHO_WORKER_ALLOWLIST", ""))
        return refusal_reason(image, allow, digest=digest)
    except WorkerImageError as exc:
        return f"worker image allowlist is unreadable: {exc}"


def refusal_reason(
    image: str, allow: AllowList, *, digest: str | None = None
) -> str | None:
    """Return a refusal reason, or ``None`` when ``image`` may run."""
    try:
        parsed = parse_image_ref(image)
        variant_by_tag(parsed.tag)
    except WorkerImageError:
        shown = image.strip() or "(empty)"
        return f"worker image {shown} is not a pinned serve worker image"
    entry = allow.entry_for(parsed.tag)
    if entry is None:
        return f"worker image {parsed.tag} is not allowed by the console"
    return _digest_reason(parsed, entry, digest)


def load_allowlist(value: str | None) -> AllowList:
    """Load a console allowlist from a path, a JSON object, or the default.

    An empty value is the published matrix. A path that exists is read as
    JSON. A value that starts with ``{`` is parsed as JSON. Any other value
    is a path and must exist.
    """
    text = (value or "").strip()
    if not text:
        return default_allowlist()
    if text.startswith("{"):
        return allowlist_from_json(text)
    path = Path(text)
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerImageError(f"allowlist file {path} is unreadable") from exc
    return allowlist_from_json(payload)


def allowlist_from_json(payload: str) -> AllowList:
    """Parse an allowlist document. Unknown tags and bad digests are refused."""
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise WorkerImageError("allowlist is not JSON") from exc
    if not isinstance(document, dict):
        raise WorkerImageError("allowlist must be a JSON object")
    schema = document.get("schema", ALLOWLIST_SCHEMA)
    if schema != ALLOWLIST_SCHEMA:
        raise WorkerImageError(f"allowlist schema must be {ALLOWLIST_SCHEMA}")
    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise WorkerImageError("allowlist images must be a non-empty list")
    entries: list[AllowEntry] = []
    seen: set[str] = set()
    for item in images:
        entry = _entry(item)
        if entry.tag in seen:
            raise WorkerImageError(f"allowlist lists {entry.tag} more than once")
        seen.add(entry.tag)
        entries.append(entry)
    return AllowList(tuple(entries))


def _entry(item: object) -> AllowEntry:
    if isinstance(item, str):
        parsed = parse_image_ref(item)
        variant_by_tag(parsed.tag)
        if parsed.digest:
            return AllowEntry(parsed.tag, parsed.digest)
        return AllowEntry(parsed.tag)
    if not isinstance(item, dict):
        raise WorkerImageError("allowlist image entries must be strings or objects")
    tag = item.get("tag")
    if not isinstance(tag, str):
        raise WorkerImageError("allowlist image tag must be a string")
    parsed = parse_image_ref(tag)
    variant_by_tag(parsed.tag)
    digest = item.get("digest", parsed.digest)
    if digest is None:
        return AllowEntry(parsed.tag)
    if not isinstance(digest, str) or not _digest_ok(digest):
        raise WorkerImageError("allowlist digest must be sha256 and 64 hex characters")
    return AllowEntry(parsed.tag, digest)


def _digest_reason(
    parsed: ImageRef, entry: AllowEntry, runtime_digest: str | None
) -> str | None:
    claimed = runtime_digest or parsed.digest
    if runtime_digest and parsed.digest and runtime_digest != parsed.digest:
        return (
            f"worker image digest {runtime_digest} does not match the image reference"
        )
    if entry.digest is None:
        return None
    if claimed is None:
        return f"worker image {parsed.tag} is missing the digest the console allows"
    if claimed != entry.digest:
        return f"worker image digest {claimed} does not match the console allowlist"
    return None


def _digest_ok(value: str) -> bool:
    return _DIGEST.fullmatch(value) is not None
