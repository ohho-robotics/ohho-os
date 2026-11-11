"""Pinned serve-worker image matrix.

The JSON file in this package is the source of truth for image tags and the
torch, ONNX Runtime, TensorRT, CUDA, and JetPack versions those tags carry.
Dockerfiles and ``ohho targets verify`` both read it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from ohho.targets.schema import get_profile
from ohho.worker_images.errors import WorkerImageError

SCHEMA_ID = "ohho.serve_worker_matrix.v1"
REPOSITORY = "ohho-serve-worker"
VARIANTS = ("cpu", "cuda", "jetson")
CUDA_IDENTITIES = ("absent", "required", "jetpack")
_TAG = re.compile(
    r"^ohho-serve-worker:(cpu|cuda|jetson)-([0-9]+\.[0-9]+\.[0-9]+)"
    r"(?:@(sha256:[0-9a-f]{64}))?$"
)
_CACHE: tuple[WorkerVariant, ...] | None = None


@dataclass(frozen=True)
class ImageRef:
    """A parsed ``ohho-serve-worker`` tag, optionally with a digest."""

    tag: str
    variant: str
    version: str
    digest: str | None


@dataclass(frozen=True)
class WorkerVariant:
    """One pinned worker image."""

    variant: str
    tag: str
    version: str
    platforms: tuple[str, ...]
    arches: tuple[str, ...]
    base_image: str
    torch: str
    torch_index: str
    ort: str
    ort_package: str
    tensorrt: str | None
    tensorrt_pip: bool
    cuda: str | None
    jetpack: str | None
    l4t: str | None
    target_profile: str
    cuda_identity: str

    def pins(self) -> dict[str, str | None]:
        """Versions a job result and a verify report both record."""
        return {
            "torch": self.torch,
            "ort": self.ort,
            "tensorrt": self.tensorrt,
            "cuda": self.cuda,
            "jetpack": self.jetpack,
            "l4t": self.l4t,
        }


def matrix_text() -> str:
    """Raw matrix JSON shipped with the package."""
    resource = files("ohho.worker_images").joinpath("worker_matrix.json")
    return resource.read_text(encoding="utf-8")


def variants() -> tuple[WorkerVariant, ...]:
    """Pinned images in matrix order: cpu, cuda, jetson."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load(json.loads(matrix_text()))
    return _CACHE


def get_variant(variant: str) -> WorkerVariant:
    """Return one variant. Unknown names raise ``WorkerImageError``."""
    for row in variants():
        if row.variant == variant:
            return row
    known = ", ".join(row.variant for row in variants())
    raise WorkerImageError(f"Unknown worker variant {variant!r}. Known: {known}.")


def variant_by_tag(tag: str) -> WorkerVariant:
    """Return the row whose tag is exactly ``tag`` (no digest)."""
    for row in variants():
        if row.tag == tag:
            return row
    raise WorkerImageError(f"{tag} is not a pinned serve worker image")


def parse_image_ref(value: str) -> ImageRef:
    """Parse ``ohho-serve-worker:<variant>-<version>[@sha256:...]``."""
    text = value.strip()
    match = _TAG.fullmatch(text)
    if match is None:
        raise WorkerImageError(f"{value} is not a pinned serve worker image")
    variant, version, digest = match.group(1), match.group(2), match.group(3)
    tag = f"{REPOSITORY}:{variant}-{version}"
    return ImageRef(tag=tag, variant=variant, version=version, digest=digest)


def _load(document: Mapping[str, Any]) -> tuple[WorkerVariant, ...]:
    if document.get("schema") != SCHEMA_ID:
        raise WorkerImageError(f"Worker matrix schema must be {SCHEMA_ID}.")
    if document.get("repository") != REPOSITORY:
        raise WorkerImageError(f"Worker matrix repository must be {REPOSITORY}.")
    version = document.get("version")
    if not isinstance(version, str) or not version:
        raise WorkerImageError("Worker matrix version is missing.")
    from ohho import __version__

    if version != __version__:
        raise WorkerImageError(
            f"Worker matrix version {version} does not match ohho {__version__}."
        )
    rows = document.get("variants")
    if not isinstance(rows, list) or not rows:
        raise WorkerImageError("Worker matrix variants must be a non-empty list.")
    parsed = tuple(_row(item, version) for item in rows)
    names = tuple(item.variant for item in parsed)
    if names != VARIANTS:
        raise WorkerImageError(
            f"Worker matrix variants must be {', '.join(VARIANTS)} in that order."
        )
    return parsed


def _row(item: Any, version: str) -> WorkerVariant:
    if not isinstance(item, Mapping):
        raise WorkerImageError("Each worker variant must be an object.")
    variant = item.get("variant")
    if variant not in VARIANTS:
        raise WorkerImageError(f"Unknown worker variant {variant!r}.")
    tag = f"{REPOSITORY}:{variant}-{version}"
    platforms = _strings(item.get("platforms"), "platforms")
    arches = _strings(item.get("arches"), "arches")
    base_image = _text(item.get("base_image"), "base_image")
    torch = _text(item.get("torch"), "torch")
    torch_index = _text(item.get("torch_index"), "torch_index")
    ort = _text(item.get("ort"), "ort")
    ort_package = _text(item.get("ort_package"), "ort_package")
    if ort_package not in {"onnxruntime", "onnxruntime-gpu"}:
        raise WorkerImageError(
            f"{variant} ort_package {ort_package!r} is not supported."
        )
    tensorrt = _optional(item.get("tensorrt"), "tensorrt")
    tensorrt_pip = item.get("tensorrt_pip")
    if not isinstance(tensorrt_pip, bool):
        raise WorkerImageError(f"{variant} tensorrt_pip must be a boolean.")
    cuda = _optional(item.get("cuda"), "cuda")
    jetpack = _optional(item.get("jetpack"), "jetpack")
    l4t = _optional(item.get("l4t"), "l4t")
    target_profile = _text(item.get("target_profile"), "target_profile")
    cuda_identity = item.get("cuda_identity")
    if cuda_identity not in CUDA_IDENTITIES:
        raise WorkerImageError(f"{variant} cuda_identity {cuda_identity!r} is invalid.")
    row = WorkerVariant(
        variant=variant,
        tag=tag,
        version=version,
        platforms=platforms,
        arches=arches,
        base_image=base_image,
        torch=torch,
        torch_index=torch_index,
        ort=ort,
        ort_package=ort_package,
        tensorrt=tensorrt,
        tensorrt_pip=tensorrt_pip,
        cuda=cuda,
        jetpack=jetpack,
        l4t=l4t,
        target_profile=target_profile,
        cuda_identity=cuda_identity,
    )
    _check_row(row)
    return row


def _check_row(row: WorkerVariant) -> None:
    profile = get_profile(row.target_profile)
    if profile.arch not in row.arches:
        raise WorkerImageError(
            f"{row.variant} arch {profile.arch} is not in {', '.join(row.arches)}."
        )
    if row.tensorrt is None:
        if profile.tensorrt is not None or row.tensorrt_pip:
            raise WorkerImageError(f"{row.variant} pins no TensorRT.")
    else:
        if profile.tensorrt is None or not profile.tensorrt.allows(row.tensorrt):
            raise WorkerImageError(
                f"{row.variant} TensorRT {row.tensorrt} is outside {row.target_profile}."
            )
    if row.cuda_identity == "absent":
        if row.cuda or row.jetpack or row.l4t or row.tensorrt:
            raise WorkerImageError(
                f"{row.variant} must not pin CUDA, JetPack, or TensorRT."
            )
    elif row.cuda_identity == "required":
        if not row.cuda or row.jetpack or row.l4t or not row.tensorrt:
            raise WorkerImageError(
                f"{row.variant} must pin CUDA and TensorRT and no JetPack."
            )
    elif not row.cuda or not row.jetpack or not row.l4t or not row.tensorrt:
        raise WorkerImageError(
            f"{row.variant} must pin CUDA, TensorRT, JetPack, and L4T."
        )
    if row.variant == "jetson" and "l4t" not in row.base_image:
        raise WorkerImageError("The jetson image must use an L4T base image.")
    if row.variant == "cuda" and not row.base_image.startswith("nvidia/cuda:"):
        raise WorkerImageError("The cuda image must use an nvidia/cuda base image.")


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise WorkerImageError(
            f"Worker matrix {field} must be a non-empty string list."
        )
    return tuple(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerImageError(f"Worker matrix {field} must be a non-empty string.")
    return value


def _optional(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkerImageError(f"Worker matrix {field} must be a string or null.")
    return value
