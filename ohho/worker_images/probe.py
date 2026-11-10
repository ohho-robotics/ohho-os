"""Read-only probe of the process that is about to serve."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ohho.optimize.toolchain import detect_jetpack
from ohho.targets.detect import normalize_arch
from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.stamp import read_stamp

_DEFAULT_STAMP = Path("/etc/ohho/worker-image.json")
_UNSET = object()


@dataclass(frozen=True)
class WorkerProbe:
    """What this process can see. Tests pass one instead of reading the host."""

    arch: str
    torch: str | None
    ort: str | None
    tensorrt: str | None
    cuda: str | None
    l4t: str | None
    image: str | None
    digest: str | None
    stack: str
    require_stack: bool
    base_image: str | None = None


def collect_probe(
    *,
    environ: Mapping[str, str] | None = None,
    arch: str | None = None,
    l4t: str | None | object = _UNSET,
    packages: Mapping[str, str | None] | None = None,
    stamp_path: Path | None = None,
) -> WorkerProbe:
    """Build a probe from the environment, the stamp file, and installed packages.

    ``packages`` replaces distribution lookups. ``l4t`` replaces
    ``/etc/nv_tegra_release``. Passing ``l4t=None`` means the file is absent.
    """
    env = os.environ if environ is None else environ
    digest = env.get("OHHO_WORKER_IMAGE_DIGEST", "").strip() or None
    image, digest = _split_image(env.get("OHHO_WORKER_IMAGE", ""), digest)
    stamp = _load_stamp(env, stamp_path)
    if stamp is not None:
        if image and image != stamp["tag"]:
            raise WorkerImageError(
                f"OHHO_WORKER_IMAGE is {image}, stamp is {stamp['tag']}."
            )
        image = image or str(stamp["tag"])
    stack = "platform"
    require_stack = env.get("OHHO_WORKER_REQUIRE_STACK", "").strip() == "1"
    base_image = None
    if stamp is not None:
        stack = str(stamp["stack"])
        require_stack = require_stack or bool(stamp["require_stack"])
        base_image = str(stamp["base_image"])
    elif env.get("OHHO_WORKER_STACK", "").strip() in {"platform", "full"}:
        stack = env["OHHO_WORKER_STACK"].strip()
        require_stack = require_stack or stack == "full"
    found = packages if packages is not None else _installed_packages()
    if l4t is _UNSET:
        release = detect_jetpack()
    else:
        release = None if l4t is None else str(l4t)
    return WorkerProbe(
        arch=normalize_arch(arch or _machine()),
        torch=found.get("torch"),
        ort=found.get("ort"),
        tensorrt=found.get("tensorrt"),
        cuda=_cuda_version(env),
        l4t=release,
        image=image,
        digest=digest,
        stack=stack,
        require_stack=require_stack,
        base_image=base_image,
    )


def _split_image(value: str, digest: str | None) -> tuple[str | None, str | None]:
    text = value.strip()
    if not text:
        return None, digest
    if "@" not in text:
        return text, digest
    tag, ref_digest = text.split("@", 1)
    if digest and digest != ref_digest:
        raise WorkerImageError(
            f"OHHO_WORKER_IMAGE_DIGEST is {digest}, tag digest is {ref_digest}."
        )
    return tag, digest or ref_digest


def _machine() -> str:
    import platform

    return platform.machine()


def _load_stamp(env: Mapping[str, str], stamp_path: Path | None) -> dict | None:
    if stamp_path is not None:
        return read_stamp(stamp_path)
    named = env.get("OHHO_WORKER_STAMP", "").strip()
    if named:
        return read_stamp(named)
    if _DEFAULT_STAMP.is_file():
        return read_stamp(_DEFAULT_STAMP)
    return None


def _cuda_version(env: Mapping[str, str]) -> str | None:
    for key in ("CUDA_VERSION", "NVIDIA_CUDA_VERSION"):
        value = env.get(key, "").strip()
        if value:
            return value
    return None


def _installed_packages() -> dict[str, str | None]:
    return {
        "torch": _dist_version("torch"),
        "ort": _dist_version("onnxruntime"),
        "tensorrt": _dist_version("tensorrt"),
    }


def _dist_version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None
