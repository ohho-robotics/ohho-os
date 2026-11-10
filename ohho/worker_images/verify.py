"""``ohho targets verify`` — compare this process to a pinned worker image."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import (
    WorkerVariant,
    parse_image_ref,
    variant_by_tag,
)
from ohho.worker_images.probe import WorkerProbe, collect_probe

VERIFY_SCHEMA = "ohho.worker_verify.v1"


@dataclass(frozen=True)
class WorkerVerifyReport:
    """Successful verify. Failures raise ``WorkerImageError`` before this exists."""

    tag: str
    variant: str
    arch: str
    stack: str
    require_stack: bool
    base_image: str
    image_digest: str | None
    pins: dict[str, str | None]
    probed: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        """JSON body for ``ohho targets verify --json``."""
        return {
            "schema": VERIFY_SCHEMA,
            "ok": True,
            "image": self.tag,
            "image_digest": self.image_digest,
            "variant": self.variant,
            "arch": self.arch,
            "stack": self.stack,
            "require_stack": self.require_stack,
            "base_image": self.base_image,
            "pins": self.pins,
            "probed": self.probed,
        }

    def text(self) -> str:
        """Multi-line report. The first line is ``ok: <tag>``."""
        lines = [f"ok: {self.tag}", f"  arch: {self.arch}", f"  stack: {self.stack}"]
        for name in ("torch", "ort", "tensorrt", "cuda", "jetpack", "l4t"):
            pin = self.pins.get(name)
            got = self.probed.get(name)
            shown = pin if pin is not None else "none"
            if pin is not None and got is None:
                shown = f"{pin} (not installed)"
            lines.append(f"  {name}: {shown}")
        if self.image_digest:
            lines.append(f"  image_digest: {self.image_digest}")
        return "\n".join(lines)


def verify_worker(
    *,
    image: str | None = None,
    variant: str | None = None,
    require_stack: bool = False,
    probe: WorkerProbe | None = None,
) -> WorkerVerifyReport:
    """Accept this process or raise ``WorkerImageError``.

    ``probe`` replaces the live read. ``image`` overrides the tag the probe
    claims. ``require_stack`` demands that pinned wheels are installed.
    """
    current = probe if probe is not None else collect_probe()
    if image:
        current = replace(current, image=image)
    if require_stack:
        current = replace(current, require_stack=True)
    if not current.image:
        raise WorkerImageError("no worker image is set (OHHO_WORKER_IMAGE)")
    parsed = parse_image_ref(current.image)
    if variant and variant != parsed.variant:
        raise WorkerImageError(
            f"{parsed.tag} is variant {parsed.variant}, not {variant}."
        )
    row = variant_by_tag(parsed.tag)
    if parsed.version != row.version:
        raise WorkerImageError(f"{parsed.tag} is not a pinned serve worker image")
    _check_arch(row, current)
    _check_identity(row, current)
    _check_packages(row, current)
    digest = current.digest or parsed.digest
    return WorkerVerifyReport(
        tag=row.tag,
        variant=row.variant,
        arch=current.arch,
        stack=current.stack,
        require_stack=current.require_stack,
        base_image=row.base_image,
        image_digest=digest,
        pins=row.pins(),
        probed={
            "torch": current.torch,
            "ort": current.ort,
            "tensorrt": current.tensorrt,
            "cuda": current.cuda,
            "jetpack": row.jetpack if current.l4t else None,
            "l4t": _l4t_text(current.l4t),
        },
    )


def _check_arch(row: WorkerVariant, probe: WorkerProbe) -> None:
    if probe.arch not in row.arches:
        arches = ", ".join(row.arches)
        raise WorkerImageError(
            f"{row.tag} requires arch {arches}, this process is {probe.arch}"
        )


def _check_identity(row: WorkerVariant, probe: WorkerProbe) -> None:
    found_l4t = _l4t_key(probe.l4t)
    if row.cuda_identity == "absent":
        if found_l4t is not None:
            raise WorkerImageError(
                f"{row.tag} does not use an L4T base, found {_show_l4t(probe.l4t)}."
            )
        if probe.cuda:
            raise WorkerImageError(f"{row.tag} does not pin CUDA, found {probe.cuda}.")
        return
    if row.cuda_identity == "required":
        if found_l4t is not None:
            raise WorkerImageError(
                f"{row.tag} does not use an L4T base, found {_show_l4t(probe.l4t)}."
            )
        if not probe.cuda:
            raise WorkerImageError(f"{row.tag} CUDA version is not set.")
        if not versions_match(probe.cuda, row.cuda):
            raise WorkerImageError(
                f"{row.tag} CUDA version is {probe.cuda}, pin is {row.cuda}."
            )
        return
    pin = _l4t_key(row.l4t)
    if found_l4t is None:
        raise WorkerImageError(f"{row.tag} L4T release is not set.")
    if found_l4t != pin:
        raise WorkerImageError(
            f"{row.tag} L4T release is {_show_l4t(probe.l4t)}, pin is {row.l4t}."
        )
    if probe.cuda and not versions_match(probe.cuda, row.cuda):
        raise WorkerImageError(
            f"{row.tag} CUDA version is {probe.cuda}, pin is {row.cuda}."
        )


def _check_packages(row: WorkerVariant, probe: WorkerProbe) -> None:
    _one(row, "torch", probe.torch, row.torch, probe.require_stack)
    _one(row, "onnxruntime", probe.ort, row.ort, probe.require_stack)
    if row.tensorrt is None:
        if probe.tensorrt:
            raise WorkerImageError(f"TensorRT is installed and {row.tag} pins none.")
        return
    _one(
        row,
        "TensorRT",
        probe.tensorrt,
        row.tensorrt,
        probe.require_stack and row.tensorrt_pip,
    )


def _one(
    row: WorkerVariant,
    label: str,
    installed: str | None,
    pin: str,
    required: bool,
) -> None:
    if installed is None:
        if required:
            raise WorkerImageError(f"{label} {pin} is not installed.")
        return
    if not versions_match(installed, pin):
        raise WorkerImageError(f"{label} is {installed}, pin is {pin}.")


def versions_match(installed: str | None, pin: str | None) -> bool:
    """True when ``installed`` starts with the pinned ``major.minor.patch``.

    A null pin matches a missing install. ``2.5.1+cu124`` matches ``2.5.1``.
    ``10.0.1.26`` matches ``10.0.1``. A shorter install does not match a longer pin.
    """
    if pin is None:
        return installed in (None, "")
    if not installed:
        return False
    got = _numbers(installed)
    want = _numbers(pin)
    if not got or not want or len(got) < len(want):
        return False
    return got[: len(want)] == want


def _numbers(text: str) -> tuple[int, ...]:
    head = text.split("+", 1)[0].strip()
    if head.upper().startswith("R"):
        head = head[1:]
    parts: list[int] = []
    for part in head.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def _l4t_key(value: str | None) -> tuple[int, int, int] | None:
    if value is None or not str(value).strip():
        return None
    nums = _numbers(str(value))
    if len(nums) < 2:
        return None
    return (nums + (0, 0, 0))[:3]


def _l4t_text(value: str | None) -> str | None:
    key = _l4t_key(value)
    if key is None:
        return None
    return ".".join(str(part) for part in key)


def _show_l4t(value: str | None) -> str:
    return _l4t_text(value) or str(value)
