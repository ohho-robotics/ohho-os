"""Dockerfiles for the pinned serve worker images.

``render_dockerfile`` is the source used by the committed files under
``docker/serve-worker``. The files do not install git. The L4T base image
used for the jetson variant does not ship git, and it does not ship
``/etc/nv_tegra_release``. The jetson Dockerfile writes that file from the
pinned L4T release when the image has no parseable copy.
"""

from __future__ import annotations

from pathlib import Path

from ohho.optimize.toolchain import jetpack_from_tegra_release
from ohho.worker_images.errors import WorkerImageError
from ohho.worker_images.matrix import WorkerVariant, get_variant

_ROOT = Path(__file__).resolve().parents[2]
_APT = """\
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -euo pipefail; \\
    if ! apt-get update -qq; then \\
      echo "apt-get update failed. Commenting NVIDIA apt entries and retrying."; \\
      shopt -s nullglob; \\
      for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do \\
        if [ -f "${f}" ]; then \\
          sed -i -E '/nvidia|repo\\.download\\.nvidia|jetson/s/^/# ohho-ci /' "${f}"; \\
        fi; \\
      done; \\
      shopt -u nullglob; \\
      apt-get update -qq; \\
    fi; \\
    apt-get install -y -qq --no-install-recommends \\
      python3 \\
      python3-pip \\
      python3-venv \\
      ca-certificates; \\
    rm -rf /var/lib/apt/lists/*; \\
    id ohho >/dev/null 2>&1 || useradd --create-home --uid 10001 --user-group ohho
"""


def tegra_release_line(l4t: str) -> str:
    """``/etc/nv_tegra_release`` line for a pinned ``major.minor[.patch]``.

    ``ohho targets verify`` reads that file through
    ``jetpack_from_tegra_release``. A pin that would not parse back to the
    same release is refused.
    """
    parts = l4t.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts):
        raise WorkerImageError(
            f"L4T release {l4t!r} must be dotted integers (major.minor.patch)."
        )
    major, revision = parts[0], ".".join(parts[1:])
    line = (
        f"# R{major} (release), REVISION: {revision}, GCID: 0, "
        "BOARD: generic, EABI: aarch64, DATE: Thu Jan 1 00:00:00 UTC 1970"
    )
    parsed = jetpack_from_tegra_release(line)
    if parsed != f"R{l4t}":
        raise WorkerImageError(
            f"L4T release {l4t} would be read as {parsed or 'unset'}."
        )
    return line


def dockerfile_path(variant: str) -> Path:
    """Committed Dockerfile for one variant."""
    return _ROOT / "docker" / "serve-worker" / f"Dockerfile.{variant}"


def render_dockerfile(variant: str) -> str:
    """Return the Dockerfile text for ``cpu``, ``cuda``, or ``jetson``."""
    row = get_variant(variant)
    parts = [
        "# Generated from ohho/worker_images/worker_matrix.json. Do not edit by hand.",
        f"# {row.tag}",
        "# CI passes --build-arg OHHO_WORKER_STACK=platform.",
        "# The default full stack installs the pinned wheels.",
        f"ARG BASE_IMAGE={row.base_image}",
        "FROM ${BASE_IMAGE}",
        "",
        "ARG OHHO_WORKER_STACK=full",
        *_env_lines(row),
        "ENV OHHO_WORKER_STACK=$OHHO_WORKER_STACK \\",
        "    PYTHONDONTWRITEBYTECODE=1 \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    PATH=/opt/ohho/bin:$PATH",
        "",
        _APT.rstrip("\n"),
        *_record_l4t(row),
        "",
        "COPY . /opt/ohho-src",
        "WORKDIR /opt/ohho-src",
        "",
        _pip_run(row),
        "",
        "RUN mkdir -p /etc/ohho \\",
        '    && python -c "from ohho.worker_images.stamp import write_image_stamp; '
        "write_image_stamp('/etc/ohho/worker-image.json')\"",
        "",
        "USER ohho",
        "WORKDIR /home/ohho",
        'ENTRYPOINT ["ohho"]',
        'CMD ["targets", "verify"]',
        "",
    ]
    return "\n".join(parts)


def write_dockerfiles() -> None:
    """Rewrite the committed Dockerfiles from the matrix."""
    for name in ("cpu", "cuda", "jetson"):
        path = dockerfile_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_dockerfile(name), encoding="utf-8", newline="\n")


def _record_l4t(row: WorkerVariant) -> list[str]:
    """Write the pinned release when the base image has no parseable file.

    ``nvcr.io/nvidia/l4t-base`` does not ship ``/etc/nv_tegra_release``. On a
    Jetson the NVIDIA runtime bind-mounts the host file, which is the host
    release. A GitHub arm64 runner has no GPU and does not perform that mount,
    so ``ohho targets verify`` would otherwise report that the L4T release is
    not set. A file the base already provides, and that verify can parse, is
    left in place and still has to match the pin.
    """
    if not row.l4t:
        return []
    line = tegra_release_line(row.l4t)
    return [
        "",
        "# l4t-base does not ship /etc/nv_tegra_release. The NVIDIA runtime",
        "# bind-mounts the host file when present. Record the pinned release",
        "# when this image has no parseable file so targets verify can read",
        "# it on a runner without that runtime and without a GPU.",
        "RUN if ! grep -Eq "
        "'R[0-9]+[[:space:]]*\\(release\\),[[:space:]]*REVISION:"
        "[[:space:]]*[0-9]+(\\.[0-9]+)*' "
        "/etc/nv_tegra_release 2>/dev/null; then \\",
        f"      printf '%s\\n' '{line}' > /etc/nv_tegra_release; \\",
        "      chmod 644 /etc/nv_tegra_release; \\",
        "    fi",
    ]


def _env_lines(row: WorkerVariant) -> list[str]:
    pairs = (
        ("OHHO_WORKER_IMAGE", row.tag),
        ("OHHO_WORKER_VARIANT", row.variant),
        ("OHHO_WORKER_TORCH", row.torch),
        ("OHHO_WORKER_ORT", row.ort),
        ("OHHO_WORKER_TENSORRT", row.tensorrt or ""),
        ("OHHO_WORKER_CUDA", row.cuda or ""),
        ("OHHO_WORKER_JETPACK", row.jetpack or ""),
        ("OHHO_WORKER_L4T", row.l4t or ""),
        ("OHHO_WORKER_BASE", row.base_image),
    )
    lines: list[str] = []
    last = len(pairs) - 1
    for index, (key, value) in enumerate(pairs):
        cont = " \\" if index < last else ""
        prefix = "ENV " if index == 0 else "    "
        lines.append(f"{prefix}{key}={value}{cont}")
    return lines


def _pip_run(row: WorkerVariant) -> str:
    installs = [
        (
            "/opt/ohho/bin/pip install --no-cache-dir "
            f"--index-url {row.torch_index} 'torch=={row.torch}'"
        ),
        (f"/opt/ohho/bin/pip install --no-cache-dir '{row.ort_package}=={row.ort}'"),
    ]
    if row.tensorrt_pip and row.tensorrt:
        installs.append(
            f"/opt/ohho/bin/pip install --no-cache-dir 'tensorrt=={row.tensorrt}'"
        )
    body = ["         " + installs[0]]
    for command in installs[1:]:
        body.append("         && " + command)
    joined = " \\\n".join(body)
    return "\n".join(
        [
            "RUN python3 -m venv /opt/ohho \\",
            "    && /opt/ohho/bin/pip install --upgrade pip setuptools \\",
            "    && /opt/ohho/bin/pip install --no-cache-dir /opt/ohho-src \\",
            '    && if [ "$OHHO_WORKER_STACK" != "full" ] && '
            '[ "$OHHO_WORKER_STACK" != "platform" ]; then \\',
            '         echo "OHHO_WORKER_STACK must be full or platform" >&2; \\',
            "         exit 1; \\",
            "       fi \\",
            '    && if [ "$OHHO_WORKER_STACK" = "full" ]; then \\',
            joined + "; \\",
            "       fi",
        ]
    )
