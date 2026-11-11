"""SBOM and signature attachment names for a released worker tag.

CI attaches these files to the git tag. This module does not sign, scan,
or push an image.
"""

from __future__ import annotations

from ohho.worker_images.matrix import variants


def cyclonedx_name(variant: str) -> str:
    """CycloneDX SBOM file name for one variant."""
    return f"sbom-{variant}.cdx.json"


def spdx_name(variant: str) -> str:
    """SPDX SBOM file name for one variant."""
    return f"sbom-{variant}.spdx.json"


def scan_name(variant: str) -> str:
    """Grype JSON file name for one variant."""
    return f"scan-{variant}.json"


def signature_name(variant: str) -> str:
    """Cosign bundle file name for one image digest."""
    return f"{variant}.image.sig.bundle"


def cosign_sign_blob_command(variant: str) -> list[str]:
    """``cosign sign-blob`` argv. The digest bytes are in ``digest.txt``.

    Keyless signing needs a GitHub OIDC token on a tag build. Callers do not
    run this command from unit tests.
    """
    return [
        "cosign",
        "sign-blob",
        "--yes",
        "--bundle",
        signature_name(variant),
        "digest.txt",
    ]


def release_plan() -> list[dict[str, str]]:
    """One record per pinned tag: SBOM, scan, signature, and the sign command."""
    plan: list[dict[str, str]] = []
    for row in variants():
        plan.append(
            {
                "tag": row.tag,
                "variant": row.variant,
                "cyclonedx": cyclonedx_name(row.variant),
                "spdx": spdx_name(row.variant),
                "scan": scan_name(row.variant),
                "signature": signature_name(row.variant),
                "job_result": "worker-job-result.json",
                "sign_command": " ".join(cosign_sign_blob_command(row.variant)),
            }
        )
    return plan
