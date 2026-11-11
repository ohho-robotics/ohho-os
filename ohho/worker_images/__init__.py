"""Pinned ``ohho-serve-worker`` images for cpu, cuda, and jetson.

Import submodules directly. ``ohho targets verify`` checks a process against
the matrix in ``worker_matrix.json``.
"""

from __future__ import annotations

__all__ = [
    "SCHEMA_ID",
    "get_variant",
    "parse_image_ref",
    "refusal_for_environment",
    "variants",
    "verify_worker",
    "write_job_result",
]


def __getattr__(name: str) -> object:
    if name in {"SCHEMA_ID", "get_variant", "parse_image_ref", "variants"}:
        from ohho.worker_images import matrix

        return getattr(matrix, name)
    if name == "refusal_for_environment":
        from ohho.worker_images.policy import refusal_for_environment

        return refusal_for_environment
    if name == "verify_worker":
        from ohho.worker_images.verify import verify_worker

        return verify_worker
    if name == "write_job_result":
        from ohho.worker_images.results import write_job_result

        return write_job_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
