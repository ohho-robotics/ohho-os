"""Dataset versions: build, pin, and stats for local episodes.

A version is an immutable, content-hashed LeRobot dataset built from local
episodes. Training code keeps taking a filesystem path; call
:func:`resolve_dataset_version` to turn a version id into that path.

See ``docs/dataset-versions.md`` for the on-disk layout and the cloud table
it maps onto. No cloud resources are created here.
"""

from __future__ import annotations

from .build import build_version, edit_version, prepare_spec
from .model import (
    SCHEMA_NAME,
    Augmentation,
    AxisRange,
    DatasetBuildError,
    DatasetSpec,
    DatasetStats,
    DatasetVersion,
    DatasetVersionError,
    DatasetVersionFrozen,
    DatasetVersionNotFound,
    Filters,
    SourceQuery,
    Split,
    schema_path,
)
from .store import (
    list_versions,
    load_version,
    overwrite_version,
    pin_version,
    resolve_dataset_version,
    resolve_store,
    version_stats,
)

__all__ = [
    "SCHEMA_NAME",
    "Augmentation",
    "AxisRange",
    "DatasetBuildError",
    "DatasetSpec",
    "DatasetStats",
    "DatasetVersion",
    "DatasetVersionError",
    "DatasetVersionFrozen",
    "DatasetVersionNotFound",
    "Filters",
    "SourceQuery",
    "Split",
    "build_version",
    "edit_version",
    "list_versions",
    "load_version",
    "overwrite_version",
    "pin_version",
    "prepare_spec",
    "resolve_dataset_version",
    "resolve_store",
    "schema_path",
    "version_stats",
]
