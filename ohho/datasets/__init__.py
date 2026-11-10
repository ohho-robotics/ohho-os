"""Dataset versions: build, pin, stats, lineage, and Hugging Face import.

A version is an immutable, content-hashed LeRobot dataset built from local
episodes. Training code keeps taking a filesystem path; call
:func:`resolve_dataset_version` to turn a version id into that path.

Lineage edges (episode, run, model version) live in a JSON file. See
``docs/dataset-lineage.md``. No cloud resources are created here.
"""

from __future__ import annotations

from .build import build_version, edit_version, prepare_spec
from .hf_import import (
    DatasetImport,
    DatasetImportError,
    import_hf_dataset,
    is_non_commercial,
    resolved_licence,
)
from .lineage import (
    EpisodeDeletion,
    EpisodeRecord,
    EpisodeTraining,
    JsonLineageStore,
    LineageConflict,
    LineageError,
    LineageNotFound,
    LineageStore,
    ModelLineage,
    episode_from_canonical,
    record_version_lineage,
)
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
    "DatasetImport",
    "DatasetImportError",
    "DatasetSpec",
    "DatasetStats",
    "DatasetVersion",
    "DatasetVersionError",
    "DatasetVersionFrozen",
    "DatasetVersionNotFound",
    "EpisodeDeletion",
    "EpisodeRecord",
    "EpisodeTraining",
    "Filters",
    "JsonLineageStore",
    "LineageConflict",
    "LineageError",
    "LineageNotFound",
    "LineageStore",
    "ModelLineage",
    "SourceQuery",
    "Split",
    "build_version",
    "edit_version",
    "episode_from_canonical",
    "import_hf_dataset",
    "is_non_commercial",
    "list_versions",
    "load_version",
    "overwrite_version",
    "pin_version",
    "prepare_spec",
    "record_version_lineage",
    "resolve_dataset_version",
    "resolve_store",
    "resolved_licence",
    "schema_path",
    "version_stats",
]
