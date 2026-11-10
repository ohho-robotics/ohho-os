# Dataset versions

A dataset version is an immutable snapshot of local episodes. The same spec
builds the same content hash, so a training run can cite the version instead
of a directory that might still be changing.

This page is the ohho-os side: a JSON Schema, a dataclass, a local directory
layout, and `ohho dataset build|pin|stats`. It does not create cloud tables
or object-storage buckets. The layout below is the follow-up contract for the
console.

The schema is a data-version contract. It is not part of the transport
protocol manifest in `schemas/manifest.json`.

## Record

`ohho.dataset_version.v1` is checked by
`ohho/datasets/dataset_version.schema.json`.

| Field | Role |
| --- | --- |
| `version_id` | `dv_` plus 32 hex chars. Same spec, same id. |
| `dataset_name` | Label. Not a path. |
| `parent_version_id` | Version this one was edited from, or null. |
| `pinned` | When true, the version cannot be modified. |
| `content_hash` | `sha256:` of the canonical episode bytes. |
| `source_query` | Local episode roots. |
| `filters` | Task substring, robot type, min/max frames. |
| `split` | `all`, `train`, `val`, or `test`, plus weights. |
| `seed` | Shuffle and noise seed. |
| `augmentation` | Scale, offset, noise, and copies. |
| `stats` | Episodes, frames, action ranges, state ranges. |
| `licence` | Licence string stored with the version. |
| `storage_prefix` | `versions/<version_id>/dataset`. |
| `created_at` | UTC timestamp, `YYYY-MM-DDTHH:MM:SSZ`. |

`pinned`, `created_at`, `stats`, `parent_version_id`, and `storage_prefix`
are stored on the record and are not part of the version id.

## Local layout

The store root is `--store`, or `OHHO_DATASETS_ROOT`, or `~/.ohho/datasets`.

```text
<store>/
  versions/
    dv_<32 hex>/
      version.json
      dataset/
        meta/info.json
        meta/tasks.jsonl
        meta/episodes.jsonl
        meta/episodes_stats.jsonl
        meta/stats.json
        meta/ohho_content.json
        data/chunk-000/episode_000000.jsonl   # or .parquet
        README.md
```

`version.json` is the row. `dataset/` is the LeRobot tree a trainer already
knows how to read. `meta/ohho_content.json` is the canonical payload that
`content_hash` digests (UTF-8, one trailing newline).

Listing versions is a scan of `versions/*/version.json`, which is the local
form of `select * from dataset_versions`.

## How a version is built

1. Read each source with `ohho.data.DatasetReader`. A source is a LeRobot
   directory (`meta/info.json`) or a directory whose children are.
2. Keep episodes that match the filters. Order is source path, then episode
   index, then task.
3. `split=all` keeps that order and ignores the seed. `train`, `val`, and
   `test` shuffle with `random.Random.random` (the stable `random()` stream)
   and take a Hamilton share of the weights.
4. Apply scale, offset, and noise. Noise is zero unless the std is positive,
   so a zero-noise build does not depend on the seed. Numbers are rendered
   with 8 decimal places.
5. Write a LeRobot dataset and `meta/ohho_content.json`.
6. Store stats on `version.json`: episode count, frame count, and the
   min/max of each action and state dimension.

Building that spec again returns the existing record and does not rewrite
it. Video files are not copied; the hashed content is the state and action
table the reader returns. Copying video bytes is a follow-up.

## Pin and edits

`ohho dataset pin VERSION` sets `pinned` to true. That is the only in-place
change. A second pin does not write. `overwrite_version` refuses a pinned
record with `DatasetVersionFrozen`.

`edit_version` (Python) changes a spec field and allocates a new version id.
The pinned directory is left byte for byte. A no-op edit returns the same id.

## Stats

`ohho dataset stats VERSION` prints the stats object already stored on the
version. It does not recompute and it does not write. Console display of
that object is a follow-up.

## Training recipes

Recipes keep taking a dataset path. Resolve a version id first:

```python
from ohho.datasets import resolve_dataset_version

root = resolve_dataset_version("dv_...")  # or an existing dataset path
# pass str(root) to any trainer that accepts a LeRobot directory
```

`resolve_dataset_version` checks `content_hash` against
`meta/ohho_content.json` when given a version id. It does not import
`ohho.train`.

## Proposed cloud tables (follow-up)

Do not create these tables from ohho-os. One local `version.json` is one row.
One local `dataset/` tree is one object prefix.

### `dataset_versions`

| Column | JSON field | Type |
| --- | --- | --- |
| `version_id` | `version_id` | text primary key |
| `schema_version` | `schema` | text not null |
| `dataset_name` | `dataset_name` | text not null |
| `parent_version_id` | `parent_version_id` | text null, references `dataset_versions(version_id)` |
| `pinned` | `pinned` | boolean not null |
| `content_hash` | `content_hash` | text not null |
| `source_query` | `source_query` | jsonb not null |
| `filters` | `filters` | jsonb not null |
| `split` | `split` | jsonb not null |
| `seed` | `seed` | integer not null |
| `augmentation` | `augmentation` | jsonb not null |
| `stats` | `stats` | jsonb not null |
| `licence` | `licence` | text not null |
| `storage_prefix` | `storage_prefix` | text not null |
| `created_at` | `created_at` | timestamptz not null |

`stats` stays on the row so a console can show episodes, frames, and ranges
without reading the dataset.

### Object storage

Bucket prefix `dataset-versions/`. The object key for a file is
`versions/<version_id>/dataset/<relative path>`, which is `storage_prefix`
plus the path under the local `dataset/` directory.

| Local path | Object key |
| --- | --- |
| `<store>/versions/<id>/version.json` | row in `dataset_versions`, not an object |
| `<store>/versions/<id>/dataset/meta/info.json` | `versions/<id>/dataset/meta/info.json` |
| `<store>/versions/<id>/dataset/data/...` | `versions/<id>/dataset/data/...` |

A pinned row is immutable. An edit inserts a new row and writes a new
prefix. The previous prefix stays.

## Commands

```text
ohho dataset build --source episodes/ --name demos --licence Apache-2.0
ohho dataset pin dv_0123...
ohho dataset stats dv_0123...
ohho dataset import-hf owner/name --revision main
```

Repeat `--source` to combine roots. `--store DIR` selects the store.
`--json` prints the machine-readable result. `--split train` with the
default fractions keeps every episode as train; pass `--train-fraction`,
`--val-fraction`, and `--test-fraction` to partition.

## Lineage and Hub import

Episode, run, and model-version edges, and the Hugging Face import record,
are described in [`dataset-lineage.md`](dataset-lineage.md).
