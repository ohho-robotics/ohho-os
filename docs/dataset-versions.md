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
| `content_hash` | `sha256:` of the canonical bytes. Those bytes include the episode rows and the `split` and `augmentation` configs. |
| `source_query` | Local episode roots. |
| `filters` | Task substring, robot type, min/max frames. |
| `split` | Slice name, weights, strategy (`episode`, `session`, `task`, `scene`), and held-out scenes. |
| `seed` | Shuffle and noise seed. Part of the version id. With `split.name=all` it leaves the content hash unchanged. |
| `augmentation` | Build-time scale, offset, noise, and copies, plus train-time crop, resize, colour jitter, blur, temporal stride, and per-camera toggles. |
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
3. Assign every filtered episode to train, val, or test. The episode moves
   as a whole: its frames stay in one split. `strategy=episode` shuffles
   episodes with `random.Random.random` (the stable `random()` stream) and
   takes a Hamilton share of the weights. `session` groups by the `session`
   string on `episodes.jsonl`, otherwise by `day`, otherwise by the UTC day
   of the first frame timestamp when that timestamp is a Unix time at or
   after `1000000000`, otherwise one undated group, and shuffles those
   groups. `task` groups by the task string. `scene` groups by the `scene`
   string (episodes with no scene stay in one group). Scenes named in
   `held_out_scenes` are appended to test and stay out of the shuffle.
   Weights are Hamilton shares of the remaining groups. `name=all` keeps
   source order. `name=train`, `val`, or `test` writes that part only.
4. Apply scale, offset, noise, and copies to the materialised rows. Zero
   noise leaves the seeded stream untouched. Numbers are rendered with 8
   decimal places. Crop, resize, colour jitter, blur, temporal stride, and
   camera toggles are stored on the version and applied later by
   `apply_augmentation`.
5. Write a LeRobot dataset and `meta/ohho_content.json`. The hashed JSON
   object contains `episodes`, `split`, and `augmentation`, so a train-time
   setting changes the content hash when the stored rows stay the same.
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

## Splits

`ohho.datasets.assign_splits` is the partition the builder uses. Pass
`SplitEpisode` rows and a `Split`. The same list, split, and seed return
the same `SplitAssignment`.

| Strategy | Group |
| --- | --- |
| `episode` | one episode |
| `session` | `session`, else `day`, else the UTC day of a Unix timestamp `>= 1000000000`, else one undated group |
| `task` | the task string, matched exactly |
| `scene` | the `scene` string; a missing scene shares one group |

`episodes.jsonl` may include string fields `session`, `day`, and `scene`.
The LeRobot writer leaves them unset. Session, day, and scene labels are
stripped. A `session` keeps episodes from different days together. Held-out
scene names are stored sorted and unique. A listed scene that matches no
selected episode fails the build.

Fractions are weights of groups. For `episode` each group is one episode, so
the counts match a Hamilton split of episodes. For the other strategies a
large session can land entirely in train.

## Train-time augmentation

`ohho.datasets.apply_augmentation(sample, augmentation, seed=...)` is pure
Python. Numpy arrays of shape `(height, width, 3)` and PIL images are
accepted at the edge. Pillow is imported only for a PIL image. Numpy is
imported only when the return value keeps a numpy image. List pixels need
neither package.

The sample is an object:

```python
{
    "episode_id": "ep-1",
    "frames": [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "observation_state": [0.0, 0.1],
            "action": [0.2],
            "cameras": {
                "front": {"height": 2, "width": 2, "pixels": [0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255]}
            },
        }
    ],
}
```

`pixels` is row-major RGB, integers 0 through 255, length `height * width * 3`.
The function returns `episode_id` and `frames`. Cameras come back as a list
sorted by name, each with `name`, `height`, `width`, and `pixels`. The input
object is left unchanged.

Operations on an enabled camera run in this order: random crop, nearest
resize, colour (brightness, contrast, saturation, hue), Gaussian blur.
`blur` is the sigma in pixels. `temporal_stride` of N keeps frames whose
position in the sample satisfies `index % N == phase`. State and action use
`value * scale + offset + jitter`. `copies` stays a build-time repeat of
episodes and does not change one sample.

Draws use `random.Random` streams whose seeds are the first 8 bytes of
SHA-256 over `ohho.augment.v1:{seed}:{label}`. Labels are `color`, `crop`,
`temporal`, `state`, and `action`. A magnitude of zero leaves that stream
untouched. Colour factors are drawn once per call. Crop, state noise, and
action noise run on the frames temporal sub-sampling kept. Cameras are
visited in name order. A camera listed with `enabled: false` is copied
through and consumes no crop draw. A camera omitted from `cameras` is
augmented.

Same sample, config, and seed: same result.

### ACT and Diffusion Policy

ACT (`policy="act"`) and Diffusion Policy (`policy="diffusion"`) keep their
dataset readers. Resolve the version, then call `apply_augmentation` on each
training sample. OHH-508 wires this into the recipe data loaders.

```python
from ohho.datasets import (
    apply_augmentation,
    load_version,
    materialised_apply_config,
    resolve_dataset_version,
)

record = load_version(version_id)
dataset_path = resolve_dataset_version(record.version_id)

# The loader reads dataset_path like any other LeRobot directory.
# sample_index is the loader's own deterministic index
# (episode index, or episode index and chunk index).
augmented = apply_augmentation(
    sample,
    materialised_apply_config(record.augmentation),
    seed=record.seed + sample_index,
)
```

`materialised_apply_config` keeps crop, resize, colour jitter, blur,
temporal stride, and camera toggles. Scale, offset, noise, and copies are
already in the stored frames, so the loader passes scale 1, offset 0, and
noise 0 into `apply_augmentation`.

A loader that reads the source episodes, rather than the materialised
version, passes `record.augmentation` itself so scale, offset, and noise run
inside `apply_augmentation`:

```python
augmented = apply_augmentation(sample, record.augmentation, seed=record.seed + sample_index)
```

## Preview samples

`preview_augmentation(sample, augmentation, seed=...)` returns the object the
console preview renders. The console sends one raw sample and draws `before`
and `after`. It does not run the augmentation itself.

```json
{
  "schema": "ohho.augmentation_preview.v1",
  "seed": 0,
  "augmentation": {},
  "before": {
    "episode_id": "ep-1",
    "frames": [
      {
        "frame_index": 0,
        "timestamp": 0.0,
        "observation_state": [0.0],
        "action": [0.0],
        "cameras": [
          {
            "name": "front",
            "height": 2,
            "width": 2,
            "pixels": [0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255]
          }
        ]
      }
    ]
  },
  "after": {
    "episode_id": "ep-1",
    "frames": []
  }
}
```

`augmentation` is the normalised config stored on the version. `pixels` is
row-major RGB, integers 0 through 255, length `height * width * 3`. Cameras
are sorted by name. Match a before frame to an after frame by
`frame_index`. Temporal sub-sampling leaves some source frames out of
`after.frames`. A disabled camera has the same pixels in `before` and
`after`. Preview applies the full config, including scale, offset, and
noise, to the raw sample the console supplied.

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
without reading the dataset. `split` and `augmentation` stay single JSON
columns. `split` carries `strategy` and `held_out_scenes`. `augmentation`
carries `crop`, `resize`, `color_jitter`, `blur`, `temporal_stride`, and
`cameras`.

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
`--val-fraction`, and `--test-fraction` to partition. `--split-strategy` is
`episode` (default), `session`, `task`, or `scene`. Repeat
`--held-out-scene` when the strategy is `scene`. Train-time image settings
are `--crop-height`, `--crop-width`, `--resize-height`, `--resize-width`,
`--brightness`, `--contrast`, `--saturation`, `--hue`, `--blur`,
`--temporal-stride`, and repeatable `--camera-off`.

## Lineage and Hub import

Episode, run, and model-version edges, and the Hugging Face import record,
are described in [`dataset-lineage.md`](dataset-lineage.md).
