# Dataset lineage

A training run cites one dataset version. A model version cites the run that
produced it. A dataset version cites the episodes it materialised. Those three
edges answer two questions:

* which episodes a model version trained on
* which model versions trained on one episode

This page is the ohho-os side: a Python interface, a JSON file, and an import
record. It does not create cloud tables. The tables below are the follow-up
contract for the console database. The same `LineageStore` methods are what
that database implements. Run rows store a run id and a dataset version id.
A training manifest is a separate record.

The schemas are data contracts. They are not part of the transport protocol
manifest in `schemas/manifest.json`.

| File | Schema |
| --- | --- |
| `ohho/datasets/lineage.schema.json` | `ohho.lineage.v1` |
| `ohho/datasets/dataset_import.schema.json` | `ohho.dataset_import.v1` |

## Local layout

The store root is `--store`, or `OHHO_DATASETS_ROOT`, or `~/.ohho/datasets`.
Dataset versions stay where [`dataset-versions.md`](dataset-versions.md)
puts them. Lineage adds three paths:

```text
<store>/
  lineage/graph.json
  imports/hf/<sha256(repo_id)>/<commit>.json
  .cache/hf/<sha256(repo_id)>/<commit>/    # Hub download cache
```

`lineage/graph.json` is the whole edge set. The file is replaced atomically.
The local process is a single writer. The console database is the concurrent
implementation. There is no SQLite database.

`graph.json` shape:

| Field | Role |
| --- | --- |
| `schema` | `ohho.lineage.v1` |
| `episodes` | Map of `episode_id` → episode row |
| `version_episodes` | Map of `version_id` → episode ids, in version order. List index is `position`. |
| `runs` | Map of `run_id` → `{run_id, dataset_version_id}` |
| `models` | Map of `model_version_id` → `{model_version_id, run_id}` |

Unknown fields are refused, including on a run or model row.

Episode id is `ep_` plus the first 32 hex characters of `content_hash`.
The hash is `sha256:` of the canonical episode payload
(`ohho.episode_id.v1`: task, fps, frames). The same frames share an id in
every version that copied them. A run id and a model version id are opaque
references, up to 128 characters, matching
`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`.

## Interface

`ohho.datasets.LineageStore` is the interface. `JsonLineageStore` is the
file implementation. `record_version_lineage` registers canonical episodes
and links them to a version id.

| Method | Effect |
| --- | --- |
| `put_episode(episode)` | Insert the row. The same id and content returns the stored row. |
| `episode(episode_id)` | Read one row, including a soft-deleted row. |
| `link_version_episodes(version_id, episode_ids)` | Store the ordered episode list. The same list is a no-op. |
| `link_run(run_id, dataset_version_id)` | Cite one dataset version from a run id. |
| `link_model(model_version_id, run_id)` | Cite the run that produced a model version. |
| `episodes_for_version(version_id)` | Episode ids in stored order. |
| `versions_for_episode(episode_id)` | Dataset versions that cite the episode. |
| `lineage_from_model(model_version_id)` | Model version → run → dataset version → episodes. |
| `models_trained_on(episode_id)` | Every model version trained on the episode. |
| `delete_episode(episode_id)` | Remove the episode, or soft-delete it when a pin still cites it. |

A run cites one dataset version. Linking it to a second version conflicts.
A model version cites one run. The version must already list episodes, and
the run must already exist, so a stored chain is complete.

```python
from ohho.datasets import JsonLineageStore

graph = JsonLineageStore()
chain = graph.lineage_from_model("mv_demo")
# chain.run_id, chain.dataset_version_id, chain.episode_ids
uses = graph.models_trained_on(chain.episode_ids[0])
```

`lineage_from_model` returns `ModelLineage`. Episode ids stay in the version's
order. A soft-deleted episode remains in that tuple.

`models_trained_on` returns `EpisodeTraining` rows sorted by dataset version
id, run id, then model version id. An episode that no run used returns an
empty tuple. A soft-deleted episode still answers.

## Delete

`delete_episode` reads `pinned` from each linked `version.json`.

* When every linked version is present and unpinned, the episode row is
  removed and dropped from each version's episode list. Runs and model
  versions stay. The dataset directory is left as it is.
* When any linked version is pinned, the hard delete is blocked. The episode
  stays on every edge, `deleted` becomes true, `deleted_at` is a UTC
  timestamp, and `delete_warning` explains the block. The result's
  `blocked` and `soft_deleted` fields are true and `warning` is that text.
  A second delete keeps the first `deleted_at` and does not rewrite the
  graph when the warning is unchanged.
* When a linked version id has no readable `version.json`, that edge is
  treated as pinned. The episode is soft-deleted with a warning.

The warning text is:

```text
soft-deleted: episode ep_… is used by pinned dataset version dv_… and was not removed
```

Several versions use `versions` and a comma-separated id list.

## Hugging Face import

```text
ohho dataset import-hf owner/name --revision main --store DIR
```

`--revision` may be a branch, tag, or commit. The stored revision is the
40-character commit the Hub resolved. `--json` prints `revision`,
`requested_revision`, `licence`, `non_commercial`, `revision_pinned`,
`licence_pinned`, and `warning`.

The import writes a normal dataset version (`dv_…`) whose `source_query`
root is `hf://<repo_id>@<commit>`, whose `licence` is the card licence, and
whose `pinned` flag is true. The content hash is the same canonical hash a
local build would compute for those frames. Video bytes are not copied; the
pinned commit identifies the Hub snapshot. The version's episodes are linked
in the lineage graph.

The import record (`ohho.dataset_import.v1`) is one file:

| Field | Role |
| --- | --- |
| `repo_id` | Hugging Face dataset id, `owner/name` or `name`. |
| `revision` | Resolved commit. Pinned. |
| `requested_revision` | The `--revision` argument, or null when it was omitted. |
| `licence` | Card licence. Pinned. Missing cards store `unspecified`. |
| `non_commercial` | True when the licence is non-commercial. |
| `revision_pinned` | Always true. |
| `licence_pinned` | Always true. |
| `version_id` | The dataset version this import materialised. |
| `created_at` | UTC timestamp, `YYYY-MM-DDTHH:MM:SSZ`. |

Importing that commit again returns the same record and does not rewrite it.
A later response that names a different licence for that commit is refused.
The directory name is the sha256 of `repo_id`, so the repo id is not a path.

`huggingface_hub` is optional. Without it, the default client raises
`DatasetImportError` and names `pip install 'ohho-os[datasets]'`. Tests pass
a hub object and do not open a network connection. The download cache under
`.cache/hf/` can be deleted; the pinned version directory remains.

A licence is non-commercial when it is a Creative Commons NonCommercial id
(`cc-by-nc`, `cc-by-nc-sa`, `cc-by-nc-nd`, with or without a version) or its
text contains "non-commercial". Comparison ignores case, spaces, and
underscores. Apache, MIT, BSD, CC BY, CC BY-SA, CC BY-ND, and `unspecified`
are not flagged. The import still completes. The CLI prints:

```text
warning: licence cc-by-nc-4.0 is non-commercial
```

## Proposed console tables (follow-up)

Do not create these tables from ohho-os. One local `graph.json` is the edge
set. One import file is one `dataset_imports` row. `dataset_versions` is
unchanged: an import sets the existing `licence` and `pinned` columns and
stores `hf://<repo_id>@<commit>` in `source_query`.

### `lineage_episodes`

| Column | JSON field | Type |
| --- | --- | --- |
| `episode_id` | `episode_id` | text primary key |
| `content_hash` | `content_hash` | text not null |
| `task` | `task` | text not null |
| `fps` | `fps` | text not null |
| `frame_count` | `frame_count` | integer not null |
| `source` | `source` | text not null |
| `source_index` | `source_index` | integer not null |
| `deleted` | `deleted` | boolean not null |
| `deleted_at` | `deleted_at` | timestamptz null |
| `delete_warning` | `delete_warning` | text null |

### `dataset_version_episodes`

| Column | JSON field | Type |
| --- | --- | --- |
| `dataset_version_id` | key of `version_episodes` | text not null, references `dataset_versions(version_id)` |
| `episode_id` | list entry | text not null, references `lineage_episodes(episode_id)` |
| `position` | list index, from 0 | integer not null |

Primary key `(dataset_version_id, episode_id)`. Unique
`(dataset_version_id, position)`. Index `(episode_id)`.

### `lineage_runs`

| Column | JSON field | Type |
| --- | --- | --- |
| `run_id` | `run_id` | text primary key |
| `dataset_version_id` | `dataset_version_id` | text not null, references `dataset_versions(version_id)` |

Index `(dataset_version_id)`. No manifest columns.

### `model_version_runs`

| Column | JSON field | Type |
| --- | --- | --- |
| `model_version_id` | `model_version_id` | text primary key |
| `run_id` | `run_id` | text not null, references `lineage_runs(run_id)` |

Index `(run_id)`.

### `dataset_imports`

| Column | JSON field | Type |
| --- | --- | --- |
| `repo_id` | `repo_id` | text not null |
| `revision` | `revision` | text not null |
| `requested_revision` | `requested_revision` | text null |
| `licence` | `licence` | text not null |
| `non_commercial` | `non_commercial` | boolean not null |
| `version_id` | `version_id` | text not null, references `dataset_versions(version_id)` |
| `created_at` | `created_at` | timestamptz not null |

Primary key `(repo_id, revision)`. `revision` and `licence` are immutable.
`non_commercial` is true exactly when the licence rule above matches.

### Queries

Model version → run → dataset version → episodes, in version order:

```sql
select m.model_version_id, m.run_id, r.dataset_version_id,
       e.episode_id, e.position
from model_version_runs m
join lineage_runs r on r.run_id = m.run_id
join dataset_version_episodes e
  on e.dataset_version_id = r.dataset_version_id
where m.model_version_id = $1
order by e.position;
```

Episode → every model version trained on it:

```sql
select e.episode_id, e.dataset_version_id, r.run_id, m.model_version_id
from dataset_version_episodes e
join lineage_runs r on r.dataset_version_id = e.dataset_version_id
join model_version_runs m on m.run_id = r.run_id
where e.episode_id = $1
order by e.dataset_version_id, r.run_id, m.model_version_id;
```

Pinned edges for a delete. A missing version row is pinned:

```sql
select e.dataset_version_id
from dataset_version_episodes e
left join dataset_versions v on v.version_id = e.dataset_version_id
where e.episode_id = $1
  and (v.version_id is null or v.pinned);
```

When that query returns a row, set `deleted`, `deleted_at`, and
`delete_warning` on `lineage_episodes` and keep the edges. When it returns
nothing, delete the `dataset_version_episodes` rows for that episode and
then the episode row. Leave runs and model versions in place.
