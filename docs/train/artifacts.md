# Artifact retention

Each training run keeps its files in one directory, next to `run_manifest.json`. Retention never rewrites that manifest or `metrics.json`. Checkpoints it removes are recorded in `artifact_index.json` (`ohho.artifacts/v1`).

The schema file is package data:

`ohho/train/schema/ohho.artifacts.v1.json`

Its `$id` is `https://ohho-robotics.com/schemas/ohho.artifacts/v1.json` and its title is `ohho.artifacts/v1`.

## Layout

```text
<experiment>/<run>/
  run_manifest.json          # ohho.run/v1; never deleted or rewritten
  metrics.json               # never deleted or rewritten
  artifact_index.json        # pin, promote, prune, and deletion records
  config.json                # run-root files are not deletion candidates
  policy.pt                  # trainer final weights; counted, not deleted
  checkpoint.json            # mock-trainer final weights; counted, not deleted
  model_final/               # counted, not deleted
  checkpoints/
    best/                    # role best; kept
    last.pt                  # role last; kept
    step-000100.pt           # role step; intermediate
    epoch-3/                 # role epoch; intermediate
  logs/                      # every file kept
    train.log
  eval/                      # every report kept
    report.json
```

`ensure_layout(run_dir)` creates `checkpoints/`, `logs/`, and `eval/`. A directory that contains `run_manifest.json` is one run. A directory of those runs is an experiment tree. Retention does not descend into a run looking for another manifest.

The role is the checkpoint's file or directory name with one weight suffix removed (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.bin`, `.json`):

| Stem | Role | Retention |
| --- | --- | --- |
| `best` | best | kept |
| `last` | last | kept |
| `step-<n>` or `step_<n>` | step | intermediate |
| `epoch-<n>` or `epoch_<n>` | epoch | intermediate |
| anything else directly under `checkpoints/` | other | intermediate |

Names are case-sensitive. `best-ema.pt` is `other`, not `best`.

## What retention deletes

An intermediate checkpoint is deleted only when all of these are true:

- the trial is pruned (`artifact_index.json` has `"pruned": true`)
- the run is not pinned and not promoted
- the checkpoint is not pinned and not promoted
- the newest mtime in the checkpoint is strictly older than N days (default 14)

The clock is that mtime. `pruned_at` records when the trial was marked pruned and is not the clock. A checkpoint stamped exactly N days ago is kept.

These are never deletion candidates:

- `run_manifest.json`, `metrics.json`, and `artifact_index.json`
- every file under `logs/` and `eval/`
- files at the run root, including `policy.pt`, `checkpoint.json`, `model_final/`, and `config.json`
- a checkpoint path that is not a direct child of `checkpoints/`, or that resolves outside the run

A symlink stored as a checkpoint is unlinked. The target is not removed. A pinned or promoted run keeps every checkpoint. A pinned or promoted checkpoint is kept even when the trial is pruned and the file is old.

An invalid `artifact_index.json` deletes nothing for that run. The job reports the error and leaves the manifest and metrics bytes unchanged.

## Commands

```bash
ohho run retain checkpoints/act --dry-run
ohho run retain checkpoints/act --days 14
ohho run retain experiments --experiment walk --json
ohho run storage checkpoints/act --json
ohho run storage experiments --experiment walk
```

`--dry-run` prints the checkpoints that would be deleted. It does not remove them and does not write `artifact_index.json`.

`python -m ohho.train.artifacts retain <path> --dry-run` is the same job.

| Exit | Meaning |
| --- | --- |
| 0 | The job finished. Nothing to delete is still success. |
| 1 | The path is missing, no run matches, the index is invalid, or a delete failed. |
| 2 | The command line is wrong, including a negative `--days`. |

A path to `run_manifest.json` is accepted as well as the directory that contains it. `--experiment` keeps a run when the sidecar's `experiment` field matches. When that field is null, the parent directory name is used instead.

## Console API

```python
from ohho.train.artifacts import (
    apply_retention,
    set_checkpoint_retention,
    set_run_retention,
    storage_used,
    storage_used_experiment,
)

set_run_retention(run_dir, pruned=True, experiment="walk")
set_run_retention(run_dir, pinned=True)
set_run_retention(run_dir, promoted=True)
set_checkpoint_retention(run_dir, "checkpoints/step-000100.pt", pinned=True)
set_checkpoint_retention(run_dir, "step-000100.pt", promoted=True)

usage = storage_used(run_dir)
rolled = storage_used_experiment(experiments, experiment="walk")
preview = apply_retention(experiments, dry_run=True)
apply_retention(experiments)
```

`storage_used` is the per-run total the console calls. `storage_used_experiment` sums every matching run and does not count files that sit outside those runs. Both return objects with `to_dict()`.

`storage_used` fields:

| Field | Meaning |
| --- | --- |
| `total_bytes` | Bytes still on disk. |
| `checkpoints_bytes` | `checkpoints/**`, plus root `policy.pt`, `checkpoint.json`, and `model_final/**`. |
| `logs_bytes` | `logs/**` |
| `eval_bytes` | `eval/**` |
| `metrics_bytes` | root `metrics.json` |
| `manifest_bytes` | root `run_manifest.json` |
| `other_bytes` | everything else on disk, including `artifact_index.json` and `config.json` |
| `deleted_bytes` | Sum of sidecar deletion records. Not included in `total_bytes`. |
| `experiment` | Sidecar experiment id, or null. |

`apply_retention` returns `dry_run`, `max_age_days`, `deleted_bytes`, and one entry per visited run. Each run entry has `deleted`, `kept`, and `errors`. A deleted item has `path`, `role`, `bytes`, `reason`, and `deleted_at`.

`set_run_retention(..., experiment=None)` clears the experiment id. Omitting `experiment` leaves it. Marking a trial pruned sets `pruned_at` once. Clearing prune clears `pruned_at`.

## Sidecar

`artifact_index.json` holds `experiment`, `pinned`, `promoted`, `pruned`, `pruned_at`, the checkpoints still on disk, and `deletions`. A deletion entry has `path`, `role`, `bytes`, `deleted_at` (`YYYY-MM-DDTHH:MM:SSZ`), and `reason`. The manifest file is not modified to point at this sidecar. The console reads `artifact_index.json` from the same directory as `run_manifest.json`.

This module does not open a database, read credentials, or use the network.
