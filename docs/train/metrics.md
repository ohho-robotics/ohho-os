# Training metrics (`ohho.metrics/v1`)

Built-in trainers record samples with one call:

```python
from ohho import log
from ohho.metrics import open_metrics

with open_metrics("checkpoints/act"):
    log(step, {"loss": loss})
```

`log` buffers samples in memory. They are appended to disk when the buffer fills (32 samples by default), when `flush()` is called, when the `with` block exits, and when the process exits. The base install does not depend on W&B or any other metrics service.

`open_metrics(run_dir)` binds the call to that directory. A nested open of the same directory reuses the active run. Diffusion, SmolVLA, OpenVLA, and PPO have no trainer in this build, so they do not log. ACT does, from `ohho.train.act.train_act`.

## Files

Both files sit in the run directory, next to `run_manifest.json`.

| File | Role |
| --- | --- |
| `metrics.jsonl` | Append-only log. One JSON object per line. This is the source of truth. |
| `metrics.json` | Snapshot rewritten from that log on each flush, for a dashboard that reads one object. |

`read_metrics(path)` accepts either the JSONL file or the run directory and returns the same object `metrics.json` stores.

## JSONL record

```json
{"schema":"ohho.metrics/v1","run_id":"ohho-<32 hex chars>","step":0,"metrics":{"loss":0.5},"wall_time":"2026-11-09T21:00:00Z"}
```

| Field | Meaning |
| --- | --- |
| `schema` | Always `ohho.metrics/v1`. |
| `run_id` | Native id for this run. A later open of the same file keeps this id. |
| `step` | Non-negative integer supplied by the trainer. |
| `metrics` | JSON object. Names map to numbers, strings, booleans, null, or nested JSON. |
| `wall_time` | UTC `YYYY-MM-DDTHH:MM:SSZ` when the sample was recorded. |

Lines are appended. A flush does not rewrite earlier lines.

## Dashboard object

`metrics.json` and `read_metrics()` return:

| Field | Meaning |
| --- | --- |
| `schema` | `ohho.metrics/v1`. |
| `run_id` | Native run id, or null when the file has no samples. |
| `records` | The JSONL objects, in order. |
| `series` | Numeric metrics only. Each name maps to `[{"step": <int>, "value": <number>}, ...]` in log order. Booleans are not series points. |
| `latest` | The last value logged for each metric name, including strings. |
| `final_loss` | Last numeric `loss`, or null when no `loss` sample exists. |
| `loss_curve` | `loss` series as `[step, value]` pairs. |

A chart can plot `series.loss` or `loss_curve`. A table can read `latest` or walk `records`.

## Uploader

ohho-os does not send metrics over the network. A console metrics store implements `upload(batch)` and installs it before training:

```python
from ohho.metrics import set_metrics_uploader

class LocalStore:
    def upload(self, batch: dict) -> None:
        """Persist batch locally. batch keys: schema, run_id, run_dir, records."""

set_metrics_uploader(LocalStore())
```

`upload` runs after the batch is on disk. An exception is printed on stderr and does not fail the run. `set_metrics_uploader(None)` clears the hook. A function of one argument works in place of an object. The console page that would chart these records is a follow-up; this package only writes the files and the hook.

## W&B mirror

The same `log` call is mirrored to Weights & Biases only when all of these are true:

1. `WANDB_API_KEY` is set in the environment.
2. `WANDB_MODE` is not `offline`, `disabled`, or `dryrun`.
3. `WANDB_DISABLED` is not true.
4. The `wandb` package imports.

The package is imported on that path only. A key saved by `wandb login` and not exported as `WANDB_API_KEY` does not turn the mirror on. Install the optional extra with `pip install 'ohho-os[wandb]'`, or install `wandb` yourself. The base install stays dependency-free.

`wandb.init` receives `config={"ohho_run_id": <native run id>}`. That is the native id on the W&B side. The project is `WANDB_PROJECT` or `ohho`.

`run_manifest.json` field `wandb_url` is the string returned by the client's `run.url` or `run.get_url()`. A blank string is ignored. OhhO does not build a URL from the entity, project, or run id. When the mirror is off, or the client returns no link, `wandb_url` is null. `ohho run show` prints that field.
