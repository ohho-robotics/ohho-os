"""Training metrics: one call, a local append-only store, an optional W&B mirror.

``ohho.log(step, metrics)`` is the only metrics call built-in trainers use.
Samples stay in memory until the buffer fills, :meth:`MetricsSession.flush`
runs, or the run closes. Closing (and process exit) appends them to
``metrics.jsonl`` beside the run and rewrites ``metrics.json``, a snapshot a
dashboard can read. The same records are handed to a :class:`MetricsUploader`
when one is installed. ohho-os does not open a network connection for that
hook.

The W&B mirror starts only when ``WANDB_API_KEY`` is set and the mode is not
offline or disabled. ``wandb`` is imported on that path only. A run link is
stored only when the client returns one. The native run id is passed to the
client as ``config["ohho_run_id"]``.

The on-disk format is documented in ``docs/train/metrics.md``.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SCHEMA_ID = "ohho.metrics/v1"
METRICS_FILENAME = "metrics.jsonl"
SUMMARY_FILENAME = "metrics.json"
DEFAULT_BUFFER_SIZE = 32
_WALL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_OFFLINE_MODES = frozenset({"offline", "disabled", "dryrun"})
_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})

_STACK: list[MetricsSession] = []
_LAST: MetricsSession | None = None
_CLOSED: dict[str, MetricsSession] = {}
_UPLOADER: Any = None
_CLIENT: Any = None
_CLIENT_OVERRIDE = False
_LOCK = threading.Lock()


class MetricsError(RuntimeError):
    """A metrics run could not be opened, written, or read."""


class MetricsUploader(Protocol):
    """Sink for one flushed batch of metric records.

    ``upload`` is called after the batch is appended to ``metrics.jsonl``.
    ohho-os does not ship a network implementation. A console metrics store
    can install one with :func:`set_metrics_uploader`. An exception from
    ``upload`` is reported on stderr and does not fail the training run.
    """

    def upload(self, batch: Mapping[str, Any]) -> None:
        """Accept one flushed batch.

        ``batch`` has ``schema``, ``run_id``, ``run_dir``, and ``records``.
        """
        ...


class MetricsSession:
    """One open metrics run bound to a directory."""

    def __init__(self, run_dir: Path, *, run_id: str | None, buffer_size: int) -> None:
        self.run_dir = run_dir
        self.path = run_dir / METRICS_FILENAME
        self.summary_path = run_dir / SUMMARY_FILENAME
        self.buffer_size = buffer_size
        self.run_id = _adopt_run_id(self.path, run_id)
        self.wandb_url: str | None = None
        self.closed = False
        self._buffer: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = (
            _read_records(self.path) if self.path.is_file() else []
        )
        self._lock = threading.Lock()
        self._mirror = _start_mirror(self.run_id)

    def record(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Buffer one sample. Flush when the buffer is full."""
        sample = _sample(self.run_id, step, metrics)
        with self._lock:
            if self.closed:
                raise MetricsError("metrics run is closed")
            self._buffer.append(sample)
            full = len(self._buffer) >= self.buffer_size
        if full:
            self.flush()

    def flush(self) -> None:
        """Append buffered samples to the metrics file and notify sinks."""
        batch = self._take_batch(rewrite_summary=True)
        self._publish(batch)

    def close(self) -> None:
        """Flush, finish the W&B mirror, and keep this session as the last run."""
        global _LAST
        with self._lock:
            if self.closed:
                return
            batch = self._take_locked(rewrite_summary=True)
            self.closed = True
        _LAST = self
        self._publish(batch)
        if self._mirror is not None:
            self._mirror.finish()
            if self._mirror.url:
                self.wandb_url = self._mirror.url
        _CLOSED[_dir_key(self.run_dir)] = self

    def _take_batch(self, *, rewrite_summary: bool) -> list[dict[str, Any]]:
        with self._lock:
            if self.closed:
                return []
            return self._take_locked(rewrite_summary=rewrite_summary)

    def _take_locked(self, *, rewrite_summary: bool) -> list[dict[str, Any]]:
        batch = list(self._buffer)
        if batch:
            _append_records(self.path, batch)
            self._history.extend(batch)
            self._buffer.clear()
        if batch or rewrite_summary:
            _write_summary(self.summary_path, _dashboard(self.run_id, self._history))
        return batch

    def _publish(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        if self._mirror is not None:
            for sample in batch:
                self._mirror.log(int(sample["step"]), sample["metrics"])
        _notify_uploader(self, batch)


def log(step: int, metrics: Mapping[str, Any]) -> None:
    """Record one training sample for the active metrics run.

    ``step`` is a non-negative integer. ``metrics`` is a JSON object of
    names to numbers, strings, booleans, nulls, or nested JSON values.
    Samples stay buffered until the buffer fills or the run is closed.
    """
    session = _current()
    if session is None:
        raise MetricsError(
            "ohho.log() needs an open metrics run; open one with open_metrics()"
        )
    session.record(step, metrics)


@contextmanager
def open_metrics(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> Iterator[MetricsSession]:
    """Bind ``ohho.log`` to ``run_dir`` until the block exits.

    A nested open of the same directory reuses the active session and flushes
    it on the way out without finishing the mirror. The outermost block
    flushes, writes the summary, and finishes W&B.
    """
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int):
        raise ValueError("buffer_size must be an integer >= 1")
    if buffer_size < 1:
        raise ValueError("buffer_size must be an integer >= 1")
    resolved = Path(run_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    current = _current()
    if (
        current is not None
        and not current.closed
        and _same_dir(current.run_dir, resolved)
    ):
        try:
            yield current
        finally:
            if not current.closed:
                current.flush()
        return
    session = MetricsSession(resolved, run_id=run_id, buffer_size=buffer_size)
    with _LOCK:
        _STACK.append(session)
    try:
        yield session
    finally:
        try:
            session.close()
        finally:
            with _LOCK:
                if session in _STACK:
                    _STACK.remove(session)


def read_metrics(path: str | Path) -> dict[str, Any]:
    """Read ``metrics.jsonl`` (or a run directory) into a dashboard document.

    The document has ``schema``, ``run_id``, ``records``, ``series``,
    ``latest``, ``final_loss``, and ``loss_curve``. ``series`` lists numeric
    metrics as ``{"step", "value"}`` in log order. ``loss_curve`` is the
    ``loss`` series as ``[step, value]`` pairs.
    """
    file = _locate(path)
    records = _read_records(file)
    run_id = records[0]["run_id"] if records else None
    if records and any(record["run_id"] != run_id for record in records):
        raise MetricsError(f"{file}: mixed run ids")
    return _dashboard(run_id, records)


def summary_path(run_dir: str | Path) -> Path:
    """Path of the dashboard snapshot (``metrics.json``) for ``run_dir``."""
    return Path(run_dir).expanduser() / SUMMARY_FILENAME


def log_path(run_dir: str | Path) -> Path:
    """Path of the append-only metrics file (``metrics.jsonl``) for ``run_dir``."""
    return Path(run_dir).expanduser() / METRICS_FILENAME


def set_metrics_uploader(uploader: MetricsUploader | Any | None) -> None:
    """Install the console uploader, or ``None`` to clear it.

    An object with ``upload(batch)`` or a function of one argument both work.
    The default is no uploader. This process does not contact the network.
    """
    global _UPLOADER
    _UPLOADER = uploader


def set_wandb_client(client: Any | None) -> None:
    """Use ``client`` instead of importing wandb. ``None`` restores the import.

    Tests pass a fake module whose ``init`` returns a run. Production leaves
    this unset. The client is used only when :func:`wandb_configured` is true.
    """
    global _CLIENT, _CLIENT_OVERRIDE
    _CLIENT = client
    _CLIENT_OVERRIDE = client is not None


def wandb_configured(environ: Mapping[str, str] | None = None) -> bool:
    """True when the W&B mirror should start.

    The API key has to be in ``WANDB_API_KEY``. ``WANDB_MODE`` of ``offline``,
    ``disabled``, or ``dryrun``, and ``WANDB_DISABLED`` of true, keep the
    mirror off. This does not import wandb and does not read a login file.
    """
    env = os.environ if environ is None else environ
    disabled = env.get("WANDB_DISABLED", "").strip().lower()
    if disabled in _DISABLED_VALUES:
        return False
    mode = env.get("WANDB_MODE", "").strip().lower()
    if mode in _OFFLINE_MODES:
        return False
    return bool(env.get("WANDB_API_KEY", "").strip())


def last_metrics_run() -> MetricsSession | None:
    """The session that most recently closed, or the one close has started."""
    return _LAST


def wandb_url_for(run_dir: str | Path | None) -> str | None:
    """Client-provided W&B link for ``run_dir``, or ``None`` when there is none."""
    if run_dir is None:
        return None
    session = _CLOSED.get(_dir_key(Path(run_dir)))
    if session is None or not session.wandb_url:
        return None
    return session.wandb_url


def clear_last_metrics_run() -> None:
    """Forget closed sessions. Does not delete files."""
    global _LAST
    _LAST = None
    _CLOSED.clear()


def reset_metrics_state() -> None:
    """Close open runs and clear the uploader, W&B client, and last run.

    Tests use this between cases. It does not delete metrics files.
    """
    global _LAST, _UPLOADER, _CLIENT, _CLIENT_OVERRIDE
    while True:
        with _LOCK:
            if not _STACK:
                break
            session = _STACK[-1]
        session.close()
        with _LOCK:
            if session in _STACK:
                _STACK.remove(session)
    _LAST = None
    _CLOSED.clear()
    _UPLOADER = None
    _CLIENT = None
    _CLIENT_OVERRIDE = False


def _current() -> MetricsSession | None:
    with _LOCK:
        return _STACK[-1] if _STACK else None


def _flush_open_runs() -> None:
    """atexit hook: append any samples still buffered in an open run."""
    with _LOCK:
        sessions = list(_STACK)
    for session in sessions:
        try:
            session.flush()
        except Exception as exc:
            print(
                f"metrics flush on exit failed: {type(exc).__name__}",
                file=sys.stderr,
            )


atexit.register(_flush_open_runs)


def _adopt_run_id(path: Path, requested: str | None) -> str:
    if requested is not None and (
        not isinstance(requested, str) or not requested.strip()
    ):
        raise ValueError("run_id must be a non-empty string")
    chosen = requested.strip() if isinstance(requested, str) else _new_run_id()
    if not path.is_file():
        return chosen
    records = _read_records(path)
    if not records:
        return chosen
    existing = records[0]["run_id"]
    if requested is not None and requested.strip() != existing:
        raise MetricsError(
            f"{path}: run_id is {existing}; refusing to continue as {requested.strip()}"
        )
    return str(existing)


def _new_run_id() -> str:
    return "ohho-" + uuid.uuid4().hex


def _sample(run_id: str, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "step": step,
        "metrics": _metrics_object(metrics),
        "wall_time": _utc_now(),
    }


def _metrics_object(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping of names to JSON values")
    try:
        encoded = json.dumps(dict(metrics), allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metrics must be JSON values: {exc}") from exc
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("metrics must be a JSON object")
    return loaded


def _dashboard(run_id: str | None, records: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = {}
    latest: dict[str, Any] = {}
    for record in records:
        step = record["step"]
        values = record["metrics"]
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            latest[str(key)] = value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            series.setdefault(str(key), []).append({"step": step, "value": value})
    loss = series.get("loss", [])
    final_loss = loss[-1]["value"] if loss else None
    return {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "records": records,
        "series": series,
        "latest": latest,
        "final_loss": final_loss,
        "loss_curve": [[point["step"], point["value"]] for point in loss],
    }


def _locate(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / METRICS_FILENAME
    if not candidate.is_file():
        raise MetricsError(f"metrics file does not exist: {candidate}")
    return candidate


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetricsError(f"{path}:{number}: invalid JSON") from exc
        _check_record(item, path, number)
        records.append(item)
    if records:
        run_id = records[0]["run_id"]
        if any(record["run_id"] != run_id for record in records):
            raise MetricsError(f"{path}: mixed run ids")
    return records


def _check_record(item: Any, path: Path, number: int) -> None:
    where = f"{path}:{number}"
    if not isinstance(item, dict):
        raise MetricsError(f"{where}: record must be an object")
    if item.get("schema") != SCHEMA_ID:
        raise MetricsError(f"{where}: schema must be {SCHEMA_ID}")
    run_id = item.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise MetricsError(f"{where}: run_id must be a non-empty string")
    step = item.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise MetricsError(f"{where}: step must be a non-negative integer")
    if not isinstance(item.get("metrics"), dict):
        raise MetricsError(f"{where}: metrics must be an object")
    wall = item.get("wall_time")
    if not isinstance(wall, str) or _WALL_TIME.search(wall) is None:
        raise MetricsError(f"{where}: wall_time must be UTC YYYY-MM-DDTHH:MM:SSZ")


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record, allow_nan=False, ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_summary(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _notify_uploader(session: MetricsSession, batch: list[dict[str, Any]]) -> None:
    uploader = _UPLOADER
    if uploader is None:
        return
    copies: list[dict[str, Any]] = []
    for record in batch:
        item = dict(record)
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            item["metrics"] = dict(metrics)
        copies.append(item)
    payload = {
        "schema": SCHEMA_ID,
        "run_id": session.run_id,
        "run_dir": str(session.run_dir),
        "records": copies,
    }
    try:
        upload = getattr(uploader, "upload", None)
        if callable(upload):
            upload(payload)
        elif callable(uploader):
            uploader(payload)
        else:
            raise TypeError("metrics uploader must provide upload() or be a function")
    except Exception as exc:
        print(
            f"metrics upload failed: {type(exc).__name__}",
            file=sys.stderr,
        )


def _utc_now() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dir_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def _same_dir(left: Path, right: Path) -> bool:
    return _dir_key(left) == _dir_key(right)


def _import_wandb() -> Any:
    import wandb

    return wandb


def _load_client() -> Any:
    if _CLIENT_OVERRIDE:
        return _CLIENT
    return _import_wandb()


def _start_mirror(run_id: str) -> _WandbMirror | None:
    if not wandb_configured():
        return None
    try:
        client = _load_client()
    except ImportError:
        print("W&B mirror is not active: wandb is not installed", file=sys.stderr)
        return None
    mirror = _WandbMirror(client, run_id)
    try:
        mirror.start()
    except Exception as exc:
        print(
            f"W&B mirror is not active: {type(exc).__name__}",
            file=sys.stderr,
        )
        return None
    return mirror


class _WandbMirror:
    """Lazy W&B client. A link is kept only when the client returns one."""

    def __init__(self, client: Any, run_id: str) -> None:
        self._client = client
        self._run_id = run_id
        self._run: Any = None
        self._disabled = False
        self._finished = False
        self.url: str | None = None

    def start(self) -> None:
        project = os.environ.get("WANDB_PROJECT", "").strip() or "ohho"
        self._run = self._client.init(
            project=project,
            config={"ohho_run_id": self._run_id},
            reinit=True,
        )
        if self._run is None:
            self._run = getattr(self._client, "run", None)
        self._stamp_run_id()
        self._capture_url()

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        if self._disabled or self._run is None:
            return
        payload = dict(metrics)
        try:
            self._run.log(payload, step=step)
        except Exception as exc:
            self._disabled = True
            print(
                f"W&B mirror stopped: {type(exc).__name__}",
                file=sys.stderr,
            )

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._capture_url()
        finish = getattr(self._run, "finish", None) if self._run is not None else None
        if callable(finish):
            try:
                finish()
            except Exception as exc:
                print(
                    f"W&B mirror finish failed: {type(exc).__name__}",
                    file=sys.stderr,
                )
        self._capture_url()

    def _stamp_run_id(self) -> None:
        if self._run is None:
            return
        config = getattr(self._run, "config", None)
        if isinstance(config, dict):
            config["ohho_run_id"] = self._run_id
            return
        update = getattr(config, "update", None)
        if callable(update):
            update({"ohho_run_id": self._run_id}, allow_val_change=True)

    def _capture_url(self) -> None:
        found = _url_from(self._run)
        if found is None:
            found = _url_from(getattr(self._client, "run", None))
        if found:
            self.url = found


def _url_from(run: Any) -> str | None:
    """Return a link the client already has. Never build one."""
    if run is None:
        return None
    for name in ("url", "get_url"):
        value = getattr(run, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "METRICS_FILENAME",
    "SCHEMA_ID",
    "SUMMARY_FILENAME",
    "MetricsError",
    "MetricsSession",
    "MetricsUploader",
    "clear_last_metrics_run",
    "last_metrics_run",
    "log",
    "log_path",
    "open_metrics",
    "read_metrics",
    "reset_metrics_state",
    "set_metrics_uploader",
    "set_wandb_client",
    "summary_path",
    "wandb_configured",
    "wandb_url_for",
]
