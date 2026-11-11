"""Optional W&B exporter.

The library is imported only when a run selects this scheme. Ohho never
reads the API key and never copies it onto the result or the run config.
The library uses the process environment on a machine the user controls.
A managed worker must not be given that key; the console exports afterwards.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ohho.bench.exports.base import ExportContext, ExportError, ExportJob
from ohho.bench.exports.metrics import summary_numbers

_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class WandbExporter:
    """One W&B run for one bench or optimize result."""

    scheme = "wandb"

    def export(
        self,
        document: Mapping[str, Any],
        job: ExportJob,
        target: str,
        context: ExportContext,
    ) -> dict[str, Any]:
        """Open one run, write config and summary, and return its URL."""
        library = _library(context)
        entity, project = _entity_project(target)
        refs = _artifact_refs(document, job.manifest)
        config = {
            "kind": job.kind,
            "spec": dict(job.spec),
            "manifest": dict(job.manifest),
            "artifact_refs": refs,
        }
        kwargs: dict[str, Any] = {
            "project": project,
            "config": config,
            "job_type": job.kind,
            "reinit": True,
        }
        if entity:
            kwargs["entity"] = entity
        run = library.init(**kwargs)
        if run is None:
            raise ExportError("run did not return a url")
        link: dict[str, str] | None = None
        try:
            summary = summary_numbers(document)
            _apply_summary(run, library, summary)
            _attach_artifacts(run, library, refs, _run_id(run, job))
            recorded = {
                "url": _run_url(run),
                "project": str(getattr(run, "project", "") or project),
            }
            entity_out = str(getattr(run, "entity", "") or entity or "")
            run_id = str(getattr(run, "id", "") or "")
            if entity_out:
                recorded["entity"] = entity_out
            if run_id:
                recorded["id"] = run_id
            link = recorded
        finally:
            finish = getattr(run, "finish", None)
            if callable(finish):
                try:
                    finish()
                except Exception as exc:
                    if link is None or not link.get("url"):
                        raise ExportError(str(exc) or "run finish failed") from exc
        if link is None or not link.get("url"):
            raise ExportError("run did not return a url")
        return link


def exporter() -> WandbExporter:
    """Factory used by exporter discovery."""
    return WandbExporter()


def _library(context: ExportContext) -> Any:
    if context.module is not None:
        return context.module
    loader = context.loader or _load_library
    try:
        return loader()
    except ImportError as exc:
        raise ExportError("the wandb package is not installed") from exc


def _load_library() -> Any:
    import wandb

    return wandb


def _entity_project(target: str) -> tuple[str | None, str]:
    chosen = target.strip()
    env_entity = os.environ.get("WANDB_ENTITY", "").strip() or None
    env_project = os.environ.get("WANDB_PROJECT", "").strip() or "ohho"
    if not chosen:
        return env_entity, env_project
    if chosen.count("/") > 1:
        raise ExportError("expected entity/project")
    if "/" in chosen:
        entity, project = (part.strip() for part in chosen.split("/", 1))
        if not entity or not project:
            raise ExportError("expected entity/project")
        return entity, project
    return env_entity, chosen


def _apply_summary(run: Any, library: Any, summary: Mapping[str, Any]) -> None:
    holder = getattr(run, "summary", None)
    if holder is not None and hasattr(holder, "update"):
        holder.update(dict(summary))
    log = getattr(run, "log", None)
    if not callable(log):
        log = getattr(library, "log", None)
    if callable(log):
        log(dict(summary))


def _attach_artifacts(
    run: Any,
    library: Any,
    refs: list[str],
    run_id: str,
) -> None:
    artifact_type = getattr(library, "Artifact", None)
    if not callable(artifact_type) or not refs:
        return
    try:
        artifact = artifact_type(_artifact_name(run_id), type="model")
    except Exception:
        return
    add = getattr(artifact, "add_reference", None)
    if callable(add):
        for ref in refs:
            try:
                add(ref)
            except Exception:
                continue
    log_artifact = getattr(run, "log_artifact", None)
    if not callable(log_artifact):
        return
    try:
        log_artifact(artifact)
    except Exception:
        return


def _artifact_refs(
    document: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[str]:
    refs: list[str] = []
    raw = manifest.get("manifest_ref")
    if not isinstance(raw, str) or not raw.strip():
        run = document.get("run")
        if isinstance(run, Mapping):
            raw = run.get("manifest_ref")
    if isinstance(raw, str) and raw.strip():
        text = raw.strip()
        refs.append(text)
        path = Path(text)
        if path.is_file():
            uri = path.resolve().as_uri()
            if uri not in refs:
                refs.append(uri)
    digest = manifest.get("artifact_sha")
    if not isinstance(digest, str):
        run = document.get("run")
        if isinstance(run, Mapping):
            digest = run.get("artifact_sha")
    if isinstance(digest, str) and digest.strip():
        token = digest.strip()
        if not token.startswith("sha256:"):
            token = "sha256:" + token
        if token not in refs:
            refs.append(token)
    return refs


def _artifact_name(run_id: str) -> str:
    cleaned = _NAME.sub("-", run_id).strip("-.") or "result"
    return f"ohho-{cleaned}"[:128]


def _run_id(run: Any, job: ExportJob) -> str:
    found = getattr(run, "id", None)
    if isinstance(found, str) and found.strip():
        return found.strip()
    manifest_id = job.manifest.get("run_id")
    if isinstance(manifest_id, str) and manifest_id.strip():
        return manifest_id.strip()
    return "result"


def _run_url(run: Any) -> str:
    get_url = getattr(run, "get_url", None)
    if callable(get_url):
        try:
            found = get_url()
        except Exception:
            found = None
        if found:
            return str(found)
    found = getattr(run, "url", None)
    if found:
        return str(found)
    return ""
