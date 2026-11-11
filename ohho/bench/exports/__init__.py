"""Optional exporters for a bench or optimize result.

``export_result`` is the post-run hook. ``ohho bench --export`` calls it.
An optimize command calls the same function with ``kind="optimize"``. This
repository does not ship that command. Console-side export for a managed
cloud job is also outside this package: the worker must not receive an API
key, and the console calls this hook after the job with the key it stores.

A failure is a warning string. The function does not raise for a bad spec,
a missing optional library, or a refused push. It may add ``exports`` on
``document``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ohho.bench.exports.base import (
    ExportContext,
    ExportError,
    ExportJob,
    ExportSpecError,
    check_ttl,
    parse_spec,
    public_value,
)

__all__ = [
    "ExportContext",
    "ExportError",
    "ExportJob",
    "ExportSpecError",
    "export_result",
    "parse_spec",
]


def export_result(
    document: dict[str, Any],
    specs: Sequence[str],
    *,
    kind: str,
    spec: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    ttl_minutes: float = 5,
    exporters: Sequence[Any] | None = None,
    context: ExportContext | None = None,
) -> list[str]:
    """Publish ``document`` to each spec. Return warning strings.

    ``spec`` is the job (artifact, target, batch, and the caller's other
    settings). ``manifest`` defaults to the identity stored on
    ``document["run"]``. ``ttl_minutes`` is how long a Pushgateway grouping
    key stays before that adapter deletes it. Remote-write ignores the TTL.
    """
    warnings: list[str] = []
    items = list(specs)
    if not items:
        return warnings
    try:
        ttl = check_ttl(ttl_minutes)
    except ExportError as exc:
        warnings.append(str(exc))
        return warnings
    job = ExportJob(
        kind=(kind or "bench").strip() or "bench",
        spec=_as_mapping(spec),
        manifest=_manifest(document, manifest),
        ttl_minutes=ttl,
    )
    try:
        found = list(exporters) if exporters is not None else _load_exporters()
    except Exception as exc:
        warnings.append(f"export failed: {exc}")
        return warnings
    table: dict[str, Any] = {}
    for plugin in found:
        scheme = getattr(plugin, "scheme", None)
        if isinstance(scheme, str) and scheme and scheme not in table:
            table[scheme] = plugin
    ctx = context or ExportContext()
    for raw in items:
        _one(document, raw, job, table, ctx, warnings)
    return warnings


def _load_exporters() -> list[Any]:
    from ohho.bench.exports.registry import discover_exporters

    return discover_exporters()


def _one(
    document: dict[str, Any],
    raw: object,
    job: ExportJob,
    table: Mapping[str, Any],
    context: ExportContext,
    warnings: list[str],
) -> None:
    try:
        scheme, target = parse_spec(raw)
    except ExportSpecError as exc:
        warnings.append(str(exc))
        return
    plugin = table.get(scheme)
    if plugin is None:
        warnings.append(f"unknown exporter {scheme!r}")
        return
    try:
        link = plugin.export(document, job, target, context)
    except Exception as exc:
        warnings.append(f"{scheme} export failed: {exc}")
        return
    if link is None:
        return
    if not isinstance(link, dict):
        warnings.append(
            f"{scheme} export failed: exporter returned {type(link).__name__}"
        )
        return
    exports = document.get("exports")
    if exports is None:
        exports = {}
        document["exports"] = exports
    if not isinstance(exports, dict):
        warnings.append(
            f"{scheme} export failed: result exports field is not an object"
        )
        return
    note = link.get("delete_error")
    exports[scheme] = link
    if isinstance(note, str) and note:
        warnings.append(f"{scheme} export failed: {note}")


def _as_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    copied = public_value(dict(value))
    if isinstance(copied, dict):
        return copied
    return {}


def _manifest(
    document: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        copied = public_value(dict(manifest))
        if isinstance(copied, dict):
            return copied
        return {}
    run = document.get("run")
    if not isinstance(run, Mapping):
        return {}
    return _as_mapping(
        {
            "manifest_ref": run.get("manifest_ref"),
            "artifact_sha": run.get("artifact_sha"),
            "model_sha": run.get("model_sha"),
            "run_id": run.get("id"),
            "workload": run.get("target"),
        }
    )
