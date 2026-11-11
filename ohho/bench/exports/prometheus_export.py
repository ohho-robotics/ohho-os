"""Optional Prometheus exporter.

A Pushgateway URL receives text gauges, then this process deletes that
grouping key after the job TTL. Pushed series stay until something deletes
them, so the key is ``job``, ``model``, ``target``, and ``run_id``.
A remote-write URL receives a snappy-compressed protobuf and is not deleted.
The codec is stdlib-only. No Prometheus client library is imported.
"""

from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from ohho.bench.exports.base import ExportContext, ExportError, ExportJob
from ohho.bench.exports.metrics import grouping_labels, summary_numbers

_TEXT = "text/plain; version=0.0.4"
_USER_AGENT = "ohho-os"
_LATENCY = ("mean", "median", "p95", "p99", "min", "max")
_Sample = tuple[str, dict[str, str], float]


class PrometheusExporter:
    """Push gauges for one result, then delete a Pushgateway group after the TTL."""

    scheme = "prometheus"

    def export(
        self,
        document: Mapping[str, Any],
        job: ExportJob,
        target: str,
        context: ExportContext,
    ) -> dict[str, Any]:
        """Push ``document``. Return the link record stored on the result."""
        mode, url = _parse_target(target)
        labels = grouping_labels(document, kind=job.kind, spec=job.spec)
        samples = _samples(document, labels)
        if not samples:
            raise ExportError("result has no numeric gauges")
        if mode == "remote-write":
            endpoint = _clean_url(url)
            body = snappy_compress(
                encode_write_request(samples, _timestamp_ms(context))
            )
            _send(
                context,
                endpoint,
                body,
                "POST",
                {
                    "Content-Type": "application/x-protobuf",
                    "Content-Encoding": "snappy",
                    "X-Prometheus-Remote-Write-Version": "0.1.0",
                    "User-Agent": _USER_AGENT,
                },
            )
            return {
                "mode": "remote-write",
                "url": endpoint,
                "samples": len(samples),
            }
        endpoint = _grouping_url(url, labels)
        _send(
            context,
            endpoint,
            render_exposition(samples).encode("utf-8"),
            "POST",
            {"Content-Type": _TEXT, "User-Agent": _USER_AGENT},
        )
        deleted = True
        error = ""
        try:
            if job.ttl_minutes > 0:
                _sleep(context, job.ttl_minutes * 60.0)
            _send(
                context,
                endpoint,
                None,
                "DELETE",
                {"User-Agent": _USER_AGENT},
            )
        except ExportError as exc:
            deleted = False
            error = str(exc)
        link: dict[str, Any] = {
            "mode": "pushgateway",
            "url": endpoint,
            "ttl_minutes": job.ttl_minutes,
            "deleted": deleted,
        }
        if error:
            link["delete_error"] = error
        return link


def exporter() -> PrometheusExporter:
    """Factory used by exporter discovery."""
    return PrometheusExporter()


def render_exposition(samples: list[_Sample]) -> str:
    """Prometheus text exposition. No network I/O."""
    lines: list[str] = []
    seen: set[str] = set()
    for name, labels, value in samples:
        if name not in seen:
            lines.append(f"# TYPE {name} gauge")
            seen.add(name)
        body = ",".join(f'{key}="{_escape(labels[key])}"' for key in sorted(labels))
        lines.append(f"{name}{{{body}}} {format(float(value), '.10g')}")
    lines.append("")
    return "\n".join(lines)


def snappy_compress(data: bytes) -> bytes:
    """Snappy block encoding that stores literals only.

    Prometheus remote-write uses this block format. A general compressor is
    not required: a literals-only block is valid and decodes with the usual
    library.
    """
    out = bytearray()
    out.extend(_varint(len(data)))
    start = 0
    while start < len(data):
        chunk = data[start : start + 60]
        start += len(chunk)
        out.append((len(chunk) - 1) << 2)
        out.extend(chunk)
    return bytes(out)


def snappy_decompress(data: bytes) -> bytes:
    """Decode a literals-only block written by :func:`snappy_compress`."""
    length, pos = _read_varint(data, 0)
    out = bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        if tag & 0x03 != 0:
            raise ValueError("unsupported snappy element")
        kind = tag >> 2
        if kind < 60:
            size = kind + 1
        else:
            extra = kind - 59
            if pos + extra > len(data):
                raise ValueError("truncated snappy length")
            size = int.from_bytes(data[pos : pos + extra], "little") + 1
            pos += extra
        if pos + size > len(data):
            raise ValueError("truncated snappy literal")
        out.extend(data[pos : pos + size])
        pos += size
    if len(out) != length:
        raise ValueError("snappy length mismatch")
    return bytes(out)


def encode_write_request(samples: list[_Sample], timestamp_ms: int) -> bytes:
    """Protobuf ``WriteRequest`` for the remote-write endpoint.

    Each series is a gauge. Field numbers follow the Prometheus remote-write
    messages: TimeSeries 1, Sample value 1, Sample timestamp 2, Label name 1,
    Label value 2, Metadata 3, metadata type 1 (GAUGE = 2), family name 2.
    """
    if timestamp_ms < 0:
        raise ExportError("remote-write timestamp must be >= 0")
    parts: list[bytes] = []
    names: list[str] = []
    for name, labels, value in samples:
        parts.append(_length(1, _series(name, labels, value, timestamp_ms)))
        if name not in names:
            names.append(name)
    for name in names:
        parts.append(_length(3, _metadata(name)))
    return b"".join(parts)


def _samples(document: Mapping[str, Any], labels: Mapping[str, str]) -> list[_Sample]:
    numbers = summary_numbers(document)
    base = dict(labels)
    rows: list[_Sample] = []
    for stat in _LATENCY:
        value = numbers.get(f"latency/{stat}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(("ohho_latency_ms", {**base, "stat": stat}, float(value)))
    rate = numbers.get("throughput/hz")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        rows.append(("ohho_throughput_hz", dict(base), float(rate)))
    score = numbers.get("accuracy")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        accuracy_labels = dict(base)
        name = numbers.get("accuracy_name")
        if isinstance(name, str) and name:
            accuracy_labels["name"] = name
        rows.append(("ohho_accuracy", accuracy_labels, float(score)))
    energy = numbers.get("energy/mj")
    if isinstance(energy, (int, float)) and not isinstance(energy, bool):
        rows.append(("ohho_energy_mj", dict(base), float(energy)))
    power = numbers.get("power/w")
    if isinstance(power, (int, float)) and not isinstance(power, bool):
        rows.append(("ohho_power_w", dict(base), float(power)))
    return rows


def _parse_target(target: str) -> tuple[str, str]:
    text = target.strip()
    if not text:
        raise ExportError("prometheus export needs a pushgateway or remote-write URL")
    folded = text.casefold()
    if folded.startswith("remote-write:"):
        return "remote-write", text[len("remote-write:") :].strip()
    if folded.startswith("pushgateway:"):
        return "pushgateway", text[len("pushgateway:") :].strip()
    path = urlsplit(text).path.rstrip("/")
    if path.endswith("/api/v1/write"):
        return "remote-write", text
    return "pushgateway", text


def _grouping_url(url: str, labels: Mapping[str, str]) -> str:
    root = _gateway_root(url)
    path = "/metrics"
    for name in ("job", "model", "target", "run_id"):
        path += "/" + quote(name, safe="") + "/" + quote(labels[name], safe="")
    return root + path


def _gateway_root(url: str) -> str:
    cleaned = _clean_url(url)
    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/")
    if path.endswith("/metrics"):
        path = path[: -len("/metrics")]
    else:
        marker = path.find("/metrics/")
        if marker >= 0:
            path = path[:marker]
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))


def _clean_url(url: str) -> str:
    text = url.strip()
    if not text:
        raise ExportError("prometheus export needs a URL")
    parts = urlsplit(text)
    if parts.username or parts.password:
        raise ExportError("exporter URL must not include credentials")
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ExportError("exporter URL must be http or https")
    if not parts.netloc:
        raise ExportError("exporter URL must include a host")
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _send(
    context: ExportContext,
    url: str,
    body: bytes | None,
    method: str,
    headers: Mapping[str, str],
) -> None:
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=dict(headers),
    )
    opener = context.opener or urllib.request.urlopen
    try:
        with opener(request, timeout=5) as response:
            status = _status(response)
            _drain(response)
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        finally:
            exc.close()
        raise ExportError(f"HTTP {exc.code}") from exc
    except ExportError:
        raise
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        if len(detail) > 200:
            detail = detail[:200]
        raise ExportError(detail) from exc
    if status >= 400:
        raise ExportError(f"HTTP {status}")


def _sleep(context: ExportContext, seconds: float) -> None:
    if seconds <= 0:
        return
    sleeper = context.sleeper or time.sleep
    try:
        sleeper(seconds)
    except ExportError:
        raise
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise ExportError(detail) from exc


def _timestamp_ms(context: ExportContext) -> int:
    clock = context.now or time.time
    try:
        seconds = float(clock())
    except Exception as exc:
        raise ExportError("remote-write clock failed") from exc
    if seconds < 0:
        return 0
    return int(seconds * 1000)


def _status(response: Any) -> int:
    raw = getattr(response, "status", None)
    if raw is None:
        getcode = getattr(response, "getcode", None)
        raw = getcode() if callable(getcode) else 200
    return int(raw)


def _drain(response: Any) -> None:
    read = getattr(response, "read", None)
    if not callable(read):
        return
    try:
        read()
    except Exception:
        return


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _series(
    name: str, labels: Mapping[str, str], value: float, timestamp_ms: int
) -> bytes:
    pairs = {key: labels[key] for key in labels if key != "__name__"}
    pairs["__name__"] = name
    body = b"".join(_length(1, _label(key, pairs[key])) for key in sorted(pairs))
    body += _length(2, _sample(value, timestamp_ms))
    return body


def _label(name: str, value: str) -> bytes:
    return _string(1, name) + _string(2, value)


def _sample(value: float, timestamp_ms: int) -> bytes:
    return _fixed64(1, float(value)) + _key(2, 0) + _varint(timestamp_ms)


def _metadata(name: str) -> bytes:
    # MetricMetadata.type = GAUGE (enum 2), metric_family_name = name.
    return _key(1, 0) + _varint(2) + _string(2, name)


def _string(field: int, text: str) -> bytes:
    return _length(field, text.encode("utf-8"))


def _length(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _fixed64(field: int, value: float) -> bytes:
    return _key(field, 1) + struct.pack("<d", value)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _varint(value: int) -> bytes:
    if value < 0:
        raise ExportError("negative varint")
    out = bytearray()
    number = value
    while True:
        bits = number & 0x7F
        number >>= 7
        if number:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data) or shift > 63:
            raise ValueError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
