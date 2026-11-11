"""Shared command, URL, and version parsing for serving backends.

Engines stay external programs. Nothing in this module imports one.
"""

from __future__ import annotations

import re
import shlex
import socket
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ServingBackendError

_SEMVER = re.compile(r"\b[vV]?(\d+\.\d+(?:\.\d+)?(?:\.post\d+|rc\d+|[ab]\d+)?)\b")


def as_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy a backend config, or an empty dict."""
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ServingBackendError("config must be a mapping")
    return dict(config)


def require_model(model: str) -> str:
    """Return a stripped model id, or raise when it is empty."""
    text = str(model).strip()
    if not text:
        raise ServingBackendError("model is required")
    return text


def resolve_host(config: Mapping[str, Any]) -> str:
    """Return the bind host. The default is loopback."""
    host = str(config.get("host") or "127.0.0.1").strip()
    if not host or any(char.isspace() for char in host) or host.startswith("-"):
        raise ServingBackendError(f"invalid host {host!r}")
    return host


def free_port() -> int:
    """Ask the OS for an unused TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_port(config: Mapping[str, Any], default: int) -> int:
    """Return the listen port.

    A dry run with no port uses the engine's documented default. A real
    launch with no port picks a free loopback port.
    """
    raw = config.get("port")
    if raw is None or raw == "":
        if config.get("dry_run"):
            return default
        return free_port()
    try:
        port = int(raw)
    except (TypeError, ValueError) as exc:
        raise ServingBackendError(f"invalid port {raw!r}") from exc
    if port < 0 or port > 65535:
        raise ServingBackendError(f"port out of range: {port}")
    if port == 0:
        return free_port()
    return port


def extra_args(config: Mapping[str, Any]) -> list[str]:
    """Return extra argv entries. A shell string is refused."""
    raw = config.get("extra_args") or []
    if isinstance(raw, str):
        raise ServingBackendError(
            "extra_args must be a list of strings, not a shell command"
        )
    if not isinstance(raw, (list, tuple)):
        raise ServingBackendError("extra_args must be a list of strings")
    args: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ServingBackendError("extra_args must be a list of strings")
        args.append(item)
    return args


def command_prefix(config: Mapping[str, Any], default: Sequence[str]) -> list[str]:
    """Resolve argv before the engine's own arguments.

    ``argv_prefix`` replaces the whole prefix (used by tests). ``executable``
    replaces the first token of ``default``.
    """
    prefix = config.get("argv_prefix")
    if prefix is not None:
        if isinstance(prefix, str):
            parts = [prefix]
        elif isinstance(prefix, (list, tuple)) and prefix:
            parts = [str(part) for part in prefix]
        else:
            raise ServingBackendError("argv_prefix must be a non-empty list of strings")
        if not all(parts):
            raise ServingBackendError("argv_prefix must be a non-empty list of strings")
        return parts
    exe = config.get("executable")
    if exe:
        text = str(exe).strip()
        if not text:
            raise ServingBackendError("executable must be a non-empty string")
        return [text, *[str(part) for part in default[1:]]]
    return [str(part) for part in default]


def version_command(config: Mapping[str, Any], default: Sequence[str]) -> list[str]:
    """Return the argv that prints an engine version."""
    custom = config.get("version_argv")
    if custom is not None:
        if not isinstance(custom, (list, tuple)) or not custom:
            raise ServingBackendError(
                "version_argv must be a non-empty list of strings"
            )
        return [str(part) for part in custom]
    if config.get("argv_prefix") is not None or config.get("executable"):
        return command_prefix(config, default[:1] or default) + ["--version"]
    return [str(part) for part in default]


def env_overrides(config: Mapping[str, Any]) -> dict[str, str] | None:
    """Return environment updates for the engine process."""
    raw = config.get("env")
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise ServingBackendError("env must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def working_directory(config: Mapping[str, Any]) -> str | None:
    """Return an optional working directory."""
    raw = config.get("cwd")
    if raw is None or raw == "":
        return None
    return str(raw)


def origin(host: str, port: int) -> str:
    """Return ``http://host:port``, bracketing IPv6 hosts."""
    if ":" in host and not host.startswith("["):
        return f"http://[{host}]:{port}"
    return f"http://{host}:{port}"


def openai_base(host: str, port: int) -> str:
    """Return the OpenAI-compatible base URL, including ``/v1``."""
    return origin(host, port) + "/v1"


def quote_command(argv: Sequence[str]) -> str:
    """Return a copy-pasteable command line for an error message."""
    return " ".join(shlex.quote(part) for part in argv)


def semver_from_text(text: str) -> str | None:
    """Return the first semver, preferring a line that says ``version``."""
    for line in text.splitlines():
        if "version" in line.lower():
            match = _SEMVER.search(line)
            if match:
                return match.group(1)
    match = _SEMVER.search(text)
    if match is None:
        return None
    return match.group(1)


def positive_int(value: object, name: str) -> int:
    """Return a positive integer config value."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ServingBackendError(f"{name} must be an integer") from exc
    if number < 1:
        raise ServingBackendError(f"{name} must be >= 1")
    return number


def optional_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    """Return a config int when the key is present."""
    if key not in config or config[key] is None:
        return None
    return positive_int(config[key], key)
