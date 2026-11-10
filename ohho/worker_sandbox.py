"""Code-only per-job worker sandbox (OHH-485).

:func:`container_run_spec` describes the container a managed worker should run
in. This module does not start a container, mount a host path, or open a
socket. :class:`JobScratch` is the local stand-in for the wiped scratch
directory. :func:`check_egress` decides whether a host is on the allowlist;
tests pass a fake resolver and a fake socket.
"""

from __future__ import annotations

import ipaddress
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONSOLE_API_HOST = "api.ohho-robotics.com"
REGISTRY_STORAGE_HOST = "registry.ohho-robotics.com"
JOB_TOKEN_ENV = "OHHO_JOB_TOKEN"

FORBIDDEN_CREDENTIAL_ENV = (
    "OHHO_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "WANDB_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
_FORBIDDEN_UPPER = frozenset(name.upper() for name in FORBIDDEN_CREDENTIAL_ENV)


@dataclass(frozen=True)
class EgressTarget:
    """One destination the worker may contact."""

    host: str
    ports: tuple[int, ...]
    purpose: str


DEFAULT_EGRESS: tuple[EgressTarget, ...] = (
    EgressTarget(CONSOLE_API_HOST, (443,), "console API"),
    EgressTarget(REGISTRY_STORAGE_HOST, (443,), "registry storage"),
)


@dataclass(frozen=True)
class WorkerSandboxConfig:
    """Isolation limits for one worker job.

    The root filesystem is read-only, there are no host mounts, and the
    process user is not root. Egress cannot name a host outside the console
    API and the registry storage allowlist. The scratch directory is wiped
    when the job ends. GPU count is recorded for the orchestrator; this
    module does not attach a device.
    """

    image: str = "ohho-worker:cpu"
    read_only_root: bool = True
    host_mounts: tuple[str, ...] = ()
    user: str = "ohho"
    uid: int = 10001
    gid: int = 10001
    cpu_limit: float = 2.0
    memory_mb: int = 4096
    gpu_count: int = 0
    pids_limit: int = 256
    scratch_mount: str = "/scratch"
    wipe_scratch: bool = True
    egress: tuple[EgressTarget, ...] = DEFAULT_EGRESS

    def __post_init__(self) -> None:
        if not self.image or not isinstance(self.image, str):
            raise ValueError("sandbox image must be a non-empty name")
        if self.read_only_root is not True:
            raise ValueError("sandbox root filesystem must be read-only")
        if tuple(self.host_mounts):
            raise ValueError("sandbox refuses host mounts")
        if not isinstance(self.user, str) or self.user.strip().lower() in {"", "root"}:
            raise ValueError("sandbox user must be non-root")
        if (
            isinstance(self.uid, bool)
            or not isinstance(self.uid, int)
            or self.uid <= 0
            or isinstance(self.gid, bool)
            or not isinstance(self.gid, int)
            or self.gid <= 0
        ):
            raise ValueError("sandbox user must be non-root")
        if (
            isinstance(self.cpu_limit, bool)
            or not isinstance(self.cpu_limit, (int, float))
            or self.cpu_limit <= 0
        ):
            raise ValueError("cpu_limit must be a positive number")
        if (
            isinstance(self.memory_mb, bool)
            or not isinstance(self.memory_mb, int)
            or self.memory_mb <= 0
        ):
            raise ValueError("memory_mb must be a positive integer")
        if (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count < 0
        ):
            raise ValueError("gpu_count must be zero or a positive integer")
        if (
            isinstance(self.pids_limit, bool)
            or not isinstance(self.pids_limit, int)
            or self.pids_limit <= 0
        ):
            raise ValueError("pids_limit must be a positive integer")
        if self.wipe_scratch is not True:
            raise ValueError("sandbox scratch directory must be wiped after the job")
        if not self.scratch_mount.startswith("/") or self.scratch_mount == "/":
            raise ValueError("scratch_mount must be an absolute in-container path")
        _validate_egress(self.egress)


def _validate_egress(egress: tuple[EgressTarget, ...]) -> None:
    default_ports = {item.host: set(item.ports) for item in DEFAULT_EGRESS}
    for target in egress:
        if not isinstance(target, EgressTarget):
            raise ValueError("egress entries must be EgressTarget values")
        if target.host not in default_ports:
            raise ValueError(
                f"egress host {target.host} is outside the console API and "
                "registry storage allowlist"
            )
        if not target.ports:
            raise ValueError(f"egress host {target.host} needs a port")
        for port in target.ports:
            if isinstance(port, bool) or not isinstance(port, int):
                raise ValueError(f"egress port for {target.host} is invalid")
        if not set(target.ports).issubset(default_ports[target.host]):
            raise ValueError(f"egress ports for {target.host} cannot be widened")


def container_run_spec(
    config: WorkerSandboxConfig | None = None,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready container spec. Does not start a container.

    ``extra_env`` must not carry the user's external-platform credentials or
    a raw ``OHHO_JOB_TOKEN``. The job token is named in the spec and injected
    by the orchestrator at runtime.
    """
    sandbox = config if config is not None else WorkerSandboxConfig()
    _reject_credential_env(extra_env or {})
    return {
        "apiVersion": "ohho.worker/v1",
        "kind": "WorkerJobSandbox",
        "image": sandbox.image,
        "readOnlyRootFilesystem": True,
        "hostMounts": [],
        "privileged": False,
        "capDrop": ["ALL"],
        "noNewPrivileges": True,
        "user": {
            "name": sandbox.user,
            "uid": sandbox.uid,
            "gid": sandbox.gid,
            "runAsNonRoot": True,
        },
        "resources": {
            "cpu": sandbox.cpu_limit,
            "memoryMb": sandbox.memory_mb,
            "pids": sandbox.pids_limit,
            "gpu": {
                "count": sandbox.gpu_count,
                "attached": False,
            },
        },
        "scratch": {
            "mountPath": sandbox.scratch_mount,
            "wipeOnExit": True,
        },
        "network": {
            "mode": "allowlist",
            "hostNetwork": False,
            "egress": [
                {
                    "host": target.host,
                    "ports": list(target.ports),
                    "purpose": target.purpose,
                }
                for target in sandbox.egress
            ],
        },
        "env": [
            {"name": "OHHO_MANAGED_WORKER", "value": "1"},
            {"name": JOB_TOKEN_ENV, "valueFrom": "runtime-job-token"},
        ],
        "credentials": "short-lived-job-token-only",
        "forbiddenEnv": list(FORBIDDEN_CREDENTIAL_ENV),
    }


def _reject_credential_env(extra_env: Mapping[str, str]) -> None:
    for name in extra_env:
        if name.upper() == JOB_TOKEN_ENV:
            raise ValueError(
                "the run spec must not embed OHHO_JOB_TOKEN; the orchestrator "
                "injects a short-lived job token at runtime"
            )
        if name.upper() in _FORBIDDEN_UPPER:
            raise ValueError(f"refusing external-platform credential env {name}")


@dataclass(frozen=True)
class EgressDecision:
    """Result of an egress check. ``allowed`` is false when the socket must not open."""

    allowed: bool
    host: str
    port: int
    reason: str
    addresses: tuple[str, ...] = ()


def check_egress(
    host: str,
    port: int,
    *,
    resolve: Callable[[str], Sequence[str]],
    config: WorkerSandboxConfig | None = None,
) -> EgressDecision:
    """Decide whether ``host:port`` is on the job allowlist.

    ``resolve`` returns IP addresses. This function does not open a socket.
    A name that is not on the allowlist is rejected before ``resolve`` is called.
    Any resolved address that is not a public unicast address is rejected, so a
    public name that resolves to a loopback, private, or link-local address
    (including the cloud metadata address) is refused.
    """
    sandbox = config if config is not None else WorkerSandboxConfig()
    try:
        normalized = _normalize_host(host)
    except ValueError as exc:
        return EgressDecision(False, host, _port_or_zero(port), str(exc))
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return EgressDecision(
            False, normalized, _port_or_zero(port), "egress port is invalid"
        )
    match = _match_egress(normalized, port, sandbox)
    if match is None:
        return EgressDecision(
            False,
            normalized,
            port,
            "host is outside the console API and registry storage allowlist",
        )
    try:
        resolved = resolve(normalized)
    except Exception as exc:
        return EgressDecision(
            False,
            normalized,
            port,
            "name could not be resolved: " + exc.__class__.__name__,
        )
    if isinstance(resolved, str) or not isinstance(resolved, Sequence):
        return EgressDecision(False, normalized, port, "resolver returned no addresses")
    addresses = tuple(str(item) for item in resolved)
    if not addresses:
        return EgressDecision(False, normalized, port, "resolver returned no addresses")
    for address in addresses:
        reason = _blocked_address_reason(address)
        if reason:
            return EgressDecision(False, normalized, port, reason, addresses)
    return EgressDecision(
        True,
        normalized,
        port,
        f"allowed for {match.purpose}",
        addresses,
    )


def open_allowed_connection(
    host: str,
    port: int,
    *,
    resolve: Callable[[str], Sequence[str]],
    socket_factory: Callable[[], Any],
    config: WorkerSandboxConfig | None = None,
) -> EgressDecision:
    """Open a socket only after :func:`check_egress` allows the destination.

    ``socket_factory`` is not called when the destination is refused.
    """
    decision = check_egress(host, port, resolve=resolve, config=config)
    if not decision.allowed:
        return decision
    sock = socket_factory()
    try:
        sock.connect((decision.addresses[0], decision.port))
    except Exception:
        close = getattr(sock, "close", None)
        if callable(close):
            close()
        raise
    return decision


def _port_or_zero(port: object) -> int:
    if isinstance(port, bool) or not isinstance(port, int):
        return 0
    return port


def _normalize_host(host: str) -> str:
    if not isinstance(host, str):
        raise ValueError("egress host must be a name")
    text = host.strip().lower().rstrip(".")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text or any(char in text for char in " \t\r\n/@\\"):
        raise ValueError("egress host is not a bare name")
    try:
        return text.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("egress host is not a bare name") from exc


def _match_egress(
    host: str, port: int, config: WorkerSandboxConfig
) -> EgressTarget | None:
    for target in config.egress:
        if host == target.host and port in target.ports:
            return target
    return None


def _blocked_address_reason(address: str) -> str:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return f"resolved address {address} is not an IP address"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global:
        return f"resolved address {address} is not a public address"
    return ""


class JobScratch:
    """A job scratch directory that is empty after :meth:`wipe`.

    The directory itself remains so a caller can check :meth:`is_empty`.
    Symlinks inside the scratch directory are unlinked and their targets are
    left in place. Use this as a context manager around the job body.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            self._root = Path(tempfile.mkdtemp(prefix="ohho-job-"))
        else:
            self._root = Path(root)
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The scratch directory."""
        return self._root

    def prepare(self) -> Path:
        """Wipe any leftover files and return the empty directory."""
        self.wipe()
        return self._root

    def wipe(self) -> None:
        """Remove every entry under the scratch directory."""
        self._root.mkdir(parents=True, exist_ok=True)
        for child in list(self._root.iterdir()):
            _discard(child)

    def is_empty(self) -> bool:
        """True when the directory exists and has no entries."""
        if not self._root.is_dir():
            return False
        return next(self._root.iterdir(), None) is None

    def verify_empty(self) -> None:
        """Raise ``RuntimeError`` when anything remains after the job."""
        if self.is_empty():
            return
        left = sorted(item.name for item in self._root.iterdir())
        raise RuntimeError("scratch dir is not empty after the job: " + ", ".join(left))

    def __enter__(self) -> JobScratch:
        self.prepare()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.wipe()
        self.verify_empty()
        return False


def _discard(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if path.is_file():
        _chmod_write(path)
        path.unlink()
        return
    for child in list(path.iterdir()):
        _discard(child)
    _chmod_write(path)
    path.rmdir()


def _chmod_write(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWRITE)
    except OSError:
        return


__all__ = [
    "CONSOLE_API_HOST",
    "DEFAULT_EGRESS",
    "FORBIDDEN_CREDENTIAL_ENV",
    "JOB_TOKEN_ENV",
    "REGISTRY_STORAGE_HOST",
    "EgressDecision",
    "EgressTarget",
    "JobScratch",
    "WorkerSandboxConfig",
    "check_egress",
    "container_run_spec",
    "open_allowed_connection",
]
