"""``ohho doctor --agent`` readiness report (OHH-271 phase 7).

Checks are independent. A ``skip`` is not a failure: login is optional for
``sim://``, and ``ohho rules install`` (phase 6) is not in this build yet.
The access token is never placed in a check, a log line, or either output
format.

Phase 6 hook
------------
``ohho rules install`` does not exist yet. When it does, it should ship a
module ``ohho.agent_rules`` with::

    def installed_clients() -> dict[str, str | None]:
        '''Client id → rules path, or None when that client has no rules.'''

This report imports that function when the module is present. Until then
the rules check is ``skip: ohho rules install not available in this build``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO

from . import __version__
from .auth import AuthError, default_credentials_path, load_credentials, whoami
from .mcp_connect import inspect_ohho_servers
from .registry import UnknownRobot
from .robot import Robot
from .schema import ConnectionState

_INSTALL_METHODS = (
    "pipx",
    "pip --user",
    "venv",
    "editable source",
    "unknown",
)
_RULES_SKIP = "skip: ohho rules install not available in this build"


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def _under(path: str, parent: str) -> bool:
    child = _norm(path).rstrip("/")
    root = _norm(parent).rstrip("/")
    return child == root or child.startswith(root + "/")


def classify_install_method(
    package_file: str,
    prefix: str,
    base_prefix: str,
    user_site: Optional[str],
    editable: bool,
) -> str:
    """Label an install as pipx, pip --user, venv, editable source, or unknown.

    ``pipx`` wins over the others because a pipx install is also a venv.
    An editable checkout wins over ``pip --user`` and ``venv``. ``unknown``
    is a successful classification, not an error.
    """
    pkg = _norm(package_file)
    pre = _norm(prefix)
    if "/pipx/" in pkg or "/pipx/" in pre:
        return "pipx"
    if editable:
        return "editable source"
    if user_site and _under(package_file, user_site):
        return "pip --user"
    if _norm(prefix) != _norm(base_prefix):
        return "venv"
    return "unknown"


def _looks_editable(package_file: str) -> bool:
    """Best-effort: PEP 610 ``direct_url.json``, else a source checkout."""
    try:
        dist = importlib.metadata.distribution("ohho-os")
    except importlib.metadata.PackageNotFoundError:
        root = _package_root(package_file)
        return (root / "pyproject.toml").is_file()
    raw: Optional[str]
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, UnicodeError):
        raw = None
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            info = data.get("dir_info")
            if isinstance(info, dict) and info.get("editable") is True:
                return True
    norm = _norm(package_file)
    if "site-packages" in norm or "dist-packages" in norm:
        return False
    return (_package_root(package_file) / "pyproject.toml").is_file()


def _package_root(package_file: str) -> Path:
    return Path(package_file).resolve().parent.parent


def detect_install_method() -> str:
    """How this process's ``ohho`` package was installed.

    Raises ``OSError`` when the install layout cannot be inspected.
    """
    import ohho

    package_file = str(Path(ohho.__file__).resolve())
    prefix = str(Path(sys.prefix).resolve())
    base = getattr(sys, "base_prefix", sys.prefix)
    base_prefix = str(Path(base).resolve())
    try:
        user_site: Optional[str] = site.getusersitepackages()
    except (AttributeError, OSError):
        user_site = None
    editable = _looks_editable(package_file)
    return classify_install_method(
        package_file, prefix, base_prefix, user_site, editable
    )


@dataclass
class Check:
    """One readiness check. ``extra`` is merged into the JSON object."""

    id: str
    status: str
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "detail": self.detail,
        }
        for key in sorted(self.extra):
            body[key] = self.extra[key]
        return body


@dataclass
class AgentReport:
    """The ``ohho doctor --agent`` report. ``ok`` is false when any check failed."""

    version: str
    checks: list[Check]
    secrets: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "checks": [check.as_dict() for check in self.checks],
        }


def _version_ok(version: str) -> bool:
    parts = version.split(".")
    if len(parts) < 2:
        return False
    return parts[0].isdigit() and parts[1].isdigit()


def check_cli(version: Optional[str] = None) -> Check:
    """``ohho-os`` version and a best-effort install method."""
    ver = __version__ if version is None else version
    try:
        method = detect_install_method()
    except OSError:
        return Check("cli", "skip", "could not inspect the ohho-os install")
    if method not in _INSTALL_METHODS:
        method = "unknown"
    if not ver or not _version_ok(ver):
        return Check(
            "cli",
            "fail",
            "ohho-os version is missing",
            extra={"version": ver or "", "install": method},
        )
    return Check(
        "cli",
        "pass",
        f"ohho-os {ver} ({method})",
        extra={"version": ver, "install": method},
    )


def check_python(version_info: object = None) -> Check:
    """Pass on Python 3.10 or newer. Older interpreters fail."""
    info = sys.version_info if version_info is None else version_info
    try:
        major = int(info[0])  # type: ignore[index]
        minor = int(info[1])  # type: ignore[index]
        micro = int(info[2]) if len(info) > 2 else 0  # type: ignore[arg-type]
    except (TypeError, ValueError, IndexError):
        return Check("python", "skip", "could not read the Python version")
    label = f"{major}.{minor}.{micro}"
    if (major, minor) < (3, 10):
        return Check(
            "python",
            "fail",
            f"{label} is below 3.10",
            extra={"version": label},
        )
    return Check("python", "pass", label, extra={"version": label})


def _token_of(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    return token or None


def _contains_secret(value: Optional[str], secrets: Sequence[str]) -> bool:
    if not value:
        return False
    return any(secret and secret in value for secret in secrets)


def _scrub_text(text: str, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in redacted:
            redacted = redacted.replace(secret, "")
    return redacted


def _scrub_check(check: Check, secrets: Sequence[str]) -> Check:
    detail = _scrub_text(check.detail, secrets)
    extra: dict[str, Any] = {}
    for key, value in check.extra.items():
        if isinstance(value, str) and _contains_secret(value, secrets):
            extra[key] = None
        else:
            extra[key] = value
    email = extra.get("email")
    if isinstance(email, str) and _contains_secret(email, secrets):
        extra["email"] = None
        detail = _scrub_text(detail, secrets)
    return Check(check.id, check.status, detail, extra)


def check_login(
    environ: Mapping[str, str],
    *,
    online: bool = False,
    whoami_fn: Callable[..., Any] = whoami,
) -> Check:
    """Signed-in state from ``OHHO_TOKEN`` or ``~/.ohho/credentials.json``.

    Offline (the default) does not call the network. It reports the source
    (``env`` or ``file``) and no email, because the credentials file does not
    store one. ``online=True`` calls ``whoami_fn`` (the console whoami
    endpoint, which honours ``OHHO_CONSOLE_URL``). Signed out is ``skip``:
    login is optional for simulation. The token is not a field.
    """
    secrets: list[str] = []
    source: Optional[str] = None
    env_token = _token_of(environ.get("OHHO_TOKEN"))
    if env_token is not None:
        source = "env"
        secrets.append(env_token)
    else:
        try:
            creds = load_credentials(default_credentials_path(environ))
        except AuthError:
            return Check(
                "login",
                "fail",
                "credentials unreadable",
                extra={"source": "file", "email": None},
            )
        if creds is not None:
            source = "file"
            secrets.append(creds.token)
    if source is None:
        return Check(
            "login",
            "skip",
            "not signed in (optional for sim)",
            extra={"source": None, "email": None},
        )
    if not online:
        return _scrub_check(
            Check(
                "login",
                "pass",
                f"signed in (source: {source})",
                extra={"source": source, "email": None},
            ),
            secrets,
        )
    account = whoami_fn(environ=environ)
    if getattr(account, "signed_in", False):
        email = getattr(account, "email", None)
        if not isinstance(email, str) or _contains_secret(email, secrets):
            email = None
        shown = email or "unknown"
        return _scrub_check(
            Check(
                "login",
                "pass",
                f"signed in as {shown} (source: {source})",
                extra={"source": source, "email": email},
            ),
            secrets,
        )
    error = getattr(account, "error", None)
    if error == "credentials_unreadable":
        status, detail = "fail", "credentials unreadable"
    elif error == "console_unavailable":
        status, detail = "fail", "console unavailable"
    else:
        status, detail = "skip", "not signed in (optional for sim)"
    return _scrub_check(
        Check(
            "login",
            status,
            detail,
            extra={"source": source, "email": None},
        ),
        secrets,
    )


def _rules_module() -> object:
    return importlib.import_module("ohho.agent_rules")


def check_rules(loader: Optional[Callable[[], object]] = None) -> Check:
    """Agent rules installed per client, via the phase-6 hook.

    See the module docstring. A missing hook is a skip, not a failure.
    """
    load = loader or _rules_module
    try:
        module = load()
    except ImportError:
        return Check("rules", "skip", _RULES_SKIP)
    fn = getattr(module, "installed_clients", None)
    if not callable(fn):
        return Check("rules", "skip", _RULES_SKIP)
    try:
        installed = fn()
    except Exception:
        return Check("rules", "fail", "ohho rules install check failed")
    if not isinstance(installed, dict):
        return Check(
            "rules", "fail", "ohho rules install returned an unexpected result"
        )
    clients: dict[str, Optional[str]] = {}
    missing: list[str] = []
    present: list[str] = []
    for key, value in installed.items():
        name = str(key)
        if isinstance(value, str) and value.strip():
            clients[name] = value
            present.append(name)
        elif value is None or value == "":
            clients[name] = None
            missing.append(name)
        else:
            return Check(
                "rules",
                "fail",
                "ohho rules install returned an unexpected result",
                extra={"clients": {}},
            )
    if not clients:
        return Check(
            "rules",
            "fail",
            "ohho rules install reported no clients",
            extra={"clients": {}},
        )
    extra = {"clients": clients}
    if missing:
        names = ", ".join(missing)
        return Check("rules", "fail", f"rules missing: {names}", extra=extra)
    names = ", ".join(present)
    return Check("rules", "pass", f"installed: {names}", extra=extra)


def check_mcp(
    environ: Mapping[str, str],
    *,
    platform_name: Optional[str] = None,
) -> Check:
    """Per-client ``ohho`` MCP config. Read-only; at least one client must match."""
    plat = sys.platform if platform_name is None else platform_name
    try:
        found = inspect_ohho_servers(environ=environ, platform_name=plat)
    except OSError:
        return Check(
            "mcp",
            "skip",
            "could not read MCP client config",
            extra={"clients": {}},
        )
    if found is None:
        return Check(
            "mcp",
            "skip",
            "could not resolve the home directory",
            extra={"clients": {}},
        )
    configured = [name for name, info in found.items() if info.get("configured")]
    if configured:
        detail = "configured: " + ", ".join(configured)
        status = "pass"
    else:
        detail = "no ohho MCP server configured"
        status = "fail"
    return Check("mcp", status, detail, extra={"clients": found})


def _connect_sim(robot_id: str) -> Robot:
    return Robot.connect(robot_id, transport="sim://", runtime="native")


def check_sim(
    robot_id: str = "omnibot",
    connect: Optional[Callable[[str], Robot]] = None,
) -> Check:
    """Connect ``robot_id`` on ``sim://`` and read telemetry once. No motion."""
    opener = connect or _connect_sim
    bot: Optional[Robot] = None
    try:
        try:
            bot = opener(robot_id)
        except UnknownRobot:
            return Check(
                "sim",
                "fail",
                f"unknown robot {robot_id}",
                extra={"robot": robot_id},
            )
        except Exception:
            return Check(
                "sim",
                "fail",
                "sim:// connect failed",
                extra={"robot": robot_id},
            )
        try:
            state = bot.status().state
        except Exception:
            return Check(
                "sim",
                "fail",
                "sim:// telemetry failed",
                extra={"robot": robot_id},
            )
        if state != ConnectionState.CONNECTED:
            return Check(
                "sim",
                "skip",
                "sim:// is not connected",
                extra={"robot": robot_id},
            )
        try:
            telemetry = bot.telemetry()
        except Exception:
            return Check(
                "sim",
                "fail",
                "sim:// telemetry failed",
                extra={"robot": robot_id},
            )
        if telemetry is None or telemetry.odom is None:
            return Check(
                "sim",
                "fail",
                "sim:// telemetry missing",
                extra={"robot": robot_id},
            )
        return Check(
            "sim",
            "pass",
            "sim:// telemetry ok",
            extra={"robot": robot_id},
        )
    finally:
        if bot is not None:
            try:
                bot.disconnect()
            except Exception:
                pass


def _collect_secrets(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Token strings to strip from the rendered report. Not themselves rendered."""
    found: list[str] = []
    env_token = _token_of(environ.get("OHHO_TOKEN"))
    if env_token is not None and len(env_token) >= 8:
        found.append(env_token)
    try:
        creds = load_credentials(default_credentials_path(environ))
    except AuthError:
        creds = None
    if creds is not None and len(creds.token) >= 8 and creds.token not in found:
        found.append(creds.token)
    return tuple(found)


def build_agent_report(
    *,
    online: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    version: Optional[str] = None,
    version_info: object = None,
    whoami_fn: Callable[..., Any] = whoami,
    rules_loader: Optional[Callable[[], object]] = None,
    sim_connect: Optional[Callable[[str], Robot]] = None,
) -> AgentReport:
    """Run every agent check. Does not print. Does not call the network unless
    ``online`` is true.
    """
    env = os.environ if environ is None else environ
    secrets = _collect_secrets(env)
    checks = [
        check_cli(version),
        check_python(version_info),
        check_login(env, online=online, whoami_fn=whoami_fn),
        check_rules(rules_loader),
        check_mcp(env, platform_name=platform_name),
        check_sim(connect=sim_connect),
    ]
    # A last pass so a token cannot survive inside any detail string.
    scrubbed = [_scrub_check(check, secrets) for check in checks]
    return AgentReport(version=__version__, checks=scrubbed, secrets=secrets)


def format_agent_report(report: AgentReport, *, as_json: bool) -> str:
    """Text table, or one JSON object. The token is not included."""
    if as_json:
        text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    else:
        lines = [f"OhhO OS {report.version} agent readiness"]
        for check in report.checks:
            lines.append(f"  {check.id:<7} {check.status:<4}  {check.detail}")
        text = "\n".join(lines) + "\n"
    return _scrub_text(text, report.secrets)


def print_agent_report(
    *,
    as_json: bool = False,
    online: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    stdout: Optional[TextIO] = None,
    version: Optional[str] = None,
    version_info: object = None,
    whoami_fn: Callable[..., Any] = whoami,
    rules_loader: Optional[Callable[[], object]] = None,
    sim_connect: Optional[Callable[[str], Robot]] = None,
) -> int:
    """Print the report and return 0 when no check failed, else 1."""
    report = build_agent_report(
        online=online,
        environ=environ,
        platform_name=platform_name,
        version=version,
        version_info=version_info,
        whoami_fn=whoami_fn,
        rules_loader=rules_loader,
        sim_connect=sim_connect,
    )
    stream = stdout if stdout is not None else sys.stdout
    stream.write(format_agent_report(report, as_json=as_json))
    return 0 if report.ok else 1
