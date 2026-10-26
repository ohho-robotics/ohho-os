"""Account sign-in for the ``ohho`` CLI (OHH-273).

The console (OHH-272) speaks RFC 8628:

* ``POST /api/cli/device/code``
* ``POST /api/cli/device/token``
* ``GET /api/cli/whoami`` with ``Authorization: Bearer``

There is no CLI revoke endpoint. ``logout`` deletes the local file only.
Tokens are revoked from the console account page.

The access token and device code are never written to stdout, stderr, or
logs. ``OHHO_TOKEN``, when set, overrides the credentials file and is never
written to disk.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

DEFAULT_CONSOLE_URL = "https://console.ohho-robotics.com"
ENV_CONSOLE_URL = "OHHO_CONSOLE_URL"
ENV_TOKEN = "OHHO_TOKEN"
CREDENTIALS_VERSION = 1
DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
CLIENT_NAME = "OhhO CLI"
DEFAULT_SCOPE = "mcp"

_CODE_PATH = "/api/cli/device/code"
_TOKEN_PATH = "/api/cli/device/token"
_WHOAMI_PATH = "/api/cli/whoami"

_DEFAULT_INTERVAL_S = 5.0
_MAX_INTERVAL_S = 60.0
_DEFAULT_EXPIRES_S = 600.0
_MAX_EXPIRES_S = 3600.0
_MAX_POLLS = 10_000
_REQUEST_TIMEOUT_S = 30.0

_SIGNED_OUT_ERRORS = frozenset({"not_signed_in", "invalid_token"})


class AuthError(Exception):
    """Sign-in failed. The message is safe to print: it has no token."""


@dataclass
class Credentials:
    """On-disk account. ``token`` is a secret and is omitted from ``repr``."""

    token: str
    token_type: str
    scope: str
    console_url: str

    def __repr__(self) -> str:
        return (
            "Credentials(token='[redacted]', "
            f"token_type={self.token_type!r}, scope={self.scope!r}, "
            f"console_url={self.console_url!r})"
        )


@dataclass(frozen=True)
class Whoami:
    """Account description returned to the CLI. Never includes a token."""

    signed_in: bool
    email: Optional[str] = None
    scopes: tuple[str, ...] = ()
    user_id: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None

    def public_dict(self) -> dict[str, object]:
        """JSON object for ``ohho whoami --json``.

        Signed in, the keys are ``signed_in``, ``email``, ``scopes``,
        ``user_id``. Otherwise ``error`` is added. ``error`` is
        ``not_signed_in``, ``invalid_token``, ``credentials_unreadable``,
        or ``console_unavailable``.
        """
        body: dict[str, object] = {
            "signed_in": self.signed_in,
            "email": self.email,
            "scopes": list(self.scopes),
            "user_id": self.user_id,
        }
        if not self.signed_in:
            body["error"] = self.error or "not_signed_in"
        return body


@dataclass(frozen=True)
class LogoutResult:
    """What ``logout`` did. ``env_token`` means ``OHHO_TOKEN`` is still set."""

    removed: bool
    env_token: bool


def default_credentials_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """``~/.ohho/credentials.json`` using ``HOME`` / ``USERPROFILE``.

    This does not call :meth:`pathlib.Path.home`, so tests can point the file
    at a temporary directory.
    """
    return _home(_env(environ)) / ".ohho" / "credentials.json"


def console_origin(environ: Optional[Mapping[str, str]] = None) -> str:
    """API origin. ``OHHO_CONSOLE_URL`` when set, otherwise production."""
    return _explicit_origin(_env(environ)) or DEFAULT_CONSOLE_URL


def load_credentials(path: Path) -> Optional[Credentials]:
    """Load credentials, or ``None`` when the file is missing or has no token.

    A file that cannot be parsed raises :class:`AuthError`. The message does
    not include the file body.
    """
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuthError("Could not read credentials.") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise AuthError("Could not read credentials.") from None
    if not isinstance(data, dict):
        raise AuthError("Could not read credentials.")
    token = data.get("token")
    if not isinstance(token, str) or not token.strip():
        return None
    token_type = data.get("token_type")
    if not isinstance(token_type, str) or not token_type.strip():
        token_type = "Bearer"
    scope = data.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        scope = DEFAULT_SCOPE
    stored = data.get("console_url")
    if isinstance(stored, str) and stored.strip():
        try:
            origin = _parse_origin(stored.strip())
        except AuthError:
            raise AuthError("Credentials file has an invalid console_url.") from None
    else:
        origin = DEFAULT_CONSOLE_URL
    return Credentials(
        token=token.strip(),
        token_type=token_type.strip(),
        scope=scope.strip(),
        console_url=origin,
    )


def save_credentials(path: Path, credentials: Credentials) -> None:
    """Atomically write credentials so only the current user can read them.

    POSIX mode is ``0600``. On Windows the DACL grants the current user
    full control and nobody else.
    """
    payload = {
        "version": CREDENTIALS_VERSION,
        "token": credentials.token,
        "token_type": credentials.token_type,
        "scope": credentials.scope,
        "console_url": credentials.console_url,
    }
    try:
        _write_private(path, json.dumps(payload, indent=2) + "\n")
    except AuthError:
        raise
    except OSError:
        raise AuthError("Could not store credentials.") from None


def login(
    *,
    no_browser: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    credentials_file: Optional[Path] = None,
    sleep: Optional[Callable[[float], None]] = None,
    now: Optional[Callable[[], float]] = None,
    open_browser: Optional[Callable[[str], bool]] = None,
    out: Optional[TextIO] = None,
) -> None:
    """Run the device flow and store the access token.

    Raises :class:`AuthError` on denial, expiry, or an unreachable console.
    The new token is the only secret written. ``OHHO_TOKEN`` is left untouched.
    """
    env = _env(environ)
    stream = out if out is not None else sys.stdout
    sleeper = sleep if sleep is not None else time.sleep
    clock = now if now is not None else time.monotonic
    browse = open_browser if open_browser is not None else _default_open_browser
    origin = console_origin(env)
    path = credentials_file or default_credentials_path(env)

    issued = _request_json("POST", origin + _CODE_PATH, payload=_code_body())
    payload = _object_body(issued)
    device_code = _required_text(payload, "device_code")
    user_code = _required_text(payload, "user_code")
    url = _verification_url(payload, device_code)
    interval = _bounded_interval(payload.get("interval"), default=_DEFAULT_INTERVAL_S)
    expires_in = _bounded_expires(payload.get("expires_in"))
    deadline = clock() + expires_in

    _write_line(stream, f"Code: {user_code}")
    _write_line(stream, f"URL: {url}")
    if _host_of(url) != _host_of(origin):
        _write_line(
            stream,
            "If that page does not accept the code, open "
            f"{origin}/device and enter it there.",
        )
    if no_browser:
        _write_line(stream, "Browser not opened (--no-browser).")
    else:
        opened = False
        try:
            opened = bool(browse(url))
        except Exception:
            opened = False
        if not opened:
            _write_line(
                stream, "No browser opened. Open the URL above to finish sign-in."
            )
    _write_line(stream, "Waiting for approval...")
    stream.flush()

    token, token_type, scope = _poll_token(
        origin,
        device_code,
        interval=interval,
        deadline=deadline,
        sleep=sleeper,
        now=clock,
    )
    save_credentials(
        path,
        Credentials(
            token=token,
            token_type=token_type,
            scope=scope,
            console_url=origin,
        ),
    )
    account = _fetch_whoami(token, origin, secrets=(token, device_code))
    if account.signed_in and account.email:
        _write_line(stream, f"Signed in as {account.email}.")
    else:
        _write_line(stream, "Signed in.")
    if _env_token(env) is not None:
        _write_line(
            stream,
            "OHHO_TOKEN is still set and takes precedence over the saved "
            "credentials. It was not written to disk.",
        )


def whoami(
    *,
    environ: Optional[Mapping[str, str]] = None,
    credentials_file: Optional[Path] = None,
) -> Whoami:
    """Resolve ``OHHO_TOKEN`` or the credentials file and ask the console.

    ``OHHO_TOKEN`` wins. This function does not create or rewrite the file.
    """
    env = _env(environ)
    try:
        origin_override = _explicit_origin(env)
    except AuthError as exc:
        return Whoami(signed_in=False, error="console_unavailable", detail=str(exc))

    env_token = _env_token(env)
    if env_token is not None:
        origin = origin_override or DEFAULT_CONSOLE_URL
        return _fetch_whoami(env_token, origin, secrets=(env_token,))

    try:
        path = (
            credentials_file
            if credentials_file is not None
            else default_credentials_path(env)
        )
        creds = load_credentials(path)
    except AuthError as exc:
        return Whoami(signed_in=False, error="credentials_unreadable", detail=str(exc))
    if creds is None:
        return Whoami(signed_in=False, error="not_signed_in")
    origin = origin_override or creds.console_url
    return _fetch_whoami(creds.token, origin, secrets=(creds.token,))


def format_whoami(account: Whoami, *, as_json: bool) -> str:
    """Render :class:`Whoami` for stdout.

    Text for a missing or rejected token is exactly ``not signed in``.
    Network and credentials errors return an empty string; the CLI prints
    ``account.detail`` on stderr. JSON always includes the public object.
    """
    if as_json:
        return json.dumps(account.public_dict(), ensure_ascii=False) + "\n"
    if account.signed_in:
        email = account.email or "unknown"
        scopes = " ".join(account.scopes)
        return f"email: {email}\nscopes: {scopes}\n"
    if account.error in _SIGNED_OUT_ERRORS or account.error is None:
        return "not signed in\n"
    return ""


def logout(
    *,
    environ: Optional[Mapping[str, str]] = None,
    credentials_file: Optional[Path] = None,
) -> LogoutResult:
    """Delete local credentials.

    OHH-272 does not expose a CLI revoke route. The account page sets
    ``cli_tokens.revoked_at`` for the browser session. This function does
    not contact the console. ``OHHO_TOKEN`` is not cleared and is not written
    to disk.
    """
    env = _env(environ)
    path = credentials_file or default_credentials_path(env)
    removed = False
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            raise AuthError("Could not delete credentials.") from None
        removed = True
    return LogoutResult(removed=removed, env_token=_env_token(env) is not None)


def _poll_token(
    origin: str,
    device_code: str,
    *,
    interval: float,
    deadline: float,
    sleep: Callable[[float], None],
    now: Callable[[], float],
) -> tuple[str, str, str]:
    secrets = [device_code]
    first = True
    polls = 0
    while True:
        polls += 1
        if polls > _MAX_POLLS:
            raise AuthError(
                "Sign-in polling stopped because the console never finished."
            )
        if now() >= deadline:
            raise AuthError("The sign-in code expired before it was approved.")
        if not first:
            remaining = deadline - now()
            step = min(max(interval, 0.0), remaining)
            sleep(step)
            if now() >= deadline:
                raise AuthError("The sign-in code expired before it was approved.")
        first = False
        status, headers, body = _request_json(
            "POST",
            origin + _TOKEN_PATH,
            payload={"grant_type": DEVICE_GRANT_TYPE, "device_code": device_code},
        )
        payload = body if isinstance(body, dict) else {}
        if status == 200:
            token = payload.get("access_token")
            if isinstance(token, str) and token.strip():
                token_type = payload.get("token_type")
                if not isinstance(token_type, str) or not token_type.strip():
                    token_type = "Bearer"
                scope = payload.get("scope")
                if not isinstance(scope, str) or not scope.strip():
                    scope = DEFAULT_SCOPE
                return token.strip(), token_type.strip(), scope.strip()
            raise AuthError("The console approved sign-in but did not return a token.")
        error = payload.get("error") if isinstance(payload, dict) else None
        updated = _interval_from(payload, headers)
        if error == "authorization_pending":
            if updated is not None:
                interval = updated
            continue
        if error == "slow_down" or status == 429:
            interval = (
                updated if updated is not None else min(_MAX_INTERVAL_S, interval + 5.0)
            )
            continue
        if error == "expired_token":
            raise AuthError("The sign-in code expired.")
        if error == "access_denied":
            raise AuthError("Sign-in was denied.")
        detail = _safe_description(payload, secrets)
        code = _safe_error_code(error)
        if detail and code:
            raise AuthError(f"Sign-in failed ({code}). {detail}")
        if detail:
            raise AuthError(f"Sign-in failed. {detail}")
        if code:
            raise AuthError(f"Sign-in failed ({code}).")
        raise AuthError(f"Sign-in failed (HTTP {status}).")


def _fetch_whoami(token: str, origin: str, *, secrets: tuple[str, ...]) -> Whoami:
    try:
        status, _headers, body = _request_json(
            "GET",
            origin + _WHOAMI_PATH,
            headers={"Authorization": f"Bearer {token}"},
        )
    except AuthError as exc:
        return Whoami(signed_in=False, error="console_unavailable", detail=str(exc))
    if status == 401:
        return Whoami(signed_in=False, error="invalid_token")
    if status != 200 or not isinstance(body, dict):
        return Whoami(
            signed_in=False,
            error="console_unavailable",
            detail=f"The console returned HTTP {status}.",
        )
    email = _clean(_optional_text(body.get("email")), secrets)
    user_id = _clean(_optional_text(body.get("user_id")), secrets)
    scopes = tuple(
        scope
        for scope in _scope_list(body.get("scopes"))
        if _clean(scope, secrets) == scope
    )
    return Whoami(signed_in=True, email=email, scopes=scopes, user_id=user_id)


def _code_body() -> dict[str, str]:
    return {"client_name": CLIENT_NAME, "scope": DEFAULT_SCOPE}


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> tuple[int, dict[str, str], object]:
    hdrs = {"Accept": "application/json", "User-Agent": _user_agent()}
    data: Optional[bytes] = None
    if payload is not None:
        data = json.dumps(dict(payload)).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)
    request = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with _OPENER.open(request, timeout=_REQUEST_TIMEOUT_S) as response:
            status = int(getattr(response, "status", response.getcode()))
            raw = response.read()
            header_map = {key.lower(): value for key, value in response.headers.items()}
    except AuthError:
        raise
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
        header_map = (
            {key.lower(): value for key, value in exc.headers.items()}
            if exc.headers
            else {}
        )
    except urllib.error.URLError:
        raise AuthError(f"Could not reach the console at {_origin_of(url)}.") from None
    if not raw:
        return status, header_map, None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise AuthError(
            f"The console returned HTTP {status} with a body that was not JSON."
        ) from None
    return status, header_map, parsed


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect. A follow would resend ``Authorization``."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        raise AuthError(f"The console redirected sign-in (HTTP {code}).")


def _build_opener() -> urllib.request.OpenerDirector:
    # Honour the process proxy env (corporate networks). Redirects are refused
    # so a proxy or the console cannot bounce the bearer token elsewhere.
    return urllib.request.build_opener(_RefuseRedirect())


_OPENER = _build_opener()


def _object_body(result: tuple[int, dict[str, str], object]) -> dict[str, object]:
    status, _headers, body = result
    if status == 200 and isinstance(body, dict):
        return body
    detail = _safe_description(body, [])
    if detail:
        raise AuthError(f"Could not start sign-in (HTTP {status}). {detail}")
    raise AuthError(f"Could not start sign-in (HTTP {status}).")


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise AuthError("The console returned an incomplete device code.")


def _verification_url(payload: Mapping[str, object], device_code: str) -> str:
    complete = payload.get("verification_uri_complete")
    uri = payload.get("verification_uri")
    user_code = payload.get("user_code")
    if isinstance(complete, str) and complete.strip() and device_code not in complete:
        return complete.strip()
    if isinstance(uri, str) and uri.strip() and isinstance(user_code, str):
        base = uri.strip()
        if device_code not in base:
            join = "&" if "?" in base else "?"
            return f"{base}{join}user_code={urllib.parse.quote(user_code.strip())}"
    raise AuthError(
        "The console returned a verification URL that is not safe to print."
    )


def _bounded_expires(value: object) -> float:
    parsed = _as_float(value)
    if parsed is None:
        return _DEFAULT_EXPIRES_S
    if parsed < 0:
        return 0.0
    return min(parsed, _MAX_EXPIRES_S)


def _bounded_interval(value: object, *, default: Optional[float]) -> float:
    parsed = _as_float(value)
    if parsed is None:
        if default is None:
            raise AuthError("The console returned an invalid poll interval.")
        parsed = default
    if parsed < 0:
        parsed = 0.0
    return min(parsed, _MAX_INTERVAL_S)


def _interval_from(
    payload: Mapping[str, object], headers: Mapping[str, str]
) -> Optional[float]:
    for value in (payload.get("interval"), headers.get("retry-after")):
        parsed = _as_float(value)
        if parsed is None:
            continue
        if parsed < 0:
            return 0.0
        return min(parsed, _MAX_INTERVAL_S)
    return None


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_error_code(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if value and len(value) <= 64 and all(ch.isalnum() or ch == "_" for ch in value):
        return value
    return None


def _safe_description(payload: object, secrets: list[str]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    text = payload.get("error_description")
    if not isinstance(text, str):
        return None
    cleaned = _redact(text.strip(), secrets)
    if not cleaned or len(cleaned) > 300:
        return None
    if "ohho_" in cleaned or "Bearer " in cleaned:
        return None
    return cleaned


def _redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _clean(value: Optional[str], secrets: tuple[str, ...]) -> Optional[str]:
    if value is None:
        return None
    for secret in secrets:
        if secret and secret in value:
            return None
    return value


def _optional_text(value: object) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _scope_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part for part in value.split() if part)
    if isinstance(value, list):
        return tuple(
            str(part) for part in value if isinstance(part, str) and part.strip()
        )
    return ()


def _env(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _env_token(environ: Mapping[str, str]) -> Optional[str]:
    raw = environ.get(ENV_TOKEN)
    if raw is None:
        return None
    token = raw.strip()
    return token or None


def _explicit_origin(environ: Mapping[str, str]) -> Optional[str]:
    raw = (environ.get(ENV_CONSOLE_URL) or "").strip()
    if not raw:
        return None
    return _parse_origin(raw)


def _parse_origin(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AuthError("OHHO_CONSOLE_URL must be an http or https origin.")
    if parsed.username or parsed.password:
        raise AuthError("OHHO_CONSOLE_URL must not include credentials.")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise AuthError("OHHO_CONSOLE_URL must be an origin with no path.")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


def _origin_of(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_CONSOLE_URL
    return f"{parsed.scheme}://{parsed.netloc}"


def _host_of(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def _home(environ: Mapping[str, str]) -> Path:
    if sys.platform == "win32":
        raw = environ.get("USERPROFILE") or environ.get("HOME")
    else:
        raw = environ.get("HOME") or environ.get("USERPROFILE")
    if not raw:
        raise AuthError("Could not find a home directory (set HOME or USERPROFILE).")
    return Path(raw)


def _user_agent() -> str:
    from . import __version__

    return f"ohho-os/{__version__}"


def _default_open_browser(url: str) -> bool:
    import webbrowser

    return bool(webbrowser.open(url, new=1, autoraise=True))


def _write_line(stream: TextIO, text: str) -> None:
    stream.write(text + "\n")


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` only after the file is readable by the current user alone."""
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = -1
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(tmp), flags, 0o600)
        if os.name != "nt" and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        os.close(fd)
        fd = -1
        _restrict_private(tmp)
        fd = os.open(str(tmp), os.O_WRONLY | os.O_TRUNC)
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(str(tmp), str(path))
        _restrict_private(path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


def _restrict_private(path: Path) -> None:
    try:
        if os.name == "nt":
            _restrict_windows(path)
            if not _windows_is_user_private(path):
                raise AuthError("Could not restrict credentials to the current user.")
            return
        os.chmod(path, 0o600)
        if path.stat().st_mode & 0o777 != 0o600:
            raise AuthError("Could not set credentials file mode to 0600.")
    except AuthError:
        path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        path.unlink(missing_ok=True)
        code = getattr(exc, "winerror", None) or exc.errno
        raise AuthError(
            f"Could not restrict credentials to the current user ({code})."
        ) from None


def _restrict_windows(path: Path) -> None:
    api = _winapi()
    sid = api.current_user_sid()
    entry = _EXPLICIT_ACCESS_W()
    entry.grfAccessPermissions = _FILE_ALL_ACCESS
    entry.grfAccessMode = _SET_ACCESS
    entry.grfInheritance = 0
    entry.Trustee.TrusteeForm = _TRUSTEE_IS_SID
    entry.Trustee.TrusteeType = _TRUSTEE_IS_USER
    entry.Trustee.ptstrName = ctypes.cast(sid, ctypes.c_void_p)
    acl = ctypes.c_void_p()
    error = api.set_entries(1, ctypes.byref(entry), acl)
    if error:
        raise OSError(error, "SetEntriesInAclW failed")
    try:
        error = api.set_named(
            str(path),
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            acl,
        )
        if error:
            raise OSError(error, "SetNamedSecurityInfoW failed")
    finally:
        api.local_free(acl)


def _windows_is_user_private(path: Path) -> bool:
    api = _winapi()
    sid = api.current_user_sid()
    described = api.dacl(str(path))
    if described is None:
        return False
    security, dacl = described
    try:
        if not dacl:
            return False
        info = _ACL_SIZE_INFORMATION()
        if not api.acl_info(dacl, info):
            return False
        if info.AceCount != 1:
            return False
        ace = ctypes.c_void_p()
        if not api.get_ace(dacl, 0, ctypes.byref(ace)) or not ace.value:
            return False
        header = ctypes.cast(ace, ctypes.POINTER(_ACCESS_ALLOWED_ACE)).contents
        if header.AceType != _ACCESS_ALLOWED_ACE_TYPE:
            return False
        if header.AceFlags & _INHERITED_ACE:
            return False
        ace_sid = ctypes.c_void_p(ace.value + _ACCESS_ALLOWED_ACE.SidStart.offset)
        return bool(api.equal_sid(ace_sid, ctypes.cast(sid, ctypes.c_void_p)))
    finally:
        api.local_free(security)


# Windows ACL types are defined on every platform. They are only used when
# ``os.name == "nt"``.
_FILE_ALL_ACCESS = 0x1F01FF
_SET_ACCESS = 2
_TRUSTEE_IS_SID = 0
_TRUSTEE_IS_USER = 1
_SE_FILE_OBJECT = 1
_DACL_SECURITY_INFORMATION = 0x4
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_ACCESS_ALLOWED_ACE_TYPE = 0
_INHERITED_ACE = 0x10
_ACL_SIZE_INFORMATION_CLASS = 2


class _TRUSTEE_W(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),
    ]


class _EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _TRUSTEE_W),
    ]


class _ACCESS_ALLOWED_ACE(ctypes.Structure):
    _fields_ = [
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", wintypes.WORD),
        ("Mask", wintypes.DWORD),
        ("SidStart", wintypes.DWORD),
    ]


class _ACL_SIZE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _WinApi:
    def __init__(self) -> None:
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel = self.kernel32
        adv = self.advapi32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        adv.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        adv.OpenProcessToken.restype = wintypes.BOOL
        adv.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        adv.GetTokenInformation.restype = wintypes.BOOL
        adv.GetLengthSid.argtypes = [ctypes.c_void_p]
        adv.GetLengthSid.restype = wintypes.DWORD
        adv.CopySid.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
        adv.CopySid.restype = wintypes.BOOL
        adv.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        adv.EqualSid.restype = wintypes.BOOL
        adv.SetEntriesInAclW.argtypes = [
            wintypes.ULONG,
            ctypes.POINTER(_EXPLICIT_ACCESS_W),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        adv.SetEntriesInAclW.restype = wintypes.DWORD
        adv.SetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        adv.SetNamedSecurityInfoW.restype = wintypes.DWORD
        adv.GetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        adv.GetNamedSecurityInfoW.restype = wintypes.DWORD
        adv.GetAclInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        adv.GetAclInformation.restype = wintypes.BOOL
        adv.GetAce.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        adv.GetAce.restype = wintypes.BOOL

    def current_user_sid(self) -> ctypes.Array:
        token = wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            self.kernel32.GetCurrentProcess(),
            _TOKEN_QUERY,
            ctypes.byref(token),
        ):
            raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
        try:
            needed = wintypes.DWORD(0)
            self.advapi32.GetTokenInformation(
                token, _TOKEN_USER, None, 0, ctypes.byref(needed)
            )
            buffer = ctypes.create_string_buffer(needed.value)
            if not self.advapi32.GetTokenInformation(
                token, _TOKEN_USER, buffer, needed, ctypes.byref(needed)
            ):
                raise OSError(ctypes.get_last_error(), "GetTokenInformation failed")
            sid_ptr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            length = self.advapi32.GetLengthSid(sid_ptr)
            copy = ctypes.create_string_buffer(length)
            if not self.advapi32.CopySid(length, copy, sid_ptr):
                raise OSError(ctypes.get_last_error(), "CopySid failed")
            return copy
        finally:
            self.kernel32.CloseHandle(token)

    def set_entries(self, count: int, entry: object, acl: ctypes.c_void_p) -> int:
        return int(
            self.advapi32.SetEntriesInAclW(count, entry, None, ctypes.byref(acl))
        )

    def set_named(self, path: str, kind: int, info: int, acl: ctypes.c_void_p) -> int:
        return int(
            self.advapi32.SetNamedSecurityInfoW(path, kind, info, None, None, acl, None)
        )

    def dacl(self, path: str) -> Optional[tuple[ctypes.c_void_p, ctypes.c_void_p]]:
        security = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        error = self.advapi32.GetNamedSecurityInfoW(
            path,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(security),
        )
        if error:
            return None
        return security, dacl

    def acl_info(self, dacl: ctypes.c_void_p, info: _ACL_SIZE_INFORMATION) -> bool:
        return bool(
            self.advapi32.GetAclInformation(
                dacl,
                ctypes.byref(info),
                ctypes.sizeof(info),
                _ACL_SIZE_INFORMATION_CLASS,
            )
        )

    def get_ace(self, dacl: ctypes.c_void_p, index: int, ace: object) -> bool:
        return bool(self.advapi32.GetAce(dacl, index, ace))

    def equal_sid(self, left: ctypes.c_void_p, right: ctypes.c_void_p) -> bool:
        return bool(self.advapi32.EqualSid(left, right))

    def local_free(self, pointer: ctypes.c_void_p) -> None:
        if pointer.value:
            self.kernel32.LocalFree(pointer)


_WIN: Optional[_WinApi] = None


def _winapi() -> _WinApi:
    global _WIN
    if os.name != "nt":
        raise AuthError("Windows credentials ACL is only available on Windows.")
    if _WIN is None:
        _WIN = _WinApi()
    return _WIN
