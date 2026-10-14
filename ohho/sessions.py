"""Session registry for the VR garage (OHH-206).

``ohho sim up --target local`` and ``ohho robot announce`` upsert a row in
the Supabase ``robot_sessions`` table and heartbeat it every 10 seconds.
The headset lists ``robot_sessions_visible``, which hides rows whose
``expires_at`` has passed. The OHH-196 gateway token is stored only as a
SHA-256 hex digest.

The HTTP client is stdlib ``urllib``. Tests inject a fake transport or a
fake client so CI does not open a socket. ``--target cloud`` refuses with
a message that names OHH-216 and does not provision a VM.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from .registry import UnknownRobot, get_spec

HEARTBEAT_INTERVAL_S = 10.0
STALE_AFTER_S = 30.0
MIN_TOKEN_LENGTH = 16
MAX_TOKEN_LENGTH = 512
HTTP_TIMEOUT_S = 15.0

STATUS_ACTIVE = "active"
KIND_SIM = "sim"
KIND_REAL = "real"
TARGET_LOCAL = "local"
TARGET_CLOUD = "cloud"
TARGET_ROBOT = "robot"
KINDS = frozenset({KIND_SIM, KIND_REAL})
TARGETS = frozenset({TARGET_LOCAL, TARGET_CLOUD, TARGET_ROBOT})

CLOUD_TARGET_MESSAGE = (
    "ohho sim up --target cloud needs OHH-216. No cloud VM was created."
)
OHH204_COMPOSE_HINT = (
    "docker compose --profile gpu -f ohho/sim/docker-compose.yml up -d"
)
ON_CONFLICT = "user_id,robot_id,kind,target"
FRONT_CAMERA_TOPIC = "/camera/front/image_raw"

SESSION_COLUMNS = (
    "id",
    "user_id",
    "robot_id",
    "kind",
    "target",
    "rosbridge_url",
    "video_urls",
    "webrtc_signal_url",
    "token_hash",
    "status",
    "heartbeat_at",
    "expires_at",
    "created_at",
    "updated_at",
)
_TOKEN_KEYS = frozenset({"token", "gateway_token", "pairing_code", "access_token"})
_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_ENDPOINT_SCHEMES = frozenset({"ws", "wss", "http", "https"})
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

SUPABASE_ENV_KEYS = (
    "OHHO_SUPABASE_URL",
    "SUPABASE_URL",
    "OHHO_SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OHHO_SUPABASE_ANON_KEY",
    "SUPABASE_ANON_KEY",
    "OHHO_SUPABASE_JWT",
    "SUPABASE_ACCESS_TOKEN",
    "OHHO_USER_ID",
)

Http = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]
Clock = Callable[[], datetime]


class SessionError(RuntimeError):
    """The session registry could not complete a request."""


class SessionConfigError(SessionError):
    """Environment or arguments are not enough to register a session."""


class SessionGone(SessionError):
    """A heartbeat targeted a row that is no longer in the table."""


class CloudTargetUnavailable(SessionConfigError):
    """``--target cloud`` is reserved for OHH-216."""


@dataclass(frozen=True)
class SupabaseAuth:
    """Credentials for one PostgREST call. The raw gateway token is not here."""

    url: str
    api_key: str
    bearer: str
    user_id: str


@dataclass(frozen=True)
class ContainerLaunch:
    """Result of a docker compose attempt. ``command`` is argv, not a shell line."""

    started: bool
    command: list[str]
    detail: str


class SessionClient(Protocol):
    """Upsert, heartbeat, delete, and list. The fake test client matches this."""

    user_id: str

    def upsert(self, row: dict[str, Any]) -> dict[str, Any]: ...

    def heartbeat(self, session_id: str, fields: dict[str, str]) -> dict[str, Any]: ...

    def delete_slot(self, robot_id: str, kind: str, target: str) -> None: ...

    def list_sessions(self, robot_id: str | None = None) -> list[dict[str, Any]]: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_iso(moment: datetime) -> str:
    """ISO-8601 with a numeric offset. Python 3.10 ``fromisoformat`` rejects ``Z``."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def require_user_id(user_id: str) -> str:
    if not _UUID.fullmatch(user_id or ""):
        raise SessionConfigError("OHHO_USER_ID must be the auth.users UUID")
    return user_id.lower()


def validate_gateway_token(token: str) -> str:
    """Same length and whitespace rules as the OHH-196 gateway."""
    cleaned = token.strip()
    if len(cleaned) < MIN_TOKEN_LENGTH or len(cleaned) > MAX_TOKEN_LENGTH:
        raise SessionConfigError(
            f"gateway token must be {MIN_TOKEN_LENGTH} to {MAX_TOKEN_LENGTH} characters"
        )
    if any(character.isspace() for character in cleaned):
        raise SessionConfigError("gateway token must not contain whitespace")
    return cleaned


def hash_gateway_token(token: str) -> str:
    """SHA-256 hex of the stripped token. This is the only form that is stored."""
    cleaned = validate_gateway_token(token)
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def user_id_from_jwt(jwt: str) -> str | None:
    """Read the ``sub`` claim. Supabase verifies the signature; this only routes the row."""
    parts = jwt.split(".")
    if len(parts) < 2 or not parts[1]:
        return None
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    subject = payload.get("sub") if isinstance(payload, dict) else None
    if not isinstance(subject, str):
        return None
    try:
        return require_user_id(subject)
    except SessionConfigError:
        return None


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _first(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = env.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return ""


def _read_token_file(path: Path) -> str:
    try:
        if path.stat().st_size > 4096:
            raise SessionConfigError(f"token file is too large: {path}")
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SessionConfigError(f"cannot read token file: {path}") from exc
    return validate_gateway_token(raw)


def _write_token_file(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def default_pairing_file() -> Path:
    return Path.home() / ".ohho" / "gateway" / "pairing-code"


def default_state_dir() -> Path:
    override = os.environ.get("OHHO_STATE_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ohho" / "sessions"


def default_compose_file() -> Path:
    return Path(__file__).resolve().parent / "sim" / "docker-compose.yml"


def resolve_gateway_token(
    *,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    generate: bool = True,
) -> tuple[str, str]:
    """Return ``(token, source)`` where source is env, file, pairing, or generated.

    Order matches the OHH-196 gateway: ``OHHO_GATEWAY_TOKEN``, then
    ``OHHO_GATEWAY_TOKEN_FILE``, then ``~/.ohho/gateway/pairing-code``.
    A missing pairing file is created so the gateway reuses the same code.
    """
    values = _env(env)
    env_token = _first(values, ("OHHO_GATEWAY_TOKEN",))
    if env_token:
        return validate_gateway_token(env_token), "env"
    token_file = _first(values, ("OHHO_GATEWAY_TOKEN_FILE",))
    if token_file:
        return _read_token_file(Path(token_file)), "file"
    path = pairing_file
    if path is None:
        custom = _first(values, ("OHHO_GATEWAY_PAIRING_FILE",))
        path = Path(custom) if custom else default_pairing_file()
    if path.is_file():
        return _read_token_file(path), "pairing"
    if not generate:
        raise SessionConfigError(
            "set OHHO_GATEWAY_TOKEN, or run ohho-gateway once so "
            "~/.ohho/gateway/pairing-code exists"
        )
    token = secrets.token_urlsafe(18)
    _write_token_file(path, token)
    return token, "generated"


def _origin(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"https", "http"} or not parts.netloc:
        raise SessionConfigError("OHHO_SUPABASE_URL must be an http(s) origin")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def load_supabase_auth(env: Mapping[str, str] | None = None) -> SupabaseAuth:
    """Service role, or anon key plus a user JWT. Messages never include the secrets."""
    values = _env(env)
    raw_url = _first(values, ("OHHO_SUPABASE_URL", "SUPABASE_URL"))
    if not raw_url:
        raise SessionConfigError("set OHHO_SUPABASE_URL (or SUPABASE_URL)")
    origin = _origin(raw_url)
    service = _first(
        values, ("OHHO_SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    )
    anon = _first(values, ("OHHO_SUPABASE_ANON_KEY", "SUPABASE_ANON_KEY"))
    jwt = _first(values, ("OHHO_SUPABASE_JWT", "SUPABASE_ACCESS_TOKEN"))
    requested = _first(values, ("OHHO_USER_ID",))
    jwt_user = user_id_from_jwt(jwt) if jwt else None
    if requested and jwt_user and requested.lower() != jwt_user:
        raise SessionConfigError("OHHO_USER_ID does not match the JWT sub claim")
    if service:
        user_id = requested or jwt_user or ""
        if not user_id:
            raise SessionConfigError(
                "set OHHO_USER_ID to your auth.users UUID "
                "(the service role token is not a user)"
            )
        auth = SupabaseAuth(
            url=origin,
            api_key=service,
            bearer=service,
            user_id=require_user_id(user_id),
        )
    elif jwt and anon:
        user_id = requested or jwt_user or ""
        if not user_id:
            raise SessionConfigError(
                "the user JWT has no sub claim and OHHO_USER_ID is unset"
            )
        auth = SupabaseAuth(
            url=origin,
            api_key=anon,
            bearer=jwt,
            user_id=require_user_id(user_id),
        )
    else:
        raise SessionConfigError(
            "set OHHO_SUPABASE_SERVICE_ROLE_KEY, or both "
            "OHHO_SUPABASE_ANON_KEY and OHHO_SUPABASE_JWT"
        )
    for secret in (auth.api_key, auth.bearer):
        if secret and secret in auth.url:
            raise SessionConfigError("OHHO_SUPABASE_URL must not contain the API key")
    return auth


def _validate_host(host: str) -> str:
    cleaned = host.strip()
    if (
        not cleaned
        or any(character.isspace() for character in cleaned)
        or "://" in cleaned
        or "/" in cleaned
        or "@" in cleaned
    ):
        raise SessionConfigError("host must be an IP address or hostname")
    return cleaned


def advertise_host(env: Mapping[str, str] | None = None) -> str:
    """LAN address the headset should open. ``OHHO_SIM_HOST`` wins."""
    configured = _first(_env(env), ("OHHO_SIM_HOST",))
    if configured:
        return _validate_host(configured)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects a local address and does not send a datagram.
        sock.connect(("192.0.2.1", 9))
        host = sock.getsockname()[0] or "127.0.0.1"
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    return host


def _quote_query(
    value: str, safe: str, encoding: str | None, errors: str | None
) -> str:
    del encoding, errors
    return quote(value, safe="/")


def sanitize_endpoint(url: str) -> str:
    """Drop userinfo, fragments, and any ``token`` query parameter."""
    raw = url.strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if parts.scheme.lower() not in _ENDPOINT_SCHEMES:
        raise SessionConfigError("endpoint URL must be ws, wss, http, or https")
    host = parts.hostname
    if not host:
        raise SessionConfigError("endpoint URL is missing a host")
    netloc = f"[{host}]" if ":" in host else host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() != "token"
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            netloc,
            parts.path,
            urlencode(query, quote_via=_quote_query),
            "",
        )
    )


def _host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def default_endpoints(kind: str, target: str, host: str) -> tuple[str, list[str], str]:
    """URLs with no gateway token. The headset appends ``token`` from OHH-196."""
    shown = _host_for_url(_validate_host(host))
    if kind == KIND_SIM and target == TARGET_LOCAL:
        return (
            f"ws://{shown}:9090",
            [f"http://{shown}:8080/stream?topic={FRONT_CAMERA_TOPIC}"],
            "",
        )
    if kind == KIND_REAL and target == TARGET_ROBOT:
        return (
            f"wss://{shown}:8443/rosbridge",
            [f"https://{shown}:8443/video/stream?topic={FRONT_CAMERA_TOPIC}"],
            f"wss://{shown}:8443/webrtc/signaling",
        )
    raise SessionConfigError(f"no default endpoints for {kind}/{target}")


def _reject_token_leak(row: dict[str, Any], token: str) -> None:
    if _TOKEN_KEYS & set(row):
        raise SessionConfigError("session rows store token_hash only")
    blob = " ".join(
        [
            str(row.get("rosbridge_url") or ""),
            str(row.get("webrtc_signal_url") or ""),
            " ".join(str(item) for item in row.get("video_urls") or []),
        ]
    )
    if token and token in blob:
        raise SessionConfigError("refusing to store the gateway token in a session URL")


def build_session_row(
    *,
    user_id: str,
    robot_id: str,
    kind: str,
    target: str,
    rosbridge_url: str,
    video_urls: list[str],
    webrtc_signal_url: str,
    token: str,
    now: datetime,
) -> dict[str, Any]:
    owner = require_user_id(user_id)
    if kind not in KINDS:
        raise SessionConfigError(f"kind must be sim or real, got {kind!r}")
    if target not in TARGETS:
        raise SessionConfigError(
            f"target must be local, cloud, or robot, got {target!r}"
        )
    if not robot_id or any(character.isspace() for character in robot_id):
        raise SessionConfigError("robot_id is required")
    if not isinstance(video_urls, list) or not all(
        isinstance(item, str) for item in video_urls
    ):
        raise SessionConfigError("video_urls must be a list of URL strings")
    rosbridge = sanitize_endpoint(rosbridge_url)
    if not rosbridge:
        raise SessionConfigError("rosbridge_url is required")
    videos = [sanitize_endpoint(item) for item in video_urls if item.strip()]
    webrtc = sanitize_endpoint(webrtc_signal_url) if webrtc_signal_url else ""
    cleaned = validate_gateway_token(token)
    stamped = format_iso(now)
    row = {
        "user_id": owner,
        "robot_id": robot_id,
        "kind": kind,
        "target": target,
        "rosbridge_url": rosbridge,
        "video_urls": videos,
        "webrtc_signal_url": webrtc,
        "token_hash": hash_gateway_token(cleaned),
        "status": STATUS_ACTIVE,
        "heartbeat_at": stamped,
        "expires_at": format_iso(now + timedelta(seconds=STALE_AFTER_S)),
        "updated_at": stamped,
    }
    _reject_token_leak(row, cleaned)
    return row


def session_is_visible(row: Mapping[str, Any], now: datetime) -> bool:
    """Active rows whose ``expires_at`` is still in the future. Equal means stale."""
    if row.get("status") != STATUS_ACTIVE:
        return False
    expires = row.get("expires_at")
    if not isinstance(expires, str) or not expires:
        return False
    try:
        deadline = parse_iso(expires)
    except ValueError:
        return False
    return deadline > now


def resolve_endpoints(
    *,
    kind: str,
    target: str,
    host: str | None,
    rosbridge_url: str | None,
    video_urls: list[str] | None,
    webrtc_signal_url: str | None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, list[str], str]:
    chosen = _validate_host(host) if host else advertise_host(env)
    ros_default, video_default, webrtc_default = default_endpoints(kind, target, chosen)
    rosbridge = sanitize_endpoint(rosbridge_url) if rosbridge_url else ros_default
    if video_urls is None:
        videos = list(video_default)
    else:
        videos = [sanitize_endpoint(item) for item in video_urls if item.strip()]
    if webrtc_signal_url is None:
        webrtc = webrtc_default
    else:
        webrtc = (
            sanitize_endpoint(webrtc_signal_url) if webrtc_signal_url.strip() else ""
        )
    return rosbridge, videos, webrtc


class SessionRegistry:
    """Builds rows, heartbeats them, and hides stale ones."""

    def __init__(
        self,
        client: SessionClient,
        *,
        token: str,
        clock: Clock | None = None,
    ) -> None:
        self.client = client
        self.user_id = require_user_id(client.user_id)
        self.token = validate_gateway_token(token)
        self.clock = clock or utc_now

    def upsert(
        self,
        *,
        robot_id: str,
        kind: str,
        target: str,
        rosbridge_url: str,
        video_urls: list[str],
        webrtc_signal_url: str,
    ) -> dict[str, Any]:
        row = build_session_row(
            user_id=self.user_id,
            robot_id=robot_id,
            kind=kind,
            target=target,
            rosbridge_url=rosbridge_url,
            video_urls=video_urls,
            webrtc_signal_url=webrtc_signal_url,
            token=self.token,
            now=self.clock(),
        )
        saved = self.client.upsert(row)
        if not saved.get("id"):
            raise SessionError("supabase upsert did not return a session id")
        return saved

    def heartbeat(self, session_id: str) -> dict[str, Any]:
        now = self.clock()
        stamped = format_iso(now)
        return self.client.heartbeat(
            session_id,
            {
                "heartbeat_at": stamped,
                "expires_at": format_iso(now + timedelta(seconds=STALE_AFTER_S)),
                "status": STATUS_ACTIVE,
                "updated_at": stamped,
            },
        )

    def end(self, robot_id: str, kind: str, target: str) -> None:
        self.client.delete_slot(robot_id, kind, target)

    def list_visible(
        self,
        robot_id: str | None = None,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        moment = now or self.clock()
        rows = [
            row
            for row in self.client.list_sessions(robot_id)
            if session_is_visible(row, moment)
        ]
        rows.sort(key=lambda row: str(row.get("heartbeat_at") or ""), reverse=True)
        return rows


def _redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        redacted = redacted.replace(secret, "[redacted]")
    if len(redacted) > 300:
        return redacted[:300]
    return redacted


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would send the service-role key to whatever host answered."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, msg, headers, newurl
        raise SessionError(f"supabase refused a redirect ({code})")


def _opener() -> urllib.request.OpenerDirector:
    # Empty ProxyHandler ignores HTTP_PROXY so the bearer token stays on-box.
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )


def _default_http(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _opener().open(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except urllib.error.URLError as exc:
        raise SessionError(f"supabase request failed: {exc.reason}") from exc


class SupabaseSessionClient:
    """PostgREST client for ``robot_sessions``. No third-party SDK."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        bearer: str,
        user_id: str,
        http: Http | None = None,
        timeout: float = HTTP_TIMEOUT_S,
    ) -> None:
        self._origin = _origin(url)
        self._api_key = api_key
        self._bearer = bearer
        self.user_id = require_user_id(user_id)
        self._http = http or _default_http
        self._timeout = timeout
        for secret in (api_key, bearer):
            if secret and secret in self._origin:
                raise SessionConfigError(
                    "OHHO_SUPABASE_URL must not contain the API key"
                )

    def upsert(self, row: dict[str, Any]) -> dict[str, Any]:
        leaked = _TOKEN_KEYS & set(row)
        if leaked:
            raise SessionError("refusing to send a raw token to supabase")
        if row.get("user_id") != self.user_id:
            raise SessionError("refusing to write a session for a different user")
        url = f"{self._origin}/rest/v1/robot_sessions?on_conflict={ON_CONFLICT}"
        rows = self._request(
            "POST",
            url,
            json.dumps(row).encode("utf-8"),
            prefer="resolution=merge-duplicates,return=representation",
        )
        if not rows:
            raise SessionError("supabase upsert returned no row")
        return rows[0]

    def heartbeat(self, session_id: str, fields: dict[str, str]) -> dict[str, Any]:
        url = (
            f"{self._origin}/rest/v1/robot_sessions?"
            f"id=eq.{quote(session_id, safe='')}"
            f"&user_id=eq.{quote(self.user_id, safe='')}"
        )
        rows = self._request(
            "PATCH",
            url,
            json.dumps(fields).encode("utf-8"),
            prefer="return=representation",
        )
        if not rows:
            raise SessionGone(f"session {session_id} is gone")
        return rows[0]

    def delete_slot(self, robot_id: str, kind: str, target: str) -> None:
        url = (
            f"{self._origin}/rest/v1/robot_sessions?"
            f"user_id=eq.{quote(self.user_id, safe='')}"
            f"&robot_id=eq.{quote(robot_id, safe='')}"
            f"&kind=eq.{quote(kind, safe='')}"
            f"&target=eq.{quote(target, safe='')}"
        )
        self._request("DELETE", url, None, prefer=None)

    def list_sessions(self, robot_id: str | None = None) -> list[dict[str, Any]]:
        # The view compares expires_at to the database clock. Filtering with
        # this machine's clock here would hide a live row when the clocks differ.
        select = ",".join(SESSION_COLUMNS)
        filters = [
            f"user_id=eq.{quote(self.user_id, safe='')}",
            "status=eq.active",
        ]
        if robot_id is not None:
            filters.append(f"robot_id=eq.{quote(robot_id, safe='')}")
        query = "&".join([f"select={select}", *filters, "order=heartbeat_at.desc"])
        url = f"{self._origin}/rest/v1/robot_sessions_visible?{query}"
        return self._request("GET", url, None, prefer=None)

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        *,
        prefer: str | None,
    ) -> list[dict[str, Any]]:
        headers = {
            "apikey": self._api_key,
            "Authorization": f"Bearer {self._bearer}",
            "Accept": "application/json",
            "User-Agent": "ohho-os",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        status, payload = self._http(method, url, headers, body, self._timeout)
        text = payload.decode("utf-8", "replace") if payload else ""
        if status >= 400:
            detail = _redact(text, [self._api_key, self._bearer])
            raise SessionError(f"supabase {method} failed ({status}): {detail}")
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SessionError(f"supabase {method} returned non-JSON") from exc
        if not isinstance(data, list):
            raise SessionError(f"supabase {method} returned an unexpected payload")
        return data


def open_supabase_client(
    env: Mapping[str, str] | None = None,
    *,
    http: Http | None = None,
) -> SupabaseSessionClient:
    auth = load_supabase_auth(env)
    return SupabaseSessionClient(
        url=auth.url,
        api_key=auth.api_key,
        bearer=auth.bearer,
        user_id=auth.user_id,
        http=http,
    )


def _run_argv(argv: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", "docker not on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", "docker timed out"
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _compose_detail(extra: str) -> str:
    return (
        f"{extra} The GPU image is OHH-204 and is not published. "
        f"When it lands: {OHH204_COMPOSE_HINT}."
    )


def launch_local_sim(
    *,
    compose_file: Path | None = None,
    run: Callable[[list[str]], tuple[int, str, str]] | None = None,
) -> ContainerLaunch:
    """Run ``docker compose up -d``. A missing image is documented, not fatal."""
    path = compose_file or default_compose_file()
    command = ["docker", "compose", "-f", str(path), "up", "-d"]
    if not path.is_file():
        return ContainerLaunch(
            False,
            command,
            _compose_detail(f"compose file not found: {path}."),
        )
    if run is None and shutil.which("docker") is None:
        return ContainerLaunch(False, command, _compose_detail("docker not on PATH."))
    runner = run or _run_argv
    code, out, err = runner(command)
    if code != 0:
        tail = (err or out or "docker compose up failed").strip()
        return ContainerLaunch(False, command, _compose_detail(tail))
    ps_code, ps_out, _ps_err = runner(
        ["docker", "compose", "-f", str(path), "ps", "-q"]
    )
    if ps_code == 0 and ps_out.strip():
        return ContainerLaunch(True, command, "local sim container is running")
    return ContainerLaunch(False, command, _compose_detail("no container is running."))


def stop_local_sim(
    *,
    compose_file: Path | None = None,
    run: Callable[[list[str]], tuple[int, str, str]] | None = None,
) -> ContainerLaunch:
    path = compose_file or default_compose_file()
    command = ["docker", "compose", "-f", str(path), "down"]
    if run is None and shutil.which("docker") is None:
        return ContainerLaunch(False, command, "docker not on PATH")
    runner = run or _run_argv
    code, out, err = runner(command)
    if code != 0:
        return ContainerLaunch(
            False, command, (err or out or "docker compose down failed").strip()
        )
    return ContainerLaunch(True, command, "docker compose down finished")


def _safe_robot_id(robot_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", robot_id)


def session_paths(
    state_dir: Path, kind: str, target: str, robot_id: str
) -> tuple[Path, Path]:
    stem = f"{kind}-{target}-{_safe_robot_id(robot_id)}"
    return state_dir / f"{stem}.json", state_dir / f"{stem}.stop"


def _print_token_source(token: str, source: str) -> None:
    if source == "generated":
        print(f"gateway pairing code (enter on the headset once): {token}")
        return
    if source == "pairing":
        print(
            "gateway token: pairing file ~/.ohho/gateway/pairing-code "
            "(value not printed)"
        )
        return
    if source == "file":
        print("gateway token: OHHO_GATEWAY_TOKEN_FILE (value not printed)")
        return
    if source == "env":
        print("gateway token: OHHO_GATEWAY_TOKEN (value not printed)")
        return
    print("gateway token: provided (value not printed)")


def _warn_loopback(url: str) -> None:
    host = urlsplit(url).hostname or ""
    if host in _LOOPBACK:
        print(
            "session host is loopback. Set OHHO_SIM_HOST to this PC's LAN address "
            "so the headset can reach rosbridge.",
            file=sys.stderr,
        )


def _print_banner(saved: Mapping[str, Any], *, down_command: str) -> None:
    print(
        f"session {saved.get('id')} robot={saved.get('robot_id')} "
        f"kind={saved.get('kind')} target={saved.get('target')}"
    )
    print(f"rosbridge: {saved.get('rosbridge_url')}")
    videos = saved.get("video_urls") or []
    if videos:
        print("video: " + ", ".join(str(item) for item in videos))
    if saved.get("webrtc_signal_url"):
        print(f"webrtc: {saved.get('webrtc_signal_url')}")
    print(
        f"heartbeat every {int(HEARTBEAT_INTERVAL_S)}s — "
        f"Ctrl-C or `{down_command}` removes it"
    )


def _pause(
    interval_s: float,
    sleep: Callable[[float], None],
    should_stop: Callable[[], bool],
) -> bool:
    """Sleep ``interval_s``, checking ``should_stop`` about once a second."""
    whole = int(interval_s)
    frac = interval_s - whole
    for _ in range(whole):
        if should_stop():
            return False
        sleep(1.0)
    if frac > 0:
        if should_stop():
            return False
        sleep(frac)
    return not should_stop()


def _heartbeat_loop(
    registry: SessionRegistry,
    session_id: str,
    *,
    interval_s: float,
    sleep: Callable[[float], None],
    should_stop: Callable[[], bool],
    max_beats: int | None,
    on_beat: Callable[[int], None] | None,
) -> int:
    beats = 0
    while max_beats is None or beats < max_beats:
        if not _pause(interval_s, sleep, should_stop):
            break
        try:
            registry.heartbeat(session_id)
        except SessionGone:
            break
        except SessionError as exc:
            print(f"heartbeat failed: {exc}", file=sys.stderr)
            continue
        beats += 1
        if on_beat is not None:
            on_beat(beats)
    return beats


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _clear_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _run_session(
    *,
    robot_id: str,
    kind: str,
    target: str,
    down_command: str,
    host: str | None,
    rosbridge_url: str | None,
    video_urls: list[str] | None,
    webrtc_signal_url: str | None,
    client: SessionClient | None,
    token: str | None,
    env: Mapping[str, str] | None,
    pairing_file: Path | None,
    state_dir: Path | None,
    clock: Clock | None,
    manage_container: bool,
    launch: Callable[[], ContainerLaunch] | None,
    stop_container: Callable[[], ContainerLaunch] | None,
    sleep: Callable[[float], None] | None,
    max_beats: int | None,
    on_beat: Callable[[int], None] | None,
) -> int:
    directory = state_dir or default_state_dir()
    state_path, stop_path = session_paths(directory, kind, target, robot_id)
    _clear_paths(stop_path)
    code = 0
    registered = False
    launched = False
    registry: SessionRegistry | None = None
    try:
        resolved, source = (
            (validate_gateway_token(token), "argument")
            if token is not None
            else resolve_gateway_token(
                env=env, pairing_file=pairing_file, generate=True
            )
        )
        active = client if client is not None else open_supabase_client(env)
        registry = SessionRegistry(active, token=resolved, clock=clock)
        rosbridge, videos, webrtc = resolve_endpoints(
            kind=kind,
            target=target,
            host=host,
            rosbridge_url=rosbridge_url,
            video_urls=video_urls,
            webrtc_signal_url=webrtc_signal_url,
            env=env,
        )
        _print_token_source(resolved, source)
        _warn_loopback(rosbridge)
        if manage_container:
            starter = launch or (
                lambda: launch_local_sim(compose_file=default_compose_file())
            )
            result = starter()
            launched = True
            print(f"container: {result.detail}")
            print("container command: " + shlex.join(result.command))
        if not stop_path.exists():
            saved = registry.upsert(
                robot_id=robot_id,
                kind=kind,
                target=target,
                rosbridge_url=rosbridge,
                video_urls=videos,
                webrtc_signal_url=webrtc,
            )
            registered = True
            _write_state(
                state_path,
                {
                    "session_id": saved.get("id"),
                    "user_id": registry.user_id,
                    "robot_id": robot_id,
                    "kind": kind,
                    "target": target,
                    "rosbridge_url": saved.get("rosbridge_url"),
                    "pid": os.getpid(),
                },
            )
            if not stop_path.exists():
                _print_banner(saved, down_command=down_command)
                _heartbeat_loop(
                    registry,
                    str(saved["id"]),
                    interval_s=HEARTBEAT_INTERVAL_S,
                    sleep=sleep or time.sleep,
                    should_stop=stop_path.exists,
                    max_beats=max_beats,
                    on_beat=on_beat,
                )
    except KeyboardInterrupt:
        print("\nstopping.")
        code = 0
    except SessionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    finally:
        if registered and registry is not None:
            try:
                registry.end(robot_id, kind, target)
            except SessionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                code = 2
        _clear_paths(state_path, stop_path)
        if launched:
            stopper = stop_container or (
                lambda: stop_local_sim(compose_file=default_compose_file())
            )
            try:
                stopped = stopper()
            except OSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                code = 2
            else:
                print(stopped.detail)
    return code


def _ensure_robot(robot_id: str) -> bool:
    try:
        get_spec(robot_id)
    except UnknownRobot as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    return True


def run_sim_up(
    *,
    robot_id: str = "omnibot",
    target: str = TARGET_LOCAL,
    host: str | None = None,
    rosbridge_url: str | None = None,
    video_urls: list[str] | None = None,
    webrtc_signal_url: str | None = None,
    client: SessionClient | None = None,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    state_dir: Path | None = None,
    clock: Clock | None = None,
    launch: Callable[[], ContainerLaunch] | None = None,
    stop_container: Callable[[], ContainerLaunch] | None = None,
    sleep: Callable[[float], None] | None = None,
    max_beats: int | None = None,
    on_beat: Callable[[int], None] | None = None,
) -> int:
    """Register a sim session and heartbeat until Ctrl-C, ``sim down``, or ``max_beats``."""
    if target == TARGET_CLOUD:
        print(CloudTargetUnavailable(CLOUD_TARGET_MESSAGE), file=sys.stderr)
        return 2
    if target != TARGET_LOCAL:
        print(f"error: unknown sim target {target!r}", file=sys.stderr)
        return 2
    if not _ensure_robot(robot_id):
        return 2
    return _run_session(
        robot_id=robot_id,
        kind=KIND_SIM,
        target=TARGET_LOCAL,
        down_command=f"ohho sim down --robot {robot_id}",
        host=host,
        rosbridge_url=rosbridge_url,
        video_urls=video_urls,
        webrtc_signal_url=webrtc_signal_url,
        client=client,
        token=token,
        env=env,
        pairing_file=pairing_file,
        state_dir=state_dir,
        clock=clock,
        manage_container=True,
        launch=launch,
        stop_container=stop_container,
        sleep=sleep,
        max_beats=max_beats,
        on_beat=on_beat,
    )


def run_robot_announce(
    *,
    robot_id: str = "omnibot",
    down: bool = False,
    host: str | None = None,
    rosbridge_url: str | None = None,
    video_urls: list[str] | None = None,
    webrtc_signal_url: str | None = None,
    client: SessionClient | None = None,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    state_dir: Path | None = None,
    clock: Clock | None = None,
    sleep: Callable[[float], None] | None = None,
    max_beats: int | None = None,
    on_beat: Callable[[int], None] | None = None,
) -> int:
    """Publish a real robot (OHH-196 gateway URLs) and heartbeat until stopped."""
    if down:
        return _stop_session(
            robot_id=robot_id,
            kind=KIND_REAL,
            target=TARGET_ROBOT,
            client=client,
            env=env,
            state_dir=state_dir,
            manage_container=False,
            stop_container=None,
        )
    if not _ensure_robot(robot_id):
        return 2
    return _run_session(
        robot_id=robot_id,
        kind=KIND_REAL,
        target=TARGET_ROBOT,
        down_command=f"ohho robot announce --down --robot {robot_id}",
        host=host,
        rosbridge_url=rosbridge_url,
        video_urls=video_urls,
        webrtc_signal_url=webrtc_signal_url,
        client=client,
        token=token,
        env=env,
        pairing_file=pairing_file,
        state_dir=state_dir,
        clock=clock,
        manage_container=False,
        launch=None,
        stop_container=None,
        sleep=sleep,
        max_beats=max_beats,
        on_beat=on_beat,
    )


def _stop_session(
    *,
    robot_id: str,
    kind: str,
    target: str,
    client: SessionClient | None,
    env: Mapping[str, str] | None,
    state_dir: Path | None,
    manage_container: bool,
    stop_container: Callable[[], ContainerLaunch] | None,
) -> int:
    if kind == KIND_SIM and target == TARGET_CLOUD:
        print(CLOUD_TARGET_MESSAGE, file=sys.stderr)
        return 2
    if not _ensure_robot(robot_id):
        return 2
    directory = state_dir or default_state_dir()
    state_path, stop_path = session_paths(directory, kind, target, robot_id)
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("stop\n", encoding="utf-8")
    removed = False
    try:
        active = client if client is not None else open_supabase_client(env)
        active.delete_slot(robot_id, kind, target)
        removed = True
    except SessionConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "stop file written. A running heartbeat will exit and delete the row.",
            file=sys.stderr,
        )
    except SessionError as exc:
        print(f"error: {exc}", file=sys.stderr)
    if manage_container:
        stopper = stop_container or (
            lambda: stop_local_sim(compose_file=default_compose_file())
        )
        try:
            stopped = stopper()
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(stopped.detail)
    state_path.unlink(missing_ok=True)
    if removed:
        print(f"removed {kind}/{target} session for {robot_id}")
        return 0
    return 2


def run_sim_down(
    *,
    robot_id: str = "omnibot",
    target: str = TARGET_LOCAL,
    client: SessionClient | None = None,
    env: Mapping[str, str] | None = None,
    state_dir: Path | None = None,
    stop_container: Callable[[], ContainerLaunch] | None = None,
) -> int:
    """Delete the sim session and stop the local compose project."""
    if target == TARGET_CLOUD:
        print(CloudTargetUnavailable(CLOUD_TARGET_MESSAGE), file=sys.stderr)
        return 2
    if target != TARGET_LOCAL:
        print(f"error: unknown sim target {target!r}", file=sys.stderr)
        return 2
    return _stop_session(
        robot_id=robot_id,
        kind=KIND_SIM,
        target=TARGET_LOCAL,
        client=client,
        env=env,
        state_dir=state_dir,
        manage_container=True,
        stop_container=stop_container,
    )
