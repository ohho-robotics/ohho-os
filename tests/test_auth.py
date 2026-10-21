"""Device-flow login, whoami, and logout against a local HTTP console.

These tests never call the production console. They pass a temporary HOME /
USERPROFILE and never read or write the developer machine's credentials.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.parse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch

from ohho.auth import (
    AuthError,
    Credentials,
    console_origin,
    default_credentials_path,
    format_whoami,
    login,
    save_credentials,
    whoami,
)
from ohho.cli import build_parser, main

_REAL_CREDENTIALS = Path.home() / ".ohho" / "credentials.json"
_REAL_CREDENTIALS_BYTES = (
    _REAL_CREDENTIALS.read_bytes() if _REAL_CREDENTIALS.is_file() else None
)

_DEVICE_CODE = "device-test-code-" + ("d" * 24)
_FILE_TOKEN = "ohho_test_token_" + ("t" * 24)
_ENV_TOKEN = "ohho_env_token_" + ("e" * 24)
_USER_CODE = "WXYZ-2345"
_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "OHHO_TOKEN",
    "OHHO_CONSOLE_URL",
)


def tearDownModule() -> None:
    if _REAL_CREDENTIALS_BYTES is None:
        if _REAL_CREDENTIALS.exists():
            raise AssertionError("tests created the real credentials file")
        return
    current = _REAL_CREDENTIALS.read_bytes() if _REAL_CREDENTIALS.is_file() else None
    if current != _REAL_CREDENTIALS_BYTES:
        raise AssertionError("tests modified the real credentials file")


@dataclass
class Step:
    status: int
    body: dict
    headers: Optional[Dict[str, str]] = None


@dataclass
class Flow:
    device_code: str = _DEVICE_CODE
    user_code: str = _USER_CODE
    token: str = _FILE_TOKEN
    email: str = "ada@example.com"
    user_id: str = "user-123"
    scopes: List[str] = field(default_factory=lambda: ["mcp"])
    expires_in: int = 600
    interval: int = 0
    token_script: List[Step] = field(default_factory=list)
    redirect_code: bool = False
    verification_uri: str = ""
    accounts: Dict[str, dict] = field(default_factory=dict)
    requests: List[Tuple[str, dict]] = field(default_factory=list)
    origin: str = ""

    def allow(self, token: Optional[str] = None, email: Optional[str] = None) -> None:
        access = token or self.token
        self.token_script.append(
            Step(
                200,
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "scope": "mcp",
                },
            )
        )
        self.accounts[access] = {
            "email": email or self.email,
            "scopes": list(self.scopes),
            "user_id": self.user_id,
        }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def _handle(self) -> None:
        state: Flow = self.server.state  # type: ignore[attr-defined]
        path = urllib.parse.urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body: dict = {}
        if raw:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                body = parsed
        state.requests.append((path, body))
        if path == "/stolen":
            self._send(200, {"ok": False})
            return
        if path == "/api/cli/device/code":
            if state.redirect_code:
                location = state.origin + "/stolen"
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            uri = state.verification_uri or (state.origin + "/device")
            self._send(
                200,
                {
                    "device_code": state.device_code,
                    "user_code": state.user_code,
                    "verification_uri": uri,
                    "verification_uri_complete": (
                        f"{uri}?user_code={urllib.parse.quote(state.user_code)}"
                    ),
                    "expires_in": state.expires_in,
                    "interval": state.interval,
                },
            )
            return
        if path == "/api/cli/device/token":
            if not state.token_script:
                self._send(400, {"error": "expired_token"})
                return
            step = state.token_script.pop(0)
            self._send(step.status, step.body, step.headers)
            return
        if path == "/api/cli/whoami":
            auth = self.headers.get("Authorization", "")
            presented = auth[7:] if auth.startswith("Bearer ") else ""
            account = state.accounts.get(presented)
            if account is None:
                self._send(
                    401,
                    {
                        "error": "invalid_token",
                        "error_description": "The access token is missing, unknown, or revoked.",
                    },
                )
                return
            self._send(200, account)
            return
        self._send(404, {"error": "not_found"})

    def _send(
        self, status: int, body: dict, headers: Optional[Dict[str, str]] = None
    ) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)


class Console:
    def __init__(self, flow: Optional[Flow] = None) -> None:
        self.flow = flow or Flow()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> Flow:
        handler = _Handler
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        self._httpd.state = self.flow  # type: ignore[attr-defined]
        port = self._httpd.server_address[1]
        self.flow.origin = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.flow

    def __exit__(self, *exc: object) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: List[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


@contextmanager
def cli_env(
    home: Path, origin: Optional[str] = None, token: Optional[str] = None
) -> Iterator[None]:
    updates = {"HOME": str(home), "USERPROFILE": str(home)}
    if origin is not None:
        updates["OHHO_CONSOLE_URL"] = origin
    with patch.dict(os.environ, updates, clear=False):
        if token is None:
            os.environ.pop("OHHO_TOKEN", None)
        else:
            os.environ["OHHO_TOKEN"] = token
        if origin is None:
            os.environ.pop("OHHO_CONSOLE_URL", None)
        yield


def _home_env(home: Path, origin: str, token: Optional[str] = None) -> Dict[str, str]:
    env = {"HOME": str(home), "USERPROFILE": str(home), "OHHO_CONSOLE_URL": origin}
    if token is not None:
        env["OHHO_TOKEN"] = token
    return env


def _assert_private(test: unittest.TestCase, path: Path) -> None:
    if os.name != "nt":
        test.assertEqual(path.stat().st_mode & 0o777, 0o600)
        return
    output = subprocess.check_output(
        ["icacls", str(path)], text=True, encoding="utf-8", errors="replace"
    )
    username = os.environ.get("USERNAME", "")
    test.assertTrue(username, "USERNAME is required to check the Windows ACL")
    test.assertIn(f"{username}:(", output)
    test.assertEqual(output.count(":("), 1)
    test.assertNotIn("(I)", output)
    for banned in (
        "Everyone",
        "BUILTIN\\Users",
        "Authenticated Users",
        "NT AUTHORITY\\SYSTEM",
        "Administrators",
    ):
        test.assertNotIn(banned, output)


def _assert_secret_absent(test: unittest.TestCase, text: str, *secrets: str) -> None:
    for secret in secrets:
        test.assertTrue(secret not in text, "a secret was printed")


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in _PROXY_KEYS}
        for key in _PROXY_KEYS:
            os.environ.pop(key, None)
        self._tmp = tempfile.TemporaryDirectory(prefix="ohho-auth-")
        self.home = Path(self._tmp.name)
        self.creds = self.home / ".ohho" / "credentials.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LoginFlowTests(AuthTests):
    def test_pending_then_success_honours_interval(self) -> None:
        flow = Flow(interval=9, expires_in=600)
        flow.token_script.append(
            Step(400, {"error": "authorization_pending", "interval": 4})
        )
        flow.allow()
        clock = Clock()
        out = StringIO()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                sleep=clock.sleep,
                now=clock.now,
                open_browser=lambda _url: False,
                out=out,
            )
        self.assertEqual(clock.slept, [4])
        self.assertIn(_USER_CODE, out.getvalue())
        self.assertIn("/device?user_code=", out.getvalue())
        self.assertIn("Signed in as ada@example.com.", out.getvalue())
        _assert_secret_absent(self, out.getvalue(), _DEVICE_CODE, _FILE_TOKEN)
        stored = json.loads(self.creds.read_text(encoding="utf-8"))
        self.assertTrue(stored["token"] == _FILE_TOKEN, "saved token mismatch")
        self.assertEqual(stored["scope"], "mcp")
        self.assertEqual(stored["token_type"], "Bearer")
        self.assertEqual(stored["console_url"], server.origin)
        self.assertEqual(stored["version"], 1)
        _assert_private(self, self.creds)
        posted = [body for path, body in server.requests if path.endswith("/token")]
        self.assertGreaterEqual(len(posted), 2)
        self.assertEqual(posted[0]["grant_type"], _GRANT)
        self.assertTrue(
            posted[0]["device_code"] == _DEVICE_CODE, "device_code was not sent"
        )
        code_body = next(
            body for path, body in server.requests if path.endswith("/code")
        )
        self.assertEqual(code_body["scope"], "mcp")
        self.assertEqual(code_body["client_name"], "OhhO CLI")

    def test_slow_down_uses_server_interval(self) -> None:
        flow = Flow(interval=1)
        flow.token_script.append(
            Step(
                400,
                {
                    "error": "slow_down",
                    "interval": 7,
                    "error_description": "Polling too often.",
                },
            )
        )
        flow.allow()
        clock = Clock()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                sleep=clock.sleep,
                now=clock.now,
                out=StringIO(),
            )
        self.assertEqual(clock.slept, [7.0])

    def test_slow_down_without_interval_adds_five_seconds(self) -> None:
        flow = Flow(interval=1)
        flow.token_script.append(Step(400, {"error": "slow_down"}))
        flow.allow()
        clock = Clock()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                sleep=clock.sleep,
                now=clock.now,
                out=StringIO(),
            )
        self.assertEqual(clock.slept, [6.0])

    def test_retry_after_header_is_honoured(self) -> None:
        flow = Flow(interval=1)
        flow.token_script.append(
            Step(429, {"error": "slow_down"}, headers={"Retry-After": "3"})
        )
        flow.allow()
        clock = Clock()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                sleep=clock.sleep,
                now=clock.now,
                out=StringIO(),
            )
        self.assertEqual(clock.slept, [3.0])

    def test_client_stops_when_the_code_expires(self) -> None:
        flow = Flow(interval=5, expires_in=5)
        flow.token_script.append(
            Step(400, {"error": "authorization_pending", "interval": 5})
        )
        flow.allow()
        clock = Clock()
        out = StringIO()
        with Console(flow) as server:
            with self.assertRaises(AuthError) as caught:
                login(
                    no_browser=True,
                    environ=_home_env(self.home, server.origin),
                    credentials_file=self.creds,
                    sleep=clock.sleep,
                    now=clock.now,
                    out=out,
                )
        self.assertIn("expired", str(caught.exception).lower())
        self.assertFalse(self.creds.exists())
        self.assertEqual(len(server.requests), 2)
        _assert_secret_absent(self, out.getvalue(), _DEVICE_CODE, _FILE_TOKEN)
        self.assertIn(_USER_CODE, out.getvalue())

    def test_server_expired_token(self) -> None:
        flow = Flow(interval=0, expires_in=600)
        flow.token_script.append(
            Step(
                400,
                {
                    "error": "expired_token",
                    "error_description": "The device code is expired, unknown, or already used.",
                },
            )
        )
        with Console(flow) as server:
            with self.assertRaises(AuthError) as caught:
                login(
                    no_browser=True,
                    environ=_home_env(self.home, server.origin),
                    credentials_file=self.creds,
                    out=StringIO(),
                )
        self.assertIn("expired", str(caught.exception).lower())
        self.assertFalse(self.creds.exists())

    def test_access_denied(self) -> None:
        flow = Flow()
        flow.token_script.append(Step(400, {"error": "access_denied"}))
        with Console(flow) as server:
            with self.assertRaises(AuthError) as caught:
                login(
                    no_browser=True,
                    environ=_home_env(self.home, server.origin),
                    credentials_file=self.creds,
                    out=StringIO(),
                )
        self.assertIn("denied", str(caught.exception).lower())
        self.assertFalse(self.creds.exists())

    def test_redirect_is_not_followed(self) -> None:
        flow = Flow(redirect_code=True)
        with Console(flow) as server:
            with self.assertRaises(AuthError) as caught:
                login(
                    no_browser=True,
                    environ=_home_env(self.home, server.origin),
                    credentials_file=self.creds,
                    out=StringIO(),
                )
            stolen = [path for path, _body in server.requests if path == "/stolen"]
        self.assertIn("redirected", str(caught.exception).lower())
        self.assertEqual(stolen, [])
        self.assertFalse(self.creds.exists())

    def test_browser_opens_the_verification_url(self) -> None:
        flow = Flow()
        flow.allow()
        opened: List[str] = []
        with Console(flow) as server:
            login(
                no_browser=False,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                open_browser=lambda url: opened.append(url) or True,
                out=StringIO(),
            )
            expected = f"{server.origin}/device?user_code={_USER_CODE}"
        self.assertEqual(opened, [expected])

    def test_no_browser_does_not_call_opener(self) -> None:
        flow = Flow()
        flow.allow()
        opened: List[str] = []
        out = StringIO()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                open_browser=lambda url: opened.append(url) or True,
                out=out,
            )
        self.assertEqual(opened, [])
        self.assertIn("Browser not opened (--no-browser).", out.getvalue())

    def test_preview_hint_when_verification_host_differs(self) -> None:
        flow = Flow(verification_uri="https://device.example/device")
        flow.allow()
        out = StringIO()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin),
                credentials_file=self.creds,
                out=out,
            )
        text = out.getvalue()
        self.assertIn("https://device.example/device?user_code=", text)
        self.assertIn(f"{server.origin}/device", text)
        _assert_secret_absent(self, text, _DEVICE_CODE, _FILE_TOKEN)

    def test_env_token_is_not_written_during_login(self) -> None:
        flow = Flow()
        flow.allow()
        out = StringIO()
        with Console(flow) as server:
            login(
                no_browser=True,
                environ=_home_env(self.home, server.origin, token=_ENV_TOKEN),
                credentials_file=self.creds,
                out=out,
            )
        raw = self.creds.read_text(encoding="utf-8")
        self.assertTrue(_FILE_TOKEN in raw, "device token was not saved")
        self.assertTrue(_ENV_TOKEN not in raw, "OHHO_TOKEN was written to disk")
        text = out.getvalue()
        self.assertIn("OHHO_TOKEN is still set", text)
        _assert_secret_absent(self, text, _FILE_TOKEN, _ENV_TOKEN, _DEVICE_CODE)

    def test_login_does_not_log_secrets(self) -> None:
        flow = Flow()
        flow.allow()
        handler = _LogCapture()
        root = logging.getLogger()
        previous = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            with Console(flow) as server:
                login(
                    no_browser=True,
                    environ=_home_env(self.home, server.origin),
                    credentials_file=self.creds,
                    out=StringIO(),
                )
        finally:
            root.removeHandler(handler)
            root.setLevel(previous)
        _assert_secret_absent(self, "\n".join(handler.lines), _FILE_TOKEN, _DEVICE_CODE)

    def test_bad_console_url_does_not_call_the_network(self) -> None:
        flow = Flow()
        flow.allow()
        with Console(flow) as server:
            with self.assertRaises(AuthError):
                login(
                    no_browser=True,
                    environ={
                        "HOME": str(self.home),
                        "USERPROFILE": str(self.home),
                        "OHHO_CONSOLE_URL": "ftp://files.example/path",
                    },
                    credentials_file=self.creds,
                    out=StringIO(),
                )
            self.assertEqual(server.requests, [])


class WhoamiLogoutTests(AuthTests):
    def test_whoami_json_schema_when_signed_in(self) -> None:
        flow = Flow()
        flow.accounts[_FILE_TOKEN] = {
            "email": "ada@example.com",
            "scopes": ["mcp"],
            "user_id": "user-123",
        }
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )
        with Console(flow) as server:
            with cli_env(self.home, server.origin):
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["whoami", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            list(payload),
            ["signed_in", "email", "scopes", "user_id"],
        )
        self.assertEqual(
            payload,
            {
                "signed_in": True,
                "email": "ada@example.com",
                "scopes": ["mcp"],
                "user_id": "user-123",
            },
        )
        self.assertEqual(stderr.getvalue(), "")
        _assert_secret_absent(self, stdout.getvalue(), _FILE_TOKEN)
        _assert_private(self, self.creds)

    def test_whoami_text_and_not_signed_in(self) -> None:
        with Console() as server:
            save_credentials(
                self.creds,
                Credentials(
                    token=_FILE_TOKEN,
                    token_type="Bearer",
                    scope="mcp",
                    console_url=server.origin,
                ),
            )
            server.accounts[_FILE_TOKEN] = {
                "email": "ada@example.com",
                "scopes": ["mcp"],
                "user_id": "user-123",
            }
            with cli_env(self.home, server.origin):
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["whoami"])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "email: ada@example.com\nscopes: mcp\n")
            _assert_secret_absent(
                self, stdout.getvalue() + stderr.getvalue(), _FILE_TOKEN
            )

            self.creds.unlink()
            stdout, stderr = StringIO(), StringIO()
            with cli_env(self.home, server.origin):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["whoami"])
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "not signed in\n")

            stdout = StringIO()
            with cli_env(self.home, server.origin):
                with redirect_stdout(stdout):
                    code = main(["whoami", "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                list(payload),
                ["signed_in", "email", "scopes", "user_id", "error"],
            )
            self.assertEqual(
                payload,
                {
                    "signed_in": False,
                    "email": None,
                    "scopes": [],
                    "user_id": None,
                    "error": "not_signed_in",
                },
            )
            self.assertEqual(server.requests, [("/api/cli/whoami", {})])

    def test_whoami_drops_fields_that_contain_the_token(self) -> None:
        flow = Flow()
        flow.accounts[_FILE_TOKEN] = {
            "email": f"user-{_FILE_TOKEN}@example.com",
            "scopes": ["mcp", _FILE_TOKEN],
            "user_id": _FILE_TOKEN,
        }
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )
        with Console(flow) as server:
            with cli_env(self.home, server.origin):
                stdout, stderr = StringIO(), StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["whoami", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIsNone(payload["email"])
        self.assertEqual(payload["scopes"], ["mcp"])
        self.assertIsNone(payload["user_id"])
        _assert_secret_absent(self, stdout.getvalue() + stderr.getvalue(), _FILE_TOKEN)

    def test_invalid_token_is_not_signed_in(self) -> None:
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="http://127.0.0.1",
            ),
        )
        with Console() as server:
            with cli_env(self.home, server.origin):
                text_out, text_err = StringIO(), StringIO()
                with redirect_stdout(text_out), redirect_stderr(text_err):
                    text_code = main(["whoami"])
                json_out = StringIO()
                with redirect_stdout(json_out):
                    json_code = main(["whoami", "--json"])
        self.assertEqual(text_code, 1)
        self.assertEqual(text_out.getvalue(), "not signed in\n")
        self.assertEqual(text_err.getvalue(), "")
        self.assertEqual(json_code, 1)
        self.assertEqual(json.loads(json_out.getvalue())["error"], "invalid_token")
        _assert_secret_absent(
            self, text_out.getvalue() + json_out.getvalue(), _FILE_TOKEN
        )

    def test_ohho_token_overrides_file_and_is_not_written(self) -> None:
        flow = Flow()
        flow.accounts[_FILE_TOKEN] = {
            "email": "file@example.com",
            "scopes": ["mcp"],
            "user_id": "file-user",
        }
        flow.accounts[_ENV_TOKEN] = {
            "email": "env@example.com",
            "scopes": ["mcp"],
            "user_id": "env-user",
        }
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )
        before = self.creds.read_bytes()
        with Console(flow) as server:
            with cli_env(self.home, server.origin, token=_ENV_TOKEN):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(["whoami", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["email"], "env@example.com")
        self.assertEqual(payload["user_id"], "env-user")
        self.assertTrue(
            self.creds.read_bytes() == before, "credentials file was rewritten"
        )
        raw = self.creds.read_text(encoding="utf-8")
        self.assertTrue(_ENV_TOKEN not in raw, "OHHO_TOKEN was written to disk")
        _assert_secret_absent(self, stdout.getvalue(), _FILE_TOKEN, _ENV_TOKEN)

    def test_corrupt_credentials_do_not_echo_the_token(self) -> None:
        self.creds.parent.mkdir(parents=True)
        self.creds.write_text("not-json " + _FILE_TOKEN, encoding="utf-8")
        stdout, stderr = StringIO(), StringIO()
        with cli_env(self.home):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["whoami", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["error"], "credentials_unreadable")
        _assert_secret_absent(self, stdout.getvalue() + stderr.getvalue(), _FILE_TOKEN)

    def test_logout_deletes_the_file_and_does_not_call_the_console(self) -> None:
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )

        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("logout called the console")

        with patch("ohho.auth._request_json", boom):
            with cli_env(self.home):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(["logout"])
        self.assertEqual(code, 0)
        self.assertFalse(self.creds.exists())
        self.assertIn("Local credentials removed.", stdout.getvalue())
        with cli_env(self.home):
            stdout = StringIO()
            with redirect_stdout(stdout):
                again = main(["logout"])
        self.assertEqual(again, 0)
        self.assertIn("Signed out.", stdout.getvalue())

    def test_logout_leaves_ohho_token_in_the_environment(self) -> None:
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )
        flow = Flow()
        flow.accounts[_ENV_TOKEN] = {
            "email": "env@example.com",
            "scopes": ["mcp"],
            "user_id": "env-user",
        }
        with Console(flow) as server:
            with cli_env(self.home, server.origin, token=_ENV_TOKEN):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(["logout"])
                self.assertEqual(code, 0)
                self.assertFalse(self.creds.exists())
                self.assertIn("was not written to disk", stdout.getvalue())
                self.assertTrue(
                    _ENV_TOKEN not in stdout.getvalue(), "env token printed"
                )
                who = StringIO()
                with redirect_stdout(who):
                    who_code = main(["whoami", "--json"])
        self.assertEqual(who_code, 0)
        self.assertEqual(json.loads(who.getvalue())["email"], "env@example.com")
        self.assertFalse(self.creds.exists())

    def test_console_unavailable_json(self) -> None:
        env = {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "OHHO_TOKEN": _ENV_TOKEN,
            "OHHO_CONSOLE_URL": "http://127.0.0.1:9",
        }
        account = whoami(environ=env, credentials_file=self.creds)
        self.assertFalse(account.signed_in)
        self.assertEqual(account.error, "console_unavailable")
        rendered = json.loads(format_whoami(account, as_json=True))
        self.assertEqual(rendered["error"], "console_unavailable")
        self.assertIsNone(rendered["email"])
        _assert_secret_absent(self, json.dumps(rendered), _ENV_TOKEN)

    def test_credentials_path_follows_environ(self) -> None:
        with cli_env(self.home):
            with patch.object(
                Path, "home", side_effect=AssertionError("used Path.home")
            ):
                self.assertEqual(default_credentials_path(), self.creds)
        self.assertEqual(console_origin({}), "https://console.ohho-robotics.com")
        self.assertEqual(
            console_origin({"OHHO_CONSOLE_URL": "https://preview.example/"}),
            "https://preview.example",
        )


class SignedOutCommandsTests(AuthTests):
    def test_parser_accepts_account_commands(self) -> None:
        login_args = build_parser().parse_args(["login", "--no-browser"])
        self.assertEqual(login_args.command, "login")
        self.assertTrue(login_args.no_browser)
        who_args = build_parser().parse_args(["whoami", "--json"])
        self.assertTrue(who_args.json)
        self.assertEqual(build_parser().parse_args(["logout"]).command, "logout")

    def test_doctor_does_not_print_a_token(self) -> None:
        save_credentials(
            self.creds,
            Credentials(
                token=_FILE_TOKEN,
                token_type="Bearer",
                scope="mcp",
                console_url="https://console.ohho-robotics.com",
            ),
        )
        stdout, stderr = StringIO(), StringIO()
        with cli_env(self.home, token=_ENV_TOKEN):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["doctor"])
        self.assertEqual(code, 0)
        _assert_secret_absent(
            self, stdout.getvalue() + stderr.getvalue(), _FILE_TOKEN, _ENV_TOKEN
        )

    def test_sim_and_mcp_connect_work_signed_out(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with cli_env(self.home, token=_ENV_TOKEN):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                sim = main(["sim", "--robot", "omnibot", "--seconds", "0"])
                listed = main(["mcp", "connect", "--list"])
                version = main(["version"])
        self.assertEqual(sim, 0)
        self.assertEqual(listed, 0)
        self.assertEqual(version, 0)
        self.assertIn("cursor", stdout.getvalue())
        _assert_secret_absent(self, stdout.getvalue() + stderr.getvalue(), _ENV_TOKEN)

    def test_cli_login_no_browser_does_not_open_a_browser(self) -> None:
        flow = Flow(interval=0)
        flow.allow()
        with Console(flow) as server:
            with cli_env(self.home, server.origin):
                opened: List[str] = []

                def _open(url: str, *args: object, **kwargs: object) -> bool:
                    opened.append(url)
                    return True

                with patch("webbrowser.open", side_effect=_open):
                    stdout, stderr = StringIO(), StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = main(["login", "--no-browser"])
        self.assertEqual(opened, [])
        self.assertEqual(code, 0)
        self.assertIn(_USER_CODE, stdout.getvalue())
        self.assertIn("Signed in as ada@example.com.", stdout.getvalue())
        _assert_secret_absent(
            self, stdout.getvalue() + stderr.getvalue(), _FILE_TOKEN, _DEVICE_CODE
        )
        _assert_private(self, self.creds)


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


if __name__ == "__main__":
    unittest.main()
