"""Session registry tests. No socket to Supabase and no docker daemon."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from ohho.cli import build_parser, main
from ohho.sessions import (
    CLOUD_TARGET_MESSAGE,
    HEARTBEAT_INTERVAL_S,
    OHH204_COMPOSE_HINT,
    ON_CONFLICT,
    SESSION_COLUMNS,
    STALE_AFTER_S,
    SUPABASE_ENV_KEYS,
    CloudTargetUnavailable,
    ContainerLaunch,
    SessionError,
    SessionGone,
    SessionRegistry,
    SupabaseSessionClient,
    _NoRedirect,
    _opener,
    advertise_host,
    build_session_row,
    default_compose_file,
    format_iso,
    hash_gateway_token,
    launch_local_sim,
    load_supabase_auth,
    resolve_gateway_token,
    run_robot_announce,
    run_sim_down,
    run_sim_up,
    sanitize_endpoint,
    session_is_visible,
    session_paths,
    stop_local_sim,
    user_id_from_jwt,
)

USER = "00000000-0000-0000-0000-0000000000aa"
OTHER = "00000000-0000-0000-0000-0000000000bb"
TOKEN = "example-gateway-token"
TOKEN_HASH = "acf72e1e008bd82c8a08ebee883ac904d34e70865ca615bbb64bb8d615247b26"
ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "supabase" / "migrations" / "20261014120000_robot_sessions.sql"
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, moment: datetime = NOW) -> None:
        self.now = moment

    def __call__(self) -> datetime:
        return self.now


class FakeSupabase:
    """In-memory PostgREST stand-in. Returns stale rows so the registry must filter."""

    def __init__(self, user_id: str = USER) -> None:
        self.user_id = user_id
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self._seq = 0

    def upsert(self, row: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("upsert", row))
        if row.get("user_id") != self.user_id:
            raise SessionError("wrong user")
        key = (row["user_id"], row["robot_id"], row["kind"], row["target"])
        existing = next(
            (
                saved
                for saved in self.rows.values()
                if (saved["user_id"], saved["robot_id"], saved["kind"], saved["target"])
                == key
            ),
            None,
        )
        saved = dict(existing or {})
        saved.update(row)
        if "id" not in saved:
            self._seq += 1
            saved["id"] = f"session-{self._seq}"
        self.rows[str(saved["id"])] = saved
        return dict(saved)

    def heartbeat(self, session_id: str, fields: dict[str, str]) -> dict[str, Any]:
        self.calls.append(("heartbeat", session_id, fields))
        row = self.rows.get(session_id)
        if row is None or row.get("user_id") != self.user_id:
            raise SessionGone(session_id)
        row.update(fields)
        return dict(row)

    def delete_slot(self, robot_id: str, kind: str, target: str) -> None:
        self.calls.append(("delete", robot_id, kind, target))
        doomed = [
            session_id
            for session_id, row in self.rows.items()
            if row["user_id"] == self.user_id
            and row["robot_id"] == robot_id
            and row["kind"] == kind
            and row["target"] == target
        ]
        for session_id in doomed:
            del self.rows[session_id]

    def list_sessions(self, robot_id: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("list", robot_id))
        found = []
        for row in self.rows.values():
            if row["user_id"] != self.user_id:
                continue
            if robot_id is not None and row["robot_id"] != robot_id:
                continue
            found.append(dict(row))
        return found


def _jwt(sub: str) -> str:
    import base64

    def enc(payload: bytes) -> str:
        return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")

    header = enc(b'{"alg":"none","typ":"JWT"}')
    body = enc(json.dumps({"sub": sub}).encode("utf-8"))
    return f"{header}.{body}.sig"


def _not_sleep(_seconds: float) -> None:
    raise AssertionError("heartbeat slept")


class SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.fake = FakeSupabase()
        self.launches: list[ContainerLaunch] = []
        self.stops: list[ContainerLaunch] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def launch(self) -> ContainerLaunch:
        result = ContainerLaunch(
            False, ["docker", "compose", "up", "-d"], "OHH-204 stub"
        )
        self.launches.append(result)
        return result

    def stop(self) -> ContainerLaunch:
        result = ContainerLaunch(True, ["docker", "compose", "down"], "down")
        self.stops.append(result)
        return result

    def up(self, **kwargs: Any) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        options: dict[str, Any] = {
            "robot_id": "omnibot",
            "target": "local",
            "host": "192.0.2.10",
            "client": self.fake,
            "token": TOKEN,
            "env": {},
            "state_dir": self.dir,
            "launch": self.launch,
            "stop_container": self.stop,
            "sleep": _not_sleep,
            "max_beats": 0,
        }
        options.update(kwargs)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run_sim_up(**options)
        return code, stdout.getvalue(), stderr.getvalue()


class TestTokenAndVisibility(SessionTestCase):
    def test_hash_matches_ohh196_sha256(self) -> None:
        self.assertEqual(HEARTBEAT_INTERVAL_S, 10.0)
        self.assertEqual(STALE_AFTER_S, 30.0)
        self.assertEqual(hash_gateway_token(TOKEN), TOKEN_HASH)
        self.assertEqual(
            hash_gateway_token(TOKEN),
            hashlib.sha256(TOKEN.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(hash_gateway_token(f"  {TOKEN}\n"), TOKEN_HASH)

    def test_short_token_rejected(self) -> None:
        with self.assertRaises(SessionError):
            hash_gateway_token("too-short")

    def test_resolve_env_does_not_write_pairing_file(self) -> None:
        pairing = self.dir / "pairing-code"
        token, source = resolve_gateway_token(
            env={"OHHO_GATEWAY_TOKEN": TOKEN},
            pairing_file=pairing,
        )
        self.assertEqual(token, TOKEN)
        self.assertEqual(source, "env")
        self.assertFalse(pairing.exists())

    def test_resolve_generates_pairing_file(self) -> None:
        pairing = self.dir / "pairing-code"
        token, source = resolve_gateway_token(env={}, pairing_file=pairing)
        self.assertEqual(source, "generated")
        self.assertGreaterEqual(len(token), 16)
        self.assertEqual(pairing.read_text(encoding="utf-8").strip(), token)

    def test_jwt_sub(self) -> None:
        self.assertEqual(user_id_from_jwt(_jwt(USER.upper())), USER)
        self.assertIsNone(user_id_from_jwt("not-a-jwt"))

    def test_service_role_requires_user_id(self) -> None:
        with self.assertRaises(SessionError) as caught:
            load_supabase_auth(
                {
                    "OHHO_SUPABASE_URL": "https://example.supabase.co",
                    "OHHO_SUPABASE_SERVICE_ROLE_KEY": "service-role-test-key",
                }
            )
        self.assertNotIn("service-role-test-key", str(caught.exception))
        self.assertIn("OHHO_USER_ID", str(caught.exception))

    def test_user_jwt_sub_and_mismatch(self) -> None:
        auth = load_supabase_auth(
            {
                "OHHO_SUPABASE_URL": "https://example.supabase.co/rest/v1",
                "OHHO_SUPABASE_ANON_KEY": "anon-test-key",
                "OHHO_SUPABASE_JWT": _jwt(USER),
            }
        )
        self.assertEqual(auth.user_id, USER)
        self.assertEqual(auth.api_key, "anon-test-key")
        self.assertEqual(auth.bearer, _jwt(USER))
        self.assertEqual(auth.url, "https://example.supabase.co")
        with self.assertRaises(SessionError):
            load_supabase_auth(
                {
                    "OHHO_SUPABASE_URL": "https://example.supabase.co",
                    "OHHO_SUPABASE_ANON_KEY": "anon-test-key",
                    "OHHO_SUPABASE_JWT": _jwt(USER),
                    "OHHO_USER_ID": OTHER,
                }
            )

    def test_missing_supabase_url(self) -> None:
        with self.assertRaises(SessionError) as caught:
            load_supabase_auth({})
        self.assertIn("OHHO_SUPABASE_URL", str(caught.exception))

    def test_sanitize_strips_token_and_userinfo(self) -> None:
        cleaned = sanitize_endpoint(
            f"ws://user:{TOKEN}@192.0.2.10:9090/rosbridge?token={TOKEN}&foo=1"
        )
        self.assertEqual(cleaned, "ws://192.0.2.10:9090/rosbridge?foo=1")
        self.assertNotIn(TOKEN, cleaned)

    def test_advertise_host_env(self) -> None:
        self.assertEqual(advertise_host({"OHHO_SIM_HOST": "192.0.2.20"}), "192.0.2.20")

    def test_stale_and_ended_are_hidden(self) -> None:
        fresh = {
            "status": "active",
            "expires_at": format_iso(NOW + timedelta(seconds=1)),
        }
        self.assertTrue(session_is_visible(fresh, NOW))
        self.assertFalse(
            session_is_visible({"status": "active", "expires_at": format_iso(NOW)}, NOW)
        )
        self.assertFalse(
            session_is_visible(
                {
                    "status": "ended",
                    "expires_at": format_iso(NOW + timedelta(hours=1)),
                },
                NOW,
            )
        )
        self.assertFalse(session_is_visible({"status": "active"}, NOW))
        self.assertFalse(
            session_is_visible({"status": "active", "expires_at": "nope"}, NOW)
        )

    def test_row_stores_hash_only(self) -> None:
        row = build_session_row(
            user_id=USER,
            robot_id="omnibot",
            kind="sim",
            target="local",
            rosbridge_url=f"ws://192.0.2.10:9090/rosbridge?token={TOKEN}",
            video_urls=[f"http://192.0.2.10:8080/stream?topic=/cam&token={TOKEN}"],
            webrtc_signal_url="",
            token=TOKEN,
            now=NOW,
        )
        encoded = json.dumps(row)
        self.assertNotIn(TOKEN, encoded)
        self.assertNotIn("token=", encoded)
        self.assertEqual(row["token_hash"], TOKEN_HASH)
        self.assertNotIn("token", row)
        self.assertEqual(row["expires_at"], format_iso(NOW + timedelta(seconds=30)))
        self.assertIn("topic=/cam", row["video_urls"][0])


class TestSupabaseClient(SessionTestCase):
    def _client(self, http) -> SupabaseSessionClient:
        return SupabaseSessionClient(
            url="https://example.supabase.co",
            api_key="service-role-test-key",
            bearer="service-role-test-key",
            user_id=USER,
            http=http,
        )

    def test_upsert_posts_hash_and_on_conflict(self) -> None:
        seen: dict[str, Any] = {}

        def http(method, url, headers, body, timeout):
            seen.update(
                method=method, url=url, headers=headers, body=body, timeout=timeout
            )
            row = json.loads(body)
            row["id"] = "session-1"
            return 201, json.dumps([row]).encode()

        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            saved = self._client(http).upsert(
                build_session_row(
                    user_id=USER,
                    robot_id="omnibot",
                    kind="sim",
                    target="local",
                    rosbridge_url="ws://192.0.2.10:9090",
                    video_urls=[],
                    webrtc_signal_url="",
                    token=TOKEN,
                    now=NOW,
                )
            )
        self.assertEqual(saved["id"], "session-1")
        self.assertEqual(seen["method"], "POST")
        self.assertIn(f"on_conflict={ON_CONFLICT}", seen["url"])
        self.assertIn("resolution=merge-duplicates", seen["headers"]["Prefer"])
        self.assertEqual(
            seen["headers"]["Authorization"], "Bearer service-role-test-key"
        )
        payload = json.loads(seen["body"])
        self.assertEqual(payload["token_hash"], TOKEN_HASH)
        self.assertNotIn(TOKEN, seen["body"].decode())
        self.assertNotIn("token", payload)

    def test_list_uses_visible_view_and_owner_filter(self) -> None:
        seen: dict[str, str] = {}

        def http(method, url, headers, body, timeout):
            del method, headers, body, timeout
            seen["url"] = url
            return 200, b"[]"

        self._client(http).list_sessions("omnibot")
        parts = urlsplit(seen["url"])
        self.assertTrue(parts.path.endswith("/robot_sessions_visible"))
        query = parse_qs(parts.query)
        select = query["select"][0]
        for column in SESSION_COLUMNS:
            self.assertIn(column, select)
        self.assertNotIn("*", select)
        self.assertEqual(query["user_id"][0], f"eq.{USER}")
        self.assertEqual(query["robot_id"][0], "eq.omnibot")
        self.assertEqual(query["status"][0], "eq.active")
        self.assertNotIn(TOKEN, seen["url"])
        self.assertNotIn("service-role-test-key", seen["url"])

    def test_error_redacts_the_key_and_refuses_a_raw_token(self) -> None:
        def http(method, url, headers, body, timeout):
            del method, url, headers, body, timeout
            return 401, b"bad service-role-test-key"

        client = self._client(http)
        with self.assertRaises(SessionError) as caught:
            client.upsert(
                build_session_row(
                    user_id=USER,
                    robot_id="omnibot",
                    kind="sim",
                    target="local",
                    rosbridge_url="ws://192.0.2.10:9090",
                    video_urls=[],
                    webrtc_signal_url="",
                    token=TOKEN,
                    now=NOW,
                )
            )
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("service-role-test-key", str(caught.exception))
        with self.assertRaises(SessionError):
            client.upsert({"token": TOKEN, "user_id": USER})
        with self.assertRaises(SessionError):
            client.upsert({"user_id": OTHER, "robot_id": "omnibot"})

    def test_empty_heartbeat_is_gone(self) -> None:
        def http(method, url, headers, body, timeout):
            del method, headers, body, timeout
            self.assertIn(f"user_id=eq.{USER}", url)
            return 200, b"[]"

        with self.assertRaises(SessionGone):
            self._client(http).heartbeat("missing", {"status": "active"})

    def test_redirects_and_proxies_are_disabled(self) -> None:
        # An empty ProxyHandler skips urllib's env-proxy default and installs
        # no proxy openers, so HTTP_PROXY cannot see the service-role key.
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.example:8080",
                "http_proxy": "http://proxy.example:8080",
            },
        ):
            opener = _opener()
        self.assertFalse(
            any(getattr(handler, "proxies", None) for handler in opener.handlers)
        )
        with self.assertRaises(SessionError) as caught:
            _NoRedirect().redirect_request(
                None, None, 302, "Found", {}, "https://evil.example/steal"
            )
        self.assertIn("302", str(caught.exception))


class TestRegistry(SessionTestCase):
    def test_heartbeat_refreshes_expiry_and_hides_stale(self) -> None:
        clock = Clock()
        other = FakeSupabase(OTHER)
        other.rows["foreign"] = {
            "id": "foreign",
            "user_id": OTHER,
            "robot_id": "omnibot",
            "kind": "sim",
            "target": "local",
            "status": "active",
            "expires_at": format_iso(NOW + timedelta(hours=1)),
            "heartbeat_at": format_iso(NOW),
        }
        registry = SessionRegistry(self.fake, token=TOKEN, clock=clock)
        live = registry.upsert(
            robot_id="omnibot",
            kind="sim",
            target="local",
            rosbridge_url="ws://192.0.2.10:9090",
            video_urls=["http://192.0.2.10:8080/stream?topic=/camera/front/image_raw"],
            webrtc_signal_url="",
        )
        self.fake.rows["stale"] = {
            "id": "stale",
            "user_id": USER,
            "robot_id": "omnibot",
            "kind": "sim",
            "target": "cloud",
            "status": "active",
            "heartbeat_at": format_iso(NOW - timedelta(minutes=5)),
            "expires_at": format_iso(NOW - timedelta(seconds=1)),
        }
        self.fake.rows["ended"] = {
            "id": "ended",
            "user_id": USER,
            "robot_id": "omnibot",
            "kind": "real",
            "target": "robot",
            "status": "ended",
            "heartbeat_at": format_iso(NOW),
            "expires_at": format_iso(NOW + timedelta(hours=1)),
        }
        self.fake.rows["go2"] = {
            "id": "go2",
            "user_id": USER,
            "robot_id": "unitree-go2",
            "kind": "sim",
            "target": "local",
            "status": "active",
            "heartbeat_at": format_iso(NOW),
            "expires_at": format_iso(NOW + timedelta(seconds=30)),
        }
        visible = registry.list_visible("omnibot")
        self.assertEqual([row["id"] for row in visible], [live["id"]])
        clock.now = NOW + timedelta(seconds=31)
        self.assertEqual(registry.list_visible("omnibot"), [])
        clock.now = NOW + timedelta(seconds=10)
        beaten = registry.heartbeat(str(live["id"]))
        self.assertEqual(
            beaten["expires_at"], format_iso(clock.now + timedelta(seconds=30))
        )
        clock.now = NOW + timedelta(seconds=31)
        self.assertEqual(
            [row["id"] for row in registry.list_visible("omnibot")],
            [live["id"]],
        )
        foreign = SessionRegistry(other, token=TOKEN).list_visible("omnibot")
        self.assertEqual([row["id"] for row in foreign], ["foreign"])
        self.assertNotIn(live["id"], [row["id"] for row in foreign])
        registry.end("omnibot", "sim", "local")
        self.assertEqual(registry.list_visible("omnibot"), [])
        self.assertIn("go2", self.fake.rows)


class TestLaunch(SessionTestCase):
    def test_compose_file_documents_ohh_204(self) -> None:
        path = default_compose_file()
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("OHH-204", text)
        self.assertIn("gpu", text)
        self.assertNotIn("OHHO_GATEWAY_TOKEN", text)
        self.assertIn("ohho/sim/docker-compose.yml", OHH204_COMPOSE_HINT)

    def test_launch_without_a_running_container(self) -> None:
        def run(argv: list[str]) -> tuple[int, str, str]:
            if "ps" in argv:
                return 0, "", ""
            return 0, "", ""

        result = launch_local_sim(compose_file=default_compose_file(), run=run)
        self.assertFalse(result.started)
        self.assertIn("OHH-204", result.detail)
        self.assertIn("docker", result.command[0])

    def test_launch_when_ps_shows_a_container(self) -> None:
        def run(argv: list[str]) -> tuple[int, str, str]:
            if "ps" in argv:
                return 0, "abc\n", ""
            return 0, "", ""

        result = launch_local_sim(compose_file=default_compose_file(), run=run)
        self.assertTrue(result.started)

    def test_launch_when_docker_is_missing(self) -> None:
        with patch("ohho.sessions.shutil.which", return_value=None):
            result = launch_local_sim(compose_file=default_compose_file())
        self.assertFalse(result.started)
        self.assertIn("OHH-204", result.detail)
        self.assertIn("docker not on PATH", result.detail)

    def test_stop_reports_docker_failure(self) -> None:
        result = stop_local_sim(
            compose_file=default_compose_file(),
            run=lambda _argv: (1, "", "boom"),
        )
        self.assertFalse(result.started)
        self.assertIn("boom", result.detail)


class TestCommands(SessionTestCase):
    def test_local_up_registers_then_down_removes(self) -> None:
        seen: dict[str, Any] = {}
        state_path, _stop = session_paths(self.dir, "sim", "local", "omnibot")

        def on_beat(_count: int) -> None:
            seen["state"] = state_path.read_text(encoding="utf-8")
            seen["visible"] = SessionRegistry(self.fake, token=TOKEN).list_visible(
                "omnibot"
            )

        slept: list[float] = []
        code, stdout, stderr = self.up(
            max_beats=2,
            sleep=slept.append,
            on_beat=on_beat,
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(sum(slept), 20.0)
        self.assertEqual(slept.count(1.0), 20)
        self.assertNotIn(TOKEN, stdout)
        self.assertNotIn(TOKEN, stderr)
        self.assertIn("needs OHH-216", CLOUD_TARGET_MESSAGE)
        self.assertEqual(seen["visible"][0]["token_hash"], TOKEN_HASH)
        self.assertIn(
            "topic=/camera/front/image_raw", seen["visible"][0]["video_urls"][0]
        )
        self.assertNotIn(TOKEN, seen["state"])
        self.assertNotIn(TOKEN, json.dumps(self.fake.calls))
        self.assertEqual(self.fake.rows, {})
        self.assertFalse(state_path.exists())
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(len(self.stops), 1)
        self.assertIn("OHH-204", stdout)

        down_out, down_err = io.StringIO(), io.StringIO()
        with redirect_stdout(down_out), redirect_stderr(down_err):
            down_code = run_sim_down(
                robot_id="omnibot",
                client=self.fake,
                env={},
                state_dir=self.dir,
                stop_container=self.stop,
            )
        self.assertEqual(down_code, 0, down_err.getvalue())
        self.assertIn("removed", down_out.getvalue())
        self.assertTrue(session_paths(self.dir, "sim", "local", "omnibot")[1].exists())

    def test_stop_file_prevents_another_heartbeat(self) -> None:
        _state, stop = session_paths(self.dir, "sim", "local", "omnibot")

        def on_beat(_count: int) -> None:
            stop.write_text("stop\n", encoding="utf-8")

        slept: list[float] = []
        code, _stdout, stderr = self.up(
            max_beats=5, sleep=slept.append, on_beat=on_beat
        )
        self.assertEqual(code, 0, stderr)
        heartbeats = [call for call in self.fake.calls if call[0] == "heartbeat"]
        self.assertEqual(len(heartbeats), 1)
        self.assertEqual(sum(slept), 10.0)
        self.assertEqual(self.fake.rows, {})

    def test_ctrl_c_deletes_the_row(self) -> None:
        def sleep(_seconds: float) -> None:
            raise KeyboardInterrupt

        code, _stdout, stderr = self.up(max_beats=1, sleep=sleep)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.fake.rows, {})
        self.assertTrue(any(call[0] == "delete" for call in self.fake.calls))

    def test_cloud_does_not_register_or_launch(self) -> None:
        code, _stdout, stderr = self.up(target="cloud", robot_id="no-such-robot")
        self.assertEqual(code, 2)
        self.assertIn("needs OHH-216", stderr)
        self.assertIn("No cloud VM was created", stderr)
        self.assertIsInstance(
            CloudTargetUnavailable(CLOUD_TARGET_MESSAGE), CloudTargetUnavailable
        )
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.launches, [])
        down_err = io.StringIO()
        with redirect_stderr(down_err):
            down_code = run_sim_down(target="cloud", client=self.fake, env={})
        self.assertEqual(down_code, 2)
        self.assertIn("needs OHH-216", down_err.getvalue())
        self.assertEqual(self.fake.calls, [])

    def test_missing_supabase_does_not_launch(self) -> None:
        code, stdout, stderr = self.up(client=None, token=TOKEN)
        self.assertEqual(code, 2)
        self.assertIn("OHHO_SUPABASE_URL", stderr)
        self.assertNotIn(TOKEN, stdout + stderr)
        self.assertEqual(self.launches, [])

    def test_unknown_robot(self) -> None:
        code, _stdout, stderr = self.up(robot_id="no-such-robot")
        self.assertEqual(code, 2)
        self.assertIn("unknown robot", stderr)
        self.assertEqual(self.launches, [])

    def test_generated_pairing_code_is_not_stored(self) -> None:
        pairing = self.dir / "pairing-code"
        code, stdout, stderr = self.up(token=None, pairing_file=pairing)
        self.assertEqual(code, 0, stderr)
        token = pairing.read_text(encoding="utf-8").strip()
        self.assertIn(token, stdout)
        upserts = [call[1] for call in self.fake.calls if call[0] == "upsert"]
        self.assertNotIn(token, json.dumps(upserts))
        self.assertEqual(upserts[0]["token_hash"], hash_gateway_token(token))

    def test_robot_announce_uses_gateway_urls_and_skips_docker(self) -> None:
        with patch(
            "ohho.sessions.launch_local_sim", side_effect=AssertionError("docker")
        ):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = run_robot_announce(
                    robot_id="omnibot",
                    host="192.0.2.10",
                    client=self.fake,
                    token=TOKEN,
                    env={},
                    state_dir=self.dir,
                    max_beats=0,
                    sleep=_not_sleep,
                )
        self.assertEqual(code, 0, stderr.getvalue())
        upsert = next(call[1] for call in self.fake.calls if call[0] == "upsert")
        self.assertEqual(upsert["kind"], "real")
        self.assertEqual(upsert["target"], "robot")
        self.assertEqual(upsert["rosbridge_url"], "wss://192.0.2.10:8443/rosbridge")
        self.assertIn("/webrtc/signaling", upsert["webrtc_signal_url"])
        self.assertNotIn(TOKEN, json.dumps(upsert))
        self.assertEqual(self.launches, [])
        self.assertEqual(self.fake.rows, {})

        down_out, down_err = io.StringIO(), io.StringIO()
        with redirect_stdout(down_out), redirect_stderr(down_err):
            down_code = run_robot_announce(
                robot_id="omnibot",
                down=True,
                client=self.fake,
                env={},
                state_dir=self.dir,
                sleep=_not_sleep,
            )
        self.assertEqual(down_code, 0, down_err.getvalue())
        self.assertTrue(any(call[0] == "delete" for call in self.fake.calls))

    def test_down_without_credentials_writes_stop_file(self) -> None:
        code_out, code_err = io.StringIO(), io.StringIO()
        with redirect_stdout(code_out), redirect_stderr(code_err):
            code = run_sim_down(
                robot_id="omnibot",
                client=None,
                env={},
                state_dir=self.dir,
                stop_container=self.stop,
            )
        self.assertEqual(code, 2)
        self.assertIn("OHHO_SUPABASE_URL", code_err.getvalue())
        self.assertTrue(session_paths(self.dir, "sim", "local", "omnibot")[1].exists())
        self.assertEqual(len(self.stops), 1)


class TestCli(SessionTestCase):
    def test_sim_pattern_parser_is_unchanged(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["sim", "--robot", "unitree-go2", "--seconds", "1"])
        self.assertEqual(args.func.__name__, "_cmd_sim")
        self.assertEqual(args.robot, "unitree-go2")
        self.assertEqual(args.seconds, 1)
        self.assertIsNone(args.sim_command)
        up = parser.parse_args(
            ["sim", "up", "--robot", "unitree-go2", "--target", "local"]
        )
        self.assertEqual(up.func.__name__, "_cmd_sim_up")
        self.assertEqual(up.robot, "unitree-go2")
        self.assertEqual(up.target, "local")

    def test_cli_cloud_and_missing_config_do_not_touch_network(self) -> None:
        with (
            patch(
                "ohho.sessions.launch_local_sim", side_effect=AssertionError("docker")
            ),
            patch("urllib.request.urlopen", side_effect=AssertionError("network")),
        ):
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = main(["sim", "up", "--robot", "omnibot", "--target", "cloud"])
            self.assertEqual(code, 2)
            self.assertIn("needs OHH-216", err.getvalue())

            saved = {key: os.environ.get(key) for key in SUPABASE_ENV_KEYS}
            saved["OHHO_GATEWAY_TOKEN"] = os.environ.get("OHHO_GATEWAY_TOKEN")
            saved["OHHO_STATE_DIR"] = os.environ.get("OHHO_STATE_DIR")
            try:
                for key in SUPABASE_ENV_KEYS:
                    os.environ.pop(key, None)
                os.environ["OHHO_GATEWAY_TOKEN"] = TOKEN
                os.environ["OHHO_STATE_DIR"] = str(self.dir)
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    missing = main(
                        [
                            "sim",
                            "up",
                            "--robot",
                            "omnibot",
                            "--target",
                            "local",
                            "--host",
                            "192.0.2.10",
                        ]
                    )
                self.assertEqual(missing, 2)
                self.assertIn("OHHO_SUPABASE_URL", stderr.getvalue())
                self.assertNotIn(TOKEN, stdout.getvalue() + stderr.getvalue())
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_sim_seconds_zero_still_runs(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["sim", "--robot", "omnibot", "--seconds", "0"])
        self.assertEqual(code, 0)


class TestMigrationContract(unittest.TestCase):
    def test_sql_is_owner_only(self) -> None:
        sql = SQL_PATH.read_text(encoding="utf-8")
        lowered = sql.lower()
        for column in (
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
        ):
            self.assertIn(column, sql)
        self.assertNotRegex(lowered, r"\btoken\s+text\b")
        self.assertIn("enable row level security", lowered)
        self.assertIn("force row level security", lowered)
        self.assertIn("auth.uid() = user_id", sql)
        self.assertIn("security_invoker = true", sql)
        self.assertNotIn("using (true)", lowered)
        for action in ("select", "insert", "update", "delete"):
            self.assertIn(f"for {action}", lowered)
        self.assertIn(
            "revoke all on table public.robot_sessions from public, anon", lowered
        )
        self.assertIn("robot_sessions_visible", sql)
        self.assertIn("expires_at > now()", sql)


if __name__ == "__main__":
    unittest.main()
