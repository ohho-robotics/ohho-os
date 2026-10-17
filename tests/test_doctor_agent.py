"""``ohho doctor --agent`` readiness checks. No production console calls."""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from ohho import __version__
from ohho.agent_doctor import (
    _RULES_SKIP,
    Check,
    build_agent_report,
    check_cli,
    check_login,
    check_mcp,
    check_python,
    check_rules,
    check_sim,
    classify_install_method,
    format_agent_report,
    print_agent_report,
)
from ohho.auth import Credentials, Whoami, save_credentials
from ohho.cli import main
from ohho.mcp_connect import (
    CLIENTS,
    client_config_path,
    inspect_ohho_servers,
    resolve_home,
)
from ohho.schema import ConnectionState, Odometry, Telemetry, TransportStatus

_TOKEN = "ohho_test_token_271_phase7_do_not_print"
_FILE_TOKEN = "ohho_file_token_271_phase7_do_not_print"
_ENV_KEYS = (
    "OHHO_TOKEN",
    "OHHO_TRANSPORT",
    "OHHO_ROBOT_TRANSPORT",
    "OHHO_VERIFY_TRANSPORT",
    "OHHO_URI",
    "OHHO_ROBOT_CONFIG",
    "OHHO_CONSOLE_URL",
    "OHHO_SAFETY_CONFIG",
    "OHHO_ARM_HARDWARE",
)


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@contextmanager
def isolated(home: Path, **updates: str) -> Iterator[None]:
    """Point HOME at ``home`` and drop account/transport variables unless set."""
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "OHHO_SAFETY_DIR": str(home / "safety"),
    }
    env.update(updates)
    with patch.dict(os.environ, env, clear=False):
        for key in _ENV_KEYS:
            if key not in updates:
                os.environ.pop(key, None)
        yield


def _home_env(home: Path, **updates: str) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "OHHO_SAFETY_DIR": str(home / "safety"),
    }
    env.update(updates)
    return env


def _assert_secret_absent(test: unittest.TestCase, text: str, *secrets: str) -> None:
    for secret in secrets:
        test.assertNotIn(secret, text)
        test.assertNotIn(secret[:12], text)


class InstallMethodTests(unittest.TestCase):
    def test_classify_pipx_user_venv_editable_and_unknown(self) -> None:
        self.assertEqual(
            classify_install_method(
                "/home/a/.local/pipx/venvs/ohho-os/lib/ohho/__init__.py",
                "/home/a/.local/pipx/venvs/ohho-os",
                "/usr",
                None,
                False,
            ),
            "pipx",
        )
        self.assertEqual(
            classify_install_method(
                "/home/a/.local/lib/python3.12/site-packages/ohho/__init__.py",
                "/usr",
                "/usr",
                "/home/a/.local/lib/python3.12/site-packages",
                False,
            ),
            "pip --user",
        )
        self.assertEqual(
            classify_install_method(
                "/work/venv/lib/python3.12/site-packages/ohho/__init__.py",
                "/work/venv",
                "/usr",
                "/home/a/.local/lib/python3.12/site-packages",
                False,
            ),
            "venv",
        )
        self.assertEqual(
            classify_install_method(
                "/src/ohho-os/ohho/__init__.py",
                "/src/ohho-os/.venv",
                "/usr",
                None,
                True,
            ),
            "editable source",
        )
        self.assertEqual(
            classify_install_method(
                "/usr/lib/python3.12/site-packages/ohho/__init__.py",
                "/usr",
                "/usr",
                "/home/a/.local/lib/python3.12/site-packages",
                False,
            ),
            "unknown",
        )

    def test_pipx_wins_over_editable(self) -> None:
        self.assertEqual(
            classify_install_method(
                "/pipx/venvs/ohho/lib/ohho/__init__.py",
                "/usr",
                "/usr",
                None,
                True,
            ),
            "pipx",
        )


class CliCheckTests(unittest.TestCase):
    def test_pass_reports_version_and_install(self) -> None:
        check = check_cli()
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.extra["version"], __version__)
        self.assertIn(
            check.extra["install"],
            {
                "pipx",
                "pip --user",
                "venv",
                "editable source",
                "unknown",
            },
        )
        self.assertIn(__version__, check.detail)

    def test_fail_when_version_is_missing(self) -> None:
        check = check_cli(version="")
        self.assertEqual(check.status, "fail")
        self.assertIn("missing", check.detail)

    def test_skip_when_install_layout_cannot_be_read(self) -> None:
        with patch(
            "ohho.agent_doctor.detect_install_method",
            side_effect=OSError("unread"),
        ):
            check = check_cli()
        self.assertEqual(check.status, "skip")
        self.assertIn("could not inspect", check.detail)


class PythonCheckTests(unittest.TestCase):
    def test_pass_on_this_interpreter(self) -> None:
        check = check_python()
        self.assertEqual(check.status, "pass")
        self.assertGreaterEqual(
            tuple(int(p) for p in check.detail.split(".")[:2]), (3, 10)
        )

    def test_fail_below_3_10(self) -> None:
        check = check_python(version_info=(3, 9, 18))
        self.assertEqual(check.status, "fail")
        self.assertIn("3.9.18", check.detail)
        self.assertIn("below 3.10", check.detail)

    def test_skip_when_version_cannot_be_read(self) -> None:
        check = check_python(version_info=object())
        self.assertEqual(check.status, "skip")
        self.assertIn("could not read", check.detail)


class LoginCheckTests(unittest.TestCase):
    def test_skip_when_signed_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            check = check_login(_home_env(Path(tmp)))
        self.assertEqual(check.status, "skip")
        self.assertIn("not signed in", check.detail)
        self.assertIsNone(check.extra["email"])
        self.assertIsNone(check.extra["source"])

    def test_pass_offline_from_env_without_email_or_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(Path(tmp), OHHO_TOKEN=_TOKEN)
            check = check_login(env, online=False)
            text = format_agent_report(
                build_agent_report(environ=env, sim_connect=_connected),
                as_json=False,
            )
            body = format_agent_report(
                build_agent_report(environ=env, sim_connect=_connected),
                as_json=True,
            )
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.extra["source"], "env")
        self.assertIsNone(check.extra["email"])
        self.assertNotIn(_TOKEN, check.detail)
        _assert_secret_absent(self, text + body, _TOKEN)

    def test_pass_offline_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = _home_env(home)
            path = home / ".ohho" / "credentials.json"
            save_credentials(
                path,
                Credentials(
                    token=_FILE_TOKEN,
                    token_type="Bearer",
                    scope="mcp",
                    console_url="https://console.ohho-robotics.com",
                ),
            )
            check = check_login(env, online=False)
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.extra["source"], "file")
        self.assertIsNone(check.extra["email"])
        self.assertNotIn(_FILE_TOKEN, check.detail)

    def test_fail_when_credentials_are_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / ".ohho" / "credentials.json"
            path.parent.mkdir(parents=True)
            path.write_text("{not-json " + _FILE_TOKEN, encoding="utf-8")
            check = check_login(_home_env(home))
            rendered = check.detail + json.dumps(check.as_dict())
        self.assertEqual(check.status, "fail")
        self.assertEqual(check.detail, "credentials unreadable")
        _assert_secret_absent(self, rendered, _FILE_TOKEN)

    def test_online_pass_reports_email_and_source_only(self) -> None:
        seen: list[dict] = []

        def fake_whoami(**kwargs: object) -> Whoami:
            seen.append(kwargs)
            return Whoami(
                signed_in=True,
                email="ada@example.com",
                scopes=("mcp",),
                user_id="user-1",
            )

        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(
                Path(tmp),
                OHHO_TOKEN=_TOKEN,
                OHHO_CONSOLE_URL="https://preview.example",
            )
            check = check_login(env, online=True, whoami_fn=fake_whoami)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["environ"], env)
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.extra["email"], "ada@example.com")
        self.assertEqual(check.extra["source"], "env")
        self.assertNotIn("user_id", check.extra)
        self.assertNotIn("scopes", check.extra)
        _assert_secret_absent(self, json.dumps(check.as_dict()), _TOKEN)

    def test_online_drops_email_that_contains_the_token(self) -> None:
        def fake_whoami(**kwargs: object) -> Whoami:
            return Whoami(
                signed_in=True,
                email=_TOKEN,
                scopes=(_TOKEN,),
                user_id=_TOKEN,
            )

        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(Path(tmp), OHHO_TOKEN=_TOKEN)
            check = check_login(env, online=True, whoami_fn=fake_whoami)
            report = build_agent_report(
                online=True,
                environ=env,
                whoami_fn=fake_whoami,
                sim_connect=_connected,
            )
            rendered = format_agent_report(report, as_json=True) + format_agent_report(
                report, as_json=False
            )
        self.assertEqual(check.status, "pass")
        self.assertIsNone(check.extra["email"])
        self.assertIn("unknown", check.detail)
        _assert_secret_absent(self, rendered, _TOKEN)

    def test_online_invalid_token_is_skip(self) -> None:
        def fake_whoami(**kwargs: object) -> Whoami:
            return Whoami(signed_in=False, error="invalid_token")

        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(Path(tmp), OHHO_TOKEN=_TOKEN)
            check = check_login(env, online=True, whoami_fn=fake_whoami)
        self.assertEqual(check.status, "skip")
        self.assertNotIn(_TOKEN, check.detail)

    def test_online_console_unavailable_is_fail(self) -> None:
        def fake_whoami(**kwargs: object) -> Whoami:
            return Whoami(
                signed_in=False,
                error="console_unavailable",
                detail=f"nope {_TOKEN}",
            )

        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(Path(tmp), OHHO_TOKEN=_TOKEN)
            check = check_login(env, online=True, whoami_fn=fake_whoami)
        self.assertEqual(check.status, "fail")
        self.assertEqual(check.detail, "console unavailable")
        self.assertNotIn(_TOKEN, json.dumps(check.as_dict()))

    def test_offline_does_not_call_whoami(self) -> None:
        def fake_whoami(**kwargs: object) -> Whoami:
            raise AssertionError("network")

        with tempfile.TemporaryDirectory() as tmp:
            env = _home_env(Path(tmp), OHHO_TOKEN=_TOKEN)
            check = check_login(env, online=False, whoami_fn=fake_whoami)
        self.assertEqual(check.status, "pass")


class RulesCheckTests(unittest.TestCase):
    def test_skip_when_hook_is_missing(self) -> None:
        check = check_rules()
        self.assertEqual(check.status, "skip")
        self.assertEqual(check.detail, _RULES_SKIP)

    def test_skip_when_loader_raises_import_error(self) -> None:
        def load() -> object:
            raise ImportError("ohho rules install not available")

        check = check_rules(load)
        self.assertEqual(check.status, "skip")
        self.assertEqual(check.detail, _RULES_SKIP)

    def test_skip_when_installed_clients_is_absent(self) -> None:
        import types

        check = check_rules(lambda: types.ModuleType("ohho.agent_rules"))
        self.assertEqual(check.status, "skip")
        self.assertEqual(check.detail, _RULES_SKIP)

    def test_pass_when_every_client_has_a_path(self) -> None:
        import types

        module = types.ModuleType("ohho.agent_rules")
        module.installed_clients = lambda: {  # type: ignore[attr-defined]
            "cursor": "/home/a/.cursor/rules/ohho.md",
            "codex": "/home/a/.codex/ohho.md",
        }
        check = check_rules(lambda: module)
        self.assertEqual(check.status, "pass")
        self.assertIn("cursor", check.detail)
        self.assertEqual(
            check.extra["clients"]["cursor"],
            "/home/a/.cursor/rules/ohho.md",
        )

    def test_fail_when_no_client_has_rules(self) -> None:
        import types

        module = types.ModuleType("ohho.agent_rules")
        module.installed_clients = lambda: {  # type: ignore[attr-defined]
            "cursor": None,
            "codex": None,
        }
        check = check_rules(lambda: module)
        self.assertEqual(check.status, "fail")
        self.assertIn("cursor", check.detail)

    def test_fail_when_hook_raises(self) -> None:
        import types

        module = types.ModuleType("ohho.agent_rules")

        def boom() -> dict[str, str]:
            raise RuntimeError("disk")

        module.installed_clients = boom  # type: ignore[attr-defined]
        check = check_rules(lambda: module)
        self.assertEqual(check.status, "fail")
        self.assertNotIn("disk", check.detail)


class McpCheckTests(unittest.TestCase):
    def test_skip_without_a_home_directory(self) -> None:
        check = check_mcp({}, platform_name="linux")
        self.assertEqual(check.status, "skip")
        self.assertIn("home", check.detail)

    def test_fail_when_no_client_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            check = check_mcp(_home_env(Path(tmp)), platform_name="linux")
        self.assertEqual(check.status, "fail")
        self.assertIn("no ohho MCP server", check.detail)
        self.assertEqual(set(check.extra["clients"]), {spec.id for spec in CLIENTS})
        self.assertTrue(
            all(not info["configured"] for info in check.extra["clients"].values())
        )

    def test_pass_reads_each_client_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = _home_env(home)
            root = resolve_home(env, "linux")
            cursor = client_config_path(
                "cursor", home=root, environ=env, platform_name="linux"
            )
            cursor.parent.mkdir(parents=True)
            cursor.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "ohho": {"command": "ohho", "args": ["mcp", "serve"]}
                        }
                    }
                ),
                encoding="utf-8",
            )
            codex = client_config_path(
                "codex", home=root, environ=env, platform_name="linux"
            )
            codex.parent.mkdir(parents=True)
            codex.write_text(
                '[mcp_servers.ohho]\ncommand = "ohho"\nargs = ["mcp", "serve"]\n',
                encoding="utf-8",
            )
            opencode = client_config_path(
                "opencode", home=root, environ=env, platform_name="linux"
            )
            opencode.parent.mkdir(parents=True)
            opencode.write_text(
                json.dumps(
                    {
                        "mcp": {
                            "ohho": {
                                "type": "local",
                                "command": ["ohho", "mcp", "serve"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            broken = client_config_path(
                "gemini", home=root, environ=env, platform_name="linux"
            )
            broken.parent.mkdir(parents=True)
            broken.write_text("{", encoding="utf-8")
            originals = {
                path: path.read_bytes() for path in (cursor, codex, opencode, broken)
            }
            listing = _snapshot(home)
            found = inspect_ohho_servers(environ=env, platform_name="linux")
            check = check_mcp(env, platform_name="linux")
            after = _snapshot(home)
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)
        self.assertEqual(listing, after)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertTrue(found["cursor"]["configured"])
        self.assertTrue(found["codex"]["configured"])
        self.assertTrue(found["opencode"]["configured"])
        self.assertFalse(found["gemini"]["configured"])
        self.assertEqual(found["gemini"]["detail"], "config unreadable")
        for spec in CLIENTS:
            self.assertEqual(
                found[spec.id]["path"],
                str(
                    client_config_path(
                        spec.id, home=root, environ=env, platform_name="linux"
                    )
                ),
            )
        self.assertEqual(check.status, "pass")
        self.assertIn("cursor", check.detail)


def _snapshot(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


class _FakeBot:
    def __init__(
        self, state: ConnectionState, odom: bool = True, boom: bool = False
    ) -> None:
        self._state = state
        self._odom = odom
        self._boom = boom
        self.disconnected = False

    def status(self) -> TransportStatus:
        return TransportStatus(protocol="simulated", state=self._state, label="fake")

    def telemetry(self) -> Telemetry:
        if self._boom:
            raise RuntimeError("telemetry down")
        if not self._odom:
            return Telemetry()
        return Telemetry(odom=Odometry(x=0.1))

    def disconnect(self) -> None:
        self.disconnected = True


def _connected(robot_id: str) -> _FakeBot:
    return _FakeBot(ConnectionState.CONNECTED)


class SimCheckTests(unittest.TestCase):
    def test_pass_reads_sim_telemetry_without_motion(self) -> None:
        from ohho.adapters.sim import SimTransport

        calls: list[object] = []
        original = SimTransport.send_velocity

        def spy(self: SimTransport, vel: object) -> None:
            calls.append(vel)
            original(self, vel)  # type: ignore[arg-type]

        with patch.object(SimTransport, "send_velocity", spy):
            check = check_sim("omnibot")
        self.assertEqual(check.status, "pass")
        self.assertEqual(check.extra["robot"], "omnibot")
        self.assertEqual(calls, [])

    def test_fail_when_connect_raises(self) -> None:
        def connect(robot_id: str) -> _FakeBot:
            raise RuntimeError("down")

        check = check_sim(connect=connect)  # type: ignore[arg-type]
        self.assertEqual(check.status, "fail")
        self.assertIn("connect failed", check.detail)

    def test_fail_when_telemetry_is_missing(self) -> None:
        bot = _FakeBot(ConnectionState.CONNECTED, odom=False)
        check = check_sim(connect=lambda _robot: bot)  # type: ignore[arg-type]
        self.assertEqual(check.status, "fail")
        self.assertIn("missing", check.detail)
        self.assertTrue(bot.disconnected)

    def test_skip_when_sim_is_not_connected(self) -> None:
        bot = _FakeBot(ConnectionState.IDLE)
        check = check_sim(connect=lambda _robot: bot)  # type: ignore[arg-type]
        self.assertEqual(check.status, "skip")
        self.assertIn("not connected", check.detail)
        self.assertTrue(bot.disconnected)


class ReportTests(unittest.TestCase):
    def test_json_schema_and_exit_code_when_mcp_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            stream = io.StringIO()
            code = print_agent_report(
                as_json=True,
                environ=_home_env(home),
                platform_name="linux",
                stdout=stream,
                sim_connect=_connected,  # type: ignore[arg-type]
            )
            text = stream.getvalue()
        self.assertEqual(code, 1)
        body = json.loads(text)
        self.assertEqual(list(body), ["ok", "version", "checks"])
        self.assertIs(body["ok"], False)
        self.assertEqual(body["version"], __version__)
        ids = [item["id"] for item in body["checks"]]
        self.assertEqual(ids, ["cli", "python", "login", "rules", "mcp", "sim"])
        for item in body["checks"]:
            self.assertIn(item["status"], {"pass", "fail", "skip"})
            self.assertIsInstance(item["detail"], str)
        by_id = {item["id"]: item for item in body["checks"]}
        self.assertEqual(by_id["mcp"]["status"], "fail")
        self.assertEqual(by_id["login"]["status"], "skip")
        self.assertEqual(by_id["rules"]["status"], "skip")
        self.assertEqual(by_id["rules"]["detail"], _RULES_SKIP)
        self.assertEqual(by_id["sim"]["status"], "pass")

    def test_exit_zero_when_nothing_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = _home_env(home)
            root = resolve_home(env, "linux")
            cursor = client_config_path(
                "cursor", home=root, environ=env, platform_name="linux"
            )
            cursor.parent.mkdir(parents=True)
            cursor.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "ohho": {"command": "ohho", "args": ["mcp", "serve"]}
                        }
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()
            code = print_agent_report(
                as_json=False,
                environ=env,
                platform_name="linux",
                stdout=stream,
                sim_connect=_connected,  # type: ignore[arg-type]
            )
        self.assertEqual(code, 0)
        self.assertIn("agent readiness", stream.getvalue())
        self.assertIn("pass", stream.getvalue())

    def test_token_never_appears_in_text_json_or_logs(self) -> None:
        handler = _LogCapture()
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                path = home / ".ohho" / "credentials.json"
                save_credentials(
                    path,
                    Credentials(
                        token=_FILE_TOKEN,
                        token_type="Bearer",
                        scope="mcp",
                        console_url="https://console.ohho-robotics.com",
                    ),
                )
                env = _home_env(home, OHHO_TOKEN=_TOKEN)

                def fake_whoami(**kwargs: object) -> Whoami:
                    return Whoami(signed_in=True, email=f"ada+{_TOKEN}@example.com")

                report = build_agent_report(
                    online=True,
                    environ=env,
                    platform_name="linux",
                    whoami_fn=fake_whoami,
                    sim_connect=_connected,  # type: ignore[arg-type]
                )
                rendered = format_agent_report(
                    report, as_json=True
                ) + format_agent_report(report, as_json=False)
        finally:
            root.removeHandler(handler)
        _assert_secret_absent(self, rendered, _TOKEN, _FILE_TOKEN)
        _assert_secret_absent(self, "\n".join(handler.lines), _TOKEN, _FILE_TOKEN)
        self.assertNotIn("token", json.dumps(report.as_dict()))


class CliDoctorTests(unittest.TestCase):
    def test_plain_doctor_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with isolated(Path(tmp), OHHO_TOKEN=_TOKEN):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch("sys.stdout", stdout),
                    patch("sys.stderr", stderr),
                ):
                    plain = main(["doctor"])
                    flagged = main(["doctor", "--json"])
        text = stdout.getvalue()
        self.assertEqual(plain, 0)
        self.assertEqual(flagged, 0)
        self.assertIn("runtimes", text)
        self.assertIn("agent brain", text)
        self.assertNotIn("agent readiness", text)
        self.assertNotIn('"checks"', text)
        _assert_secret_absent(self, text + stderr.getvalue(), _TOKEN)

    def test_agent_json_and_token_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with isolated(home, OHHO_TOKEN=_TOKEN):
                path = home / ".ohho" / "credentials.json"
                save_credentials(
                    path,
                    Credentials(
                        token=_FILE_TOKEN,
                        token_type="Bearer",
                        scope="mcp",
                        console_url="https://console.ohho-robotics.com",
                    ),
                )
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    code = main(["doctor", "--agent", "--json"])
        text = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(code, 1)
        body = json.loads(stdout.getvalue())
        self.assertIn("checks", body)
        _assert_secret_absent(self, text, _TOKEN, _FILE_TOKEN)

    def test_online_uses_console_url_and_offline_does_not(self) -> None:
        urls: list[str] = []

        class _Response:
            status = 200
            headers = {"content-type": "application/json"}

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def read(self) -> bytes:
                return b'{"email":"ada@example.com","scopes":["mcp"],"user_id":"u1"}'

            def getcode(self) -> int:
                return 200

        def opener(request: object, timeout: float = 0) -> _Response:
            urls.append(getattr(request, "full_url", ""))
            return _Response()

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with isolated(
                home,
                OHHO_TOKEN=_TOKEN,
                OHHO_CONSOLE_URL="https://preview.example",
            ):
                with patch(
                    "ohho.auth._OPENER.open", side_effect=AssertionError("offline")
                ):
                    stdout = io.StringIO()
                    with patch("sys.stdout", stdout):
                        offline = main(["doctor", "--agent", "--json"])
                self.assertIn("signed in", stdout.getvalue())
                with patch("ohho.auth._OPENER.open", opener):
                    stdout = io.StringIO()
                    with patch("sys.stdout", stdout):
                        online = main(["doctor", "--agent", "--online", "--json"])
                rendered = stdout.getvalue()
        self.assertEqual(offline, 1)
        self.assertEqual(online, 1)
        self.assertEqual(urls, ["https://preview.example/api/cli/whoami"])
        self.assertIn("ada@example.com", rendered)
        _assert_secret_absent(self, rendered, _TOKEN)


class _SchemaSmoke(unittest.TestCase):
    def test_check_dict_order(self) -> None:
        body = Check(
            "cli", "pass", "ok", extra={"install": "venv", "version": "1.2.3"}
        ).as_dict()
        self.assertEqual(list(body)[:3], ["id", "status", "detail"])


if __name__ == "__main__":
    unittest.main()
