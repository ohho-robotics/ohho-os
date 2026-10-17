"""Golden-file tests for ``ohho rules install``.

Every install passes a temporary HOME / USERPROFILE / APPDATA / XDG_CONFIG_HOME.
These tests must not read or write the developer machine's real client files.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from unittest.mock import patch

from ohho.agent_rules import (
    CLIENTS,
    RULES_VERSION,
    install_rules,
    installed_clients,
    list_rules,
    render_new_file,
    rules_path,
    rules_source,
    show_rules,
)
from ohho.cli import main
from ohho.mcp_connect import CLIENTS as MCP_CLIENTS

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rules"
FIXED = datetime(2026, 10, 17, 16, 0, tzinfo=timezone.utc)

_SAFETY = (
    "sim://",
    "OHHO_ARM_HARDWARE",
    "--arm",
    "release-stop",
    "release_stop",
    "Emergency stop is always allowed",
    "SafetyGate",
    "Do not retry around it",
    "anything physical",
    "state you have verified",
    "ohho doctor --agent",
    "ohho verify --sim",
    "ohho mcp serve",
    "connect",
    "drive",
    "telemetry",
    "nav_goto",
    "look",
    "https://ohho-robotics.com/onboard.md",
)


def _snapshot(root: Path) -> List[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.exists()
    )


def _golden(client: str, name: str) -> bytes:
    return (FIXTURES / client / name).read_bytes()


# pathlib objects are slotted, so the TemporaryDirectory is kept here.
_LIVE: List[tempfile.TemporaryDirectory[str]] = []


def _fresh() -> tuple[Path, Dict[str, str]]:
    tmp = tempfile.TemporaryDirectory()
    _LIVE.append(tmp)
    home = Path(tmp.name)
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "XDG_CONFIG_HOME": str(home / ".config"),
    }
    return home, env


def _path(
    env: Dict[str, str],
    client: str,
    *,
    project: Optional[Path] = None,
    platform_name: str = "linux",
) -> Path:
    home = Path(env["USERPROFILE"])
    return rules_path(
        client,
        home=None if project is not None else home,
        environ=env,
        platform_name=platform_name,
        project=project,
    )


def _install(
    env: Dict[str, str],
    clients: Optional[Sequence[str]],
    **kwargs: object,
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    platform = kwargs.pop("platform_name", "linux")
    code = install_rules(
        clients,
        environ=env,
        platform_name=str(platform),
        stdout=out,
        stderr=err,
        now=lambda: FIXED,
        **kwargs,  # type: ignore[arg-type]
    )
    return code, out.getvalue(), err.getvalue()


class RulesSourceTests(unittest.TestCase):
    def test_source_contains_the_safety_rules(self) -> None:
        body = rules_source()
        self.assertTrue(body.endswith("\n"))
        self.assertNotIn("\r", body)
        for phrase in _SAFETY:
            self.assertIn(phrase, body)
        lowered = body.lower()
        self.assertNotIn("mcp server refuses", lowered)
        self.assertNotIn("mcp server blocks", lowered)
        self.assertNotIn("server enforces", lowered)
        self.assertNotIn("<!-- ohho-agent-rules:end", body)

    def test_every_render_contains_the_source_body(self) -> None:
        source = rules_source()
        self.assertTrue(source.endswith("\n"))
        for spec in CLIENTS:
            for project in (False, True):
                with self.subTest(client=spec.id, project=project):
                    rendered = render_new_file(spec.id, project=project)
                    self.assertIn(source, rendered)
                    self.assertIn(
                        f"<!-- ohho-agent-rules:start {RULES_VERSION} -->", rendered
                    )
                    self.assertIn("<!-- ohho-agent-rules:end -->", rendered)

    def test_cursor_and_windsurf_shapes(self) -> None:
        cursor = render_new_file("cursor")
        self.assertTrue(cursor.startswith("---\n"))
        self.assertIn("alwaysApply: true", cursor)
        windsurf = render_new_file("windsurf", project=False)
        self.assertFalse(windsurf.startswith("---"))
        self.assertLess(len(windsurf), 6000)
        project = render_new_file("windsurf", project=True)
        self.assertIn("trigger: always_on", project)
        self.assertLess(len(project), 12000)

    def test_client_ids_match_mcp_connect(self) -> None:
        self.assertEqual(
            [spec.id for spec in CLIENTS], [spec.id for spec in MCP_CLIENTS]
        )


class RulesPathTests(unittest.TestCase):
    def test_posix_user_and_project_paths(self) -> None:
        home = Path("/home/ada")
        env = {
            "HOME": "/home/ada",
            "XDG_CONFIG_HOME": "/xdg",
            "CLAUDE_CONFIG_DIR": "/claude",
            "CODEX_HOME": "/codex",
        }
        expect = {
            "claude-code": Path("/claude/CLAUDE.md"),
            "codex": Path("/codex/AGENTS.md"),
            "cursor": home / ".cursor" / "rules" / "ohho.mdc",
            "vscode": home / ".copilot" / "copilot-instructions.md",
            "gemini": home / ".gemini" / "GEMINI.md",
            "windsurf": home / ".codeium" / "windsurf" / "memories" / "global_rules.md",
            "opencode": Path("/xdg/opencode/AGENTS.md"),
        }
        for client, path in expect.items():
            self.assertEqual(
                rules_path(client, home=home, environ=env, platform_name="linux"),
                path,
                client,
            )
        plain = {"HOME": "/home/ada"}
        self.assertEqual(
            rules_path("claude-code", home=home, environ=plain, platform_name="linux"),
            home / ".claude" / "CLAUDE.md",
        )
        self.assertEqual(
            rules_path("opencode", home=home, environ=plain, platform_name="linux"),
            home / ".config" / "opencode" / "AGENTS.md",
        )
        project = Path("/work")
        project_expect = {
            "claude-code": project / "CLAUDE.md",
            "codex": project / "AGENTS.md",
            "cursor": project / ".cursor" / "rules" / "ohho.mdc",
            "vscode": project / ".github" / "copilot-instructions.md",
            "gemini": project / "GEMINI.md",
            "windsurf": project / ".windsurf" / "rules" / "ohho.md",
            "opencode": project / "AGENTS.md",
        }
        for client, path in project_expect.items():
            self.assertEqual(
                rules_path(
                    client,
                    home=home,
                    environ=plain,
                    platform_name="linux",
                    project=project,
                ),
                path,
                client,
            )

    def test_windows_paths_use_userprofile_not_appdata(self) -> None:
        home = Path(r"C:\Users\ada")
        env = {
            "USERPROFILE": r"C:\Users\ada",
            "APPDATA": r"C:\Users\ada\AppData\Roaming",
        }
        vscode = rules_path("vscode", home=home, environ=env, platform_name="win32")
        self.assertEqual(vscode, home / ".copilot" / "copilot-instructions.md")
        self.assertNotEqual(
            vscode, Path(env["APPDATA"]) / "Code" / "User" / "copilot-instructions.md"
        )
        self.assertEqual(
            rules_path("opencode", home=home, environ=env, platform_name="win32"),
            home / ".config" / "opencode" / "AGENTS.md",
        )
        self.assertEqual(
            rules_path("cursor", home=home, environ=env, platform_name="win32"),
            home / ".cursor" / "rules" / "ohho.mdc",
        )


class RulesGoldenTests(unittest.TestCase):
    def test_each_client_matches_the_golden_files(self) -> None:
        source = rules_source().encode("utf-8")
        unrelated = (FIXTURES / "unrelated_in.md").read_bytes()
        old = (FIXTURES / "old_in.md").read_bytes()
        for spec in CLIENTS:
            client = spec.id
            with self.subTest(client=client, case="created"):
                _home, env = _fresh()
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                path = _path(env, client)
                data = path.read_bytes()
                self.assertEqual(data, _golden(client, "created.md"))
                self.assertIn(source, data)
                self.assertEqual(list(path.parent.glob(path.name + ".bak.*")), [])
                before = _snapshot(Path(env["HOME"]))
                mtime = path.stat().st_mtime_ns
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(path.stat().st_mtime_ns, mtime)
                self.assertEqual(_snapshot(Path(env["HOME"])), before)

            with self.subTest(client=client, case="empty"):
                _home, env = _fresh()
                path = _path(env, client)
                path.parent.mkdir(parents=True)
                path.write_bytes(b"")
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                self.assertEqual(path.read_bytes(), _golden(client, "empty.md"))
                self.assertIn(source, path.read_bytes())
                backup = path.with_name(path.name + ".bak.20261017T160000Z")
                self.assertEqual(backup.read_bytes(), b"")

            with self.subTest(client=client, case="unrelated"):
                _home, env = _fresh()
                path = _path(env, client)
                path.parent.mkdir(parents=True)
                path.write_bytes(unrelated)
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                merged = path.read_bytes()
                self.assertEqual(merged, _golden(client, "unrelated.md"))
                self.assertTrue(merged.startswith(unrelated))
                self.assertIn(source, merged)
                backup = path.with_name(path.name + ".bak.20261017T160000Z")
                self.assertEqual(backup.read_bytes(), unrelated)
                code, _out, err = _install(env, [client], remove=True)
                self.assertEqual(code, 0, err)
                self.assertEqual(path.read_bytes(), unrelated)

            with self.subTest(client=client, case="old"):
                _home, env = _fresh()
                path = _path(env, client)
                path.parent.mkdir(parents=True)
                path.write_bytes(old)
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                updated = path.read_bytes()
                self.assertEqual(updated, _golden(client, "old.md"))
                self.assertTrue(updated.startswith(b"# keep\n\n"))
                self.assertTrue(updated.endswith(b"\n# tail\n"))
                self.assertIn(source, updated)
                self.assertNotIn(b"start v0", updated)
                backup = path.with_name(path.name + ".bak.20261017T160000Z")
                self.assertEqual(backup.read_bytes(), old)

            with self.subTest(client=client, case="remove-created"):
                _home, env = _fresh()
                code, _out, err = _install(env, [client])
                self.assertEqual(code, 0, err)
                path = _path(env, client)
                code, _out, err = _install(env, [client], remove=True)
                self.assertEqual(code, 0, err)
                self.assertFalse(path.exists())
                backups = list(path.parent.glob(path.name + ".bak.*"))
                self.assertEqual(len(backups), 1)
                self.assertIn(source, backups[0].read_bytes())

            with self.subTest(client=client, case="dry-run"):
                _home, env = _fresh()
                home = Path(env["HOME"])
                before = _snapshot(home)
                code, out, err = _install(env, [client], dry_run=True)
                self.assertEqual(code, 0, err)
                self.assertEqual(_snapshot(home), before)
                self.assertIn("dry-run", out)
                self.assertIn("created", out)
                path = _path(env, client)
                path.parent.mkdir(parents=True)
                path.write_bytes(unrelated)
                stamped = path.stat().st_mtime_ns
                code, out, err = _install(env, [client], dry_run=True)
                self.assertEqual(code, 0, err)
                self.assertEqual(path.read_bytes(), unrelated)
                self.assertEqual(path.stat().st_mtime_ns, stamped)
                self.assertEqual(list(path.parent.glob("*.bak.*")), [])
                self.assertIn("dry-run", out)

    def test_crlf_outside_the_block_is_preserved(self) -> None:
        _home, env = _fresh()
        original = "keep-me-Ω\r\n# end\r\n".encode("utf-8")
        path = _path(env, "codex")
        path.parent.mkdir(parents=True)
        path.write_bytes(original)
        code, _out, err = _install(env, ["codex"])
        self.assertEqual(code, 0, err)
        merged = path.read_bytes()
        self.assertTrue(merged.startswith(original))
        self.assertIn(rules_source().encode("utf-8"), merged)
        code, _out, err = _install(env, ["codex"], remove=True)
        self.assertEqual(code, 0, err)
        self.assertEqual(path.read_bytes(), original)

    def test_project_files_contain_the_source(self) -> None:
        _home, env = _fresh()
        project = Path(env["HOME"]) / "proj"
        code, _out, err = _install(env, None, all_clients=True, project=str(project))
        self.assertEqual(code, 0, err)
        source = rules_source()
        for spec in CLIENTS:
            path = _path(env, spec.id, project=project)
            self.assertTrue(path.is_file(), spec.id)
            text = path.read_text(encoding="utf-8")
            self.assertIn(source, text)
            if spec.id == "windsurf":
                self.assertEqual(
                    text.encode("utf-8"), _golden("windsurf", "project_created.md")
                )
            else:
                self.assertEqual(text.encode("utf-8"), _golden(spec.id, "created.md"))
        self.assertFalse((Path(env["HOME"]) / ".claude").exists())
        self.assertFalse((Path(env["HOME"]) / ".cursor").exists())


class RulesBehaviorTests(unittest.TestCase):
    def test_aliases_all_json_and_refusals(self) -> None:
        _home, env = _fresh()
        code, _out, err = _install(
            env, ["claude", "vs-code", "gemini-cli", "open-code"]
        )
        self.assertEqual(code, 0, err)
        for client in ("claude-code", "vscode", "gemini", "opencode"):
            self.assertTrue(_path(env, client).is_file(), client)

        home, env = _fresh()
        code, out, err = _install(env, None, all_clients=True, as_json=True)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        body = json.loads(out)
        self.assertEqual(list(body), ["dry_run", "remove", "version", "results"])
        self.assertIs(body["dry_run"], False)
        self.assertIs(body["remove"], False)
        self.assertEqual(body["version"], RULES_VERSION)
        self.assertEqual(
            [item["client"] for item in body["results"]],
            [spec.id for spec in CLIENTS],
        )
        for item in body["results"]:
            self.assertEqual(
                list(item), ["client", "action", "path", "backup", "changed"]
            )
            self.assertEqual(item["action"], "created")
            self.assertIs(item["changed"], True)
            self.assertIsNone(item["backup"])
            self.assertIn(
                rules_source(), Path(item["path"]).read_text(encoding="utf-8")
            )

        code, _out, err = _install(env, ["not-a-client"])
        self.assertEqual(code, 2)
        self.assertIn("unknown client", err)
        code, _out, err = _install(env, [])
        self.assertEqual(code, 2)
        self.assertIn("--client", err)
        code, _out, err = _install(env, ["not-a-client"], all_clients=True)
        self.assertEqual(code, 2)
        self.assertIn("unknown client", err)

    def test_non_utf8_and_directory_are_left_untouched(self) -> None:
        _home, env = _fresh()
        path = _path(env, "gemini")
        path.parent.mkdir(parents=True)
        payload = b"\xff\xfe not utf-8"
        path.write_bytes(payload)
        code, _out, err = _install(env, ["gemini"])
        self.assertEqual(code, 1)
        self.assertIn("UTF-8", err)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(list(path.parent.glob("*.bak.*")), [])

        home, env = _fresh()
        path = _path(env, "cursor")
        path.mkdir(parents=True)
        code, _out, err = _install(env, ["cursor"])
        self.assertEqual(code, 1)
        self.assertIn("not a file", err)
        self.assertTrue(path.is_dir())

    def test_block_without_created_marker_is_not_deleted(self) -> None:
        _home, env = _fresh()
        code, _out, err = _install(env, ["opencode"])
        self.assertEqual(code, 0, err)
        path = _path(env, "opencode")
        text = path.read_text(encoding="utf-8").replace(
            "<!-- ohho-agent-rules:created -->\n", ""
        )
        path.write_text(text, encoding="utf-8", newline="\n")
        code, _out, err = _install(env, ["opencode"], remove=True)
        self.assertEqual(code, 0, err)
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), b"")

    def test_windows_install_uses_the_temp_profile(self) -> None:
        _home, env = _fresh()
        code, _out, err = _install(env, ["vscode", "opencode"], platform_name="win32")
        self.assertEqual(code, 0, err)
        home = Path(env["USERPROFILE"])
        self.assertTrue((home / ".copilot" / "copilot-instructions.md").is_file())
        self.assertTrue((home / ".config" / "opencode" / "AGENTS.md").is_file())
        self.assertFalse((home / "AppData").exists())

    def test_list_show_and_installed_clients(self) -> None:
        _home, env = _fresh()
        out, err = io.StringIO(), io.StringIO()
        code = list_rules(environ=env, platform_name="linux", stdout=out, stderr=err)
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("not installed", text)
        self.assertIn("no file to merge", text)
        for spec in CLIENTS:
            self.assertIn(spec.id, text)

        code, _out, err = _install(env, ["cursor"])
        self.assertEqual(code, 0, err)
        found = installed_clients(environ=env, platform_name="linux")
        self.assertEqual(set(found), {spec.id for spec in CLIENTS})
        self.assertEqual(found["cursor"], str(_path(env, "cursor")))
        self.assertIsNone(found["codex"])

        out = io.StringIO()
        code = show_rules(stdout=out)
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), rules_source())
        out = io.StringIO()
        code = show_rules("cursor", stdout=out)
        self.assertEqual(code, 0)
        self.assertEqual(
            out.getvalue().encode("utf-8"), _golden("cursor", "created.md")
        )

        out = io.StringIO()
        code = list_rules(environ=env, platform_name="linux", as_json=True, stdout=out)
        body = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(len(body["clients"]), len(CLIENTS))
        by_id = {row["id"]: row for row in body["clients"]}
        self.assertIs(by_id["cursor"]["installed"], True)
        self.assertIs(by_id["codex"]["installed"], False)

    def test_missing_home_reports_every_client_as_absent(self) -> None:
        found = installed_clients(environ={}, platform_name="linux")
        self.assertEqual(found, {spec.id: None for spec in CLIENTS})
        code, _out, err = _install({}, ["cursor"])
        self.assertEqual(code, 2)
        self.assertIn("home", err)


class RulesCliTests(unittest.TestCase):
    def test_cli_install_json_remove_and_help(self) -> None:
        _home, env = _fresh()
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
            os.environ.pop("CODEX_HOME", None)
            os.environ.pop("OHHO_ARM_HARDWARE", None)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = main(["rules", "install", "--client", "gemini", "--json"])
            self.assertEqual(code, 0, stderr.getvalue())
            body = json.loads(stdout.getvalue())
            path = Path(body["results"][0]["path"])
            self.assertEqual(body["results"][0]["action"], "created")
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(Path(env["HOME"])))
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = main(["rules", "remove", "--client", "gemini"])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertFalse(path.exists())
            self.assertIn("deleted", stdout.getvalue())
            with self.assertRaises(SystemExit) as ctx:
                main(["rules"])
            self.assertEqual(ctx.exception.code, 2)

    def test_cli_dry_run_and_list(self) -> None:
        _home, env = _fresh()
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
            os.environ.pop("CODEX_HOME", None)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = main(
                    ["rules", "install", "--client", "cursor", "--dry-run", "--json"]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            body = json.loads(stdout.getvalue())
            self.assertIs(body["dry_run"], True)
            self.assertEqual(body["results"][0]["action"], "created")
            self.assertFalse(Path(body["results"][0]["path"]).exists())
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = main(["rules", "show"])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("https://ohho-robotics.com/onboard.md", stdout.getvalue())
            self.assertIn("Never arm hardware", stdout.getvalue())


class RulesPackageTests(unittest.TestCase):
    def test_wheel_includes_the_rules_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            # Isolated build: pip installs setuptools>=77 from [build-system].
            # --no-build-isolation uses the image's setuptools and fails CI.
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "-w",
                    tmp,
                    str(root),
                ],
                check=False,
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + "\n" + completed.stderr,
            )
            wheels = list(Path(tmp).glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                self.assertIn("ohho/agent_rules/ohho-rules.md", names)
                body = archive.read("ohho/agent_rules/ohho-rules.md").decode("utf-8")
        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        self.assertEqual(normalized, rules_source())
        self.assertNotIn("\r", rules_source())
        for phrase in ("sim://", "Never arm hardware", "SafetyGate"):
            self.assertIn(phrase, body)
