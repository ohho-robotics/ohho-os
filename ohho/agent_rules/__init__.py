"""Install Ohho agent rules into AI coding clients (OHH-271 phase 6).

The rules body lives only in ``ohho-rules.md`` next to this module. Every
client file receives that body between stable markers. Text outside the
markers is kept byte-for-byte.

User-scope paths (re-checked 17 Oct 2026; citations in ``docs/rules.md``):

- Claude Code: ``~/.claude/CLAUDE.md`` (or ``$CLAUDE_CONFIG_DIR/CLAUDE.md``).
  A user skill is not used: Claude Code loads ``CLAUDE.md`` every session and
  loads a skill body only when the skill is invoked.
- Codex: ``~/.codex/AGENTS.md`` (or ``$CODEX_HOME/AGENTS.md``).
- Cursor: ``~/.cursor/rules/ohho.mdc``. Account-synced Customize rules have
  no file. ``--project DIR`` writes ``DIR/.cursor/rules/ohho.mdc``.
- VS Code: ``~/.copilot/copilot-instructions.md``.
- Gemini CLI: ``~/.gemini/GEMINI.md``.
- Windsurf: ``~/.codeium/windsurf/memories/global_rules.md``.
  ``--project DIR`` writes ``DIR/.windsurf/rules/ohho.md``.
- OpenCode: ``$XDG_CONFIG_HOME/opencode/AGENTS.md`` or
  ``~/.config/opencode/AGENTS.md``.

``installed_clients()`` is the ``ohho doctor --agent`` hook.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, TextIO, Tuple

from ..mcp_connect import CLIENTS, McpConnectError, normalize_client, resolve_home

RULES_VERSION = "v1"
CREATED_MARKER = "<!-- ohho-agent-rules:created -->"
_START = f"<!-- ohho-agent-rules:start {RULES_VERSION} -->"
_END = "<!-- ohho-agent-rules:end -->"
_SOURCE_NAME = "ohho-rules.md"

# Version token is optional so an older block still matches.
_BLOCK_RE = re.compile(
    r"<!-- ohho-agent-rules:start(?: [^\s>]+)? -->.*?<!-- ohho-agent-rules:end -->(?:\r\n|\n)?",
    re.DOTALL,
)

_CURSOR_HEADER = (
    "---\n"
    'description: "Ohho robot safety and setup. Use sim://. Never arm hardware."\n'
    "alwaysApply: true\n"
    "---\n"
)
_WINDSURF_PROJECT_HEADER = (
    '---\ntrigger: always_on\ndescription: "Ohho robot safety and setup"\n---\n'
)

_CURSOR_NOTE = (
    "Cursor Settings rules sync with the account and have no file to merge. "
    "This command writes ~/.cursor/rules/ohho.mdc (local, not synced). "
    "Pass --project DIR to write DIR/.cursor/rules/ohho.mdc instead."
)

Now = Callable[[], datetime]


class RulesError(Exception):
    """A rules request was refused. ``code`` is the process exit status."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RulesResult:
    """One client's install or remove outcome."""

    client: str
    action: str
    path: str
    backup: Optional[str]
    changed: bool

    def as_dict(self) -> dict:
        return {
            "client": self.client,
            "action": self.action,
            "path": self.path,
            "backup": self.backup,
            "changed": self.changed,
        }


def rules_source() -> str:
    """The only rules body, exactly as shipped in the package.

    Newlines are ``\\n`` even when a Windows checkout stored ``\\r\\n``.
    """
    raw = files("ohho.agent_rules").joinpath(_SOURCE_NAME).read_bytes()
    text = raw.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def rules_block(source: Optional[str] = None) -> str:
    """Marked block. ``source`` defaults to :func:`rules_source`."""
    body = rules_source() if source is None else source
    if not body.endswith("\n"):
        body += "\n"
    return f"{_START}\n{body}{_END}\n"


def _header(client_id: str, *, project: bool) -> str:
    if client_id == "cursor":
        return _CURSOR_HEADER
    if client_id == "windsurf" and project:
        return _WINDSURF_PROJECT_HEADER
    return ""


def render_new_file(client_id: str, *, project: bool = False) -> str:
    """File text a fresh install writes for ``client_id``."""
    header = _header(client_id, project=project)
    return header + CREATED_MARKER + "\n" + rules_block()


def rules_path(
    client_id: str,
    *,
    home: Optional[Path],
    environ: Mapping[str, str],
    platform_name: str,
    project: Optional[Path] = None,
) -> Path:
    """Absolute rules file for a canonical ``client_id``.

    ``platform_name`` selects Windows versus POSIX home resolution. Client
    paths below do not otherwise fork on the OS. ``project`` switches to the
    project file and ignores ``home``.
    """
    if project is not None:
        root = project
        if client_id == "claude-code":
            return root / "CLAUDE.md"
        if client_id == "codex":
            return root / "AGENTS.md"
        if client_id == "cursor":
            return root / ".cursor" / "rules" / "ohho.mdc"
        if client_id == "vscode":
            return root / ".github" / "copilot-instructions.md"
        if client_id == "gemini":
            return root / "GEMINI.md"
        if client_id == "windsurf":
            return root / ".windsurf" / "rules" / "ohho.md"
        if client_id == "opencode":
            return root / "AGENTS.md"
        raise RulesError(f"unknown client {client_id!r}", code=2)

    if home is None:
        raise RulesError(
            "cannot resolve the home directory (set HOME or USERPROFILE)",
            code=2,
        )
    if client_id == "claude-code":
        override = environ.get("CLAUDE_CONFIG_DIR")
        base = Path(override) if override else home / ".claude"
        return base / "CLAUDE.md"
    if client_id == "codex":
        override = environ.get("CODEX_HOME")
        base = Path(override) if override else home / ".codex"
        return base / "AGENTS.md"
    if client_id == "cursor":
        return home / ".cursor" / "rules" / "ohho.mdc"
    if client_id == "vscode":
        return home / ".copilot" / "copilot-instructions.md"
    if client_id == "gemini":
        return home / ".gemini" / "GEMINI.md"
    if client_id == "windsurf":
        return home / ".codeium" / "windsurf" / "memories" / "global_rules.md"
    if client_id == "opencode":
        xdg = environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else home / ".config"
        return base / "opencode" / "AGENTS.md"
    raise RulesError(f"unknown client {client_id!r}", code=2)


def _select_clients(
    clients: Optional[Sequence[str]], *, all_clients: bool
) -> List[str]:
    if all_clients:
        if clients:
            for name in clients:
                normalize_client(name)
        return [spec.id for spec in CLIENTS]
    if not clients:
        raise RulesError("pass --client or --all", code=2)
    chosen: List[str] = []
    for name in clients:
        client_id = normalize_client(name)
        if client_id not in chosen:
            chosen.append(client_id)
    return chosen


def _read_text(path: Path) -> Optional[str]:
    """Return file text, ``None`` if it is missing.

    Bytes are decoded as UTF-8 with newline characters preserved. A non-file
    or non-UTF-8 path is refused and left untouched.
    """
    try:
        exists = path.exists()
    except OSError as exc:
        raise RulesError(f"cannot read {path} ({exc})", code=1) from exc
    if not exists:
        return None
    if not path.is_file():
        raise RulesError(f"{path} is not a file. Refusing to modify it.", code=1)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RulesError(f"cannot read {path} ({exc})", code=1) from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RulesError(
            f"{path} is not valid UTF-8 ({exc}). Refusing to modify it.",
            code=1,
        ) from exc


def _has_rules(text: str) -> bool:
    return "<!-- ohho-agent-rules:start" in text


def _replace_blocks(text: str, block: str) -> Tuple[str, int]:
    matches = list(_BLOCK_RE.finditer(text))
    if not matches:
        return text, 0
    parts: List[str] = []
    last = 0
    for index, match in enumerate(matches):
        parts.append(text[last : match.start()])
        if index == 0:
            parts.append(block)
        last = match.end()
    parts.append(text[last:])
    return "".join(parts), len(matches)


def _owned_and_empty(text: str, header: str) -> bool:
    """True when only the Ohho header and created-marker remain."""
    if CREATED_MARKER not in text:
        return False
    rest = text
    if header and rest.startswith(header):
        rest = rest[len(header) :]
    rest = rest.replace(CREATED_MARKER + "\r\n", "")
    rest = rest.replace(CREATED_MARKER + "\n", "")
    rest = rest.replace(CREATED_MARKER, "")
    return rest.strip() == ""


def _plan(
    old: Optional[str], client_id: str, *, project: bool, remove: bool
) -> Tuple[Optional[str], str]:
    """Return ``(new_text, action)``. ``new_text is None`` deletes the file."""
    block = rules_block()
    header = _header(client_id, project=project)
    if remove:
        if old is None or not _has_rules(old):
            return old, "missing" if old is None else "unchanged"
        new = _BLOCK_RE.sub("", old)
        if new == old:
            return old, "unchanged"
        if _owned_and_empty(new, header):
            return None, "deleted"
        return new, "removed"
    if old is None:
        return header + CREATED_MARKER + "\n" + block, "created"
    new, count = _replace_blocks(old, block)
    if count == 0:
        sep = "" if old.endswith("\n") or old == "" else "\n"
        return old + sep + block, "updated"
    if new == old:
        return old, "unchanged"
    return new, "updated"


def _stamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _backup_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.bak.{stamp}")
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{stamp}.{number}")
        number += 1
    return candidate


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _apply_one(
    client_id: str,
    path: Path,
    *,
    project: bool,
    remove: bool,
    dry_run: bool,
    stamp: str,
) -> RulesResult:
    old = _read_text(path)
    new, action = _plan(old, client_id, project=project, remove=remove)
    changed = action in {"created", "updated", "removed", "deleted"}
    backup: Optional[str] = None
    if changed and not dry_run:
        if old is not None:
            bak = _backup_path(path, stamp)
            shutil.copy2(path, bak)
            backup = str(bak)
        if action == "deleted" or new is None:
            path.unlink()
        else:
            try:
                _write_atomic(path, new)
            except OSError as exc:
                raise RulesError(f"cannot update {path} ({exc})", code=1) from exc
    return RulesResult(
        client=client_id,
        action=action,
        path=str(path),
        backup=backup,
        changed=changed,
    )


def _home_and_project(
    environ: Mapping[str, str],
    platform_name: str,
    project: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    project_path: Optional[Path] = None
    if project is not None:
        if not str(project).strip():
            raise RulesError("--project requires a directory", code=2)
        project_path = Path(project)
        if project_path.exists() and not project_path.is_dir():
            raise RulesError(f"{project_path} is not a directory", code=2)
    if project_path is not None:
        return None, project_path
    try:
        home = resolve_home(environ, platform_name)
    except McpConnectError as exc:
        raise RulesError(str(exc), code=exc.code) from exc
    return home, None


def _resolve_targets(
    client_ids: Sequence[str],
    *,
    environ: Mapping[str, str],
    platform_name: str,
    project: Optional[str],
) -> List[Tuple[str, Path, bool]]:
    home, project_path = _home_and_project(environ, platform_name, project)
    found: List[Tuple[str, Path, bool]] = []
    for client_id in client_ids:
        path = rules_path(
            client_id,
            home=home,
            environ=environ,
            platform_name=platform_name,
            project=project_path,
        )
        found.append((client_id, path, project_path is not None))
    return found


def install_rules(
    clients: Optional[Sequence[str]],
    *,
    all_clients: bool = False,
    project: Optional[str] = None,
    dry_run: bool = False,
    remove: bool = False,
    as_json: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    now: Optional[Now] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Install or remove rules. Returns a process exit code.

    Pass ``environ`` (HOME, USERPROFILE, APPDATA, XDG_CONFIG_HOME,
    CLAUDE_CONFIG_DIR, CODEX_HOME) from tests. The CLI uses the process
    environment. Nothing is written when ``dry_run`` is true.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        return _install(
            clients,
            all_clients=all_clients,
            project=project,
            dry_run=dry_run,
            remove=remove,
            as_json=as_json,
            environ=os.environ if environ is None else environ,
            platform_name=sys.platform if platform_name is None else platform_name,
            now=_utc_now if now is None else now,
            stdout=out,
        )
    except (RulesError, McpConnectError) as exc:
        code = exc.code if isinstance(exc, (RulesError, McpConnectError)) else 1
        err.write(f"error: {exc}\n")
        return code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _install(
    clients: Optional[Sequence[str]],
    *,
    all_clients: bool,
    project: Optional[str],
    dry_run: bool,
    remove: bool,
    as_json: bool,
    environ: Mapping[str, str],
    platform_name: str,
    now: Now,
    stdout: TextIO,
) -> int:
    client_ids = _select_clients(clients, all_clients=all_clients)
    targets = _resolve_targets(
        client_ids,
        environ=environ,
        platform_name=platform_name,
        project=project,
    )
    stamp = _stamp(now())
    results = [
        _apply_one(
            client_id,
            path,
            project=is_project,
            remove=remove,
            dry_run=dry_run,
            stamp=stamp,
        )
        for client_id, path, is_project in targets
    ]
    if as_json:
        payload = {
            "dry_run": dry_run,
            "remove": remove,
            "version": RULES_VERSION,
            "results": [item.as_dict() for item in results],
        }
        stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    else:
        stdout.write(_format_results(results, dry_run=dry_run))
    return 0


def _format_results(results: Sequence[RulesResult], *, dry_run: bool) -> str:
    prefix = "dry-run " if dry_run else ""
    lines = [
        f"{prefix}{item.client:<14} {item.action:<10} {item.path}"
        + (f"  backup {item.backup}" if item.backup else "")
        for item in results
    ]
    return "\n".join(lines) + "\n"


def _file_installed(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        return False
    return _has_rules(text)


def installed_clients(
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    project: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """Client id → rules path, or None when that client has no rules.

    ``ohho doctor --agent`` calls this with no arguments. User scope is the
    default. A missing home yields None for every client and does not raise.
    """
    env = os.environ if environ is None else environ
    plat = sys.platform if platform_name is None else platform_name
    try:
        targets = _resolve_targets(
            [spec.id for spec in CLIENTS],
            environ=env,
            platform_name=plat,
            project=project,
        )
    except (RulesError, McpConnectError):
        return {spec.id: None for spec in CLIENTS}
    found: dict[str, Optional[str]] = {}
    for client_id, path, _project in targets:
        found[client_id] = str(path) if _file_installed(path) else None
    return found


def list_rules(
    *,
    project: Optional[str] = None,
    as_json: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Print each client's target path and whether the rules block is present."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    env = os.environ if environ is None else environ
    plat = sys.platform if platform_name is None else platform_name
    try:
        targets = _resolve_targets(
            [spec.id for spec in CLIENTS],
            environ=env,
            platform_name=plat,
            project=project,
        )
    except (RulesError, McpConnectError) as exc:
        err.write(f"error: {exc}\n")
        code = exc.code if isinstance(exc, (RulesError, McpConnectError)) else 1
        return code
    titles = {spec.id: spec.title for spec in CLIENTS}
    rows = []
    for client_id, path, _project in targets:
        installed = _file_installed(path)
        rows.append(
            {
                "id": client_id,
                "title": titles[client_id],
                "installed": installed,
                "path": str(path),
            }
        )
    if as_json:
        payload = {"version": RULES_VERSION, "project": project, "clients": rows}
        out.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return 0
    lines = []
    for row in rows:
        state = "installed" if row["installed"] else "not installed"
        lines.append(f"{row['id']:<14} {row['title']:<16} {state:<14} {row['path']}")
    lines.append(_CURSOR_NOTE)
    out.write("\n".join(lines) + "\n")
    return 0


def show_rules(
    client: Optional[str] = None,
    *,
    project: bool = False,
    as_json: bool = False,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Print the rules source, or the fresh file for one client."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        if client is None:
            client_id: Optional[str] = None
            text = rules_source()
            if not text.endswith("\n"):
                text += "\n"
        else:
            client_id = normalize_client(client)
            text = render_new_file(client_id, project=project)
    except McpConnectError as exc:
        err.write(f"error: {exc}\n")
        return exc.code
    if as_json:
        payload = {
            "version": RULES_VERSION,
            "client": client_id,
            "project": project,
            "text": text,
        }
        out.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return 0
    out.write(text)
    return 0
