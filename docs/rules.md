# `ohho rules`

Agent rules for the seven clients `ohho mcp connect` already knows. The
command merges one safety text into each client's user-scope instructions.
It does not arm hardware and it does not change safety limits.

The body is only `ohho/agent_rules/ohho-rules.md`, shipped as package data.
Every client file contains that body unchanged, between:

```text
<!-- ohho-agent-rules:start v1 -->
...
<!-- ohho-agent-rules:end -->
```

Text outside the markers is kept byte-for-byte. An older
`ohho-agent-rules:start` block is replaced in place. A second run that would
not change the bytes does not write and does not make a backup.

```bash
ohho rules list
ohho rules show
ohho rules install --client cursor
ohho rules install --all
ohho rules install --client claude-code --dry-run
ohho rules install --client cursor --project .
ohho rules install --remove --client cursor
ohho rules remove --client cursor
ohho rules install --client vscode --json
```

| Flag | Effect |
|------|--------|
| `--client` | Client id. Repeat the flag. Names and aliases match `ohho mcp connect --list`. |
| `--all` | Every client that command knows. |
| `--project DIR` | Write the project file under `DIR` instead of the user file. |
| `--dry-run` | Print the action. Create no directory, file, or backup. |
| `--json` | One JSON object on stdout. |
| `--remove` | On `install`, delete only the Ohho block. `ohho rules remove` is the same. |

`--remove` deletes the file only when Ohho created it (the file contains
`<!-- ohho-agent-rules:created -->`) and nothing of yours would remain. A file
you already had keeps everything outside the block, including when that
remainder is empty.

Before changing an existing file the command copies it to
`<name>.bak.<UTC timestamp>`, for example `AGENTS.md.bak.20261017T160000Z`.
The new text is written by a temp file in the same directory and `os.replace`.

Client ids: `claude-code` (alias `claude`), `codex`, `cursor`, `vscode`
(alias `vs-code`, `code`), `gemini` (alias `gemini-cli`), `windsurf`,
`opencode` (alias `open-code`).

`ohho doctor --agent` calls `ohho.agent_rules.installed_clients()`. The rules
check passes only when every client has the block. Until then the detail is
`rules missing: ...`. See [cli.md](cli.md).

## What the rules tell an agent

- Use `sim://` first and by default.
- Never arm hardware. Never set `OHHO_ARM_HARDWARE`. Never type the arming
  confirmation. Never pass `--arm`.
- Never call `release-stop` or `release_stop`. Never change safety limits.
- Emergency stop is always allowed.
- On any SafetyGate or other safety refusal, stop and tell the user. Do not
  retry around it.
- Ask the human for anything physical (power, placing the robot, arming).
- Report only state that was verified.
- Use `ohho doctor --agent`, `ohho verify --sim`, and the `ohho mcp serve`
  tools `connect`, `drive`, `telemetry`, `nav_goto`, and `look` with `sim://`.

The text tells the agent not to arm and not to point MCP tools at a hardware
URI. It does not claim the server enforces that. The server-side guarantee
is OHH-278, described in [mcp.md](mcp.md) and [safety.md](safety.md):
`ohho mcp serve` is `sim://` unless a human passes `--allow-hardware`, the
process never arms and never releases an e-stop, and `emergency_stop` latches.
The CLI already wraps non-sim transports in `SafetyGate`.

Setup playbook: https://ohho-robotics.com/onboard.md

## Paths

Home is `USERPROFILE` on Windows and `HOME` otherwise. `XDG_CONFIG_HOME` and
`APPDATA` are read when the client's own docs use them. Tests pass a
temporary home and never touch the real profile.

| Client | User file | Project file (`--project DIR`) | Source |
|--------|-----------|--------------------------------|--------|
| Claude Code | `$CLAUDE_CONFIG_DIR/CLAUDE.md`, else `~/.claude/CLAUDE.md` | `DIR/CLAUDE.md` | [Memory](https://code.claude.com/docs/en/memory), [directory layout](https://code.claude.com/docs/en/claude-directory) |
| Codex | `$CODEX_HOME/AGENTS.md`, else `~/.codex/AGENTS.md` | `DIR/AGENTS.md` | [AGENTS.md](https://developers.openai.com/codex/guides/agents-md) |
| Cursor | `~/.cursor/rules/ohho.mdc` | `DIR/.cursor/rules/ohho.mdc` | [Rules](https://cursor.com/docs/rules), [help](https://cursor.com/help/customization/rules) |
| VS Code | `~/.copilot/copilot-instructions.md` | `DIR/.github/copilot-instructions.md` | [Custom instructions](https://code.visualstudio.com/docs/agent-customization/custom-instructions) |
| Gemini CLI | `~/.gemini/GEMINI.md` | `DIR/GEMINI.md` | [GEMINI.md](https://geminicli.com/docs/cli/gemini-md/) |
| Windsurf | `~/.codeium/windsurf/memories/global_rules.md` | `DIR/.windsurf/rules/ohho.md` | [Memories and rules](https://docs.devin.ai/desktop/cascade/memories) |
| OpenCode | `$XDG_CONFIG_HOME/opencode/AGENTS.md`, else `~/.config/opencode/AGENTS.md` | `DIR/AGENTS.md` | [Rules](https://opencode.ai/docs/rules/), [instructions](https://opencode.ai/v2/docs/instructions/) |

### Claude Code

User instructions in `~/.claude/CLAUDE.md` load at the start of every session
([memory](https://code.claude.com/docs/en/memory)). `CLAUDE_CONFIG_DIR`
replaces `~/.claude` ([directory layout](https://code.claude.com/docs/en/claude-directory)).

A personal skill at `~/.claude/skills/ohho/SKILL.md` is the other documented
place ([skills](https://code.claude.com/docs/en/skills)). This command does
not use it. Claude Code's own guidance is to put "never do X" rules in
`CLAUDE.md`, because a skill body is loaded only when the skill is invoked.
Safety rules have to be present even when the model does not choose the
skill. On Windows, `~/.claude` is `%USERPROFILE%\.claude`.

`--project` writes `DIR/CLAUDE.md`. `DIR/.claude/CLAUDE.md` is also read by
Claude Code; this command uses the project-root file.

### Codex

Codex reads `AGENTS.md` in the Codex home (`~/.codex`, or `CODEX_HOME`) before
project files ([AGENTS.md](https://developers.openai.com/codex/guides/agents-md)).
If `AGENTS.override.md` exists in that directory, Codex uses the override and
does not read `AGENTS.md`. This command writes `AGENTS.md` and does not create
or edit an override file.

### Cursor

Cursor's account rules (Customize → Rules) apply in every project and sync
with the account. Those rules have no documented file, so this command cannot
merge into them ([Rules](https://cursor.com/docs/rules)).

Cursor also loads local rule files from `~/.cursor/rules/` (Windows:
`%USERPROFILE%\.cursor\rules`). They apply to all projects on this machine and
do not sync ([help](https://cursor.com/help/customization/rules)). A fresh
install writes `ohho.mdc` there with `alwaysApply: true`, which is the
project-rule shape that is always included ([Rules](https://cursor.com/docs/rules)).
A plain `.md` file in `.cursor/rules` is ignored.

`--project DIR` writes `DIR/.cursor/rules/ohho.mdc` instead of the user file.
Use that when you want the rule in a repository. `ohho rules list` prints
this limitation.

### VS Code

Copilot sessions read personal always-on instructions from
`~/.copilot/copilot-instructions.md`
([custom instructions](https://code.visualstudio.com/docs/agent-customization/custom-instructions)).
That path is under the user profile (`%USERPROFILE%` on Windows), not under
`%APPDATA%\Code\User`. Profile storage is for the Local harness; Copilot
agent sessions use `~/.copilot`.

`--project` writes `DIR/.github/copilot-instructions.md`, the workspace
always-on file from the same document.

### Gemini CLI

`~/.gemini/GEMINI.md` is the global context file, loaded for every project
([GEMINI.md](https://geminicli.com/docs/cli/gemini-md/)). `--project` writes
`DIR/GEMINI.md`.

### Windsurf

The always-on global rules file is
`~/.codeium/windsurf/memories/global_rules.md`. It has no frontmatter and is
limited to 6,000 characters
([memories and rules](https://docs.devin.ai/desktop/cascade/memories)). The
Ohho block stays under that limit.

`--project` writes `DIR/.windsurf/rules/ohho.md` with `trigger: always_on`.
Current Devin Desktop docs prefer `.devin/rules/` and still read
`.windsurf/rules/` as the fallback. The Windsurf client id uses the
`.windsurf` path so older Windsurf installs still see the file. Workspace
rules are limited to 12,000 characters.

### OpenCode

Global instructions are `~/.config/opencode/AGENTS.md`
([rules](https://opencode.ai/docs/rules/),
[instructions](https://opencode.ai/v2/docs/instructions/)). When
`XDG_CONFIG_HOME` is set, the file is `$XDG_CONFIG_HOME/opencode/AGENTS.md`.
On Windows without that variable the file is
`%USERPROFILE%\.config\opencode\AGENTS.md`, the same `.config/opencode`
directory `ohho mcp connect` uses. `--project` writes `DIR/AGENTS.md`.
