# OhhO CLI account (`ohho login`, `ohho whoami`, `ohho logout`)

Sign-in is optional. `sim://`, `ohho sim`, and the local `ohho mcp serve` /
`ohho mcp connect` stdio server keep working with no account. `ohho mcp
connect --hosted` stays refused until hosted MCP auth (OHH-274).

The access token is never printed by `login`, `whoami`, `logout`, or
`ohho doctor`, and it is not written to the log.

## Console

Device flow from OHH-272 (RFC 8628):

| Step | Method | Path |
|------|--------|------|
| Request a code | POST | `/api/cli/device/code` |
| Poll for a token | POST | `/api/cli/device/token` |
| Current account | GET | `/api/cli/whoami` |

The default API origin is `https://console.ohho-robotics.com`. Set
`OHHO_CONSOLE_URL` to an `http` or `https` origin (no path, no userinfo) to
talk to a preview host. The CLI sends `scope=mcp` and `client_name=OhhO CLI`.

`OHHO_TOKEN`, when set to a non-empty value, is the access token for
`ohho whoami`. It wins over the credentials file. It is never written to
disk, and `ohho logout` does not clear it.

## `ohho login`

```bash
ohho login
ohho login --no-browser
```

The command prints the user code and the verification URL from the console,
then opens a browser unless `--no-browser` is set or no browser is available.
It polls `POST /api/cli/device/token` until the code is approved, denied, or
expired. The wait between polls is the console's `interval`. A `slow_down`
error (or HTTP 429) switches to the returned `interval` or `Retry-After`,
or adds five seconds when the console does not send one. The wait is capped
at 60 seconds. Polling also stops at `expires_in` (at most one hour).

On success the access token is stored in `~/.ohho/credentials.json`. The
file mode is `0600` on POSIX. On Windows the DACL grants the current user
full control and removes inherited entries. The token is not printed.

If `OHHO_CONSOLE_URL` points at a different host than the verification URL,
the CLI still opens the URL the console returned, and also prints
`{OHHO_CONSOLE_URL}/device`. OHH-272 currently advertises
`https://console.ohho-robotics.com/device` from every deployment.

## `ohho whoami`

```bash
ohho whoami
ohho whoami --json
```

Text, when the console accepts the token:

```text
email: ada@example.com
scopes: mcp
```

`scopes` is space-separated. A missing email is printed as `unknown`.

Text, when no token is configured or the console returns 401:

```text
not signed in
```

The exit code is 1. A credentials file that cannot be parsed, a bad
`OHHO_CONSOLE_URL`, or a console that cannot be reached prints `error: …`
on stderr and exits 1. Those cases do not print the token or the file body.

### `ohho whoami --json`

UTF-8 JSON, one object, one line, no other stdout. Field order is fixed.

Signed in (exit 0):

| Field | Type | Meaning |
|-------|------|---------|
| `signed_in` | `true` | The console accepted the token. |
| `email` | string or null | Account email from `GET /api/cli/whoami`. |
| `scopes` | array of strings | Scopes from that response. The device flow requests `mcp`. |
| `user_id` | string or null | Console user id. |

```json
{"signed_in": true, "email": "ada@example.com", "scopes": ["mcp"], "user_id": "5b1c0b0e-0000-4000-8000-000000000001"}
```

Not signed in, or the console call failed (exit 1):

| Field | Type | Meaning |
|-------|------|---------|
| `signed_in` | `false` | |
| `email` | null | |
| `scopes` | `[]` | |
| `user_id` | null | |
| `error` | string | Why `signed_in` is false. |

`error` is one of:

| Value | When |
|-------|------|
| `not_signed_in` | No `OHHO_TOKEN` and no credentials file (or the file has no token). |
| `invalid_token` | The console returned 401 (unknown or revoked). |
| `credentials_unreadable` | The credentials file or home directory could not be read. |
| `console_unavailable` | `OHHO_CONSOLE_URL` is invalid, or the console could not be reached, or it returned a non-401 error. |

```json
{"signed_in": false, "email": null, "scopes": [], "user_id": null, "error": "not_signed_in"}
```

The access token is not a field. `email`, `user_id`, and `scopes` are dropped
when they contain the token.

## `ohho logout`

```bash
ohho logout
```

Deletes `~/.ohho/credentials.json` when that file exists. Running it again
still exits 0.

```text
Signed out. Local credentials removed.
```

OHH-272 has no CLI revoke endpoint. Logout does not call the console. A
token stays valid until it is revoked on the console account page (the
signed-in browser session sets `cli_tokens.revoked_at`). When `OHHO_TOKEN`
is set, logout also prints that the variable is still in effect and was not
written to disk. Later `ohho whoami` calls keep using it.

## Credentials file

Path: `~/.ohho/credentials.json` (`USERPROFILE` on Windows, otherwise
`HOME`). The CLI does not use `Path.home()`, so a process can point the file
elsewhere by setting those variables.

```json
{
  "version": 1,
  "token": "ohho_…",
  "token_type": "Bearer",
  "scope": "mcp",
  "console_url": "https://console.ohho-robotics.com"
}
```

`token` is the secret. Do not commit this file. `console_url` is the origin
that issued it. `OHHO_CONSOLE_URL` still wins for the next API call.

## Human TODO

- Run `ohho login`, `ohho whoami`, and `ohho logout` once against a console
  preview (`OHHO_CONSOLE_URL=https://<preview>`). This repository does not
  call the production console from tests. If the printed verification URL is
  the production `/device` page, open `<preview>/device` and enter the same
  user code.
- Server-side revoke from the CLI needs a console endpoint. Until that
  exists, revoke the token from the console account page after `ohho logout`.
