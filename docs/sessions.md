# Session registry (OHH-206)

`ohho sim up` and `ohho robot announce` publish a row the VR garage can
list. The headset polls Supabase. It does not scan the LAN for a sim.

This repository holds the migration, the CLI, and the C# contract. OhhO-VR
is not modified here. A later VR change implements `SessionClient` from the
contract below.

## Apply the migration (Varun)

The OhhO website git clone is not required. The SQL file in this repo is
the whole migration.

1. Open the Supabase dashboard for the Ohho project. It is the project
   OhhO-VR already uses: the garage reads `user_robots` there.
2. Open **SQL Editor → New query**.
3. Paste `supabase/migrations/20261014120000_robot_sessions.sql` and run it.
4. Confirm `public.robot_sessions` exists and **RLS is enabled** (forced).
5. Confirm the view `public.robot_sessions_visible` exists.
6. Confirm four policies, each `auth.uid() = user_id`: select, insert,
   update, delete. There is no policy for `anon` and no `using (true)`.
7. The script ends with `notify pgrst, 'reload schema'` so PostgREST picks
   up the view. If the garage still 404s the view, run that `notify` again.

Re-running the script is safe for policies, the trigger, and the view. The
table is created with `if not exists`, so a second run does not wipe rows.

The headset must keep using the signed-in user's access token and the anon
key, the same way `GarageClient` reads `user_robots`. The service-role key
stays on the PC that runs `ohho sim up`. Do not put it in the headset, in
git, or in a log.

### Operator-controlled PC environment

Use this setup only on the operator-controlled PC that runs `ohho sim up`.
Do not put the service-role key on an end-user PC or in the headset.

PowerShell, for the current user (replace the placeholders; do not commit them):

```powershell
[System.Environment]::SetEnvironmentVariable("OHHO_SUPABASE_URL", "https://<project-ref>.supabase.co", "User")
[System.Environment]::SetEnvironmentVariable("OHHO_SUPABASE_SERVICE_ROLE_KEY", "<service-role-key>", "User")
[System.Environment]::SetEnvironmentVariable("OHHO_USER_ID", "<auth.users uuid>", "User")
[System.Environment]::SetEnvironmentVariable("OHHO_GATEWAY_TOKEN", "<same token the gateway checks>", "User")
[System.Environment]::SetEnvironmentVariable("OHHO_SIM_HOST", "<this PC LAN address>", "User")
```

Open a new terminal so the process sees them.

`OHHO_USER_ID` is required with the service role because that key is not a
user and bypasses RLS. The row's `user_id` is what the headset's RLS policy
matches.

Alternative to the service role: set `OHHO_SUPABASE_ANON_KEY` and
`OHHO_SUPABASE_JWT` (a user access token). `user_id` is the JWT `sub`
unless `OHHO_USER_ID` is set. If both are set they must be the same UUID.

Accepted URL and key names are `OHHO_SUPABASE_URL` or `SUPABASE_URL`, and
the same `OHHO_` / unprefixed pairs for the service role, anon key, and JWT
(`SUPABASE_ACCESS_TOKEN`).

## Table

| Column | Values |
|---|---|
| `user_id` | `auth.users.id`. Owner-only RLS. |
| `robot_id` | Registry id, for example `omnibot`. |
| `kind` | `sim` or `real`. |
| `target` | `local`, `cloud`, or `robot`. |
| `rosbridge_url` | WebSocket the headset opens. No token query. |
| `video_urls` | JSON array of video URLs. No token query. |
| `webrtc_signal_url` | Signaling URL, or `""`. |
| `token_hash` | SHA-256 hex of the OHH-196 gateway token. |
| `status` | `active` or `ended`. |
| `heartbeat_at` | Last heartbeat, UTC. |
| `expires_at` | `heartbeat_at` + 30 seconds. |

Unique slot: `(user_id, robot_id, kind, target)`. A second `sim up` for the
same robot updates that row.

There is no `token` column. A leaked database does not reveal the bearer
token the gateway checks.

## Token

The gateway (OHH-196) accepts `OHHO_GATEWAY_TOKEN`, then
`OHHO_GATEWAY_TOKEN_FILE`, then `~/.ohho/gateway/pairing-code`. This CLI
uses that same order. If none of them exist it writes a new pairing code
to `~/.ohho/gateway/pairing-code` so a later `ohho-gateway` process reuses
it, and prints the code once. An env token or a token file is not printed.

The stored hash is SHA-256 over the UTF-8 bytes of the stripped token,
lowercase hex. Example, not a deployed secret:

```text
token: example-gateway-token
token_hash: acf72e1e008bd82c8a08ebee883ac904d34e70865ca615bbb64bb8d615247b26
```

URLs stored in the row have userinfo, fragments, and any `token` query
parameter removed. The headset already has the pairing code from OHH-196
(`PlayerPrefs` key `ohho.gateway.token`). It appends that token only to
absolute `wss` and `https` URLs. Cleartext `ws` and `http` endpoints,
including the local simulator defaults, never receive the token.

## Heartbeat and stale rows

The command stays in the foreground and updates `heartbeat_at` and
`expires_at` every **10 seconds**. `expires_at` is 30 seconds after the
heartbeat, so one missed beat does not hide the row. Three missed beats
do. `ohho sim down` deletes the row immediately and writes a stop file so
the heartbeat process exits instead of upserting again.

`robot_sessions_visible` is the list the garage should query:

```sql
status = 'active' and expires_at > now()
```

The view is `security_invoker`, so RLS still limits the result to the
signed-in user. Ended rows and stale rows are hidden even though they may
still sit in `robot_sessions` until a later delete.

## Commands

### Local sim

```bash
ohho sim up --robot omnibot --target local
ohho sim down --robot omnibot
```

What it does:

1. Refuses `--target cloud` before any other work (see below).
2. Resolves the gateway token and the Supabase credentials.
3. Runs `docker compose -f ohho/sim/docker-compose.yml up -d`.
4. Upserts the session and prints the rosbridge URL.
5. Heartbeats every 10 seconds until Ctrl-C or `ohho sim down`.

The compose file's only service is on the `gpu` profile. The default
`up -d` does not pull an image and does not start a container, because
the GPU image is OHH-204 and is not published. The CLI still upserts the
session and prints:

```text
docker compose --profile gpu -f ohho/sim/docker-compose.yml up -d
```

That is the command to run once OHH-204 publishes `ohho/omnibot-sim:ohh-204`
(override the name with `OHHO_SIM_IMAGE`). Ports `9090` (rosbridge) and
`8080` (web video) are what the session row advertises.

Default local URLs, with no token in them:

```text
ws://<host>:9090
http://<host>:8080/stream?topic=/camera/front/image_raw
```

`<host>` is `--host`, else `OHHO_SIM_HOST`, else the address this PC would
use to reach the LAN. `127.0.0.1` is printed with a warning because a
headset cannot open loopback on the PC. WebRTC signaling is empty for the
local sim until OHH-210.

`--rosbridge`, `--video` (repeatable), and `--webrtc` replace those
defaults. The existing `ohho sim --robot omnibot --seconds N` pattern
drive is unchanged.

### Real robot

```bash
ohho robot announce --robot omnibot
ohho robot announce --down --robot omnibot
```

`kind=real`, `target=robot`. No container is started. Default URLs are the
OHH-196 gateway on port 8443:

```text
wss://<host>:8443/rosbridge
https://<host>:8443/video/stream?topic=/camera/front/image_raw
wss://<host>:8443/webrtc/signaling
```

The signaling path answers 503 until OHH-210. The gateway still requires
the pairing token on the request.

### Cloud

```bash
ohho sim up --robot omnibot --target cloud
ohho sim down --target cloud
```

Both exit with code 2 and print:

```text
ohho sim up --target cloud needs OHH-216. No cloud VM was created.
```

Nothing is upserted and no VM is provisioned.

## OhhO-VR SessionClient contract

Implement this in OhhO-VR. Do not invent a second JSON shape. Namespace and
HTTP style match `GarageClient` / `SupabaseAuthService`
(`OmniBot.VR.Core.Platform`).

### Methods

```csharp
namespace OmniBot.VR.Core.Platform
{
    public class SessionClient : MonoBehaviour
    {
        // Point at the same Supabase project as GarageClient.
        public void Configure(VrAuthConfig auth);

        // Sessions for one robot. Callback is (rows, error).
        // error is null on success. rows is null when error is set.
        public void ListForRobot(string robotId, Action<List<RobotSession>, string> done);

        // True when the garage should show the row.
        public bool IsFresh(RobotSession session, DateTime utcNow);
    }
}
```

`ListForRobot` is the method the garage calls. Poll it every 10 seconds
while the garage is open, and once when the panel opens, so a PC that just
ran `ohho sim up` shows up within 10 seconds. `ohho sim down` deletes the
row; the next poll drops it. There is no push channel.

`IsFresh` returns true only when `status == "active"` and `expires_at` is
strictly after `utcNow`. A missing or unparseable `expires_at` is stale.
The SQL view applies the same rule with the database clock. The headset
applies it again so a row that expires between polls disappears.

### Request

```http
GET {supabaseUrl}/rest/v1/robot_sessions_visible?select=id,user_id,robot_id,kind,target,rosbridge_url,video_urls,webrtc_signal_url,token_hash,status,heartbeat_at,expires_at,created_at,updated_at&robot_id=eq.{robotId}&order=heartbeat_at.desc
apikey: {anon key}
Authorization: Bearer {user access token}
```

URL-encode `robotId`. Do not send `select=*`. Do not add a `token` query
parameter. RLS (`auth.uid() = user_id`) is what scopes the rows; the client
does not send `user_id` as the authority. The service-role key is not
available on the headset.

A fleet screen that is not filtered to one robot uses the same path and
select list without `robot_id=eq.`. That call is optional. Per-robot list
is the required one.

### JSON shape

PostgREST returns a JSON array. Property names are snake_case.

```csharp
public class RobotSession
{
    [JsonProperty("id")] public string Id;
    [JsonProperty("user_id")] public string UserId;
    [JsonProperty("robot_id")] public string RobotId;
    [JsonProperty("kind")] public string Kind;                 // "sim" | "real"
    [JsonProperty("target")] public string Target;             // "local" | "cloud" | "robot"
    [JsonProperty("rosbridge_url")] public string RosbridgeUrl;
    [JsonProperty("video_urls")] public List<string> VideoUrls;
    [JsonProperty("webrtc_signal_url")] public string WebrtcSignalUrl;
    [JsonProperty("token_hash")] public string TokenHash;      // sha256 hex, not a bearer token
    [JsonProperty("status")] public string Status;             // "active" | "ended"
    [JsonProperty("heartbeat_at")] public string HeartbeatAt;  // ISO-8601
    [JsonProperty("expires_at")] public string ExpiresAt;
    [JsonProperty("created_at")] public string CreatedAt;
    [JsonProperty("updated_at")] public string UpdatedAt;
}
```

Example row (token hash of the example string above):

```json
{
  "id": "8f1c0c2e-4b3a-4f1d-9c0a-6e5d4c3b2a10",
  "user_id": "2b0e1111-2222-4333-8444-555566667777",
  "robot_id": "omnibot",
  "kind": "sim",
  "target": "local",
  "rosbridge_url": "ws://192.0.2.10:9090",
  "video_urls": [
    "http://192.0.2.10:8080/stream?topic=/camera/front/image_raw"
  ],
  "webrtc_signal_url": "",
  "token_hash": "acf72e1e008bd82c8a08ebee883ac904d34e70865ca615bbb64bb8d615247b26",
  "status": "active",
  "heartbeat_at": "2026-10-14T11:40:00+00:00",
  "expires_at": "2026-10-14T11:40:30+00:00",
  "created_at": "2026-10-14T11:40:00+00:00",
  "updated_at": "2026-10-14T11:40:00+00:00"
}
```

### Connecting

Append the OHH-196 pairing token at connect time, and only when the URL
is an absolute `wss` or `https` endpoint. Cleartext `ws` and `http` URLs,
including the local simulator defaults, never receive the token. Real-robot
defaults use `wss` and `https` on port 8443, and those keep the token.
Do not read a token field from this JSON. There isn't one.

```csharp
string token = Uri.EscapeDataString(PlayerPrefs.GetString("ohho.gateway.token", ""));
bool secure = Uri.TryCreate(url, UriKind.Absolute, out var endpoint)
    && (string.Equals(endpoint.Scheme, "wss", StringComparison.OrdinalIgnoreCase)
        || string.Equals(endpoint.Scheme, "https", StringComparison.OrdinalIgnoreCase));
string authed = url;
if (secure && !string.IsNullOrEmpty(token))
{
    string join = url.Contains("?") ? "&" : "?";
    authed = $"{url}{join}token={token}";
}
```

Use that for `rosbridge_url` and each `video_urls` entry. Optional check,
not a substitute for the gateway's own 401: SHA-256 of the trimmed UTF-8
token, lowercase hex, compared to `token_hash` ordinal-ignore-case. A
mismatch means the headset has a different pairing code than the PC that
registered the session.

Cloud rows will not appear until OHH-216. An empty `webrtc_signal_url` on
a local sim is expected. The real-robot signaling URL may answer 503 until
OHH-210; keep the MJPEG `video_urls` fallback.
