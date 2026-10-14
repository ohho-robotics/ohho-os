# Cloud sim target (OHH-216)

`ohho cloud` is the orchestrator for a GPU VM that runs the OHH-204 sim
container behind a Caddy gateway. The headset still discovers the session
through the OHH-206 `robot_sessions` table (`kind=sim`, `target=cloud`).

This build does not create a cloud account, VM, DNS name, or bill. The AWS
adapter is a dry-run skeleton. `FakeProvider` is the in-memory provider the
unit tests use.

`ohho sim up --target cloud` and `ohho sim down --target cloud` still exit 2.
They do not upsert a row and they do not create a VM. The message names
OHH-216 and points at `ohho cloud start`.

## Provider choice

Varun picks the provider. The recommendation below is an input to that
decision, not the decision.

**Recommendation:** AWS `g6.xlarge` in `eu-west-2` (London), on-demand Linux.

Why this shape is the one the skeleton targets:

- The issue's default region is EU-West / London. `eu-west-2` is the AWS
  London region. The GCP London region is `europe-west2`.
- `g6.xlarge` is one NVIDIA L4 (24 GB). In the table below it is the cheaper
  of the two AWS GPU shapes quoted for London (`g6` L4 versus `g5` A10G).
- A full VM can run Caddy on 443, the token check, coturn, and the sim
  container with published ports bound to `127.0.0.1`. That matches the
  OHH-196 route map. A GPU pod is a weaker place to terminate a certificate
  you control.

The code default region is `eu-west-2`. Override with `--region` or
`OHHO_CLOUD_REGION`. The hourly rate is **not** hard-coded. Set
`OHHO_CLOUD_HOURLY_USD` to the figure you confirm, and
`OHHO_CLOUD_MONTHLY_BUDGET_USD` to the cap you accept.

### Prices read on 14 October 2026

Instance price only, unless the note says otherwise. VAT, disk, public IPv4,
and egress are extra. If a cell says **TODO verify**, that number was not
taken from a page fetched for this doc.

| Provider | Shape | GPU | Region | USD / hour | What was read |
|---|---|---|---|---|---|
| AWS | `g5.xlarge` | 1× A10G 24 GB | `eu-west-2` London | 1.277 on-demand Linux | [cloud-credits.org g5.xlarge eu-west-2](https://cloud-credits.org/prices/ec2/g5-xlarge/eu-west-2), read 14 October 2026. That page says the figure is from the AWS Price List API dated 2026-10-04. It is a third-party page, not the AWS console. |
| AWS | `g6.xlarge` | 1× L4 24 GB | `eu-west-2` London | 1.022 on-demand Linux | [cloud-credits.org g6.xlarge eu-west-2](https://cloud-credits.org/prices/ec2/g6-xlarge/eu-west-2), read 14 October 2026, same AWS Price List API date 2026-10-04. Recommended shape if Varun picks AWS. |
| GCP | `g2-standard-4` | 1× L4 24 GB, 4 vCPU, 16 GB RAM | `europe-west2` London | 0.805 on-demand, 0.483 spot | [DevZero europe-west2](https://www.devzero.io/instances/gcp/regions/europe-west2), read 14 October 2026. Google's GPU pricing page timed out in this pass. **TODO verify** on [cloud.google.com GPU pricing](https://cloud.google.com/compute/gpus-pricing) before buying. |
| RunPod | L4 pod (24 GB, card shows 50 GB RAM, 12 vCPU) | L4 24 GB | **TODO verify** London / EU-West | 0.44 Community Cloud, 0.49 Secure Cloud | [RunPod GPU pricing](https://www.runpod.io/gpu-instance/pricing), read 14 October 2026. The page says "Updated July 27, 2026" and "30+ regions". It does not name London. The L4 card on that fetch shows $0.49/hr; the page's schema.org block lists Community $0.44/hr and Secure $0.49/hr. |
| Lambda | 1× NVIDIA A10 (no L4, g5, or g6 on the page) | A10 24 GB, 30 vCPU, 226 GiB RAM, 1.3 TiB | **TODO verify** | 1.29 per GPU-hour, plus tax | [lambda.ai/pricing](https://lambda.ai/pricing), read 14 October 2026. The 1-GPU table is the closest published 24 GB card. The page says "plus applicable sales tax/VAT/GST". It does not name a region. |

The official AWS on-demand pricing page fetched the same day did not include
the GPU instance table in the text that came back, so it is not the source
of the $1.277 and $1.022 figures.

Worked example, arithmetic on the cited `g6.xlarge` rate, not a measured
bill: 20 hours × $1.022 = $20.44. A cap of `25` would cover that example
with a little room and would then refuse another start. Confirm the rate
before typing it into `OHHO_CLOUD_HOURLY_USD`.

## Commands

```bash
ohho cloud start --robot omnibot
ohho cloud status --robot omnibot
ohho cloud stop --robot omnibot
```

`start` defaults to `--provider aws` and region `eu-west-2`. It exits 0
after printing a dry-run plan when both of these are set:

- `OHHO_AWS_ACCESS_KEY_ID` and `OHHO_AWS_SECRET_ACCESS_KEY`
  (or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
- `OHHO_CLOUD_MONTHLY_BUDGET_USD` and `OHHO_CLOUD_HOURLY_USD`

and a gateway token is available (`OHHO_GATEWAY_TOKEN`, the token file, or
the pairing file). The plan records that `OHHO_GATEWAY_TOKEN` would be
injected. It does not print the token or the secret key. It does not open
a socket, pull an image, or write a session row.

Missing credentials, a missing budget, a non-positive budget, or
`--live` / `OHHO_CLOUD_LIVE=1` / `OHHO_CLOUD_DRY_RUN=0` exits 2. Live mode
is refused after the credential and budget checks. No VM is created.

`--provider fake` starts an in-memory instance, upserts `robot_sessions`
with `target=cloud`, heartbeats every 10 seconds, and stops the instance
when the process exits. Tests use this provider. It still refuses to start
when the local monthly ledger has reached the cap.

### Session row

URLs are TLS on port 443 and do not contain the token. The headset appends
`token` the same way it does for a local session (see `docs/sessions.md`).

```text
wss://<host>/rosbridge
https://<host>/video/stream?topic=/camera/front/image_raw
wss://<host>/webrtc/signaling
```

The stored credential is `token_hash` (SHA-256 hex). The orchestrator passes
the raw token only to the provider's in-memory injection list and, on a
future live start, the instance environment `OHHO_GATEWAY_TOKEN`.

The garage visibility window is unchanged: heartbeat every 10 seconds,
`expires_at` 30 seconds later. That is not the idle shutdown.

### Idle shutdown

A running sim is stopped when `now - last_heartbeat >= 15 minutes`
(900 seconds). A heartbeat at time T keeps the VM up through T+899 seconds
and stops it at T+900 seconds. `ohho cloud status` applies this before it
prints. The foreground `ohho cloud start` heartbeats every 10 seconds, so
the idle watch fires when those heartbeats stop (the process was killed, or
a later `status` observes a stale ledger).

Ctrl-C or `ohho cloud stop` stops immediately and deletes the session row.

The guest script `deploy/cloud/idle-watch.sh` is the on-VM backstop. An
established connection to local port 9090 or 8080 refreshes
`/var/lib/ohho/heartbeat`. Age of 900 seconds runs `shutdown -h now`.
**TODO Varun:** confirm that watch on a real guest. This repo does not run it.

### Monthly cap

The ledger is `~/.ohho/cloud/instances.json` (override `OHHO_CLOUD_STATE_DIR`).
Spend for the current UTC month is `hourly_usd * seconds / 3600`, including
time a still-running row has already used. Seconds outside the current month
do not count. When spend is greater than or equal to
`OHHO_CLOUD_MONTHLY_BUDGET_USD`, `start` refuses and does not call the
provider. A running row that crosses the cap is stopped. If the budget
environment is removed while a row is still running, `status` stops it
(fail closed).

This file is a local ledger, not an invoice. **TODO Varun:** reconcile it
with the provider bill. A VM started outside `ohho cloud` is invisible here.

## Gateway assets

`deploy/cloud/` is templates. No real token, TURN secret, or account id.

| File | Role |
|---|---|
| `Caddyfile` | Automatic HTTPS for `{$OHHO_CLOUD_DOMAIN}`, token check, then `/rosbridge` → `127.0.0.1:9090`, `/video/*` → `127.0.0.1:8080` (prefix stripped), `/foxglove` → `127.0.0.1:8765`. `/webrtc/*` responds `OHH-210` with status 503. Anything else is 401. |
| `turnserver.conf` | coturn on 3478 and 5349, relay ports 49152–49252. Secret and realm are placeholders. |
| `start-sim.sh` | `docker pull` of `ghcr.io/ohho-robotics/omnibot-sim:latest` (override `OHHO_SIM_IMAGE`) and `docker run --gpus all` with ports bound to `127.0.0.1`. Refuses a placeholder token and does not print the token. |
| `idle-watch.sh` | 15 minute guest shutdown. |
| `cloud-init.yaml` | Package list and boot path. Does not name an AMI. |

OHH-204 publishes `ghcr.io/<owner>/omnibot-sim` with tags `latest` and the
git sha on pushes to `main` (`.github/workflows/docker-sim.yml`). The GPU
compose file uses `ghcr.io/ohho-robotics/omnibot-sim:latest` and notes a
manual tag `ghcr.io/ohho-robotics/omnibot-sim:vr-<sha>`. **TODO Varun:**
confirm which tag is the one to pull. The image default is llvmpipe; the
startup script sets `LIBGL_ALWAYS_SOFTWARE=0` the same way as
`docker-compose.gpu.yml`. **TODO Varun:** install the NVIDIA driver and the
container toolkit on the guest. This repo does not install them and does
not pull the image.

Copy `deploy/cloud/` to `/opt/ohho` and install `/etc/ohho/cloud.env` mode
0600 before associating the user-data. The example env file in the
cloud-init template uses `REPLACE_WITH_*` placeholders.

## VR Cloud button and console button

Not implemented in this repository. Do not edit OhhO-VR or the console repo
from this tree. Both should call the same contract and then attach with the
existing `SessionClient` (`docs/sessions.md`) once `target` is `cloud`.

There is no HTTP server in this repo. The contract the console and the
headset should implement against a future wrapper:

```http
POST /api/cloud/sim/start
Content-Type: application/json

{"robot_id": "omnibot", "region": "eu-west-2"}
```

```http
GET /api/cloud/sim/status?robot_id=omnibot
```

```http
POST /api/cloud/sim/stop
Content-Type: application/json

{"robot_id": "omnibot"}
```

Response JSON. Empty strings when that field does not apply. `error` is
empty on success.

```json
{
  "state": "running",
  "robot_id": "omnibot",
  "region": "eu-west-2",
  "provider": "aws",
  "instance_id": "",
  "host": "",
  "rosbridge_url": "",
  "video_urls": [],
  "webrtc_signal_url": "",
  "session_id": "",
  "idle_deadline": "",
  "month_spend_usd": "0.000000",
  "monthly_cap_usd": "",
  "error": ""
}
```

`state` is one of `starting`, `running`, `stopped`, `dry-run`. This CLI's
fake provider becomes `running` inside `start` and does not emit
`starting`. The AWS adapter returns `dry-run` and does not insert a session
row. A future live provider should emit `starting` until the guest is
accepting connections, then `running`.

Headset Cloud control:

1. `POST /api/cloud/sim/start` with the signed-in user's robot id.
2. Poll `GET /api/cloud/sim/status` about once a second.
3. Stop polling when `state` is `running` or `dry-run`, when `error` is
   set, or when 4 minutes have passed.
4. On `running`, list `robot_sessions_visible` for that robot and attach
   the row whose `target` is `cloud`, using the pairing token already on
   the headset. Drive and video use `rosbridge_url` and `video_urls`.

Console button: the same three calls and the same polling rule. The console
does not embed a gateway token in the session row.

CLI equivalents today: `ohho cloud start`, `ohho cloud status`,
`ohho cloud stop`.

## Human TODOs

- Varun picks the provider. The recommendation is AWS `g6.xlarge` in
  `eu-west-2`.
- Confirm the hourly prices on the provider's own price list. GCP London,
  RunPod region, and Lambda region are **TODO verify** as marked above.
- Cloud account, monthly budget amount, DNS name, and ACME email.
- Ubuntu 24.04 AMI id in `eu-west-2`, NVIDIA driver, container toolkit.
- GHCR tag to pull. This environment did not pull it.
- Apply nothing here to a live account. No VM was created by these tests.
- Headset acceptance: Cloud → sim ready in under 4 minutes, drive and video
  over `wss` / WebRTC, then an idle shutdown with a billing screenshot.
  That run needs the account and the headset. It is not done.
- Reconcile `instances.json` with the provider invoice.
- OhhO-VR Cloud button and the console button, from the contract above.
- OHH-210 still owns WebRTC signaling. The gateway answers 503 on
  `/webrtc/*` until then. coturn is configured so the relay exists when
  signaling does.
- Guest `idle-watch.sh` on a real VM.
