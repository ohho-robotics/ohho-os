"""Cloud sim orchestrator. No sockets, no docker, no cloud API."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ohho.cli import build_parser, main
from ohho.cloud import (
    DEFAULT_IMAGE,
    DEFAULT_REGION,
    ENV_AWS_KEY,
    ENV_AWS_SECRET,
    ENV_DRY_RUN,
    ENV_HOURLY,
    ENV_MONTHLY,
    IDLE_AFTER_S,
    AwsEc2Provider,
    Budget,
    BudgetExceeded,
    CloudConfigError,
    CloudLiveDisabled,
    CloudOrchestrator,
    FakeProvider,
    LaunchSpec,
    accrue,
    cloud_endpoints,
    is_idle,
    overlap_seconds,
    run_cloud_start,
    run_cloud_status,
    run_cloud_stop,
)
from ohho.sessions import (
    KIND_SIM,
    TARGET_CLOUD,
    SessionError,
    SessionGone,
    SessionRegistry,
    hash_gateway_token,
)

USER = "00000000-0000-0000-0000-0000000000aa"
TOKEN = "example-gateway-token"
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
SCRUB = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    ENV_AWS_KEY,
    ENV_AWS_SECRET,
    ENV_MONTHLY,
    ENV_HOURLY,
    "OHHO_CLOUD_LIVE",
    ENV_DRY_RUN,
    "OHHO_CLOUD_REGION",
    "OHHO_CLOUD_INSTANCE_TYPE",
    "OHHO_CLOUD_STATE_DIR",
    "OHHO_SIM_IMAGE",
    "OHHO_GATEWAY_TOKEN",
    "OHHO_GATEWAY_TOKEN_FILE",
    "OHHO_GATEWAY_PAIRING_FILE",
    "OHHO_SUPABASE_URL",
    "SUPABASE_URL",
    "OHHO_SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OHHO_SUPABASE_ANON_KEY",
    "SUPABASE_ANON_KEY",
    "OHHO_SUPABASE_JWT",
    "SUPABASE_ACCESS_TOKEN",
    "OHHO_USER_ID",
    "OHHO_STATE_DIR",
)


class Clock:
    def __init__(self, moment: datetime = NOW) -> None:
        self.now = moment

    def __call__(self) -> datetime:
        return self.now


class FakeDb:
    def __init__(self, user_id: str = USER) -> None:
        self.user_id = user_id
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self._seq = 0

    def upsert(self, row: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("upsert", row))
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
        if row is None:
            raise SessionGone(session_id)
        row.update(fields)
        return dict(row)

    def delete_slot(self, robot_id: str, kind: str, target: str) -> None:
        self.calls.append(("delete", robot_id, kind, target))
        doomed = [
            session_id
            for session_id, row in self.rows.items()
            if row["robot_id"] == robot_id
            and row["kind"] == kind
            and row["target"] == target
        ]
        for session_id in doomed:
            del self.rows[session_id]

    def list_sessions(self, robot_id: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.rows.values())
        if robot_id is not None:
            rows = [row for row in rows if row["robot_id"] == robot_id]
        return [dict(row) for row in rows]


class BoomUpsert(FakeDb):
    def upsert(self, row: dict[str, Any]) -> dict[str, Any]:
        raise SessionError("upsert refused")


def _aws_env() -> dict[str, str]:
    return {
        ENV_AWS_KEY: "test-access-key",
        ENV_AWS_SECRET: "test-secret-key",
        ENV_MONTHLY: "25",
        ENV_HOURLY: "1.022",
        "OHHO_GATEWAY_TOKEN": TOKEN,
    }


def _spec() -> LaunchSpec:
    return LaunchSpec(
        robot_id="omnibot",
        region=DEFAULT_REGION,
        image=DEFAULT_IMAGE,
        instance_type="g6.xlarge",
        gateway_token=TOKEN,
    )


class CloudTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = FakeDb()
        self.clock = Clock()
        self.provider = FakeProvider()
        self.orch = CloudOrchestrator(
            self.provider,
            budget=Budget(Decimal("10"), Decimal("1")),
            token=TOKEN,
            registry=SessionRegistry(self.db, token=TOKEN, clock=self.clock),
            state_dir=self.dir,
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_start_registers_hash_and_injects_token(self) -> None:
        result = self.orch.begin("omnibot")
        record = result["record"]
        self.assertFalse(result["dry_run"])
        self.assertEqual(record["region"], "eu-west-2")
        self.assertEqual(self.provider.start_calls, 1)
        self.assertEqual(self.provider.injected["fake-1"], TOKEN)
        self.assertNotIn("gateway_token", self.provider.instances["fake-1"])
        upsert = next(call[1] for call in self.db.calls if call[0] == "upsert")
        self.assertEqual(upsert["kind"], KIND_SIM)
        self.assertEqual(upsert["target"], TARGET_CLOUD)
        self.assertEqual(upsert["token_hash"], hash_gateway_token(TOKEN))
        self.assertNotIn(TOKEN, json.dumps(upsert))
        self.assertNotIn("token=", upsert["rosbridge_url"])
        self.assertIn("/camera/front/image_raw", upsert["video_urls"][0])
        self.assertTrue(upsert["rosbridge_url"].startswith("wss://"))
        self.assertIn("/webrtc/signaling", upsert["webrtc_signal_url"])
        ledger = (self.dir / "instances.json").read_text(encoding="utf-8")
        self.assertNotIn(TOKEN, ledger)
        self.assertNotIn(TOKEN, repr(self.provider))
        self.assertNotIn(TOKEN, repr(_spec()))

    def test_status_stop_and_second_stop(self) -> None:
        self.orch.begin("omnibot")
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_cloud_status(
                robot_id="omnibot",
                provider="fake",
                provider_obj=self.provider,
                client=self.db,
                token=TOKEN,
                env={ENV_MONTHLY: "10", ENV_HOURLY: "1"},
                state_dir=self.dir,
                clock=self.clock,
                pairing_file=self.dir / "missing-pairing",
            )
        self.assertEqual(code, 0)
        self.assertIn("state=running", out.getvalue())
        self.assertNotIn(TOKEN, out.getvalue())
        stopped = io.StringIO()
        with redirect_stdout(stopped):
            stop_code = run_cloud_stop(
                robot_id="omnibot",
                provider="fake",
                provider_obj=self.provider,
                client=self.db,
                token=TOKEN,
                env={ENV_MONTHLY: "10", ENV_HOURLY: "1"},
                state_dir=self.dir,
                clock=self.clock,
                pairing_file=self.dir / "missing-pairing",
            )
        self.assertEqual(stop_code, 0)
        self.assertIn("stopped", stopped.getvalue())
        self.assertEqual(self.db.rows, {})
        self.assertEqual(self.provider.instances["fake-1"]["state"], "stopped")
        self.assertNotIn(
            TOKEN, (self.dir / "instances.json").read_text(encoding="utf-8")
        )
        again = io.StringIO()
        with redirect_stdout(again):
            again_code = run_cloud_stop(
                robot_id="omnibot",
                provider="fake",
                provider_obj=self.provider,
                client=self.db,
                token=TOKEN,
                env={ENV_MONTHLY: "10", ENV_HOURLY: "1"},
                state_dir=self.dir,
                clock=self.clock,
                pairing_file=self.dir / "missing-pairing",
            )
        self.assertEqual(again_code, 0)
        self.assertIn("no running cloud sim", again.getvalue())

    def test_foreground_start_heartbeats_then_stops(self) -> None:
        seen: dict[str, Any] = {}

        def on_beat(_count: int) -> None:
            seen["rows"] = SessionRegistry(
                self.db, token=TOKEN, clock=self.clock
            ).list_visible("omnibot")

        slept: list[float] = []
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run_cloud_start(
                robot_id="omnibot",
                provider="fake",
                provider_obj=self.provider,
                client=self.db,
                token=TOKEN,
                env={ENV_MONTHLY: "10", ENV_HOURLY: "1"},
                state_dir=self.dir,
                clock=self.clock,
                pairing_file=self.dir / "missing-pairing",
                max_beats=1,
                sleep=slept.append,
                on_beat=on_beat,
            )
        text = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(code, 0, text)
        self.assertEqual(sum(slept), 10.0)
        self.assertEqual(seen["rows"][0]["token_hash"], hash_gateway_token(TOKEN))
        self.assertNotIn(TOKEN, text)
        self.assertNotIn(TOKEN, json.dumps(self.db.calls))
        self.assertEqual(self.db.rows, {})
        self.assertEqual(self.provider.instances["fake-1"]["state"], "stopped")
        self.assertIn("wss://", text)

    def test_upsert_failure_stops_the_provider_and_writes_no_ledger(self) -> None:
        provider = FakeProvider()
        orch = CloudOrchestrator(
            provider,
            budget=Budget(Decimal("10"), Decimal("1")),
            token=TOKEN,
            registry=SessionRegistry(BoomUpsert(), token=TOKEN, clock=self.clock),
            state_dir=self.dir,
            clock=self.clock,
        )
        with self.assertRaises(SessionError):
            orch.begin("omnibot")
        self.assertEqual(provider.start_calls, 1)
        self.assertEqual(provider.stop_calls, ["fake-1"])
        self.assertEqual(provider.instances["fake-1"]["state"], "stopped")
        self.assertFalse((self.dir / "instances.json").exists())

    def test_second_start_does_not_create_another_instance(self) -> None:
        self.orch.begin("omnibot")
        with self.assertRaises(Exception) as caught:
            self.orch.begin("omnibot")
        self.assertIn("already running", str(caught.exception))
        self.assertIn("No additional VM was created", str(caught.exception))
        self.assertEqual(self.provider.start_calls, 1)

    def test_idle_boundary_and_heartbeat_reset(self) -> None:
        self.assertFalse(is_idle(NOW, NOW + timedelta(seconds=IDLE_AFTER_S - 1)))
        self.assertTrue(is_idle(NOW, NOW + timedelta(seconds=IDLE_AFTER_S)))
        self.orch.begin("omnibot")
        self.clock.now = NOW + timedelta(seconds=IDLE_AFTER_S - 1)
        self.assertEqual(self.orch.reap(), [])
        self.assertEqual(self.provider.instances["fake-1"]["state"], "running")
        self.clock.now = NOW + timedelta(seconds=IDLE_AFTER_S)
        self.assertEqual(self.orch.reap(), ["idle"])
        self.assertEqual(self.provider.instances["fake-1"]["state"], "stopped")
        self.assertEqual(self.db.rows, {})
        ledger = json.loads((self.dir / "instances.json").read_text(encoding="utf-8"))
        self.assertEqual(ledger["instances"][0]["stop_reason"], "idle")

        provider = FakeProvider()
        clock = Clock()
        db = FakeDb()
        orch = CloudOrchestrator(
            provider,
            budget=Budget(Decimal("100"), Decimal("1")),
            token=TOKEN,
            registry=SessionRegistry(db, token=TOKEN, clock=clock),
            state_dir=self.dir / "idle2",
            clock=clock,
        )
        orch.begin("omnibot")
        clock.now = NOW + timedelta(seconds=800)
        orch.heartbeat("omnibot")
        clock.now = NOW + timedelta(seconds=800 + IDLE_AFTER_S - 1)
        self.assertEqual(orch.reap(), [])
        clock.now = NOW + timedelta(seconds=800 + IDLE_AFTER_S)
        self.assertEqual(orch.reap(), ["idle"])

    def test_budget_refuses_start_and_stops_a_running_vm(self) -> None:
        self.orch._append(
            {
                "instance_id": "old",
                "robot_id": "unitree-go2",
                "provider": "fake",
                "region": "eu-west-2",
                "host": "old.cloud.example",
                "state": "stopped",
                "started_at": format_at(NOW - timedelta(hours=10)),
                "last_heartbeat_at": format_at(NOW),
                "stopped_at": format_at(NOW),
                "stop_reason": "stop",
                "session_id": "",
                "hourly_usd": "1",
                "image": DEFAULT_IMAGE,
                "instance_type": "g6.xlarge",
            }
        )
        with self.assertRaises(BudgetExceeded) as caught:
            self.orch.begin("omnibot")
        self.assertIn("No VM was created", str(caught.exception))
        self.assertEqual(self.provider.start_calls, 0)

        provider = FakeProvider()
        clock = Clock()
        db = FakeDb()
        orch = CloudOrchestrator(
            provider,
            budget=Budget(Decimal("1"), Decimal("1")),
            token=TOKEN,
            registry=SessionRegistry(db, token=TOKEN, clock=clock),
            state_dir=self.dir / "budget",
            clock=clock,
        )
        orch.begin("omnibot")
        record = orch._running_for("omnibot")
        assert record is not None
        record["started_at"] = format_at(NOW - timedelta(hours=2))
        orch._replace(record)
        self.assertEqual(orch.reap(), ["budget"])
        self.assertEqual(provider.instances["fake-1"]["state"], "stopped")
        self.assertEqual(db.rows, {})

    def test_missing_budget_stops_a_running_row(self) -> None:
        self.orch.begin("omnibot")
        self.orch.budget = None
        self.assertEqual(self.orch.reap(), ["budget-unconfigured"])
        self.assertEqual(self.provider.stop_calls, ["fake-1"])

    def test_previous_month_does_not_count(self) -> None:
        spanning = {
            "started_at": "2026-09-30T22:00:00+00:00",
            "stopped_at": "2026-10-01T02:00:00+00:00",
            "hourly_usd": "10",
        }
        september = {
            "started_at": "2026-09-01T00:00:00+00:00",
            "stopped_at": "2026-09-01T03:00:00+00:00",
            "hourly_usd": "10",
        }
        now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        self.assertEqual(
            overlap_seconds(
                datetime(2026, 9, 30, 22, tzinfo=timezone.utc),
                datetime(2026, 10, 1, 2, tzinfo=timezone.utc),
                now,
            ),
            2 * 3600,
        )
        self.assertEqual(accrue([spanning, september], now), Decimal("20.000000"))
        self.assertEqual(accrue([september], now), Decimal("0.000000"))

    def test_aws_dry_run_injects_token_without_network(self) -> None:
        provider = AwsEc2Provider(_aws_env())
        spec = _spec()

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("network")

        with (
            patch.object(urllib.request, "urlopen", side_effect=boom),
            patch.object(socket, "create_connection", side_effect=boom),
            patch.object(subprocess, "run", side_effect=boom),
        ):
            instance = provider.start(spec)
            stopped = provider.stop(instance.instance_id)
        self.assertFalse(instance.created)
        self.assertEqual(instance.state, "dry-run")
        self.assertEqual(instance.region, "eu-west-2")
        self.assertEqual(stopped.state, "stopped")
        self.assertEqual(provider.injected, [TOKEN])
        blob = json.dumps(provider.public_plans)
        self.assertNotIn(TOKEN, blob)
        self.assertNotIn("test-secret-key", blob)
        self.assertNotIn("test-access-key", blob)
        self.assertEqual(provider.public_plans[0]["token_injected"], "yes")
        self.assertNotIn(TOKEN, repr(provider))
        self.assertNotIn("test-secret-key", repr(provider))

    def test_aws_refuses_without_credentials_budget_or_live_calls(self) -> None:
        spec = _spec()
        missing_creds = AwsEc2Provider({ENV_MONTHLY: "25", ENV_HOURLY: "1"})
        with self.assertRaises(CloudConfigError) as creds:
            missing_creds.start(spec)
        self.assertIn(ENV_AWS_KEY, str(creds.exception))
        self.assertIn("No VM was created", str(creds.exception))

        missing_budget = AwsEc2Provider(
            {ENV_AWS_KEY: "test-access-key", ENV_AWS_SECRET: "test-secret-key"}
        )
        with self.assertRaises(CloudConfigError) as budget:
            missing_budget.start(spec)
        self.assertIn(ENV_MONTHLY, str(budget.exception))

        live = AwsEc2Provider(_aws_env(), live=True)
        with self.assertRaises(CloudLiveDisabled) as refused:
            live.start(spec)
        self.assertIn("No VM was created", str(refused.exception))
        self.assertEqual(live.injected, [])

        zero = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(zero):
            code = run_cloud_start(
                provider="aws",
                env={**_aws_env(), ENV_MONTHLY: "0"},
                state_dir=self.dir,
                pairing_file=self.dir / "missing-pairing",
                token=TOKEN,
            )
        self.assertEqual(code, 2)
        self.assertIn("positive", zero.getvalue())

    def test_cli_dry_run_and_live_do_not_touch_network(self) -> None:
        saved = {key: os.environ.get(key) for key in SCRUB}
        pairing = self.dir / "missing-pairing"
        try:
            for key in SCRUB:
                os.environ.pop(key, None)
            os.environ[ENV_AWS_KEY] = "test-access-key"
            os.environ[ENV_AWS_SECRET] = "test-secret-key"
            os.environ[ENV_MONTHLY] = "25"
            os.environ[ENV_HOURLY] = "1.022"
            os.environ["OHHO_GATEWAY_TOKEN"] = TOKEN
            os.environ["OHHO_GATEWAY_PAIRING_FILE"] = str(pairing)
            os.environ["OHHO_CLOUD_STATE_DIR"] = str(self.dir)

            def boom(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("network")

            with (
                patch.object(urllib.request, "urlopen", side_effect=boom),
                patch.object(socket, "create_connection", side_effect=boom),
                patch.object(subprocess, "run", side_effect=boom),
            ):
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(["cloud", "start", "--robot", "omnibot"])
                self.assertEqual(code, 0, stderr.getvalue())
                text = stdout.getvalue()
                self.assertIn("dry-run: no VM was created", text)
                self.assertIn("eu-west-2", text)
                self.assertIn(DEFAULT_IMAGE, text)
                self.assertNotIn(TOKEN, text)
                self.assertNotIn("test-secret-key", text)
                self.assertFalse((self.dir / "instances.json").exists())
                self.assertFalse(pairing.exists())

                live_out, live_err = io.StringIO(), io.StringIO()
                with redirect_stdout(live_out), redirect_stderr(live_err):
                    live_code = main(["cloud", "start", "--robot", "omnibot", "--live"])
                self.assertEqual(live_code, 2)
                self.assertIn("No VM was created", live_err.getvalue())
                self.assertNotIn(TOKEN, live_out.getvalue() + live_err.getvalue())
                self.assertFalse((self.dir / "instances.json").exists())
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_cli_without_credentials_exits_2(self) -> None:
        saved = {key: os.environ.get(key) for key in SCRUB}
        try:
            for key in SCRUB:
                os.environ.pop(key, None)
            os.environ["OHHO_GATEWAY_PAIRING_FILE"] = str(self.dir / "missing-pairing")
            os.environ["OHHO_CLOUD_STATE_DIR"] = str(self.dir)
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = main(["cloud", "start", "--robot", "omnibot"])
            self.assertEqual(code, 2)
            self.assertIn(ENV_MONTHLY, err.getvalue())
            self.assertIn("No VM was created", err.getvalue())
            self.assertFalse((self.dir / "missing-pairing").exists())
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_parser_and_endpoints(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cloud", "start", "--robot", "omnibot"])
        self.assertEqual(args.func.__name__, "_cmd_cloud_start")
        self.assertEqual(args.provider, "aws")
        self.assertFalse(args.live)
        self.assertIsNone(args.region)
        ros, videos, webrtc = cloud_endpoints("sim-1.cloud.example")
        self.assertEqual(ros, "wss://sim-1.cloud.example/rosbridge")
        self.assertNotIn("token", videos[0])
        self.assertIn("/webrtc/signaling", webrtc)

    def test_deploy_assets_are_templates(self) -> None:
        deploy = ROOT / "deploy" / "cloud"
        caddy = (deploy / "Caddyfile").read_text(encoding="utf-8")
        turn = (deploy / "turnserver.conf").read_text(encoding="utf-8")
        start = (deploy / "start-sim.sh").read_text(encoding="utf-8")
        idle = (deploy / "idle-watch.sh").read_text(encoding="utf-8")
        init = (deploy / "cloud-init.yaml").read_text(encoding="utf-8")
        blob = "\n".join([caddy, turn, start, idle, init])
        self.assertNotIn("tls internal", caddy)
        self.assertIn("{$OHHO_CLOUD_DOMAIN}", caddy)
        self.assertIn("{$OHHO_GATEWAY_TOKEN}", caddy)
        self.assertIn("/rosbridge", caddy)
        self.assertIn("/video/", caddy)
        self.assertIn("401", caddy)
        self.assertIn("REPLACE_WITH_OHHO_TURN_SECRET", turn)
        self.assertIn("3478", turn)
        self.assertIn("ghcr.io/ohho-robotics/omnibot-sim:latest", start)
        self.assertIn("docker pull", start)
        self.assertIn("900", idle)
        self.assertIn("ghcr.io/ohho-robotics/omnibot-sim:latest", init)
        self.assertIn("idle-watch.sh", init)
        self.assertNotIn(TOKEN, blob)
        self.assertNotIn("AKIA", blob)
        self.assertNotIn("BEGIN PRIVATE", blob)
        self.assertNotIn("BEGIN RSA", blob)

    def test_doc_cites_fetched_prices_and_leaves_the_choice_open(self) -> None:
        doc = (ROOT / "docs" / "cloud-sim.md").read_text(encoding="utf-8")
        for needle in (
            "Varun",
            "eu-west-2",
            "europe-west2",
            "g6.xlarge",
            "g5.xlarge",
            "g2-standard-4",
            "1.277",
            "1.022",
            "0.805",
            "0.44",
            "0.49",
            "1.29",
            "https://cloud-credits.org/prices/ec2/g5-xlarge/eu-west-2",
            "https://cloud-credits.org/prices/ec2/g6-xlarge/eu-west-2",
            "https://www.devzero.io/instances/gcp/regions/europe-west2",
            "https://www.runpod.io/gpu-instance/pricing",
            "https://lambda.ai/pricing",
            "TODO verify",
            "14 October 2026",
            "15 minutes",
            "OHHO_CLOUD_MONTHLY_BUDGET_USD",
            "OhhO-VR",
            "console",
            "dry-run",
        ):
            self.assertIn(needle, doc)


def format_at(moment: datetime) -> str:
    from ohho.sessions import format_iso

    return format_iso(moment)


if __name__ == "__main__":
    unittest.main()
