"""On-demand cloud sim orchestrator (OHH-216).

``ohho cloud start`` / ``status`` / ``stop`` sit behind a provider interface.
``FakeProvider`` is an in-memory stand-in for tests. ``AwsEc2Provider`` is a
dry-run skeleton: it refuses to run without credentials and a monthly budget,
and this build never calls AWS. No account, VM, DNS record, or spend is created.

A running sim is registered through the OHH-206 session registry (``target=cloud``).
The gateway token is injected into the provider request and stored only as a
SHA-256 hash on the session row. The VM is stopped after 15 minutes without a
heartbeat, and a start is refused when this month's local ledger has reached
the configured cap.

The local ledger is not a cloud invoice. Reconcile it with the provider bill
before treating it as the amount AWS charged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .registry import UnknownRobot, get_spec
from .sessions import (
    FRONT_CAMERA_TOPIC,
    HEARTBEAT_INTERVAL_S,
    KIND_SIM,
    TARGET_CLOUD,
    SessionError,
    SessionRegistry,
    format_iso,
    open_supabase_client,
    parse_iso,
    resolve_gateway_token,
    validate_gateway_token,
)

IDLE_AFTER_S = 15 * 60
DEFAULT_REGION = "eu-west-2"
DEFAULT_INSTANCE_TYPE = "g6.xlarge"
DEFAULT_IMAGE = "ghcr.io/ohho-robotics/omnibot-sim:latest"
LEDGER_VERSION = 1
MONEY = Decimal("0.000001")

ENV_MONTHLY = "OHHO_CLOUD_MONTHLY_BUDGET_USD"
ENV_HOURLY = "OHHO_CLOUD_HOURLY_USD"
ENV_AWS_KEY = "OHHO_AWS_ACCESS_KEY_ID"
ENV_AWS_SECRET = "OHHO_AWS_SECRET_ACCESS_KEY"
ENV_LIVE = "OHHO_CLOUD_LIVE"
ENV_DRY_RUN = "OHHO_CLOUD_DRY_RUN"
ENV_REGION = "OHHO_CLOUD_REGION"
ENV_IMAGE = "OHHO_SIM_IMAGE"
ENV_INSTANCE = "OHHO_CLOUD_INSTANCE_TYPE"
ENV_STATE = "OHHO_CLOUD_STATE_DIR"

_SECRET_KEYS = frozenset(
    {
        "gateway_token",
        "ohho_gateway_token",
        "token",
        "aws_secret_access_key",
        "ohho_aws_secret_access_key",
        "aws_access_key_id",
        "secret",
    }
)


class CloudError(RuntimeError):
    """The cloud orchestrator could not finish a request."""


class CloudConfigError(CloudError):
    """Credentials, budget, or arguments are not enough to continue."""


class BudgetExceeded(CloudError):
    """This month's ledger has reached the configured cap."""


class CloudLiveDisabled(CloudError):
    """This build refuses live provider calls. No VM is created."""


Clock = Callable[[], datetime]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class Budget:
    """Operator-configured cap. Both numbers come from the environment."""

    monthly_usd: Decimal
    hourly_usd: Decimal


@dataclass(frozen=True)
class LaunchSpec:
    """What a provider would start. ``repr`` hides the gateway token."""

    robot_id: str
    region: str
    image: str
    instance_type: str
    gateway_token: str

    def __repr__(self) -> str:
        return (
            "LaunchSpec("
            f"robot_id={self.robot_id!r}, region={self.region!r}, "
            f"image={self.image!r}, instance_type={self.instance_type!r}, "
            "gateway_token='[redacted]')"
        )


@dataclass(frozen=True)
class Instance:
    """Public view of a provider instance. It has no token and no credentials."""

    instance_id: str
    state: str
    host: str
    region: str
    created: bool
    provider: str
    image: str
    instance_type: str


class CloudProvider(Protocol):
    """Start, stop, and report. Implementations must not log the gateway token."""

    name: str
    creates_instances: bool

    def start(self, spec: LaunchSpec) -> Instance: ...

    def stop(self, instance_id: str) -> Instance: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_state_dir() -> Path:
    override = os.environ.get(ENV_STATE, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ohho" / "cloud"


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is None:
        return os.environ
    return env


def _first(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = env.get(key, "")
        if value.strip():
            return value.strip()
    return ""


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def _parse_usd(name: str, raw: str) -> Decimal:
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise CloudConfigError(
            f"{name} must be a positive USD amount. No VM was created."
        ) from exc
    if not value.is_finite() or value <= 0:
        raise CloudConfigError(
            f"{name} must be a positive USD amount. No VM was created."
        )
    return value


def load_budget(env: Mapping[str, str] | None = None) -> Budget | None:
    """Return the cap, or ``None`` when either number is unset."""
    values = _env(env)
    monthly = _first(values, (ENV_MONTHLY,))
    hourly = _first(values, (ENV_HOURLY,))
    if not monthly and not hourly:
        return None
    if not monthly or not hourly:
        missing = ENV_MONTHLY if not monthly else ENV_HOURLY
        raise CloudConfigError(
            f"set {ENV_MONTHLY} and {ENV_HOURLY} (missing {missing}). "
            "No VM was created."
        )
    return Budget(
        monthly_usd=_parse_usd(ENV_MONTHLY, monthly),
        hourly_usd=_parse_usd(ENV_HOURLY, hourly),
    )


def aws_credentials_present(env: Mapping[str, str] | None = None) -> bool:
    """True when both an access key id and a secret are non-empty.

    ``OHHO_AWS_*`` wins over ``AWS_*``. The secret is not returned.
    """
    values = _env(env)
    key = _first(values, (ENV_AWS_KEY, "AWS_ACCESS_KEY_ID"))
    secret = _first(values, (ENV_AWS_SECRET, "AWS_SECRET_ACCESS_KEY"))
    return bool(key and secret)


def month_bounds(moment: datetime) -> tuple[datetime, datetime]:
    current = moment.astimezone(timezone.utc)
    start = datetime(current.year, current.month, 1, tzinfo=timezone.utc)
    if current.month == 12:
        end = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def overlap_seconds(start: datetime, end: datetime, now: datetime) -> int:
    """Whole seconds of ``[start, end)`` that fall in ``now``'s UTC month."""
    if end <= start:
        return 0
    month_start, month_end = month_bounds(now)
    left = max(start, month_start)
    right = min(end, month_end)
    if right <= left:
        return 0
    delta = right - left
    return delta.days * 86400 + delta.seconds


def is_idle(last_heartbeat: datetime, now: datetime) -> bool:
    """True once 15 minutes have passed with no heartbeat."""
    if last_heartbeat.tzinfo is None or now.tzinfo is None:
        raise CloudError("heartbeat timestamps must be timezone-aware")
    return now - last_heartbeat >= timedelta(seconds=IDLE_AFTER_S)


def idle_deadline(last_heartbeat: datetime) -> datetime:
    return last_heartbeat + timedelta(seconds=IDLE_AFTER_S)


def accrue(records: list[dict[str, Any]], now: datetime) -> Decimal:
    """USD charged in ``now``'s month, including time a still-running VM has used."""
    total = Decimal("0")
    for record in records:
        start = parse_iso(str(record["started_at"]))
        stopped = record.get("stopped_at")
        end = parse_iso(str(stopped)) if stopped else now
        if end > now:
            end = now
        seconds = overlap_seconds(start, end, now)
        hourly = Decimal(str(record["hourly_usd"]))
        total += hourly * Decimal(seconds) / Decimal(3600)
    return _money(total)


def cloud_endpoints(host: str) -> tuple[str, list[str], str]:
    """TLS URLs on port 443. The headset appends ``token``; these strings do not."""
    shown = host.strip()
    if ":" in shown and not shown.startswith("["):
        shown = f"[{shown}]"
    topic = FRONT_CAMERA_TOPIC
    return (
        f"wss://{shown}/rosbridge",
        [f"https://{shown}/video/stream?topic={topic}"],
        f"wss://{shown}/webrtc/signaling",
    )


def _reject_secret_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                raise CloudError("cloud state must not store secrets")
            _reject_secret_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secret_keys(item)


def _validate_region(region: str) -> str:
    cleaned = region.strip()
    if (
        not cleaned
        or any(character.isspace() for character in cleaned)
        or "/" in cleaned
        or "\\" in cleaned
    ):
        raise CloudConfigError(
            "region must be a provider region id such as eu-west-2. No VM was created."
        )
    return cleaned


def _validate_label(name: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned or any(character.isspace() for character in cleaned):
        raise CloudConfigError(
            f"{name} must not contain whitespace. No VM was created."
        )
    return cleaned


class FakeProvider:
    """In-memory provider. ``injected`` holds tokens for tests and is not serialized."""

    name = "fake"
    creates_instances = True

    def __init__(self) -> None:
        self.instances: dict[str, dict[str, str]] = {}
        self.injected: dict[str, str] = {}
        self.start_calls = 0
        self.stop_calls: list[str] = []
        self._seq = 0

    def __repr__(self) -> str:
        return f"FakeProvider(n={len(self.instances)})"

    def start(self, spec: LaunchSpec) -> Instance:
        self.start_calls += 1
        if not spec.gateway_token:
            raise CloudConfigError("refusing to inject an empty session token")
        self._seq += 1
        instance_id = f"fake-{self._seq}"
        host = f"sim-{self._seq}.cloud.example"
        self.instances[instance_id] = {
            "state": "running",
            "host": host,
            "region": spec.region,
            "image": spec.image,
            "instance_type": spec.instance_type,
        }
        self.injected[instance_id] = spec.gateway_token
        return self._view(instance_id)

    def stop(self, instance_id: str) -> Instance:
        self.stop_calls.append(instance_id)
        row = self.instances.get(instance_id)
        if row is None:
            return Instance(
                instance_id=instance_id,
                state="stopped",
                host="",
                region="",
                created=False,
                provider=self.name,
                image="",
                instance_type="",
            )
        row["state"] = "stopped"
        return self._view(instance_id)

    def _view(self, instance_id: str) -> Instance:
        row = self.instances[instance_id]
        return Instance(
            instance_id=instance_id,
            state=row["state"],
            host=row["host"],
            region=row["region"],
            created=row["state"] == "running",
            provider=self.name,
            image=row["image"],
            instance_type=row["instance_type"],
        )


class AwsEc2Provider:
    """Dry-run EC2 skeleton for ``g6.xlarge`` in ``eu-west-2``.

    There is no boto3 import and no HTTP client. ``start`` checks credentials
    and budget, then returns a plan. Live mode raises ``CloudLiveDisabled``
    after those checks and still does not contact AWS.
    """

    name = "aws"
    creates_instances = False

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        live: bool = False,
        dry_run: bool | None = None,
    ) -> None:
        self._env = dict(_env(env))
        live_env = self._env.get(ENV_LIVE, "").strip() == "1"
        self.live = live or live_env
        if dry_run is None:
            dry_run = self._env.get(ENV_DRY_RUN, "1").strip() != "0"
        self.dry_run = bool(dry_run) and not self.live
        self.public_plans: list[dict[str, str]] = []
        self.injected: list[str] = []

    def __repr__(self) -> str:
        return f"AwsEc2Provider(mode={self.mode})"

    @property
    def mode(self) -> str:
        if self.live or not self.dry_run:
            return "refused-live"
        return "dry-run"

    def start(self, spec: LaunchSpec) -> Instance:
        budget = self._require_credentials_and_budget()
        if self.mode != "dry-run":
            raise CloudLiveDisabled(
                "refusing live cloud calls. This build does not create VMs. "
                "No VM was created."
            )
        if not spec.gateway_token:
            raise CloudConfigError(
                "refusing to inject an empty session token. No VM was created."
            )
        if spec.gateway_token in spec.image or spec.gateway_token in spec.region:
            raise CloudConfigError(
                "refusing to place the gateway token in the launch plan. "
                "No VM was created."
            )
        plan = {
            "provider": self.name,
            "region": spec.region,
            "instance_type": spec.instance_type,
            "image": spec.image,
            "token_injected": "yes",
            "monthly_usd": format(budget.monthly_usd, "f"),
            "hourly_usd": format(budget.hourly_usd, "f"),
            "user_data_env": "OHHO_GATEWAY_TOKEN",
            "dry_run": "yes",
        }
        if spec.gateway_token in json.dumps(plan):
            raise CloudError("refusing to put the gateway token in the provider plan")
        self.public_plans.append(plan)
        self.injected.append(spec.gateway_token)
        return Instance(
            instance_id="dry-run",
            state="dry-run",
            host="",
            region=spec.region,
            created=False,
            provider=self.name,
            image=spec.image,
            instance_type=spec.instance_type,
        )

    def stop(self, instance_id: str) -> Instance:
        self._require_credentials_and_budget()
        if self.mode != "dry-run":
            raise CloudLiveDisabled("refusing live cloud calls. No VM was created.")
        return Instance(
            instance_id=instance_id,
            state="stopped",
            host="",
            region="",
            created=False,
            provider=self.name,
            image="",
            instance_type="",
        )

    def _require_credentials_and_budget(self) -> Budget:
        if not aws_credentials_present(self._env):
            raise CloudConfigError(
                f"set {ENV_AWS_KEY} and {ENV_AWS_SECRET} "
                "(or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY). "
                "No VM was created."
            )
        budget = load_budget(self._env)
        if budget is None:
            raise CloudConfigError(
                f"set {ENV_MONTHLY} and {ENV_HOURLY}. No VM was created."
            )
        return budget


def build_provider(
    name: str,
    env: Mapping[str, str] | None = None,
    *,
    live: bool = False,
) -> CloudProvider:
    cleaned = name.strip().lower()
    if cleaned == "fake":
        return FakeProvider()
    if cleaned == "aws":
        return AwsEc2Provider(env, live=live)
    raise CloudConfigError(
        f"unknown cloud provider {name!r}. Known providers: aws, fake. "
        "No VM was created."
    )


class CloudOrchestrator:
    """Budget, idle shutdown, token injection, and ``robot_sessions`` registration."""

    def __init__(
        self,
        provider: CloudProvider,
        *,
        budget: Budget | None,
        token: str,
        registry: SessionRegistry | None = None,
        state_dir: Path,
        clock: Clock | None = None,
        image: str = DEFAULT_IMAGE,
        instance_type: str = DEFAULT_INSTANCE_TYPE,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.token = validate_gateway_token(token) if token else ""
        self.registry = registry
        self.state_dir = state_dir
        self.clock = clock or utc_now
        self.image = image
        self.instance_type = instance_type
        self.last_plan: dict[str, str] | None = None

    def spent(self, now: datetime | None = None) -> Decimal:
        moment = now or self.clock()
        return accrue(self._records(), moment)

    def begin(self, robot_id: str, *, region: str = DEFAULT_REGION) -> dict[str, Any]:
        """Start or, for the AWS skeleton, return a dry-run plan and create nothing."""
        get_spec(robot_id)
        chosen = _validate_region(region)
        image = _validate_label("image", self.image)
        instance_type = _validate_label("instance type", self.instance_type)
        if self.budget is None:
            raise CloudConfigError(
                f"set {ENV_MONTHLY} and {ENV_HOURLY}. No VM was created."
            )
        self.reap()
        moment = self.clock()
        spent = self.spent(moment)
        if spent >= self.budget.monthly_usd:
            raise BudgetExceeded(
                "monthly cloud budget cap reached "
                f"({format(spent, 'f')} USD of "
                f"{format(self.budget.monthly_usd, 'f')} USD). "
                "No VM was created."
            )
        if self._running_for(robot_id) is not None:
            raise CloudError(
                f"cloud sim already running for {robot_id}. "
                "No additional VM was created."
            )
        if not self.token:
            raise CloudConfigError("set OHHO_GATEWAY_TOKEN. No VM was created.")
        if self.token in image:
            raise CloudConfigError(
                "image name must not contain the gateway token. No VM was created."
            )
        spec = LaunchSpec(
            robot_id=robot_id,
            region=chosen,
            image=image,
            instance_type=instance_type,
            gateway_token=self.token,
        )
        instance = self.provider.start(spec)
        if not instance.created:
            plan = {
                "provider": instance.provider,
                "region": instance.region or chosen,
                "instance_type": instance.instance_type or instance_type,
                "image": instance.image or image,
                "token_injected": "yes",
                "monthly_usd": format(self.budget.monthly_usd, "f"),
                "hourly_usd": format(self.budget.hourly_usd, "f"),
            }
            if self.token in json.dumps(plan):
                raise CloudError(
                    "refusing to record the gateway token in the dry-run plan"
                )
            self.last_plan = plan
            return {"dry_run": True, "plan": plan, "record": None}
        registered = False
        try:
            if self.registry is None:
                raise CloudConfigError(
                    "session registry is required to register a running cloud sim"
                )
            rosbridge, videos, webrtc = cloud_endpoints(instance.host)
            saved = self.registry.upsert(
                robot_id=robot_id,
                kind=KIND_SIM,
                target=TARGET_CLOUD,
                rosbridge_url=rosbridge,
                video_urls=videos,
                webrtc_signal_url=webrtc,
            )
            registered = True
            stamped = format_iso(moment)
            record = {
                "instance_id": instance.instance_id,
                "robot_id": robot_id,
                "provider": instance.provider,
                "region": instance.region,
                "host": instance.host,
                "state": "running",
                "started_at": stamped,
                "last_heartbeat_at": stamped,
                "stopped_at": None,
                "stop_reason": "",
                "session_id": str(saved["id"]),
                "hourly_usd": format(self.budget.hourly_usd, "f"),
                "image": image,
                "instance_type": instance_type,
                "rosbridge_url": saved.get("rosbridge_url", rosbridge),
                "video_urls": list(saved.get("video_urls") or videos),
                "webrtc_signal_url": saved.get("webrtc_signal_url", webrtc),
            }
            _reject_secret_keys(record)
            if self.token in json.dumps(record):
                raise CloudError("refusing to write the gateway token into cloud state")
            self._append(record)
            return {"dry_run": False, "plan": None, "record": record}
        except Exception:
            self.provider.stop(instance.instance_id)
            if registered and self.registry is not None:
                try:
                    self.registry.end(robot_id, KIND_SIM, TARGET_CLOUD)
                except SessionError:
                    pass
            raise

    def heartbeat(self, robot_id: str) -> dict[str, Any]:
        record = self._running_for(robot_id)
        if record is None:
            raise CloudError(f"no running cloud sim for {robot_id}")
        if self.registry is None or not record.get("session_id"):
            raise CloudError("session registry is not available")
        moment = self.clock()
        saved = self.registry.heartbeat(str(record["session_id"]))
        record["last_heartbeat_at"] = format_iso(moment)
        self._replace(record)
        return saved

    def reap(self, now: datetime | None = None) -> list[str]:
        """Stop idle VMs and any VM that has reached the monthly cap."""
        moment = now or self.clock()
        reasons: list[str] = []
        if self.budget is None:
            for record in list(self._running()):
                self._halt(record, moment, "budget-unconfigured")
                reasons.append("budget-unconfigured")
            return reasons
        for record in list(self._running()):
            if self.spent(moment) >= self.budget.monthly_usd:
                self._halt(record, moment, "budget")
                reasons.append("budget")
                continue
            last = parse_iso(str(record["last_heartbeat_at"]))
            if is_idle(last, moment):
                self._halt(record, moment, "idle")
                reasons.append("idle")
        return reasons

    def stop(self, robot_id: str) -> dict[str, Any] | None:
        record = self._running_for(robot_id)
        if record is None:
            return None
        return self._halt(record, self.clock(), "stop")

    def run_loop(
        self,
        robot_id: str,
        *,
        sleep: Sleep,
        max_beats: int | None,
        stop_path: Path,
        on_beat: Callable[[int], None] | None = None,
    ) -> int:
        beats = 0
        while max_beats is None or beats < max_beats:
            if not _pause(HEARTBEAT_INTERVAL_S, sleep, stop_path.exists):
                break
            if self._running_for(robot_id) is None:
                break
            self.heartbeat(robot_id)
            self.reap()
            if self._running_for(robot_id) is None:
                break
            beats += 1
            if on_beat is not None:
                on_beat(beats)
        return beats

    def _halt(
        self, record: dict[str, Any], moment: datetime, reason: str
    ) -> dict[str, Any]:
        if str(record.get("provider") or "") != self.provider.name:
            raise CloudError(
                "ledger instance "
                f"{record.get('instance_id')} uses provider "
                f"{record.get('provider')}. Stop it with that provider. "
                "No VM was created."
            )
        self.provider.stop(str(record["instance_id"]))
        record["state"] = "stopped"
        record["stopped_at"] = format_iso(moment)
        record["stop_reason"] = reason
        self._replace(record)
        if record.get("session_id"):
            if self.registry is None:
                raise CloudConfigError(
                    "cloud sim stopped, but the session row was not deleted "
                    "(Supabase credentials missing)"
                )
            self.registry.end(str(record["robot_id"]), KIND_SIM, TARGET_CLOUD)
        return record

    def _path(self) -> Path:
        return self.state_dir / "instances.json"

    def _records(self) -> list[dict[str, Any]]:
        path = self._path()
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CloudError(f"cloud ledger is unreadable: {path}") from exc
        _reject_secret_keys(payload)
        if not isinstance(payload, dict) or payload.get("version") != LEDGER_VERSION:
            raise CloudError(f"cloud ledger version is not {LEDGER_VERSION}")
        rows = payload.get("instances")
        if not isinstance(rows, list):
            raise CloudError("cloud ledger is missing instances")
        return [dict(row) for row in rows]

    def _running(self) -> list[dict[str, Any]]:
        return [row for row in self._records() if row.get("state") == "running"]

    def _running_for(self, robot_id: str) -> dict[str, Any] | None:
        for row in self._running():
            if row.get("robot_id") == robot_id:
                return row
        return None

    def _write(self, rows: list[dict[str, Any]]) -> None:
        _reject_secret_keys(rows)
        blob = json.dumps({"version": LEDGER_VERSION, "instances": rows}, indent=2)
        if self.token and self.token in blob:
            raise CloudError("refusing to write the gateway token into cloud state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self._path()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(blob + "\n", encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(path)

    def _append(self, record: dict[str, Any]) -> None:
        rows = self._records()
        rows.append(record)
        self._write(rows)

    def _replace(self, record: dict[str, Any]) -> None:
        rows = self._records()
        replaced = False
        for index, row in enumerate(rows):
            if row.get("instance_id") == record.get("instance_id"):
                rows[index] = record
                replaced = True
                break
        if not replaced:
            rows.append(record)
        self._write(rows)


def _pause(interval_s: float, sleep: Sleep, should_stop: Callable[[], bool]) -> bool:
    whole = int(interval_s)
    frac = interval_s - whole
    for _ in range(whole):
        if should_stop():
            return False
        sleep(1.0)
    if frac > 0 and not should_stop():
        sleep(frac)
    return not should_stop()


def _announce_token(token: str, source: str) -> None:
    if source == "generated":
        print(f"gateway pairing code (enter on the headset once): {token}")
        return
    if source == "argument":
        print("gateway token: provided (value not printed)")
        return
    if source == "pairing":
        print(
            "gateway token: pairing file ~/.ohho/gateway/pairing-code "
            "(value not printed)"
        )
        return
    if source == "file":
        print("gateway token: OHHO_GATEWAY_TOKEN_FILE (value not printed)")
        return
    print("gateway token: OHHO_GATEWAY_TOKEN (value not printed)")


def _print_plan(plan: Mapping[str, str]) -> None:
    print("dry-run: no VM was created")
    print(f"provider: {plan.get('provider')}")
    print(f"region: {plan.get('region')}")
    print(f"instance_type: {plan.get('instance_type')}")
    print(f"image: {plan.get('image')}")
    print(f"token_injected: {plan.get('token_injected')}")
    print(f"monthly cap USD: {plan.get('monthly_usd')}")
    print(f"hourly USD: {plan.get('hourly_usd')}")


def _print_running(record: Mapping[str, Any]) -> None:
    print(
        f"cloud sim robot={record.get('robot_id')} "
        f"provider={record.get('provider')} state=running "
        f"region={record.get('region')}"
    )
    print(f"host: {record.get('host')}")
    print(f"rosbridge: {record.get('rosbridge_url')}")
    videos = record.get("video_urls") or []
    if videos:
        print("video: " + ", ".join(str(item) for item in videos))
    if record.get("webrtc_signal_url"):
        print(f"webrtc: {record.get('webrtc_signal_url')}")
    print(f"session: {record.get('session_id')}")
    print(
        f"heartbeat every {int(HEARTBEAT_INTERVAL_S)}s — "
        f"idle shutdown after {IDLE_AFTER_S // 60} min without a heartbeat — "
        f"Ctrl-C or `ohho cloud stop --robot {record.get('robot_id')}` stops it"
    )


def _resolve_token(
    *,
    token: str | None,
    env: Mapping[str, str] | None,
    pairing_file: Path | None,
    generate: bool,
) -> tuple[str, str]:
    if token is not None:
        return validate_gateway_token(token), "argument"
    return resolve_gateway_token(env=env, pairing_file=pairing_file, generate=generate)


def _orchestrator(
    *,
    robot_id: str,
    provider_name: str,
    provider_obj: CloudProvider | None,
    region: str | None,
    live: bool,
    client: Any,
    token: str | None,
    env: Mapping[str, str] | None,
    pairing_file: Path | None,
    state_dir: Path | None,
    clock: Clock | None,
) -> tuple[CloudOrchestrator, str, str, str]:
    del robot_id
    values = _env(env)
    provider = provider_obj or build_provider(provider_name, values, live=live)
    budget = load_budget(values)
    if budget is None:
        raise CloudConfigError(
            f"set {ENV_MONTHLY} and {ENV_HOURLY}. No VM was created."
        )
    if isinstance(provider, AwsEc2Provider):
        provider._require_credentials_and_budget()
        if provider.mode != "dry-run":
            raise CloudLiveDisabled(
                "refusing live cloud calls. This build does not create VMs. "
                "No VM was created."
            )
    directory = state_dir or default_state_dir()
    image = _first(values, (ENV_IMAGE,)) or DEFAULT_IMAGE
    instance_type = _first(values, (ENV_INSTANCE,)) or DEFAULT_INSTANCE_TYPE
    chosen_region = region or _first(values, (ENV_REGION,)) or DEFAULT_REGION
    resolved, source = _resolve_token(
        token=token,
        env=values,
        pairing_file=pairing_file,
        generate=provider.creates_instances,
    )
    registry = None
    if provider.creates_instances:
        active = client if client is not None else open_supabase_client(values)
        registry = SessionRegistry(active, token=resolved, clock=clock)
    orchestrator = CloudOrchestrator(
        provider,
        budget=budget,
        token=resolved,
        registry=registry,
        state_dir=directory,
        clock=clock,
        image=image,
        instance_type=instance_type,
    )
    return orchestrator, resolved, source, _validate_region(chosen_region)


def run_cloud_start(
    *,
    robot_id: str = "omnibot",
    provider: str = "aws",
    region: str | None = None,
    live: bool = False,
    client: Any = None,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    state_dir: Path | None = None,
    clock: Clock | None = None,
    provider_obj: CloudProvider | None = None,
    sleep: Sleep | None = None,
    max_beats: int | None = None,
    on_beat: Callable[[int], None] | None = None,
) -> int:
    """Start a cloud sim, or print an AWS dry-run plan and create no VM."""
    code = 0
    orchestrator: CloudOrchestrator | None = None
    started = False
    try:
        try:
            get_spec(robot_id)
        except UnknownRobot as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        orchestrator, resolved, source, chosen = _orchestrator(
            robot_id=robot_id,
            provider_name=provider,
            provider_obj=provider_obj,
            region=region,
            live=live,
            client=client,
            token=token,
            env=env,
            pairing_file=pairing_file,
            state_dir=state_dir,
            clock=clock,
        )
        _announce_token(resolved, source)
        result = orchestrator.begin(robot_id, region=chosen)
        if result["dry_run"]:
            _print_plan(result["plan"])
            return 0
        started = True
        _print_running(result["record"])
        stop_path = orchestrator.state_dir / f"{robot_id}.stop"
        stop_path.unlink(missing_ok=True)
        try:
            orchestrator.run_loop(
                robot_id,
                sleep=sleep or time.sleep,
                max_beats=max_beats,
                stop_path=stop_path,
                on_beat=on_beat,
            )
        except KeyboardInterrupt:
            print("\nstopping.")
    except (CloudError, SessionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    finally:
        if started and orchestrator is not None:
            try:
                orchestrator.stop(robot_id)
            except (CloudError, SessionError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                code = 2
    return code


def _load_for_read(
    *,
    provider_name: str,
    provider_obj: CloudProvider | None,
    live: bool,
    client: Any,
    token: str | None,
    env: Mapping[str, str] | None,
    pairing_file: Path | None,
    state_dir: Path | None,
    clock: Clock | None,
) -> CloudOrchestrator:
    values = _env(env)
    directory = state_dir or default_state_dir()
    reader = CloudOrchestrator(
        FakeProvider(),
        budget=None,
        token="",
        registry=None,
        state_dir=directory,
        clock=clock,
    )
    running = reader._running()
    name = provider_name
    if running:
        name = str(running[0].get("provider") or provider_name)
    provider = provider_obj or build_provider(name, values, live=live)
    budget: Budget | None
    try:
        budget = load_budget(values)
    except CloudConfigError:
        budget = None
    resolved = ""
    registry = None
    if provider.creates_instances and running:
        try:
            resolved, _source = _resolve_token(
                token=token,
                env=values,
                pairing_file=pairing_file,
                generate=False,
            )
            active = client if client is not None else open_supabase_client(values)
            registry = SessionRegistry(active, token=resolved, clock=clock)
        except (CloudError, SessionError):
            resolved = ""
            registry = None
    image = _first(values, (ENV_IMAGE,)) or DEFAULT_IMAGE
    instance_type = _first(values, (ENV_INSTANCE,)) or DEFAULT_INSTANCE_TYPE
    orchestrator = CloudOrchestrator(
        provider,
        budget=budget,
        token=resolved,
        registry=registry,
        state_dir=directory,
        clock=clock,
        image=image,
        instance_type=instance_type,
    )
    return orchestrator


def _print_status(orchestrator: CloudOrchestrator, robot_id: str) -> None:
    matches = [
        row for row in orchestrator._records() if row.get("robot_id") == robot_id
    ]
    record = matches[-1] if matches else None
    running = [
        row for row in orchestrator._running() if row.get("robot_id") == robot_id
    ]
    if not running:
        print(f"no running cloud sim for {robot_id}")
        if record and record.get("state") == "stopped":
            print(f"last stop reason: {record.get('stop_reason') or 'stopped'}")
        return
    current = running[-1]
    last = parse_iso(str(current["last_heartbeat_at"]))
    deadline = idle_deadline(last)
    spent = orchestrator.spent()
    cap = (
        format(orchestrator.budget.monthly_usd, "f")
        if orchestrator.budget is not None
        else "unset"
    )
    print(
        f"cloud sim robot={current.get('robot_id')} "
        f"provider={current.get('provider')} state={current.get('state')} "
        f"region={current.get('region')}"
    )
    print(f"host: {current.get('host')}")
    print(f"session: {current.get('session_id')}")
    print(f"idle deadline: {format_iso(deadline)}")
    print(f"month spend USD: {format(spent, 'f')} of {cap}")


def run_cloud_status(
    *,
    robot_id: str = "omnibot",
    provider: str = "aws",
    live: bool = False,
    client: Any = None,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    state_dir: Path | None = None,
    clock: Clock | None = None,
    provider_obj: CloudProvider | None = None,
) -> int:
    """Print status. Idle and over-budget sims are stopped first."""
    try:
        get_spec(robot_id)
    except UnknownRobot as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        orchestrator = _load_for_read(
            provider_name=provider,
            provider_obj=provider_obj,
            live=live,
            client=client,
            token=token,
            env=env,
            pairing_file=pairing_file,
            state_dir=state_dir,
            clock=clock,
        )
        orchestrator.reap()
        _print_status(orchestrator, robot_id)
        return 0
    except (CloudError, SessionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def run_cloud_stop(
    *,
    robot_id: str = "omnibot",
    provider: str = "aws",
    live: bool = False,
    client: Any = None,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    pairing_file: Path | None = None,
    state_dir: Path | None = None,
    clock: Clock | None = None,
    provider_obj: CloudProvider | None = None,
) -> int:
    """Stop one cloud sim. A second stop is a no-op."""
    try:
        get_spec(robot_id)
    except UnknownRobot as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    code = 0
    try:
        orchestrator = _load_for_read(
            provider_name=provider,
            provider_obj=provider_obj,
            live=live,
            client=client,
            token=token,
            env=env,
            pairing_file=pairing_file,
            state_dir=state_dir,
            clock=clock,
        )
        stop_path = orchestrator.state_dir / f"{robot_id}.stop"
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        stop_path.write_text("stop\n", encoding="utf-8")
        orchestrator.reap()
        stopped = orchestrator.stop(robot_id)
        if stopped is None:
            print(f"no running cloud sim for {robot_id}")
        else:
            print(f"stopped cloud sim for {robot_id} ({stopped.get('stop_reason')})")
    except (CloudError, SessionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    return code
