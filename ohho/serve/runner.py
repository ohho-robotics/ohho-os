"""Closed-loop policy runner — query ohho serve and drive through the Robot API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from ..data.recorder import _telemetry_to_state
from ..robot import Robot

BASE_ACTION_SCHEMA_VERSION = 1
_PREDICT_TIMEOUT_S = 0.2


class _HttpResponse(Protocol):
    status_code: int

    def json(self) -> dict[str, Any]: ...


@dataclass
class PolicyRunResult:
    steps: int
    stopped_reason: str
    last_odom_x: float = 0.0


def _default_post(url: str, payload: dict[str, Any], timeout_s: float) -> _HttpResponse:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        return _SimpleResponse(e.code, body)
    except urllib.error.URLError as e:
        raise ConnectionError(str(e)) from e

    return _SimpleResponse(resp.status, body)


class _SimpleResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, Any]:
        return json.loads(self._body.decode("utf-8"))


def parse_base_action(data: dict[str, Any]) -> tuple[float, float, float]:
    """Extract base velocity from a /predict JSON body (schema v1 or legacy vector)."""
    action = data.get("action") or {}
    if "base" in action and isinstance(action["base"], dict):
        b = action["base"]
        return float(b.get("vx", 0.0)), float(b.get("vy", 0.0)), float(b.get("wz", 0.0))
    vec = action.get("vector") or []
    if len(vec) >= 3:
        return float(vec[0]), float(vec[1]), float(vec[2])
    raise ValueError("predict response missing base velocity action")


class PolicyRunner:
    """Poll a policy server and apply base velocity through ``Robot.drive``."""

    def __init__(
        self,
        robot: Robot,
        url: str,
        hz: float = 10.0,
        *,
        predict_timeout_s: float = _PREDICT_TIMEOUT_S,
        post: Optional[Callable[..., _HttpResponse]] = None,
        monotonic: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        step: Optional[Callable[[], None]] = None,
    ) -> None:
        self.robot = robot
        self.url = url.rstrip("/")
        self.period_s = 1.0 / max(hz, 1e-6)
        self.predict_timeout_s = predict_timeout_s
        self._post = post or _default_post
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._step = step
        self._running = False

    def run(
        self,
        instruction: str,
        *,
        max_steps: Optional[int] = None,
        duration_s: Optional[float] = None,
        goal_x: Optional[float] = None,
    ) -> PolicyRunResult:
        """Run until max_steps, duration, goal_x (odom.x), or a predict fault."""
        stand = getattr(self.robot, "stand", None)
        if callable(stand):
            stand()
        self._running = True
        steps = 0
        reason = "completed"
        t_end = self._monotonic() + duration_s if duration_s is not None else None

        while self._running:
            if max_steps is not None and steps >= max_steps:
                break
            if t_end is not None and self._monotonic() >= t_end:
                break

            t = self.robot.telemetry()
            odom_x = t.odom.x if t and t.odom else 0.0
            if goal_x is not None and odom_x >= goal_x:
                reason = "goal_reached"
                break

            if not self._tick(instruction, odom_x):
                reason = "predict_fault"
                break

            steps += 1
            self._sleep(self.period_s)

        self.robot.stop()
        self._running = False
        t = self.robot.telemetry()
        last_x = t.odom.x if t and t.odom else 0.0
        return PolicyRunResult(steps=steps, stopped_reason=reason, last_odom_x=last_x)

    def stop(self) -> None:
        self._running = False
        self.robot.stop()

    def _tick(self, instruction: str, odom_x: float) -> bool:
        state = _telemetry_to_state(self.robot)
        t = self.robot.telemetry()
        odom = None
        if t and t.odom:
            odom = {
                "x": t.odom.x,
                "y": t.odom.y,
                "theta": t.odom.theta,
            }
        payload = {
            "instruction": instruction,
            "image_base64": "",
            "config": {"state": state, "odom": odom},
        }
        t0 = self._monotonic()
        try:
            resp = self._post(
                f"{self.url}/predict",
                payload,
                self.predict_timeout_s,
            )
        except (ConnectionError, TimeoutError, OSError):
            self.stop()
            return False

        elapsed = self._monotonic() - t0
        if elapsed > self.predict_timeout_s or resp.status_code != 200:
            self.stop()
            return False

        try:
            vx, vy, wz = parse_base_action(resp.json())
        except (ValueError, json.JSONDecodeError, TypeError):
            self.stop()
            return False

        self.robot.drive(vx=vx, vy=vy, w=wz)
        if self._step is not None:
            self._step()
        return True


__all__ = [
    "BASE_ACTION_SCHEMA_VERSION",
    "PolicyRunner",
    "PolicyRunResult",
    "parse_base_action",
]
