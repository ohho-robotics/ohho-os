"""Composite transport — combines a base link with an arm link.

Mobile manipulators have two independent physical links: a base and an arm.
The ``Robot`` abstraction takes
one ``Transport``; this composite wraps both and delegates by responsibility:

- ``send_velocity`` → **base** (the arm has no wheels)
- ``send_joint_command`` → **arm** (the base board doesn't drive the arm)
- ``read`` / telemetry → **merge** (odom from base, joints from arm)
- ``emergency_stop`` / ``release_stop`` → **both**
- ``connect`` / ``disconnect`` → **both**

Nothing above the transport interface changes — a single ``Robot`` drives the
whole mobile-manipulator. Telemetry callbacks fire on each constituent update,
merged with the latest snapshot from the other link.
"""

from __future__ import annotations

import time
from typing import Optional

from ..schema import (
    Telemetry,
    TransportStatus,
    Velocity,
)
from ..transport import BaseTransport, Transport


class CompositeTransport(BaseTransport):
    """Merge a base transport and an arm transport into one ``Transport``."""

    protocol = "composite"

    def __init__(self, base: Transport, arm: Transport) -> None:
        super().__init__()
        self._base = base
        self._arm = arm
        self._latest_base: Optional[Telemetry] = None
        self._latest_arm: Optional[Telemetry] = None
        self._base.on_telemetry(self._on_base_tele)
        self._arm.on_telemetry(self._on_arm_tele)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> TransportStatus:
        self._base.connect()
        self._arm.connect()
        s = self.status()
        self._emit_status(s)
        return s

    def disconnect(self) -> None:
        try:
            self._arm.disconnect()
        finally:
            self._base.disconnect()
        self._emit_status(self.status())

    def status(self) -> TransportStatus:
        bs = self._base.status()
        return TransportStatus(
            protocol=self.protocol,
            state=bs.state,
            label=f"composite({self._base.protocol}+{self._arm.protocol})",
            connected_since=bs.connected_since,
        )

    @property
    def base(self) -> Transport:
        return self._base

    @property
    def arm(self) -> Transport:
        return self._arm

    # ── commands ──────────────────────────────────────────────────────────────
    def send_velocity(self, vel: Velocity) -> None:
        self._base.send_velocity(vel)

    def send_motor(self, fl: int, fr: int, rl: int, rr: int) -> None:
        send_m = getattr(self._base, "send_motor", None)
        if callable(send_m):
            send_m(fl, fr, rl, rr)

    def send_joint_command(self, name: str, position: float) -> None:
        self._arm.send_joint_command(name, position)

    def emergency_stop(self) -> None:
        self._base.emergency_stop()
        self._arm.emergency_stop()

    def release_stop(self) -> None:
        self._base.release_stop()
        self._arm.release_stop()

    # ── telemetry ─────────────────────────────────────────────────────────────
    def read(self) -> Telemetry:
        return self._merge(self._base.read(), self._arm.read())

    def get_wheel_encoders(self) -> dict[str, int]:
        get_enc = getattr(self._base, "get_wheel_encoders", None)
        if callable(get_enc):
            return get_enc()
        return {}

    def get_imu_rate(self) -> float:
        get_imu = getattr(self._base, "get_imu_rate", None)
        if callable(get_imu):
            return get_imu()
        return 0.0

    def get_imu_samples(self) -> list[float]:
        get_s = getattr(self._base, "get_imu_samples", None)
        if callable(get_s):
            return get_s()
        return []

    def read_servo_diagnostics(self) -> list[dict]:
        read_diag = getattr(self._arm, "read_servo_diagnostics", None)
        if callable(read_diag):
            return read_diag()
        return []

    def _merge(self, base_t: Telemetry, arm_t: Telemetry) -> Telemetry:
        joints = list(arm_t.joints)
        return Telemetry(
            odom=base_t.odom,
            joints=joints,
            battery=base_t.battery,
            custom={**base_t.custom, **arm_t.custom},
            timestamp=time.time(),
        )

    def _on_base_tele(self, t: Telemetry) -> None:
        self._latest_base = t
        self._emit_merged()

    def _on_arm_tele(self, t: Telemetry) -> None:
        self._latest_arm = t
        self._emit_merged()

    def _emit_merged(self) -> None:
        base_t = self._latest_base or self._base.read()
        arm_t = self._latest_arm or self._arm.read()
        self._emit_telemetry(self._merge(base_t, arm_t))
