"""Wrap an ``ohho`` transport as a service.

``connect`` is start and ``disconnect`` is stop. Readiness is the link
(``ConnectionState.CONNECTED``). Liveness is telemetry freshness.

When the transport is a :class:`ohho.safety.SafetyGate`, the freshness window
is that gate's public ``profile.state_timeout_s`` and a sample counts only
when the gate itself would refresh state (no fault, and orientation present
when the profile requires it). The wrapper does not read private gate fields,
does not call ``tick``, and does not send commands. It does not change the
gate's caps, watchdogs, arming, or e-stop latch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ohho.safety import SafetyGate, SafetyProfile
from ohho.schema import ConnectionState, Telemetry, TransportStatus
from ohho.transport import Transport

from ohho_runtime.log import note_exception
from ohho_runtime.service.base import ServiceBase


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        note_exception(
            exc,
            service="transport",
            event="telemetry_field",
            level="DEBUG",
            message="telemetry field is not a float",
        )
        return math.nan


def _orientation(sample: Telemetry) -> tuple[float, float] | None:
    """Same inputs the gate uses: ``roll``/``pitch`` or ``imu_rpy``."""
    custom = sample.custom
    if "roll" in custom or "pitch" in custom:
        return _as_float(custom.get("roll")), _as_float(custom.get("pitch"))
    rpy = custom.get("imu_rpy")
    if isinstance(rpy, (list, tuple)) and len(rpy) >= 2:
        return _as_float(rpy[0]), _as_float(rpy[1])
    return None


def _faulted(profile: SafetyProfile, sample: Telemetry) -> bool:
    """True when the gate would reject this sample instead of refreshing state."""
    custom = sample.custom
    if "error_code" in custom:
        try:
            code = int(custom["error_code"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            note_exception(
                exc,
                service="transport",
                event="telemetry_error_code",
                level="DEBUG",
                message="telemetry error_code is not an int",
            )
            return True
        if code != 0:
            return True
    orientation = _orientation(sample)
    if orientation is not None:
        limit = profile.tilt_limit_rad
        roll, pitch = orientation
        if not (abs(roll) <= limit and abs(pitch) <= limit):
            return True
    return False


def sample_refreshes_state(profile: SafetyProfile, sample: Telemetry) -> bool:
    """Whether this sample refreshes SafetyGate state freshness.

    A faulted sample does not. A sample with no orientation does not when
    ``profile.require_orientation`` is set. The age limit is
    ``profile.state_timeout_s``, applied by the service liveness window.
    """
    if _faulted(profile, sample):
        return False
    if profile.require_orientation and _orientation(sample) is None:
        return False
    return True


class TransportService(ServiceBase):
    """Service view of an existing :class:`ohho.transport.Transport`."""

    def __init__(
        self,
        transport: Transport,
        *,
        clock: Callable[[], float] | None = None,
        readiness_timeout_s: float = 5.0,
        liveness_timeout_s: float = 1.0,
        grace_s: float = 1.0,
    ) -> None:
        super().__init__(
            clock=clock,
            readiness_timeout_s=readiness_timeout_s,
            liveness_timeout_s=liveness_timeout_s,
            grace_s=grace_s,
        )
        self._transport = transport
        self._gate_profile: SafetyProfile | None = None
        self._link_up = False
        self._subscribed = False
        self._unsub_tele: Callable[[], None] | None = None
        self._unsub_status: Callable[[], None] | None = None

    @property
    def transport(self) -> Transport:
        return self._transport

    def ready(self) -> bool:
        """True when the link reports ``connected``. Independent of liveness."""
        return self._link_up

    def on_start(self) -> None:
        if isinstance(self._transport, SafetyGate):
            self._gate_profile = self._transport.profile
            # The gate's window is the freshness definition. Do not lengthen it.
            self.adopt_liveness_window(self._gate_profile.state_timeout_s)
        self._unsub_tele = self._transport.on_telemetry(self._on_telemetry)
        self._unsub_status = self._transport.on_status(self._on_status)
        self._subscribed = True
        status = self._transport.connect()
        self._link_up = status.state == ConnectionState.CONNECTED

    def on_stop(self) -> None:
        if not self._subscribed:
            return
        unsub_tele = self._unsub_tele
        unsub_status = self._unsub_status
        self._unsub_tele = None
        self._unsub_status = None
        self._subscribed = False
        if unsub_tele is not None:
            unsub_tele()
        if unsub_status is not None:
            unsub_status()
        self._transport.disconnect()
        self._link_up = False

    def _on_status(self, status: TransportStatus) -> None:
        self._link_up = status.state == ConnectionState.CONNECTED

    def _on_telemetry(self, sample: Telemetry) -> None:
        profile = self._gate_profile
        if profile is not None and not sample_refreshes_state(profile, sample):
            return
        self.heartbeat()
