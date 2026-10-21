"""SIGTERM shutdown: call ``stop()`` and record it when grace is exceeded."""

from __future__ import annotations

import signal
from collections.abc import Callable

from ohho_runtime.service.base import ServiceBase
from ohho_runtime.service.lifecycle import IllegalTransition, ShutdownRecord


class SigtermShutdown:
    """Invoke ``service.stop()`` when SIGTERM arrives.

    The handler does not arm a robot and does not command motion. If ``stop``
    returns within ``grace_s`` (measured on the service clock), the record is
    ``sigterm``. If the clock shows ``stop`` took longer, the record is
    ``grace_exceeded`` and ``forced`` is set. A second signal, once the
    service can no longer stop, is recorded and does not raise.
    """

    def __init__(self, service: ServiceBase, *, grace_s: float | None = None) -> None:
        self._service = service
        self._grace_s = service.grace_s if grace_s is None else grace_s
        if self._grace_s <= 0:
            raise ValueError("grace_s must be > 0")

    @property
    def grace_s(self) -> float:
        return self._grace_s

    def handle(self, signum: int, _frame: object = None) -> None:
        """Signal handler. Only SIGTERM stops the service."""
        if signum != signal.SIGTERM:
            return
        started = self._service.clock()
        try:
            self._service.stop()
        except IllegalTransition as exc:
            self._service.note_shutdown(
                ShutdownRecord(
                    action="sigterm",
                    forced=False,
                    grace_s=self._grace_s,
                    signal=int(signal.SIGTERM),
                    detail=str(exc),
                )
            )
            return
        elapsed = self._service.clock() - started
        if elapsed > self._grace_s:
            self._service.note_shutdown(
                ShutdownRecord(
                    action="grace_exceeded",
                    forced=True,
                    grace_s=self._grace_s,
                    signal=int(signal.SIGTERM),
                    detail="SIGTERM stop exceeded the grace period",
                )
            )
            return
        self._service.note_shutdown(
            ShutdownRecord(
                action="sigterm",
                forced=False,
                grace_s=self._grace_s,
                signal=int(signal.SIGTERM),
                detail="SIGTERM",
            )
        )

    def install(self) -> Callable[[], None]:
        """Install ``handle`` as the process SIGTERM handler.

        Returns a callable that restores the previous handler.
        """
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.handle)

        def restore() -> None:
            signal.signal(signal.SIGTERM, previous)

        return restore
