"""In-process service: a target callable on a daemon thread."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ohho_runtime.log import note_exception
from ohho_runtime.service.base import ServiceBase
from ohho_runtime.service.lifecycle import LifecycleState, ShutdownRecord


class InProcessService(ServiceBase):
    """Run ``target(service)`` on a thread.

    ``target`` should heartbeat and return when :meth:`stopping` is true.
    ``stop`` sets the event and joins for ``grace_s``. A thread that is still
    alive after that join is recorded as ``grace_exceeded``. Python cannot
    force-kill a thread; the record is the escalation.

    ``ready_check`` overrides the default readiness probe (the thread is
    alive). Liveness stays the heartbeat, so a live thread can still be
    not ready.
    """

    def __init__(
        self,
        target: Callable[[InProcessService], None],
        *,
        ready_check: Callable[[], bool] | None = None,
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
        self._target = target
        self._ready_check = ready_check
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def on_start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._main,
            name="ohho-runtime-service",
            daemon=True,
        )
        self._thread.start()

    def _main(self) -> None:
        try:
            self._target(self)
        except Exception as exc:
            note_exception(
                exc,
                service=self._service_name(),
                event="service_error",
                logger=self._logger,
                metrics=self._metrics,
            )
            if self.state not in (
                LifecycleState.STOPPING,
                LifecycleState.STOPPED,
                LifecycleState.FAILED,
            ):
                self._fail(f"service error: {exc}")

    def ready(self) -> bool:
        if self._ready_check is not None:
            return bool(self._ready_check())
        thread = self._thread
        return thread is not None and thread.is_alive()

    def on_stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=self.grace_s)
        if thread.is_alive():
            self.note_shutdown(
                ShutdownRecord(
                    action="grace_exceeded",
                    forced=True,
                    grace_s=self.grace_s,
                    detail="in-process service did not stop within the grace period",
                )
            )
