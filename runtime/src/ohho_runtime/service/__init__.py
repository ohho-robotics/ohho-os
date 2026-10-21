"""Service lifecycle: configure, start, readiness, liveness, and shutdown.

This is the onboard runtime's service API. It is not ``ohho.runtime``, the
in-process executor port.
"""

from __future__ import annotations

from ohho_runtime.service.base import Service, ServiceBase
from ohho_runtime.service.inprocess import InProcessService
from ohho_runtime.service.lifecycle import (
    LEGAL_TRANSITIONS,
    Health,
    HealthStatus,
    IllegalTransition,
    LifecycleState,
    ManualClock,
    ServiceError,
    ShutdownRecord,
    UnixSocketUnavailable,
    move,
)
from ohho_runtime.service.shutdown import SigtermShutdown
from ohho_runtime.service.subprocess import (
    SubprocessService,
    child_env,
    stop_child,
    unix_sockets_available,
)
from ohho_runtime.service.transport import TransportService, sample_refreshes_state

__all__ = [
    "LEGAL_TRANSITIONS",
    "Health",
    "HealthStatus",
    "IllegalTransition",
    "InProcessService",
    "LifecycleState",
    "ManualClock",
    "Service",
    "ServiceBase",
    "ServiceError",
    "ShutdownRecord",
    "SigtermShutdown",
    "SubprocessService",
    "TransportService",
    "UnixSocketUnavailable",
    "child_env",
    "move",
    "sample_refreshes_state",
    "stop_child",
    "unix_sockets_available",
]
