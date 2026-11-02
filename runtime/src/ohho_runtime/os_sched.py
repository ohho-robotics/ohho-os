"""Linux scheduling calls for one service process.

This is the only module that touches niceness, the scheduler, CPU
affinity, or ``mlockall``. Tests replace these calls. Nothing here
imports ``ohho``. Importing the module does not change the process.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

# linux/capability.h
CAP_SYS_NICE = 23

# bits/mman-linux.h. CURRENT locks pages now; FUTURE locks later allocations.
MCL_CURRENT = 1
MCL_FUTURE = 2

MLOCK_ENV = "OHHO_RUNTIME_MLOCKALL"
MLOCK_STATUS_ENV = "OHHO_RUNTIME_MLOCK_STATUS"

_LOG = logging.getLogger("ohho_runtime.os_sched")
_LIBC: ctypes.CDLL | None = None


def cap_sys_nice_from_status(text: str) -> bool:
    """Return whether ``CapEff`` in a ``/proc/self/status`` body has ``CAP_SYS_NICE``."""
    for line in text.splitlines():
        if not line.startswith("CapEff:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            mask = int(raw, 16)
        except ValueError:
            return False
        return bool(mask & (1 << CAP_SYS_NICE))
    return False


def read_cap_sys_nice(status_path: Path | None = None) -> bool:
    """Read the effective ``CAP_SYS_NICE`` bit. Missing ``/proc`` is false."""
    path = Path("/proc/self/status") if status_path is None else status_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return cap_sys_nice_from_status(text)


def privileged_linux() -> bool:
    """True when this process is root or holds ``CAP_SYS_NICE`` on Linux."""
    if sys.platform != "linux":
        return False
    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid) and geteuid() == 0:
        return True
    return read_cap_sys_nice()


def _as_oserror(exc: Exception) -> OSError:
    if isinstance(exc, OSError):
        return exc
    return OSError(f"scheduling call is not available: {exc}")


def _rlimit(name: str) -> tuple[int, int]:
    try:
        import resource
    except ModuleNotFoundError as exc:
        raise OSError("resource limits are not available") from exc
    try:
        soft, hard = resource.getrlimit(getattr(resource, name))
    except (AttributeError, ValueError, OSError) as exc:
        raise _as_oserror(exc) from exc
    return int(soft), int(hard)


def _infinity() -> int:
    try:
        import resource
    except ModuleNotFoundError as exc:
        raise OSError("resource limits are not available") from exc
    return int(resource.RLIM_INFINITY)


class LinuxOsCalls:
    """Real ``CAP_SYS_NICE``, rlimit, scheduler, affinity, and ``mlockall`` calls.

    Methods raise :class:`OSError` when the platform has no such call.
    They do not substitute a different priority.
    """

    def cap_sys_nice(self) -> bool:
        """Effective ``CAP_SYS_NICE`` for this process."""
        return read_cap_sys_nice()

    def rlimit_rtprio(self) -> tuple[int, int]:
        """``RLIMIT_RTPRIO`` ``(soft, hard)``."""
        return _rlimit("RLIMIT_RTPRIO")

    def rlimit_memlock(self) -> tuple[int, int]:
        """``RLIMIT_MEMLOCK`` ``(soft, hard)`` in bytes."""
        return _rlimit("RLIMIT_MEMLOCK")

    def rlimit_infinity(self) -> int:
        """The platform's unlimited rlimit sentinel."""
        return _infinity()

    def get_nice(self, pid: int) -> int:
        """Absolute niceness of ``pid``."""
        try:
            return int(os.getpriority(os.PRIO_PROCESS, pid))
        except (AttributeError, OSError) as exc:
            raise _as_oserror(exc) from exc

    def set_nice(self, pid: int, value: int) -> None:
        """Set absolute niceness. ``value`` is not an increment."""
        try:
            os.setpriority(os.PRIO_PROCESS, pid, value)
        except (AttributeError, OSError) as exc:
            raise _as_oserror(exc) from exc

    def set_scheduler(self, pid: int, policy: str, priority: int) -> None:
        """Set ``fifo`` or ``rr`` to exactly ``priority``.

        ``priority`` is passed through. This does not clamp it.
        """
        if policy == "fifo":
            number = getattr(os, "SCHED_FIFO", None)
        elif policy == "rr":
            number = getattr(os, "SCHED_RR", None)
        else:
            raise OSError(f"unsupported scheduler policy {policy!r}")
        if number is None or not hasattr(os, "sched_setscheduler"):
            raise OSError("sched_setscheduler is not available")
        param_type = getattr(os, "sched_param", None)
        if param_type is None:
            raise OSError("sched_param is not available")
        try:
            os.sched_setscheduler(pid, number, param_type(priority))
        except (AttributeError, OSError) as exc:
            raise _as_oserror(exc) from exc

    def set_affinity(self, pid: int, cpus: tuple[int, ...]) -> None:
        """Pin ``pid`` to exactly ``cpus``. The mask is not widened or shrunk."""
        try:
            os.sched_setaffinity(pid, set(cpus))
        except (AttributeError, OSError) as exc:
            raise _as_oserror(exc) from exc

    def mlockall(self) -> None:
        """``mlockall(MCL_CURRENT | MCL_FUTURE)`` for this process."""
        mlockall_current()


def mlockall_current() -> None:
    """Lock this process's current and future pages.

    Raises :class:`OSError` when the call is missing or the kernel refuses.
    Does not lock another process.
    """
    global _LIBC
    if sys.platform != "linux":
        raise OSError("mlockall is only supported on Linux")
    if _LIBC is None:
        name = ctypes.util.find_library("c")
        if not name:
            raise OSError("libc was not found")
        library = ctypes.CDLL(name, use_errno=True)
        library.mlockall.argtypes = [ctypes.c_int]
        library.mlockall.restype = ctypes.c_int
        _LIBC = library
    ctypes.set_errno(0)
    rc = _LIBC.mlockall(MCL_CURRENT | MCL_FUTURE)
    if rc != 0:
        err = ctypes.get_errno()
        message = os.strerror(err) if err else "mlockall failed"
        raise OSError(err, message)


def lock_memory_if_requested(
    environ: Mapping[str, str] | None = None,
    lock: Callable[[], None] | None = None,
) -> str | None:
    """Lock this process when ``OHHO_RUNTIME_MLOCKALL`` is ``1``.

    Returns ``None`` when the variable is unset. Returns ``ok`` or a
    ``refused:`` string. A refusal is logged and does not raise, so the
    service process still starts. When ``OHHO_RUNTIME_MLOCK_STATUS`` is
    set, the same string is written there for the supervisor.
    """
    env = os.environ if environ is None else environ
    if env.get(MLOCK_ENV) != "1":
        return None
    try:
        if lock is None:
            mlockall_current()
        else:
            lock()
    except OSError as exc:
        _LOG.warning("%s", f"mlockall refused: {exc}")
        result = f"refused: {exc}"
    else:
        result = "ok"
    status = env.get(MLOCK_STATUS_ENV, "")
    if status:
        Path(status).write_text(result + "\n", encoding="utf-8")
    return result


__all__ = [
    "CAP_SYS_NICE",
    "MCL_CURRENT",
    "MCL_FUTURE",
    "MLOCK_ENV",
    "MLOCK_STATUS_ENV",
    "LinuxOsCalls",
    "cap_sys_nice_from_status",
    "lock_memory_if_requested",
    "mlockall_current",
    "privileged_linux",
    "read_cap_sys_nice",
]
