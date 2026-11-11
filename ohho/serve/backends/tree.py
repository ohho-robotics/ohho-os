"""Process tree for a serving engine.

The engine is a child of a small supervisor. ``kill`` stops that supervisor
and every process it started. On Windows the supervisor is placed in a job
object that dies with its members. On POSIX it is its own session so a
signal reaches the group. A supervisor that sees its engine exit also kills
descendants that would otherwise be reparented.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from .errors import ServingBackendError

_NO_JOB = "OHHO_SERVE_NO_JOB"


def http_ok(url: str, timeout: float = 0.3) -> bool:
    """True when a GET returns a 2xx status. Connection errors are false."""
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return 200 <= int(status) < 300
    except Exception:
        return False


def pid_alive(pid: int) -> bool:
    """True when ``pid`` still refers to a running process.

    On Windows ``OpenProcess`` succeeds for a process that has already
    exited while its object is still open. That is not a running process.
    ``GetExitCodeProcess`` returns 259 (``STILL_ACTIVE``) only while it runs.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = _k32()
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def kill_pid(pid: int, *, tree: bool = True) -> None:
    """Stop ``pid``. On Windows, ``tree`` also stops processes it created.

    The supervisor passes ``tree=False`` because it already enumerated
    descendants. Asking ``taskkill /T`` twice for one pid can block when
    that pid is reused or a console host is still exiting.
    """
    if pid <= 0:
        return
    if os.name == "nt":
        command = ["taskkill", "/F", "/PID", str(pid)]
        if tree:
            command.insert(2, "/T")
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3,
            )
        except subprocess.TimeoutExpired:
            return
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def descendants(root: int) -> set[int]:
    """Return child and grandchild pids of ``root``. ``root`` is not included."""
    found: set[int] = set()
    stack = [root]
    mapping = _parent_map()
    while stack:
        current = stack.pop()
        for child in mapping.get(current, []):
            if child in found or child == root:
                continue
            found.add(child)
            stack.append(child)
    return found


def _parent_map() -> dict[int, list[int]]:
    if os.name == "nt":
        return _windows_parent_map()
    proc = Path("/proc")
    if proc.is_dir():
        return _linux_parent_map(proc)
    return _ps_parent_map()


def _linux_parent_map(proc: Path) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    try:
        entries = list(proc.iterdir())
    except OSError:
        return children
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            text = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                children.setdefault(ppid, []).append(pid)
                break
    return children


def _ps_parent_map() -> dict[int, list[int]]:
    try:
        output = subprocess.check_output(
            ["ps", "-A", "-o", "pid=", "-o", "ppid="],
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def _k32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    return kernel32


def _windows_parent_map() -> dict[int, list[int]]:
    kernel32 = _k32()
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid:
        return {}
    entry = _PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    children: dict[int, list[int]] = {}
    try:
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return {}
        while True:
            children.setdefault(int(entry.th32ParentProcessID), []).append(
                int(entry.th32ProcessID)
            )
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return children


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),
        ("cntUsage", ctypes.c_ulong),
        ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_ulong),
        ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class ProcessTree:
    """One supervisor process and the engine commands it starts."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._job: int | None = None
        self._lines: list[str] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def poll(self) -> int | None:
        """Return the supervisor exit code, or None while it is running."""
        proc = self._proc
        if proc is None:
            return None
        return proc.poll()

    def start(
        self,
        commands: Sequence[Sequence[str]],
        *,
        env: Mapping[str, str] | None,
        cwd: str | None,
    ) -> None:
        """Start the supervisor. It spawns ``commands`` after this returns."""
        if self._proc is not None:
            raise ServingBackendError("serving process is already running")
        if not commands:
            raise ServingBackendError("serving command is empty")
        payload = (
            json.dumps(
                {
                    "commands": [list(command) for command in commands],
                    "cwd": cwd,
                    "env": None if env is None else dict(env),
                }
            ).encode("utf-8")
            + b"\n"
        )
        gate_env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[3])
        previous = gate_env.get("PYTHONPATH")
        gate_env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
        popen_kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": gate_env,
        }
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            popen_kwargs["creationflags"] = flags | no_window
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-m", "ohho.serve.backends.gate"],
            **popen_kwargs,  # type: ignore[arg-type]
        )
        self._proc = proc
        self._drain(proc.stdout)
        self._drain(proc.stderr)
        try:
            self._arm_job(proc.pid)
            self._write(payload)
        except Exception as exc:
            tail = self.output_tail(2000).strip()
            self.kill()
            if isinstance(exc, ServingBackendError):
                if tail and tail not in str(exc):
                    raise ServingBackendError(f"{exc}\n{tail}") from exc
                raise
            extra = f" {tail}" if tail else ""
            raise ServingBackendError(
                f"failed to start the serving process.{extra}"
            ) from exc

    def output_tail(self, limit: int = 4000) -> str:
        """Return the recent combined stdout and stderr of the tree."""
        with self._lock:
            text = "".join(self._lines)
        if len(text) <= limit:
            return text
        return text[-limit:]

    def kill(self) -> None:
        """Stop the supervisor and every process in its tree. Idempotent."""
        proc = self._proc
        job = self._job
        self._job = None
        if os.name == "nt":
            kernel32 = _k32()
            if job:
                kernel32.TerminateJobObject(job, 1)
                kernel32.CloseHandle(job)
            if proc is not None and proc.poll() is None:
                kill_pid(proc.pid)
        elif proc is not None and proc.pid:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                try:
                    proc.kill()
                except OSError:
                    pass
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        for thread in self._threads:
            thread.join(timeout=1)
        self._close_pipes()
        self._proc = None
        self._threads = []

    def _close_pipes(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def _write(self, payload: bytes) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ServingBackendError("serving process has no stdin")
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            raise ServingBackendError(
                "serving process gate exited before reading its command"
            ) from exc

    def _arm_job(self, pid: int) -> None:
        if os.name != "nt" or os.environ.get(_NO_JOB) == "1":
            return
        job = _create_job()
        if job is None:
            return
        if _assign_job(job, pid):
            self._job = job
            return
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)

    def _drain(self, pipe: object) -> None:
        def read() -> None:
            try:
                while True:
                    chunk = pipe.readline()  # type: ignore[attr-defined]
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    with self._lock:
                        self._lines.append(text)
                        if len(self._lines) > 400:
                            del self._lines[:200]
            except Exception:
                return

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        self._threads.append(thread)


def _create_job() -> int | None:
    kernel32 = _k32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x2000
    ok = kernel32.SetInformationJobObject(
        ctypes.c_void_p(job),
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _assign_job(job: int, pid: int) -> bool:
    kernel32 = _k32()
    handle = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)
    if not handle:
        handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), handle))
    finally:
        kernel32.CloseHandle(handle)


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]
