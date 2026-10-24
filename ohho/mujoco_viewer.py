"""Native MuJoCo window for a sim backend that already owns the physics.

Headless stays the default. ``--viewer`` calls
``mujoco.viewer.launch_passive`` and the backend syncs that handle from its
existing loop. A missing display (CI, or SSH without a window server) raises
:class:`ViewerUnavailable` before GLFW is launched.

On macOS the window has to be created on the process main thread. The
ordinary ``python`` launcher does not do that. Run::

    mjpython -m ohho sim --viewer

``mjpython`` is installed with the ``mujoco`` package.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping

_MACOS_HINT = (
    "On macOS, launch the viewer with mjpython so the window runs on the "
    "main thread: mjpython -m ohho sim --viewer"
)

_NO_DISPLAY = (
    "no display available for the MuJoCo viewer "
    "(headless, CI, or SSH without a display). "
    "Headless simulation is the default. " + _MACOS_HINT
)

_NO_GL = (
    "the MuJoCo viewer could not open a window because this process "
    "cannot create an OpenGL context. "
    "Headless simulation is the default. " + _MACOS_HINT
)


class ViewerUnavailable(RuntimeError):
    """The native MuJoCo viewer could not open a window."""


class ViewerNotSupported(RuntimeError):
    """This simulator has no MuJoCo model to show."""


def display_available(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """True when a window system is likely to accept a native viewer.

    Linux needs ``DISPLAY`` or ``WAYLAND_DISPLAY``. An SSH session on macOS
    needs one of those too. A local Mac is treated as having a display;
    ``mjpython`` is checked when the window is launched. Windows needs at
    least one monitor.
    """
    plat = sys.platform if platform is None else platform
    env = os.environ if environ is None else environ
    if plat.startswith("linux") or plat.startswith("freebsd"):
        return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    if plat == "darwin":
        remote = (
            env.get("SSH_CONNECTION") or env.get("SSH_TTY") or env.get("SSH_CLIENT")
        )
        if remote:
            return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
        return True
    if plat == "win32":
        return _monitor_count() > 0
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _monitor_count() -> int:
    """Interactive monitors on Windows. Zero means there is nowhere to draw."""
    if sys.platform != "win32":
        return 1
    import ctypes

    try:
        # SM_CMONITORS
        return int(ctypes.windll.user32.GetSystemMetrics(80))
    except (AttributeError, OSError, ValueError):
        return 0


def open_passive_viewer(model: object, data: object) -> object:
    """Open ``mujoco.viewer.launch_passive`` on ``model`` and ``data``.

    Raises :class:`ViewerUnavailable` when there is no display, when this
    process cannot create an OpenGL context, when GLFW cannot start, or
    when macOS is not running under ``mjpython``. The launcher is not
    called on those paths. On Windows, ``glfwCreateWindow`` aborts the
    process when WGL cannot create a context, so that case is probed in
    a child process first.
    """
    if not display_available():
        raise ViewerUnavailable(_NO_DISPLAY)
    if not _opengl_context_available():
        raise ViewerUnavailable(_NO_GL)
    try:
        return _launch_passive(model, data)
    except ViewerUnavailable:
        raise
    except Exception as exc:
        raise ViewerUnavailable(_failure_message(exc)) from exc


# A failed WGL context makes glfwCreateWindow abort the process. The probe
# runs in a child so the parent can raise ViewerUnavailable instead.
_WIN_GL_PROBE = (
    "import glfw\n"
    "ok = bool(glfw.init())\n"
    "win = glfw.create_window(64, 64, 'ohho', None, None) if ok else None\n"
    "raise SystemExit(0 if win else 1)\n"
)


def _opengl_context_available() -> bool:
    """False when a native viewer would be unable to create a GL context.

    Non-Windows platforms report True here. Linux and macOS surface a
    missing display or the ``mjpython`` error before GLFW is asked to
    open a window. Windows asks a child process to open a tiny GLFW
    window, because a failed WGL context aborts ``glfwCreateWindow``
    in-process.
    """
    if sys.platform != "win32":
        return True
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WIN_GL_PROBE],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _launch_passive(model: object, data: object) -> object:
    try:
        import mujoco.viewer as viewer_mod
    except Exception as exc:
        raise ViewerUnavailable(
            "MuJoCo viewer is not installed "
            "(pip install 'ohho-os[humanoid-sim]'). " + _MACOS_HINT
        ) from exc
    # GLFW has to be initialised on the thread that creates the window.
    # ``launch_passive`` does that itself. Initialising it here first, on
    # the caller, crashes the native viewer on Windows.
    return viewer_mod.launch_passive(model, data)


def _failure_message(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if "mjpython" in text.lower() or sys.platform == "darwin":
        return f"the MuJoCo viewer could not open a window. {_MACOS_HINT} ({text})"
    return f"the MuJoCo viewer could not open a window: {text}. {_NO_DISPLAY}"


def sync_viewer(handle: object) -> None:
    """Copy the current simulation state into a passive viewer handle."""
    running = getattr(handle, "is_running", None)
    try:
        if callable(running) and not running():
            return
        sync = getattr(handle, "sync", None)
        if callable(sync):
            sync()
    except Exception:
        return


def close_viewer(handle: object) -> None:
    """Close a passive viewer. A second close, or a failed close, is ignored."""
    close = getattr(handle, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        return


def attach_passive_viewer(transport: object) -> object:
    """Open a passive viewer on ``transport`` and hand it the handle.

    The transport must expose ``mujoco_model`` and ``mujoco_data`` after
    ``connect``, and ``attach_viewer`` so its own loop can call
    :func:`sync_viewer`. A kinematic simulator raises
    :class:`ViewerNotSupported`.
    """
    model = getattr(transport, "mujoco_model", None)
    data = getattr(transport, "mujoco_data", None)
    if model is None or data is None:
        raise ViewerNotSupported(
            "--viewer needs a MuJoCo sim backend; this simulator has no MuJoCo model"
        )
    attach = getattr(transport, "attach_viewer", None)
    if not callable(attach):
        raise ViewerNotSupported(
            "--viewer needs a MuJoCo sim backend that can sync a native viewer"
        )
    handle = open_passive_viewer(model, data)
    try:
        attach(handle)
    except Exception:
        close_viewer(handle)
        raise
    return handle


__all__ = [
    "ViewerNotSupported",
    "ViewerUnavailable",
    "attach_passive_viewer",
    "close_viewer",
    "display_available",
    "open_passive_viewer",
    "sync_viewer",
]
