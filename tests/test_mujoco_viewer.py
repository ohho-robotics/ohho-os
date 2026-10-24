"""Native MuJoCo viewer flag. The no-display path must not call GLFW."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import build_parser, main
from ohho.mujoco_viewer import (
    ViewerNotSupported,
    ViewerUnavailable,
    _opengl_context_available,
    attach_passive_viewer,
    display_available,
    open_passive_viewer,
)
from ohho.registry import _REGISTRY, register
from ohho.robot import Robot
from ohho.robot_spec import RobotAssets, RobotClass
from ohho.humanoid_sim import HUMANOID_MUJOCO_BACKEND
from ohho.registry import RobotSpec

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
_HINGE = Path(__file__).resolve().parent / "fixtures" / "humanoid" / "hinge.xml"


class TestDisplayAvailable(unittest.TestCase):
    def test_linux_without_a_display_is_headless(self) -> None:
        self.assertFalse(display_available(platform="linux", environ={}))
        self.assertFalse(display_available(platform="linux", environ={"CI": "true"}))
        self.assertFalse(
            display_available(platform="linux", environ={"SSH_CONNECTION": "1 2 3 4"})
        )
        self.assertTrue(display_available(platform="linux", environ={"DISPLAY": ":1"}))
        self.assertTrue(
            display_available(
                platform="linux", environ={"WAYLAND_DISPLAY": "wayland-0"}
            )
        )
        self.assertFalse(display_available(platform="linux", environ={"DISPLAY": ""}))

    def test_macos_ssh_needs_a_forwarded_display(self) -> None:
        self.assertFalse(display_available(platform="darwin", environ={"SSH_TTY": "1"}))
        self.assertTrue(
            display_available(
                platform="darwin", environ={"SSH_CONNECTION": "1", "DISPLAY": ":0"}
            )
        )
        self.assertTrue(display_available(platform="darwin", environ={}))

    def test_windows_without_monitors_is_headless(self) -> None:
        with patch("ohho.mujoco_viewer._monitor_count", return_value=0):
            self.assertFalse(display_available(platform="win32", environ={}))
        with patch("ohho.mujoco_viewer._monitor_count", return_value=2):
            self.assertTrue(display_available(platform="win32", environ={}))


class TestNoDisplayPath(unittest.TestCase):
    def test_windows_child_crash_is_no_opengl_context(self) -> None:
        crashed = subprocess.CompletedProcess(args=[], returncode=3221225477)
        with (
            patch("ohho.mujoco_viewer.sys.platform", "win32"),
            patch("ohho.mujoco_viewer.subprocess.run", return_value=crashed),
        ):
            self.assertFalse(_opengl_context_available())
        opened = subprocess.CompletedProcess(args=[], returncode=0)
        with (
            patch("ohho.mujoco_viewer.sys.platform", "win32"),
            patch("ohho.mujoco_viewer.subprocess.run", return_value=opened),
        ):
            self.assertTrue(_opengl_context_available())

    def test_missing_opengl_context_does_not_launch(self) -> None:
        with (
            patch("ohho.mujoco_viewer.display_available", return_value=True),
            patch("ohho.mujoco_viewer._opengl_context_available", return_value=False),
            patch("ohho.mujoco_viewer._launch_passive") as launch,
        ):
            with self.assertRaises(ViewerUnavailable) as caught:
                open_passive_viewer(object(), object())
        launch.assert_not_called()
        message = str(caught.exception)
        self.assertIn("OpenGL", message)
        self.assertIn("mjpython", message)
        self.assertNotIn("Traceback", message)

    def test_open_does_not_launch_when_there_is_no_display(self) -> None:
        with (
            patch("ohho.mujoco_viewer.display_available", return_value=False),
            patch("ohho.mujoco_viewer._launch_passive") as launch,
        ):
            with self.assertRaises(ViewerUnavailable) as caught:
                open_passive_viewer(object(), object())
        launch.assert_not_called()
        message = str(caught.exception)
        self.assertIn("no display", message)
        self.assertIn("mjpython", message)
        self.assertNotIn("Traceback", message)

    def test_attach_stops_before_launch_when_headless(self) -> None:
        class _Link:
            def __init__(self) -> None:
                self.mujoco_model = object()
                self.mujoco_data = object()
                self.attached = None

            def attach_viewer(self, handle: object) -> None:
                self.attached = handle

        link = _Link()
        with (
            patch("ohho.mujoco_viewer.display_available", return_value=False),
            patch("ohho.mujoco_viewer._launch_passive") as launch,
        ):
            with self.assertRaises(ViewerUnavailable):
                attach_passive_viewer(link)
        launch.assert_not_called()
        self.assertIsNone(link.attached)

    def test_mjpython_error_stays_a_viewer_error(self) -> None:
        def boom(model: object, data: object) -> object:
            del model, data
            raise RuntimeError(
                "`launch_passive` requires that the Python script be run under "
                "`mjpython` on macOS"
            )

        with (
            patch("ohho.mujoco_viewer.display_available", return_value=True),
            patch("ohho.mujoco_viewer._opengl_context_available", return_value=True),
            patch("ohho.mujoco_viewer._launch_passive", side_effect=boom),
        ):
            with self.assertRaises(ViewerUnavailable) as caught:
                open_passive_viewer(object(), object())
        self.assertIn("mjpython", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_a_transport_without_a_model_is_not_a_crash(self) -> None:
        with self.assertRaises(ViewerNotSupported) as caught:
            attach_passive_viewer(object())
        self.assertIn("MuJoCo", str(caught.exception))
        self.assertNotIn("Traceback", str(caught.exception))


@unittest.skipUnless(_HAS_MUJOCO, "mujoco is not installed")
class TestConnectNoDisplay(unittest.TestCase):
    def test_robot_connect_viewer_does_not_launch_or_start_the_loop(self) -> None:
        spec = RobotSpec(
            id="viewer-hinge",
            name="Viewer hinge",
            category="humanoid",
            robot_class=RobotClass.HUMANOID,
            sim_backend=HUMANOID_MUJOCO_BACKEND,
            joint_names=("hinge",),
            joint_groups={"link": ("hinge",)},
            assets=RobotAssets(mjcf=str(_HINGE), licence="Apache-2.0"),
        )
        register(spec)
        self.addCleanup(_REGISTRY.pop, spec.id, None)
        with (
            patch("ohho.mujoco_viewer.display_available", return_value=False),
            patch("ohho.mujoco_viewer._launch_passive") as launch,
        ):
            with self.assertRaises(ViewerUnavailable) as caught:
                Robot.connect(
                    spec.id, transport="sim://", runtime="native", viewer=True
                )
        launch.assert_not_called()
        self.assertIn("no display", str(caught.exception))
        self.assertIn("mjpython", str(caught.exception))
        live = [
            thread.name
            for thread in threading.enumerate()
            if thread.name == "ohho-humanoid-mujoco" and thread.is_alive()
        ]
        self.assertEqual(live, [])


class TestCliViewer(unittest.TestCase):
    def test_viewer_defaults_off(self) -> None:
        sim = build_parser().parse_args(["sim", "--robot", "omnibot", "--seconds", "1"])
        self.assertFalse(sim.viewer)
        served = build_parser().parse_args(["sim-serve", "--robot", "omnibot"])
        self.assertFalse(served.viewer)
        flagged = build_parser().parse_args(
            ["sim", "--robot", "omnibot", "--viewer", "--seconds", "1"]
        )
        self.assertTrue(flagged.viewer)

    def test_sim_without_viewer_stays_headless(self) -> None:
        with (
            patch("ohho.mujoco_viewer.open_passive_viewer") as launch,
            redirect_stdout(io.StringIO()),
        ):
            code = main(["sim", "--robot", "omnibot", "--seconds", "0"])
        self.assertEqual(code, 0)
        launch.assert_not_called()

    def test_sim_viewer_on_a_kinematic_robot_is_a_clean_error(self) -> None:
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sim", "--robot", "omnibot", "--viewer", "--seconds", "1"])
        self.assertEqual(code, 2)
        text = err.getvalue()
        self.assertIn("MuJoCo", text)
        self.assertNotIn("Traceback", text)
        live = [
            thread.name
            for thread in threading.enumerate()
            if thread.name == "ohho-sim" and thread.is_alive()
        ]
        self.assertEqual(live, [])

    def test_sim_serve_viewer_on_ros2_is_a_clean_error(self) -> None:
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sim-serve", "--backend", "ros2", "--viewer"])
        self.assertEqual(code, 2)
        text = err.getvalue()
        self.assertIn("--viewer", text)
        self.assertNotIn("Traceback", text)

    def test_sim_serve_mujoco_backend_stays_not_built(self) -> None:
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sim-serve", "--backend", "mujoco", "--viewer"])
        self.assertEqual(code, 1)
        self.assertIn("not built", out.getvalue() + err.getvalue())
