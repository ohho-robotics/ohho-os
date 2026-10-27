"""``ohho verify --sim`` drives only ``sim://`` through SafetyGate."""

from __future__ import annotations

import inspect
import io
import json
import os
import tempfile
import unittest
import urllib.error
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.robots.omnibot.policy import OMNIBOT_PROFILE
from ohho.safety import CMD_TIMEOUT_S, SafetyGate
from ohho.schema import Velocity
from ohho.verify import (
    COMMAND_PERIOD_S,
    DRIVE_MPS,
    DRIVE_S,
    TOLERANCE_M,
    is_sim_uri,
    non_sim_refusal,
    run_verify,
    run_verify_cli,
)

_TOKEN = "ohho_verify_token_271_phase7_do_not_print"


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def _env(home: Path, **updates: str) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "OHHO_SAFETY_DIR": str(home / "safety"),
    }
    env.update(updates)
    return env


class ManualVerifyTests(unittest.TestCase):
    def _run(self, home: Path, **kwargs: object):
        clock = _Clock()
        log: list[Velocity] = []
        options: dict[str, object] = {"environ": _env(home)}
        options.update(kwargs)
        result = run_verify(
            "omnibot",
            clock=clock,
            sleep=clock.sleep,
            command_log=log,
            **options,  # type: ignore[arg-type]
        )
        return result, log

    def test_happy_path_moves_within_tolerance_through_the_gate(self) -> None:
        seen: list[SafetyGate] = []
        original = SafetyGate.send_velocity

        def spy(gate: SafetyGate, vel: Velocity) -> None:
            seen.append(gate)
            original(gate, vel)

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch.object(SafetyGate, "send_velocity", spy):
                result, log = self._run(home)
            audit = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (home / "safety").glob("*.jsonl")
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.transport, "sim://")
        self.assertEqual(result.robot, "omnibot")
        self.assertGreater(result.distance_m, 0.0)
        self.assertLessEqual(result.distance_m, DRIVE_MPS * DRIVE_S + TOLERANCE_M)
        self.assertTrue(result.gated)
        self.assertIsInstance(result.gate, SafetyGate)
        assert result.gate is not None
        self.assertEqual(result.gate.protocol, "simulated")
        self.assertFalse(result.gate.requires_hardware_arming)
        self.assertFalse(result.gate.armed)
        self.assertAlmostEqual(result.gate.profile.max_vx, OMNIBOT_PROFILE.max_vx)
        self.assertAlmostEqual(result.gate.profile.max_vy, OMNIBOT_PROFILE.max_vy)
        self.assertTrue(seen)
        self.assertTrue(all(isinstance(gate, SafetyGate) for gate in seen))
        self.assertTrue(any(vel.linear_x > 0.0 for vel in log))
        self.assertEqual(log[-1], Velocity())
        self.assertTrue(all(vel.linear_x <= DRIVE_MPS + 1e-9 for vel in log))
        self.assertIn('"event": "arm"', audit)
        self.assertNotIn("estop_release", audit)
        self.assertLess(COMMAND_PERIOD_S, CMD_TIMEOUT_S)

    def test_stop_is_sent_when_drive_raises(self) -> None:
        def on_step(step: int) -> None:
            if step == 3:
                raise RuntimeError("mid-drive")

        with tempfile.TemporaryDirectory() as tmp:
            clock = _Clock()
            log: list[Velocity] = []
            with self.assertRaises(RuntimeError):
                run_verify(
                    clock=clock,
                    sleep=clock.sleep,
                    environ=_env(Path(tmp)),
                    command_log=log,
                    on_step=on_step,
                )
        self.assertTrue(any(vel.linear_x > 0.0 for vel in log))
        self.assertEqual(log[-1], Velocity())

    def test_stop_is_sent_on_keyboard_interrupt(self) -> None:
        def on_step(step: int) -> None:
            if step == 2:
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            clock = _Clock()
            log: list[Velocity] = []
            with self.assertRaises(KeyboardInterrupt):
                run_verify(
                    clock=clock,
                    sleep=clock.sleep,
                    environ=_env(Path(tmp)),
                    command_log=log,
                    on_step=on_step,
                )
        self.assertEqual(log[-1], Velocity())

    def test_arm_hardware_and_safety_config_do_not_change_the_sim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = home / "raise.json"
            cfg.write_text(
                json.dumps(
                    {
                        "hard_ceiling": {"max_lin": 1.5, "max_ang": 2.0},
                        "profiles": {"omnibot": {"max_vx": 1.4, "max_vy": 1.4}},
                    }
                ),
                encoding="utf-8",
            )
            plain, _log = self._run(home)
            armed, _log2 = self._run(
                home,
                environ=_env(
                    home,
                    OHHO_ARM_HARDWARE="1",
                    OHHO_SAFETY_CONFIG=str(cfg),
                    OHHO_TRANSPORT="sim://",
                ),
            )
        self.assertTrue(plain.ok and armed.ok)
        self.assertAlmostEqual(plain.distance_m, armed.distance_m)
        assert armed.gate is not None
        self.assertAlmostEqual(armed.gate.profile.max_vx, 0.2)
        self.assertFalse(armed.gate.requires_hardware_arming)
        self.assertEqual(armed.transport, "sim://")

    def test_release_stop_and_arm_env_are_not_in_the_drive_path(self) -> None:
        import ohho.verify as verify_mod

        source = "\n".join(
            inspect.getsource(fn)
            for fn in (
                verify_mod.run_verify,
                verify_mod._drive,
                verify_mod._stop,
                verify_mod._arm,
            )
        )
        self.assertNotIn("release_stop", source)
        self.assertNotIn("OHHO_ARM_HARDWARE", source)

    def test_no_network_unless_console_is_set(self) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("network")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("ohho.verify.urlopen", boom),
                patch("ohho.auth._OPENER.open", boom),
            ):
                result, _log = self._run(Path(tmp))
        self.assertTrue(result.ok)
        self.assertIsNone(result.console)

    def test_console_404_is_skip_and_does_not_fail_the_drive(self) -> None:
        def opener(request: object, timeout: float = 0) -> None:
            raise urllib.error.HTTPError(
                "https://console.ohho-robotics.com/sim",
                404,
                "missing",
                EmailMessage(),
                None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            result, _log = self._run(Path(tmp), console=True, opener=opener)
        self.assertTrue(result.ok)
        assert result.console is not None
        self.assertEqual(result.console.status, "skip")
        self.assertIn("404", result.console.detail)

    def test_console_200_is_reachable(self) -> None:
        class _Response:
            status = 200

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

            def getcode(self) -> int:
                return 200

        def opener(request: object, timeout: float = 0) -> _Response:
            self.assertTrue(str(getattr(request, "full_url", "")).endswith("/sim"))
            return _Response()

        with tempfile.TemporaryDirectory() as tmp:
            result, _log = self._run(Path(tmp), console=True, opener=opener)
        assert result.console is not None
        self.assertEqual(result.console.status, "pass")
        self.assertEqual(result.console.detail, "reachable")

    def test_console_outage_and_bad_origin_are_unavailable(self) -> None:
        def opener(*_args: object, **_kwargs: object) -> None:
            raise urllib.error.URLError("down")

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            down, _log = self._run(home, console=True, opener=opener)
            bad, _log2 = self._run(
                home,
                console=True,
                opener=opener,
                environ=_env(home, OHHO_CONSOLE_URL="not a url"),
            )
        assert down.console is not None and bad.console is not None
        self.assertEqual(down.console.status, "skip")
        self.assertEqual(bad.console.detail, "unavailable")
        self.assertTrue(down.ok and bad.ok)


class RefusalTests(unittest.TestCase):
    def test_sim_uri_helper(self) -> None:
        self.assertTrue(is_sim_uri("sim://"))
        self.assertTrue(is_sim_uri(" SIM:// "))
        self.assertTrue(is_sim_uri("simulated://"))
        self.assertFalse(is_sim_uri("dds://eth0"))
        self.assertFalse(is_sim_uri("serial:///dev/ttyUSB0"))

    def test_non_sim_flag_env_and_robot_config_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cfg = home / "robot.json"
            cfg.write_text(
                json.dumps({"id": "omnibot", "transport": "dds://eth0"}),
                encoding="utf-8",
            )
            benign = home / "manifest.json"
            benign.write_text(
                json.dumps({"id": "my-rover", "adapter": "yahboom-serial"}),
                encoding="utf-8",
            )
            self.assertIn(
                "--transport",
                non_sim_refusal(_env(home), transport="serial:///dev/ttyUSB0") or "",
            )
            self.assertIn(
                "OHHO_TRANSPORT",
                non_sim_refusal(_env(home, OHHO_TRANSPORT="ros2://")) or "",
            )
            self.assertIn(
                "OHHO_URI",
                non_sim_refusal(_env(home, OHHO_URI="feetech:///dev/ttyACM0")) or "",
            )
            self.assertIn(
                "--robot",
                non_sim_refusal(_env(home), robot="dds://eth0") or "",
            )
            self.assertIn(
                "robot config",
                non_sim_refusal(_env(home, OHHO_ROBOT_CONFIG=str(cfg))) or "",
            )
            self.assertIsNone(
                non_sim_refusal(_env(home, OHHO_ROBOT_CONFIG=str(benign)))
            )
            self.assertIsNone(non_sim_refusal(_env(home, OHHO_TRANSPORT="sim://")))
            broken = home / "broken.json"
            broken.write_text("{", encoding="utf-8")
            self.assertIn(
                "unreadable",
                non_sim_refusal(_env(home, OHHO_ROBOT_CONFIG=str(broken))) or "",
            )

    def test_cli_refuses_before_building_a_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = _env(home, OHHO_TRANSPORT="dds://eth0")
            with patch("ohho.verify.SimTransport", side_effect=AssertionError("sim")):
                missing = run_verify_cli(
                    sim=False,
                    as_json=True,
                    robot="omnibot",
                    console=False,
                    transport=None,
                    environ=env,
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                )
                stdout, stderr = io.StringIO(), io.StringIO()
                refused = run_verify_cli(
                    sim=True,
                    as_json=True,
                    robot="omnibot",
                    console=False,
                    transport="serial:///dev/ttyUSB0",
                    environ=_env(home),
                    stdout=stdout,
                    stderr=stderr,
                )
        self.assertEqual(missing, 2)
        self.assertEqual(refused, 2)
        body = json.loads(stdout.getvalue())
        self.assertIs(body["ok"], False)
        self.assertEqual(body["transport"], "sim://")
        self.assertIn("refusing --transport", body["error"])
        self.assertNotIn("serial://", stdout.getvalue() + stderr.getvalue())
        self.assertIn("error:", stderr.getvalue())

    def test_hardware_adapters_are_not_constructed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch(
                    "ohho.robots.omnibot.yahboom.YahboomTransport",
                    side_effect=AssertionError("serial"),
                ),
                patch(
                    "ohho.adapters.unitree.UnitreeDdsTransport",
                    side_effect=AssertionError("dds"),
                ),
            ):
                result, _log = ManualVerifyTests()._run(
                    home,
                    environ=_env(home, OHHO_ARM_HARDWARE="1"),
                )
        self.assertTrue(result.ok)
        self.assertEqual(result.transport, "sim://")


class CliVerifyTests(unittest.TestCase):
    def test_main_requires_sim_and_refuses_non_sim_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = _env(home)
            with patch.dict(os.environ, env, clear=False):
                for key in (
                    "OHHO_TOKEN",
                    "OHHO_TRANSPORT",
                    "OHHO_ROBOT_TRANSPORT",
                    "OHHO_VERIFY_TRANSPORT",
                    "OHHO_URI",
                    "OHHO_ROBOT_CONFIG",
                    "OHHO_ARM_HARDWARE",
                ):
                    os.environ.pop(key, None)
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                    missing = main(["verify"])
                    refused = main(
                        ["verify", "--sim", "--transport", "dds://eth0", "--json"]
                    )
        self.assertEqual(missing, 2)
        self.assertEqual(refused, 2)
        self.assertIn("requires --sim", stderr.getvalue())
        self.assertNotIn("dds://eth0", stdout.getvalue() + stderr.getvalue())

    def test_main_sim_json_reports_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch.dict(os.environ, _env(home, OHHO_ARM_HARDWARE="1"), clear=False):
                os.environ.pop("OHHO_TRANSPORT", None)
                os.environ.pop("OHHO_TOKEN", None)
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch("sys.stdout", stdout),
                    patch("sys.stderr", stderr),
                    patch("ohho.verify.urlopen", side_effect=AssertionError("network")),
                    patch(
                        "ohho.auth._OPENER.open", side_effect=AssertionError("network")
                    ),
                ):
                    code = main(["verify", "--sim", "--json", "--robot", "omnibot"])
        self.assertNotIn(_TOKEN, stderr.getvalue())
        self.assertNotIn("dds://", stderr.getvalue())
        body = json.loads(stdout.getvalue())
        self.assertEqual(
            list(body),
            [
                "ok",
                "robot",
                "transport",
                "distance_m",
                "tolerance_m",
                "commanded_mps",
                "duration_s",
                "detail",
                "console",
            ],
        )
        self.assertIs(body["ok"], True)
        self.assertEqual(body["transport"], "sim://")
        self.assertEqual(body["robot"], "omnibot")
        self.assertEqual(body["duration_s"], 1.0)
        self.assertEqual(body["commanded_mps"], 0.1)
        self.assertIsNone(body["console"])
        self.assertGreater(body["distance_m"], 0.0)
        self.assertEqual(code, 0)
        self.assertNotIn(_TOKEN, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
