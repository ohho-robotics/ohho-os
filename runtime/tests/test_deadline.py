"""Deadline, wire heartbeat, and timeout conformance (OHH-359)."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from ohho.robot_spec import RobotClass
from ohho.safety import CMD_TIMEOUT_S, STATE_TIMEOUT_S, SafetyProfile
from ohho.sim_serve import DEADMAN_TIMEOUT_S
from ohho_runtime.deadline import (
    GATE_TICK_S,
    HEARTBEAT_HZ,
    STOP_BUDGET_S,
    Deadline,
    Heartbeat,
    TimeoutConformanceError,
    check_registered_timeouts,
    check_timeout_conformance,
    class_timeout_ceiling,
    safety_state_json,
    wire_contract,
)
from ohho_runtime.service.lifecycle import ManualClock

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "runtime" / "src"
_DEADLINE_SRC = _SRC / "ohho_runtime" / "deadline.py"
_WATCHDOG_LITERAL = re.compile(r"0\.3(00)?\b")


class TestDeadline(unittest.TestCase):
    def test_command_and_state_read_the_profile(self) -> None:
        public = SafetyProfile()
        tighter = SafetyProfile(
            cmd_timeout_s=public.cmd_timeout_s / 2,
            state_timeout_s=public.state_timeout_s / 2,
        )
        clock = ManualClock()
        command = Deadline.command(tighter, clock=clock)
        state = Deadline.state(tighter, clock=clock)
        self.assertEqual(command.timeout_s, tighter.cmd_timeout_s)
        self.assertEqual(state.timeout_s, tighter.state_timeout_s)
        self.assertNotEqual(command.timeout_s, state.timeout_s)
        ceiling = class_timeout_ceiling(RobotClass.MOBILE_BASE)
        self.assertEqual(Deadline.command(public).timeout_s, ceiling.cmd_timeout_s)
        self.assertEqual(Deadline.state(public).timeout_s, ceiling.state_timeout_s)

    def test_expires_at_the_profile_timeout(self) -> None:
        profile = SafetyProfile()
        clock = ManualClock()
        deadline = Deadline.command(profile, clock=clock)
        self.assertTrue(deadline.expired())
        self.assertIsNone(deadline.age_s())
        self.assertEqual(deadline.remaining_s(), 0.0)
        deadline.refresh()
        self.assertTrue(deadline.fresh())
        self.assertFalse(deadline.expired(profile.cmd_timeout_s / 2))
        self.assertTrue(deadline.expired(profile.cmd_timeout_s))
        clock.advance(profile.cmd_timeout_s)
        self.assertTrue(deadline.expired())
        self.assertFalse(deadline.fresh())

    def test_refresh_opens_a_new_window(self) -> None:
        profile = SafetyProfile()
        clock = ManualClock()
        deadline = Deadline.state(profile, clock=clock)
        deadline.refresh()
        clock.advance(profile.state_timeout_s)
        self.assertTrue(deadline.expired())
        deadline.refresh()
        self.assertTrue(deadline.fresh())
        self.assertGreater(deadline.remaining_s(), 0.0)

    def test_backwards_clock_expires(self) -> None:
        deadline = Deadline.command(SafetyProfile(), clock=ManualClock())
        deadline.refresh()
        self.assertTrue(deadline.expired(-1.0))
        self.assertLess(deadline.age_s(-1.0) or 0.0, 0.0)

    def test_rejects_a_bad_timeout_or_profile(self) -> None:
        with self.assertRaises(ValueError):
            Deadline(0.0)
        with self.assertRaises(ValueError):
            Deadline(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Deadline.command(object())  # type: ignore[arg-type]
        deadline = Deadline.command(SafetyProfile(), clock=ManualClock())
        with self.assertRaises(ValueError):
            deadline.refresh(True)  # type: ignore[arg-type]

    def test_default_clock_is_monotonic(self) -> None:
        deadline = Deadline.state(SafetyProfile())
        deadline.refresh()
        self.assertFalse(deadline.expired())


class TestHeartbeat(unittest.TestCase):
    def test_miss_uses_the_command_timeout(self) -> None:
        public = SafetyProfile()
        tighter = SafetyProfile(
            cmd_timeout_s=public.cmd_timeout_s / 2,
            state_timeout_s=public.state_timeout_s,
        )
        clock = ManualClock()
        heartbeat = Heartbeat(tighter, clock=clock)
        self.assertEqual(heartbeat.timeout_s, tighter.cmd_timeout_s)
        self.assertNotEqual(heartbeat.timeout_s, tighter.state_timeout_s)
        self.assertTrue(heartbeat.expired())
        heartbeat.beat()
        self.assertTrue(heartbeat.fresh(tighter.cmd_timeout_s / 2))
        self.assertTrue(heartbeat.expired(tighter.cmd_timeout_s))
        clock.advance(tighter.cmd_timeout_s)
        self.assertTrue(heartbeat.expired())

    def test_default_profile_matches_the_class_ceiling(self) -> None:
        heartbeat = Heartbeat(clock=ManualClock())
        ceiling = class_timeout_ceiling("mobile_base")
        self.assertEqual(heartbeat.timeout_s, ceiling.cmd_timeout_s)
        self.assertEqual(heartbeat.rate_hz, HEARTBEAT_HZ)
        self.assertEqual(heartbeat.period_s, 1.0 / HEARTBEAT_HZ)

    def test_publish_grid_and_a_late_beat(self) -> None:
        clock = ManualClock()
        heartbeat = Heartbeat(SafetyProfile(), clock=clock)
        self.assertFalse(heartbeat.due())
        heartbeat.start()
        self.assertTrue(heartbeat.due())
        self.assertTrue(heartbeat.expired())
        heartbeat.beat()
        self.assertFalse(heartbeat.due())
        self.assertTrue(heartbeat.fresh())
        clock.advance(heartbeat.period_s)
        self.assertTrue(heartbeat.due())
        clock.advance(heartbeat.period_s * 2.5)
        self.assertTrue(heartbeat.due())
        heartbeat.beat()
        self.assertFalse(heartbeat.due())

    def test_rejects_a_bad_profile_or_rate(self) -> None:
        with self.assertRaises(TypeError):
            Heartbeat(object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Heartbeat(SafetyProfile(), rate_hz=0)


class TestWireContract(unittest.TestCase):
    def test_default_contract_reads_the_profile(self) -> None:
        profile = SafetyProfile()
        contract = wire_contract()
        self.assertEqual(contract.topic, "/ohho/heartbeat")
        self.assertEqual(contract.message_type, "std_msgs/Empty")
        self.assertEqual(contract.rate_hz, 10.0)
        self.assertEqual(contract.period_s, 1.0 / contract.rate_hz)
        self.assertEqual(contract.timeout_s, profile.cmd_timeout_s)
        self.assertEqual(contract.state_timeout_s, profile.state_timeout_s)
        self.assertEqual(contract.gate_tick_s, GATE_TICK_S)
        self.assertEqual(contract.gate_tick_s, 0.01)
        self.assertEqual(contract.stop_budget_s, STOP_BUDGET_S)
        self.assertEqual(contract.stop_budget_s, 0.35)
        self.assertGreater(
            contract.stop_budget_s, profile.cmd_timeout_s + contract.gate_tick_s
        )
        self.assertEqual(contract.safety_state_topic, "/ohho/safety/state")
        self.assertEqual(contract.safety_state_type, "std_msgs/String")
        self.assertEqual(contract.safety_state_fields, ("armed", "latched", "reason"))
        self.assertEqual(contract.estop_reset_service, "/ohho/estop_reset")
        self.assertEqual(contract.estop_reset_type, "std_srvs/Trigger")
        self.assertLess(contract.period_s, contract.timeout_s)

    def test_tighter_profile_shrinks_the_timeout_only(self) -> None:
        public = wire_contract()
        profile = SafetyProfile()
        tighter = SafetyProfile(
            cmd_timeout_s=profile.cmd_timeout_s / 2,
            state_timeout_s=profile.state_timeout_s / 2,
        )
        contract = wire_contract(tighter)
        self.assertEqual(contract.timeout_s, tighter.cmd_timeout_s)
        self.assertEqual(contract.state_timeout_s, tighter.state_timeout_s)
        slack = public.stop_budget_s - public.timeout_s
        self.assertEqual(contract.stop_budget_s, tighter.cmd_timeout_s + slack)
        self.assertLess(contract.stop_budget_s, public.stop_budget_s)
        self.assertLess(contract.timeout_s, public.timeout_s)
        self.assertEqual(contract.topic, public.topic)
        self.assertEqual(contract.rate_hz, public.rate_hz)
        self.assertEqual(contract.safety_state_topic, public.safety_state_topic)

    def test_safety_state_json(self) -> None:
        text = safety_state_json(armed=False, latched=True, reason="startup")
        self.assertEqual(
            text,
            '{"armed": false, "latched": true, "reason": "startup"}',
        )
        with self.assertRaises(TypeError):
            safety_state_json(armed=1, latched=False, reason="startup")  # type: ignore[arg-type]

    def test_module_has_no_watchdog_literal(self) -> None:
        text = _DEADLINE_SRC.read_text(encoding="utf-8")
        self.assertIsNone(_WATCHDOG_LITERAL.search(text))

    def test_shipped_windows_match_the_profile(self) -> None:
        profile = SafetyProfile()
        self.assertEqual(CMD_TIMEOUT_S, profile.cmd_timeout_s)
        self.assertEqual(STATE_TIMEOUT_S, profile.state_timeout_s)
        self.assertEqual(DEADMAN_TIMEOUT_S, profile.cmd_timeout_s)


class TestTimeoutConformance(unittest.TestCase):
    def test_every_class_uses_the_public_profile(self) -> None:
        profile = SafetyProfile()
        for kind in RobotClass:
            ceiling = class_timeout_ceiling(kind)
            self.assertEqual(ceiling.robot_class, kind.value)
            self.assertEqual(ceiling.cmd_timeout_s, profile.cmd_timeout_s)
            self.assertEqual(ceiling.state_timeout_s, profile.state_timeout_s)
            check_timeout_conformance(profile, kind)
            check_timeout_conformance(
                {
                    "cmd_timeout_s": profile.cmd_timeout_s,
                    "state_timeout_s": profile.state_timeout_s,
                },
                kind.value,
            )

    def test_a_raised_timeout_fails(self) -> None:
        ceiling = class_timeout_ceiling(RobotClass.QUADRUPED)
        raised = {
            "cmd_timeout_s": ceiling.cmd_timeout_s,
            "state_timeout_s": ceiling.state_timeout_s * 2,
        }
        with self.assertRaises(TimeoutConformanceError) as caught:
            check_timeout_conformance(raised, RobotClass.QUADRUPED)
        self.assertIn("state_timeout_s", str(caught.exception))
        self.assertIn("exceeds class maximum", str(caught.exception))
        self.assertEqual(caught.exception.robot_class, "quadruped")

        raised_cmd = {
            "cmd_timeout_s": ceiling.cmd_timeout_s * 2,
            "state_timeout_s": ceiling.state_timeout_s,
        }
        with self.assertRaises(TimeoutConformanceError) as caught_cmd:
            check_timeout_conformance(raised_cmd, "mobile_manipulator")
        self.assertIn("cmd_timeout_s", str(caught_cmd.exception))

    def test_a_tighter_ceiling_rejects_the_default_profile(self) -> None:
        public = SafetyProfile()
        tighter = SafetyProfile(
            cmd_timeout_s=public.cmd_timeout_s / 2,
            state_timeout_s=public.state_timeout_s / 2,
        )
        with self.assertRaises(TimeoutConformanceError):
            check_timeout_conformance(public, RobotClass.AERIAL, ceiling=tighter)
        check_timeout_conformance(tighter, RobotClass.AERIAL, ceiling=tighter)

    def test_invalid_declared_timeouts_fail(self) -> None:
        ceiling = class_timeout_ceiling(RobotClass.ARM)
        with self.assertRaises(TimeoutConformanceError):
            check_timeout_conformance({}, RobotClass.ARM)
        with self.assertRaises(TimeoutConformanceError):
            check_timeout_conformance(
                {"cmd_timeout_s": True, "state_timeout_s": ceiling.state_timeout_s},
                "arm",
            )
        with self.assertRaises(TimeoutConformanceError):
            check_timeout_conformance(
                {"cmd_timeout_s": 0, "state_timeout_s": ceiling.state_timeout_s},
                RobotClass.HUMANOID,
            )
        with self.assertRaises(TimeoutConformanceError):
            check_timeout_conformance(
                {
                    "cmd_timeout_s": math.nan,
                    "state_timeout_s": ceiling.state_timeout_s,
                },
                RobotClass.MOBILE_BASE,
            )
        with self.assertRaises(ValueError):
            check_timeout_conformance(SafetyProfile(), "not-a-class")
        with self.assertRaises(TypeError):
            check_timeout_conformance(
                SafetyProfile(),
                RobotClass.ARM,
                ceiling=object(),  # type: ignore[arg-type]
            )

    def test_registered_robots_stay_inside_the_ceiling(self) -> None:
        count = check_registered_timeouts()
        self.assertGreater(count, 0)


class TestDeadlineImport(unittest.TestCase):
    def test_package_import_does_not_load_the_deadline(self) -> None:
        env = dict(os.environ)
        prior = env.get("PYTHONPATH", "")
        paths = [str(_SRC), str(_ROOT)]
        if prior:
            paths.append(prior)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        probe = (
            "import ohho_runtime, sys\n"
            "bad = [name for name in ('ohho', 'ohho_runtime.deadline') "
            "if name in sys.modules]\n"
            "raise SystemExit(0 if not bad else ','.join(bad))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
