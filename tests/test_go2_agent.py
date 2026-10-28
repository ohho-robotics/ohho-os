import unittest
from unittest.mock import patch

from ohho.adapters.sim import SimTransport
from ohho.agent import Agent, ScriptedBrain
from ohho.brains import build_tool_registry, harness_available
from ohho.evals import load_sim_eval, run_scripted_sim_eval
from ohho.planner import goal_blockers
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime

_HAS_AGENT_ENGINE = harness_available()
_SKIP_REASON = "agent_engine (+ numpy) not importable — install 'ohho-os[agent]'"


def _bot(robot_id="unitree-go2"):
    spec = get_spec(robot_id)
    tp = SimTransport(spec)
    tp.connect()
    return Robot(spec, tp, NativeRuntime())


class TestGo2GoalBlockers(unittest.TestCase):
    def test_refuse_manipulation_goal(self):
        reasons = goal_blockers(_bot(), "Pick up the box in front of you.")
        self.assertTrue(any("manipulation" in r for r in reasons))

    def test_refuse_stairs(self):
        reasons = goal_blockers(_bot(), "Climb the stairs to the second floor.")
        self.assertTrue(any("stairs" in r for r in reasons))


class TestLeggedToolGating(unittest.TestCase):
    @unittest.skipUnless(_HAS_AGENT_ENGINE, _SKIP_REASON)
    def test_go2_has_legged_tools_not_arm(self):
        reg = build_tool_registry(_bot("unitree-go2"))
        for tool in ("stand", "lie_down", "sit", "recover", "set_body_pose"):
            self.assertIn(tool, reg)
        self.assertNotIn("move_joints", reg)

    @unittest.skipUnless(_HAS_AGENT_ENGINE, _SKIP_REASON)
    def test_omnibot_has_arm_not_legged_pose_tools(self):
        reg = build_tool_registry(_bot("omnibot"))
        self.assertIn("move_joints", reg)
        for tool in ("stand", "lie_down", "sit", "recover", "set_body_pose"):
            self.assertNotIn(tool, reg)

    def test_go2_sim_stand_before_drive(self):
        bot = _bot("unitree-go2")
        self.assertEqual(bot.legged_posture(), "lying")
        bot.drive(vx=0.2)
        bot.transport.step(0.1)  # type: ignore[attr-defined]
        t_lying = bot.telemetry()
        self.assertAlmostEqual(t_lying.odom.x, 0.0, places=3)
        bot.stand()
        self.assertEqual(bot.legged_posture(), "standing")
        bot.drive(vx=0.2)
        bot.transport.step(0.2)  # type: ignore[attr-defined]
        t_standing = bot.telemetry()
        self.assertGreater(t_standing.odom.x, 0.02)


class TestGo2ScriptedEval(unittest.TestCase):
    def test_scripted_passes_goals_1_and_5_without_api_key(self):
        spec = load_sim_eval()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            results = run_scripted_sim_eval(spec, only_scripted_pass=True)
        self.assertEqual({g.id for g, _, _ in results}, {1, 5})
        for goal, ok, log in results:
            self.assertTrue(ok, f"goal {goal.id} failed: {log}")

    def test_scripted_refuses_goal_3(self):
        log = Agent(_bot(), brain=ScriptedBrain()).run(
            "Pick up the box in front of you."
        )
        self.assertTrue(any("refused:" in line for line in log))


if __name__ == "__main__":
    unittest.main()
