"""OHH-535: ohho eval runner, Wilson interval, and ohho.eval/v1 reports."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from ohho.cli import main
from ohho.eval import runner as eval_runner
from ohho.eval import (
    ProofScenario,
    SCHEMA_ID,
    SUITES,
    EvalError,
    EvalValidationError,
    Z_95,
    evaluate_after_train,
    list_suites,
    load_scenarios,
    load_schema,
    resolve_scenario_file,
    run_eval,
    run_eval_job,
    validate_report,
    wilson_interval,
)

_FIXTURE = """\
version: 1
frame_id: map
world: apartment
scenarios:
  - id: clear_goal
    start: {x: 0.0, y: 0.0, yaw_deg: 0.0}
    goal: {x: 1.0, y: 0.0, yaw_deg: 0.0}
    obstacle:
      name: off_path
      pose: {x: 0.5, y: 3.0, yaw_deg: 0.0}
      size: {x: 0.2, y: 0.2, z: 0.2}
  - id: turn_in_place
    start: {x: 0.0, y: 0.0, yaw_deg: 0.0}
    goal: {x: 0.0, y: 0.0, yaw_deg: 180.0}
    obstacle:
      name: far
      pose: {x: 5.0, y: 5.0, yaw_deg: 0.0}
      size: {x: 0.2, y: 0.2, z: 0.2}
  - id: blocked
    start: {x: 0.0, y: 0.0, yaw_deg: 0.0}
    goal: {x: 1.5, y: 0.0, yaw_deg: 0.0}
    obstacle:
      name: on_path
      pose: {x: 0.5, y: 0.0, yaw_deg: 0.0}
      size: {x: 0.4, y: 0.4, z: 0.4}
"""

_REL = "omnibot-ros2/omnibot_navigation/config/nav_regression_scenarios.yaml"


def _pose_scenario(
    scenario_id: str,
    goal_x: float,
    obstacle_x: float,
    obstacle_y: float,
) -> str:
    def num(value: float) -> str:
        return format(value, ".12f")

    return (
        "version: 1\n"
        "frame_id: map\n"
        "world: apartment\n"
        "scenarios:\n"
        f"  - id: {scenario_id}\n"
        "    start: {x: 0.0, y: 0.0, yaw_deg: 0.0}\n"
        f"    goal: {{x: {num(goal_x)}, y: 0.0, yaw_deg: 0.0}}\n"
        "    obstacle:\n"
        "      name: box\n"
        f"      pose: {{x: {num(obstacle_x)}, y: {num(obstacle_y)}, yaw_deg: 0.0}}\n"
        "      size: {x: 0.2, y: 0.2, z: 0.2}\n"
    )


def _omnibot_root() -> Path | None:
    roots: list[Path] = []
    for key in ("OHHO_PROOF_ROOT", "OHHO_OMNIBOT_ROOT", "OMNIBOT_ROOT"):
        value = os.environ.get(key, "").strip()
        if value:
            roots.append(Path(value))
    sibling = Path(__file__).resolve().parents[2]
    roots.append(sibling / "OmniBot")
    roots.append(sibling / "OmniBot-ohh-395")
    for root in roots:
        if (root / _REL).is_file():
            return root
    return None


class TestWilsonInterval(unittest.TestCase):
    def test_z95_is_the_normal_quantile(self) -> None:
        self.assertAlmostEqual(Z_95, 1.959963984540054, places=9)

    def test_empty_sample_is_uninformative(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_zero_and_perfect_match_the_closed_form(self) -> None:
        z2 = Z_95 * Z_95
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, z2 / (10.0 + z2))
        low, high = wilson_interval(10, 10)
        self.assertAlmostEqual(high, 1.0)
        self.assertAlmostEqual(low, 10.0 / (10.0 + z2))

    def test_interval_contains_the_rate_and_is_symmetric(self) -> None:
        for trials in (1, 5, 20):
            for successes in range(trials + 1):
                low, high = wilson_interval(successes, trials)
                rate = successes / trials
                self.assertLessEqual(low, rate)
                self.assertGreaterEqual(high, rate)
                self.assertGreaterEqual(low, 0.0)
                self.assertLessEqual(high, 1.0)
                mirror = wilson_interval(trials - successes, trials)
                self.assertAlmostEqual(low, 1.0 - mirror[1])
                self.assertAlmostEqual(high, 1.0 - mirror[0])

    def test_rejects_bad_counts(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(-1, 4)
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(TypeError):
            wilson_interval(True, 1)  # type: ignore[arg-type]


class TestScenarioReference(unittest.TestCase):
    def test_suite_pointer_is_not_a_copied_definition(self) -> None:
        ref = SUITES["gazebo-nav-regression"]
        self.assertEqual(ref.relative_path, _REL)
        self.assertIn("ohho-robotics/OmniBot", ref.repository)
        self.assertEqual(ref.source, f"{ref.repository}#{ref.relative_path}")
        root = Path(__file__).resolve().parents[1]
        banned = ("origin_to_kitchen", "mid_corridor_box")
        for path in (root / "ohho" / "eval").rglob("*"):
            if path.suffix not in {".py", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in banned:
                self.assertNotIn(needle, text, str(path))
        schema = (root / "schemas" / "ohho.eval.v1.json").read_text(encoding="utf-8")
        for needle in banned:
            self.assertNotIn(needle, schema)

    def test_missing_checkout_names_the_reference(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"OHHO_PROOF_ROOT": "", "OHHO_EVAL_SCENARIOS": ""},
        ):
            with self.assertRaises(EvalError) as caught:
                resolve_scenario_file("gazebo-nav-regression")
        message = str(caught.exception)
        self.assertIn(_REL, message)
        self.assertIn("not vendored", message)

    def test_unknown_suite_lists_known_names(self) -> None:
        with mock.patch.dict(os.environ, {"OHHO_EVAL_SCENARIOS": ""}):
            with self.assertRaises(EvalError) as caught:
                resolve_scenario_file("not-a-suite")
        self.assertIn("gazebo-nav-regression", str(caught.exception))
        self.assertEqual(list_suites()[0].name, "gazebo-nav-regression")

    def test_fixture_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.yaml"
            path.write_text(_FIXTURE, encoding="utf-8")
            loaded = load_scenarios(path)
        self.assertEqual(
            [item.id for item in loaded],
            ["clear_goal", "turn_in_place", "blocked"],
        )
        blocked = loaded[2]
        self.assertAlmostEqual(blocked.obstacle_x, 0.5)
        self.assertAlmostEqual(blocked.size_z, 0.4)
        self.assertEqual(blocked.world, "apartment")

    def test_duplicate_id_is_rejected(self) -> None:
        text = _FIXTURE.replace("id: blocked", "id: clear_goal", 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(EvalError):
                load_scenarios(path)

    def test_packaged_schema_matches_schemas_dir(self) -> None:
        root = Path(__file__).resolve().parents[1]
        published = (root / "schemas" / "ohho.eval.v1.json").read_text(encoding="utf-8")
        packaged = (root / "ohho" / "eval" / "schemas" / "ohho.eval.v1.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(published, packaged)
        self.assertEqual(load_schema()["properties"]["schema"]["const"], SCHEMA_ID)

    def test_omnibot_file_when_present(self) -> None:
        root = _omnibot_root()
        if root is None:
            self.skipTest("OmniBot nav regression YAML is not checked out")
        loaded = load_scenarios(root / _REL)
        ids = [item.id for item in loaded]
        self.assertIn("origin_to_kitchen", ids)
        self.assertGreaterEqual(len(ids), 3)
        with mock.patch.dict(
            os.environ,
            {
                "OHHO_PROOF_ROOT": str(root),
                "OHHO_EVAL_SCENARIOS": "",
            },
        ):
            path, source = resolve_scenario_file("gazebo-nav-regression")
        self.assertEqual(path.resolve(), (root / _REL).resolve())
        self.assertEqual(source, SUITES["gazebo-nav-regression"].source)


class TestEvalRunner(unittest.TestCase):
    def _write_checkpoint(self, directory: Path, speed: float) -> Path:
        path = directory / f"ckpt-{speed}.json"
        path.write_text(
            json.dumps({"eval": {"speed_mps": speed, "heading_gain": 2.5}}),
            encoding="utf-8",
        )
        return path

    def _run(self, directory: Path, checkpoint: Path, **kwargs: object) -> dict:
        scenarios = directory / "scenarios.yaml"
        if not scenarios.exists():
            scenarios.write_text(_FIXTURE, encoding="utf-8")
        return run_eval(
            checkpoint,
            "gazebo-nav-regression",
            scenarios=scenarios,
            **kwargs,  # type: ignore[arg-type]
        )

    def _episode_speed(self, checkpoint: Path, seed: int) -> float:
        from ohho.registry import get_spec

        digest = eval_runner.checkpoint_sha256(checkpoint)
        base_speed, _gain = eval_runner.controller_gains(checkpoint, digest)
        scale = 0.85 + 0.15 * eval_runner._unit(digest, seed)
        return min(float(get_spec("sim").max_lin), base_speed * scale)

    def _run_pose_case(
        self,
        root: Path,
        checkpoint: Path,
        scenario_id: str,
        *,
        goal_x: float,
        obstacle_x: float,
        obstacle_y: float,
    ) -> dict:
        scenarios = root / f"{scenario_id}.yaml"
        scenarios.write_text(
            _pose_scenario(scenario_id, goal_x, obstacle_x, obstacle_y),
            encoding="utf-8",
        )
        return run_eval(
            checkpoint,
            "gazebo-nav-regression",
            episodes=1,
            seed=0,
            scenarios=scenarios,
            videos=root / f"{scenario_id}-videos",
        )

    def test_same_checkpoint_and_seed_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_checkpoint(root, 0.4)
            first = self._run(root, checkpoint, episodes=2, seed=7)
            second = self._run(root, checkpoint, episodes=2, seed=7)
        self.assertEqual(first["success_rate"], second["success_rate"])
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertEqual(first["success_ci95"], second["success_ci95"])
        self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])
        validate_report(first)
        reloaded = json.loads(json.dumps(first))
        validate_report(reloaded)

    def test_failure_modes_and_yaw_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_checkpoint(root, 0.45)
            videos = root / "videos"
            report = self._run(root, checkpoint, episodes=2, seed=1, videos=videos)
            self.assertTrue(all(Path(path).is_file() for path in report["videos"]))
        by_id: dict[str, list[dict]] = {}
        for episode in report["episodes"]:
            by_id.setdefault(episode["scenario_id"], []).append(episode)
        self.assertTrue(all(row["success"] for row in by_id["clear_goal"]))
        self.assertTrue(all(row["success"] for row in by_id["turn_in_place"]))
        self.assertTrue(all(not row["success"] for row in by_id["blocked"]))
        self.assertTrue(
            all(row["failure_mode"] == "collision" for row in by_id["blocked"])
        )
        self.assertEqual(report["failure_modes"], {"collision": 2})
        self.assertEqual(report["success_rate"], 4 / 6)
        self.assertEqual(len(report["videos"]), 6)
        self.assertEqual(report["backend"], "kinematic-sim")
        self.assertEqual(report["schema"], SCHEMA_ID)
        low = report["success_ci95"]["low"]
        high = report["success_ci95"]["high"]
        self.assertEqual((low, high), wilson_interval(4, 6))
        self.assertLessEqual(low, report["success_rate"])
        self.assertGreaterEqual(high, report["success_rate"])

    def test_last_step_goal_is_success_and_collision_still_wins(self) -> None:
        """Arrival on step MAX_STEPS counts; contact on that step stays collision."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_checkpoint(root, 0.5)
            speed = self._episode_speed(checkpoint, seed=0)
            x = 0.0
            positions = [0.0]
            for _step in range(eval_runner.MAX_STEPS):
                x, _y, _theta = eval_runner._integrate(
                    x, 0.0, 0.0, speed, 0.0, eval_runner.DT_S
                )
                positions.append(x)
            before = positions[-2]
            after = positions[-1]
            step_m = after - before
            tol = eval_runner.GOAL_XY_TOLERANCE_M
            # Just outside before the last integration, inside after it.
            goal = before + tol + 0.5 * step_m
            self.assertGreater(goal - before, tol)
            self.assertLessEqual(abs(goal - after), tol)

            success = self._run_pose_case(
                root,
                checkpoint,
                "last_step_goal",
                goal_x=goal,
                obstacle_x=-50.0,
                obstacle_y=-50.0,
            )
            hit_goal = after + tol - 0.25 * step_m
            half_x = 0.2 / 2.0 + eval_runner.BODY_RADIUS_M
            # Leading edge sits between the last two predicted poses.
            obstacle_x = before + 0.5 * step_m + half_x
            hit = ProofScenario(
                id="last_step_hit",
                start_x=0.0,
                start_y=0.0,
                start_yaw=0.0,
                goal_x=hit_goal,
                goal_y=0.0,
                goal_yaw=0.0,
                obstacle_name="box",
                obstacle_x=obstacle_x,
                obstacle_y=0.0,
                obstacle_yaw=0.0,
                size_x=0.2,
                size_y=0.2,
                size_z=0.2,
                world="apartment",
                frame_id="map",
            )
            self.assertGreater(hit_goal - before, tol)
            self.assertLessEqual(abs(hit_goal - after), tol)
            self.assertFalse(eval_runner._hits_obstacle(before, 0.0, hit))
            self.assertTrue(eval_runner._hits_obstacle(after, 0.0, hit))
            collision = self._run_pose_case(
                root,
                checkpoint,
                "last_step_hit",
                goal_x=hit_goal,
                obstacle_x=obstacle_x,
                obstacle_y=0.0,
            )

            success_row = success["episodes"][0]
            self.assertTrue(success_row["success"])
            self.assertIsNone(success_row["failure_mode"])
            self.assertEqual(success_row["length"], eval_runner.MAX_STEPS)
            self.assertEqual(success["success_rate"], 1.0)
            self.assertEqual(success["failure_modes"], {})
            success_poses = json.loads(
                Path(success_row["video_path"]).read_text(encoding="utf-8")
            )["poses"]
            self.assertEqual(len(success_poses), eval_runner.MAX_STEPS + 1)
            self.assertGreater(
                math.hypot(goal - success_poses[-2][0], success_poses[-2][1]),
                tol,
            )
            self.assertLessEqual(
                math.hypot(goal - success_poses[-1][0], success_poses[-1][1]),
                tol,
            )
            validate_report(success)

            collision_row = collision["episodes"][0]
            self.assertFalse(collision_row["success"])
            self.assertEqual(collision_row["failure_mode"], "collision")
            self.assertEqual(collision_row["length"], eval_runner.MAX_STEPS)
            self.assertEqual(collision["success_rate"], 0.0)
            self.assertEqual(collision["failure_modes"], {"collision": 1})
            collision_poses = json.loads(
                Path(collision_row["video_path"]).read_text(encoding="utf-8")
            )["poses"]
            # The contacting step is not applied, so the trace stops one short.
            self.assertEqual(len(collision_poses), eval_runner.MAX_STEPS)
            validate_report(collision)

    def test_checkpoint_changes_the_clear_episode_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slow = self._run(
                root, self._write_checkpoint(root, 0.2), episodes=1, seed=0
            )
            fast = self._run(
                root, self._write_checkpoint(root, 0.8), episodes=1, seed=0
            )
        slow_len = next(
            row["length"]
            for row in slow["episodes"]
            if row["scenario_id"] == "clear_goal"
        )
        fast_len = next(
            row["length"]
            for row in fast["episodes"]
            if row["scenario_id"] == "clear_goal"
        )
        self.assertNotEqual(slow["checkpoint_sha256"], fast["checkpoint_sha256"])
        self.assertGreater(slow_len, fast_len)
        self.assertEqual(slow["success_rate"], fast["success_rate"])

    def test_report_schema_rejects_a_broken_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self._run(
                root, self._write_checkpoint(root, 0.4), episodes=1, seed=0
            )
        report["success_rate"] = 1.5
        with self.assertRaises(EvalValidationError):
            validate_report(report)
        report["success_rate"] = 1.0
        del report["schema"]
        with self.assertRaises(EvalValidationError):
            validate_report(report)

    def test_hook_and_job(self) -> None:
        self.assertIsNone(
            evaluate_after_train("missing-checkpoint", "gazebo-nav-regression")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_checkpoint(root, 0.4)
            scenarios = root / "scenarios.yaml"
            scenarios.write_text(_FIXTURE, encoding="utf-8")
            report = evaluate_after_train(
                checkpoint,
                "gazebo-nav-regression",
                enabled=True,
                episodes=1,
                seed=4,
                scenarios=scenarios,
            )
            self.assertIsNotNone(report)
            job = run_eval_job(
                {
                    "type": "eval",
                    "checkpoint": str(checkpoint),
                    "suite": "gazebo-nav-regression",
                    "episodes": 1,
                    "seed": 4,
                    "scenarios": str(scenarios),
                    "enabled": True,
                }
            )
        assert report is not None and job is not None
        self.assertEqual(report["episodes"], job["episodes"])
        self.assertIsNone(
            run_eval_job(
                {
                    "type": "eval",
                    "checkpoint": "ignored",
                    "suite": "gazebo-nav-regression",
                    "enabled": False,
                }
            )
        )
        with self.assertRaises(EvalError):
            run_eval_job({"type": "train", "checkpoint": "x", "suite": "y"})

    def test_cli_writes_a_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write_checkpoint(root, 0.4)
            scenarios = root / "scenarios.yaml"
            scenarios.write_text(_FIXTURE, encoding="utf-8")
            out = root / "report.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "eval",
                        str(checkpoint),
                        "--suite",
                        "gazebo-nav-regression",
                        "--scenarios",
                        str(scenarios),
                        "--episodes",
                        "1",
                        "--seed",
                        "2",
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("ohho.eval/v1", stdout.getvalue())
            report = json.loads(out.read_text(encoding="utf-8"))
        validate_report(report)
        self.assertEqual(report["seed"], 2)
        with redirect_stderr(io.StringIO()):
            code = main(
                [
                    "eval",
                    "missing-checkpoint",
                    "--suite",
                    "gazebo-nav-regression",
                    "--episodes",
                    "1",
                ]
            )
        self.assertEqual(code, 1)

    def test_real_suite_repeats_when_the_checkout_is_present(self) -> None:
        root = _omnibot_root()
        if root is None:
            self.skipTest("OmniBot nav regression YAML is not checked out")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            checkpoint.write_text("{}\n", encoding="utf-8")
            kwargs = {
                "episodes": 1,
                "seed": 11,
                "scenarios": root / _REL,
            }
            first = run_eval(checkpoint, "gazebo-nav-regression", **kwargs)
            second = run_eval(checkpoint, "gazebo-nav-regression", **kwargs)
        self.assertEqual(first["success_rate"], second["success_rate"])
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertEqual(len(first["scenario_ids"]), len(first["episodes"]))
        validate_report(first)

    def test_package_does_not_import_train_or_sweep(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ohho" / "eval"
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("ohho.train", text, str(path))
            self.assertNotIn("ohho.sweep", text, str(path))
            self.assertNotIn("ohho.recipe", text, str(path))


if __name__ == "__main__":
    unittest.main()
