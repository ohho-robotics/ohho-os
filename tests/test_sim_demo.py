"""Tests for the OHH-134 Sim Demo Harness and Evaluation Logger."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from ohho.cli import main as cli_main
from ohho.sim_demo import DEFAULT_SHOTS, SimDemoHarness, main as harness_main


class TestSimDemoHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.schema_path = self.repo_root / "evals" / "schemas" / "sim_demo_schema.json"
        self.example_log_path = (
            self.repo_root / "evals" / "examples" / "ohh-134-go2-mujoco-demo.json"
        )

    def test_schema_validates_example_log(self) -> None:
        """Validate that the canonical example log adheres strictly to the schema."""
        self.assertTrue(
            self.schema_path.exists(), f"Missing schema at {self.schema_path}"
        )
        self.assertTrue(
            self.example_log_path.exists(),
            f"Missing example log at {self.example_log_path}",
        )

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(self.example_log_path, "r", encoding="utf-8") as f:
            example_log = json.load(f)

        try:
            import jsonschema

            jsonschema.validate(instance=example_log, schema=schema)
        except ImportError:
            # Fallback manual validation if jsonschema is not installed
            self.assertEqual(example_log.get("schema_version"), "1.0.0")
            self.assertTrue(example_log.get("simulation_only"))
            self.assertFalse(example_log.get("hardware_tested"))
            self.assertEqual(len(example_log.get("shots", [])), 6)
            self.assertEqual(
                example_log.get("summary", {}).get("overall_result"), "PASS"
            )

    def test_shot_list_conformance(self) -> None:
        """Verify the 6 shots and their timing parameters within the 60-90s window."""
        self.assertEqual(len(DEFAULT_SHOTS), 6)
        expected_ids = [
            "doctor",
            "sim_shim_start",
            "typed_goal",
            "tool_call_log",
            "safety_gate_demo",
            "end_card",
        ]
        actual_ids = [s["id"] for s in DEFAULT_SHOTS]
        self.assertEqual(actual_ids, expected_ids)

        total_target_s = sum(s["target_duration_s"] for s in DEFAULT_SHOTS)
        self.assertGreaterEqual(total_target_s, 60.0)
        self.assertLessEqual(total_target_s, 90.0)

    def test_harness_runs_end_to_end_in_sim(self) -> None:
        """Run the full demo harness in fast sim mode and verify output and schema compliance."""
        out_buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "eval_report.json"
            harness = SimDemoHarness(
                robot_id="unitree-go2",
                sim_mode=True,
                fast=True,
                out_path=str(out_file),
                out_stream=out_buf,
            )
            report = harness.run()

            # Verify report structure
            self.assertEqual(report["schema_version"], "1.0.0")
            self.assertTrue(report["simulation_only"])
            self.assertFalse(report["hardware_tested"])
            self.assertEqual(report["robot"]["id"], "unitree-go2")
            self.assertEqual(len(report["shots"]), 6)

            summary = report["summary"]
            self.assertEqual(summary["total_shots"], 6)
            self.assertEqual(summary["passed_shots"], 6)
            self.assertEqual(summary["failed_shots"], 0)
            self.assertTrue(summary["safety_clamp_verified"])
            self.assertTrue(summary["prohibited_action_refused"])
            self.assertTrue(summary["estop_latched"])
            self.assertTrue(summary["simulation_only_banner_verified"])
            self.assertEqual(summary["overall_result"], "PASS")

            # Verify file was written and matches schema
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["eval_id"], report["eval_id"])

            with open(self.schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            try:
                import jsonschema

                jsonschema.validate(instance=saved, schema=schema)
            except ImportError:
                pass

        # Verify terminal output contains required banners and cues
        output = out_buf.getvalue()
        self.assertIn("SIMULATION ONLY", output)
        self.assertIn("NOT TESTED ON PHYSICAL HARDWARE", output)
        self.assertIn("OHH-135", output)
        self.assertIn("OVERLAY CUE:", output)
        self.assertIn(
            "SafetyGate: CLAMPED requested vx=2.0 m/s -> applied vx=0.500 m/s", output
        )
        self.assertIn("SafetyGate: REFUSED", output)
        self.assertIn("EMERGENCY STOP LATCHED", output)

    def test_cli_sim_demo_invocation(self) -> None:
        """Verify that 'ohho sim-demo' CLI command runs and exits 0."""
        code = cli_main(["sim-demo", "--sim", "--fast", "--check-overlays"])
        self.assertEqual(code, 0)

    def test_harness_standalone_main(self) -> None:
        """Verify harness main function execution."""
        code = harness_main(["--sim", "--fast"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
