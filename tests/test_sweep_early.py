"""ASHA and median early stopping, including the 32-trial step budget."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ohho.train.sweep import SCHEMA_ID, SweepError, expand, format_plan, validate_sweep
from ohho.train.sweep_early import (
    ACTION_CONTINUE,
    ACTION_STOP,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PRUNED,
    STATUS_RUNNING,
    EarlyStopping,
    simulate_asha_budget,
    synthetic_learning_curve,
)
from ohho.train.sweep_run import resume_sweep, start_sweep


def _rows(count: int) -> list[dict]:
    return [
        {
            "index": index,
            "status": "pending",
            "checkpoint": None,
            "metrics": [],
            "decision": None,
            "error": None,
        }
        for index in range(count)
    ]


def _asha(**overrides: object) -> dict:
    config: dict = {
        "metric": "val_loss",
        "mode": "min",
        "rule": "asha",
        "rungs": [1],
        "reduction_factor": 2,
        "grace_period": 0,
    }
    config.update(overrides)
    return config


class TestEarlyConfig(unittest.TestCase):
    def test_rejects_bad_rules_rungs_and_grace(self) -> None:
        samples = [
            {"metric": "loss", "rule": "hyperband"},
            {"metric": "loss", "rule": "asha"},
            {"metric": "loss", "rule": "asha", "rungs": []},
            {"metric": "loss", "rule": "median", "rungs": [3, 1]},
            {"metric": "loss", "rule": "median", "rungs": [1, 2], "grace_period": 3},
            {"metric": "loss", "reduction_factor": 1},
            {"metric": "loss", "rungs": [1, 1]},
            {"metric": "loss", "rungs": [0, 1]},
        ]
        for early in samples:
            with self.subTest(early=early):
                with self.assertRaises(SweepError):
                    validate_sweep({"schema": SCHEMA_ID, "early_stopping": early})

    def test_plan_text_shows_asha_rungs(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "early_stopping": {
                    "metric": "val_loss",
                    "rule": "asha",
                    "rungs": [1, 3, 9, 27, 81],
                    "reduction_factor": 3,
                    "grace_period": 1,
                },
            }
        )
        text = format_plan(plan)
        self.assertIn("rule=asha", text)
        self.assertIn("reduction_factor=3", text)
        self.assertIn("grace_period=1", text)
        self.assertIn("rungs=1,3,9,27,81", text)
        self.assertEqual(plan.early_stopping["mode"], "min")


class TestAsha(unittest.TestCase):
    def test_synchronous_rung_keeps_the_top_fraction(self) -> None:
        stopper = EarlyStopping(_asha(reduction_factor=2, rungs=[1, 2]))
        rows = _rows(4)
        for index in range(4):
            stopper.report(rows, index, 1, index, checkpoint=f"{index}:1")
        decisions = [stopper.decide(rows, index) for index in range(4)]
        self.assertEqual(
            [item.action for item in decisions],
            [ACTION_CONTINUE, ACTION_CONTINUE, ACTION_STOP, ACTION_STOP],
        )
        self.assertEqual(rows[2]["status"], STATUS_PRUNED)
        self.assertEqual(rows[2]["checkpoint"], "2:1")
        self.assertEqual(rows[2]["metrics"], [{"step": 1, "value": 2}])
        self.assertIsNone(rows[2]["error"])
        self.assertNotEqual(rows[2]["status"], STATUS_FAILED)
        for index in (0, 1):
            stopper.report(rows, index, 2, index, checkpoint=f"{index}:2")
        second = [stopper.decide(rows, index) for index in (0, 1)]
        self.assertEqual(second[0].action, ACTION_CONTINUE)
        self.assertEqual(second[1].action, ACTION_STOP)
        self.assertEqual(rows[1]["checkpoint"], "1:2")
        self.assertEqual(len(rows[1]["metrics"]), 2)
        self.assertEqual(rows[0]["status"], STATUS_RUNNING)

    def test_maximize_keeps_the_larger_values(self) -> None:
        stopper = EarlyStopping(_asha(mode="max", reduction_factor=2))
        rows = _rows(4)
        for index, value in enumerate((1, 2, 3, 4)):
            stopper.report(rows, index, 1, value)
        actions = [stopper.decide(rows, index).action for index in range(4)]
        self.assertEqual(
            actions,
            [ACTION_STOP, ACTION_STOP, ACTION_CONTINUE, ACTION_CONTINUE],
        )

    def test_ties_at_the_cutoff_continue(self) -> None:
        stopper = EarlyStopping(_asha(reduction_factor=2))
        rows = _rows(3)
        for index, value in enumerate((1, 1, 2)):
            stopper.report(rows, index, 1, value)
        actions = [stopper.decide(rows, index).action for index in range(3)]
        self.assertEqual(actions, [ACTION_CONTINUE, ACTION_CONTINUE, ACTION_STOP])

    def test_async_continue_is_not_revoked(self) -> None:
        stopper = EarlyStopping(_asha(reduction_factor=3))
        rows = _rows(3)
        stopper.report(rows, 0, 1, 9)
        first = stopper.decide(rows, 0)
        self.assertEqual(first.action, ACTION_CONTINUE)
        self.assertEqual(first.reason, "not enough trials at this step")
        stopper.report(rows, 1, 1, 0)
        stopper.report(rows, 2, 1, 1)
        again = stopper.decide(rows, 0)
        self.assertEqual(again.action, ACTION_CONTINUE)
        self.assertEqual(again.reason, "recorded decision")
        self.assertEqual(rows[0]["status"], STATUS_RUNNING)
        self.assertEqual(stopper.decide(rows, 2).action, ACTION_STOP)
        self.assertEqual(rows[2]["status"], STATUS_PRUNED)

    def test_same_values_stop_when_the_rung_is_reported_first(self) -> None:
        stopper = EarlyStopping(_asha(reduction_factor=3))
        rows = _rows(3)
        stopper.report(rows, 0, 1, 9)
        stopper.report(rows, 1, 1, 0)
        stopper.report(rows, 2, 1, 1)
        self.assertEqual(stopper.decide(rows, 0).action, ACTION_STOP)
        self.assertEqual(rows[0]["status"], STATUS_PRUNED)

    def test_before_grace_and_off_rung_continue(self) -> None:
        stopper = EarlyStopping(_asha(rungs=[2], grace_period=2, reduction_factor=2))
        rows = _rows(1)
        stopper.report(rows, 0, 1, 5)
        early = stopper.decide(rows, 0)
        self.assertEqual(early.action, ACTION_CONTINUE)
        self.assertEqual(early.reason, "before grace period")
        stopper = EarlyStopping(_asha(rungs=[2], grace_period=0))
        rows = _rows(1)
        stopper.report(rows, 0, 1, 5)
        off = stopper.decide(rows, 0)
        self.assertEqual(off.reason, "step is not a rung")


class TestMedianAndPatience(unittest.TestCase):
    def test_median_stops_only_after_grace_and_enough_peers(self) -> None:
        stopper = EarlyStopping(
            {
                "metric": "loss",
                "rule": "median",
                "mode": "min",
                "reduction_factor": 2,
                "grace_period": 2,
            }
        )
        rows = _rows(2)
        stopper.report(rows, 0, 1, 5)
        stopper.report(rows, 1, 1, 0)
        self.assertEqual(stopper.decide(rows, 0).reason, "before grace period")
        self.assertEqual(stopper.decide(rows, 1).action, ACTION_CONTINUE)
        stopper.report(rows, 0, 2, 5)
        stopper.report(rows, 1, 2, 0)
        stopped = stopper.decide(rows, 0)
        self.assertEqual(stopped.action, ACTION_STOP)
        self.assertEqual(stopped.reason, "worse than the median")
        self.assertEqual(stopper.decide(rows, 1).reason, "at or better than the median")
        self.assertEqual(rows[0]["status"], STATUS_PRUNED)
        self.assertEqual(rows[1]["status"], STATUS_RUNNING)

        alone = EarlyStopping(
            {
                "metric": "loss",
                "rule": "median",
                "mode": "max",
                "reduction_factor": 2,
                "grace_period": 0,
            }
        )
        one = _rows(1)
        alone.report(one, 0, 1, 1)
        self.assertEqual(alone.decide(one, 0).reason, "not enough trials at this step")
        pair = _rows(2)
        alone.report(pair, 0, 1, 1)
        alone.report(pair, 1, 1, 5)
        self.assertEqual(alone.decide(pair, 0).action, ACTION_STOP)
        self.assertEqual(alone.decide(pair, 1).action, ACTION_CONTINUE)

    def test_patience_uses_min_delta_and_keeps_the_checkpoint(self) -> None:
        stopper = EarlyStopping(
            {
                "metric": "loss",
                "rule": "patience",
                "mode": "min",
                "patience": 2,
                "min_delta": 0.1,
                "grace_period": 0,
            }
        )
        rows = _rows(1)
        stopper.report(rows, 0, 1, 1.0, checkpoint="first")
        self.assertEqual(stopper.decide(rows, 0).action, ACTION_CONTINUE)
        stopper.report(rows, 0, 2, 0.95)
        self.assertEqual(stopper.decide(rows, 0).reason, "within patience")
        stopper.report(rows, 0, 3, 0.95)
        stopped = stopper.decide(rows, 0)
        self.assertEqual(stopped.action, ACTION_STOP)
        self.assertEqual(stopped.reason, "patience exhausted")
        self.assertEqual(rows[0]["status"], STATUS_PRUNED)
        self.assertEqual(rows[0]["checkpoint"], "first")
        self.assertEqual(len(rows[0]["metrics"]), 3)

        improving = EarlyStopping(
            {
                "metric": "loss",
                "rule": "patience",
                "mode": "min",
                "patience": 1,
                "min_delta": 0.1,
            }
        )
        good = _rows(1)
        for step, value in ((1, 1.0), (2, 0.8), (3, 0.6)):
            improving.report(good, 0, step, value)
            self.assertEqual(improving.decide(good, 0).action, ACTION_CONTINUE)

        grace = EarlyStopping(
            {
                "metric": "loss",
                "rule": "patience",
                "patience": 1,
                "min_delta": 0,
                "grace_period": 3,
            }
        )
        waited = _rows(1)
        grace.report(waited, 0, 1, 5)
        grace.report(waited, 0, 2, 5)
        self.assertEqual(grace.decide(waited, 0).reason, "before grace period")
        grace.report(waited, 0, 3, 5)
        self.assertEqual(grace.decide(waited, 0).action, ACTION_CONTINUE)
        grace.report(waited, 0, 4, 5)
        self.assertEqual(grace.decide(waited, 0).action, ACTION_STOP)


class TestTrialTable(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._old = os.environ.get("OHHO_SWEEP_DIR")
        os.environ["OHHO_SWEEP_DIR"] = str(self.root)

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("OHHO_SWEEP_DIR", None)
        else:
            os.environ["OHHO_SWEEP_DIR"] = self._old
        self._tmp.cleanup()

    def test_pruned_checkpoint_survives_resume(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "strategy": "grid",
            "seed": 1,
            "dimensions": [{"name": "i", "type": "int", "values": [0, 1, 2, 3, 4, 5]}],
            "objectives": [
                {"name": "val_loss", "direction": "minimize", "primary": True}
            ],
            "early_stopping": _asha(rungs=[1, 2], reduction_factor=2, grace_period=1),
        }
        run = start_sweep(doc, sweep_id="asha", root=self.root)
        self.assertEqual([trial.index for trial in run.ask(4)], [0, 1, 2, 3])
        for index in range(4):
            run.report(index, 1, index, checkpoint=f"{index}:1")
        actions = [run.decide(index).action for index in range(4)]
        self.assertEqual(
            actions,
            [ACTION_CONTINUE, ACTION_CONTINUE, ACTION_STOP, ACTION_STOP],
        )
        for index in (0, 1):
            run.report(index, 2, index, checkpoint=f"{index}:2")
        self.assertEqual(run.decide(0).action, ACTION_CONTINUE)
        self.assertEqual(run.decide(1).action, ACTION_STOP)
        run.tell(0, {"val_loss": 0})
        with self.assertRaises(SweepError):
            run.tell(1, {"val_loss": 1})
        with self.assertRaises(SweepError):
            run.mark_failed(2, "should stay pruned")
        saved = json.loads(
            (self.root / "asha" / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved["trials"][2]["status"], STATUS_PRUNED)
        self.assertEqual(saved["trials"][2]["checkpoint"], "2:1")
        self.assertEqual(saved["trials"][1]["checkpoint"], "1:2")
        self.assertIsNone(saved["trials"][2]["values"])
        self.assertEqual(saved["trials"][0]["status"], STATUS_COMPLETED)
        self.assertEqual(saved["trials"][0]["values"], {"val_loss": 0})
        self.assertIsNone(saved["trials"][2]["error"])

        resumed = resume_sweep("asha", root=self.root)
        table = {row["index"]: row for row in resumed.trial_table()}
        self.assertEqual(table[2]["status"], STATUS_PRUNED)
        self.assertEqual(table[2]["checkpoint"], "2:1")
        self.assertEqual(table[2]["metrics"], [{"step": 1, "value": 2}])
        self.assertEqual(table[1]["status"], STATUS_PRUNED)
        self.assertEqual(len(table[1]["metrics"]), 2)
        self.assertEqual(table[0]["status"], STATUS_COMPLETED)
        self.assertEqual(table[0]["metrics"][1], {"step": 2, "value": 0})
        self.assertEqual(resumed.decide(2).action, ACTION_STOP)
        self.assertEqual(resumed.decide(2).reason, "trial is pruned")
        nxt = resumed.ask()[0]
        self.assertEqual(nxt.index, 4)
        self.assertEqual(resumed.trial_table()[2]["status"], STATUS_PRUNED)

    def test_failed_is_distinct_and_survives_resume(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "seed": 2,
            "objectives": [{"name": "loss", "direction": "minimize", "primary": True}],
            "early_stopping": {"metric": "loss", "rule": "patience", "patience": 3},
        }
        run = start_sweep(doc, sweep_id="failed", root=self.root)
        run.ask()
        run.report(0, 1, 0.4, checkpoint="ckpt-a")
        run.mark_failed(0, error="diverged")
        with self.assertRaises(SweepError):
            run.report(0, 2, 0.3, checkpoint="ckpt-b")
        with self.assertRaises(SweepError):
            run.tell(0, {"loss": 0.4})
        row = run.trial_table()[0]
        self.assertEqual(row["status"], STATUS_FAILED)
        self.assertNotEqual(row["status"], STATUS_PRUNED)
        self.assertEqual(row["checkpoint"], "ckpt-a")
        self.assertEqual(row["metrics"], [{"step": 1, "value": 0.4}])
        self.assertEqual(row["error"], "diverged")
        self.assertIsNone(row["values"])
        resumed = resume_sweep("failed", root=self.root)
        again = resumed.trial_table()[0]
        self.assertEqual(again["status"], STATUS_FAILED)
        self.assertEqual(again["checkpoint"], "ckpt-a")
        self.assertEqual(again["error"], "diverged")
        self.assertEqual(resumed.decide(0).reason, "trial failed")

    def test_bad_report_does_not_change_the_saved_row(self) -> None:
        doc = {
            "schema": SCHEMA_ID,
            "early_stopping": {"metric": "loss", "rule": "median", "grace_period": 0},
        }
        run = start_sweep(doc, sweep_id="rollback", root=self.root)
        run.ask()
        run.report(0, 1, 0.2, checkpoint="a")
        with self.assertRaises(SweepError):
            run.report(0, 1, 0.3, checkpoint="b")
        with self.assertRaises(SweepError):
            run.report(0, 2, float("nan"))
        row = run.trial_table()[0]
        self.assertEqual(row["metrics"], [{"step": 1, "value": 0.2}])
        self.assertEqual(row["checkpoint"], "a")
        self.assertEqual(row["status"], STATUS_RUNNING)
        with self.assertRaises(SweepError):
            run.report(1, 1, 0.2)
        plain = start_sweep({"schema": SCHEMA_ID}, sweep_id="plain", root=self.root)
        plain.ask()
        with self.assertRaises(SweepError) as caught:
            plain.report(0, 1, 0.1)
        self.assertIn("early_stopping", str(caught.exception))


class TestBenchmark(unittest.TestCase):
    def test_curves_do_not_cross(self) -> None:
        for step in (1, 3, 9, 27, 81):
            losses = [synthetic_learning_curve(trial, step) for trial in range(32)]
            self.assertEqual(losses, sorted(losses))
            self.assertEqual(len(set(losses)), 32)

    def test_thirty_two_trial_asha_uses_at_most_half_the_steps(self) -> None:
        result = simulate_asha_budget()
        self.assertEqual(result["trials"], 32)
        self.assertEqual(result["rungs"], [1, 3, 9, 27, 81])
        self.assertEqual(result["reduction_factor"], 3)
        self.assertEqual(result["steps_without_asha"], 32 * 81)
        self.assertEqual(result["steps_with_asha"], 142)
        self.assertLessEqual(
            result["steps_with_asha"], result["steps_without_asha"] * 0.5
        )
        self.assertLessEqual(result["ratio"], 0.5)
        self.assertEqual(result["pruned"], 31)
        self.assertEqual(result["finished"], 1)
        self.assertEqual(result["finished_trials"], [0])
        self.assertEqual(result["failed"], 0)
        self.assertTrue(result["pruned_keep_checkpoint_and_metrics"])


if __name__ == "__main__":
    unittest.main()
