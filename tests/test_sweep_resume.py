"""Derived sweep seeds and disk resume of a stopped sweep."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ohho.cli import build_parser, main
from ohho.train.sweep import (
    SCHEMA_ID,
    SweepError,
    expand,
    sampler_seed,
    trial_training_seed,
)
from ohho.train.sweep_sampler import SAMPLER_SCHEMA_ID, SweepSampler
from ohho.train.sweep_run import (
    RUN_SCHEMA_ID,
    load_run_schema,
    resume_sweep,
    start_sweep,
)
from ohho.train.sweep_seed import derived_trial_seed


def _sha(sweep_seed: int, domain: str, trial_index: int | None = None) -> int:
    parts: list[object] = ["ohho.sweep.seed/v1", sweep_seed, domain]
    if trial_index is not None:
        parts.append(trial_index)
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _grid(seed: int = 0) -> dict:
    return {
        "schema": SCHEMA_ID,
        "strategy": "grid",
        "seed": seed,
        "recipe": {"task": "push", "optimizer": {"weight_decay": 0.0}},
        "estimate": {"gpu_hours_per_trial": 0.25},
        "dimensions": [
            {"name": "a", "type": "categorical", "values": ["a", "b"]},
            {"name": "b", "type": "int", "values": [1, 2, 3]},
        ],
    }


def _random(seed: int = 1) -> dict:
    return {
        "schema": SCHEMA_ID,
        "strategy": "random",
        "seed": seed,
        "budget": {"max_trials": 6},
        "dimensions": [
            {"name": "x", "type": "float", "low": 0, "high": 1},
            {"name": "n", "type": "int", "low": 0, "high": 4},
        ],
    }


def _adaptive(strategy: str, seed: int = 4) -> dict:
    objectives: list[dict] = [
        {"name": "loss", "direction": "minimize", "primary": True}
    ]
    if strategy == "multi":
        objectives = [
            {"name": "success_rate", "direction": "maximize"},
            {"name": "latency_p95_ms", "direction": "minimize"},
        ]
    return {
        "schema": SCHEMA_ID,
        "strategy": strategy,
        "seed": seed,
        "budget": {"max_trials": 6},
        "sampler": {"startup_trials": 2, "candidates": 8, "gamma": 0.25},
        "objectives": objectives,
        "dimensions": [{"name": "x", "type": "float", "low": 0, "high": 1}],
    }


def _values(doc: dict, overrides: dict) -> dict:
    names = [item["name"] for item in doc.get("objectives") or []]
    if not names:
        return {}
    x = float(next(iter(overrides.values())))
    if names == ["loss"]:
        return {"loss": (x - 0.2) ** 2}
    return {names[0]: x, names[1]: 1.0 - x}


def _play(doc: dict, count: int) -> list:
    sampler = SweepSampler.from_sweep(doc)
    played = []
    for _ in range(count):
        trial = sampler.ask()[0]
        sampler.tell(trial.index, _values(doc, trial.overrides))
        played.append(trial)
    return played


class TestDerivedSeeds(unittest.TestCase):
    def test_sha256_derivation_is_stable(self) -> None:
        self.assertEqual(sampler_seed(0), 0)
        self.assertEqual(sampler_seed(99), 99)
        self.assertEqual(trial_training_seed(1, 0), _sha(1, "trial", 0))
        self.assertEqual(trial_training_seed(99, 4), _sha(99, "trial", 4))
        self.assertNotEqual(sampler_seed(1), trial_training_seed(1, 0))
        seeds = [trial_training_seed(5, index) for index in range(8)]
        self.assertEqual(len(set(seeds)), 8)
        self.assertTrue(all(0 <= seed < 2**32 for seed in seeds))
        with self.assertRaises(SweepError):
            trial_training_seed(-1, 0)
        with self.assertRaises(SweepError):
            trial_training_seed(True, 0)  # type: ignore[arg-type]
        with self.assertRaises(SweepError):
            derived_trial_seed({"seed": -1})
        with self.assertRaises(SweepError):
            derived_trial_seed({"seed": True})
        with self.assertRaises(SweepError):
            derived_trial_seed([])  # type: ignore[arg-type]

    def test_same_spec_and_seed_repeat_grid_random_and_tpe(self) -> None:
        for name, doc in (
            ("grid", _grid(3)),
            ("random", _random(3)),
            ("tpe", _adaptive("tpe", 3)),
            ("bayesian", _adaptive("bayesian", 3)),
            ("multi", _adaptive("multi", 3)),
        ):
            with self.subTest(strategy=name):
                left = _play(doc, 6)
                right = _play(doc, 6)
                self.assertEqual(
                    [trial.overrides for trial in left],
                    [trial.overrides for trial in right],
                )
                self.assertEqual(
                    [trial.seed for trial in left],
                    [trial_training_seed(3, index) for index in range(6)],
                )
                self.assertEqual(
                    [trial.seed for trial in left],
                    [trial.seed for trial in right],
                )

    def test_grid_order_ignores_seed_but_training_seeds_do_not(self) -> None:
        left = expand(_grid(1))
        right = expand(_grid(2))
        self.assertEqual(
            [trial.sets for trial in left.trials],
            [trial.sets for trial in right.trials],
        )
        self.assertNotEqual(
            [trial.seed for trial in left.trials],
            [trial.seed for trial in right.trials],
        )
        self.assertEqual(left.sampler_seed, sampler_seed(1))
        self.assertEqual(
            [trial.seed for trial in left.trials],
            [trial_training_seed(1, index) for index in range(len(left.trials))],
        )

    def test_random_plan_seeds_match_the_sampler(self) -> None:
        doc = _random(8)
        plan = expand(doc)
        got = SweepSampler.from_sweep(doc).ask(len(plan.trials))
        self.assertEqual(
            [trial.overrides for trial in got],
            [trial.overrides for trial in plan.trials],
        )
        self.assertEqual(
            [trial.seed for trial in got],
            [trial.seed for trial in plan.trials],
        )
        self.assertEqual(
            [derived_trial_seed({"seed": trial.seed}) for trial in got],
            [trial.seed for trial in plan.trials],
        )


class TestResumeRun(unittest.TestCase):
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

    def test_stopped_sweep_continues_without_duplicate_trials(self) -> None:
        cases = {
            "grid": _grid(4),
            "random": _random(4),
            "tpe": _adaptive("tpe", 4),
            "bayesian": _adaptive("bayesian", 4),
            "multi": _adaptive("multi", 4),
        }
        for name, doc in cases.items():
            with self.subTest(strategy=name):
                expected = _play(doc, 6)
                run = start_sweep(doc, sweep_id=name, root=self.root)
                for index in range(3):
                    trial = run.ask()[0]
                    self.assertEqual(trial.overrides, expected[index].overrides)
                    self.assertEqual(trial.seed, expected[index].seed)
                    run.tell(trial.index, _values(doc, trial.overrides))
                saved = json.loads(
                    (self.root / name / "run.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["schema"], RUN_SCHEMA_ID)
                self.assertEqual(saved["sampler"]["schema"], SAMPLER_SCHEMA_ID)
                self.assertEqual(saved["sampler_seed"], sampler_seed(4))
                self.assertEqual([row["index"] for row in saved["trials"]], [0, 1, 2])
                self.assertEqual(
                    [row["seed"] for row in saved["trials"]],
                    [trial_training_seed(4, index) for index in range(3)],
                )
                resumed = resume_sweep(name, root=self.root)
                self.assertEqual(
                    [row["index"] for row in resumed.trial_table()], [0, 1, 2]
                )
                for index in range(3, 6):
                    trial = resumed.ask()[0]
                    self.assertEqual(trial.index, index)
                    self.assertEqual(trial.overrides, expected[index].overrides)
                    self.assertEqual(trial.seed, expected[index].seed)
                    row = resumed.trial_table()[-1]
                    self.assertEqual(derived_trial_seed(row), trial.seed)
                    self.assertEqual(row["index"], index)
                    resumed.tell(trial.index, _values(doc, trial.overrides))
                indexes = [row["index"] for row in resumed.trial_table()]
                self.assertEqual(indexes, list(range(6)))
                self.assertEqual(len(indexes), len(set(indexes)))
                again = resume_sweep(name)
                with self.assertRaises(SweepError) as caught:
                    again.ask()
                self.assertIn("exhausted", str(caught.exception))
                self.assertEqual(
                    [row["index"] for row in again.trial_table()], list(range(6))
                )

    def test_pending_trial_is_kept_and_not_proposed_again(self) -> None:
        doc = _random(2)
        run = start_sweep(doc, sweep_id="pending", root=self.root)
        first = run.ask()[0]
        resumed = resume_sweep("pending", root=self.root)
        table = resumed.trial_table()
        self.assertEqual([row["index"] for row in table], [0])
        self.assertIsNone(table[0]["values"])
        self.assertEqual(table[0]["seed"], first.seed)
        nxt = resumed.ask()[0]
        self.assertEqual(nxt.index, 1)
        self.assertEqual(
            [row["index"] for row in resumed.trial_table()],
            [0, 1],
        )

    def test_bad_id_duplicate_and_tampered_seed_are_rejected(self) -> None:
        doc = _grid(1)
        start_sweep(doc, sweep_id="kept", root=self.root)
        with self.assertRaises(SweepError) as caught:
            start_sweep(doc, sweep_id="kept", root=self.root)
        self.assertIn("already exists", str(caught.exception))
        with self.assertRaises(SweepError):
            start_sweep(doc, sweep_id="../escape", root=self.root)
        with self.assertRaises(SweepError) as caught:
            resume_sweep("missing", root=self.root)
        self.assertIn("not found", str(caught.exception))
        path = self.root / "kept" / "run.json"
        run = resume_sweep("kept", root=self.root)
        run.ask()
        payload = json.loads(path.read_text(encoding="utf-8"))
        original = payload["trials"][0]["seed"]
        wrong = 0 if original else 1
        payload["trials"][0]["seed"] = wrong
        payload["sampler"]["trials"][0]["seed"] = wrong
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SweepError) as caught:
            resume_sweep("kept", root=self.root)
        self.assertIn("seed", str(caught.exception))
        payload["trials"][0]["seed"] = original
        payload["sampler"]["trials"][0]["seed"] = original
        payload["trials"].append(json.loads(json.dumps(payload["trials"][0])))
        payload["sampler"]["trials"].append(
            json.loads(json.dumps(payload["sampler"]["trials"][0]))
        )
        payload["sampler"]["cursor"] = len(payload["sampler"]["trials"])
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SweepError) as caught:
            resume_sweep("kept", root=self.root)
        self.assertIn("0..n-1", str(caught.exception))

    def test_run_schema_is_packaged(self) -> None:
        schema = load_run_schema()
        self.assertEqual(schema["title"], RUN_SCHEMA_ID)
        self.assertIn("seed", schema["$defs"]["trial"]["required"])


class TestResumeCli(unittest.TestCase):
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

    def test_parser(self) -> None:
        args = build_parser().parse_args(
            ["sweep", "resume", "abc", "--count", "3", "--json"]
        )
        self.assertEqual(args.sweep_command, "resume")
        self.assertEqual(args.sweep_id, "abc")
        self.assertEqual(args.count, 3)
        self.assertTrue(args.json)

    def test_start_and_resume_do_not_repeat_trials(self) -> None:
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(_random(2)), encoding="utf-8")
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "start", str(spec), "--id", "cli-run"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("id: cli-run", out.getvalue())
        self.assertIn(f"sampler_seed: {sampler_seed(2)}", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "resume", "cli-run", "--count", "2", "--json"])
        self.assertEqual(code, 0, err.getvalue())
        first = json.loads(out.getvalue())
        self.assertEqual([row["index"] for row in first["proposed"]], [0, 1])
        self.assertEqual(
            [row["seed"] for row in first["proposed"]],
            [trial_training_seed(2, 0), trial_training_seed(2, 1)],
        )
        self.assertEqual(
            [row["seed"] for row in first["trial_table"]],
            [trial_training_seed(2, 0), trial_training_seed(2, 1)],
        )

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "resume", "cli-run", "--count", "2"])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        indexes = [
            int(line.split(":", 1)[1])
            for line in text.splitlines()
            if line.startswith("trial:")
        ]
        self.assertEqual(indexes, [2, 3])
        self.assertIn(f"seed: {trial_training_seed(2, 2)}", text)
        self.assertNotIn("trial: 0\n", text)
        self.assertNotIn("trial: 1\n", text)

        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sweep", "start", str(spec), "--id", "cli-run"])
        self.assertEqual(code, 1)
        self.assertIn("already exists", err.getvalue())
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["sweep", "resume", "missing"])
        self.assertEqual(code, 1)
        self.assertIn("not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
