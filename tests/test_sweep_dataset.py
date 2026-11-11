"""Dataset version as a sweep dimension, worker cache, and grouping."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.data.recorder import Episode, Frame
from ohho.data.writer import write_dataset
from ohho.datasets import (
    DatasetSpec,
    SourceQuery,
    build_version,
    resolve_dataset_version,
)
from ohho.train.sweep import (
    SCHEMA_ID,
    SweepError,
    expand,
    format_plan,
    load_schema,
    plan_as_dict,
)
from ohho.train.sweep_dataset import (
    GROUPS_SCHEMA_ID,
    DatasetVersionCache,
    format_dataset_groups,
    group_by_dataset_version,
    groups_as_dict,
    load_groups_schema,
    resolve_trial_dataset,
    resolve_trial_datasets,
    worker_dataset_cache,
)
from ohho.train.sweep_run import resume_sweep, start_sweep
from ohho.train.sweep_sampler import SweepSampler

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "ohho" / "train" / "schemas" / "examples" / "dataset_3x2.json"
_VERSIONS = (
    "dv_" + "a" * 32,
    "dv_" + "b" * 32,
    "dv_" + "c" * 32,
)


def _spec() -> dict:
    return {
        "schema": SCHEMA_ID,
        "strategy": "grid",
        "seed": 1,
        "recipe": {"task": "push"},
        "dimensions": [
            {"name": "dataset", "type": "dataset", "values": list(_VERSIONS)},
            {
                "name": "model",
                "type": "categorical",
                "values": ["act", "diffusion"],
            },
        ],
    }


def _run(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestDatasetDimension(unittest.TestCase):
    def test_schema_names_the_dataset_axis(self) -> None:
        schema = load_schema()
        self.assertIn(
            "dataset", schema["$defs"]["dimension"]["properties"]["type"]["enum"]
        )
        self.assertIn("dataset version", schema["description"])
        groups = load_groups_schema()
        self.assertEqual(groups["title"], GROUPS_SCHEMA_ID)
        packaged = (
            files("ohho.train")
            .joinpath("schemas", "sweep.dataset_groups.v1.json")
            .read_text(encoding="utf-8")
        )
        on_disk = (
            _ROOT / "ohho" / "train" / "schemas" / "sweep.dataset_groups.v1.json"
        ).read_text(encoding="utf-8")
        self.assertEqual(packaged, on_disk)

    def test_three_versions_by_two_models_is_six_trials(self) -> None:
        plan = expand(_EXAMPLE)
        self.assertEqual(len(plan.trials), 6)
        pairs = [
            (trial.dataset_version_id, trial.overrides["model"])
            for trial in plan.trials
        ]
        self.assertEqual(
            pairs,
            [
                (_VERSIONS[0], "act"),
                (_VERSIONS[0], "diffusion"),
                (_VERSIONS[1], "act"),
                (_VERSIONS[1], "diffusion"),
                (_VERSIONS[2], "act"),
                (_VERSIONS[2], "diffusion"),
            ],
        )
        for trial, (version_id, model) in zip(plan.trials, pairs):
            self.assertEqual(trial.overrides["dataset"], version_id)
            self.assertEqual(trial.recipe["dataset"], version_id)
            self.assertEqual(trial.recipe["model"], model)
            self.assertEqual(trial.recipe["task"], "push")
            self.assertIn(f'dataset="{version_id}"', trial.sets)
        text = format_plan(plan)
        self.assertIn("trials: 6", text)
        for version_id in _VERSIONS:
            self.assertEqual(text.count(f"dataset_version_id: {version_id}"), 2)
        payload = plan_as_dict(plan)
        self.assertEqual(
            [row["dataset_version_id"] for row in payload["trial_recipes"]],
            [pair[0] for pair in pairs],
        )
        self.assertEqual(expand(_spec()).trials[0].overrides, plan.trials[0].overrides)

    def test_custom_path_still_records_the_version_id(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "versions",
                        "path": "data.version_id",
                        "type": "dataset",
                        "values": list(_VERSIONS[:2]),
                    }
                ],
            }
        )
        self.assertEqual(
            [trial.dataset_version_id for trial in plan.trials],
            list(_VERSIONS[:2]),
        )
        self.assertEqual(plan.trials[0].recipe["data"]["version_id"], _VERSIONS[0])
        self.assertEqual(plan.trials[0].overrides["data.version_id"], _VERSIONS[0])

    def test_categorical_values_are_not_a_dataset_axis(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {"name": "tag", "type": "categorical", "values": [_VERSIONS[0]]}
                ],
            }
        )
        self.assertIsNone(plan.trials[0].dataset_version_id)
        self.assertNotIn("dataset_version_id", plan_as_dict(plan)["trial_recipes"][0])

    def test_conditional_dataset_dimension(self) -> None:
        plan = expand(
            {
                "schema": SCHEMA_ID,
                "dimensions": [
                    {
                        "name": "model",
                        "type": "categorical",
                        "values": ["act", "vla"],
                    },
                    {
                        "name": "dataset",
                        "type": "dataset",
                        "values": list(_VERSIONS[:2]),
                        "when": {"path": "model", "eq": "vla"},
                    },
                ],
            }
        )
        self.assertEqual(
            [trial.dataset_version_id for trial in plan.trials],
            [None, _VERSIONS[0], _VERSIONS[1]],
        )
        self.assertNotIn("dataset", plan.trials[0].overrides)
        groups = group_by_dataset_version(plan.trials)
        self.assertEqual(
            [group.dataset_version_id for group in groups],
            [None, _VERSIONS[0], _VERSIONS[1]],
        )
        self.assertEqual(len(groups[0].trials), 1)

    def test_rejects_a_bad_version_id_and_a_second_axis(self) -> None:
        bad_values = [
            ["not-a-version"],
            ["dv_" + "A" * 32],
            [1],
            [_VERSIONS[0], _VERSIONS[0]],
        ]
        for values in bad_values:
            doc = _spec()
            doc["dimensions"] = [
                {"name": "dataset", "type": "dataset", "values": values}
            ]
            with self.assertRaises(SweepError):
                expand(doc)
        missing = _spec()
        missing["dimensions"] = [{"name": "dataset", "type": "dataset"}]
        with self.assertRaises(SweepError):
            expand(missing)
        doubled = _spec()
        doubled["dimensions"] = [
            {"name": "left", "type": "dataset", "values": [_VERSIONS[0]]},
            {
                "name": "right",
                "path": "other",
                "type": "dataset",
                "values": [_VERSIONS[1]],
            },
        ]
        with self.assertRaises(SweepError) as caught:
            expand(doubled)
        self.assertIn("at most one dataset dimension", str(caught.exception))

    def test_random_draws_stay_inside_the_version_list(self) -> None:
        doc = _spec()
        doc["strategy"] = "random"
        doc["seed"] = 7
        doc["budget"] = {"max_trials": 8}
        first = expand(doc)
        second = expand(doc)
        self.assertEqual(len(first.trials), 8)
        self.assertEqual(
            [trial.dataset_version_id for trial in first.trials],
            [trial.dataset_version_id for trial in second.trials],
        )
        self.assertTrue(
            all(trial.dataset_version_id in _VERSIONS for trial in first.trials)
        )

    def test_sampler_records_the_id_on_state_and_leaderboard(self) -> None:
        doc = _spec()
        doc["objectives"] = [{"name": "loss", "direction": "minimize"}]
        doc["strategy"] = "tpe"
        doc["budget"] = {"max_trials": 4}
        doc["sampler"] = {"startup_trials": 2, "candidates": 4}
        sampler = SweepSampler.from_sweep(doc)
        trial = sampler.ask()[0]
        self.assertIn(trial.dataset_version_id, _VERSIONS)
        sampler.tell(trial.index, {"loss": 0.2})
        board = sampler.leaderboard()
        self.assertEqual(
            board["trials"][0]["dataset_version_id"], trial.dataset_version_id
        )
        state = sampler.dump_state()
        self.assertEqual(
            state["trials"][0]["dataset_version_id"], trial.dataset_version_id
        )
        restored = SweepSampler.load_state(json.loads(json.dumps(state)))
        self.assertEqual(
            restored.ask()[0].dataset_version_id,
            sampler.ask()[0].dataset_version_id,
        )


class TestDatasetCache(unittest.TestCase):
    def test_second_trial_does_not_fetch_or_copy(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "src"

            def fetcher(version_id: str) -> Path:
                calls.append(version_id)
                source = sources / version_id
                source.mkdir(parents=True, exist_ok=True)
                (source / "marker.txt").write_text(version_id, encoding="utf-8")
                return source

            cache = DatasetVersionCache(fetcher, root / "cache")
            trials = expand(_spec()).trials
            first = resolve_trial_datasets(trials, cache)
            self.assertEqual(calls, list(_VERSIONS))
            self.assertEqual(cache.fetch_count, 3)
            self.assertEqual(cache.copy_count, 3)
            self.assertEqual(first[0], first[1])
            self.assertNotEqual(first[0], first[2])
            kept = first[0] / "worker.txt"
            kept.write_text("stay", encoding="utf-8")
            second = resolve_trial_datasets(trials, cache)
            self.assertEqual(calls, list(_VERSIONS))
            self.assertEqual(cache.fetch_count, 3)
            self.assertEqual(cache.copy_count, 3)
            self.assertEqual(second, first)
            self.assertEqual(kept.read_text(encoding="utf-8"), "stay")
            self.assertEqual(
                (first[0] / "marker.txt").read_text(encoding="utf-8"), _VERSIONS[0]
            )

            again = DatasetVersionCache(fetcher, root / "cache")
            hit = again.resolve(_VERSIONS[0])
            self.assertEqual(hit, first[0])
            self.assertEqual(again.fetch_count, 0)
            self.assertEqual(again.copy_count, 0)
            self.assertEqual(calls, list(_VERSIONS))
            self.assertEqual(kept.read_text(encoding="utf-8"), "stay")

            def refuse(version_id: str) -> Path:
                raise AssertionError(version_id)

            cold = DatasetVersionCache(refuse, root / "other")
            with self.assertRaises(SweepError):
                cold.resolve("not-a-version")
            bare = expand(
                {
                    "schema": SCHEMA_ID,
                    "dimensions": [
                        {"name": "model", "type": "categorical", "values": ["act"]}
                    ],
                }
            ).trials[0]
            with self.assertRaises(SweepError) as caught:
                resolve_trial_dataset(bare, cold)
            self.assertIn("no dataset version", str(caught.exception))

    def test_default_fetcher_copies_a_stored_version_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "episodes"
            frame = Frame(
                observation_state=[0.0, 0.0, 0.0],
                action=[0.1, 0.0, 0.0],
                timestamp=0.0,
                frame_index=0,
                episode_index=0,
                task_index=0,
                next_done=True,
            )
            write_dataset(
                [Episode(task="hold", frames=[frame], episode_index=0, fps=10.0)],
                str(source),
                repo_id="local/cache",
                fps=10.0,
                tasks=["hold"],
                robot_type="sim",
            )
            store = root / "store"
            record = build_version(
                DatasetSpec(
                    source_query=SourceQuery((str(source),)),
                    dataset_name="hold",
                    licence="unspecified",
                ),
                store=store,
            )
            cache = worker_dataset_cache(root / "cache", store=store)
            with patch(
                "ohho.datasets.resolve_dataset_version",
                wraps=resolve_dataset_version,
            ) as fetched:
                first = cache.resolve(record.version_id)
                marker = first / "worker.txt"
                marker.write_text("stay", encoding="utf-8")
                second = cache.resolve(record.version_id)
            self.assertEqual(fetched.call_count, 1)
            self.assertEqual(cache.copy_count, 1)
            self.assertEqual(first, second)
            self.assertTrue((first / "meta" / "info.json").is_file())
            self.assertEqual(marker.read_text(encoding="utf-8"), "stay")
            stored = resolve_dataset_version(record.version_id, store=store)
            self.assertNotEqual(first.resolve(), stored.resolve())


class TestDatasetGrouping(unittest.TestCase):
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

    def test_run_roundtrip_and_group_cli(self) -> None:
        doc = _spec()
        doc["objectives"] = [{"name": "loss", "direction": "minimize"}]
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(doc), encoding="utf-8")
        plan = expand(doc)
        run = start_sweep(spec, sweep_id="ds-grid", root=self.root)
        first = run.ask(2)
        stopped = resume_sweep("ds-grid", root=self.root)
        rest = stopped.ask(4)
        got = (*first, *rest)
        self.assertEqual(
            [trial.dataset_version_id for trial in got],
            [trial.dataset_version_id for trial in plan.trials],
        )
        for trial in got:
            stopped.tell(trial.index, {"loss": float(trial.index)})
        saved = json.loads(
            (self.root / "ds-grid" / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [row["dataset_version_id"] for row in saved["trials"]],
            list(trial.dataset_version_id for trial in plan.trials),
        )

        groups = group_by_dataset_version(stopped.trial_table())
        self.assertEqual(
            [group.dataset_version_id for group in groups],
            list(_VERSIONS),
        )
        self.assertTrue(all(len(group.trials) == 2 for group in groups))
        self.assertEqual(
            [row["overrides"]["model"] for row in groups[0].trials],
            ["act", "diffusion"],
        )
        self.assertEqual(groups[0].trials[0]["values"], {"loss": 0.0})
        payload = groups_as_dict("ds-grid", groups)
        self.assertEqual(payload["schema"], GROUPS_SCHEMA_ID)
        self.assertEqual(len(payload["groups"]), 3)
        text = format_dataset_groups("ds-grid", groups)
        self.assertIn("groups: 3", text)
        self.assertEqual(text.count("trials: 2"), 3)

        code, out, err = _run(["sweep", "group", "ds-grid"])
        self.assertEqual(code, 0, err)
        self.assertIn(f"dataset_version_id: {_VERSIONS[0]}", out)
        self.assertIn('values: {"loss": 0.0}', out)
        code, out, err = _run(["sweep", "group", "ds-grid", "--json"])
        self.assertEqual(code, 0, err)
        body = json.loads(out)
        self.assertEqual(body["schema"], GROUPS_SCHEMA_ID)
        self.assertEqual(
            [group["dataset_version_id"] for group in body["groups"]],
            list(_VERSIONS),
        )
        self.assertEqual(len(body["groups"][2]["trials"]), 2)

        code, out, err = _run(["sweep", "plan", str(_EXAMPLE)])
        self.assertEqual(code, 0, err)
        self.assertIn("trials: 6", out)
        self.assertIn(f"dataset_version_id: {_VERSIONS[2]}", out)

    def test_missing_version_id_does_not_reload(self) -> None:
        doc = _spec()
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(doc), encoding="utf-8")
        run = start_sweep(spec, sweep_id="kept", root=self.root)
        run.ask(1)
        path = self.root / "kept" / "run.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        del saved["trials"][0]["dataset_version_id"]
        del saved["sampler"]["trials"][0]["dataset_version_id"]
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises(SweepError) as caught:
            resume_sweep("kept", root=self.root)
        self.assertIn("dataset version", str(caught.exception))

        saved = json.loads(path.read_text(encoding="utf-8"))
        # The failed resume did not rewrite the file. Put a bad id on the row.
        saved["trials"][0]["dataset_version_id"] = "nope"
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises(SweepError):
            resume_sweep("kept", root=self.root)


if __name__ == "__main__":
    unittest.main()
