"""Dataset versions: stable content hash, pin immutability, stored stats."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.data.reader import DatasetReader
from ohho.data.recorder import Episode, Frame
from ohho.data.writer import write_dataset
from ohho.datasets import (
    Augmentation,
    DatasetBuildError,
    DatasetSpec,
    DatasetVersionError,
    DatasetVersionFrozen,
    DatasetVersionNotFound,
    Filters,
    SourceQuery,
    Split,
    build_version,
    edit_version,
    list_versions,
    load_version,
    overwrite_version,
    pin_version,
    resolve_dataset_version,
    resolve_store,
    schema_path,
    version_stats,
)
from ohho.datasets.model import validate_document


def _run(argv: list[str]) -> tuple[int, str, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _frame(
    episode: int, index: int, state: list[float], action: list[float], done: bool
) -> Frame:
    return Frame(
        observation_state=state,
        action=action,
        timestamp=index / 10.0,
        frame_index=index,
        episode_index=episode,
        task_index=0 if episode == 0 else 1,
        next_done=done,
    )


def _write_source(root: Path) -> None:
    episodes = [
        Episode(
            task="approach",
            frames=[
                _frame(0, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], False),
                _frame(0, 1, [1.0, 0.0, 0.0], [0.5, 0.0, 0.0], False),
                _frame(0, 2, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], True),
            ],
            episode_index=0,
            task_index=0,
            fps=10.0,
        ),
        Episode(
            task="retreat",
            frames=[
                _frame(1, 0, [3.0, 1.0, 0.0], [-1.0, 0.0, 0.0], False),
                _frame(1, 1, [4.0, 1.0, 0.0], [0.0, 0.25, 0.0], True),
            ],
            episode_index=1,
            task_index=1,
            fps=10.0,
        ),
    ]
    write_dataset(
        episodes,
        str(root),
        repo_id="local/episodes",
        fps=10.0,
        tasks=["approach", "retreat"],
        robot_type="sim",
    )


def _write_indexed(root: Path, count: int) -> None:
    episodes = []
    for index in range(count):
        episodes.append(
            Episode(
                task=f"task-{index}",
                frames=[
                    Frame(
                        observation_state=[float(index), 0.0, 0.0],
                        action=[float(index) / 10.0, 0.0, 0.0],
                        timestamp=0.0,
                        frame_index=0,
                        episode_index=index,
                        task_index=index,
                        next_done=True,
                    )
                ],
                episode_index=index,
                task_index=index,
                fps=10.0,
            )
        )
    write_dataset(
        episodes,
        str(root),
        repo_id="local/indexed",
        fps=10.0,
        tasks=[f"task-{index}" for index in range(count)],
        robot_type="sim",
    )


def _spec(source: Path, **changes) -> DatasetSpec:
    spec = DatasetSpec(
        source_query=SourceQuery((str(source),)),
        dataset_name="demos",
        licence="unspecified",
    )
    if changes:
        return replace(spec, **changes)
    return spec


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class TestDatasetVersions(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.source = self.root / "episodes"
        self.store = self.root / "store"
        _write_source(self.source)

    def _record_path(self, version_id: str) -> Path:
        return self.store / "versions" / version_id / "version.json"

    def _content_path(self, version_id: str) -> Path:
        return (
            self.store
            / "versions"
            / version_id
            / "dataset"
            / "meta"
            / "ohho_content.json"
        )

    def test_same_spec_twice_has_the_same_content_hash(self) -> None:
        spec = _spec(self.source)
        other = self.root / "other-store"
        first = build_version(spec, store=self.store)
        raw = self._record_path(first.version_id).read_bytes()
        content = self._content_path(first.version_id).read_bytes()
        second = build_version(spec, store=self.store)
        third = build_version(spec, store=other)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.version_id, second.version_id)
        self.assertEqual(first.content_hash, third.content_hash)
        self.assertEqual(first.version_id, third.version_id)
        self.assertEqual(self._record_path(first.version_id).read_bytes(), raw)
        self.assertEqual(self._content_path(first.version_id).read_bytes(), content)
        self.assertEqual(
            self._content_path(first.version_id).read_bytes(),
            (
                other
                / "versions"
                / third.version_id
                / "dataset"
                / "meta"
                / "ohho_content.json"
            ).read_bytes(),
        )
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        self.assertEqual(digest, first.content_hash)
        self.assertEqual(len(list_versions(store=self.store)), 1)

    def test_seed_does_not_change_the_all_split_content_hash(self) -> None:
        first = build_version(_spec(self.source, seed=0), store=self.store)
        second = build_version(_spec(self.source, seed=1), store=self.store)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.version_id, second.version_id)

    def test_filter_and_augmentation_change_the_content_hash(self) -> None:
        base = build_version(_spec(self.source), store=self.store)
        filtered = build_version(
            _spec(self.source, filters=Filters(task_contains="approach")),
            store=self.store,
        )
        scaled = build_version(
            _spec(self.source, augmentation=Augmentation(action_scale=2.0)),
            store=self.store,
        )
        copied = build_version(
            _spec(self.source, augmentation=Augmentation(copies=2)),
            store=self.store,
        )
        self.assertEqual(filtered.stats.episodes, 1)
        self.assertEqual(filtered.stats.frames, 3)
        self.assertNotEqual(filtered.content_hash, base.content_hash)
        self.assertEqual(scaled.stats.action_ranges[0].minimum, -2.0)
        self.assertEqual(scaled.stats.action_ranges[0].maximum, 2.0)
        self.assertNotEqual(scaled.content_hash, base.content_hash)
        self.assertEqual(copied.stats.episodes, 4)
        self.assertEqual(copied.stats.frames, 10)
        self.assertNotEqual(copied.content_hash, base.content_hash)
        with self.assertRaises(DatasetBuildError):
            build_version(
                _spec(self.source, filters=Filters(robot_type="other")),
                store=self.store,
            )

    def test_noise_is_deterministic_for_a_seed(self) -> None:
        spec = _spec(self.source, augmentation=Augmentation(action_noise=0.1), seed=0)
        first = build_version(spec, store=self.store)
        second = build_version(spec, store=self.root / "noise-copy")
        other = build_version(replace(spec, seed=1), store=self.store)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.content_hash, other.content_hash)

    def test_split_is_deterministic(self) -> None:
        source = self.root / "indexed"
        _write_indexed(source, 4)
        spec = DatasetSpec(
            source_query=SourceQuery((str(source),)),
            dataset_name="split-demos",
            split=Split(name="train", train=0.5, val=0.25, test=0.25),
            seed=0,
        )
        first = build_version(spec, store=self.store)
        second = build_version(spec, store=self.store)
        other = build_version(replace(spec, seed=1), store=self.store)
        self.assertEqual(first.stats.episodes, 2)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.content_hash, other.content_hash)

    def test_parent_directory_of_datasets_is_a_source(self) -> None:
        parent = self.root / "bag"
        _write_source(parent / "left")
        _write_source(parent / "right")
        record = build_version(
            DatasetSpec(source_query=SourceQuery((str(parent),)), dataset_name="bag"),
            store=self.store,
        )
        self.assertEqual(record.stats.episodes, 4)
        self.assertEqual(record.stats.frames, 10)

    def test_pinned_version_cannot_be_modified_and_edits_create_a_new_one(self) -> None:
        original = build_version(_spec(self.source), store=self.store)
        pinned = pin_version(original.version_id, store=self.store)
        record_path = self._record_path(pinned.version_id)
        dataset = record_path.parent / "dataset"
        pinned_bytes = record_path.read_bytes()
        dataset_digest = _tree_digest(dataset)
        again = pin_version(pinned.version_id, store=self.store)
        self.assertTrue(again.pinned)
        self.assertEqual(record_path.read_bytes(), pinned_bytes)

        edited = edit_version(pinned.version_id, store=self.store, licence="CC-BY-4.0")
        self.assertNotEqual(edited.version_id, pinned.version_id)
        self.assertEqual(edited.content_hash, pinned.content_hash)
        self.assertEqual(edited.parent_version_id, pinned.version_id)
        self.assertEqual(edited.licence, "CC-BY-4.0")
        self.assertFalse(edited.pinned)

        reloaded = load_version(pinned.version_id, store=self.store)
        self.assertTrue(reloaded.pinned)
        self.assertEqual(reloaded.licence, "unspecified")
        self.assertEqual(record_path.read_bytes(), pinned_bytes)
        self.assertEqual(_tree_digest(dataset), dataset_digest)
        reader = DatasetReader(
            str(resolve_dataset_version(pinned.version_id, store=self.store))
        )
        self.assertEqual(reader.episode_count, 2)

        with self.assertRaises(DatasetVersionFrozen):
            overwrite_version(
                replace(reloaded, created_at="2000-01-01T00:00:00Z"),
                store=self.store,
            )
        self.assertEqual(record_path.read_bytes(), pinned_bytes)

        same = edit_version(pinned.version_id, store=self.store)
        self.assertEqual(same.version_id, pinned.version_id)
        self.assertEqual(record_path.read_bytes(), pinned_bytes)

    def test_unpinned_overwrite_is_refused(self) -> None:
        record = build_version(_spec(self.source), store=self.store)
        with self.assertRaises(DatasetVersionError) as caught:
            overwrite_version(record, store=self.store)
        self.assertNotIsInstance(caught.exception, DatasetVersionFrozen)

    def test_stats_are_stored_with_the_version(self) -> None:
        record = build_version(_spec(self.source), store=self.store)
        document = json.loads(
            self._record_path(record.version_id).read_text(encoding="utf-8")
        )
        self.assertEqual(document["stats"]["episodes"], 2)
        self.assertEqual(document["stats"]["frames"], 5)
        self.assertEqual(
            document["stats"]["action_ranges"][0], {"min": -1.0, "max": 1.0}
        )
        self.assertEqual(document["stats"]["state_ranges"][0], {"min": 0.0, "max": 4.0})
        self.assertEqual(len(document["stats"]["action_ranges"]), 3)
        self.assertEqual(len(document["stats"]["state_ranges"]), 3)
        stored = version_stats(record.version_id, store=self.store)
        self.assertEqual(stored.episodes, 2)
        self.assertEqual(stored.frames, 5)
        self.assertEqual(stored.to_dict(), document["stats"])

        reader = DatasetReader(
            str(resolve_dataset_version(record.version_id, store=self.store))
        )
        self.assertEqual(reader.episode_count, stored.episodes)
        self.assertEqual(reader.frame_count, stored.frames)
        ranges = reader.stats()
        self.assertAlmostEqual(
            ranges["action"]["dim0_min"], stored.action_ranges[0].minimum
        )
        self.assertAlmostEqual(
            ranges["action"]["dim0_max"], stored.action_ranges[0].maximum
        )
        self.assertAlmostEqual(
            ranges["observation.state"]["dim0_min"], stored.state_ranges[0].minimum
        )
        self.assertAlmostEqual(
            ranges["observation.state"]["dim0_max"], stored.state_ranges[0].maximum
        )

        schema = json.loads(schema_path().read_text(encoding="utf-8"))
        self.assertEqual(set(document), set(schema["required"]))
        validate_document(document)
        broken = dict(document)
        broken["extra"] = True
        with self.assertRaises(DatasetVersionError):
            validate_document(broken)

    def test_resolve_accepts_a_version_id_or_a_path(self) -> None:
        record = build_version(_spec(self.source), store=self.store)
        dataset = resolve_dataset_version(record.version_id, store=self.store)
        self.assertTrue((dataset / "meta" / "info.json").is_file())
        self.assertEqual(
            resolve_dataset_version(str(dataset), store=self.store), dataset.resolve()
        )
        version_dir = dataset.parent
        self.assertEqual(
            resolve_dataset_version(str(version_dir), store=self.store),
            dataset.resolve(),
        )
        self.assertEqual(
            resolve_dataset_version(str(self.source), store=self.store),
            self.source.resolve(),
        )
        with self.assertRaises(DatasetVersionNotFound):
            resolve_dataset_version("dv_" + "ab" * 16, store=self.store)

        from ohho.train import finetune

        checkpoint = finetune(
            str(dataset),
            policy="smolvla",
            device="cpu",
            mock=True,
            output_dir=str(self.root / "checkpoints"),
        )
        self.assertTrue((Path(checkpoint) / "checkpoint.json").is_file())

    def test_cli_build_pin_stats(self) -> None:
        code, out, err = _run(
            [
                "dataset",
                "build",
                "--source",
                str(self.source),
                "--store",
                str(self.store),
                "--name",
                "demos",
                "--json",
            ]
        )
        self.assertEqual(code, 0, err)
        built = json.loads(out)
        code, out, err = _run(
            [
                "dataset",
                "build",
                "--source",
                str(self.source),
                "--store",
                str(self.store),
                "--name",
                "demos",
                "--json",
            ]
        )
        self.assertEqual(code, 0, err)
        repeated = json.loads(out)
        self.assertEqual(repeated["content_hash"], built["content_hash"])
        self.assertEqual(repeated["version_id"], built["version_id"])

        code, out, err = _run(
            [
                "dataset",
                "pin",
                built["version_id"],
                "--store",
                str(self.store),
                "--json",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["pinned"])

        code, out, err = _run(
            [
                "dataset",
                "stats",
                built["version_id"],
                "--store",
                str(self.store),
                "--json",
            ]
        )
        self.assertEqual(code, 0, err)
        printed = json.loads(out)
        stored = json.loads(
            self._record_path(built["version_id"]).read_text(encoding="utf-8")
        )
        self.assertEqual(printed["episodes"], stored["stats"]["episodes"])
        self.assertEqual(printed["frames"], stored["stats"]["frames"])
        self.assertEqual(printed["action_ranges"], stored["stats"]["action_ranges"])
        self.assertEqual(printed["state_ranges"], stored["stats"]["state_ranges"])
        self.assertEqual(printed["episodes"], built["stats"]["episodes"])

        code, _out, err = _run(
            [
                "dataset",
                "build",
                "--source",
                str(self.root / "missing"),
                "--store",
                str(self.store),
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        code, _out, err = _run(
            ["dataset", "stats", "dv_" + "ab" * 16, "--store", str(self.store)]
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_dataset_command_requires_a_subcommand(self) -> None:
        import io
        from contextlib import redirect_stderr

        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                main(["dataset"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("dataset_command", err.getvalue())

    def test_default_store_root(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OHHO_DATASETS_ROOT", None)
            self.assertEqual(
                resolve_store(None), (Path.home() / ".ohho" / "datasets").resolve()
            )
            chosen = self.root / "from-env"
            os.environ["OHHO_DATASETS_ROOT"] = str(chosen)
            self.assertEqual(resolve_store(None), chosen.resolve())
            self.assertEqual(resolve_store(self.store), self.store.resolve())


if __name__ == "__main__":
    unittest.main()
