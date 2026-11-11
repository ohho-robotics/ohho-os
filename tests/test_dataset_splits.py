"""Seeded splits stay deterministic, keep each episode in one split, and hash."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ohho.cli import main
from ohho.data.reader import DatasetReader
from ohho.data.recorder import Episode, Frame
from ohho.data.writer import write_dataset
from ohho.datasets import (
    SPLIT_STRATEGIES,
    Augmentation,
    CameraToggle,
    DatasetBuildError,
    DatasetSpec,
    DatasetVersionError,
    SourceQuery,
    Split,
    build_version,
    load_version,
)
from ohho.datasets.model import validate_document
from ohho.datasets.split import SplitEpisode, assign_splits, prepare_split


def _run(argv: list[str]) -> tuple[int, str, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _episode(index: int, states: list[float], task: str) -> Episode:
    frames = []
    for frame_index, state in enumerate(states):
        frames.append(
            Frame(
                observation_state=[state, 0.0, 0.0],
                action=[state / 10.0, 0.0, 0.0],
                timestamp=frame_index / 10.0,
                frame_index=frame_index,
                episode_index=index,
                task_index=index,
                next_done=frame_index == len(states) - 1,
            )
        )
    return Episode(
        task=task,
        frames=frames,
        episode_index=index,
        task_index=index,
        fps=10.0,
    )


def _write(root: Path, episodes: list[Episode], tasks: list[str]) -> None:
    write_dataset(
        episodes,
        str(root),
        repo_id="local/splits",
        fps=10.0,
        tasks=tasks,
        robot_type="sim",
    )


def _annotate(root: Path, extra: dict[int, dict]) -> None:
    path = root / "meta" / "episodes.jsonl"
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.update(extra.get(int(row["episode_index"]), {}))
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signatures(path: Path) -> list[tuple[float, ...]]:
    reader = DatasetReader(str(path))
    found = []
    for _meta, frames in reader.iter_episodes():
        found.append(tuple(frame.observation_state[0] for frame in frames))
    return found


def _places(assignment) -> dict[str, str]:
    found = {}
    for name in ("train", "val", "test"):
        for episode in assignment.episodes(name):
            found[episode.episode_id] = name
    return found


def _assert_exclusive(test: unittest.TestCase, episodes, assignment) -> None:
    frames_by_part: dict[str, list[str]] = {}
    ids: list[str] = []
    frames: list[str] = []
    for name in ("train", "val", "test"):
        part_frames: list[str] = []
        for episode in assignment.episodes(name):
            ids.append(episode.episode_id)
            part_frames.extend(episode.frame_ids)
        frames_by_part[name] = part_frames
        frames.extend(part_frames)
    test.assertEqual(sorted(ids), sorted(episode.episode_id for episode in episodes))
    test.assertEqual(len(ids), len(set(ids)))
    expected = [frame for episode in episodes for frame in episode.frame_ids]
    test.assertEqual(sorted(frames), sorted(expected))
    test.assertEqual(len(frames), len(set(frames)))
    for episode in episodes:
        homes = [
            name
            for name, part_frames in frames_by_part.items()
            if set(episode.frame_ids) <= set(part_frames)
        ]
        test.assertEqual(len(homes), 1)


def _reference_shuffle(items, seed: int):
    out = list(items)
    rng = random.Random(seed)
    for index in range(len(out) - 1, 0, -1):
        swap = int(rng.random() * (index + 1))
        out[index], out[swap] = out[swap], out[index]
    return out


class TestAssignSplits(unittest.TestCase):
    def _rows(self) -> list[SplitEpisode]:
        return [
            SplitEpisode(f"e{index}", task=f"task-{index}", frame_ids=(f"e{index}f",))
            for index in range(6)
        ]

    def test_same_seed_is_identical_and_episode_order_matches_the_shuffle(self) -> None:
        episodes = self._rows()
        split = Split(name="train", train=0.5, val=0.25, test=0.25)
        first = assign_splits(episodes, split, 7)
        second = assign_splits(list(episodes), split, 7)
        self.assertEqual(first, second)
        self.assertEqual(
            first.train + first.val + first.test,
            tuple(_reference_shuffle(episodes, 7)),
        )
        self.assertEqual(len(episodes), 6)
        _assert_exclusive(self, episodes, first)

    def test_a_different_seed_can_change_the_episode_split(self) -> None:
        episodes = self._rows()
        split = Split(name="train", train=0.5, val=0.5, test=0.0)
        seeds = {
            tuple(
                episode.episode_id
                for episode in assign_splits(episodes, split, seed).train
            )
            for seed in range(8)
        }
        self.assertGreater(len(seeds), 1)

    def test_frames_stay_with_their_episode_for_every_strategy(self) -> None:
        episodes = [
            SplitEpisode(
                "long",
                task="pick",
                session="morning",
                day="2024-06-02",
                scene="lab",
                frame_ids=("long-0", "long-1", "long-2"),
            ),
            SplitEpisode(
                "short",
                task="place",
                session="evening",
                day="2024-06-03",
                scene="kitchen",
                frame_ids=("short-0",),
            ),
            SplitEpisode(
                "third",
                task="pick",
                session="morning",
                day="2024-06-02",
                scene="hall",
                frame_ids=("third-0", "third-1"),
            ),
        ]
        split = Split(name="train", train=1.0, val=1.0, test=1.0)
        for strategy in SPLIT_STRATEGIES:
            for seed in (0, 1, 99):
                assignment = assign_splits(
                    episodes, replace(split, strategy=strategy), seed
                )
                _assert_exclusive(self, episodes, assignment)
                self.assertEqual(
                    assignment,
                    assign_splits(episodes, replace(split, strategy=strategy), seed),
                )

    def test_session_keeps_a_session_together_across_days(self) -> None:
        episodes = [
            SplitEpisode(
                "a", session="run-1", day="2024-06-02", frame_ids=("a0", "a1")
            ),
            SplitEpisode("b", session="run-1", day="2024-06-03", frame_ids=("b0",)),
            SplitEpisode("c", session="run-2", day="2024-06-02", frame_ids=("c0",)),
        ]
        split = Split(name="train", strategy="session", train=1.0, val=1.0, test=0.0)
        for seed in (0, 1, 5):
            places = _places(assign_splits(episodes, split, seed))
            self.assertEqual(places["a"], places["b"])
            self.assertNotEqual(places["a"], places["c"])
            _assert_exclusive(self, episodes, assign_splits(episodes, split, seed))

    def test_day_groups_unix_timestamps_and_leaves_short_timestamps_undated(
        self,
    ) -> None:
        june_2 = datetime(2024, 6, 2, 15, tzinfo=timezone.utc).timestamp()
        episodes = [
            SplitEpisode("timed", timestamp=june_2, frame_ids=("t",)),
            SplitEpisode("labeled", day="2024-06-02", frame_ids=("l",)),
            SplitEpisode("early", timestamp=10.0, frame_ids=("e0",)),
            SplitEpisode("also-early", timestamp=20.0, frame_ids=("e1",)),
        ]
        split = Split(name="train", strategy="session", train=1.0, val=1.0, test=0.0)
        places = _places(assign_splits(episodes, split, 0))
        self.assertEqual(places["timed"], places["labeled"])
        self.assertEqual(places["early"], places["also-early"])
        self.assertNotEqual(places["timed"], places["early"])

    def test_task_group_stays_together_and_keeps_input_order(self) -> None:
        episodes = [
            SplitEpisode("a1", task="pick", frame_ids=("a1",)),
            SplitEpisode("b", task="place", frame_ids=("b",)),
            SplitEpisode("a2", task="pick", frame_ids=("a2",)),
        ]
        split = Split(name="train", strategy="task", train=1.0, val=1.0, test=0.0)
        for seed in range(4):
            assignment = assign_splits(episodes, split, seed)
            places = _places(assignment)
            self.assertEqual(places["a1"], places["a2"])
            self.assertNotEqual(places["a1"], places["b"])
            for name in ("train", "val", "test"):
                ids = [episode.episode_id for episode in assignment.episodes(name)]
                if "a1" in ids:
                    self.assertLess(ids.index("a1"), ids.index("a2"))

    def test_held_out_scenes_land_only_in_test(self) -> None:
        episodes = [
            SplitEpisode("l1", scene="lab", frame_ids=("l1a", "l1b")),
            SplitEpisode("l2", scene="lab", frame_ids=("l2",)),
            SplitEpisode("k", scene="kitchen", frame_ids=("k0", "k1")),
            SplitEpisode("h", scene="hall", frame_ids=("h",)),
        ]
        split = Split(
            name="train",
            strategy="scene",
            train=1.0,
            val=1.0,
            test=0.0,
            held_out_scenes=("kitchen",),
        )
        for seed in (0, 3, 8):
            assignment = assign_splits(episodes, split, seed)
            places = _places(assignment)
            self.assertEqual(places["k"], "test")
            self.assertEqual(places["l1"], places["l2"])
            self.assertNotEqual(places["l1"], "test")
            self.assertNotEqual(places["h"], "test")
            _assert_exclusive(self, episodes, assignment)

    def test_prepare_sorts_held_out_scenes_and_rejects_a_bad_strategy(self) -> None:
        prepared = prepare_split(
            Split(strategy="scene", held_out_scenes=(" kitchen ", "lab", "kitchen"))
        )
        self.assertEqual(prepared.held_out_scenes, ("kitchen", "lab"))
        with self.assertRaises(DatasetBuildError):
            prepare_split(Split(strategy="episode", held_out_scenes=("lab",)))
        with self.assertRaises(DatasetBuildError):
            prepare_split(Split(strategy="nope"))
        with self.assertRaises(DatasetBuildError):
            assign_splits(
                [SplitEpisode("a", frame_ids=("a",))],
                Split(strategy="scene", held_out_scenes=("missing",)),
                0,
            )

    def test_duplicate_episode_id_is_rejected(self) -> None:
        with self.assertRaises(DatasetBuildError):
            assign_splits(
                [
                    SplitEpisode("a", frame_ids=("a",)),
                    SplitEpisode("a", frame_ids=("b",)),
                ],
                Split(name="train", train=1.0, val=0.0, test=0.0),
                0,
            )


class TestSplitOnVersions(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"

    def _source(self, episodes: list[Episode], tasks: list[str]) -> Path:
        path = self.root / f"source-{len(list(self.root.iterdir()))}"
        _write(path, episodes, tasks)
        return path

    def _spec(self, source: Path, **changes) -> DatasetSpec:
        spec = DatasetSpec(
            source_query=SourceQuery((str(source),)),
            dataset_name="splits",
            licence="unspecified",
        )
        if changes:
            return replace(spec, **changes)
        return spec

    def _payload(self, version_id: str) -> dict:
        path = (
            self.store
            / "versions"
            / version_id
            / "dataset"
            / "meta"
            / "ohho_content.json"
        )
        raw = path.read_bytes()
        record = load_version(version_id, store=self.store)
        self.assertEqual(
            record.content_hash, "sha256:" + hashlib.sha256(raw).hexdigest()
        )
        return json.loads(raw.decode("utf-8"))

    def test_built_splits_do_not_share_frames(self) -> None:
        episodes = [
            _episode(0, [0.0, 1.0, 2.0], "a"),
            _episode(1, [10.0], "b"),
            _episode(2, [20.0, 21.0], "c"),
            _episode(3, [30.0], "d"),
        ]
        source = self._source(episodes, ["a", "b", "c", "d"])
        expected = [(0.0, 1.0, 2.0), (10.0,), (20.0, 21.0), (30.0,)]
        split = Split(name="train", strategy="episode", train=0.5, val=0.25, test=0.25)
        found = []
        for name in ("train", "val", "test"):
            record = build_version(
                self._spec(source, split=replace(split, name=name)),
                store=self.store,
            )
            found.extend(
                _signatures(self.store / "versions" / record.version_id / "dataset")
            )
        self.assertCountEqual(found, expected)

    def test_session_metadata_is_read_from_episodes_jsonl(self) -> None:
        episodes = [_episode(index, [float(index)], f"t{index}") for index in range(4)]
        source = self._source(episodes, [f"t{index}" for index in range(4)])
        _annotate(
            source,
            {
                0: {"session": "morning"},
                1: {"session": "morning"},
                2: {"session": "evening"},
                3: {"session": "evening"},
            },
        )
        split = Split(name="train", strategy="session", train=1.0, val=1.0, test=0.0)
        groups = []
        for name in ("train", "val"):
            record = build_version(
                self._spec(source, split=replace(split, name=name), dataset_name=name),
                store=self.store,
            )
            groups.append(
                frozenset(
                    sig[0]
                    for sig in _signatures(
                        self.store / "versions" / record.version_id / "dataset"
                    )
                )
            )
        self.assertCountEqual(groups, [frozenset({0.0, 1.0}), frozenset({2.0, 3.0})])

    def test_held_out_scene_is_materialised_only_as_test(self) -> None:
        episodes = [
            _episode(0, [0.0, 1.0], "lab"),
            _episode(1, [2.0], "lab-b"),
            _episode(2, [3.0, 4.0], "kitchen"),
        ]
        source = self._source(episodes, ["lab", "lab-b", "kitchen"])
        _annotate(
            source, {0: {"scene": "lab"}, 1: {"scene": "lab"}, 2: {"scene": "kitchen"}}
        )
        split = Split(
            name="train",
            strategy="scene",
            train=1.0,
            val=0.0,
            test=0.0,
            held_out_scenes=("kitchen",),
        )
        train = build_version(self._spec(source, split=split), store=self.store)
        test = build_version(
            self._spec(source, split=replace(split, name="test"), dataset_name="held"),
            store=self.store,
        )
        self.assertCountEqual(
            _signatures(self.store / "versions" / train.version_id / "dataset"),
            [(0.0, 1.0), (2.0,)],
        )
        self.assertEqual(
            _signatures(self.store / "versions" / test.version_id / "dataset"),
            [(3.0, 4.0)],
        )
        self.assertEqual(train.split.held_out_scenes, ("kitchen",))
        with self.assertRaises(DatasetBuildError):
            build_version(
                self._spec(
                    source,
                    split=replace(split, held_out_scenes=("attic",)),
                    dataset_name="missing-scene",
                ),
                store=self.store,
            )

    def test_split_and_augmentation_config_change_the_content_hash(self) -> None:
        source = self._source(
            [_episode(0, [0.0, 1.0], "approach"), _episode(1, [2.0], "retreat")],
            ["approach", "retreat"],
        )
        _annotate(source, {0: {"scene": "lab"}, 1: {"scene": "kitchen"}})
        base = build_version(self._spec(source), store=self.store)
        strategy = build_version(
            self._spec(
                source,
                split=Split(name="all", strategy="task"),
                dataset_name="by-task",
            ),
            store=self.store,
        )
        blurred = build_version(
            self._spec(
                source, augmentation=Augmentation(blur=1.0), dataset_name="blur"
            ),
            store=self.store,
        )
        again = build_version(
            self._spec(
                source, augmentation=Augmentation(blur=1.0), dataset_name="blur"
            ),
            store=self.root / "other-store",
        )
        seeded = build_version(
            self._spec(
                source,
                augmentation=Augmentation(blur=1.0),
                seed=1,
                dataset_name="blur-seed",
            ),
            store=self.store,
        )
        held = build_version(
            self._spec(
                source,
                split=Split(name="all", strategy="scene", held_out_scenes=("kitchen",)),
                dataset_name="held",
            ),
            store=self.store,
        )
        base_payload = self._payload(base.version_id)
        strategy_payload = self._payload(strategy.version_id)
        blurred_payload = self._payload(blurred.version_id)
        self.assertEqual(base_payload["episodes"], strategy_payload["episodes"])
        self.assertNotEqual(base_payload["split"], strategy_payload["split"])
        self.assertNotEqual(base.content_hash, strategy.content_hash)
        self.assertEqual(base_payload["episodes"], blurred_payload["episodes"])
        self.assertNotEqual(
            base_payload["augmentation"], blurred_payload["augmentation"]
        )
        self.assertNotEqual(base.content_hash, blurred.content_hash)
        self.assertEqual(blurred.content_hash, again.content_hash)
        self.assertEqual(blurred.version_id, again.version_id)
        self.assertEqual(blurred.content_hash, seeded.content_hash)
        self.assertNotEqual(blurred.version_id, seeded.version_id)
        self.assertNotEqual(base.content_hash, held.content_hash)
        self.assertEqual(
            self._payload(base.version_id)["episodes"],
            self._payload(held.version_id)["episodes"],
        )
        self.assertIn("split", base_payload)
        self.assertIn("augmentation", base_payload)

        document = json.loads(
            (self.store / "versions" / blurred.version_id / "version.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(document["augmentation"]["blur"], 1.0)
        self.assertEqual(document["split"]["strategy"], "episode")
        validate_document(document)
        broken = json.loads(json.dumps(document))
        broken["split"]["strategy"] = "random"
        with self.assertRaises(DatasetVersionError):
            validate_document(broken)
        broken = json.loads(json.dumps(document))
        del broken["augmentation"]["blur"]
        with self.assertRaises(DatasetVersionError):
            validate_document(broken)

    def test_cli_stores_strategy_blur_and_camera_toggle(self) -> None:
        source = self._source([_episode(0, [0.0], "hold")], ["hold"])
        code, out, err = _run(
            [
                "dataset",
                "build",
                "--source",
                str(source),
                "--store",
                str(self.store),
                "--split-strategy",
                "task",
                "--blur",
                "0.5",
                "--camera-off",
                "wrist",
                "--json",
            ]
        )
        self.assertEqual(code, 0, err)
        built = json.loads(out)
        record = load_version(built["version_id"], store=self.store)
        self.assertEqual(record.split.strategy, "task")
        self.assertEqual(record.augmentation.blur, 0.5)
        self.assertEqual(record.augmentation.cameras, (CameraToggle("wrist", False),))
        code, _out, err = _run(
            [
                "dataset",
                "build",
                "--source",
                str(source),
                "--store",
                str(self.store),
                "--crop-height",
                "4",
                "--name",
                "partial-crop",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("crop needs both", err)


if __name__ == "__main__":
    unittest.main()
