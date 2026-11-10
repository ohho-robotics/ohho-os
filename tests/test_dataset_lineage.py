"""Lineage walks and pinned-episode delete."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ohho.data.recorder import Episode, Frame
from ohho.data.writer import write_dataset
from ohho.datasets import (
    DatasetSpec,
    EpisodeRecord,
    JsonLineageStore,
    LineageConflict,
    LineageError,
    LineageNotFound,
    SourceQuery,
    build_version,
    episode_from_canonical,
    pin_version,
)
from ohho.datasets.model import CanonicalEpisode, CanonicalFrame


def _episode(
    marker: str, *, task: str = "approach", source_index: int = 0
) -> EpisodeRecord:
    digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()
    return EpisodeRecord(
        episode_id="ep_" + digest[:32],
        content_hash="sha256:" + digest,
        task=task,
        fps="10.00000000",
        frame_count=1,
        source=f"local/{marker}",
        source_index=source_index,
    )


def _write_one(root: Path) -> None:
    episode = Episode(
        task="approach",
        frames=[
            Frame(
                observation_state=[0.0, 0.0, 0.0],
                action=[0.1, 0.0, 0.0],
                timestamp=0.0,
                frame_index=0,
                episode_index=0,
                task_index=0,
                next_done=True,
            )
        ],
        episode_index=0,
        task_index=0,
        fps=10.0,
    )
    write_dataset(
        [episode],
        str(root),
        repo_id="local/one",
        fps=10.0,
        tasks=["approach"],
        robot_type="sim",
    )


def _digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class TestDatasetLineage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.graph = JsonLineageStore(self.store)
        self.ep1 = _episode("one", task="approach")
        self.ep2 = _episode("two", task="retreat", source_index=1)
        self.v1 = "dv_" + "11" * 16
        self.v2 = "dv_" + "22" * 16

    def _chain(self) -> None:
        self.graph.put_episode(self.ep1)
        self.graph.put_episode(self.ep2)
        self.graph.link_version_episodes(
            self.v1, [self.ep1.episode_id, self.ep2.episode_id]
        )
        self.graph.link_version_episodes(self.v2, [self.ep2.episode_id])
        self.graph.link_run("run_a", self.v1)
        self.graph.link_run("run_b", self.v2)
        self.graph.link_model("mv_a", "run_a")
        self.graph.link_model("mv_b", "run_a")
        self.graph.link_model("mv_c", "run_b")

    def _materialise(self, *, pinned: bool) -> str:
        source = self.root / "episodes"
        _write_one(source)
        record = build_version(
            DatasetSpec(
                source_query=SourceQuery((str(source),)),
                dataset_name="demos",
            ),
            store=self.store,
        )
        if pinned:
            record = pin_version(record.version_id, store=self.store)
        return record.version_id

    def test_model_lineage_walks_run_version_and_episodes(self) -> None:
        self._chain()
        chain = self.graph.lineage_from_model("mv_a")
        self.assertEqual(chain.model_version_id, "mv_a")
        self.assertEqual(chain.run_id, "run_a")
        self.assertEqual(chain.dataset_version_id, self.v1)
        self.assertEqual(chain.episode_ids, (self.ep1.episode_id, self.ep2.episode_id))
        other = self.graph.lineage_from_model("mv_b")
        self.assertEqual(other.run_id, "run_a")
        self.assertEqual(other.episode_ids, chain.episode_ids)
        third = self.graph.lineage_from_model("mv_c")
        self.assertEqual(third.run_id, "run_b")
        self.assertEqual(third.dataset_version_id, self.v2)
        self.assertEqual(third.episode_ids, (self.ep2.episode_id,))

    def test_episode_lists_every_model_trained_on_it(self) -> None:
        self._chain()
        trained = self.graph.models_trained_on(self.ep2.episode_id)
        self.assertEqual(
            [
                (row.dataset_version_id, row.run_id, row.model_version_id)
                for row in trained
            ],
            [
                (self.v1, "run_a", "mv_a"),
                (self.v1, "run_a", "mv_b"),
                (self.v2, "run_b", "mv_c"),
            ],
        )
        only = self.graph.models_trained_on(self.ep1.episode_id)
        self.assertEqual([row.model_version_id for row in only], ["mv_a", "mv_b"])
        for row in only:
            self.assertEqual(row.episode_id, self.ep1.episode_id)

    def test_unknown_model_and_episode_are_missing(self) -> None:
        with self.assertRaises(LineageNotFound):
            self.graph.lineage_from_model("mv_missing")
        self.graph.put_episode(self.ep1)
        self.assertEqual(self.graph.models_trained_on(self.ep1.episode_id), ())
        with self.assertRaises(LineageNotFound):
            self.graph.models_trained_on(self.ep2.episode_id)

    def test_run_and_model_edges_conflict_when_retargeted(self) -> None:
        self._chain()
        with self.assertRaises(LineageConflict):
            self.graph.link_run("run_a", self.v2)
        self.assertEqual(
            self.graph.lineage_from_model("mv_a").dataset_version_id, self.v1
        )
        with self.assertRaises(LineageConflict):
            self.graph.link_model("mv_a", "run_b")
        self.assertEqual(self.graph.lineage_from_model("mv_a").run_id, "run_a")
        with self.assertRaises(LineageError):
            self.graph.link_model("mv_new", "run_missing")
        with self.assertRaises(LineageConflict):
            self.graph.link_version_episodes(self.v1, [self.ep2.episode_id])
        self.assertEqual(
            self.graph.episodes_for_version(self.v1),
            (self.ep1.episode_id, self.ep2.episode_id),
        )

    def test_run_row_stores_only_the_version_reference(self) -> None:
        self._chain()
        path = self.graph.path
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "ohho.lineage.v1")
        self.assertEqual(
            set(document["runs"]["run_a"]), {"run_id", "dataset_version_id"}
        )
        self.assertEqual(
            set(document["models"]["mv_a"]), {"model_version_id", "run_id"}
        )
        self.assertEqual(list(self.store.rglob("*.sqlite")), [])
        self.assertEqual(list(self.store.rglob("*.db")), [])
        document["manifest"] = {"run_id": "run_a"}
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(LineageError) as caught:
            self.graph.episode(self.ep1.episode_id)
        self.assertIn("manifest", str(caught.exception))

    def test_repeat_registration_keeps_the_first_source(self) -> None:
        self.graph.put_episode(self.ep1)
        raw = self.graph.path.read_bytes()
        stored = self.graph.put_episode(
            replace(self.ep1, source="local/other", source_index=4)
        )
        self.assertEqual(stored.source, "local/one")
        self.assertEqual(stored.source_index, 0)
        self.assertEqual(self.graph.path.read_bytes(), raw)

    def test_episode_id_is_the_content_hash_prefix(self) -> None:
        episode = CanonicalEpisode(
            task="approach",
            fps="10.00000000",
            frames=(
                CanonicalFrame(
                    observation_state=("0.00000000",),
                    action=("0.10000000",),
                    action_raw=None,
                    timestamp="0.00000000",
                    frame_index=0,
                    next_done=True,
                ),
            ),
        )
        first = episode_from_canonical(episode, source="local/a", source_index=0)
        second = episode_from_canonical(episode, source="local/b", source_index=3)
        self.assertEqual(first.episode_id, second.episode_id)
        self.assertEqual(
            first.episode_id, "ep_" + first.content_hash.removeprefix("sha256:")[:32]
        )
        self.graph.put_episode(first)
        self.assertEqual(self.graph.episode(first.episode_id).frame_count, 1)

    def test_unregistered_episode_is_not_linked(self) -> None:
        with self.assertRaises(LineageError):
            self.graph.link_version_episodes(self.v1, [self.ep1.episode_id])
        self.assertFalse(self.graph.path.is_file())

    def test_pinned_episode_delete_is_blocked_and_soft_deleted(self) -> None:
        version_id = self._materialise(pinned=True)
        self.graph.put_episode(self.ep1)
        self.graph.link_version_episodes(version_id, [self.ep1.episode_id])
        self.graph.link_run("run_pinned", version_id)
        self.graph.link_model("mv_pinned", "run_pinned")
        version_dir = self.store / "versions" / version_id
        before = _digest(version_dir)
        raw = self.graph.path.read_bytes()
        deletion = self.graph.delete_episode(self.ep1.episode_id)
        self.assertTrue(deletion.blocked)
        self.assertTrue(deletion.soft_deleted)
        self.assertFalse(deletion.removed)
        self.assertEqual(deletion.pinned_version_ids, (version_id,))
        self.assertIsNotNone(deletion.warning)
        assert deletion.warning is not None
        self.assertIn("soft-deleted", deletion.warning)
        self.assertIn(self.ep1.episode_id, deletion.warning)
        self.assertIn(version_id, deletion.warning)
        self.assertEqual(_digest(version_dir), before)
        self.assertIn(self.ep1.episode_id, self.graph.episodes_for_version(version_id))
        stored = self.graph.episode(self.ep1.episode_id)
        self.assertTrue(stored.deleted)
        self.assertEqual(stored.delete_warning, deletion.warning)
        self.assertEqual(
            self.graph.lineage_from_model("mv_pinned").episode_ids,
            (self.ep1.episode_id,),
        )
        self.assertEqual(
            self.graph.models_trained_on(self.ep1.episode_id)[0].model_version_id,
            "mv_pinned",
        )
        self.assertNotEqual(self.graph.path.read_bytes(), raw)
        deleted_at = stored.deleted_at
        again_raw = self.graph.path.read_bytes()
        again = self.graph.delete_episode(self.ep1.episode_id)
        self.assertEqual(again.warning, deletion.warning)
        self.assertEqual(self.graph.episode(self.ep1.episode_id).deleted_at, deleted_at)
        self.assertEqual(self.graph.path.read_bytes(), again_raw)

    def test_unpinned_episode_delete_removes_it(self) -> None:
        version_id = self._materialise(pinned=False)
        self.graph.put_episode(self.ep1)
        self.graph.link_version_episodes(version_id, [self.ep1.episode_id])
        self.graph.link_run("run_open", version_id)
        self.graph.link_model("mv_open", "run_open")
        version_json = self.store / "versions" / version_id / "version.json"
        before = version_json.read_bytes()
        deletion = self.graph.delete_episode(self.ep1.episode_id)
        self.assertTrue(deletion.removed)
        self.assertFalse(deletion.blocked)
        self.assertFalse(deletion.soft_deleted)
        self.assertIsNone(deletion.warning)
        self.assertEqual(version_json.read_bytes(), before)
        with self.assertRaises(LineageNotFound):
            self.graph.episode(self.ep1.episode_id)
        self.assertEqual(self.graph.episodes_for_version(version_id), ())
        self.assertEqual(self.graph.lineage_from_model("mv_open").episode_ids, ())
        with self.assertRaises(LineageNotFound):
            self.graph.models_trained_on(self.ep1.episode_id)

    def test_missing_version_record_blocks_delete(self) -> None:
        self.graph.put_episode(self.ep1)
        self.graph.link_version_episodes(self.v1, [self.ep1.episode_id])
        deletion = self.graph.delete_episode(self.ep1.episode_id)
        self.assertTrue(deletion.blocked)
        self.assertTrue(deletion.soft_deleted)
        self.assertIn(self.v1, deletion.warning or "")
        self.assertTrue(self.graph.episode(self.ep1.episode_id).deleted)
        self.assertEqual(
            self.graph.episodes_for_version(self.v1), (self.ep1.episode_id,)
        )

    def test_lineage_schema_names_the_graph(self) -> None:
        import ohho.datasets.lineage as lineage_mod

        path = Path(lineage_mod.__file__).with_name("lineage.schema.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(schema["title"], "LineageGraph")
        self.assertEqual(
            set(schema["required"]),
            {"schema", "episodes", "version_episodes", "runs", "models"},
        )
        self.assertEqual(schema["properties"]["schema"]["const"], "ohho.lineage.v1")
        self.assertEqual(
            set(schema["$defs"]["run"]["required"]),
            {"run_id", "dataset_version_id"},
        )


if __name__ == "__main__":
    unittest.main()
