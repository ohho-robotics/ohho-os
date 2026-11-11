"""Hugging Face dataset import: pinned revision, licence, and lineage."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.data.reader import DatasetReader
from ohho.data.recorder import Episode, Frame
from ohho.data.writer import write_dataset
from ohho.datasets import (
    DatasetSpec,
    JsonLineageStore,
    SourceQuery,
    build_version,
    import_hf_dataset,
    is_non_commercial,
    load_version,
    resolve_dataset_version,
    resolved_licence,
)
from ohho.datasets.hf_import import (
    DatasetImportError,
    HubSnapshot,
    HuggingFaceDatasetHub,
)

SHA = "a" * 40
SHA_B = "b" * 40


def _run(argv: list[str]) -> tuple[int, str, str]:
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
                _frame(0, 1, [1.0, 0.0, 0.0], [0.5, 0.0, 0.0], True),
            ],
            episode_index=0,
            task_index=0,
            fps=10.0,
        ),
        Episode(
            task="retreat",
            frames=[
                _frame(1, 0, [2.0, 1.0, 0.0], [-0.25, 0.0, 0.0], True),
            ],
            episode_index=1,
            task_index=1,
            fps=10.0,
        ),
    ]
    write_dataset(
        episodes,
        str(root),
        repo_id="acme/pusht",
        fps=10.0,
        tasks=["approach", "retreat"],
        robot_type="sim",
    )


class FakeHub:
    """In-process stand-in for the Hugging Face Hub. No network."""

    def __init__(self, root: Path, revision: str, licence: str) -> None:
        self.root = root
        self.revision = revision
        self.licence = licence
        self.calls: list[tuple[str, str | None]] = []

    def fetch(self, repo_id: str, revision: str | None) -> HubSnapshot:
        self.calls.append((repo_id, revision))
        return HubSnapshot(
            repo_id=repo_id,
            revision=self.revision,
            licence=self.licence,
            root=self.root,
        )


def _import_file(store: Path, sha: str) -> Path:
    matches = list((store / "imports" / "hf").glob(f"*/{sha}.json"))
    if len(matches) != 1:
        raise AssertionError(f"expected one import file for {sha}, found {matches}")
    return matches[0]


class TestDatasetHfImport(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = self.root / "store"
        self.source = self.root / "snapshot"
        _write_source(self.source)

    def _hub(self, licence: str = "apache-2.0", revision: str = SHA) -> FakeHub:
        return FakeHub(self.source, revision, licence)

    def test_import_pins_resolved_revision_and_licence(self) -> None:
        hub = self._hub()
        imported = import_hf_dataset(
            "acme/pusht", revision="main", store=self.store, hub=hub
        )
        self.assertEqual(hub.calls, [("acme/pusht", "main")])
        self.assertEqual(imported.revision, SHA)
        self.assertNotEqual(imported.revision, "main")
        self.assertEqual(imported.requested_revision, "main")
        self.assertEqual(imported.licence, "apache-2.0")
        self.assertFalse(imported.non_commercial)
        self.assertIsNone(imported.warning)
        record = load_version(imported.version_id, store=self.store)
        self.assertTrue(record.pinned)
        self.assertEqual(record.licence, "apache-2.0")
        self.assertEqual(record.source_query.roots, (f"hf://acme/pusht@{SHA}",))
        self.assertEqual(record.dataset_name, "pusht")
        dataset = resolve_dataset_version(imported.version_id, store=self.store)
        reader = DatasetReader(str(dataset))
        self.assertEqual(reader.episode_count, 2)
        info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["repo_id"], "acme/pusht")
        document = json.loads(_import_file(self.store, SHA).read_text(encoding="utf-8"))
        self.assertTrue(document["revision_pinned"])
        self.assertTrue(document["licence_pinned"])
        self.assertEqual(document["revision"], SHA)
        self.assertEqual(document["licence"], "apache-2.0")
        graph = JsonLineageStore(self.store)
        episode_ids = graph.episodes_for_version(imported.version_id)
        self.assertEqual(len(episode_ids), 2)
        self.assertEqual(graph.episode(episode_ids[0]).task, "approach")
        self.assertEqual(graph.episode(episode_ids[1]).task, "retreat")
        self.assertEqual(graph.episode(episode_ids[0]).source, f"hf://acme/pusht@{SHA}")
        self.assertEqual(graph.episode(episode_ids[0]).source_index, 0)
        self.assertEqual(graph.episode(episode_ids[0]).fps, "10.00000000")

        raw = _import_file(self.store, SHA).read_bytes()
        version_raw = (
            self.store / "versions" / imported.version_id / "version.json"
        ).read_bytes()
        hub.revision = SHA_B
        moved = import_hf_dataset(
            "acme/pusht", revision="main", store=self.store, hub=hub
        )
        self.assertEqual(moved.revision, SHA_B)
        self.assertNotEqual(moved.version_id, imported.version_id)
        self.assertEqual(_import_file(self.store, SHA).read_bytes(), raw)
        self.assertEqual(
            (
                self.store / "versions" / imported.version_id / "version.json"
            ).read_bytes(),
            version_raw,
        )

    def test_repeat_import_does_not_rewrite_the_pin(self) -> None:
        hub = self._hub()
        first = import_hf_dataset(
            "acme/pusht", revision="main", store=self.store, hub=hub
        )
        raw = _import_file(self.store, SHA).read_bytes()
        version_raw = (
            self.store / "versions" / first.version_id / "version.json"
        ).read_bytes()
        graph_raw = (self.store / "lineage" / "graph.json").read_bytes()
        second = import_hf_dataset(
            "acme/pusht", revision=SHA, store=self.store, hub=hub
        )
        self.assertEqual(second, first)
        self.assertEqual(second.requested_revision, "main")
        self.assertEqual(_import_file(self.store, SHA).read_bytes(), raw)
        self.assertEqual(
            (self.store / "versions" / first.version_id / "version.json").read_bytes(),
            version_raw,
        )
        self.assertEqual(
            (self.store / "lineage" / "graph.json").read_bytes(), graph_raw
        )

    def test_changed_licence_for_the_same_revision_is_refused(self) -> None:
        hub = self._hub(licence="apache-2.0")
        import_hf_dataset("acme/pusht", revision="main", store=self.store, hub=hub)
        raw = _import_file(self.store, SHA).read_bytes()
        versions = sorted(path.name for path in (self.store / "versions").iterdir())
        hub.licence = "mit"
        with self.assertRaises(DatasetImportError) as caught:
            import_hf_dataset("acme/pusht", revision="main", store=self.store, hub=hub)
        self.assertIn("pinned licence", str(caught.exception))
        self.assertIn("apache-2.0", str(caught.exception))
        self.assertEqual(_import_file(self.store, SHA).read_bytes(), raw)
        self.assertEqual(
            sorted(path.name for path in (self.store / "versions").iterdir()),
            versions,
        )

    def test_non_commercial_licence_is_flagged(self) -> None:
        imported = import_hf_dataset(
            "acme/pusht",
            revision="v1",
            store=self.store,
            hub=self._hub(licence="cc-by-nc-4.0"),
        )
        self.assertEqual(imported.revision, SHA)
        self.assertEqual(imported.requested_revision, "v1")
        self.assertEqual(imported.licence, "cc-by-nc-4.0")
        self.assertTrue(imported.non_commercial)
        self.assertEqual(imported.warning, "licence cc-by-nc-4.0 is non-commercial")
        record = load_version(imported.version_id, store=self.store)
        self.assertEqual(record.licence, "cc-by-nc-4.0")
        self.assertTrue(record.pinned)
        document = json.loads(_import_file(self.store, SHA).read_text(encoding="utf-8"))
        self.assertTrue(document["non_commercial"])
        self.assertTrue(document["licence_pinned"])

    def test_commercial_licence_is_not_flagged(self) -> None:
        for licence in ("apache-2.0", "MIT", "cc-by-4.0", "cc-by-sa-4.0"):
            with self.subTest(licence=licence):
                self.assertFalse(is_non_commercial(licence))
        self.assertTrue(is_non_commercial("CC-BY-NC-4.0"))
        self.assertTrue(is_non_commercial("cc-by-nc-sa-4.0"))
        self.assertTrue(is_non_commercial("cc-by-nc-nd-3.0"))
        self.assertTrue(is_non_commercial("cc-by-nc"))
        self.assertTrue(is_non_commercial("CC BY-NC-SA 4.0"))
        self.assertTrue(is_non_commercial("other (non-commercial)"))
        self.assertFalse(is_non_commercial("cc-by-nd-4.0"))
        self.assertFalse(is_non_commercial("unspecified"))
        self.assertFalse(is_non_commercial("gpl-3.0"))

    def test_content_hash_matches_a_local_build(self) -> None:
        imported = import_hf_dataset(
            "acme/pusht", revision=SHA, store=self.store, hub=self._hub()
        )
        local = build_version(
            DatasetSpec(
                source_query=SourceQuery((str(self.source),)),
                dataset_name="local-copy",
                licence="unspecified",
            ),
            store=self.store,
        )
        record = load_version(imported.version_id, store=self.store)
        self.assertEqual(record.content_hash, local.content_hash)
        self.assertNotEqual(record.version_id, local.version_id)
        self.assertTrue(record.pinned)
        self.assertFalse(local.pinned)

    def test_import_episode_delete_keeps_lineage(self) -> None:
        imported = import_hf_dataset(
            "acme/pusht", revision="main", store=self.store, hub=self._hub()
        )
        graph = JsonLineageStore(self.store)
        graph.link_run("run_import", imported.version_id)
        graph.link_model("mv_import", "run_import")
        episode_ids = graph.episodes_for_version(imported.version_id)
        chain = graph.lineage_from_model("mv_import")
        self.assertEqual(chain.dataset_version_id, imported.version_id)
        self.assertEqual(chain.episode_ids, episode_ids)
        trained = graph.models_trained_on(episode_ids[0])
        self.assertEqual(trained[0].model_version_id, "mv_import")
        self.assertEqual(trained[0].run_id, "run_import")
        version_dir = self.store / "versions" / imported.version_id
        before = version_dir.joinpath("version.json").read_bytes()
        deletion = graph.delete_episode(episode_ids[0])
        self.assertTrue(deletion.blocked)
        self.assertTrue(deletion.soft_deleted)
        self.assertIn("soft-deleted", deletion.warning or "")
        self.assertIn(imported.version_id, deletion.warning or "")
        self.assertEqual(version_dir.joinpath("version.json").read_bytes(), before)
        self.assertEqual(graph.episodes_for_version(imported.version_id), episode_ids)
        self.assertTrue(graph.episode(episode_ids[0]).deleted)
        self.assertEqual(
            graph.models_trained_on(episode_ids[0])[0].model_version_id, "mv_import"
        )
        self.assertFalse(graph.episode(episode_ids[1]).deleted)

    def test_hub_must_return_a_commit(self) -> None:
        hub = self._hub(revision="main")
        with self.assertRaises(DatasetImportError) as caught:
            import_hf_dataset("acme/pusht", revision="main", store=self.store, hub=hub)
        self.assertIn("commit", str(caught.exception))
        self.assertFalse((self.store / "versions").exists())
        self.assertFalse((self.store / "imports").exists())

    def test_snapshot_must_be_a_lerobot_dataset(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        hub = FakeHub(empty, SHA, "mit")
        with self.assertRaises(DatasetImportError) as caught:
            import_hf_dataset("acme/pusht", store=self.store, hub=hub)
        self.assertIn("meta/info.json", str(caught.exception))

    def test_revision_argument_is_checked_before_the_hub(self) -> None:
        hub = self._hub()
        with self.assertRaises(DatasetImportError):
            import_hf_dataset(
                "acme/pusht", revision="../main", store=self.store, hub=hub
            )
        self.assertEqual(hub.calls, [])

    def test_resolved_licence_reads_the_card_then_the_readme(self) -> None:
        class _Card:
            license = "cc-by-nc-4.0"

        readme = "---\nlicense: mit\n---\n# card\n"
        self.assertEqual(resolved_licence(_Card(), readme), "cc-by-nc-4.0")
        self.assertEqual(
            resolved_licence({"license": ["apache-2.0", "mit"]}, None), "apache-2.0"
        )
        self.assertEqual(
            resolved_licence(None, "---\nlicense: apache-2.0\n---\n"), "apache-2.0"
        )
        listed = "---\nlicense:\n  - cc-by-nc-4.0\n---\n"
        self.assertEqual(resolved_licence(None, listed), "cc-by-nc-4.0")
        self.assertEqual(resolved_licence(None, "no front matter"), "unspecified")

    def test_missing_huggingface_hub_has_a_clear_error(self) -> None:
        hub = HuggingFaceDatasetHub(self.root / "cache")
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with self.assertRaises(DatasetImportError) as caught:
                hub.fetch("acme/pusht", None)
        message = str(caught.exception)
        self.assertIn("huggingface_hub", message)
        self.assertIn("ohho-os[datasets]", message)

    def test_cli_import_hf_json(self) -> None:
        hub = self._hub()
        with patch("ohho.datasets.hf_import.default_dataset_hub", return_value=hub):
            code, out, err = _run(
                [
                    "dataset",
                    "import-hf",
                    "acme/pusht",
                    "--revision",
                    "main",
                    "--store",
                    str(self.store),
                    "--json",
                ]
            )
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertEqual(payload["repo_id"], "acme/pusht")
        self.assertEqual(payload["revision"], SHA)
        self.assertEqual(payload["requested_revision"], "main")
        self.assertEqual(payload["licence"], "apache-2.0")
        self.assertFalse(payload["non_commercial"])
        self.assertTrue(payload["revision_pinned"])
        self.assertTrue(payload["licence_pinned"])
        self.assertTrue(payload["pinned"])
        self.assertIsNone(payload["warning"])
        self.assertEqual(hub.calls, [("acme/pusht", "main")])

    def test_cli_warns_on_non_commercial_licence(self) -> None:
        hub = self._hub(licence="cc-by-nc-4.0")
        with patch("ohho.datasets.hf_import.default_dataset_hub", return_value=hub):
            code, out, err = _run(
                [
                    "dataset",
                    "import-hf",
                    "acme/pusht",
                    "--revision",
                    "main",
                    "--store",
                    str(self.store),
                ]
            )
        self.assertEqual(code, 0, err)
        self.assertIn("warning: licence cc-by-nc-4.0 is non-commercial", err)
        self.assertIn(f"revision: {SHA}", out)
        self.assertIn("licence: cc-by-nc-4.0", out)
        self.assertIn("non_commercial: true", out)

    def test_cli_rejects_a_bad_repo_id(self) -> None:
        def fail_hub(store: object) -> None:
            raise AssertionError(f"hub should not be called for {store}")

        with patch("ohho.datasets.hf_import.default_dataset_hub", fail_hub):
            code, _out, err = _run(
                ["dataset", "import-hf", "not a repo", "--store", str(self.store)]
            )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertIn("repo id", err)

    def test_import_schema_names_the_record(self) -> None:
        import ohho.datasets.hf_import as import_mod

        path = Path(import_mod.__file__).with_name("dataset_import.schema.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(schema["title"], "DatasetImport")
        self.assertEqual(
            schema["properties"]["schema"]["const"], "ohho.dataset_import.v1"
        )
        self.assertEqual(schema["properties"]["revision_pinned"]["const"], True)
        self.assertEqual(schema["properties"]["licence_pinned"]["const"], True)
        self.assertIn("non_commercial", schema["required"])
        self.assertIn("revision", schema["required"])
        self.assertIn("licence", schema["required"])


if __name__ == "__main__":
    unittest.main()
