"""Tests for per-run artifact retention and storage totals."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ohho.cli import main as cli_main
from ohho.train.artifacts import (
    CHECKPOINTS_DIR,
    DEFAULT_MAX_AGE_DAYS,
    EVAL_DIR,
    INDEX_NAME,
    LAYOUT,
    LOGS_DIR,
    SCHEMA_ID,
    ArtifactError,
    RetentionPolicy,
    apply_retention,
    checkpoint_role,
    empty_artifact_index,
    ensure_layout,
    load_artifact_index,
    main as artifacts_main,
    schema_document,
    schema_path,
    set_checkpoint_retention,
    set_run_retention,
    storage_used,
    storage_used_experiment,
    validate_artifact_index,
)
from ohho.train.run_manifest import MANIFEST_NAME, validate_run_manifest


def _manifest() -> dict:
    return {
        "schema": "ohho.run/v1",
        "status": "succeeded",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T01:00:00Z",
        "family": "act",
        "device": "cpu",
        "resolved_config": {"family": "act"},
        "seeds": {
            "recipe": 1,
            "python": 1,
            "numpy": None,
            "torch": None,
            "pythonhashseed": None,
            "extra": {},
        },
        "git": {"commit": None, "dirty": False, "root": None},
        "dataset": {"version_id": None, "path": None, "hash": None},
        "base_weights": {
            "uri": None,
            "revision": None,
            "hash": None,
            "loaded": False,
        },
        "container_image_digest": None,
        "library_versions": {"python": "3.12.0", "ohho": "1.2.0"},
        "gpu": None,
        "final_loss": 0.5,
        "error": None,
        "recipe_path": "recipe.yaml",
        "overrides": [],
        "checkpoint_dir": None,
        "metrics_path": None,
        "declared_hooks": {"compression_hooks": [], "eval_hooks": []},
    }


def _stamp(path: Path, moment: datetime) -> None:
    value = moment.timestamp()
    os.utime(path, (value, value))


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class TestArtifactSchema(unittest.TestCase):
    def test_schema_file_and_empty_index(self) -> None:
        document = schema_document()
        self.assertEqual(document["title"], SCHEMA_ID)
        self.assertEqual(
            document["$id"],
            "https://ohho-robotics.com/schemas/ohho.artifacts/v1.json",
        )
        self.assertTrue(schema_path().is_file())
        self.assertEqual(validate_artifact_index(empty_artifact_index()), [])
        extra = empty_artifact_index()
        extra["note"] = "no"
        self.assertTrue(validate_artifact_index(extra))

    def test_layout_names_and_roles(self) -> None:
        self.assertEqual(
            LAYOUT,
            {
                "manifest": "run_manifest.json",
                "metrics": "metrics.json",
                "index": "artifact_index.json",
                "checkpoints": "checkpoints",
                "logs": "logs",
                "eval": "eval",
            },
        )
        self.assertEqual(DEFAULT_MAX_AGE_DAYS, 14)
        self.assertEqual(checkpoint_role("best"), "best")
        self.assertEqual(checkpoint_role("best.pt"), "best")
        self.assertEqual(checkpoint_role("last.json"), "last")
        self.assertEqual(checkpoint_role("step-0001.pt"), "step")
        self.assertEqual(checkpoint_role("step_2"), "step")
        self.assertEqual(checkpoint_role("epoch-3"), "epoch")
        self.assertEqual(checkpoint_role("epoch_4.ckpt"), "epoch")
        self.assertEqual(checkpoint_role("export.pt"), "other")
        self.assertEqual(checkpoint_role("best-ema.pt"), "other")
        with self.assertRaises(ArtifactError):
            RetentionPolicy(max_age_days=-1)
        with self.assertRaises(ArtifactError):
            RetentionPolicy(max_age_days=True)  # type: ignore[arg-type]

    def test_module_does_not_touch_the_network_or_a_database(self) -> None:
        source = Path(schema_path().parent.parent / "artifacts.py").read_text(
            encoding="utf-8"
        )
        for banned in (
            "urllib.request",
            "http.client",
            "socket",
            "requests",
            "sqlite3",
            "credentials.json",
        ):
            self.assertNotIn(banned, source)


class TestArtifactRetention(unittest.TestCase):
    def _run(self, root: Path, name: str = "run") -> Path:
        run = root / name
        ensure_layout(run)
        manifest = _manifest()
        errors = validate_run_manifest(manifest)
        if errors:
            raise AssertionError(errors)
        (run / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (run / "metrics.json").write_text('{"final_loss": 0.5}\n', encoding="utf-8")
        (run / "config.json").write_bytes(b"{}\n")
        (run / "policy.pt").write_bytes(b"policy")
        (run / "checkpoint.json").write_bytes(b"{}\n")
        _write(run / "model_final" / "weights.bin", b"MM")
        _write(run / CHECKPOINTS_DIR / "best" / "weights.bin", b"BBBBB")
        _write(run / CHECKPOINTS_DIR / "last.pt", b"LLLL")
        _write(run / CHECKPOINTS_DIR / "step-0001.pt", b"SSSSSSSSS")
        _write(run / CHECKPOINTS_DIR / "step-0002.pt", b"TTTT")
        _write(run / LOGS_DIR / "train.log", b"log\n")
        _write(run / EVAL_DIR / "report.json", b"{}\n")
        return run

    def _protected(self, run: Path) -> dict[str, bytes]:
        return {
            "manifest": (run / MANIFEST_NAME).read_bytes(),
            "metrics": (run / "metrics.json").read_bytes(),
            "config": (run / "config.json").read_bytes(),
            "policy": (run / "policy.pt").read_bytes(),
            "checkpoint": (run / "checkpoint.json").read_bytes(),
            "model": (run / "model_final" / "weights.bin").read_bytes(),
            "best": (run / CHECKPOINTS_DIR / "best" / "weights.bin").read_bytes(),
            "last": (run / CHECKPOINTS_DIR / "last.pt").read_bytes(),
            "log": (run / LOGS_DIR / "train.log").read_bytes(),
            "eval": (run / EVAL_DIR / "report.json").read_bytes(),
        }

    def test_storage_used_splits_a_run_and_an_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            usage = storage_used(run)
            self.assertEqual(Path(usage.run_dir), run.resolve())
            self.assertIsNone(usage.experiment)
            self.assertEqual(usage.deleted_bytes, 0)
            self.assertEqual(usage.logs_bytes, len(b"log\n"))
            self.assertEqual(usage.eval_bytes, len(b"{}\n"))
            self.assertEqual(usage.metrics_bytes, (run / "metrics.json").stat().st_size)
            self.assertEqual(usage.manifest_bytes, (run / MANIFEST_NAME).stat().st_size)
            self.assertEqual(usage.other_bytes, len(b"{}\n"))
            checkpoint_bytes = sum(
                path.stat().st_size
                for path in (
                    run / "policy.pt",
                    run / "checkpoint.json",
                    run / "model_final" / "weights.bin",
                    run / CHECKPOINTS_DIR / "best" / "weights.bin",
                    run / CHECKPOINTS_DIR / "last.pt",
                    run / CHECKPOINTS_DIR / "step-0001.pt",
                    run / CHECKPOINTS_DIR / "step-0002.pt",
                )
            )
            self.assertEqual(usage.checkpoints_bytes, checkpoint_bytes)
            self.assertEqual(
                usage.total_bytes,
                usage.checkpoints_bytes
                + usage.logs_bytes
                + usage.eval_bytes
                + usage.metrics_bytes
                + usage.manifest_bytes
                + usage.other_bytes,
            )
            body = usage.to_dict()
            self.assertEqual(
                set(body),
                {
                    "run_dir",
                    "experiment",
                    "total_bytes",
                    "checkpoints_bytes",
                    "logs_bytes",
                    "eval_bytes",
                    "metrics_bytes",
                    "manifest_bytes",
                    "other_bytes",
                    "deleted_bytes",
                },
            )
            rolled = storage_used_experiment(run / MANIFEST_NAME)
            self.assertEqual(len(rolled.runs), 1)
            self.assertEqual(rolled.total_bytes, usage.total_bytes)
            self.assertEqual(
                rolled.to_dict()["runs"][0]["total_bytes"], usage.total_bytes
            )
            with self.assertRaises(ArtifactError):
                storage_used(root)
            self.assertFalse((run / INDEX_NAME).exists())
            loaded = load_artifact_index(run)
            self.assertEqual(loaded["schema"], SCHEMA_ID)
            self.assertFalse((run / INDEX_NAME).exists())

    def test_pruned_trial_deletes_old_intermediates_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            old = datetime(2026, 1, 1, tzinfo=timezone.utc)
            young = datetime(2026, 6, 10, tzinfo=timezone.utc)
            now = datetime(2026, 6, 15, tzinfo=timezone.utc)
            _stamp(run / CHECKPOINTS_DIR / "step-0001.pt", old)
            _stamp(run / CHECKPOINTS_DIR / "step-0002.pt", young)
            _stamp(run / LOGS_DIR / "train.log", old)
            _stamp(run / EVAL_DIR / "report.json", old)
            step_bytes = (run / CHECKPOINTS_DIR / "step-0001.pt").stat().st_size
            before = self._protected(run)
            checkpoints_before = storage_used(run).checkpoints_bytes
            set_run_retention(run, pruned=True, now=now)
            self.assertEqual((run / MANIFEST_NAME).read_bytes(), before["manifest"])
            index_before = (run / INDEX_NAME).read_bytes()
            dry = apply_retention(run, now=now, dry_run=True)
            self.assertTrue(dry.dry_run)
            self.assertEqual(len(dry.runs[0].deleted), 1)
            self.assertEqual(dry.runs[0].deleted[0].path, "checkpoints/step-0001.pt")
            self.assertEqual(dry.runs[0].deleted[0].size_bytes, step_bytes)
            self.assertIn("older than 14 days", dry.runs[0].deleted[0].reason)
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertEqual((run / INDEX_NAME).read_bytes(), index_before)
            self.assertEqual(self._protected(run), before)
            result = apply_retention(run, now=now, dry_run=False)
            self.assertFalse(result.dry_run)
            self.assertEqual(result.deleted_bytes, step_bytes)
            self.assertFalse((run / CHECKPOINTS_DIR / "step-0001.pt").exists())
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0002.pt").is_file())
            self.assertEqual(self._protected(run), before)
            self.assertEqual(validate_run_manifest(json.loads(before["manifest"])), [])
            kept = {item.path: item.reason for item in result.runs[0].kept}
            self.assertEqual(kept["checkpoints/best"], "best checkpoint")
            self.assertEqual(kept["checkpoints/last.pt"], "last checkpoint")
            self.assertEqual(
                kept["checkpoints/step-0002.pt"],
                f"newer than {DEFAULT_MAX_AGE_DAYS} days",
            )
            index = load_artifact_index(run)
            self.assertEqual(validate_artifact_index(index), [])
            self.assertEqual(len(index["deletions"]), 1)
            self.assertEqual(index["deletions"][0]["path"], "checkpoints/step-0001.pt")
            self.assertEqual(index["deletions"][0]["bytes"], step_bytes)
            self.assertEqual(index["deletions"][0]["role"], "step")
            self.assertEqual(
                index["deletions"][0]["deleted_at"], "2026-06-15T00:00:00Z"
            )
            self.assertNotIn(
                "checkpoints/step-0001.pt",
                [item["path"] for item in index["checkpoints"]],
            )
            usage = storage_used(run)
            self.assertEqual(usage.deleted_bytes, step_bytes)
            self.assertEqual(usage.checkpoints_bytes, checkpoints_before - step_bytes)
            self.assertEqual(usage.manifest_bytes, len(before["manifest"]))
            self.assertEqual(usage.metrics_bytes, len(before["metrics"]))
            again = apply_retention(run, now=now)
            self.assertEqual(again.deleted_bytes, 0)
            self.assertEqual(len(load_artifact_index(run)["deletions"]), 1)
            self.assertEqual(self._protected(run), before)

    def test_age_boundary_keeps_a_checkpoint_at_the_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            cutoff = now - timedelta(days=14)
            _stamp(run / CHECKPOINTS_DIR / "step-0001.pt", cutoff)
            _stamp(
                run / CHECKPOINTS_DIR / "step-0002.pt",
                cutoff - timedelta(seconds=2),
            )
            set_run_retention(run, pruned=True, now=now)
            result = apply_retention(run, now=now)
            deleted = {item.path for item in result.runs[0].deleted}
            self.assertEqual(deleted, {"checkpoints/step-0002.pt"})
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())

    def test_unpruned_trial_keeps_old_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            _stamp(
                run / CHECKPOINTS_DIR / "step-0001.pt",
                datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            result = apply_retention(
                run, now=datetime(2026, 6, 15, tzinfo=timezone.utc)
            )
            self.assertEqual(result.deleted_bytes, 0)
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            reasons = {item.path: item.reason for item in result.runs[0].kept}
            self.assertEqual(reasons["checkpoints/step-0001.pt"], "trial is not pruned")
            self.assertFalse((run / INDEX_NAME).exists())

    def test_pinned_and_promoted_runs_are_kept(self) -> None:
        now = datetime(2026, 6, 15, tzinfo=timezone.utc)
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        for flag in ("pinned", "promoted"):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmp:
                    run = self._run(Path(tmp))
                    _stamp(run / CHECKPOINTS_DIR / "step-0001.pt", old)
                    _stamp(run / CHECKPOINTS_DIR / "step-0002.pt", old)
                    before = self._protected(run)
                    set_run_retention(run, pruned=True, now=now, **{flag: True})
                    result = apply_retention(run, now=now)
                    self.assertEqual(result.deleted_bytes, 0)
                    self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
                    self.assertTrue((run / CHECKPOINTS_DIR / "step-0002.pt").is_file())
                    self.assertEqual(self._protected(run), before)
                    self.assertEqual(load_artifact_index(run)["deletions"], [])
                    reasons = {item.reason for item in result.runs[0].kept}
                    self.assertEqual(reasons, {f"{flag} run"})

    def test_pinned_and_promoted_checkpoints_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            now = datetime(2026, 6, 15, tzinfo=timezone.utc)
            old = datetime(2020, 1, 1, tzinfo=timezone.utc)
            _write(run / CHECKPOINTS_DIR / "step-0003.pt", b"UUUUU")
            for name in ("step-0001.pt", "step-0002.pt", "step-0003.pt"):
                _stamp(run / CHECKPOINTS_DIR / name, old)
            before = self._protected(run)
            set_run_retention(run, pruned=True, now=now)
            set_checkpoint_retention(run, "step-0001.pt", pinned=True)
            set_checkpoint_retention(run, "checkpoints/step-0002.pt", promoted=True)
            self.assertEqual((run / MANIFEST_NAME).read_bytes(), before["manifest"])
            result = apply_retention(run, now=now)
            deleted = [item.path for item in result.runs[0].deleted]
            self.assertEqual(deleted, ["checkpoints/step-0003.pt"])
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0002.pt").is_file())
            self.assertFalse((run / CHECKPOINTS_DIR / "step-0003.pt").exists())
            self.assertEqual(self._protected(run), before)
            reasons = {item.path: item.reason for item in result.runs[0].kept}
            self.assertEqual(reasons["checkpoints/step-0001.pt"], "pinned checkpoint")
            self.assertEqual(reasons["checkpoints/step-0002.pt"], "promoted checkpoint")
            index = load_artifact_index(run)
            flags = {item["path"]: item for item in index["checkpoints"]}
            self.assertTrue(flags["checkpoints/step-0001.pt"]["pinned"])
            self.assertTrue(flags["checkpoints/step-0002.pt"]["promoted"])

    def test_invalid_index_deletes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            _stamp(
                run / CHECKPOINTS_DIR / "step-0001.pt",
                datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            before = self._protected(run)
            (run / INDEX_NAME).write_text("{", encoding="utf-8")
            result = apply_retention(
                run, now=datetime(2026, 6, 15, tzinfo=timezone.utc)
            )
            self.assertEqual(result.deleted_bytes, 0)
            self.assertTrue(result.errors)
            self.assertIn("could not read", result.errors[0])
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertEqual(self._protected(run), before)
            (run / INDEX_NAME).write_text(
                json.dumps({"schema": "nope"}), encoding="utf-8"
            )
            result = apply_retention(
                run, now=datetime(2026, 6, 15, tzinfo=timezone.utc)
            )
            self.assertTrue(result.errors)
            self.assertIn("not valid", result.errors[0])
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertEqual((run / "metrics.json").read_bytes(), before["metrics"])

    def test_checkpoint_path_must_stay_inside_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            outside = root / "secret.bin"
            outside.write_bytes(b"secret")
            with self.assertRaises(ArtifactError):
                set_checkpoint_retention(run, "../secret.bin", pinned=True)
            with self.assertRaises(ArtifactError):
                set_checkpoint_retention(
                    run, "checkpoints/../../secret.bin", pinned=True
                )
            self.assertEqual(outside.read_bytes(), b"secret")

    def test_symlink_checkpoint_unlinks_without_deleting_the_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            outside = root / "outside.bin"
            outside.write_bytes(b"keep-me")
            link = run / CHECKPOINTS_DIR / "step-0009.pt"
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("symlinks are not available")
            before = self._protected(run)
            set_run_retention(
                run, pruned=True, now=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            result = apply_retention(run, now=datetime(2099, 1, 1, tzinfo=timezone.utc))
            deleted = {item.path for item in result.runs[0].deleted}
            self.assertIn("checkpoints/step-0009.pt", deleted)
            self.assertFalse(link.exists())
            self.assertFalse(link.is_symlink())
            self.assertEqual(outside.read_bytes(), b"keep-me")
            self.assertEqual(self._protected(run), before)

    def test_experiment_filter_selects_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 8, 1, tzinfo=timezone.utc)
            old = datetime(2026, 1, 1, tzinfo=timezone.utc)

            def make(parent: str, name: str, *, pruned: bool) -> Path:
                run = self._run(root / parent, name)
                _stamp(run / CHECKPOINTS_DIR / "step-0001.pt", old)
                if pruned:
                    set_run_retention(run, pruned=True, now=now)
                return run

            run_a = make("walk", "run-a", pruned=True)
            run_b = make("walk", "run-b", pruned=False)
            run_c = make("other", "run-c", pruned=True)
            run_d = make("other", "run-d", pruned=True)
            set_run_retention(run_d, experiment="walk")
            run_e = make("walk", "run-e", pruned=True)
            set_run_retention(run_e, experiment="other")
            usage = storage_used_experiment(root, experiment="walk")
            self.assertEqual(usage.experiment, "walk")
            self.assertEqual(
                {Path(item.run_dir).name for item in usage.runs},
                {"run-a", "run-b", "run-d"},
            )
            self.assertEqual(
                usage.total_bytes, sum(item.total_bytes for item in usage.runs)
            )
            whole = storage_used_experiment(root)
            self.assertEqual(len(whole.runs), 5)
            self.assertIsNone(whole.experiment)
            result = apply_retention(root, now=now, experiment="walk")
            touched = {Path(item.run_dir).name for item in result.runs}
            self.assertEqual(touched, {"run-a", "run-b", "run-d"})
            self.assertFalse((run_a / CHECKPOINTS_DIR / "step-0001.pt").exists())
            self.assertTrue((run_b / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertTrue((run_c / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertFalse((run_d / CHECKPOINTS_DIR / "step-0001.pt").exists())
            self.assertTrue((run_e / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertEqual(
                (run_c / MANIFEST_NAME).read_bytes(),
                (run_a / MANIFEST_NAME).read_bytes(),
            )
            with self.assertRaises(ArtifactError):
                storage_used_experiment(root, experiment="missing")
            with self.assertRaises(ArtifactError):
                apply_retention(root, experiment="missing", now=now)


class TestArtifactCli(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        run = root / "run"
        ensure_layout(run)
        (run / MANIFEST_NAME).write_text(
            json.dumps(_manifest(), indent=2) + "\n", encoding="utf-8"
        )
        (run / "metrics.json").write_text('{"final_loss": 0.5}\n', encoding="utf-8")
        _write(run / CHECKPOINTS_DIR / "best" / "weights.bin", b"BBBBB")
        _write(run / CHECKPOINTS_DIR / "last.pt", b"LLLL")
        _write(run / CHECKPOINTS_DIR / "step-0001.pt", b"SSSSSSSSS")
        _write(run / LOGS_DIR / "train.log", b"log\n")
        _write(run / EVAL_DIR / "report.json", b"{}\n")
        return run

    def test_retain_dry_run_then_delete_leaves_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            _stamp(
                run / CHECKPOINTS_DIR / "step-0001.pt",
                datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            set_run_retention(run, pruned=True)
            manifest = (run / MANIFEST_NAME).read_bytes()
            metrics = (run / "metrics.json").read_bytes()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli_main(["run", "retain", str(run), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("dry-run: 1 checkpoint,", stdout.getvalue())
            self.assertIn("checkpoints/step-0001.pt", stdout.getvalue())
            self.assertTrue((run / CHECKPOINTS_DIR / "step-0001.pt").is_file())
            self.assertEqual((run / MANIFEST_NAME).read_bytes(), manifest)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = cli_main(["run", "retain", str(run), "--json"])
            self.assertEqual(code, 0)
            body = json.loads(stdout.getvalue())
            self.assertFalse(body["dry_run"])
            self.assertEqual(body["max_age_days"], 14)
            self.assertEqual(
                body["runs"][0]["deleted"][0]["path"], "checkpoints/step-0001.pt"
            )
            self.assertFalse((run / CHECKPOINTS_DIR / "step-0001.pt").exists())
            self.assertEqual((run / MANIFEST_NAME).read_bytes(), manifest)
            self.assertEqual((run / "metrics.json").read_bytes(), metrics)
            self.assertTrue((run / CHECKPOINTS_DIR / "best" / "weights.bin").is_file())
            self.assertTrue((run / CHECKPOINTS_DIR / "last.pt").is_file())
            self.assertEqual((run / LOGS_DIR / "train.log").read_bytes(), b"log\n")
            self.assertEqual((run / EVAL_DIR / "report.json").read_bytes(), b"{}\n")
            show_out = io.StringIO()
            with redirect_stdout(show_out), redirect_stderr(io.StringIO()):
                show_code = cli_main(["run", "show", str(run)])
            self.assertEqual(show_code, 0)
            self.assertIn("status: succeeded", show_out.getvalue())
            storage_out = io.StringIO()
            with redirect_stdout(storage_out), redirect_stderr(io.StringIO()):
                storage_code = cli_main(["run", "storage", str(run), "--json"])
            self.assertEqual(storage_code, 0)
            stored = json.loads(storage_out.getvalue())
            self.assertEqual(stored["runs"][0]["deleted_bytes"], len(b"SSSSSSSSS"))
            self.assertGreater(stored["total_bytes"], 0)

    def test_module_entry_and_usage_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                code = artifacts_main(["storage", str(run), "--json"])
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["runs"][0]["experiment"], None
            )
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = cli_main(["run", "retain", str(run), "--days", "-1"])
            self.assertEqual(code, 2)
            self.assertIn("max_age_days", stderr.getvalue())
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = cli_main(["run", "retain", str(Path(tmp) / "missing")])
            self.assertEqual(code, 1)
            self.assertIn("does not exist", stderr.getvalue())
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                code = cli_main(["run", "retain", str(run), "--experiment", "missing"])
            self.assertEqual(code, 1)
            self.assertIn("no run matches", stderr.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli_main(["run", "retain"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
