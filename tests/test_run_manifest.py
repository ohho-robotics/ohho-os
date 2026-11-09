"""Tests for the ohho.run/v1 manifest and `ohho run show`."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ohho.cli import main as cli_main
from ohho.train.recipe_registry import TrainResult, register, unregister
from ohho.train.recipe_schema import TrainFailed
from ohho.train.run_cli import EXIT_OK, EXIT_TRAIN_FAILED, execute
from ohho.train.run_manifest import (
    CONSOLE_FIELDS,
    MANIFEST_NAME,
    SCHEMA_ID,
    container_image_digest,
    format_run_text,
    git_state,
    gpu_snapshot,
    hash_base_weights,
    hash_tree,
    in_container,
    schema_document,
    schema_path,
    set_run_store,
    validate_run_manifest,
)
from ohho.train.schema_validate import validate_instance


def _run(path: Path, overrides: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "OhhO Test"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_NAME"] = "OhhO Test"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    command = [
        "git",
        "-c",
        "safe.directory=*",
        "-c",
        "core.autocrlf=false",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=OhhO Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "core.pager=cat",
        *args,
    ]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=30,
        **kwargs,
    )


def _init_repo(root: Path) -> None:
    if shutil.which("git") is None:
        raise unittest.SkipTest("git is not installed")
    root.mkdir(parents=True, exist_ok=True)
    created = _git(root, "init", "-b", "main")
    if created.returncode != 0:
        created = _git(root, "init")
    if created.returncode != 0:
        raise AssertionError(created.stderr or created.stdout)


class _RecordingFamily:
    id = "tiny_manifest"

    def __init__(self) -> None:
        self.mid: dict[str, Any] | None = None
        self.existed_before_start = False

    def available(self) -> bool:
        return True

    def run(self, recipe: dict[str, Any], emit) -> TrainResult:
        out = Path(recipe["output"]["dir"])
        manifest = out / MANIFEST_NAME
        self.existed_before_start = manifest.exists()
        emit({"event": "train_start", "family": self.id, "device": "cpu"})
        self.mid = json.loads(manifest.read_text(encoding="utf-8"))
        metrics = out / "metrics.json"
        metrics.write_text(json.dumps({"final_loss": 0.25}), encoding="utf-8")
        return TrainResult(str(out), str(metrics), "cpu", False)


class _FailingFamily:
    id = "tiny_manifest"

    def available(self) -> bool:
        return True

    def run(self, recipe: dict[str, Any], emit) -> TrainResult:
        del recipe
        emit({"event": "train_start", "family": self.id, "device": "cpu"})
        raise TrainFailed("disk full")


class _InterruptFamily:
    id = "tiny_manifest"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def available(self) -> bool:
        return True

    def run(self, recipe: dict[str, Any], emit) -> TrainResult:
        del recipe
        emit({"event": "train_start", "family": self.id, "device": "cpu"})
        raise self.exc


class TestRunManifest(unittest.TestCase):
    def tearDown(self) -> None:
        set_run_store(None)
        unregister("tiny_manifest")

    def test_schema_file_is_package_data(self) -> None:
        document = schema_document()
        self.assertEqual(document["title"], SCHEMA_ID)
        self.assertEqual(
            document["$id"],
            "https://ohho-robotics.com/schemas/ohho.run/v1.json",
        )
        self.assertTrue(schema_path().is_file())
        errors = validate_instance({"schema": "other"}, document)
        self.assertTrue(any("schema" in error for error in errors))

    def test_recipe_writes_a_valid_manifest_at_start(self) -> None:
        family = _RecordingFamily()
        register(family, source="test")
        saved: list[dict[str, Any]] = []

        class _Store:
            def save(self, manifest: dict[str, Any]) -> None:
                saved.append(manifest)

        set_run_store(_Store())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "nested").mkdir(parents=True)
            (dataset / "nested" / "frames.txt").write_text("frames\n", encoding="utf-8")
            (dataset / "meta.txt").write_bytes(b"meta")
            weights = root / "init.pt"
            weights.write_bytes(b"base-weights-v1")
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "family: tiny_manifest",
                        "dataset:",
                        "  version_id: dv_placeholder",
                        "hyperparameters:",
                        "  seed: 1",
                        "output:",
                        "  dir: placeholder",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            code, events, err = _run(
                recipe,
                [
                    "dataset.version_id=null",
                    f"dataset.path={dataset}",
                    f"output.dir={out}",
                    "hyperparameters.seed=7",
                    "hyperparameters.eval_seed=11",
                    f"base_weights.uri={weights}",
                    "base_weights.revision=rev-a",
                ],
            )
            self.assertEqual(code, EXIT_OK, err)
            self.assertFalse(family.existed_before_start)
            assert family.mid is not None
            self.assertEqual(family.mid["status"], "running")
            self.assertIsNone(family.mid["finished_at"])
            self.assertEqual(validate_run_manifest(family.mid), [])
            manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertEqual(manifest["schema"], SCHEMA_ID)
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["started_at"], family.mid["started_at"])
            self.assertGreaterEqual(manifest["finished_at"], manifest["started_at"])
            self.assertEqual(manifest["family"], "tiny_manifest")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["seeds"]["recipe"], 7)
            self.assertEqual(manifest["seeds"]["python"], 7)
            self.assertEqual(manifest["seeds"]["extra"]["eval_seed"], 11)
            if importlib.util.find_spec("numpy") is not None:
                self.assertEqual(manifest["seeds"]["numpy"], 7)
            else:
                self.assertIsNone(manifest["seeds"]["numpy"])
            if importlib.util.find_spec("torch") is not None:
                self.assertEqual(manifest["seeds"]["torch"], 7)
            else:
                self.assertIsNone(manifest["seeds"]["torch"])
            self.assertIsNone(manifest["dataset"]["version_id"])
            self.assertEqual(manifest["dataset"]["hash"], hash_tree(dataset))
            self.assertEqual(Path(manifest["dataset"]["path"]), dataset.resolve())
            self.assertEqual(
                manifest["base_weights"]["hash"],
                hashlib.sha256(b"base-weights-v1").hexdigest(),
            )
            self.assertEqual(manifest["base_weights"]["revision"], "rev-a")
            self.assertFalse(manifest["base_weights"]["loaded"])
            self.assertEqual(manifest["final_loss"], 0.25)
            self.assertIn("python", manifest["library_versions"])
            self.assertIn("ohho", manifest["library_versions"])
            self.assertEqual(
                manifest["container_image_digest"], container_image_digest()
            )
            text = format_run_text(manifest)
            for field in CONSOLE_FIELDS:
                self.assertIn(f"{field}:", text)
            shown = io.StringIO()
            with redirect_stdout(shown):
                show_code = cli_main(["run", "show", str(out)])
            self.assertEqual(show_code, 0, shown.getvalue())
            self.assertEqual(shown.getvalue(), text)
            json_out = io.StringIO()
            with redirect_stdout(json_out):
                show_code = cli_main(["run", "show", str(out), "--json"])
            self.assertEqual(show_code, 0)
            self.assertEqual(json.loads(json_out.getvalue()), manifest)
            self.assertGreaterEqual(len(saved), 2)
            self.assertEqual(saved[0]["status"], "running")
            self.assertEqual(saved[-1]["status"], "succeeded")
            self.assertTrue(any(event.get("status") == "running" for event in events))
            broken = dict(manifest)
            broken["status"] = "running"
            broken["finished_at"] = manifest["finished_at"]
            self.assertTrue(validate_run_manifest(broken))
            missing = dict(manifest)
            del missing["seeds"]
            self.assertTrue(validate_run_manifest(missing))

    def test_dirty_working_tree_is_flagged(self) -> None:
        family = _RecordingFamily()
        register(family, source="test")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            _init_repo(repo)
            recipe = repo / "recipe.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "family: tiny_manifest",
                        "dataset:",
                        "  version_id: dv_git",
                        "output:",
                        "  dir: placeholder",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            added = _git(repo, "add", "--", "recipe.yaml")
            self.assertEqual(added.returncode, 0, added.stderr)
            committed = _git(repo, "commit", "-m", "start")
            self.assertEqual(committed.returncode, 0, committed.stderr)
            clean_out = base / "clean"
            code, _events, err = _run(recipe, [f"output.dir={clean_out}"])
            self.assertEqual(code, EXIT_OK, err)
            clean = json.loads((clean_out / MANIFEST_NAME).read_text(encoding="utf-8"))
            head = _git(repo, "rev-parse", "HEAD")
            self.assertEqual(head.returncode, 0, head.stderr)
            self.assertEqual(clean["git"]["commit"], head.stdout.strip())
            self.assertFalse(clean["git"]["dirty"])
            self.assertEqual(Path(clean["git"]["root"]), repo.resolve())
            self.assertEqual(clean["dataset"]["version_id"], "dv_git")
            self.assertIsNone(clean["dataset"]["hash"])
            self.assertIsNone(clean["dataset"]["path"])
            (repo / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
            self.assertTrue(git_state(repo)["dirty"])
            dirty_out = base / "dirty"
            code, _events, err = _run(recipe, [f"output.dir={dirty_out}"])
            self.assertEqual(code, EXIT_OK, err)
            dirty = json.loads((dirty_out / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(validate_run_manifest(dirty), [])
            self.assertEqual(dirty["git"]["commit"], head.stdout.strip())
            self.assertTrue(dirty["git"]["dirty"])
            shown = io.StringIO()
            with redirect_stdout(shown):
                self.assertEqual(cli_main(["run", "show", str(dirty_out)]), 0)
            self.assertIn("git.dirty: true", shown.getvalue())

    def test_failure_and_interrupt_finalise_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "family: tiny_manifest",
                        "dataset:",
                        "  version_id: dv_fail",
                        "output:",
                        "  dir: placeholder",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            register(_FailingFamily(), source="test")
            failed_out = root / "failed"
            code, events, err = _run(recipe, [f"output.dir={failed_out}"])
            self.assertEqual(code, EXIT_TRAIN_FAILED, err)
            self.assertIn("disk full", err)
            self.assertFalse(any(event.get("event") == "done" for event in events))
            failed = json.loads(
                (failed_out / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(failed["status"], "failed")
            self.assertIn("disk full", failed["error"])
            self.assertEqual(validate_run_manifest(failed), [])
            unregister("tiny_manifest")
            register(_InterruptFamily(KeyboardInterrupt()), source="test")
            interrupted_out = root / "interrupted"
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    execute(str(recipe), [f"output.dir={interrupted_out}"])
            interrupted = json.loads(
                (interrupted_out / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["error"], "interrupted")
            self.assertIsNotNone(interrupted["finished_at"])
            self.assertEqual(validate_run_manifest(interrupted), [])
            unregister("tiny_manifest")
            register(_InterruptFamily(SystemExit(7)), source="test")
            exited_out = root / "exited"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    execute(str(recipe), [f"output.dir={exited_out}"])
            self.assertEqual(caught.exception.code, 7)
            exited = json.loads(
                (exited_out / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(exited["status"], "interrupted")
            self.assertEqual(validate_run_manifest(exited), [])

    def test_store_hook_failure_does_not_fail_the_run(self) -> None:
        register(_RecordingFamily(), source="test")

        def boom(_manifest: dict[str, Any]) -> None:
            raise RuntimeError("store down")

        set_run_store(boom)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "\n".join(
                    [
                        "schema: ohho.recipe/v1",
                        "family: tiny_manifest",
                        "dataset:",
                        "  version_id: dv_store",
                        "output:",
                        "  dir: placeholder",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            out = root / "out"
            code, _events, err = _run(recipe, [f"output.dir={out}"])
            self.assertEqual(code, EXIT_OK, err)
            self.assertIn("run store hook failed", err)
            manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(validate_run_manifest(manifest), [])

    def test_environment_probes(self) -> None:
        self.assertIsNone(hash_base_weights("https://example.com/weights.pt"))
        self.assertIsNone(hash_base_weights("hf://org/model"))
        self.assertIsNone(hash_base_weights("local:act-init"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "w.bin"
            weight.write_bytes(b"abc")
            self.assertEqual(
                hash_base_weights(str(weight)), hashlib.sha256(b"abc").hexdigest()
            )
            (root / "dir").mkdir()
            (root / "dir" / "a.txt").write_text("a", encoding="utf-8")
            self.assertEqual(hash_tree(root / "dir"), hash_tree(root / "dir"))
            outside = root / "not-a-repo"
            outside.mkdir()
            bare = git_state(outside)
            self.assertIsNone(bare["commit"])
            self.assertFalse(bare["dirty"])
            self.assertIsNone(bare["root"])
        with (
            patch("ohho.train.run_manifest.in_container", return_value=False),
            patch.dict(os.environ, {"OHHO_IMAGE_DIGEST": "sha256:abc"}),
        ):
            self.assertIsNone(container_image_digest())
        with (
            patch("ohho.train.run_manifest.in_container", return_value=True),
            patch.dict(os.environ, {"OHHO_IMAGE_DIGEST": "sha256:abc"}),
        ):
            self.assertEqual(container_image_digest(), "sha256:abc")
        with (
            patch("ohho.train.run_manifest._cuda_from_torch", return_value=None),
            patch("ohho.train.run_manifest._cuda_from_nvidia_smi", return_value=None),
        ):
            self.assertIsNone(gpu_snapshot())
        with (
            patch(
                "ohho.train.run_manifest._cuda_from_torch",
                return_value={"name": "Test GPU", "driver": None, "cuda": "12.4"},
            ),
            patch(
                "ohho.train.run_manifest._cuda_from_nvidia_smi",
                return_value={"name": "Test GPU", "driver": "560.1", "cuda": None},
            ),
        ):
            gpu = gpu_snapshot()
            assert gpu is not None
            self.assertEqual(gpu["name"], "Test GPU")
            self.assertEqual(gpu["driver"], "560.1")
            self.assertEqual(gpu["cuda"], "12.4")
        source = Path(schema_path().parent.parent / "run_manifest.py").read_text(
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
        self.assertIsInstance(in_container(), bool)

    def test_show_rejects_a_missing_or_invalid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = cli_main(["run", "show", str(root)])
            self.assertEqual(code, 1)
            self.assertIn("no run manifest", stderr.getvalue())
            bad = root / MANIFEST_NAME
            bad.write_text("{", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = cli_main(["run", "show", str(bad)])
            self.assertEqual(code, 1)
            self.assertIn("could not read", stderr.getvalue())
            bad.write_text(json.dumps({"schema": "ohho.run/v1"}), encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = cli_main(["run", "show", str(root)])
            self.assertEqual(code, 1)
            self.assertIn("not valid", stderr.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli_main(["run"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
