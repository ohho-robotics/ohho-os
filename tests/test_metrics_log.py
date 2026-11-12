"""OHH-544: ohho.log, the native metrics store, and the optional W&B mirror."""

from __future__ import annotations

import ast
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho import log
from ohho.metrics import (
    SCHEMA_ID,
    MetricsError,
    _flush_open_runs,
    clear_last_metrics_run,
    log_path,
    open_metrics,
    read_metrics,
    reset_metrics_state,
    set_metrics_uploader,
    set_wandb_client,
    summary_path,
    wandb_configured,
)
from ohho.train.act import train_act
from ohho.train.recipe_registry import TrainResult, register, unregister
from ohho.train.run_cli import EXIT_OK, execute
from ohho.train.run_manifest import (
    MANIFEST_NAME,
    load_run_manifest,
    validate_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
LINK = "https://wandb.example/runs/abc"


def _quiet_env() -> dict[str, str]:
    return {"WANDB_API_KEY": "", "WANDB_MODE": "", "WANDB_DISABLED": ""}


class _Run:
    def __init__(self, url: str | None = LINK) -> None:
        self.url = url
        self.config: dict[str, str] = {}
        self.logged: list[tuple[int | None, dict[str, object]]] = []
        self.finished = False

    def log(self, data: dict[str, object], step: int | None = None) -> None:
        self.logged.append((step, dict(data)))

    def finish(self) -> None:
        self.finished = True

    def get_url(self) -> str | None:
        return self.url


class _Client:
    def __init__(self, run: _Run | None = None) -> None:
        self.run = run or _Run()
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> _Run:
        self.init_kwargs = kwargs
        config = kwargs.get("config")
        if isinstance(config, dict):
            self.run.config.update(config)
        return self.run


class _LateUrl(_Run):
    def __init__(self) -> None:
        super().__init__(url=None)

    def finish(self) -> None:
        super().finish()
        self.url = "https://wandb.example/runs/late"


class _GetUrlOnly:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.logged: list[tuple[int | None, dict[str, object]]] = []

    def log(self, data: dict[str, object], step: int | None = None) -> None:
        self.logged.append((step, dict(data)))

    def finish(self) -> None:
        return None

    def get_url(self) -> str:
        return "https://wandb.example/runs/from-get"


class _LoggingFamily:
    id = "tiny_metrics"

    def available(self) -> bool:
        return True

    def run(self, recipe: dict[str, object], emit) -> TrainResult:
        out = Path(str(recipe["output"]["dir"]))  # type: ignore[index]
        emit({"event": "train_start", "family": self.id, "device": "cpu"})
        with open_metrics(out, buffer_size=10):
            log(0, {"loss": 0.5})
            log(1, {"loss": 0.25})
        return TrainResult(str(out), str(summary_path(out)), "cpu", False)


def _recipe(root: Path) -> Path:
    path = root / "recipe.yaml"
    path.write_text(
        "\n".join(
            [
                "schema: ohho.recipe/v1",
                "family: tiny_metrics",
                "dataset:",
                "  version_id: dv_metrics",
                "output:",
                "  dir: placeholder",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _wandb_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "wandb" or alias.name.startswith("wandb."):
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "wandb" or node.module.startswith("wandb."):
                found.append(f"from {node.module}")
        elif isinstance(node, ast.Name) and node.id == "wandb":
            found.append("name")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "wandb"
        ):
            found.append(f"wandb.{node.attr}")
    return found


def _calls(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


class TestMetricsLog(unittest.TestCase):
    def setUp(self) -> None:
        reset_metrics_state()
        unregister("tiny_metrics")
        self._env = patch.dict(os.environ, _quiet_env(), clear=False)
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        reset_metrics_state()
        unregister("tiny_metrics")
        clear_last_metrics_run()

    def test_public_log_is_the_metrics_function(self) -> None:
        from ohho import metrics

        self.assertIs(log, metrics.log)
        self.assertIn("log", __import__("ohho").__all__)

    def test_log_without_a_run_raises(self) -> None:
        with self.assertRaises(MetricsError):
            log(0, {"loss": 1.0})

    def test_buffer_flushes_on_close_and_when_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open_metrics(root, buffer_size=2) as session:
                log(0, {"loss": 1.0})
                self.assertFalse(session.path.exists())
                log(1, {"loss": 0.5})
                self.assertEqual(
                    session.path.read_text(encoding="utf-8").count("\n"), 2
                )
                held = session.path.read_text(encoding="utf-8")
                log(2, {"loss": 0.25})
                self.assertEqual(session.path.read_text(encoding="utf-8"), held)
            text = session.path.read_text(encoding="utf-8")
            self.assertGreater(text.count("\n"), held.count("\n"))
            document = read_metrics(root)
            self.assertEqual(document["schema"], SCHEMA_ID)
            self.assertEqual(document["run_id"], session.run_id)
            self.assertEqual([row["step"] for row in document["records"]], [0, 1, 2])
            self.assertEqual(document["series"]["loss"][0], {"step": 0, "value": 1.0})
            self.assertEqual(document["final_loss"], 0.25)
            self.assertEqual(document["loss_curve"], [[0, 1.0], [1, 0.5], [2, 0.25]])
            summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["final_loss"], document["final_loss"])
            self.assertEqual(summary["records"], document["records"])

    def test_append_only_and_exit_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open_metrics(root, buffer_size=1) as first:
                log(0, {"loss": 1.0, "note": "kept"})
                run_id = first.run_id
            original = log_path(root).read_text(encoding="utf-8")
            with open_metrics(root, buffer_size=1) as second:
                self.assertEqual(second.run_id, run_id)
                log(1, {"loss": 0.5})
            updated = log_path(root).read_text(encoding="utf-8")
            self.assertTrue(updated.startswith(original))
            self.assertGreater(len(updated), len(original))
            document = read_metrics(root)
            self.assertEqual(document["latest"]["note"], "kept")
            self.assertNotIn("note", document["series"])
            with self.assertRaises(MetricsError):
                open_metrics(root, run_id="ohho-other").__enter__()

            other = root / "exit"
            session = open_metrics(other, buffer_size=8)
            active = session.__enter__()
            log(3, {"loss": 0.1})
            self.assertFalse(active.path.exists())
            _flush_open_runs()
            self.assertEqual(read_metrics(other)["records"][0]["step"], 3)
            session.__exit__(None, None, None)
            self.assertIn(
                "_flush_open_runs",
                (ROOT / "ohho" / "metrics.py").read_text(encoding="utf-8"),
            )

    def test_rejects_bad_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open_metrics(tmp):
                with self.assertRaises(ValueError):
                    log(-1, {"loss": 1})
                with self.assertRaises(ValueError):
                    log(True, {"loss": 1})  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    log(0, {"loss": float("nan")})
                with self.assertRaises(ValueError):
                    log(0, ["loss"])  # type: ignore[arg-type]

    def test_uploader_receives_the_flushed_batch_only(self) -> None:
        seen: list[dict[str, object]] = []

        class _Uploader:
            def upload(self, batch: dict[str, object]) -> None:
                seen.append(batch)

        set_metrics_uploader(_Uploader())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open_metrics(root, buffer_size=8) as session:
                log(0, {"loss": 0.4})
                self.assertEqual(seen, [])
                session.flush()
                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0]["run_id"], session.run_id)
                records = seen[0]["records"]
                self.assertIsInstance(records, list)
                assert isinstance(records, list)
                self.assertEqual(records[0]["metrics"], {"loss": 0.4})
            self.assertTrue(log_path(root).is_file())

        class _Boom:
            def upload(self, batch: dict[str, object]) -> None:
                del batch
                raise RuntimeError("uploader down")

        set_metrics_uploader(_Boom())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = io.StringIO()
            with redirect_stderr(stderr), open_metrics(root, buffer_size=1):
                log(0, {"loss": 0.2})
            self.assertIn("metrics upload failed", stderr.getvalue())
            self.assertEqual(read_metrics(root)["final_loss"], 0.2)
            source = (ROOT / "ohho" / "metrics.py").read_text(encoding="utf-8")
            for banned in ("urllib", "http.client", "socket", "requests"):
                self.assertNotIn(banned, source)

    def test_wandb_stays_off_without_a_configured_client(self) -> None:
        self.assertFalse(wandb_configured())
        imported = patch(
            "ohho.metrics._import_wandb", side_effect=AssertionError("imported")
        )
        with imported as loader, tempfile.TemporaryDirectory() as tmp:
            with open_metrics(tmp) as session:
                log(0, {"loss": 1.0})
            loader.assert_not_called()
            self.assertIsNone(session.wandb_url)
        for mode in ("offline", "disabled", "dryrun"):
            with (
                patch.dict(
                    os.environ,
                    {"WANDB_API_KEY": "test-key", "WANDB_MODE": mode},
                ),
                patch(
                    "ohho.metrics._import_wandb",
                    side_effect=AssertionError("imported"),
                ) as loader,
                tempfile.TemporaryDirectory() as tmp,
            ):
                self.assertFalse(wandb_configured())
                with open_metrics(tmp) as session:
                    log(0, {"loss": 1.0})
                loader.assert_not_called()
                self.assertIsNone(session.wandb_url)
        with (
            patch.dict(
                os.environ,
                {"WANDB_API_KEY": "test-key", "WANDB_DISABLED": "true"},
            ),
            patch(
                "ohho.metrics._import_wandb",
                side_effect=AssertionError("imported"),
            ) as loader,
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp) as session:
                log(0, {"loss": 1.0})
            loader.assert_not_called()
            self.assertIsNone(session.wandb_url)

    def test_wandb_mirror_uses_a_lazy_client_and_does_not_invent_a_url(self) -> None:
        source = (ROOT / "ohho" / "metrics.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("wandb", [alias.name for alias in node.names])
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "wandb")
        self.assertNotIn("wandb.ai", source)
        self.assertNotIn("https://", source)
        self.assertNotIn(LINK, source)

        client = _Client()
        with (
            patch.dict(
                os.environ, {"WANDB_API_KEY": "test-key", "WANDB_MODE": "online"}
            ),
            patch("ohho.metrics._import_wandb", return_value=client) as imported,
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp, buffer_size=8) as session:
                log(4, {"loss": 0.2})
                self.assertEqual(client.run.logged, [])
            imported.assert_called_once()
            self.assertEqual(session.wandb_url, LINK)
            self.assertTrue(client.run.finished)
            self.assertEqual(client.run.logged, [(4, {"loss": 0.2})])
            assert client.init_kwargs is not None
            self.assertEqual(
                client.init_kwargs["config"], {"ohho_run_id": session.run_id}
            )
            self.assertEqual(client.run.config["ohho_run_id"], session.run_id)
            self.assertNotIn("url", client.init_kwargs)
            self.assertNotIn("api_key", client.init_kwargs)
            self.assertNotIn("test-key", json.dumps(client.init_kwargs))

        blank = _Client(_Run(url=None))
        blank.run.get_url = lambda: None  # type: ignore[method-assign]
        set_wandb_client(blank)
        with (
            patch.dict(
                os.environ, {"WANDB_API_KEY": "test-key", "WANDB_MODE": "online"}
            ),
            patch(
                "ohho.metrics._import_wandb", side_effect=AssertionError("imported")
            ) as imported,
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp) as session:
                log(0, {"loss": 0.8})
            imported.assert_not_called()
            self.assertIsNone(session.wandb_url)
            self.assertEqual(blank.run.config["ohho_run_id"], session.run_id)

        whitespace = _Client(_Run(url="  "))
        set_wandb_client(whitespace)
        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp) as session:
                log(0, {"loss": 0.1})
            self.assertIsNone(session.wandb_url)

        late = _Client(_LateUrl())
        set_wandb_client(late)
        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp) as session:
                log(2, {"loss": 0.3})
            self.assertEqual(session.wandb_url, "https://wandb.example/runs/late")

        via_get = _Client()
        via_get.run = _GetUrlOnly()  # type: ignore[assignment]
        set_wandb_client(via_get)
        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            tempfile.TemporaryDirectory() as tmp,
        ):
            with open_metrics(tmp) as session:
                log(0, {"loss": 0.7})
            self.assertEqual(session.wandb_url, "https://wandb.example/runs/from-get")

        set_wandb_client(None)
        missing = _Client()
        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
            patch("ohho.metrics._import_wandb", side_effect=ImportError("no wandb")),
            tempfile.TemporaryDirectory() as tmp,
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr), open_metrics(tmp) as session:
                log(0, {"loss": 0.6})
            self.assertIsNone(session.wandb_url)
            self.assertIn("not installed", stderr.getvalue())
            self.assertEqual(read_metrics(tmp)["final_loss"], 0.6)
            self.assertIsNone(missing.init_kwargs)

    def test_recipes_log_through_the_interface_and_do_not_call_wandb(self) -> None:
        offenders: list[str] = []
        for path in (ROOT / "ohho" / "train").rglob("*.py"):
            if path.name in ("metrics.py", "sweep_wandb.py"):
                continue
            uses = _wandb_uses(path)
            if uses:
                offenders.append(f"{path.relative_to(ROOT)}: {uses}")
        self.assertEqual(offenders, [])
        act = ROOT / "ohho" / "train" / "act.py"
        recipes = ROOT / "ohho" / "train" / "builtin_recipes.py"
        self.assertTrue(_calls(act, "log"))
        self.assertTrue(_calls(act, "open_metrics"))
        self.assertTrue(_calls(recipes, "open_metrics"))
        self.assertNotIn("metrics.json", act.read_text(encoding="utf-8"))
        for family in ("diffusion", "smolvla", "openvla", "ppo"):
            self.assertNotIn(
                "wandb", (ROOT / "ohho" / "train" / "builtin_recipes.py").read_text()
            )
            self.assertIn(family, recipes.read_text(encoding="utf-8").lower())

        calls: list[tuple[int, dict[str, float]]] = []
        real = log

        def spy(step: int, metrics: dict[str, float]) -> None:
            calls.append((step, dict(metrics)))
            real(step, metrics)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            out = root / "out"
            with patch("ohho.train.act.log", spy):
                train_act(str(dataset), str(out), mock=True, train_steps=100)
            self.assertEqual(calls[0], (0, {"loss": 0.8421}))
            self.assertEqual(calls[-1][0], 100)
            self.assertAlmostEqual(calls[-1][1]["loss"], 0.0412)
            document = read_metrics(out)
            self.assertAlmostEqual(document["final_loss"], 0.0412)
            summary = json.loads(summary_path(out).read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["final_loss"], 0.0412)
            self.assertTrue(summary["loss_curve"])
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertNotIn("loss_curve", config)
            self.assertNotIn("final_loss", config)

    def test_manifest_records_a_client_url_and_omits_one_it_does_not_have(self) -> None:
        register(_LoggingFamily(), source="test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "plain"
            code, _events, err = _run(_recipe(root), [f"output.dir={out}"])
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertIsNone(manifest["wandb_url"])
            encoded = json.dumps(manifest)
            self.assertNotIn("http://", encoded)
            self.assertNotIn("https://", encoded)
            self.assertEqual(read_metrics(out)["final_loss"], 0.25)
            stored = dict(manifest)
            del stored["wandb_url"]
            self.assertEqual(validate_run_manifest(stored), [])
            (out / MANIFEST_NAME).write_text(json.dumps(stored), encoding="utf-8")
            loaded = load_run_manifest(out / MANIFEST_NAME)
            self.assertIsNone(loaded.wandb_url)

        client = _Client()
        register(_LoggingFamily(), source="test")
        with (
            patch.dict(
                os.environ, {"WANDB_API_KEY": "test-key", "WANDB_MODE": "online"}
            ),
            patch("ohho.metrics._import_wandb", return_value=client),
            tempfile.TemporaryDirectory() as tmp,
        ):
            root = Path(tmp)
            out = root / "mirrored"
            code, _events, err = _run(_recipe(root), [f"output.dir={out}"])
            self.assertEqual(code, EXIT_OK, err)
            manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(validate_run_manifest(manifest), [])
            self.assertEqual(manifest["wandb_url"], LINK)
            self.assertEqual(
                client.run.config["ohho_run_id"],
                read_metrics(out)["run_id"],
            )
            self.assertEqual(
                client.run.logged, [(0, {"loss": 0.5}), (1, {"loss": 0.25})]
            )


def _run(path: Path, overrides: list[str]) -> tuple[int, list[dict[str, object]], str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
