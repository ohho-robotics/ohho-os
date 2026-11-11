"""W&B sweep translation and the native versus wandb backends.

The W&B client is always a fake. These tests do not call the network.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from ohho.cli import build_parser, main
from ohho.train.sweep import STRATEGIES, SweepError, expand
from ohho.train.sweep_agent import (
    _gpu_available,
    execute_train_job,
    main as agent_main,
    parse_agent_args,
    resolve_train_device,
)
from ohho.train.sweep_run import RUN_SCHEMA_ID, format_start
from ohho.train.sweep_wandb import (
    OHHO_TRAIN_AGENT_COMMAND,
    OHHO_TRAIN_AGENT_MODULE,
    WANDB_METHOD,
    WandbApiClient,
    WandbSweepCreated,
    WandbUnavailable,
    _early_terminate,
    _load_wandb,
    format_launch,
    launch_as_dict,
    launch_sweep,
    to_wandb_sweep_config,
    wandb_connection_configured,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "ohho" / "train" / "schemas" / "examples"
_IMPL = (
    _ROOT / "ohho" / "train" / "sweep_wandb.py",
    _ROOT / "ohho" / "train" / "sweep_agent.py",
)
_WANDB_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((_EXAMPLES / name).read_text(encoding="utf-8"))


def _grid_doc() -> dict[str, Any]:
    return {
        "schema": "ohho.sweep/v1",
        "strategy": "grid",
        "dimensions": [
            {"name": "model", "type": "categorical", "values": ["small", "large"]},
        ],
        "objectives": [
            {"name": "val_loss", "direction": "minimize", "primary": True},
        ],
        "recipe": {"dataset": "toy", "task": "push"},
    }


class FakeClient:
    """In-memory W&B client. ``url`` is returned unchanged."""

    def __init__(self, url: str | None, sweep_id: str = "sweep-from-client") -> None:
        self.url = url
        self.sweep_id = sweep_id
        self.calls: list[dict[str, Any]] = []
        self.mutate = False

    def create(
        self,
        config: Mapping[str, Any],
        *,
        entity: str | None,
        project: str,
    ) -> WandbSweepCreated:
        if self.mutate:
            config["method"] = "mutated"  # type: ignore[index]
            config["url"] = "https://evil.example/not-the-sweep"  # type: ignore[index]
        self.calls.append(
            {
                "config": copy.deepcopy(dict(config)),
                "entity": entity,
                "project": project,
            }
        )
        returned = "" if self.url is None else self.url
        return WandbSweepCreated(self.sweep_id, returned)


class FakeSweep:
    def __init__(self, url: Any) -> None:
        self.url = url


class FakeApi:
    def __init__(self, url: Any) -> None:
        self.url = url
        self.idents: list[str] = []

    def sweep(self, ident: str) -> FakeSweep:
        self.idents.append(ident)
        return FakeSweep(self.url)


class FakeWandb:
    def __init__(self, url: Any, sweep_id: str = "abc") -> None:
        self.url = url
        self.sweep_id = sweep_id
        self.api = FakeApi(url)
        self.configs: list[dict[str, Any]] = []

    def sweep(
        self,
        config: Mapping[str, Any],
        entity: str | None = None,
        project: str | None = None,
    ) -> str:
        self.configs.append(copy.deepcopy(dict(config)))
        self.entity = entity
        self.project = project
        return self.sweep_id

    def Api(self) -> FakeApi:
        return self.api


class TestTranslation(unittest.TestCase):
    def test_strategies_cover_the_sweep_schema(self) -> None:
        self.assertEqual(set(WANDB_METHOD), set(STRATEGIES))

    def test_grid_example_maps_method_metric_and_levels(self) -> None:
        doc = _load("grid_2x3x4.json")
        original = copy.deepcopy(doc)
        translation = to_wandb_sweep_config(doc)
        self.assertEqual(doc, original)
        config = translation.config
        self.assertEqual(config["method"], "grid")
        self.assertEqual(config["metric"], {"name": "val_loss", "goal": "minimize"})
        self.assertEqual(config["command"], list(OHHO_TRAIN_AGENT_COMMAND))
        self.assertIn(OHHO_TRAIN_AGENT_MODULE, config["command"])
        self.assertNotIn("train.py", " ".join(config["command"]))
        plan = expand(doc)
        lrs = [trial.overrides["optimizer.lr"] for trial in plan.trials[:4]]
        self.assertEqual(config["parameters"]["optimizer.lr"]["values"], lrs)
        self.assertEqual(config["parameters"]["model"]["values"], ["small", "large"])
        self.assertEqual(
            config["parameters"]["train.batch_size"]["values"], [8, 16, 32]
        )
        self.assertEqual(config["parameters"]["task"], {"value": "push"})
        self.assertEqual(config["parameters"]["optimizer.weight_decay"], {"value": 0.0})
        self.assertIsNone(translation.early_terminate)
        self.assertNotIn("early_terminate", config)
        self.assertNotIn("url", config)
        self.assertEqual(translation.agent_count, 100)
        notes = "\n".join(translation.notes)
        self.assertIn("TODO(OHH-520)", notes)
        self.assertIn("Hyperband", notes)
        self.assertIn("patience=5", notes)
        self.assertIn("parallelism is not a W&B field", notes)
        self.assertIn("budget.max_gpu_hours is not sent to W&B", notes)
        again = to_wandb_sweep_config(doc)
        self.assertEqual(again.config, translation.config)
        self.assertEqual(again.notes, translation.notes)

    def test_continuous_and_stepped_parameters(self) -> None:
        tpe = to_wandb_sweep_config(_load("tpe.json"))
        self.assertEqual(tpe.config["method"], "bayes")
        self.assertEqual(
            tpe.config["parameters"]["x"],
            {"distribution": "uniform", "min": 0.0, "max": 1.0},
        )
        self.assertIn("built-in TPE", "\n".join(tpe.notes))
        self.assertIn("the sweep seed is not part of the W&B config", tpe.notes)

        multi = to_wandb_sweep_config(_load("multi.json"))
        self.assertEqual(multi.config["method"], "bayes")
        self.assertNotIn("metric", multi.config)
        self.assertEqual(
            multi.config["parameters"]["lr"],
            {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
        )
        self.assertEqual(
            multi.config["parameters"]["batch_size"]["values"],
            [4, 8, 12, 16, 20, 24, 28, 32],
        )
        self.assertIn("no primary objective", "\n".join(multi.notes))

        stepped = to_wandb_sweep_config(
            {
                "schema": "ohho.sweep/v1",
                "strategy": "random",
                "budget": {"max_trials": 3},
                "dimensions": [
                    {"name": "a", "type": "int", "low": 0, "high": 5},
                    {"name": "b", "type": "int", "low": 2, "high": 10, "step": 4},
                    {
                        "name": "p",
                        "type": "float",
                        "low": 0,
                        "high": 1,
                        "n": 3,
                    },
                ],
            }
        )
        self.assertEqual(
            stepped.config["parameters"]["a"],
            {"distribution": "int_uniform", "min": 0, "max": 5},
        )
        self.assertEqual(stepped.config["parameters"]["b"]["values"], [2, 6, 10])
        self.assertEqual(stepped.config["parameters"]["p"]["values"], [0.0, 0.5, 1.0])

    def test_conditionals_constraints_and_non_scalars_are_notes(self) -> None:
        conditional = to_wandb_sweep_config(_load("conditional.json"))
        self.assertIn("model.lora.rank", conditional.config["parameters"])
        self.assertIn("lora_rank", "\n".join(conditional.notes))
        self.assertIn("conditional (when)", "\n".join(conditional.notes))

        constrained = to_wandb_sweep_config(_load("constrained.json"))
        self.assertIn("not pruned", "\n".join(constrained.notes))

        translated = to_wandb_sweep_config(
            {
                "schema": "ohho.sweep/v1",
                "recipe": {"name": "x", "flag": True, "tags": ["a"], "skip": None},
            }
        )
        self.assertEqual(translated.config["parameters"]["name"], {"value": "x"})
        self.assertIs(translated.config["parameters"]["flag"]["value"], True)
        self.assertNotIn("tags", translated.config["parameters"])
        self.assertNotIn("skip", translated.config["parameters"])
        notes = "\n".join(translated.notes)
        self.assertIn("recipe field tags", notes)
        self.assertIn("recipe field skip", notes)

    def test_early_terminate_helper_stays_unmapped(self) -> None:
        spec_doc = _load("grid_2x3x4.json")
        translation = to_wandb_sweep_config(spec_doc)
        self.assertIsNone(translation.early_terminate)
        source = Path(_early_terminate.__code__.co_filename).read_text(encoding="utf-8")
        self.assertIn("TODO(OHH-520)", source)
        self.assertIn("def _early_terminate", source)

    def test_returned_config_is_independent(self) -> None:
        doc = _grid_doc()
        first = to_wandb_sweep_config(doc)
        first.config["method"] = "changed"
        second = to_wandb_sweep_config(doc)
        self.assertEqual(second.config["method"], "grid")

    def test_config_keys_are_wandb_fields(self) -> None:
        config = to_wandb_sweep_config(_grid_doc()).config
        allowed = {
            "program",
            "command",
            "method",
            "parameters",
            "description",
            "metric",
            "early_terminate",
        }
        self.assertLessEqual(set(config), allowed)
        self.assertNotIn("://", json.dumps(config))


class TestNoFabricatedUrl(unittest.TestCase):
    def test_source_does_not_build_a_sweep_url(self) -> None:
        for path in _IMPL:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("wandb.ai", text, path.name)
            self.assertNotIn("https://", text, path.name)
            self.assertNotIn("http://", text, path.name)
            tree = ast.parse(text)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                    self.assertNotIn("wandb", roots, path.name)
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotEqual(node.module.split(".")[0], "wandb", path.name)
        loader = Path(sweep_wandb_source()).read_text(encoding="utf-8")
        self.assertIn("import wandb", loader)

    def test_import_does_not_load_wandb(self) -> None:
        code = (
            "import ohho.train.sweep_agent\n"
            "import ohho.train.sweep_wandb\n"
            "import sys\n"
            "raise SystemExit(0 if 'wandb' not in sys.modules else 1)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_native_result_has_no_url(self) -> None:
        client = FakeClient("https://example.test/sweeps/should-not-be-used")
        with tempfile.TemporaryDirectory() as tmp:
            result = launch_sweep(
                _grid_doc(),
                backend="native",
                sweep_id="local-run",
                root=tmp,
                project="ohho-train",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
        self.assertEqual(client.calls, [])
        self.assertIsNone(result.url)
        self.assertEqual(result.backend, "native")
        self.assertEqual(result.sweep_id, "local-run")
        text = format_launch(result)
        self.assertEqual(text, format_start(result.run))  # type: ignore[arg-type]
        self.assertNotIn("://", text)
        self.assertNotIn("url", text)
        payload = launch_as_dict(result)
        self.assertNotIn("url", payload)
        self.assertEqual(payload["schema"], RUN_SCHEMA_ID)
        self.assertNotIn("test-key-not-real", json.dumps(payload))

    def test_client_url_is_returned_unchanged(self) -> None:
        unusual = "https://wandb.example.test/org/proj/sweeps/real-id"
        public = "https://wandb.ai/org/proj/sweeps/abc"
        for url in (unusual, public):
            client = FakeClient(url, sweep_id="from-api")
            result = launch_sweep(
                _grid_doc(),
                backend="wandb",
                project="ohho-train",
                entity="ohho",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
            self.assertEqual(result.url, url)
            self.assertEqual(result.sweep_id, "from-api")
            self.assertEqual(len(client.calls), 1)
            sent = client.calls[0]["config"]
            self.assertEqual(sent["method"], "grid")
            self.assertEqual(sent["command"], list(OHHO_TRAIN_AGENT_COMMAND))
            self.assertNotIn("url", sent)
            self.assertNotIn("test-key-not-real", json.dumps(sent))
            text = format_launch(result)
            self.assertEqual(text.count("://"), 1)
            self.assertIn(url, text)
            self.assertEqual(launch_as_dict(result)["url"], url)

    def test_non_absolute_client_url_is_not_replaced(self) -> None:
        for url in (None, "", "abc123", "org/proj/sweeps/abc"):
            client = FakeClient(url)
            with self.assertRaises(SweepError) as caught:
                launch_sweep(
                    _grid_doc(),
                    backend="wandb",
                    project="ohho-train",
                    api_key="test-key-not-real",
                    environ={},
                    client=client,
                )
            message = str(caught.exception)
            self.assertNotIn("://", message)
            self.assertNotIn("wandb.ai", message)
            self.assertIn("absolute sweep URL", message)
            self.assertEqual(len(client.calls), 1)

    def test_mutated_client_config_does_not_become_the_url(self) -> None:
        client = FakeClient("https://example.test/sweeps/real")
        client.mutate = True
        result = launch_sweep(
            _grid_doc(),
            backend="wandb",
            project="ohho-train",
            api_key="test-key-not-real",
            environ={},
            client=client,
        )
        self.assertEqual(result.url, "https://example.test/sweeps/real")
        self.assertEqual(result.config["method"], "grid")  # type: ignore[index]
        self.assertNotIn("url", result.config)

    def test_without_a_key_the_client_is_not_called(self) -> None:
        client = FakeClient("https://example.test/sweeps/nope")
        with tempfile.TemporaryDirectory() as tmp:
            auto = launch_sweep(
                _grid_doc(),
                backend="auto",
                root=tmp,
                project="ohho-train",
                environ={},
                client=client,
            )
            self.assertEqual(auto.backend, "native")
            self.assertIsNone(auto.url)
            self.assertIn("not configured", " ".join(auto.notes))
            with self.assertRaises(SweepError) as caught:
                launch_sweep(
                    _grid_doc(),
                    backend="wandb",
                    root=tmp,
                    project="ohho-train",
                    environ={},
                    client=client,
                )
        self.assertEqual(client.calls, [])
        self.assertIn("WANDB_API_KEY", str(caught.exception))
        self.assertNotIn("://", str(caught.exception))
        self.assertFalse(wandb_connection_configured(environ={}))

    def test_offline_mode_does_not_call_the_client(self) -> None:
        client = FakeClient("https://example.test/sweeps/offline")
        with self.assertRaises(SweepError) as caught:
            launch_sweep(
                _grid_doc(),
                backend="wandb",
                project="ohho-train",
                api_key="test-key-not-real",
                environ={"WANDB_MODE": "offline"},
                client=client,
            )
        self.assertEqual(client.calls, [])
        self.assertIn("offline", str(caught.exception))
        self.assertNotIn("://", str(caught.exception))
        self.assertFalse(
            wandb_connection_configured(
                api_key="test-key-not-real",
                environ={"WANDB_MODE": "disabled"},
            )
        )

    def test_missing_project_and_local_id_do_not_call_the_client(self) -> None:
        client = FakeClient("https://example.test/sweeps/nope")
        with self.assertRaises(SweepError) as missing_project:
            launch_sweep(
                _grid_doc(),
                backend="wandb",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
        self.assertIn("project", str(missing_project.exception))
        with self.assertRaises(SweepError) as local_id:
            launch_sweep(
                _grid_doc(),
                backend="wandb",
                sweep_id="local",
                project="ohho-train",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
        self.assertIn("local sweep id", str(local_id.exception))
        self.assertEqual(client.calls, [])

    def test_multi_without_primary_does_not_call_the_client(self) -> None:
        client = FakeClient("https://example.test/sweeps/multi")
        with self.assertRaises(SweepError) as caught:
            launch_sweep(
                _load("multi.json"),
                backend="wandb",
                project="ohho-train",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
        self.assertEqual(client.calls, [])
        self.assertIn("primary", str(caught.exception))
        self.assertNotIn("://", str(caught.exception))

    def test_wandb_launch_does_not_write_a_native_run(self) -> None:
        client = FakeClient("https://example.test/sweeps/only")
        with tempfile.TemporaryDirectory() as tmp:
            launch_sweep(
                _grid_doc(),
                backend="wandb",
                root=tmp,
                project="ohho-train",
                api_key="test-key-not-real",
                environ={},
                client=client,
            )
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_api_client_uses_the_object_url_only(self) -> None:
        previous = os.environ.get("WANDB_API_KEY")
        fake = FakeWandb("https://example.test/sweeps/from-object", sweep_id="abc")
        try:
            created = WandbApiClient().create(
                {"method": "grid", "parameters": {}},
                entity="ent",
                project="proj",
                api_key="sekret-value",
                wandb_module=fake,
            )
            self.assertEqual(os.environ.get("WANDB_API_KEY"), previous)
        finally:
            if previous is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = previous
        self.assertEqual(created.url, "https://example.test/sweeps/from-object")
        self.assertEqual(created.sweep_id, "abc")
        self.assertEqual(fake.api.idents, ["ent/proj/abc"])
        self.assertNotIn("://", fake.api.idents[0])
        self.assertNotIn("sekret-value", json.dumps(fake.configs))

        missing = FakeWandb(None, sweep_id="abc")
        with self.assertRaises(SweepError) as caught:
            WandbApiClient().create(
                {"method": "grid"},
                entity=None,
                project="proj",
                wandb_module=missing,
            )
        self.assertNotIn("://", str(caught.exception))
        self.assertNotIn("wandb.ai", str(caught.exception))
        self.assertEqual(missing.api.idents, ["proj/abc"])

    def test_missing_package_does_not_invent_a_url(self) -> None:
        def importer(name: str) -> Any:
            raise ImportError(name)

        with self.assertRaises(WandbUnavailable) as caught:
            _load_wandb(None, importer=importer)
        self.assertIn("ohho-os[wandb]", str(caught.exception))
        self.assertNotIn("://", str(caught.exception))

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WandbUnavailable) as explicit:
                launch_sweep(
                    _grid_doc(),
                    backend="wandb",
                    sweep_id="local",
                    root=tmp,
                    project="ohho-train",
                    api_key="test-key-not-real",
                    environ={},
                    importer=importer,
                )
            self.assertNotIn("://", str(explicit.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            auto = launch_sweep(
                _grid_doc(),
                backend="auto",
                sweep_id="pkg-missing",
                root=tmp,
                api_key="test-key-not-real",
                environ={},
                importer=importer,
            )
        self.assertEqual(auto.backend, "native")
        self.assertIsNone(auto.url)
        self.assertEqual(auto.sweep_id, "pkg-missing")
        self.assertIn("not installed", " ".join(auto.notes))
        self.assertNotIn("://", " ".join(auto.notes))


class TestAgent(unittest.TestCase):
    def test_parse_wandb_args(self) -> None:
        parsed = parse_agent_args(
            [
                "--dataset=toy",
                "--optimizer.lr=0.01",
                "--train.batch_size",
                "8",
                "--lr=2",
                "--mock",
            ]
        )
        self.assertEqual(parsed["dataset"], "toy")
        self.assertEqual(parsed["optimizer.lr"], 0.01)
        self.assertEqual(parsed["train.batch_size"], 8)
        self.assertIs(parsed["mock"], True)
        self.assertEqual(parsed["lr"], 2)
        nested = parse_agent_args(
            ['{"optimizer": {"lr": 0.02}, "dataset": "toy"}', "--lr=3"]
        )
        self.assertEqual(nested["optimizer.lr"], 0.02)
        self.assertEqual(nested["lr"], 3)
        with self.assertRaises(SweepError):
            parse_agent_args(["hello"])

    def test_train_job_calls_finetune_fields(self) -> None:
        seen: dict[str, Any] = {}

        def trainer(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "ckpt-dir"

        result = execute_train_job(
            {
                "dataset": "toy",
                "optimizer": {"lr": 0.01},
                "train": {"batch_size": 8},
                "model": "small",
            },
            trainer=trainer,
        )
        self.assertEqual(result.job, "ohho.train")
        self.assertEqual(result.checkpoint, "ckpt-dir")
        self.assertEqual(seen["dataset"], "toy")
        self.assertEqual(seen["lr"], 0.01)
        self.assertEqual(seen["batch_size"], 8)
        self.assertEqual(seen["device"], "cpu")
        self.assertNotIn("model", seen)
        self.assertEqual(result.overrides["model"], "small")

    def test_gpu_and_tensorrt_fail_before_training(self) -> None:
        def trainer(**kwargs: Any) -> str:
            raise AssertionError(kwargs)

        for device in ("cuda", "cuda:0", "tensorrt", "TensorRT", "mps"):
            with self.assertRaises(Exception) as caught:
                execute_train_job(
                    {"dataset": "toy", "device": device},
                    trainer=trainer,
                )
            message = str(caught.exception)
            self.assertIn("No training was started", message)
            self.assertIn("GPU and TensorRT", message)
            self.assertNotIn("://", message)
        self.assertFalse(_gpu_available("cuda"))
        self.assertFalse(_gpu_available("tensorrt"))
        self.assertEqual(resolve_train_device("auto"), "cpu")
        self.assertEqual(
            resolve_train_device("cuda", available=lambda _device: True),
            "cuda",
        )

    def test_main_runs_one_job_and_refuses_gpu(self) -> None:
        def trainer(**kwargs: Any) -> str:
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["dataset"], "toy")
            return "ckpt-dir"

        out = io.StringIO()
        with redirect_stdout(out):
            code = agent_main(["--dataset=toy", "--optimizer.lr=1e-4"], trainer=trainer)
        self.assertEqual(code, 0)
        self.assertIn("job: ohho.train", out.getvalue())
        self.assertIn("checkpoint: ckpt-dir", out.getvalue())
        self.assertNotIn("://", out.getvalue())

        err = io.StringIO()
        with redirect_stderr(err):
            code = agent_main(["--device=cuda", "--dataset=toy"])
        self.assertEqual(code, 1)
        self.assertIn("GPU and TensorRT", err.getvalue())
        self.assertNotIn("://", err.getvalue())

        proc = subprocess.run(
            [sys.executable, "-m", "ohho.train.sweep_agent", "--device", "tensorrt"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("GPU and TensorRT", proc.stderr)
        self.assertNotIn("://", proc.stderr)
        self.assertNotIn("://", proc.stdout)


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = {key: os.environ.get(key) for key in _WANDB_KEYS}
        for key in _WANDB_KEYS:
            os.environ.pop(key, None)
        self._old_root = os.environ.get("OHHO_SWEEP_DIR")
        os.environ["OHHO_SWEEP_DIR"] = self._tmp.name

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("OHHO_SWEEP_DIR", None)
        else:
            os.environ["OHHO_SWEEP_DIR"] = self._old_root
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_parser_defaults_to_native(self) -> None:
        args = build_parser().parse_args(["sweep", "start", "sweep.yaml"])
        self.assertEqual(args.backend, "native")
        self.assertIsNone(args.wandb_project)

    def test_native_start_prints_no_url(self) -> None:
        spec = Path(self._tmp.name) / "spec.json"
        spec.write_text(json.dumps(_grid_doc()), encoding="utf-8")
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["sweep", "start", str(spec), "--id", "cli-native"])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("id: cli-native", out.getvalue())
        self.assertNotIn("://", out.getvalue())
        self.assertNotIn("url", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_wandb_without_a_key_prints_no_url(self) -> None:
        spec = Path(self._tmp.name) / "spec.json"
        spec.write_text(json.dumps(_grid_doc()), encoding="utf-8")
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(
                [
                    "sweep",
                    "start",
                    str(spec),
                    "--backend",
                    "wandb",
                    "--wandb-project",
                    "ohho-train",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("WANDB_API_KEY", err.getvalue())
        self.assertNotIn("://", out.getvalue() + err.getvalue())
        self.assertNotIn("wandb.ai", out.getvalue() + err.getvalue())

    def test_auto_without_a_key_stays_native(self) -> None:
        spec = Path(self._tmp.name) / "spec.json"
        spec.write_text(json.dumps(_grid_doc()), encoding="utf-8")
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(
                ["sweep", "start", str(spec), "--backend", "auto", "--id", "auto-run"]
            )
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("id: auto-run", out.getvalue())
        self.assertNotIn("://", out.getvalue())
        self.assertIn("not configured", err.getvalue())


def sweep_wandb_source() -> str:
    return str(_ROOT / "ohho" / "train" / "sweep_wandb.py")


class TestLaunchRejectsUnknownBackend(unittest.TestCase):
    def test_unknown_backend(self) -> None:
        with self.assertRaises(SweepError):
            launch_sweep(_grid_doc(), backend="other", environ={})


if __name__ == "__main__":
    unittest.main()
