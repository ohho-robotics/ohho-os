"""OHH-542: deterministic recipes, published variance, and ohho run rerun."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main as cli_main
from ohho.train.determinism import (
    FIXED_DATALOADER_WORKERS,
    dataloader_kwargs,
    deterministic_algorithms,
    ensure_deterministic_seed,
)
from ohho.train.recipe_schema import example_path, load_recipe_file, validate_recipe
from ohho.train.rerun import rerun_cli
from ohho.train.run_cli import EXIT_OK, execute
from ohho.train.run_manifest import MANIFEST_NAME
from ohho.train.variance import (
    CONSOLE_FOLLOWUP,
    all_recipe_noise,
    comparison_hint,
    doc_table_rows,
    recipe_noise,
    within_noise,
)

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "train" / "determinism.md"
_TINY_ACT = (
    Path(__file__).resolve().parents[1]
    / "ohho"
    / "train"
    / "examples"
    / "il"
    / "act-tiny-cpu.yaml"
)


def _run(path: Path, overrides: list[str]):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


class _Flag:
    def __init__(self) -> None:
        self.deterministic = False
        self.benchmark = True
        self.allow_tf32 = True
        self.enabled = True


class _Backends:
    def __init__(self) -> None:
        self.cudnn = _Flag()
        self.mkldnn = _Flag()


class _Generator:
    def __init__(self) -> None:
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = int(seed)


class _Torch:
    def __init__(self) -> None:
        self.backends = _Backends()
        self._enabled = False
        self._warn = False
        self.Generator = _Generator

    def use_deterministic_algorithms(self, mode: bool, warn_only: bool = False) -> None:
        self._enabled = bool(mode)
        self._warn = bool(warn_only)

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self._warn

    def manual_seed(self, seed: int) -> None:
        self.seed = int(seed)


class _TorchStub:
    def manual_seed(self, seed: int) -> None:
        del seed


def _torch_skip_reason() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return (
            "skipped: torch is not installed, so the deterministic tiny ACT "
            "re-run cannot train (pip install 'ohho-os[train]'; lerobot is "
            "only required to download an hf:// dataset)"
        )
    return None


def _write_metrics(output_dir: str, loss: float) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps({"final_loss": loss}),
        encoding="utf-8",
    )
    (out / "policy.pt").write_bytes(b"pt")
    return str(out)


class TestDeterministicFlag(unittest.TestCase):
    def test_schema_default_and_rejection(self) -> None:
        recipe = load_recipe_file(example_path("act"))
        self.assertIs(recipe["deterministic"], False)
        bad = json.loads(json.dumps(recipe))
        bad["deterministic"] = "yes"
        self.assertTrue(validate_recipe(bad))
        good = json.loads(json.dumps(recipe))
        good["deterministic"] = True
        self.assertEqual(validate_recipe(good), [])

    def test_omitted_seed_becomes_zero_only_when_deterministic(self) -> None:
        missing = {"deterministic": True, "hyperparameters": {}}
        self.assertEqual(ensure_deterministic_seed(missing), 0)
        self.assertEqual(missing["hyperparameters"]["seed"], 0)
        kept = {"deterministic": True, "hyperparameters": {"seed": 4}}
        self.assertEqual(ensure_deterministic_seed(kept), 4)
        self.assertEqual(kept["hyperparameters"]["seed"], 4)
        off = {"deterministic": False, "hyperparameters": {}}
        self.assertIsNone(ensure_deterministic_seed(off))
        self.assertNotIn("seed", off["hyperparameters"])

    def test_workers_are_fixed_and_batch_order_is_seeded(self) -> None:
        self.assertEqual(FIXED_DATALOADER_WORKERS, 0)
        torch = _Torch()
        plain = dataloader_kwargs(torch, deterministic=False, seed=3, shuffle=True)
        self.assertEqual(plain, {"shuffle": True})
        seeded = dataloader_kwargs(torch, deterministic=True, seed=3, shuffle=True)
        self.assertEqual(seeded["num_workers"], 0)
        self.assertTrue(seeded["shuffle"])
        self.assertEqual(seeded["generator"].seed, 3)
        held = dataloader_kwargs(torch, deterministic=True, seed=3, shuffle=False)
        self.assertEqual(held, {"shuffle": False, "num_workers": 0})

    def test_algorithm_flags_restore_after_success_and_failure(self) -> None:
        torch = _Torch()
        with deterministic_algorithms(torch, enabled=True) as applied:
            self.assertIs(applied["algorithms"], True)
            self.assertEqual(applied["workers"], 0)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.mkldnn.enabled)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.mkldnn.enabled)
        try:
            with deterministic_algorithms(torch, enabled=True):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        with deterministic_algorithms(torch, enabled=False) as skipped:
            self.assertFalse(skipped["enabled"])
        self.assertFalse(torch.are_deterministic_algorithms_enabled())


class TestPublishedVariance(unittest.TestCase):
    def test_rows_and_within_noise(self) -> None:
        act = recipe_noise("act", deterministic=True)
        self.assertEqual(act.expected_variance, 0.0)
        self.assertEqual(act.tolerance, 0.0)
        self.assertEqual(act.metric, "final_loss")
        self.assertTrue(within_noise("act", 0.25, 0.25, deterministic=True))
        self.assertFalse(within_noise("act", 0.25, 0.2500001, deterministic=True))
        self.assertTrue(within_noise("diffusion", 0.1, 0.1009, deterministic=False))
        self.assertFalse(within_noise("diffusion", 0.1, 0.1011, deterministic=False))
        self.assertFalse(within_noise("smolvla", 0.1, 0.1, deterministic=True))
        hint = comparison_hint("act", 0.2, 0.2, deterministic=True)
        self.assertEqual(hint["console_followup"], CONSOLE_FOLLOWUP)
        self.assertTrue(hint["within_noise"])
        self.assertTrue(hint["exact"])
        names = [row.recipe for row in all_recipe_noise()]
        self.assertEqual(
            names,
            [
                "act",
                "act",
                "diffusion",
                "diffusion",
                "smolvla",
                "smolvla",
                "openvla",
                "openvla",
                "ppo",
                "ppo",
            ],
        )

    def test_doc_matches_the_published_rows(self) -> None:
        text = _DOCS.read_text(encoding="utf-8")
        for row in doc_table_rows():
            self.assertIn(row, text)
        self.assertIn(CONSOLE_FOLLOWUP, text)
        self.assertIn("ohho run rerun", text)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", text)
        self.assertIn("FIXED_DATALOADER_WORKERS", text)


class TestRerunCommand(unittest.TestCase):
    def _recipe(
        self, directory: Path, *, deterministic: bool, seed: int | None
    ) -> Path:
        lines = [
            "schema: ohho.recipe/v1",
            "family: act",
            "dataset:",
            f"  path: {directory / 'data'}",
            "hyperparameters:",
            "  train_steps: 1",
            "device: cpu",
            f"deterministic: {'true' if deterministic else 'false'}",
            "output:",
            f"  dir: {directory / 'out'}",
            "",
        ]
        if seed is not None:
            lines.insert(6, f"  seed: {seed}")
        path = directory / "recipe.yaml"
        path.write_text("\n".join(lines), encoding="utf-8")
        (directory / "data").mkdir()
        return path

    def test_rerun_matches_and_does_not_touch_the_original(self) -> None:
        calls: list[dict] = []

        def fake_train_act(**kwargs):
            calls.append(kwargs)
            return _write_metrics(kwargs["output_dir"], 0.25)

        with (
            patch(
                "ohho.train.builtin_recipes._import_torch",
                return_value=_TorchStub(),
            ),
            patch("ohho.train.act.train_act", side_effect=fake_train_act),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = self._recipe(root, deterministic=True, seed=None)
                out = root / "out"
                code, events, err = _run(
                    recipe,
                    [f"output.dir={out}", "deterministic=true"],
                )
                self.assertEqual(code, EXIT_OK, err)
                started = [event for event in events if event["event"] == "train_start"]
                self.assertTrue(started[0]["deterministic"])
                manifest_path = out / MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertIs(manifest["resolved_config"]["deterministic"], True)
                self.assertEqual(manifest["seeds"]["recipe"], 0)
                before = manifest_path.read_bytes()
                self.assertTrue(calls[0]["deterministic"])
                self.assertEqual(calls[0]["seed"], 0)
                rerun_dir = root / "again"
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = cli_main(
                        ["run", "rerun", str(out), "--output", str(rerun_dir), "--json"]
                    )
                self.assertEqual(status, 0, stderr.getvalue() + stdout.getvalue())
                body = json.loads(stdout.getvalue())
                self.assertEqual(body["schema"], "ohho.run.rerun/v1")
                self.assertTrue(body["exact"])
                self.assertTrue(body["within_noise"])
                self.assertEqual(body["original_final_loss"], 0.25)
                self.assertEqual(body["rerun_final_loss"], 0.25)
                self.assertEqual(body["expected_variance"], 0)
                self.assertEqual(body["tolerance"], 0)
                self.assertEqual(body["console_followup"], CONSOLE_FOLLOWUP)
                self.assertEqual(manifest_path.read_bytes(), before)
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[1]["deterministic"])
                self.assertEqual(calls[1]["seed"], 0)
                refused = io.StringIO()
                with redirect_stderr(refused), redirect_stdout(io.StringIO()):
                    same = cli_main(["run", "rerun", str(out), "--output", str(out)])
                self.assertEqual(same, 1)
                self.assertIn("different directory", refused.getvalue())
                self.assertEqual(len(calls), 2)

    def test_within_noise_is_not_an_exact_match(self) -> None:
        losses = [0.1, 0.1005]

        def fake_train_act(**kwargs):
            return _write_metrics(kwargs["output_dir"], losses.pop(0))

        with (
            patch(
                "ohho.train.builtin_recipes._import_torch",
                return_value=_TorchStub(),
            ),
            patch("ohho.train.act.train_act", side_effect=fake_train_act),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = self._recipe(root, deterministic=False, seed=1)
                out = root / "out"
                code, _events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = rerun_cli(str(out), output_dir=str(root / "again"))
                self.assertEqual(status, 1, stderr.getvalue() + stdout.getvalue())
                self.assertIn("exact: false", stdout.getvalue())
                self.assertIn("within_noise: true", stdout.getvalue())
                self.assertIn(CONSOLE_FOLLOWUP, stdout.getvalue())

    def test_a_failed_manifest_is_not_trained_again(self) -> None:
        calls = {"n": 0}

        def fake_train_act(**kwargs):
            calls["n"] += 1
            return _write_metrics(kwargs["output_dir"], 0.2)

        with (
            patch(
                "ohho.train.builtin_recipes._import_torch",
                return_value=_TorchStub(),
            ),
            patch("ohho.train.act.train_act", side_effect=fake_train_act),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = self._recipe(root, deterministic=True, seed=3)
                out = root / "out"
                code, _events, err = _run(recipe, [f"output.dir={out}"])
                self.assertEqual(code, EXIT_OK, err)
                path = out / MANIFEST_NAME
                document = json.loads(path.read_text(encoding="utf-8"))
                document["status"] = "failed"
                document["error"] = "disk full"
                path.write_text(json.dumps(document), encoding="utf-8")
                stderr = io.StringIO()
                with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    status = cli_main(["run", "rerun", str(out)])
                self.assertEqual(status, 1)
                self.assertIn("succeeded", stderr.getvalue())
                self.assertEqual(calls["n"], 1)
                missing = io.StringIO()
                with redirect_stderr(missing), redirect_stdout(io.StringIO()):
                    status = cli_main(["run", "rerun", str(root / "missing")])
                self.assertEqual(status, 1)
                self.assertIn("no run manifest", missing.getvalue())

    def test_diffusion_receives_the_flag_and_its_seed(self) -> None:
        seen: dict = {}

        def fake_train(**kwargs):
            seen.update(kwargs)
            return _write_metrics(kwargs["output_dir"], 0.4)

        with (
            patch(
                "ohho.train.builtin_recipes._import_torch",
                return_value=_TorchStub(),
            ),
            patch(
                "ohho.train.il.diffusion_policy.train_diffusion",
                side_effect=fake_train,
            ),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = root / "recipe.yaml"
                recipe.write_text(
                    "\n".join(
                        [
                            "schema: ohho.recipe/v1",
                            "family: diffusion",
                            "dataset:",
                            f"  path: {root / 'data'}",
                            "hyperparameters:",
                            "  train_steps: 1",
                            "  seed: 7",
                            "device: cpu",
                            "deterministic: true",
                            "output:",
                            f"  dir: {root / 'out'}",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                (root / "data").mkdir()
                code, events, err = _run(recipe, [])
                self.assertEqual(code, EXIT_OK, err)
                self.assertTrue(seen["deterministic"])
                self.assertEqual(seen["seed"], 7)
                started = [event for event in events if event["event"] == "train_start"]
                self.assertEqual(started[0]["family"], "diffusion")
                self.assertTrue(started[0]["deterministic"])

    def test_rerun_without_an_id_is_usage(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli_main(["run", "rerun"])
        self.assertEqual(caught.exception.code, 2)


class TestTinyActCpuRerun(unittest.TestCase):
    """Re-run the tiny ACT recipe on CPU and require an exact final loss."""

    dataset = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._skip = _torch_skip_reason()
        cls._tmp = None
        if cls._skip:
            return
        from ohho.sim_loop import record_sim_episodes

        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = record_sim_episodes(
            num_episodes=1,
            steps_per_episode=8,
            output_dir=str(Path(cls._tmp.name) / "dataset"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmp is not None:
            cls._tmp.cleanup()

    def setUp(self) -> None:
        if self._skip:
            self.skipTest(self._skip)

    def test_final_loss_matches_exactly(self) -> None:
        import torch

        before = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "run"
                code, events, err = _run(
                    _TINY_ACT,
                    [
                        f"dataset.path={self.dataset}",
                        f"output.dir={out}",
                        "deterministic=true",
                        "device=cpu",
                        "hyperparameters.train_steps=2",
                        "hyperparameters.seed=0",
                        "hyperparameters.vision_backbone=none",
                    ],
                )
                self.assertEqual(code, EXIT_OK, err)
                manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
                self.assertIs(manifest["resolved_config"]["deterministic"], True)
                self.assertEqual(manifest["seeds"]["recipe"], 0)
                started = [event for event in events if event["event"] == "train_start"]
                self.assertTrue(started[0]["deterministic"])
                original_metrics = json.loads(
                    (out / "metrics.json").read_text(encoding="utf-8")
                )
                before_bytes = (out / MANIFEST_NAME).read_bytes()
                rerun_dir = root / "rerun"
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = cli_main(
                        [
                            "run",
                            "rerun",
                            str(out),
                            "--output",
                            str(rerun_dir),
                            "--json",
                        ]
                    )
                self.assertEqual(status, 0, stderr.getvalue() + stdout.getvalue())
                body = json.loads(stdout.getvalue())
                self.assertTrue(body["exact"], stdout.getvalue())
                self.assertTrue(body["within_noise"])
                self.assertEqual(body["expected_variance"], 0)
                self.assertEqual(body["tolerance"], 0)
                self.assertEqual(body["family"], "act")
                self.assertIs(body["deterministic"], True)
                self.assertEqual(
                    body["original_final_loss"], original_metrics["final_loss"]
                )
                self.assertEqual(body["original_final_loss"], body["rerun_final_loss"])
                rerun_metrics = json.loads(
                    (rerun_dir / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    rerun_metrics["final_loss"], original_metrics["final_loss"]
                )
                self.assertEqual(
                    rerun_metrics["loss_curve"], original_metrics["loss_curve"]
                )
                self.assertEqual(
                    rerun_metrics.get("validation_loss"),
                    original_metrics.get("validation_loss"),
                )
                self.assertTrue(body["compared"]["final_loss"]["exact"])
                self.assertTrue(body["compared"]["loss_curve"]["exact"])
                self.assertEqual((out / MANIFEST_NAME).read_bytes(), before_bytes)
        finally:
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)


if __name__ == "__main__":
    unittest.main()
