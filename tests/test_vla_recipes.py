"""OHH-509: SmolVLA and OpenVLA recipes, memory, licence, and LoRA merge."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohho.serve import _is_act_checkpoint
from ohho.train.builtin_recipes import OPENVLA, SMOLVLA
from ohho.train.recipe_registry import ensure_loaded, get_family
from ohho.train.recipe_schema import (
    FamilyUnavailable,
    RecipeConfigError,
    example_path,
    load_recipe_file,
)
from ohho.train.run_cli import EXIT_INVALID, EXIT_UNAVAILABLE, execute
from ohho.train.vla import (
    checkpoint_metadata,
    documented_parameter_lines,
    estimate_memory,
    merge_lora_for_serve,
    prepare,
    resolve_hyperparameters,
)
from ohho.train.vla import export as export_mod

_VLA = Path(__file__).resolve().parents[1] / "ohho" / "train" / "examples" / "vla"
_DOCS = Path(__file__).resolve().parents[1] / "docs" / "train" / "vla-recipes.md"
_RECIPES = Path(__file__).resolve().parents[1] / "docs" / "train" / "recipes.md"
_CLI = Path(__file__).resolve().parents[1] / "docs" / "cli.md"


def _run(path: str | Path, overrides: list[str] | None = None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    from contextlib import redirect_stderr, redirect_stdout

    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = execute(str(path), overrides)
    events = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return code, events, stderr.getvalue()


def _torch_skip() -> str | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "skipped: torch is not installed (pip install 'ohho-os[train]')"
    return None


def _peft_skip() -> str | None:
    reason = _torch_skip()
    if reason:
        return (
            "skipped: torch and peft are not installed, so the tiny LoRA merge "
            "cannot run (pip install torch peft)"
        )
    try:
        import peft  # noqa: F401
    except ImportError:
        return "skipped: peft is not installed, so the tiny LoRA merge cannot run (pip install peft)"
    return None


class TestParameterValidation(unittest.TestCase):
    def test_defaults_and_shipped_examples(self) -> None:
        smol = resolve_hyperparameters("smolvla", {})
        self.assertEqual(smol["train_steps"], 200)
        self.assertEqual(smol["batch_size"], 8)
        self.assertEqual(smol["lr"], 0.0001)
        self.assertEqual(smol["precision"], "bf16")
        self.assertIs(smol["gradient_checkpointing"], True)
        self.assertEqual(smol["tune_mode"], "full")
        self.assertEqual(smol["lora_rank"], 32)
        self.assertEqual(smol["lora_alpha"], 16)
        self.assertEqual(smol["lora_dropout"], 0.0)
        self.assertEqual(smol["lora_target_modules"], ("all-linear",))
        self.assertNotIn("seed", smol)
        self.assertNotIn("num_epochs", smol)
        openvla = resolve_hyperparameters("openvla", {})
        self.assertEqual(openvla["batch_size"], 16)
        self.assertEqual(openvla["lr"], 5e-4)
        self.assertEqual(openvla["tune_mode"], "lora")
        self.assertEqual(openvla["precision"], "bf16")
        again = resolve_hyperparameters("smolvla", smol)
        self.assertEqual(again, smol)
        shipped = load_recipe_file(example_path("smolvla"))
        resolved = resolve_hyperparameters("smolvla", shipped["hyperparameters"])
        self.assertEqual(resolved["train_steps"], 40)
        self.assertEqual(resolved["batch_size"], 4)
        self.assertEqual(resolved["precision"], "bf16")
        self.assertEqual(shipped["precision"], "bf16")
        open_shipped = load_recipe_file(example_path("openvla"))
        self.assertEqual(open_shipped["precision"], "fp16")
        open_resolved = resolve_hyperparameters(
            "openvla", open_shipped["hyperparameters"]
        )
        self.assertEqual(open_resolved["precision"], "bf16")
        self.assertEqual(open_resolved["train_steps"], 20)
        self.assertEqual(open_resolved["batch_size"], 2)
        self.assertEqual(open_resolved["tune_mode"], "lora")
        both = resolve_hyperparameters("openvla", {"num_epochs": 9, "train_steps": 4})
        self.assertEqual(both["train_steps"], 4)
        self.assertEqual(both["num_epochs"], 9)

    def test_ranges_reject_before_any_download(self) -> None:
        cases = [
            ("smolvla", {"lora_rank": 0}, "lora_rank"),
            ("smolvla", {"lora_rank": 257}, "lora_rank"),
            ("smolvla", {"lora_alpha": 0}, "lora_alpha"),
            ("smolvla", {"lora_dropout": 0.6}, "lora_dropout"),
            ("smolvla", {"precision": "fp64"}, "precision"),
            ("smolvla", {"tune_mode": "freeze"}, "tune_mode"),
            ("smolvla", {"gradient_checkpointing": "yes"}, "gradient_checkpointing"),
            ("smolvla", {"batch_size": True}, "batch_size"),
            ("smolvla", {"lr": 0}, "lr"),
            ("smolvla", {"max_grad_norm": 0}, "max_grad_norm"),
            ("openvla", {"mock": True}, "fake trainer"),
            ("openvla", {"chunk_size": 8}, "unknown hyperparameters"),
            ("openvla", {"lora_target_modules": "all-linear,q_proj"}, "cannot combine"),
            ("openvla", {"lora_target_modules": "q_proj,q_proj"}, "duplicate"),
            ("openvla", {"lora_target_modules": ""}, "lora_target_modules"),
            ("openvla", {"lora_target_modules": ["q proj"]}, "lora_target_modules"),
            ("smolvla", {"num_epochs": 1_000_000}, "outside [1, 1000000]"),
        ]
        for family, raw, needle in cases:
            with self.subTest(family=family, needle=needle):
                with patch(
                    "urllib.request.urlopen", side_effect=AssertionError("download")
                ):
                    with self.assertRaises(RecipeConfigError) as caught:
                        resolve_hyperparameters(family, raw)
                self.assertIn(needle, str(caught.exception))

    def test_target_module_forms(self) -> None:
        names = resolve_hyperparameters(
            "openvla", {"lora_target_modules": "q_proj, k_proj"}
        )
        self.assertEqual(names["lora_target_modules"], ("q_proj", "k_proj"))
        listed = resolve_hyperparameters(
            "smolvla", {"lora_target_modules": ["q_proj", "v_proj"]}
        )
        self.assertEqual(listed["lora_target_modules"], ("q_proj", "v_proj"))
        too_many = [f"m{index}" for index in range(33)]
        with self.assertRaises(RecipeConfigError):
            resolve_hyperparameters("openvla", {"lora_target_modules": too_many})

    def test_shared_precision_field_is_not_the_family_dtype(self) -> None:
        recipe = load_recipe_file(example_path("openvla"))
        recipe["precision"] = "fp32"
        prepared = prepare("openvla", recipe)
        self.assertEqual(recipe["precision"], "fp32")
        self.assertEqual(prepared.resolved["precision"], "bf16")
        wider = estimate_memory("openvla", {"precision": "fp32"})
        same = estimate_memory("openvla", {"precision": "fp16"})
        default = estimate_memory("openvla", {})
        self.assertEqual(same.total_bytes, default.total_bytes)
        self.assertGreater(wider.total_bytes, default.total_bytes)


class TestMemoryEstimate(unittest.TestCase):
    def test_smolvla_default_fits_24gb(self) -> None:
        estimate = estimate_memory("smolvla", {})
        self.assertEqual(estimate.gpu_class, "24gb")
        self.assertEqual(estimate.recommended_gpu, "single 24 GB GPU")
        self.assertEqual(estimate.launch_limit_gib, 24.0)
        self.assertEqual(estimate.tune_mode, "full")
        self.assertEqual(estimate.precision, "bf16")
        self.assertIs(estimate.gradient_checkpointing, True)
        self.assertEqual(estimate.total_gib, 17.029)
        self.assertTrue(estimate.fits_launch_class)
        self.assertEqual(estimate.warning, "")
        self.assertEqual(estimate.parameter_count, 450_000_000)
        self.assertEqual(estimate.trainable_parameters, 450_000_000)
        self.assertEqual(
            estimate.total_bytes,
            estimate.weights_bytes
            + estimate.gradients_bytes
            + estimate.optimizer_bytes
            + estimate.adapters_bytes
            + estimate.activations_bytes
            + estimate.overhead_bytes,
        )

    def test_smolvla_over_24gb_warns(self) -> None:
        large = estimate_memory("smolvla", {"batch_size": 16})
        self.assertEqual(large.total_gib, 27.029)
        self.assertFalse(large.fits_launch_class)
        self.assertIn("24 GB", large.warning)
        off = estimate_memory("smolvla", {"gradient_checkpointing": False})
        self.assertEqual(off.total_gib, 37.029)
        self.assertIn("24 GB", off.warning)
        lora = estimate_memory("smolvla", {"tune_mode": "lora"})
        self.assertEqual(lora.total_gib, 12.909)
        self.assertEqual(lora.trainable_parameters, 6_300_000)
        self.assertLess(lora.total_bytes, estimate_memory("smolvla").total_bytes)
        self.assertEqual(lora.gpu_class, "24gb")

    def test_openvla_lora_anchors_match_the_readme_points(self) -> None:
        batch_16 = estimate_memory("openvla", {})
        self.assertEqual(batch_16.gpu_class, "a100-80gb")
        self.assertEqual(batch_16.recommended_gpu, "single A100 80GB")
        self.assertEqual(batch_16.launch_limit_gib, 80.0)
        self.assertEqual(batch_16.total_gib, 72.0)
        self.assertTrue(batch_16.fits_launch_class)
        self.assertEqual(batch_16.warning, "")
        self.assertEqual(batch_16.parameter_count, 7_500_000_000)
        self.assertGreater(batch_16.overhead_bytes, 0)
        batch_1 = estimate_memory("openvla", {"batch_size": 1})
        self.assertEqual(batch_1.total_gib, 27.0)
        self.assertTrue(batch_1.fits_launch_class)
        batch_32 = estimate_memory("openvla", {"batch_size": 32})
        self.assertEqual(batch_32.total_gib, 120.0)
        self.assertIn("80 GB", batch_32.warning)
        self.assertIn("single A100 80GB", batch_32.warning)
        off = estimate_memory("openvla", {"gradient_checkpointing": False})
        self.assertEqual(off.total_gib, 168.0)
        self.assertFalse(off.fits_launch_class)
        rank_64 = estimate_memory("openvla", {"lora_rank": 64})
        self.assertGreater(rank_64.trainable_parameters, batch_16.trainable_parameters)
        self.assertGreater(rank_64.total_bytes, batch_16.total_bytes)
        one_module = estimate_memory("openvla", {"lora_target_modules": "q_proj"})
        self.assertLess(one_module.trainable_parameters, batch_16.trainable_parameters)

    def test_openvla_full_is_multi_gpu(self) -> None:
        estimate = estimate_memory("openvla", {"tune_mode": "full", "batch_size": 1})
        self.assertEqual(estimate.gpu_class, "multi-gpu-cloud")
        self.assertIsNone(estimate.launch_limit_gib)
        self.assertFalse(estimate.fits_launch_class)
        self.assertIn("multi-GPU/cloud only", estimate.warning)
        self.assertIn("8 A100 80GB", estimate.warning)
        self.assertGreater(estimate.total_gib, 80.0)
        self.assertEqual(estimate.trainable_parameters, estimate.parameter_count)
        body = estimate.as_dict()
        self.assertIsNone(body["launch_limit_gib"])
        self.assertEqual(body["warning"], estimate.warning)

    def test_grad_accumulation_does_not_scale_memory(self) -> None:
        once = estimate_memory("openvla", {"grad_accumulation_steps": 1})
        eight = estimate_memory("openvla", {"grad_accumulation_steps": 8})
        self.assertEqual(once.total_bytes, eight.total_bytes)
        self.assertEqual(eight.grad_accumulation_steps, 8)


class TestLicenceMetadata(unittest.TestCase):
    def test_published_licences_are_recorded_without_download(self) -> None:
        with patch("urllib.request.urlopen", side_effect=AssertionError("download")):
            smol = checkpoint_metadata(
                "smolvla", {}, base_weights_uri="local:smolvla-base"
            )
            openvla = checkpoint_metadata("openvla", {})
        self.assertEqual(smol["base_model"], "lerobot/smolvla_base")
        self.assertEqual(smol["base_model_licence"], "Apache-2.0")
        self.assertEqual(smol["base_model_card_licence"], "Apache-2.0")
        self.assertEqual(smol["base_weights_uri"], "local:smolvla-base")
        self.assertIn("not loaded", smol["parameter_count_note"])
        self.assertIs(smol["merged_lora"], False)
        self.assertEqual(smol["precision"], "bf16")
        self.assertIs(smol["mock"], False)
        self.assertEqual(openvla["base_model"], "openvla/openvla-7b")
        self.assertEqual(openvla["base_model_licence"], "Llama 2 Community License")
        self.assertEqual(openvla["base_model_card_licence"], "MIT")
        self.assertIn("MIT", openvla["base_model_licence_note"])
        self.assertIn("Llama-2", openvla["base_model_licence_note"])
        self.assertIs(openvla["merged_lora"], True)
        self.assertEqual(openvla["parameter_count"], 7_500_000_000)
        prepared = prepare("smolvla", load_recipe_file(example_path("smolvla")))
        self.assertEqual(prepared.metadata["base_model_licence"], "Apache-2.0")
        self.assertEqual(prepared.metadata["base_weights_uri"], "local:smolvla-base")
        self.assertEqual(prepared.memory.batch_size, 4)

    def test_vla_sources_do_not_download_weights(self) -> None:
        root = Path(export_mod.__file__).resolve().parent
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("from_pretrained", text, path.name)
            self.assertNotIn("hf_hub_download", text, path.name)
            self.assertNotIn("urlopen", text, path.name)


class TestFamilyRun(unittest.TestCase):
    def test_registered_families_preflight_then_stop(self) -> None:
        ensure_loaded()
        self.assertIs(get_family("smolvla"), SMOLVLA)
        self.assertIs(get_family("openvla"), OPENVLA)
        self.assertTrue(SMOLVLA.available())
        self.assertTrue(OPENVLA.available())

        def _emit(payload: dict) -> None:
            raise AssertionError(payload)

        with self.assertRaises(FamilyUnavailable):
            SMOLVLA.run({"hyperparameters": {}}, _emit)
        with self.assertRaises(RecipeConfigError) as caught:
            OPENVLA.run({"hyperparameters": {"lora_rank": 0}}, _emit)
        self.assertIn("lora_rank", str(caught.exception))

    def test_train_command_does_not_download_or_start(self) -> None:
        with patch("urllib.request.urlopen", side_effect=AssertionError("download")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out"
                for family in ("smolvla", "openvla"):
                    with self.subTest(family=family):
                        code, events, err = _run(
                            example_path(family), [f"output.dir={out}"]
                        )
                        self.assertEqual(code, EXIT_UNAVAILABLE, err)
                        self.assertEqual(
                            [event["event"] for event in events], ["error"]
                        )
                        self.assertEqual(events[0]["family"], family)
                        self.assertFalse(out.exists())
                code, events, err = _run(
                    example_path("smolvla"),
                    ["hyperparameters.lora_rank=0", f"output.dir={out}"],
                )
                self.assertEqual(code, EXIT_INVALID, err)
                self.assertIn("lora_rank", err)
                self.assertEqual([event["event"] for event in events], ["error"])
                code, events, err = _run(
                    example_path("openvla"),
                    ["precision=fp32", f"output.dir={out}"],
                )
                self.assertEqual(code, EXIT_UNAVAILABLE, err)
                self.assertFalse(
                    any(event["event"] == "train_start" for event in events)
                )
                code, events, err = _run(
                    example_path("act"),
                    [
                        "family=smolvla",
                        "dataset.path=null",
                        "dataset.version_id=not-resolved",
                    ],
                )
                self.assertEqual(code, EXIT_INVALID, err)
                self.assertIn("unknown hyperparameters", err)
                self.assertNotIn("version ids are not resolved", err)


class TestDocsAndExamples(unittest.TestCase):
    def test_docs_list_every_parameter_default_and_range(self) -> None:
        text = _DOCS.read_text(encoding="utf-8")
        for line in documented_parameter_lines():
            self.assertIn(line, text)
        self.assertIn("24 GB", text)
        self.assertIn("single A100 80GB", text)
        self.assertIn("multi-GPU/cloud only", text)
        self.assertIn("OHH-524", text)
        self.assertIn("Llama 2 Community License", text)
        self.assertIn("Apache-2.0", text)
        self.assertIn("does not download", text)
        recipes = _RECIPES.read_text(encoding="utf-8")
        self.assertIn("vla-recipes.md", recipes)
        self.assertIn("vla-recipes.md", _CLI.read_text(encoding="utf-8"))
        project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("train/examples/vla/*.yaml", project)

    def test_launch_examples_validate(self) -> None:
        names = sorted(path.name for path in _VLA.glob("*.yaml"))
        self.assertEqual(
            names,
            ["openvla-full.yaml", "openvla-lora-80gb.yaml", "smolvla-24gb.yaml"],
        )
        smol = load_recipe_file(_VLA / "smolvla-24gb.yaml")
        self.assertEqual(smol["device"], "cuda")
        self.assertEqual(smol["precision"], "bf16")
        resolved = resolve_hyperparameters("smolvla", smol["hyperparameters"])
        self.assertEqual(resolved["train_steps"], 20000)
        self.assertEqual(resolved["batch_size"], 8)
        self.assertEqual(resolved["tune_mode"], "full")
        self.assertEqual(resolved["precision"], "bf16")
        lora = load_recipe_file(_VLA / "openvla-lora-80gb.yaml")
        lora_resolved = resolve_hyperparameters("openvla", lora["hyperparameters"])
        self.assertEqual(lora_resolved["train_steps"], 200000)
        self.assertEqual(lora_resolved["batch_size"], 16)
        self.assertEqual(lora_resolved["tune_mode"], "lora")
        self.assertEqual(lora_resolved["lora_rank"], 32)
        self.assertEqual(
            estimate_memory("openvla", lora["hyperparameters"]).total_gib, 72.0
        )
        full = load_recipe_file(_VLA / "openvla-full.yaml")
        full_estimate = estimate_memory("openvla", full["hyperparameters"])
        self.assertEqual(full_estimate.gpu_class, "multi-gpu-cloud")
        self.assertFalse(full_estimate.fits_launch_class)


class TestLoRAMerge(unittest.TestCase):
    def test_missing_torch_and_peft_name_the_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "ohho.train.vla.export._import_torch",
                side_effect=ImportError("no torch"),
            ):
                with self.assertRaises(RecipeConfigError) as caught:
                    merge_lora_for_serve(
                        object(),
                        tmp,
                        family="smolvla",
                        hyperparameters={"tune_mode": "full"},
                    )
            self.assertIn("torch", str(caught.exception))
            with patch("ohho.train.vla.export._import_torch", return_value=object()):
                with patch(
                    "ohho.train.vla.export._import_peft",
                    side_effect=ImportError("no peft"),
                ):
                    with self.assertRaises(RecipeConfigError) as caught:
                        merge_lora_for_serve(
                            object(),
                            tmp,
                            family="openvla",
                            hyperparameters={"tune_mode": "lora"},
                        )
            self.assertIn("peft", str(caught.exception))

    def test_tiny_random_model_merges_and_records_the_licence(self) -> None:
        reason = _peft_skip()
        if reason:
            self.skipTest(reason)
        import torch
        from peft import LoraConfig, get_peft_model
        from torch import nn

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8, bias=False)
                self.k_proj = nn.Linear(8, 8, bias=False)

        torch.manual_seed(0)
        model = Tiny()
        before = model.q_proj.weight.detach().cpu().clone()
        self.assertFalse(torch.equal(before, torch.zeros_like(before)))
        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "k_proj"],
                bias="none",
            ),
        )
        found = False
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.fill_(0.5)
                    found = True
        self.assertTrue(found)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serve"
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("download")
            ):
                merge_lora_for_serve(
                    peft_model,
                    out,
                    family="openvla",
                    hyperparameters={
                        "tune_mode": "lora",
                        "lora_rank": 2,
                        "lora_alpha": 4,
                        "lora_target_modules": ["q_proj", "k_proj"],
                    },
                    base_weights_uri="openvla/openvla-7b",
                )
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["policy"], "openvla")
            self.assertEqual(config["base_model_licence"], "Llama 2 Community License")
            self.assertEqual(config["base_model"], "openvla/openvla-7b")
            self.assertEqual(config["base_weights_uri"], "openvla/openvla-7b")
            self.assertIs(config["merged_lora"], True)
            self.assertEqual(config["precision"], "bf16")
            self.assertIs(config["mock"], False)
            self.assertFalse(_is_act_checkpoint(str(out)))
            state = torch.load(out / "policy.pt", weights_only=True)
            self.assertFalse(any("lora_" in key for key in state))
            weight = state["q_proj.weight"]
            self.assertFalse(torch.allclose(weight, before))

    def test_plain_module_is_wrapped_without_downloading(self) -> None:
        reason = _peft_skip()
        if reason:
            self.skipTest(reason)
        import torch
        from torch import nn

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(8, 8, bias=False)

        torch.manual_seed(0)
        model = Tiny()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serve"
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("download")
            ):
                merge_lora_for_serve(
                    model,
                    out,
                    family="openvla",
                    hyperparameters={
                        "tune_mode": "lora",
                        "lora_target_modules": "q_proj",
                        "lora_rank": 2,
                    },
                )
            state = torch.load(out / "policy.pt", weights_only=True)
            self.assertIn("q_proj.weight", state)
            self.assertFalse(any("lora_" in key for key in state))
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["lora_target_modules"], ["q_proj"])
            self.assertIs(config["merged_lora"], True)
            self.assertEqual(config["base_model_licence"], "Llama 2 Community License")

    def test_full_export_writes_the_smolvla_licence(self) -> None:
        reason = _torch_skip()
        if reason:
            self.skipTest(reason)
        import torch
        from torch import nn

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)

        torch.manual_seed(1)
        model = Tiny()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "serve"
            with patch(
                "ohho.train.vla.export._import_peft",
                side_effect=AssertionError("peft"),
            ):
                merge_lora_for_serve(
                    model,
                    out,
                    family="smolvla",
                    hyperparameters={"tune_mode": "full"},
                    base_weights_uri="lerobot/smolvla_base",
                )
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["base_model_licence"], "Apache-2.0")
            self.assertEqual(config["tune_mode"], "full")
            self.assertIs(config["merged_lora"], False)
            state = torch.load(out / "policy.pt", weights_only=True)
            self.assertIn("q_proj.weight", state)
            self.assertFalse(_is_act_checkpoint(str(out)))


if __name__ == "__main__":
    unittest.main()
