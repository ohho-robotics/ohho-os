"""``ohho optimize`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ohho.optimize.errors import OptimizeError
from ohho.optimize.plan import PRECISION_FLAGS, OptimizeRequest
from ohho.parity.errors import ParityError, ParityFailed, TaskFailed


def register_optimize(subparsers: Any) -> None:
    """Add the ``optimize`` command."""
    parser = subparsers.add_parser(
        "optimize",
        help="export and compile a model for a hardware target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ohho optimize --model reference-act --target x86-cpu "
            "--precision fp32 --out build/act --json\n"
            "  ohho optimize --model policy.pt --submodule actor --target x86-cuda "
            "--precision fp16 --tensorrt --workspace-gb 2 --max-batch 4 "
            "--dynamic-batch --out build/fp16\n"
            "  ohho optimize --model policy.pt --target jetson-orin-nx "
            "--precision int8 --tensorrt --calib-data calib --calib-batch 8 "
            "--calib-max-images 32 --calib-cache build/int8.cache --out build/int8\n"
            "  ohho optimize --model smolvla/ --submodule lm --precision int4-awq "
            "--target INT4_PROFILE --out build/awq\n"
            "  ohho optimize --model reference-act --target x86-cpu "
            "--precision int8 --out build/int8\n"
            "\n"
            "TensorRT, the CUDA execution provider, AWQ, GPTQ, bitsandbytes, "
            "and FP8 fail fast when they are not installed. That is a GPU follow-up. "
            "FP8 is TensorRT on Ada and Hopper only. int8 on x86-cpu and rpi5-cpu "
            "is ORT static INT8. int4-bnb is a bitsandbytes NF4 research baseline.\n"
            "\n"
            "The parity gate runs after the export. A mean absolute difference "
            "above 0.05 refuses to publish unless --allow-parity-fail is set. "
            "That flag is recorded in the manifest.\n"
            "\n"
            "--tasks runs the task-accuracy gate after tensor parity and writes "
            "accuracy.json (ohho.bench/v1). A task miss also refuses to publish "
            "unless --allow-parity-fail is set."
        ),
    )
    parser.add_argument("--model", required=True, help="checkpoint path or registry id")
    parser.add_argument(
        "--target", required=True, help="target profile id or profile file"
    )
    parser.add_argument(
        "--precision",
        required=True,
        choices=PRECISION_FLAGS,
        help=(
            "fp32, fp16, int8 (ORT static on CPU; TensorRT with --tensorrt), "
            "fp8 (TensorRT, Ada/Hopper), int4, int4-awq, int4-gptq, int4-bnb, nf4"
        ),
    )
    parser.add_argument("--onnx", action="store_true", help="write an ONNX file")
    parser.add_argument(
        "--tensorrt", action="store_true", help="build a TensorRT engine"
    )
    parser.add_argument("--calib-data", default=None, help="INT8 calibration directory")
    parser.add_argument(
        "--calib-batch",
        type=int,
        default=None,
        help="INT8 calibrator batch size",
    )
    parser.add_argument(
        "--calib-cache", default=None, help="INT8 calibration cache path"
    )
    parser.add_argument(
        "--calib-max-images",
        type=int,
        default=None,
        help="maximum number of INT8 calibration samples to read",
    )
    parser.add_argument(
        "--ort-ep",
        choices=("cuda", "cpu", "tensorrt"),
        default=None,
        help="ONNX Runtime execution provider",
    )
    parser.add_argument(
        "--quant", choices=("awq", "gptq"), default=None, help="weight quantizer"
    )
    parser.add_argument(
        "--workspace-gb", type=float, default=None, help="TensorRT workspace GiB"
    )
    parser.add_argument(
        "--max-batch", type=int, default=1, help="maximum batch in the profile"
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="TensorRT profile batch runs from 1 to --max-batch",
    )
    parser.add_argument(
        "--submodule",
        default=None,
        help="dotted module path, or a role: vision, actor, encoder, lm",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--allow-parity-fail",
        action="store_true",
        help="publish even when the parity gate fails; recorded in the manifest",
    )
    parser.add_argument(
        "--parity-inputs",
        default="random",
        help="parity inputs: random, a LeRobot dataset, or recorded",
    )
    parser.add_argument(
        "--parity-n", type=int, default=256, help="parity sample count (default 256)"
    )
    parser.add_argument(
        "--parity-seed", type=int, default=0, help="seed for parity inputs"
    )
    parser.add_argument(
        "--parity-mean-abs",
        type=float,
        default=None,
        help="override the parity mean absolute difference limit (default 0.05)",
    )
    parser.add_argument(
        "--parity-thresholds",
        default=None,
        help="JSON file of per-family and per-target parity limits",
    )
    parser.add_argument(
        "--parity-family",
        default=None,
        help="model family for the parity gate (default: the model kind)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="optional task gate after tensor parity: all, or a comma-separated list",
    )
    parser.add_argument(
        "--task-seeds",
        type=int,
        default=20,
        help="fixed sim seeds 0..N-1 for the task gate (default 20)",
    )
    parser.add_argument(
        "--task-tolerances",
        default=None,
        help="JSON file of per-model task tolerances",
    )
    parser.add_argument(
        "--task-episodes",
        default=None,
        help="held-out episode set for action error",
    )
    parser.add_argument(
        "--task-detections",
        default=None,
        help="JSON detection set for mAP",
    )
    parser.add_argument(
        "--task-tokens",
        default=None,
        help="JSON labelled examples for token match",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the manifest as JSON"
    )
    parser.set_defaults(func=run_optimize)


def run_optimize(args: argparse.Namespace) -> int:
    """Run one optimize request. Validation and capability errors exit 2."""
    request = OptimizeRequest(
        model=args.model,
        target=args.target,
        precision=args.precision,
        out_dir=args.out,
        onnx=bool(args.onnx),
        tensorrt=bool(args.tensorrt),
        calib_data=args.calib_data,
        calib_batch=args.calib_batch,
        calib_cache=args.calib_cache,
        calib_max_images=args.calib_max_images,
        ort_ep=args.ort_ep,
        quant=args.quant,
        workspace_gb=args.workspace_gb,
        max_batch=args.max_batch,
        dynamic_batch=bool(args.dynamic_batch),
        submodule=args.submodule,
        allow_parity_fail=bool(args.allow_parity_fail),
        parity_inputs=args.parity_inputs,
        parity_n=args.parity_n,
        parity_seed=args.parity_seed,
        parity_mean_abs=args.parity_mean_abs,
        parity_thresholds=args.parity_thresholds,
        parity_family=args.parity_family,
        tasks=args.tasks,
        task_tolerances=args.task_tolerances,
        task_seeds=args.task_seeds,
        task_episodes=args.task_episodes,
        task_detections=args.task_detections,
        task_tokens=args.task_tokens,
    )
    try:
        from ohho.optimize.engine import optimize

        document = optimize(request)
    except TaskFailed as exc:
        if args.json:
            print(
                json.dumps({"ok": False, "error": str(exc), "accuracy": exc.document})
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except ParityFailed as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "parity": exc.report}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except ParityError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    except OptimizeError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(document, indent=2))
        return 0
    print(f"optimized {document['model']['ref']} for {document['target']['id']}")
    print(f"  precision: {document['precision']}")
    print(f"  provider: {document['execution_provider']}")
    for item in document["outputs"]:
        print(f"  {item['role']}: {item['path']}")
    parity = document.get("parity") or {}
    print(f"  parity: {parity.get('result', 'unknown')}")
    accuracy_path = Path(args.out) / "accuracy.json"
    if accuracy_path.is_file():
        task_doc = json.loads(accuracy_path.read_text(encoding="utf-8"))
        task_result = (task_doc.get("accuracy") or {}).get("result", "unknown")
        print(f"  tasks: {task_result}")
        print("  accuracy: accuracy.json")
    print("  manifest: manifest.json")
    return 0
