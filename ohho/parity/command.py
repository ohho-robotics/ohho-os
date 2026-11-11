"""``ohho parity`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def register_parity(subparsers: Any) -> None:
    """Add the ``parity`` command."""
    parser = subparsers.add_parser(
        "parity",
        help="compare an optimized artifact to a reference model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ohho parity --reference reference-act --candidate build/act "
            "--inputs random --n 256 --json\n"
            "  ohho parity --reference policy.pt --candidate build/act/model.onnx "
            "--inputs dataset/ --n 256 --mean-abs 0.02\n"
            "  ohho parity --reference reference-mlp --candidate candidate.pt "
            "--inputs recorded --seed 0\n"
            "\n"
            "The default fail is mean absolute difference above 0.05. "
            "A threshold file can set limits per model family and per target. "
            "TensorRT engines are not executed (GPU follow-up); pass the ONNX file.\n"
            "\n"
            "--tasks runs action error, sim success, detection mAP, and token "
            "match after the tensor gate. The task report is ohho.bench/v1."
        ),
    )
    parser.add_argument(
        "--reference", required=True, help="checkpoint path or registry id"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="ONNX, TorchScript, module, or output directory",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="random, a LeRobot dataset directory, recorded, or recorded:PATH",
    )
    parser.add_argument(
        "--n", type=int, default=256, help="number of samples (default 256)"
    )
    parser.add_argument("--seed", type=int, default=0, help="random and subsample seed")
    parser.add_argument(
        "--mean-abs",
        type=float,
        default=None,
        help="override the mean absolute difference limit (default 0.05)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="JSON file of per-family and per-target limits",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="model family for limits and extra metrics (default: the model kind)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="target profile id used to pick limits",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="precision used to pick limits, for example FP32 or INT8",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="where the recorded sim-loop episode is written",
    )
    parser.add_argument(
        "--out", default=None, help="write the report JSON to this path"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--tasks",
        default=None,
        help="optional task gate: all, or a comma-separated evaluator list",
    )
    parser.add_argument(
        "--task-seeds",
        type=int,
        default=20,
        help="fixed sim seeds 0..N-1 (default 20)",
    )
    parser.add_argument(
        "--task-tolerances",
        default=None,
        help="JSON file of per-model task tolerances",
    )
    parser.add_argument(
        "--task-episodes",
        default=None,
        help="held-out LeRobot dataset or JSON episode file",
    )
    parser.add_argument(
        "--task-detections",
        default=None,
        help="JSON file of detection images for mAP",
    )
    parser.add_argument(
        "--task-tokens",
        default=None,
        help="JSON file of labelled token examples",
    )
    parser.add_argument(
        "--accuracy-out",
        default=None,
        help="write the ohho.bench/v1 task report to this path",
    )
    parser.set_defaults(func=run_parity)


def run_parity(args: argparse.Namespace) -> int:
    """Score one pair. A threshold miss exits 1. A request error exits 2."""
    from ohho.parity.errors import ParityError
    from ohho.parity.gate import evaluate

    try:
        report = evaluate(
            args.reference,
            args.candidate,
            inputs=args.inputs,
            n=args.n,
            seed=args.seed,
            family=args.family,
            target=args.target,
            precision=args.precision,
            mean_abs=args.mean_abs,
            thresholds=args.thresholds,
            record_dir=args.record_dir,
        )
        accuracy = None
        if args.tasks:
            from ohho.parity.tasks.stage import evaluate_cli

            accuracy = evaluate_cli(args)
    except ParityError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2)
    if args.out:
        from pathlib import Path

        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if accuracy is not None:
        _write_accuracy(args, accuracy)
    if args.json:
        if accuracy is None:
            print(text)
        else:
            print(json.dumps({"parity": report, "accuracy": accuracy}, indent=2))
    else:
        _print_human(report)
        if accuracy is not None:
            _print_accuracy(accuracy)
    task_failed = accuracy is not None and accuracy["accuracy"]["result"] != "pass"
    if report["result"] != "pass" or task_failed:
        return 1
    return 0


def _write_accuracy(args: argparse.Namespace, accuracy: dict[str, Any]) -> None:
    from pathlib import Path

    if args.accuracy_out:
        destination = Path(args.accuracy_out)
    elif args.out:
        destination = Path(args.out).with_name("accuracy.json")
    else:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(accuracy, indent=2) + "\n", encoding="utf-8")


def _print_accuracy(document: dict[str, Any]) -> None:
    accuracy = document["accuracy"]
    print(f"tasks {accuracy['result']}")
    for item in accuracy.get("comparisons") or []:
        print(
            f"  {item['task']} {item['metric']}: "
            f"reference={item['reference']:.6g} "
            f"optimized={item['optimized']:.6g} "
            f"delta={item['delta']:.6g} {item['result']}"
        )
    for reason in accuracy.get("reasons") or []:
        print(f"  {reason}")


def _print_human(report: dict[str, Any]) -> None:
    inputs = report["inputs"]
    print(f"parity {report['result']}")
    print(
        f"  inputs: {inputs['source']}  n={inputs['n']}  "
        f"seed={inputs['seed']}  sha256={inputs['sha256']}"
    )
    for item in report["outputs"]:
        print(
            f"  {item['name']}: mean_abs={item['mean_abs']:.6g} "
            f"max_abs={item['max_abs']:.6g} rmse={item['rmse']:.6g} "
            f"cosine={item['cosine']:.6g}"
        )
        if item["per_joint"] is not None:
            joints = ", ".join(f"{value:.6g}" for value in item["per_joint"])
            print(f"    per_joint: {joints}")
        if item["action_chunk"] is not None:
            print(f"    action_chunk: {item['action_chunk']:.6g}")
        if item["top1"] is not None:
            print(f"    top1: {item['top1']:.6g}  kl: {item['kl']:.6g}")
    for reason in report["reasons"]:
        print(f"  {reason}")
    for item in report["unchecked"]:
        print(f"  unchecked {item['role']}: {item['reason']}")
