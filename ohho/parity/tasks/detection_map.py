"""Detection mean average precision.

Boxes are dicts with ``class_id``, ``x``, ``y``, ``w``, ``h``, and an
optional ``score`` (default 1). Coordinates are corner-plus-size. A
prediction matches a ground-truth box of the same class when IoU is at
least 0.5, greedily in score order, one match each. Average precision uses
the all-point interpolation. mAP is the unweighted mean over classes that
appear in the ground truth.

Each policy is called with the image dict (including ``ground_truth``) and
returns a list of boxes.
"""

from __future__ import annotations

from typing import Any

from ohho.parity.errors import ParityError
from ohho.parity.tasks.tolerances import tolerance_for
from ohho.parity.tasks.types import Comparison, TaskContext

_IOU = 0.5


class DetectionMapEvaluator:
    """mAP of reference detections and optimised detections against one set."""

    name = "detection_map"

    def run(self, context: TaskContext) -> list[Comparison]:
        images = context.detections
        if not images:
            raise ParityError(
                "detection mAP needs a detection set "
                "(--task-detections or detections=)."
            )
        reference = _mean_ap(images, context.reference)
        optimized = _mean_ap(images, context.optimized)
        return [
            Comparison(
                task=self.name,
                metric="map",
                reference=reference,
                optimized=optimized,
                tolerance=tolerance_for(context.tolerances, "map"),
                direction="higher_is_better",
                unit="fraction",
                samples=len(images),
            )
        ]


def _mean_ap(images: tuple[dict[str, Any], ...], policy) -> float:
    predictions = [_boxes(policy(dict(image)), "detection") for image in images]
    classes: list[int] = []
    for image in images:
        for box in _boxes(image.get("ground_truth"), "ground truth"):
            if box["class_id"] not in classes:
                classes.append(box["class_id"])
    if not classes:
        raise ParityError("detection set has no ground-truth boxes.")
    scores = [_average_precision(images, predictions, class_id) for class_id in classes]
    return sum(scores) / len(scores)


def _average_precision(
    images: tuple[dict[str, Any], ...],
    predictions: list[list[dict[str, float | int]]],
    class_id: int,
) -> float:
    ground: list[list[dict[str, float | int]]] = []
    total = 0
    for image in images:
        boxes = [
            box
            for box in _boxes(image.get("ground_truth"), "ground truth")
            if box["class_id"] == class_id
        ]
        ground.append(boxes)
        total += len(boxes)
    if total == 0:
        return 0.0
    ranked: list[tuple[float, int, dict[str, float | int]]] = []
    for image_index, boxes in enumerate(predictions):
        for box in boxes:
            if box["class_id"] == class_id:
                ranked.append((float(box["score"]), image_index, box))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    matched = [set() for _ in images]
    hits: list[int] = []
    for _score, image_index, box in ranked:
        best_iou = 0.0
        best_index = -1
        for gt_index, truth in enumerate(ground[image_index]):
            if gt_index in matched[image_index]:
                continue
            overlap = _iou(truth, box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index >= 0 and best_iou >= _IOU:
            matched[image_index].add(best_index)
            hits.append(1)
        else:
            hits.append(0)
    if not hits:
        return 0.0
    true_pos = 0
    false_pos = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for hit in hits:
        true_pos += hit
        false_pos += 1 - hit
        recalls.append(true_pos / total)
        precisions.append(true_pos / (true_pos + false_pos))
    return _voc_ap(recalls, precisions)


def _voc_ap(recalls: list[float], precisions: list[float]) -> float:
    recall = [0.0, *recalls, 1.0]
    precision = [0.0, *precisions, 0.0]
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    area = 0.0
    for index in range(1, len(recall)):
        if recall[index] != recall[index - 1]:
            area += (recall[index] - recall[index - 1]) * precision[index]
    return area


def _iou(left: dict[str, float | int], right: dict[str, float | int]) -> float:
    ax1, ay1 = float(left["x"]), float(left["y"])
    ax2, ay2 = ax1 + float(left["w"]), ay1 + float(left["h"])
    bx1, by1 = float(right["x"]), float(right["y"])
    bx2, by2 = bx1 + float(right["w"]), by1 + float(right["h"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    union = float(left["w"]) * float(left["h"]) + float(right["w"]) * float(right["h"])
    union -= intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _boxes(raw: object, label: str) -> list[dict[str, float | int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ParityError(f"{label} must be a list of boxes.")
    boxes: list[dict[str, float | int]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ParityError(f"{label} boxes must be objects.")
        class_id = item.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ParityError(f"{label} class_id must be an integer.")
        box: dict[str, float | int] = {"class_id": class_id}
        for key in ("x", "y", "w", "h"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParityError(f"{label} {key} must be a number.")
            box[key] = float(value)
        score = item.get("score", 1.0)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ParityError(f"{label} score must be a number.")
        box["score"] = float(score)
        boxes.append(box)
    return boxes


evaluator = DetectionMapEvaluator()
