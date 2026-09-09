"""Geometry and diagnostic matching; usable on Mac without ML frameworks."""

import math


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_box(left, right):
    return [
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    ]


def box_iou(left, right):
    intersection = box_area(intersection_box(left, right))
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def union_coverage(box, regions):
    """Fraction of a box covered by the union, without double-counting overlaps."""
    area = box_area(box)
    if area <= 0:
        raise ValueError("Coverage requires a positive-area box")
    clipped = [intersection_box(box, region) for region in regions]
    clipped = [region for region in clipped if box_area(region) > 0]
    xs = sorted({x for region in clipped for x in (region[0], region[2])})
    covered = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (region[1], region[3]) for region in clipped if region[0] < right and region[2] > left
        )
        height = 0.0
        end = -math.inf
        for low, high in intervals:
            height += max(0.0, high - max(low, end))
            end = max(end, high)
        covered += (right - left) * height
    return min(1.0, covered / area)


def tile_windows(width, height, tile_size, overlap):
    """Full coverage; the last full tile is shifted inward at each image edge."""
    if min(width, height, tile_size) <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive dimensions and overlap in [0, 1) are required")
    step = max(1, round(tile_size * (1 - overlap)))

    def starts(length):
        last = max(0, length - tile_size)
        values = list(range(0, last + 1, step))
        if values[-1] != last:
            values.append(last)
        return values

    return [
        [x, y, min(width, x + tile_size), min(height, y + tile_size)]
        for y in starts(height)
        for x in starts(width)
    ]


def restore_tile_predictions(predictions, window, image_size, tile_index):
    width, height = image_size
    restored = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["xyxy"]
        box = [
            min(width, max(0.0, x1 + window[0])),
            min(height, max(0.0, y1 + window[1])),
            min(width, max(0.0, x2 + window[0])),
            min(height, max(0.0, y2 + window[1])),
        ]
        if box_area(box) > 0:
            restored.append({**prediction, "xyxy": box, "tile_index": tile_index})
    return restored


def merge_tile_predictions(predictions, nms_iou, max_detections):
    """Confidence-ordered, car-only global NMS after coordinate restoration."""
    if not 0 < nms_iou <= 1 or max_detections <= 0:
        raise ValueError("Invalid NMS threshold or detection cap")
    remaining = sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i))
    kept = []
    while remaining:
        selected = remaining[0]
        kept.append(selected)
        remaining = [
            index
            for index in remaining[1:]
            if box_iou(predictions[selected]["xyxy"], predictions[index]["xyxy"]) <= nms_iou
        ]
    return [predictions[i] for i in kept[:max_detections]], {
        "before_merge": len(predictions),
        "after_nms": len(kept),
        "after_cap": min(len(kept), max_detections),
        "removed_by_nms": len(predictions) - len(kept),
        "removed_by_cap": max(0, len(kept) - max_detections),
    }


def diagnostic_match(ground_truth, predictions, ignored, match_iou, ignore_coverage):
    """Project diagnostic, not official AP: region filtering then greedy matching."""
    if not 0 < match_iou <= 1 or not 0 < ignore_coverage <= 1:
        raise ValueError("Matching thresholds must be in (0, 1]")
    regions = [item["xyxy"] for item in ignored if item["valid_geometry"]]
    gt_states = [
        {
            "status": "ignored"
            if union_coverage(item["xyxy"], regions) >= ignore_coverage
            else "fn",
            "prediction_index": None,
        }
        for item in ground_truth
    ]
    prediction_states = [
        {
            "status": "ignored"
            if union_coverage(item["xyxy"], regions) >= ignore_coverage
            else "fp",
            "gt_index": None,
            "iou": None,
        }
        for item in predictions
    ]
    for index in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i)):
        if prediction_states[index]["status"] == "ignored":
            continue
        candidates = [
            (box_iou(predictions[index]["xyxy"], item["xyxy"]), gt_index)
            for gt_index, item in enumerate(ground_truth)
            if gt_states[gt_index]["status"] == "fn"
        ]
        if not candidates:
            continue
        overlap, gt_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if overlap >= match_iou:
            prediction_states[index] = {"status": "tp", "gt_index": gt_index, "iou": overlap}
            gt_states[gt_index] = {"status": "tp", "prediction_index": index}
    tp = sum(item["status"] == "tp" for item in prediction_states)
    fp = sum(item["status"] == "fp" for item in prediction_states)
    fn = sum(item["status"] == "fn" for item in gt_states)
    return {
        **count_metrics(tp, fp, fn),
        "raw_gt": len(ground_truth),
        "raw_predictions": len(predictions),
        "ignored_gt": sum(item["status"] == "ignored" for item in gt_states),
        "ignored_predictions": sum(item["status"] == "ignored" for item in prediction_states),
        "gt_states": gt_states,
        "prediction_states": prediction_states,
    }


def count_metrics(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evaluated_gt": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def size_group(box, image_size, baseline_size, boundaries):
    scale = min(baseline_size / image_size[0], baseline_size / image_size[1])
    side = math.sqrt(box_area(box)) * scale
    for index, boundary in enumerate(boundaries):
        if side < boundary:
            return f"lt{boundary}" if index == 0 else f"{boundaries[index - 1]}_to_{boundary}"
    return f"ge{boundaries[-1]}"


def recovery_counts(baseline, changed):
    if len(baseline["gt_states"]) != len(changed["gt_states"]):
        raise ValueError("Recovery comparisons require the same GT identities")
    pairs = zip(baseline["gt_states"], changed["gt_states"])
    recovered = lost = retained = still_missed = 0
    for old, new in pairs:
        if (old["status"] == "ignored") != (new["status"] == "ignored"):
            raise ValueError("The ignore policy must be identical across arms")
        if old["status"] == "fn" and new["status"] == "tp":
            recovered += 1
        elif old["status"] == "tp" and new["status"] == "fn":
            lost += 1
        elif old["status"] == new["status"] == "tp":
            retained += 1
        elif old["status"] == new["status"] == "fn":
            still_missed += 1
    return {
        "recovered_gt": recovered,
        "lost_gt": lost,
        "retained_gt": retained,
        "still_missed_gt": still_missed,
        "fp_delta": changed["fp"] - baseline["fp"],
    }
