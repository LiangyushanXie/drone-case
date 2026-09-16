"""Pure-Python confidence-ordered matching and 101-point interpolated custom AP."""

from comparison_utils import count_metrics

IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def match_overlaps(overlaps, thresholds=IOU_THRESHOLDS):
    """Rows already follow stable descending confidence; one GT per prediction."""
    candidates = [
        sorted(
            ((value, i) for i, value in enumerate(row) if value >= min(thresholds)),
            key=lambda item: (-item[0], item[1]),
        )
        for row in overlaps
    ]
    result = [[False] * len(thresholds) for _ in overlaps]
    for column, threshold in enumerate(thresholds):
        matched = set()
        for index, row in enumerate(candidates):
            for overlap, gt_index in row:
                if overlap < threshold:
                    break
                if gt_index not in matched:
                    result[index][column] = True
                    matched.add(gt_index)
                    break
    return result


def interpolated_ap(correct, gt_count):
    """Mean envelope precision at recall 0,.01,...,1, with zero past max recall."""
    if gt_count == 0:
        return None
    precision, recall = [], []
    tp = 0
    for rank, positive in enumerate(correct, 1):
        tp += int(positive)
        precision.append(tp / rank)
        recall.append(tp / gt_count)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    total, index = 0.0, 0
    for step in range(101):
        target = step / 100
        while index < len(recall) and recall[index] < target:
            index += 1
        if index < len(precision):
            total += precision[index]
    return total / 101


def summarize_records(records, confidences=(0.25, 0.7, 0.8, 0.9)):
    gt_count = sum(record["evaluated_gt"] for record in records)
    rows = [
        (confidence, flags)
        for record in records
        for confidence, flags in zip(record["confidence"], record["correct"])
    ]
    rows.sort(key=lambda row: -row[0])
    aps = [
        interpolated_ap([row[1][column] for row in rows], gt_count)
        for column in range(len(IOU_THRESHOLDS))
    ]
    operating_points = []
    for column in (0, 5):
        for cutoff in confidences:
            selected = [row for row in rows if row[0] >= cutoff]
            tp = sum(row[1][column] for row in selected)
            operating_points.append(
                {
                    "match_iou": IOU_THRESHOLDS[column],
                    "confidence": cutoff,
                    **count_metrics(tp, len(selected) - tp, gt_count - tp),
                }
            )
    return {
        "images": len(records),
        "evaluated_gt": gt_count,
        "ignored_gt": sum(record["ignored_gt"] for record in records),
        "ignored_predictions": sum(record["ignored_predictions"] for record in records),
        "capped_images": sum(record["hit_prediction_cap"] for record in records),
        "custom_AP50": aps[0],
        "custom_AP75": aps[5],
        "custom_AP50_95": sum(aps) / len(aps) if gt_count else None,
        "AP_by_iou": dict(zip(map(str, IOU_THRESHOLDS), aps)),
        "operating_points": operating_points,
    }
