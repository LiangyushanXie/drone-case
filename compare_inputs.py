"""Compare fixed 640, whole-image 960 and tile-only 640 inputs on MyGPU."""

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from comparison_utils import (
    box_area,
    count_metrics,
    diagnostic_match,
    merge_tile_predictions,
    recovery_counts,
    restore_tile_predictions,
    size_group,
    tile_windows,
)
from comparison_visuals import ARM_NAMES, percent, render_comparison
from data_utils import parse_annotations, sha256, verify_manifest
from model_arch import (
    ROOT,
    build_detector,
    build_square_predictor,
    load_config,
    prediction_arguments,
)

DEFAULT_SETTINGS = ROOT / "configs/input_comparison.json"


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    for key in ("whole_image_size", "tile_size", "tile_input_size"):
        value = settings[key]
        if not isinstance(value, int) or value <= 0 or value % 32:
            raise ValueError(f"{key} must be a positive multiple of 32")
    if not 0 <= settings["tile_overlap"] < 1:
        raise ValueError("tile_overlap must be in [0, 1)")
    for key in ("match_iou", "ignore_coverage", "tile_merge_nms_iou"):
        if not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if settings["include_full_image_in_sliced"]:
        raise ValueError("This experiment compares tile-only predictions")
    boundaries = settings["size_group_boundaries_at_baseline"]
    if not boundaries or boundaries != sorted(set(boundaries)) or boundaries[0] <= 0:
        raise ValueError("Size boundaries must be increasing positive numbers")
    base = load_config(ROOT / settings["base_config"])
    if base["image_size"] != 640 or settings["tile_input_size"] != 640:
        raise ValueError("The agreed baseline and per-tile input are both 640")
    if base["sample_count"] != 30:
        raise ValueError("Keep the fixed 30-image development sample for this comparison")
    if base["half_precision"]:
        raise ValueError("Keep FP32 for all three comparison arms")
    if settings["tile_merge_nms_iou"] != base["nms_iou"]:
        raise ValueError("Global merge NMS must retain the baseline threshold")
    return settings, base


def extract_predictions(result, class_id):
    predictions = []
    for xyxy, confidence, category in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        if int(category) != class_id or not all(math.isfinite(v) for v in [*xyxy, confidence]):
            raise RuntimeError("Unexpected class or non-finite prediction")
        if box_area(xyxy) <= 0 or not 0 <= confidence <= 1:
            raise RuntimeError("Invalid prediction geometry or confidence")
        predictions.append({"xyxy": xyxy, "confidence": confidence, "coco_class_id": int(category)})
    return predictions


def run_arm(name, base, settings, manifest, weights, output):
    import cv2
    import torch

    input_size = (
        settings["whole_image_size"]
        if name == "resolution"
        else settings["tile_input_size"]
        if name == "sliced"
        else base["image_size"]
    )
    config = {**base, "image_size": input_size}
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    detector, environment = build_detector(config, weights)
    predictor_class = build_square_predictor()
    events = []
    context = {}

    def check_input(_module, inputs):
        tensor = inputs[0]
        shape = list(tensor.shape)
        if shape != [1, 3, input_size, input_size] or tensor.device.type != "cuda":
            raise RuntimeError(f"Unexpected {name} input: {shape} on {tensor.device}")
        if tensor.dtype != torch.float32:
            raise RuntimeError("This comparison requires FP32 in every arm")
        events.append(
            {**context, "shape": shape, "device": str(tensor.device), "dtype": str(tensor.dtype)}
        )

    hook = detector.model.register_forward_pre_hook(check_input)
    records = []
    try:
        with (output / f"{name}.jsonl").open("w") as stream:
            for index, sample in enumerate(manifest["samples"], 1):
                image = cv2.imread(str(ROOT / sample["image"]))
                if image is None:
                    raise ValueError(f"Cannot decode {sample['image']}")
                height, width = image.shape[:2]
                ground_truth, ignored = parse_annotations(
                    (ROOT / sample["annotation"]).read_text(), base["visdrone_class_id"]
                )
                windows = (
                    tile_windows(width, height, settings["tile_size"], settings["tile_overlap"])
                    if name == "sliced"
                    else [[0, 0, width, height]]
                )
                predictions = []
                tile_caps = 0
                before = len(events)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for tile_index, window in enumerate(windows):
                    context.clear()
                    context.update(
                        {"image_index": index, "tile_index": tile_index, "window": window}
                    )
                    x1, y1, x2, y2 = window
                    crop = image[y1:y2, x1:x2]
                    event_count = len(events)
                    result = detector.predict(
                        source=crop, predictor=predictor_class, **prediction_arguments(config)
                    )[0]
                    if len(events) <= event_count:
                        raise RuntimeError("Model input guard did not observe this prediction")
                    boxes = extract_predictions(result, base["coco_class_id"])
                    tile_caps += len(boxes) >= base["max_detections"]
                    if name == "sliced":
                        boxes = restore_tile_predictions(boxes, window, (width, height), tile_index)
                    predictions.extend(boxes)
                if name == "sliced":
                    predictions, merge = merge_tile_predictions(
                        predictions, settings["tile_merge_nms_iou"], base["max_detections"]
                    )
                else:
                    merge = {
                        "before_merge": len(predictions),
                        "after_nms": len(predictions),
                        "after_cap": len(predictions),
                        "removed_by_nms": 0,
                        "removed_by_cap": 0,
                    }
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                evaluation = diagnostic_match(
                    ground_truth,
                    predictions,
                    ignored,
                    settings["match_iou"],
                    settings["ignore_coverage"],
                )
                record = {
                    "index": index,
                    "image": sample["image"],
                    "image_size": [width, height],
                    "ground_truth": ground_truth,
                    "ignored_regions": ignored,
                    "predictions": predictions,
                    "evaluation": evaluation,
                    "windows": windows,
                    "predict_calls": len(windows),
                    "forward_events": len(events) - before,
                    "tile_calls_at_detection_cap": tile_caps,
                    "merge": merge,
                    "predict_and_merge_seconds": elapsed,
                    "includes_initial_warmup": index == 1,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.append(record)
                print(
                    f"{name} {index:02d}/30: TP={evaluation['tp']} FP={evaluation['fp']} "
                    f"FN={evaluation['fn']} | {len(windows)} view(s)",
                    flush=True,
                )
    finally:
        hook.remove()
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    return records, {
        "environment": environment,
        "input_size": input_size,
        "predict_arguments": prediction_arguments(config),
        "observed_inputs": events,
        "predict_calls": sum(record["predict_calls"] for record in records),
        "images_processed": len(records),
    }


def summarize(arms, settings, base):
    baseline = arms["baseline"]
    for name, records in arms.items():
        if len(records) != len(baseline) or any(
            old["image"] != new["image"] or old["ground_truth"] != new["ground_truth"]
            for old, new in zip(baseline, records)
        ):
            raise ValueError(f"Non-matching image or GT identities in {name}")
    summary, groups, rows, recoveries = {}, {}, [], {}
    for name, records in arms.items():
        totals = {
            key: sum(record["evaluation"][key] for record in records)
            for key in (
                "tp",
                "fp",
                "fn",
                "raw_gt",
                "raw_predictions",
                "ignored_gt",
                "ignored_predictions",
            )
        }
        summary[name] = {
            **totals,
            **count_metrics(totals["tp"], totals["fp"], totals["fn"]),
            "median_predict_and_merge_ms_excluding_first_image": statistics.median(
                record["predict_and_merge_seconds"] * 1000 for record in records[1:]
            ),
            "predict_calls": sum(record["predict_calls"] for record in records),
            "negative_image_fp": sum(
                record["evaluation"]["fp"] for record in records if not record["ground_truth"]
            ),
            "negative_images": sum(not record["ground_truth"] for record in records),
        }
        for record in records:
            evaluation = record["evaluation"]
            rows.append(
                {
                    "arm": name,
                    "index": record["index"],
                    "image": record["image"],
                    **{
                        key: value
                        for key, value in evaluation.items()
                        if key not in ("gt_states", "prediction_states")
                    },
                    "predict_calls": record["predict_calls"],
                    "predict_and_merge_ms": record["predict_and_merge_seconds"] * 1000,
                }
            )
            for gt, state in zip(record["ground_truth"], evaluation["gt_states"]):
                if state["status"] == "ignored":
                    continue
                labels = {
                    "size_at_baseline_640": size_group(
                        gt["xyxy"],
                        record["image_size"],
                        base["image_size"],
                        settings["size_group_boundaries_at_baseline"],
                    ),
                    "occlusion": {0: "visible", 1: "partial", 2: "heavy"}.get(
                        gt["occlusion"], f"unknown_{gt['occlusion']}"
                    ),
                }
                for kind, label in labels.items():
                    group = groups.setdefault((name, kind, label), {"gt": 0, "tp": 0, "fn": 0})
                    group["gt"] += 1
                    group[state["status"]] += 1
        if name != "baseline":
            counts = [
                recovery_counts(old["evaluation"], new["evaluation"])
                for old, new in zip(arms["baseline"], records)
            ]
            recoveries[name] = {key: sum(item[key] for item in counts) for key in counts[0]}
    group_rows = [
        {"arm": arm, "kind": kind, "group": label, **counts, "recall": counts["tp"] / counts["gt"]}
        for (arm, kind, label), counts in sorted(groups.items())
    ]
    return {"arms": summary, "recovery_vs_baseline": recoveries}, rows, group_rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_reference_baseline(runtime, records):
    directory = ROOT / runtime["comparison_config"]["reference_baseline_run"]
    if not (directory / "run.json").is_file():
        return {"available": False}
    reference = json.loads((directory / "run.json").read_text())
    comparable = (
        reference.get("status") == "completed"
        and reference["config"] == runtime["base_config"]
        and reference["weights_sha256"] == runtime["weights_sha256"]
        and reference["sample_manifest_sha256"] == runtime["sample_manifest_sha256"]
    )
    if not comparable:
        return {"available": True, "comparable": False}
    previous = json.loads((directory / "predictions.json").read_text())
    old_by_image = {record["image"]: record["predictions"] for record in previous}
    same_counts = set(old_by_image) == {record["image"] for record in records} and all(
        len(old_by_image[record["image"]]) == len(record["predictions"]) for record in records
    )
    coordinate_delta = confidence_delta = 0.0
    if same_counts:
        for record in records:
            old = sorted(old_by_image[record["image"]], key=lambda item: tuple(item["xyxy"]))
            new = sorted(record["predictions"], key=lambda item: tuple(item["xyxy"]))
            for left, right in zip(old, new):
                coordinate_delta = max(
                    coordinate_delta, *(abs(a - b) for a, b in zip(left["xyxy"], right["xyxy"]))
                )
                confidence_delta = max(
                    confidence_delta, abs(left["confidence"] - right["confidence"])
                )
    return {
        "available": True,
        "comparable": True,
        "same_counts": same_counts,
        "max_coordinate_delta_pixels": coordinate_delta if same_counts else None,
        "max_confidence_delta": confidence_delta if same_counts else None,
        "reproduced_within_tolerance": (
            same_counts and coordinate_delta <= 0.01 and confidence_delta <= 0.00001
        ),
        "reference_commit": reference["commit"],
    }


def render_outputs(output):
    runtime = json.loads((output / "run.json").read_text())
    settings, base = runtime["comparison_config"], runtime["base_config"]
    manifest = json.loads((output / "inputs.json").read_text())
    verify_manifest(manifest, base)
    arms = {
        name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        for name in ARM_NAMES
    }
    expected_images = [sample["image"] for sample in manifest["samples"]]
    for name, records in arms.items():
        if [record["image"] for record in records] != expected_images:
            raise ValueError(f"Image identities differ in {name}")
    summary, rows, group_rows = summarize(arms, settings, base)
    summary["reference_baseline_audit"] = audit_reference_baseline(runtime, arms["baseline"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "group_recall.csv", group_rows)
    for index, sample in enumerate(manifest["samples"]):
        render_comparison(
            ROOT / sample["image"],
            {name: arms[name][index] for name in ARM_NAMES},
            settings,
            output / f"{index + 1:02d}_{Path(sample['image']).stem}.png",
        )
    report = [
        "# YOLOv12-S input comparison",
        "",
        "Development diagnostic on the same 30 validation images; no training or official AP.",
        "",
        f"Commit: {runtime['commit']}",
        f"Confidence: {base['confidence']}; per-view NMS: {base['nms_iou']}; "
        f"tile merge NMS: {settings['tile_merge_nms_iou']}; matching IoU: {settings['match_iou']}.",
        "",
        "| Input | Predictions | TP | FP | FN | Ignored predictions | Precision | Recall | Median ms/image* |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary["arms"].items():
        report.append(
            f"| {name} | {values['raw_predictions']} | {values['tp']} | {values['fp']} | "
            f"{values['fn']} | {values['ignored_predictions']} | {percent(values['precision'])} | "
            f"{percent(values['recall'])} | "
            f"{values['median_predict_and_merge_ms_excluding_first_image']:.1f} |"
        )
    report += [
        "",
        "*Wall time includes preprocessing, model execution, NMS, CPU box copies and tile merge; "
        "excludes file decoding, matching and rendering. First image of each arm is excluded from "
        "the median because it includes automatic warmup. These are observed session timings.",
        "",
        f"Raw car labels: {summary['arms']['baseline']['raw_gt']}; "
        f"evaluated car labels: {summary['arms']['baseline']['evaluated_gt']}; "
        f"ignored car labels: {summary['arms']['baseline']['ignored_gt']}.",
        "",
        "Score-0/category-0 regions form an ignore union. GT or predictions with at least "
        f"{percent(settings['ignore_coverage'])} of their area inside it are excluded before matching. "
        "Geometry is continuous xyxy, "
        "not the official toolkit's rounded pixel mask. This car-only diagnostic is not official AP.",
        "",
        "Size groups use sqrt(GT area) after scaling to the BASELINE 640 input, with boundaries "
        f"{settings['size_group_boundaries_at_baseline']} pixels, fixed across all arms. "
        "These are diagnostic groups, not COCO AP bins. "
        "Occlusion uses the original VisDrone labels. Group recall reuses the global matching.",
        "",
        "Orange prediction boxes matched GT; red boxes did not; gray boxes are ignored. "
        "Numbers beside predictions are confidence scores, not statistical significance.",
        "",
        "| Change vs baseline | Recovered GT | Lost GT | Retained GT | Still missed GT | FP delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in summary["recovery_vs_baseline"].items():
        report.append(
            f"| {name} | {values['recovered_gt']} | {values['lost_gt']} | "
            f"{values['retained_gt']} | {values['still_missed_gt']} | {values['fp_delta']:+d} |"
        )
    report += [
        "",
        "## Prior baseline sanity check",
        "",
        json.dumps(summary["reference_baseline_audit"], indent=2),
        "",
        "## Size and occlusion groups",
        "",
        "| Arm | Grouping | Group | GT | TP | FN | Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        report.append(
            f"| {row['arm']} | {row['kind']} | {row['group']} | "
            f"{row['gt']} | {row['tp']} | {row['fn']} | {percent(row['recall'])} |"
        )
    report += ["", "## Comparison images", ""]
    for index, sample in enumerate(manifest["samples"], 1):
        image = f"{index:02d}_{Path(sample['image']).stem}.png"
        focus = " (user-selected focus)" if index in settings["focus_image_indices"] else ""
        report.append(f"- [{index:02d}{focus}]({image})")
    (output / "report.md").write_text("\n".join(report) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/yolov12s.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--render-only", type=Path)
    args = parser.parse_args()
    if args.render_only:
        if args.execute:
            parser.error("--render-only and --execute are separate operations")
        render_outputs(args.render_only.resolve())
        print(f"Rendered saved predictions: {args.render_only}")
        return
    settings, base = load_settings(args.config)
    manifest = json.loads((ROOT / base["sample_manifest"]).read_text())
    verify_manifest(manifest, base)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "sample_count": len(manifest["samples"]),
                    "baseline": prediction_arguments(base),
                    "comparison": settings,
                    "next": "Run compare_inputs.py --execute on MyGPU",
                },
                indent=2,
            )
        )
        return
    receipt = json.loads(args.weights.with_suffix(".source.json").read_text())
    weight_hash = sha256(args.weights)
    if (
        args.weights.stat().st_size != base["weights_bytes"]
        or receipt["url"] != base["weights_url"]
        or receipt["sha256"] != weight_hash
    ):
        raise ValueError("Official checkpoint size or provenance receipt mismatch")
    started = datetime.now(timezone.utc)
    output = (
        args.output or ROOT / "runs" / started.strftime("input-comparison-%Y%m%dT%H%M%SZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "schema_version": 1,
        "status": "running",
        "started_at": started.isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "base_config": base,
        "comparison_config": settings,
        "weights_sha256": weight_hash,
        "sample_manifest_sha256": sha256(ROOT / base["sample_manifest"]),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "compare_inputs.py",
                "comparison_utils.py",
                "comparison_visuals.py",
                "model_arch.py",
                "data_utils.py",
            )
        },
        "arms": {},
        "scope": "30-image development diagnostic; no training or official AP",
    }
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(runtime, indent=2) + "\n")
    try:
        for name in ARM_NAMES:
            _, metadata = run_arm(name, base, settings, manifest, args.weights, output)
            runtime["arms"][name] = metadata
            (output / "run.json").write_text(json.dumps(runtime, indent=2) + "\n")
        render_outputs(output)
        if len(list(output.glob("*.png"))) != base["sample_count"]:
            raise RuntimeError("Incomplete comparison images")
        runtime["status"] = "completed"
        runtime["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        runtime["status"] = "failed"
        runtime["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (output / "run.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
