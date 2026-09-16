"""Raise confidence cutoffs on saved predictions without loading a model."""

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from comparison_utils import count_metrics, diagnostic_match
from comparison_visuals import ARM_NAMES, percent
from data_utils import sha256
from model_arch import ROOT

DEFAULT_SOURCE = ROOT / "runs/input-comparison-20260909T080429Z"
SOURCE_FILES = (
    "run.json",
    "inputs.json",
    "summary.json",
    "baseline.jsonl",
    "resolution.jsonl",
    "sliced.jsonl",
)


def validated_thresholds(values, source_confidence):
    if not values or any(
        not math.isfinite(value) or not source_confidence <= value <= 1 for value in values
    ):
        raise ValueError(f"Thresholds must be finite and in [{source_confidence}, 1]")
    return sorted(set([source_confidence, *values]))


def sweep_records(records, thresholds, match_iou, ignore_coverage):
    totals, per_image = [], []
    for threshold in thresholds:
        counts = {
            key: 0
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
        for record in records:
            predictions = [p for p in record["predictions"] if p["confidence"] >= threshold]
            matched = diagnostic_match(
                record["ground_truth"],
                predictions,
                record["ignored_regions"],
                match_iou,
                ignore_coverage,
            )
            for key in counts:
                counts[key] += matched[key]
            per_image.append(
                {
                    "confidence": threshold,
                    "index": record["index"],
                    "image": record["image"],
                    **{
                        key: value
                        for key, value in matched.items()
                        if key not in ("gt_states", "prediction_states")
                    },
                }
            )
        totals.append(
            {
                "confidence": threshold,
                **counts,
                **count_metrics(counts["tp"], counts["fp"], counts["fn"]),
            }
        )
    baseline = totals[0]
    for row in totals:
        row["tp_lost_vs_source"] = baseline["tp"] - row["tp"]
        row["fp_removed_vs_source"] = baseline["fp"] - row["fp"]
        row["predictions_removed_vs_source"] = baseline["raw_predictions"] - row["raw_predictions"]
    return totals, per_image


def load_source(directory):
    runtime = json.loads((directory / "run.json").read_text())
    if runtime["status"] != "completed":
        raise ValueError("Only a completed comparison run can be analyzed")
    manifest = json.loads((directory / "inputs.json").read_text())
    if sha256(directory / "inputs.json") != runtime["sample_manifest_sha256"]:
        raise ValueError("Saved sample manifest no longer matches the source run")
    images = [item["image"] for item in manifest["samples"]]
    if len(images) != runtime["base_config"]["sample_count"] or len(set(images)) != len(images):
        raise ValueError("Invalid sample identities")
    arms = {
        name: [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
        for name in ARM_NAMES
    }
    for name, records in arms.items():
        if [record["image"] for record in records] != images:
            raise ValueError(f"Unexpected image ordering in {name}")
        for record, reference in zip(records, arms["baseline"]):
            if (
                record["ground_truth"] != reference["ground_truth"]
                or record["ignored_regions"] != reference["ignored_regions"]
            ):
                raise ValueError(f"Different labels or ignore regions in {name}")
            if any(
                not math.isfinite(p["confidence"])
                or not runtime["base_config"]["confidence"] <= p["confidence"] <= 1
                or p["coco_class_id"] != runtime["base_config"]["coco_class_id"]
                for p in record["predictions"]
            ):
                raise ValueError(f"Invalid cached confidence or class in {name}")
    return runtime, arms, json.loads((directory / "summary.json").read_text())


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output, requested_thresholds):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Write results outside the source run")
    source_hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    runtime, arms, previous = load_source(source)
    thresholds = validated_thresholds(requested_thresholds, runtime["base_config"]["confidence"])
    comparison = runtime["comparison_config"]
    rows, image_rows = [], []
    for name in ARM_NAMES:
        totals, details = sweep_records(
            arms[name], thresholds, comparison["match_iou"], comparison["ignore_coverage"]
        )
        for key in (
            "tp",
            "fp",
            "fn",
            "raw_gt",
            "raw_predictions",
            "ignored_gt",
            "ignored_predictions",
        ):
            if totals[0][key] != previous["arms"][name][key]:
                raise ValueError(f"Source-cutoff replay mismatch: {name}/{key}")
        rows.extend({"arm": name, **row} for row in totals)
        image_rows.extend({"arm": name, **row} for row in details)
    if source_hashes != {name: sha256(source / name) for name in SOURCE_FILES}:
        raise ValueError("Source artifacts changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_model_commit": runtime["commit"],
        "source_run": str(source),
        "source_file_sha256": source_hashes,
        "source_sample_manifest_sha256": runtime["sample_manifest_sha256"],
        "source_weights_sha256": runtime["weights_sha256"],
        "source_confidence": runtime["base_config"]["confidence"],
        "thresholds": thresholds,
        "matching_iou": comparison["match_iou"],
        "ignore_coverage": comparison["ignore_coverage"],
        "per_view_nms_iou": runtime["base_config"]["nms_iou"],
        "tile_merge_nms_iou": comparison["tile_merge_nms_iou"],
        "sample_count": runtime["base_config"]["sample_count"],
        "model_forward_calls": 0,
        "nms_rerun": False,
        "source_cutoff_reproduced": True,
        "scope": "Cached post-NMS confidence filtering and rematching; partial operating points, not full PR or AP",
    }
    save_csv(output / "summary.csv", rows)
    save_csv(output / "per_image.csv", image_rows)
    (output / "summary.json").write_text(
        json.dumps({"protocol": protocol, "rows": rows}, indent=2) + "\n"
    )
    report = [
        "# Cached confidence comparison",
        "",
        "Same saved predictions, labels and IoU rules; no model inference or training.",
        "",
        f"Matching IoU: {protocol['matching_iou']}; original per-view and tile-merge NMS: "
        f"{protocol['per_view_nms_iou']} / {protocol['tile_merge_nms_iou']}. "
        f"Source model run commit: {protocol['source_model_commit']}.",
        "",
        "| Arm | Confidence | TP | FP | FN | Precision | Recall | F1 | TP lost | FP removed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['arm']} | {row['confidence']:.2f} | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {percent(row['precision'])} | {percent(row['recall'])} | "
            f"{percent(row['f1'])} | {row['tp_lost_vs_source']} | {row['fp_removed_vs_source']} |"
        )
    report += [
        "",
        "TP lost / FP removed are relative to the original confidence cutoff for the same arm.",
        "Ignored predictions are excluded from precision. Empty-denominator ratios remain undefined.",
        "Filtering uses confidence >= cutoff, followed by fresh one-to-one matching at the unchanged IoU.",
        "The original post-NMS boxes are retained as the candidate pool; NMS is not run again.",
        "Lower cutoffs cannot be recovered from this cache. These points are not a complete PR curve, "
        "a new detector, a speed benchmark, official AP or a final test-set result.",
        "",
        "[Per-image data](per_image.csv) | [Summary CSV](summary.csv)",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    metadata = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "sweep_confidence.py",
                "comparison_utils.py",
                "comparison_visuals.py",
            )
        },
        "protocol": protocol,
        "summary_rows": len(rows),
        "per_image_rows": len(image_rows),
    }
    (output / "analysis.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n".join(report), flush=True)
    print(f"Results: {output}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.45])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime(
        "confidence-sweep-%Y%m%dT%H%M%SZ"
    )
    analyze(args.source_run, output, args.thresholds)


if __name__ == "__main__":
    main()
