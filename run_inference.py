"""Preview pretrained car detection on MyGPU; default invocation is a dry run."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from data_utils import parse_annotations, sha256, verify_manifest
from model_arch import (
    DEFAULT_CONFIG,
    ROOT,
    build_detector,
    build_square_predictor,
    load_config,
    prediction_arguments,
)


def render_pair(image_path, ground_truth, ignored, predictions, destination):
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        source = source.convert("RGB")
        width, height = source.size
        left, right = source.copy(), source.copy()
    font = ImageFont.load_default(size=max(12, width // 100))
    left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
    for region in ignored:
        if region["valid_geometry"]:
            left_draw.rectangle(region["xyxy"], outline="#888888", width=2)
    for box in ground_truth:
        left_draw.rectangle(box["xyxy"], outline="#18c77a", width=2)
    for box in predictions:
        right_draw.rectangle(box["xyxy"], outline="#ff9d2e", width=2)
        right_draw.text(
            (box["xyxy"][0], max(0, box["xyxy"][1] - 15)),
            f"{box['confidence']:.2f}",
            font=font,
            fill="#ff9d2e",
            stroke_width=1,
            stroke_fill="black",
        )
    panel = Image.new("RGB", (width * 2 + 8, height + 48), "#101820")
    panel.paste(left, (0, 48))
    panel.paste(right, (width + 8, 48))
    title = ImageDraw.Draw(panel)
    title.text(
        (12, 12),
        f"Ground truth: {len(ground_truth)} car boxes | gray = ignored",
        fill="white",
        font=font,
    )
    title.text(
        (width + 20, 12),
        f"YOLOv12-S: {len(predictions)} car predictions | scores shown",
        fill="white",
        font=font,
    )
    panel.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "checkpoints/yolov12s.pt"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads((ROOT / config["sample_manifest"]).read_text())
    verify_manifest(manifest, config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "sample_count": len(manifest["samples"]),
                    "predict_arguments": prediction_arguments(config),
                    "next": "Run with --execute on MyGPU only",
                },
                indent=2,
            )
        )
        return
    started = datetime.now(timezone.utc)
    output = args.output or ROOT / "runs" / started.strftime(
        "yolov12s-car-%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.weights.with_suffix(".source.json").read_text())
    if args.weights.stat().st_size != config["weights_bytes"]:
        raise ValueError("Checkpoint size differs from the official release")
    weight_hash = sha256(args.weights)
    if receipt["url"] != config["weights_url"] or receipt["sha256"] != weight_hash:
        raise ValueError("Checkpoint differs from the recorded official download")
    detector, environment = build_detector(config, args.weights)
    predictor_class = build_square_predictor()
    (output / "architecture.txt").write_text(str(detector.model) + "\n")
    observed_inputs = []

    def check_input(_module, inputs):
        tensor = inputs[0]
        shape = list(tensor.shape)
        expected = [1, 3, config["image_size"], config["image_size"]]
        if shape != expected or tensor.device.type != "cuda":
            raise RuntimeError(f"Unexpected model input: {shape} on {tensor.device}")
        observed_inputs.append(
            {"shape": shape, "dtype": str(tensor.dtype), "device": str(tensor.device)}
        )

    hook = detector.model.register_forward_pre_hook(check_input)
    rows, predictions_by_image = [], []
    timer = time.monotonic()
    try:
        for index, sample in enumerate(manifest["samples"], 1):
            image_path = ROOT / sample["image"]
            gt, ignored = parse_annotations(
                (ROOT / sample["annotation"]).read_text(), config["visdrone_class_id"]
            )
            before = len(observed_inputs)
            result = detector.predict(
                source=str(image_path),
                predictor=predictor_class,
                **prediction_arguments(config),
            )[0]
            if len(observed_inputs) <= before:
                raise RuntimeError("The actual model input was not observed")
            predictions = []
            for xyxy, confidence, category in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                if int(category) != config["coco_class_id"]:
                    raise RuntimeError("Unexpected predicted class")
                predictions.append(
                    {
                        "xyxy": xyxy,
                        "confidence": confidence,
                        "coco_class_id": int(category),
                    }
                )
            panel = f"{index:02d}_{image_path.stem}.png"
            render_pair(image_path, gt, ignored, predictions, output / panel)
            rows.append(
                {
                    "image": image_path.name,
                    "ground_truth_car_count": len(gt),
                    "predicted_car_count": len(predictions),
                    "panel": panel,
                }
            )
            predictions_by_image.append(
                {
                    "image": sample["image"],
                    "ground_truth": gt,
                    "ignored_regions": ignored,
                    "predictions": predictions,
                }
            )
            print(
                f"{index:02d}/{len(manifest['samples'])}: {image_path.name}, GT={len(gt)}, predictions={len(predictions)}",
                flush=True,
            )
    finally:
        hook.remove()
    with (output / "counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "predictions.json").write_text(
        json.dumps(predictions_by_image, indent=2) + "\n"
    )
    runtime = {
        "status": "completed",
        "started_at": started.isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config": config,
        "sample_manifest_sha256": sha256(ROOT / config["sample_manifest"]),
        "weights_sha256": weight_hash,
        "environment": environment,
        "observed_inputs": observed_inputs,
        "images_processed": len(rows),
        "elapsed_including_io_and_render_seconds": time.monotonic() - timer,
        "scope": "Qualitative pretrained car preview; counts are not precision, recall or AP",
    }
    (output / "run.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
