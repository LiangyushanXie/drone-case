#!/usr/bin/env python3
"""Audit original VisDrone detection files without altering their annotations."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 6471, "val": 548, "test-dev": 1610}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    report = {
        "dataset": "VisDrone2019-DET",
        "splits": {},
        "fatal_errors": [],
        "warnings": [],
    }
    image_hashes = defaultdict(list)
    for split, expected in EXPECTED.items():
        directory = ROOT / "data/raw" / f"VisDrone2019-DET-{split}"
        images = sorted((directory / "images").glob("*.jpg"))
        labels = sorted((directory / "annotations").glob("*.txt"))
        stats = {
            "images": len(images),
            "annotation_files": len(labels),
            "boxes": 0,
            "ignored_regions": 0,
            "out_of_bounds_boxes": 0,
            "nonpositive_boxes": 0,
        }
        categories = Counter()
        if len(images) != expected or {p.stem for p in images} != {
            p.stem for p in labels
        }:
            report["fatal_errors"].append(
                f"{split}: expected {expected} paired images/annotations"
            )
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                image_hashes[sha256(image_path)].append(f"{split}/{image_path.name}")
                annotation = directory / "annotations" / (image_path.stem + ".txt")
                for line_number, line in enumerate(
                    annotation.read_text().splitlines(), 1
                ):
                    if not line.strip():
                        continue
                    values = [int(value) for value in line.rstrip(",").split(",")]
                    if len(values) != 8 or not 0 <= values[5] <= 11:
                        raise ValueError(f"invalid annotation at line {line_number}")
                    left, top, box_width, box_height, score, category, _, _ = values
                    stats["boxes"] += 1
                    categories[str(category)] += 1
                    stats["ignored_regions"] += int(score == 0 or category == 0)
                    stats["nonpositive_boxes"] += int(box_width <= 0 or box_height <= 0)
                    stats["out_of_bounds_boxes"] += int(
                        left < 0
                        or top < 0
                        or left + box_width > width
                        or top + box_height > height
                    )
            except (OSError, ValueError) as error:
                report["fatal_errors"].append(f"{split}/{image_path.name}: {error}")
        stats["category_ids"] = dict(sorted(categories.items()))
        report["splits"][split] = stats
        if stats["out_of_bounds_boxes"] or stats["nonpositive_boxes"]:
            report["warnings"].append(
                f"{split}: original box geometry requires review; no corrections made"
            )
    report["duplicate_image_groups"] = [
        paths for paths in image_hashes.values() if len(paths) > 1
    ]
    report["cross_split_duplicate_groups"] = [
        paths
        for paths in report["duplicate_image_groups"]
        if len({p.split("/")[0] for p in paths}) > 1
    ]
    if report["cross_split_duplicate_groups"]:
        report["warnings"].append(
            "Exact duplicate images cross official splits; do not silently change benchmark splits"
        )
    report["status"] = (
        "failed"
        if report["fatal_errors"]
        else "passed_with_warnings"
        if report["warnings"]
        else "passed"
    )
    (ROOT / "metadata/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if "groups" not in k}, indent=2))
    return bool(report["fatal_errors"])


if __name__ == "__main__":
    raise SystemExit(main())
