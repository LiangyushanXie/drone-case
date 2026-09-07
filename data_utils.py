"""Dependency-free VisDrone parsing and reproducible preview sampling."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from model_arch import ROOT, load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotations(text, car_class_id=4):
    cars, ignored = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = [int(item.strip()) for item in line.rstrip(",").split(",")]
        if len(fields) != 8:
            raise ValueError(f"Expected eight VisDrone fields at line {number}")
        x, y, width, height, score, category, truncation, occlusion = fields
        record = {
            "xyxy": [x, y, x + width, y + height],
            "category_id": category,
            "score": score,
            "truncation": truncation,
            "occlusion": occlusion,
            "valid_geometry": width > 0 and height > 0,
        }
        if score == 0 or category == 0:
            ignored.append(record)
        elif category == car_class_id:
            if not record["valid_geometry"]:
                raise ValueError(
                    f"Invalid scored car box at line {number}; do not silently repair it"
                )
            cars.append(record)
    return cars, ignored


def choose_dispersed_images(images, count, seed):
    groups = defaultdict(list)
    for image in sorted(images):
        groups[image.name.split("_")[0]].append(image)
    if not 1 <= count <= len(groups):
        raise ValueError(
            "Need at least one distinct filename-prefix group per selected image"
        )
    rng = random.Random(seed)
    chosen_groups = rng.sample(sorted(groups), count)
    return sorted(rng.choice(groups[group]) for group in chosen_groups)


def prepare_manifest(config):
    directory = ROOT / config["validation_directory"]
    images = choose_dispersed_images(
        list((directory / "images").glob("*.jpg")),
        config["sample_count"],
        config["sample_seed"],
    )
    records = []
    for image in images:
        annotation = directory / "annotations" / (image.stem + ".txt")
        cars, ignored = parse_annotations(
            annotation.read_text(), config["visdrone_class_id"]
        )
        records.append(
            {
                "image": image.relative_to(ROOT).as_posix(),
                "annotation": annotation.relative_to(ROOT).as_posix(),
                "filename_group": image.name.split("_")[0],
                "image_sha256": sha256(image),
                "annotation_sha256": sha256(annotation),
                "gt_car_count": len(cars),
                "ignored_region_count": len(ignored),
            }
        )
    return {
        "selection": "seeded selection of 30 filename-prefix groups, then one image per group; no filtering by car count",
        "limitation": "Filename prefixes are a dispersion heuristic, not verified independent scenes",
        "seed": config["sample_seed"],
        "samples": records,
    }


def verify_manifest(manifest, config):
    samples = manifest["samples"]
    if len(samples) != config["sample_count"] or len(
        {s["image"] for s in samples}
    ) != len(samples):
        raise ValueError("Unexpected sample count or duplicate samples")
    groups = {Path(sample["image"]).name.split("_")[0] for sample in samples}
    if len(groups) != len(samples):
        raise ValueError("Samples must come from distinct filename-prefix groups")
    validation_root = (ROOT / config["validation_directory"]).resolve()
    for sample in samples:
        if Path(sample["image"]).stem != Path(sample["annotation"]).stem:
            raise ValueError("Image and annotation filenames do not match")
        for kind in ("image", "annotation"):
            path = (ROOT / sample[kind]).resolve()
            if validation_root not in path.parents:
                raise ValueError("Sample path escapes the validation dataset")
            if sha256(path) != sample[kind + "_sha256"]:
                raise ValueError(f"Sample changed: {sample[kind]}")


if __name__ == "__main__":
    config = load_config()
    manifest = prepare_manifest(config)
    target = ROOT / config["sample_manifest"]
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['samples'])} samples to {target.relative_to(ROOT)}")
