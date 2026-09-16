"""Prepare a frozen car-only view; ignore-aware training is a separate step."""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
COUNTS = {"train": 6471, "val": 548, "test-dev": 1610}
EXCLUDED = {
    "train/9999985_00000_d_0000020": "invalid scored car (line 12, width 4 height 0); whole image excluded",
    "train/0000239_06950_d_0000018": "duplicate; user visually selected 33-car version 0000239_06450_d_0000017",
    "train/9999950_00000_d_10000080": "night duplicate; user permits either, keep lexical first 9999950_00000_d_0000079",
    "val/0000022_00000_d_0000004": "duplicate; user selected car-labeled 0000023_00000_d_0000008",
}
ZERO_AREA = {
    ("train/9999985_00000_d_0000020", 12): [611, 158, 4, 0, 1, 4, 0, 0],
    ("train/0000293_03401_d_0000939", 130): [1008, 374, 3, 0, 0, 0, 0, 0],
    ("train/9999999_00590_d_0000267", 89): [545, 414, 10, 0, 0, 0, 0, 0],
}
DUPLICATES = [
    ["train/0000239_06450_d_0000017", "train/0000239_06950_d_0000018"],
    ["train/9999950_00000_d_0000079", "train/9999950_00000_d_10000080"],
    ["val/0000022_00000_d_0000004", "val/0000023_00000_d_0000008"],
]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_rows(text, size, key):
    rows, cars, ignored, warnings = [], [], [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = [int(value) for value in line.rstrip(",").split(",")]
        if len(row) != 8:
            raise ValueError(f"{key}:{number}: expected eight fields")
        x, y, width, height, score, category, truncation, occlusion = row
        if not (
            0 <= category <= 11
            and score in (0, 1)
            and truncation in (0, 1, 2)
            and occlusion in (0, 1, 2)
        ):
            raise ValueError(f"{key}:{number}: unexpected attributes")
        if x < 0 or y < 0 or x + width > size[0] or y + height > size[1]:
            raise ValueError(f"{key}:{number}: out of bounds; review instead of clipping")
        rows.append(row)
        if width <= 0 or height <= 0:
            if ZERO_AREA.get((key, number)) != row:
                raise ValueError(f"{key}:{number}: unreviewed invalid geometry")
            warnings.append({"line": number, "row": row, "action": "no rasterized area"})
            continue
        if score == 0 or category == 0:
            ignored.append(row)
        elif category == 4:
            cars.append(row)
    return rows, cars, ignored, warnings


def yolo_text(cars, size):
    lines = []
    for x, y, width, height, *_ in cars:
        values = (
            (x + width / 2) / size[0],
            (y + height / 2) / size[1],
            width / size[0],
            height / size[1],
        )
        lines.append("0 " + " ".join(f"{value:.9f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def ignore_mask(size, rows):
    mask = Image.new("L", size, 0)
    for x, y, width, height, *_ in rows:
        # PIL paste uses half-open bounds, matching integer source xywh.
        mask.paste(255, (x, y, x + width, y + height))
    return mask


def audit_sources(root):
    expected = {}
    for line in (root / "metadata/files.sha256").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        expected[relative] = checksum
    records, hashes, groups, seen_paths = [], defaultdict(list), defaultdict(set), set()
    for split, count in COUNTS.items():
        raw = root / "data/raw" / f"VisDrone2019-DET-{split}"
        images = sorted((raw / "images").glob("*.jpg"))
        labels = sorted((raw / "annotations").glob("*.txt"))
        if len(images) != count or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f"{split}: unexpected source membership")
        for image in images:
            label = raw / "annotations" / (image.stem + ".txt")
            key = f"{split}/{image.stem}"
            record = {"id": key, "split": split, "included": key not in EXCLUDED}
            for kind, path in (("image", image), ("annotation", label)):
                relative = path.relative_to(root).as_posix()
                actual = digest(path)
                if expected.get(relative) != actual:
                    raise ValueError(f"Source changed or absent from checksums: {relative}")
                record[kind] = relative
                record[kind + "_sha256"] = actual
                seen_paths.add(relative)
            with Image.open(image) as picture:
                picture.load()
                record["size"] = list(picture.size)
            rows, cars, ignored, warnings = parse_rows(label.read_text(), record["size"], key)
            record.update(rows=rows, cars=cars, ignored=ignored, warnings=warnings)
            if key in EXCLUDED:
                record["exclusion_reason"] = EXCLUDED[key]
            records.append(record)
            hashes[record["image_sha256"]].append(key)
            groups[split].add(image.stem.split("_")[0])
        print(f"Audited {split}: {count} image/annotation pairs", flush=True)
    if seen_paths != set(expected):
        raise ValueError("Checksum manifest membership differs from source pairs")
    duplicates = sorted(sorted(group) for group in hashes.values() if len(group) > 1)
    if duplicates != sorted(sorted(group) for group in DUPLICATES):
        raise ValueError("Unexpected duplicate groups; review policy first")
    return records, {
        "duplicate_groups": duplicates,
        "cross_split_exact_duplicates": [],
        "train_val_shared_filename_prefixes": sorted(groups["train"] & groups["val"]),
        "prefix_warning": "Prefixes neither prove nor rule out scene leakage; no scene independence claim",
    }


def create_previews(output, records):
    chosen = {}
    training = [r for r in records if r["included"] and r["split"] == "train"]
    validation = [r for r in records if r["included"] and r["split"] == "val"]
    chosen["small_dense"] = max(training, key=lambda r: sum(c[2] * c[3] < 32**2 for c in r["cars"]))
    chosen["occluded"] = max(training, key=lambda r: sum(c[7] == 2 for c in r["cars"]))
    chosen["clear_large"] = next(
        r
        for r in training
        if 3 <= len(r["cars"]) <= 15
        and sum(c[2] * c[3] >= 96**2 and c[7] == 0 for c in r["cars"]) >= 3
    )
    chosen["no_car"] = next(r for r in training if not r["cars"] and not r["ignored"])
    chosen["ignore_regions"] = next(r for r in training if r["ignored"] and len(r["cars"]) >= 10)
    chosen["validation"] = next(r for r in validation if len(r["cars"]) >= 20)
    chosen["retained_small_cars"] = next(r for r in training if r["id"] == DUPLICATES[0][0])
    directory = output / "previews"
    directory.mkdir()
    links = []
    for name, record in chosen.items():
        picture = Image.open(output / record["derived_image"]).convert("RGB")
        draw = ImageDraw.Draw(picture)
        for row in record["ignored"]:
            x, y, width, height = row[:4]
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#AAAAAA", width=3)
        for row in record["cars"]:
            x, y, width, height = row[:4]
            draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#00FF66", width=2)
        picture.thumbnail((1600, 1100))
        canvas = Image.new("RGB", (max(900, picture.width), picture.height + 55), "#111820")
        canvas.paste(picture, (0, 55))
        header = f"{name} | {record['id']} | cars={len(record['cars'])} | ignore={len(record['ignored'])}"
        ImageDraw.Draw(canvas).text(
            (12, 9),
            header
            + "\nGREEN: source car   GRAY: ignore regions   Annotation preview, not predictions",
            fill="white",
        )
        target = directory / (name + ".png")
        canvas.save(target)
        links.append(
            {
                "case": name,
                "id": record["id"],
                "preview": target.relative_to(output).as_posix(),
                "original_image": record["image"],
                "original_annotation": record["annotation"],
                "derived_image": record["derived_image"],
            }
        )
    write_json(directory / "sources.json", links)


def prepare(root, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {output}")
    records, audit = audit_sources(root)
    output.mkdir(parents=True)
    summaries = {}
    for split in COUNTS:
        for folder in ("images", "labels", "annotations", "ignore_masks", "sidecars"):
            (output / folder / split).mkdir(parents=True)
        included = [r for r in records if r["split"] == split and r["included"]]
        stats = Counter(images=len(included), cars=sum(len(r["cars"]) for r in included))
        occlusion = Counter()
        sizes = Counter()
        for record in included:
            stem = record["id"].split("/")[1]
            record["derived_image"] = f"images/{split}/{stem}.jpg"
            record["derived_label"] = f"labels/{split}/{stem}.txt"
            record["derived_annotation"] = f"annotations/{split}/{stem}.txt"
            record["ignore_mask"] = f"ignore_masks/{split}/{stem}.png"
            record["sidecar"] = f"sidecars/{split}/{stem}.json"
            shutil.copy2(root / record["image"], output / record["derived_image"])
            shutil.copy2(root / record["annotation"], output / record["derived_annotation"])
            (output / record["derived_label"]).write_text(yolo_text(record["cars"], record["size"]))
            mask = ignore_mask(record["size"], record["ignored"])
            mask.save(output / record["ignore_mask"], compress_level=1)
            stats["negative_images"] += not record["cars"]
            stats["images_with_ignore"] += bool(record["ignored"])
            stats["ignored_regions"] += len(record["ignored"])
            for x, y, width, height, _, _, _, occluded in record["cars"]:
                occlusion[str(occluded)] += 1
                area = width * height
                sizes[
                    "small_lt32" if area < 32**2 else "medium_lt96" if area < 96**2 else "large"
                ] += 1
                histogram = mask.crop((x, y, x + width, y + height)).histogram()
                stats["cars_overlapping_ignore"] += histogram[255] > 0
                stats["cars_ignore_coverage_ge_half"] += histogram[255] >= area / 2
            write_json(output / record["sidecar"], record)
        summaries[split] = {
            **dict(stats),
            "occlusion": dict(occlusion),
            "native_pixel_size_bins": dict(sizes),
        }
        (output / f"{split}.txt").write_text("".join(f"./{r['derived_image']}\n" for r in included))
        print(f"Prepared {split}: {len(included)} images", flush=True)
    audit.update(
        splits=summaries,
        exclusions=EXCLUDED,
        geometry_warnings=[{"id": r["id"], **w} for r in records for w in r["warnings"]],
    )
    write_json(output / "audit.json", audit)
    create_previews(output, records)
    # A record for every raw pair preserves exclusion provenance as well as included membership.
    with (output / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    checksums = {
        p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(output / "checksums.json", checksums)
    write_json(
        output / "dataset_spec.json",
        {
            "dataset": "visdrone-car-v1",
            "status": "prepared_not_training_ready",
            "requires_ignore_aware_trainer": True,
            "training_launch_ready": False,
            "source_category_to_training_class": {"4": 0},
            "names": ["car"],
            "source_checksums_sha256": digest(root / "metadata/files.sha256"),
            "manifest_sha256": digest(output / "manifest.jsonl"),
            "asset_checksums_sha256": digest(output / "checksums.json"),
            "mask_contract": "L PNG, 0 valid, 255 ignored, original image dimensions; integer xywh half-open rectangles",
            "loss_contract": "Trainer must transform images, boxes and masks together; scored cars remain positive even when overlapping ignore. Define foreground precedence and suppress ignored background loss; plain YOLO TXT does not implement this.",
            "splits": {split: f"{split}.txt" for split in COUNTS},
            "evaluation_contract": "Freeze cleaned val for BOTH pretrained and fine-tuned evaluation with identical matching/NMS/confidence/ignore rules; not the raw official benchmark. Reserve test-dev for final evaluation.",
            "raw_counts": COUNTS,
            "included_counts": {s: v["images"] for s, v in summaries.items()},
        },
    )
    print(json.dumps(audit["splits"], indent=2), flush=True)


def verify(root, output):
    spec = json.loads((output / "dataset_spec.json").read_text())
    if not spec["requires_ignore_aware_trainer"] or spec["training_launch_ready"]:
        raise ValueError("Unexpected training contract")
    if digest(root / "metadata/files.sha256") != spec["source_checksums_sha256"]:
        raise ValueError("Source checksum manifest changed")
    for file, key in (
        ("manifest.jsonl", "manifest_sha256"),
        ("checksums.json", "asset_checksums_sha256"),
    ):
        if digest(output / file) != spec[key]:
            raise ValueError(f"Changed frozen {file}")
    checksums = json.loads((output / "checksums.json").read_text())
    actual_paths = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual_paths != set(checksums) | {"checksums.json", "dataset_spec.json"}:
        raise ValueError("Unexpected derived file membership")
    for relative, checksum in checksums.items():
        if digest(output / relative) != checksum:
            raise ValueError(f"Derived asset changed: {relative}")
    records, _ = audit_sources(root)
    saved = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    if [r["id"] for r in records] != [r["id"] for r in saved]:
        raise ValueError("Source manifest membership mismatch")
    max_error = 0.0
    for raw, record in zip(records, saved):
        if any(record.get(key) != value for key, value in raw.items()):
            raise ValueError(f"Source record mismatch: {raw['id']}")
        if not record["included"]:
            continue
        for kind in ("image", "annotation"):
            if digest(output / record["derived_" + kind]) != raw[kind + "_sha256"]:
                raise ValueError("Original bytes were not preserved")
        lines = (output / record["derived_label"]).read_text().splitlines()
        if len(lines) != len(raw["cars"]):
            raise ValueError("Label count mismatch")
        width, height = raw["size"]
        for line, car in zip(lines, raw["cars"]):
            category, xc, yc, bw, bh = map(float, line.split())
            if category != 0 or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)) or min(bw, bh) <= 0:
                raise ValueError("Invalid normalized label")
            restored = [(xc - bw / 2) * width, (yc - bh / 2) * height, bw * width, bh * height]
            error = max(abs(a - b) for a, b in zip(restored, car[:4]))
            max_error = max(max_error, error)
            if error > 0.00001:
                raise ValueError("Label roundtrip changed geometry")
        with Image.open(output / record["ignore_mask"]) as mask:
            expected = ignore_mask(raw["size"], raw["ignored"])
            if (
                mask.mode != "L"
                or mask.size != tuple(raw["size"])
                or mask.tobytes() != expected.tobytes()
            ):
                raise ValueError("Ignore mask differs from original regions")
    for split in COUNTS:
        members = [r for r in saved if r["included"] and r["split"] == split]
        expected = [f"./{r['derived_image']}" for r in members]
        if (output / f"{split}.txt").read_text().splitlines() != expected:
            raise ValueError("Split membership/order changed")
        if len(members) != spec["included_counts"][split]:
            raise ValueError("Split count mismatch")
    print(
        json.dumps(
            {
                "status": "verified",
                "source_pairs": len(saved),
                "included": spec["included_counts"],
                "max_box_roundtrip_error_pixels": max_error,
                "original_bytes_preserved": True,
                "all_masks_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared/visdrone-car-v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (verify if args.verify else prepare)(ROOT, args.output.resolve())
