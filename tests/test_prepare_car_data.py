"""Semantic checks for car-only conversion and reviewed exceptions."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import prepare_car_data as preparation
from prepare_car_data import EXCLUDED, ignore_mask, parse_rows, yolo_text


class CarDataTests(unittest.TestCase):
    def test_class_mapping_and_roundtrip(self):
        rows, cars, ignored, warnings = parse_rows(
            "10,20,30,40,1,4,0,1\n2,3,4,5,1,5,0,0", (200, 100), "train/test"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            yolo_text(cars, (200, 100)), "0 0.125000000 0.400000000 0.150000000 0.400000000\n"
        )
        self.assertFalse(ignored or warnings)

    def test_score_zero_is_ignored_even_for_car(self):
        _, cars, ignored, _ = parse_rows("1,2,3,4,0,4,0,0\n6,2,3,4,1,0,0,0", (20, 20), "x")
        self.assertFalse(cars)
        self.assertEqual(len(ignored), 2)
        self.assertEqual(yolo_text(cars, (20, 20)), "")

    def test_union_mask_has_half_open_edges(self):
        mask = ignore_mask((10, 8), [[1, 2, 3, 2], [3, 2, 3, 2]])
        self.assertEqual(mask.histogram()[255], 10)
        self.assertEqual(mask.getpixel((5, 3)), 255)
        self.assertEqual(mask.getpixel((6, 3)), 0)
        self.assertEqual(mask.getpixel((1, 4)), 0)

    def test_positive_overlapping_ignore_is_preserved(self):
        _, cars, ignored, _ = parse_rows("1,1,5,5,1,4,0,2\n0,0,9,9,0,0,0,0", (10, 10), "x")
        self.assertEqual(len(cars), 1)
        self.assertEqual(ignore_mask((10, 10), ignored).getpixel((2, 2)), 255)

    def test_unknown_bad_geometry_and_out_of_bounds_fail(self):
        for line in ("1,1,4,0,1,4,0,0", "9,9,4,4,1,4,0,0", "-1,1,3,3,1,4,0,0"):
            with self.assertRaises(ValueError):
                parse_rows(line, (10, 10), "unreviewed")

    def test_known_degenerate_region_is_logged_not_expanded(self):
        text = "\n" * 129 + "1008,374,3,0,0,0,0,0"
        _, cars, ignored, warnings = parse_rows(text, (1920, 1080), "train/0000293_03401_d_0000939")
        self.assertFalse(cars or ignored)
        self.assertEqual(warnings[0]["line"], 130)
        self.assertEqual(ignore_mask((1920, 1080), ignored).getbbox(), None)

    def test_only_approved_versions_are_excluded(self):
        self.assertEqual(len(EXCLUDED), 4)
        for retained in (
            "train/0000239_06450_d_0000017",
            "train/9999950_00000_d_0000079",
            "val/0000023_00000_d_0000008",
        ):
            self.assertNotIn(retained, EXCLUDED)


class PipelineTests(unittest.TestCase):
    def test_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/VisDrone2019-DET-train"
            (raw / "images").mkdir(parents=True)
            (raw / "annotations").mkdir()
            (root / "metadata").mkdir()
            Image.new("RGB", (20, 10), "red").save(raw / "images/a.jpg")
            Image.new("RGB", (20, 10), "blue").save(raw / "images/b.jpg")
            (raw / "annotations/a.txt").write_text("2,2,4,3,1,4,0,1\n0,0,8,8,0,0,0,0\n")
            (raw / "annotations/b.txt").write_text("")
            (root / "metadata/files.sha256").write_text(
                "".join(
                    preparation.digest(p) + "  " + p.relative_to(root).as_posix() + "\n"
                    for p in sorted(raw.rglob("*"))
                    if p.is_file()
                )
            )
            output = root / "prepared"
            with (
                patch.object(preparation, "COUNTS", {"train": 2}),
                patch.object(preparation, "EXCLUDED", {}),
                patch.object(preparation, "DUPLICATES", []),
                patch.object(preparation, "create_previews"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                preparation.prepare(root, output)
                preparation.verify(root, output)
                with self.assertRaises(FileExistsError):
                    preparation.prepare(root, output)
                label = output / "labels/train/a.txt"
                original = label.read_text()
                label.write_text("0 0.1 0.1 0.1 0.1\n")
                with self.assertRaisesRegex(ValueError, "Derived asset changed"):
                    preparation.verify(root, output)
                label.write_text(original)
                (raw / "annotations/a.txt").write_text("")
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    preparation.verify(root, output)


if __name__ == "__main__":
    unittest.main()
