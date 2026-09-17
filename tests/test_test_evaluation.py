import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from data_utils import sha256
from run_finetune import evaluate
from run_test_evaluation import (
    draw_triptych,
    match_display,
    selected_test_records,
    test_dataset_document,
)


class TestSetComparisonTests(unittest.TestCase):
    def test_only_frozen_test_membership_in_original_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "id": "test-dev/a",
                    "split": "test-dev",
                    "included": True,
                    "derived_image": "images/test-dev/a.jpg",
                },
                {
                    "id": "val/b",
                    "split": "val",
                    "included": True,
                    "derived_image": "images/val/b.jpg",
                },
                {
                    "id": "test-dev/c",
                    "split": "test-dev",
                    "included": True,
                    "derived_image": "images/test-dev/c.jpg",
                },
            ]
            (root / "manifest.jsonl").write_text("\n".join(map(json.dumps, records)))
            (root / "test-dev.txt").write_text("./images/test-dev/a.jpg\n./images/test-dev/c.jpg\n")
            self.assertEqual(
                [r["id"] for r in selected_test_records(root, 2)], ["test-dev/a", "test-dev/c"]
            )
            (root / "test-dev.txt").write_text("./images/test-dev/c.jpg\n./images/test-dev/a.jpg\n")
            with self.assertRaises(ValueError):
                selected_test_records(root, 2)

    def test_test_yaml_does_not_replace_validation_membership(self):
        root = Path("/example/dataset")
        doc = test_dataset_document(root)
        self.assertEqual(doc["test"], str(root / "test-dev.txt"))
        self.assertEqual(doc["val"], str(root / "val.txt"))
        self.assertEqual(doc["names"], ["car"])

    def test_evaluation_default_and_explicit_test_split_without_ml_import(self):
        configurations = []

        class FakeValidator:
            def __init__(self, save_dir, args):
                configurations.append(args)
                self.summary = {"images": 1}
                self.records = []

            def __call__(self, model):
                pass

        fake = SimpleNamespace(
            IgnoreValidator=FakeValidator, write_json=lambda p, v: p.write_text(json.dumps(v))
        )
        config = {"imgsz": 960, "workers": 4, "eval_conf": 0.001, "nms_iou": 0.7, "max_det": 1000}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"finetune_runtime": fake}),
        ):
            evaluate("model", Path("/data.yaml"), Path(directory) / "val", config)
            evaluate("model", Path("/data.yaml"), Path(directory) / "test", config, split="test")
        self.assertEqual([c["split"] for c in configurations], ["val", "test"])
        self.assertTrue(all(c["imgsz"] == 960 and c["half"] is False for c in configurations))
        with self.assertRaises(ValueError):
            evaluate(None, None, None, config, split="train")

    def test_display_matches_tp_fp_fn_and_ignores(self):
        sidecar = {
            "cars": [[10, 10, 10, 10, 1, 4, 0, 0], [35, 10, 10, 10, 1, 4, 0, 1]],
            "ignored": [[0, 0, 5, 5, 0, 0, 0, 0]],
        }
        predictions = [
            [10, 10, 20, 20, 0.9, 0],
            [50, 10, 60, 20, 0.8, 0],
            [0, 0, 5, 5, 0.7, 0],
            [35, 10, 45, 20, 0.2, 0],
            [60, 20, 60, 20, 0.9, 0],
        ]
        _, _, filtered, match = match_display(sidecar, predictions)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(
            (match["tp"], match["fp"], match["fn"], match["ignored_predictions"]), (1, 1, 1, 1)
        )
        empty = match_display({"cars": [], "ignored": []}, [[1, 1, 4, 4, 0.8, 0]])[3]
        self.assertEqual((empty["tp"], empty["fp"], empty["fn"]), (0, 1, 0))

    def test_triptych_dimensions_and_source_preserved(self):
        sidecar = {"cars": [[10, 10, 10, 10, 1, 4, 0, 0]], "ignored": []}
        arm = match_display(sidecar, [[10, 10, 20, 20, 0.9, 0]])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            target = Path(directory) / "comparison.jpg"
            Image.new("RGB", (960, 540), "white").save(source)
            original = sha256(source)
            draw_triptych(source, arm, arm, target, "example.jpg")
            with Image.open(target) as image:
                self.assertEqual(image.size, (2880, 664))
            self.assertEqual(sha256(source), original)


if __name__ == "__main__":
    unittest.main()
