import copy
import json
import tempfile
import unittest
from pathlib import Path

from compare_inputs import load_settings, summarize
from comparison_utils import (
    diagnostic_match,
    merge_tile_predictions,
    recovery_counts,
    restore_tile_predictions,
    size_group,
    tile_windows,
    union_coverage,
)
from comparison_visuals import render_comparison

try:
    from PIL import Image
except ImportError:
    Image = None


def gt(box, occlusion=0):
    return {"xyxy": box, "occlusion": occlusion, "valid_geometry": True}


def prediction(box, confidence=0.9):
    return {"xyxy": box, "confidence": confidence, "coco_class_id": 2}


class TileTests(unittest.TestCase):
    def test_tiles_cover_edges_without_duplicate_windows(self):
        windows = tile_windows(43, 31, 16, 0.25)
        covered = set()
        for x1, y1, x2, y2 in windows:
            self.assertLessEqual(x2, 43)
            self.assertLessEqual(y2, 31)
            self.assertEqual((x2 - x1, y2 - y1), (16, 16))
            covered.update((x, y) for y in range(y1, y2) for x in range(x1, x2))
        self.assertEqual(len(covered), 43 * 31)
        self.assertEqual(len(windows), len({tuple(window) for window in windows}))

    def test_image_smaller_than_tile_is_used_once(self):
        self.assertEqual(tile_windows(80, 40, 640, 0.2), [[0, 0, 80, 40]])
        with self.assertRaises(ValueError):
            tile_windows(80, 40, 640, 1)

    def test_restore_then_nms_removes_duplicate_but_keeps_adjacent_car(self):
        first = restore_tile_predictions(
            [prediction([20, 10, 40, 30], 0.8)], [100, 50, 740, 690], (800, 700), 0
        )
        second = restore_tile_predictions(
            [prediction([0, 0, 20, 20], 0.9), prediction([22, 0, 42, 20], 0.7)],
            [120, 60, 760, 700],
            (800, 700),
            1,
        )
        boxes, counts = merge_tile_predictions(first + second, 0.7, 300)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["xyxy"], [120, 60, 140, 80])
        self.assertEqual(boxes[0]["confidence"], 0.9)
        self.assertEqual(counts["removed_by_nms"], 1)
        self.assertEqual(counts["removed_by_cap"], 0)

    def test_nms_boundary_and_cap_are_separate(self):
        boxes = [prediction([0, 0, 3, 3]), prediction([1, 0, 4, 3], 0.8)]
        kept, counts = merge_tile_predictions(boxes, 0.5, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(counts["after_nms"], 2)
        self.assertEqual(counts["removed_by_cap"], 1)


class MatchingTests(unittest.TestCase):
    def test_ignore_union_does_not_double_count_overlaps(self):
        self.assertAlmostEqual(union_coverage([0, 0, 10, 10], [[0, 0, 3, 10], [0, 0, 3, 10]]), 0.3)
        self.assertAlmostEqual(union_coverage([0, 0, 10, 10], [[0, 0, 3, 10], [7, 0, 10, 10]]), 0.6)

    def test_matching_is_confidence_ordered_and_one_to_one(self):
        boxes = [prediction([0, 0, 10, 10], 0.8), prediction([0, 0, 10, 10], 0.9)]
        result = diagnostic_match([gt([0, 0, 10, 10]), gt([20, 20, 30, 30])], boxes, [], 0.5, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["gt_states"][0]["prediction_index"], 1)
        self.assertEqual(result["prediction_states"][0]["status"], "fp")

    def test_ignored_objects_and_predictions_do_not_inflate_scores(self):
        ignored = [gt([0, 0, 6, 10])]
        result = diagnostic_match(
            [gt([0, 0, 10, 10]), gt([20, 20, 30, 30])],
            [prediction([0, 0, 10, 10])],
            ignored,
            0.5,
            0.5,
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
        self.assertEqual((result["ignored_gt"], result["ignored_predictions"]), (1, 1))

    def test_negative_images_do_not_get_artificial_perfect_recall(self):
        empty = diagnostic_match([], [], [], 0.5, 0.5)
        false_alarm = diagnostic_match([], [prediction([0, 0, 10, 10])], [], 0.5, 0.5)
        self.assertIsNone(empty["precision"])
        self.assertIsNone(empty["recall"])
        self.assertIsNone(false_alarm["recall"])
        self.assertEqual(false_alarm["fp"], 1)
        self.assertEqual(false_alarm["precision"], 0)

    def test_match_threshold_includes_equality(self):
        result = diagnostic_match([gt([0, 0, 3, 3])], [prediction([1, 0, 4, 3])], [], 0.5, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["prediction_states"][0]["iou"], 0.5)

    def test_recovery_keeps_gains_and_losses_separate(self):
        baseline = {"gt_states": [{"status": "tp"}, {"status": "fn"}], "fp": 1}
        changed = {"gt_states": [{"status": "fn"}, {"status": "tp"}], "fp": 2}
        self.assertEqual(
            recovery_counts(baseline, changed),
            {
                "recovered_gt": 1,
                "lost_gt": 1,
                "retained_gt": 0,
                "still_missed_gt": 0,
                "fp_delta": 1,
            },
        )
        with self.assertRaises(ValueError):
            recovery_counts(baseline, {"gt_states": [], "fp": 0})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.settings, self.base = load_settings(
            Path(__file__).resolve().parents[1] / "configs/input_comparison.json"
        )
        self.ground_truth = [gt([20, 30, 40, 50]), gt([60, 50, 70, 60], 2)]
        self.arms = {}
        for name, boxes in {
            "baseline": [prediction([20, 30, 40, 50])],
            "resolution": [prediction([20, 30, 40, 50]), prediction([60, 50, 70, 60])],
            "sliced": [prediction([20, 30, 40, 50]), prediction([200, 140, 220, 160])],
        }.items():
            self.arms[name] = [
                {
                    "index": i + 1,
                    "image": f"{i}.jpg",
                    "image_size": [320, 192],
                    "ground_truth": self.ground_truth,
                    "ignored_regions": [],
                    "predictions": boxes,
                    "evaluation": diagnostic_match(self.ground_truth, boxes, [], 0.5, 0.5),
                    "predict_calls": 1,
                    "predict_and_merge_seconds": 0.01,
                }
                for i in range(2)
            ]

    def test_micro_metrics_groups_and_recovered_gt(self):
        summary, rows, groups = summarize(self.arms, self.settings, self.base)
        self.assertEqual(summary["arms"]["baseline"]["recall"], 0.5)
        self.assertEqual(summary["arms"]["resolution"]["recall"], 1.0)
        self.assertEqual(summary["arms"]["sliced"]["precision"], 0.5)
        self.assertEqual(summary["recovery_vs_baseline"]["resolution"]["recovered_gt"], 2)
        self.assertEqual(len(rows), 6)
        for name in self.arms:
            self.assertEqual(
                sum(
                    row["gt"]
                    for row in groups
                    if row["arm"] == name and row["kind"] == "size_at_baseline_640"
                ),
                4,
            )
        self.assertEqual(size_group([0, 0, 32, 32], (1920, 1080), 640, [16, 32, 96]), "lt16")

    def test_reordered_images_are_rejected(self):
        changed = copy.deepcopy(self.arms)
        changed["resolution"].reverse()
        with self.assertRaises(ValueError):
            summarize(changed, self.settings, self.base)

    def test_unapproved_full_image_fusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = {**self.settings, "include_full_image_in_sliced": True}
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(settings))
            with self.assertRaises(ValueError):
                load_settings(path)

    @unittest.skipIf(Image is None, "Pillow is needed for rendering")
    def test_four_panels_preserve_pixels_and_gt_prediction_colors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (320, 192), "#202020").save(image)
            destination = root / "comparison.png"
            render_comparison(
                image,
                {name: records[0] for name, records in self.arms.items()},
                self.settings,
                destination,
            )
            with Image.open(destination) as result:
                self.assertEqual(result.size, (1316, 255))
                self.assertEqual(result.getpixel((20, 93)), (24, 199, 122))
                self.assertEqual(result.getpixel((352, 93)), (255, 157, 46))
                self.assertEqual(result.getpixel((1196, 203)), (239, 83, 80))


if __name__ == "__main__":
    unittest.main()
