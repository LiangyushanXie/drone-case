import unittest
from pathlib import Path

from finetune_metrics import interpolated_ap, match_overlaps, summarize_records
from run_finetune import load_training_config, training_arguments


class FineTuneTests(unittest.TestCase):
    def test_greedy_one_to_one_and_thresholds(self):
        matches = match_overlaps([[0.8, 0.1], [0.9, 0.2], [0.1, 0.6]])
        self.assertTrue(matches[0][5])
        self.assertFalse(matches[1][0])
        self.assertTrue(matches[2][0])
        self.assertFalse(matches[2][5])

    def test_iou_tie_uses_lowest_gt_index(self):
        self.assertEqual(match_overlaps([[0.8, 0.8], [0.8, 0.1]], (0.5,)), [[True], [False]])

    def test_ap_known_curves(self):
        self.assertEqual(interpolated_ap([True, True], 2), 1.0)
        self.assertEqual(interpolated_ap([False, False], 2), 0.0)
        self.assertEqual(interpolated_ap([False, True], 1), 0.5)
        self.assertEqual(interpolated_ap([], 1), 0.0)
        self.assertIsNone(interpolated_ap([], 0))

    def test_operating_point_counts(self):
        summary = summarize_records(
            [
                {
                    "evaluated_gt": 2,
                    "ignored_gt": 1,
                    "ignored_predictions": 1,
                    "hit_prediction_cap": False,
                    "confidence": [0.8, 0.7],
                    "correct": [[True] * 10, [False] * 10],
                }
            ]
        )
        self.assertEqual(summary["operating_points"][0]["tp"], 1)
        self.assertEqual(summary["operating_points"][0]["fp"], 1)
        self.assertEqual(summary["operating_points"][0]["fn"], 1)
        self.assertEqual(summary["operating_points"][2]["fp"], 0)

    def test_explicit_training_controls(self):
        c = load_training_config()
        a = training_arguments(c, Path("/tmp/data.yaml"), Path("/tmp/run"))
        self.assertEqual((a["epochs"], a["imgsz"], a["freeze"]), (50, 960, 0))
        self.assertEqual((a["optimizer"], a["lr0"], a["nbs"]), ("AdamW", 0.0003, 8))
        self.assertEqual(
            (a["mosaic"], a["mixup"], a["warmup_bias_lr"], a["patience"]), (0, 0, 0, 0)
        )
        self.assertEqual(
            training_arguments(c, Path("/tmp/data.yaml"), Path("/tmp/run"), batch=4)["nbs"], 8
        )


if __name__ == "__main__":
    unittest.main()
