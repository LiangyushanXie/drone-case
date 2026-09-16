import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finetune_metrics import interpolated_ap, match_overlaps, summarize_records
from run_finetune import load_training_config, training_arguments
from run_finetune_suite import build_jobs, next_capacity_batch, wait_for_existing_batch8


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

    def test_larger_batch_has_same_lr_and_matching_effective_batch(self):
        config = load_training_config()
        for batch in (16, 20):
            args = training_arguments(config, Path("/tmp/data.yaml"), Path("/tmp/run"), batch=batch)
            self.assertEqual(
                (args["batch"], args["nbs"], args["lr0"], args["epochs"]),
                (batch, batch, 0.0003, 50),
            )

    def test_followup_never_queues_full_training(self):
        jobs = build_jobs(Path("/tmp/suite"))
        self.assertEqual([j["batch"] for j in jobs], [8, 16, 20])
        self.assertTrue(all(j["kind"] == "throughput_only" for j in jobs))
        self.assertEqual(len(set(j["output"] for j in jobs)), len(jobs))

    def test_capacity_stops_on_oom_or_instability_and_has_bound(self):
        results = [{"batch": b, "status": "completed", "stable": True} for b in (8, 16, 20)]
        self.assertEqual(next_capacity_batch(results), 24)
        self.assertIsNone(
            next_capacity_batch(results + [{"batch": 24, "status": "oom", "stable": False}])
        )
        self.assertIsNone(
            next_capacity_batch(results + [{"batch": 24, "status": "completed", "stable": False}])
        )
        self.assertIsNone(
            next_capacity_batch(results + [{"batch": 64, "status": "completed", "stable": True}])
        )

    def test_followup_waits_for_old_cuda_process_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(
                json.dumps({"status": "completed", "epochs_completed": 50, "smoke": False})
            )
            with (
                patch("run_finetune_suite.process_matches", side_effect=[True, False]),
                patch("run_finetune_suite.time.sleep") as sleep,
            ):
                self.assertEqual(wait_for_existing_batch8(root, 123)["epochs_completed"], 50)
                sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
