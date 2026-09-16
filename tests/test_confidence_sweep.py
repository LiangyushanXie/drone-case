import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from data_utils import sha256
from sweep_confidence import analyze, sweep_records, validated_thresholds


class ConfidenceSweepTests(unittest.TestCase):
    def test_invalid_or_unrecoverable_thresholds_are_rejected(self):
        for values in ([], [0.2], [1.1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validated_thresholds(values, 0.25)
        self.assertEqual(validated_thresholds([0.45, 0.35, 0.35], 0.25), [0.25, 0.35, 0.45])

    def test_cutoff_equality_and_fresh_matches_without_mutating_source(self):
        records = [
            {
                "index": 1,
                "image": "fixture.jpg",
                "ground_truth": [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [20, 0, 30, 10]}],
                "ignored_regions": [{"xyxy": [40, 0, 50, 10], "valid_geometry": True}],
                "predictions": [
                    {"xyxy": [0, 0, 10, 10], "confidence": 0.45},
                    {"xyxy": [20, 0, 30, 10], "confidence": 0.35},
                    {"xyxy": [0, 0, 10, 10], "confidence": 0.30},
                    {"xyxy": [60, 0, 70, 10], "confidence": 0.25},
                    {"xyxy": [40, 0, 50, 10], "confidence": 0.50},
                ],
                "evaluation": {"tp": -999},
            }
        ]
        before = copy.deepcopy(records)
        totals, details = sweep_records(records, [0.25, 0.35, 0.45], 0.5, 0.5)
        self.assertEqual(
            [(r["tp"], r["fp"], r["fn"]) for r in totals], [(2, 2, 0), (2, 0, 0), (1, 0, 1)]
        )
        self.assertEqual([r["ignored_predictions"] for r in totals], [1, 1, 1])
        self.assertEqual(totals[2]["tp_lost_vs_source"], 1)
        self.assertEqual(totals[2]["fp_removed_vs_source"], 2)
        self.assertEqual(len(details), 3)
        self.assertEqual(records, before)

    def test_no_predictions_produces_undefined_precision_and_zero_recall(self):
        records = [
            {
                "index": 1,
                "image": "fixture.jpg",
                "ground_truth": [{"xyxy": [0, 0, 10, 10]}],
                "ignored_regions": [],
                "predictions": [{"xyxy": [0, 0, 10, 10], "confidence": 0.3}],
            }
        ]
        totals, _ = sweep_records(records, [0.25, 0.45], 0.5, 0.5)
        self.assertIsNone(totals[1]["precision"])
        self.assertEqual(totals[1]["recall"], 0)

    def test_end_to_end_tables_provenance_and_source_replay_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "inputs.json").write_text(json.dumps({"samples": [{"image": "fixture.jpg"}]}))
            runtime = {
                "status": "completed",
                "commit": "original-model-commit",
                "base_config": {
                    "sample_count": 1,
                    "confidence": 0.25,
                    "coco_class_id": 2,
                    "nms_iou": 0.7,
                },
                "comparison_config": {
                    "match_iou": 0.5,
                    "ignore_coverage": 0.5,
                    "tile_merge_nms_iou": 0.7,
                },
                "sample_manifest_sha256": sha256(source / "inputs.json"),
                "weights_sha256": "fixed-weight-hash",
            }
            record = {
                "index": 1,
                "image": "fixture.jpg",
                "ground_truth": [{"xyxy": [0, 0, 10, 10]}],
                "ignored_regions": [],
                "predictions": [{"xyxy": [0, 0, 10, 10], "confidence": 0.35, "coco_class_id": 2}],
            }
            source_totals, _ = sweep_records([record], [0.25], 0.5, 0.5)
            source_summary = {"arms": {}}
            for arm in ("baseline", "resolution", "sliced"):
                (source / f"{arm}.jsonl").write_text(json.dumps(record) + "\n")
                source_summary["arms"][arm] = source_totals[0].copy()
            (source / "run.json").write_text(json.dumps(runtime))
            (source / "summary.json").write_text(json.dumps(source_summary))
            hashes = {p.name: sha256(p) for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                rows = analyze(source, root / "output", [0.25, 0.35, 0.45])
            self.assertEqual(len(rows), 9)
            metadata = json.loads((root / "output/analysis.json").read_text())
            self.assertEqual(metadata["protocol"]["matching_iou"], 0.5)
            self.assertEqual(metadata["protocol"]["per_view_nms_iou"], 0.7)
            self.assertEqual(metadata["protocol"]["model_forward_calls"], 0)
            self.assertEqual(metadata["protocol"]["source_model_commit"], "original-model-commit")
            self.assertEqual(hashes, {p.name: sha256(p) for p in source.iterdir()})
            source_summary["arms"]["baseline"]["tp"] = 99
            (source / "summary.json").write_text(json.dumps(source_summary))
            with self.assertRaisesRegex(ValueError, "replay mismatch"):
                analyze(source, root / "bad-output", [0.35])
            self.assertFalse((root / "bad-output").exists())


if __name__ == "__main__":
    unittest.main()
