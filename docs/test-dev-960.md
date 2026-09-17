# Fixed test-dev comparison

On2026-09-17 Bruce requested real labels vs960pretrained vs960fine-tuned on the
previously held-out test-dev set. This run uses all1610images in the frozen prepared
manifest. The epoch20checkpoint was selected using yesterday's validation scores,
not by today's test result. No optimizer, training, confidence search or new model
selection occurs.

`run_test_evaluation.py` reuses the reviewed `IgnoreValidator` and `evaluate`
implementation with explicit `split="test"`. Its JSON-as-YAML dataset document
keeps train/val entries unchanged and maps `test` to `test-dev.txt`. It checks the
complete ordered membership, checkpoint/data hashes and both output image lists.

Protocol remains square960FP32, COCOcar2 vs trainedcar0, confidence floor.001,
NMS.70, max1000, union ignore coverage.50, existing confidence-greedy matching and
AP101. Report AP50/AP75/AP50:95 and P/R at confidence.25/.70/.80/.90 with matching
IoU.50/.75. This remains custom VisDrone diagnostic AP, not official benchmark AP.

Every image gets one JPEG with GT / pretrained / fine-tuned panels. GTgreen,
correct predictionorange, false predictionred, ignoredgray. Displayconfidence.25,
matchingIoU.50. Raw GT and evaluated GT counts are separated; ignored regions do
not contribute ordinary FP/FN. Full metric candidates remain in predictions.jsonl,
so display filtering does not truncate the AP calculation. Rendered TP/FP/FN totals
must replay the evaluator's matching operating point. Degenerate zero-area clipped
boxes are not drawn or evaluated as detections.

Open `index.html` for all1610lazy-loaded images; clicking a card opens the full-size
triptych. The16shortcut images are uniformly spaced file indices, selected without
looking at outcomes. Reports and gallery are local artifacts, not published images.

MyGPU command, from the confirmed VSCode SSH terminal:

```bash
.venv-runtime/bin/python -u run_test_evaluation.py --execute --output runs/test-dev-960-20260917
```

Without `--execute`, print a dry run and import no ML framework. Refuse existing
output directories. Preserve training checkpoints and source data. Test observations
can inform later questions, but any subsequent tuning using these results means this
set is no longer a completely untouched final holdout. DJI imagery remains an
additional domain-shift test, not something these public-test scores guarantee.
