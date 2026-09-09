# Whole-image resolution and tiled input comparison

This is a development experiment on the already-inspected 30 VisDrone validation
images. It preserves the COCO-pretrained YOLOv12-Turbo-S checkpoint, class mapping
(COCO car 2 / original VisDrone car 4), confidence 0.25, NMS 0.70, FP32, batch one
and maximum 300 final detections per image. There is no training or architecture
change.

| Arm | Input strategy |
| --- | --- |
| baseline | Whole original image, proportional resize and fixed square padding to 640 |
| resolution | Whole original image, proportional resize and fixed square padding to 960 |
| sliced | Original-image 640-pixel windows with nominal 20% overlap, each fed at 640; restore full-image coordinates and apply global car-only NMS at 0.70 |

The sliced arm merges tile predictions only. It does not mix in whole-image
predictions and does not combine slicing with 960 input. Each arm loads the same
checkpoint afresh and runs all 30 images in the same order. The final tiles are
shifted inward to cover image borders, so their overlap can exceed the nominal
20%. Images smaller than a tile are processed once. Per-view and final detection
caps are recorded; cap hits should be inspected before making stronger claims.

The hypothesis is that input downscaling may remove useful small-object detail.
More resolution or slicing can also introduce false positives, duplicate or
partial boxes, lost context and extra computation. An improvement on these
development images would support input processing as a useful intervention here;
it would not establish that the network architecture never matters.

## Diagnostic matching

The user selected confidence-ordered one-to-one matching at IoU >= 0.50. Matching
IoU judges agreement with labels; NMS IoU removes overlapping predictions. They
are different operations.

Annotations returned as ignored by the existing parser (score 0 or category 0)
form a union of rectangles. GT cars and predictions with >=50% of their own box
area inside this union are excluded before matching. Union area is computed in
continuous xyxy coordinates and overlapping ignore rectangles are not counted
twice. This documented car-only rule is a diagnostic approximation: it is not a
port of the official rounded-mask, ten-class, multi-threshold VisDrone AP evaluator.
The original labels are preserved, and raw/evaluated/ignored counts are separate.

After ignore filtering, each prediction can match at most one unmatched GT car.
Extra duplicate predictions count as false positives. Unmatched GT cars count as
false negatives. Precision and recall are micro-aggregated from TP/FP/FN, not from
prediction counts or a mean of per-image scores. Undefined ratios on empty images
remain null rather than being reported as perfect. Background false alarms are
also reported on the four images with no raw car labels.

Size groups use sqrt(original GT width * height), scaled to the baseline 640
input. Boundaries are 16, 32 and 96 pixels. This common grouping is fixed for all
arms; these are explicitly diagnostic groups, not official COCO size-bin AP.
Occlusion groups use the original VisDrone values 0/1/2. Group recall reuses each
arm's global matches; unmatched predictions cannot be assigned a GT-size group.

## Human observations and review focus

Bruce identified images 1, 6, 7 and 11 as useful examples: small visible cars,
larger occluded cars, adjacent vehicles, and the small/dense scene in image 11.
These are qualitative observations without exhaustive object-level annotation.
Density has no new ground-truth label in this experiment; review these examples
alongside the size/occlusion tables instead of treating density as a measured cause.
Numbers drawn beside predictions are model confidence, not statistical significance.

## Run and inspect

On Mac, importing the runner, its default dry run and the geometry/render tests
require no Torch or Ultralytics installation. Actual prediction is MyGPU-only.

    python3 compare_inputs.py
    .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
    uv tool run --from ruff==0.16.6 ruff check compare_inputs.py comparison_utils.py comparison_visuals.py tests/test_input_comparison.py

After committing, pushing and pulling the code, run in the MyGPU checkout:

    cd /home/brucex/drone-case
    .venv-runtime/bin/python compare_inputs.py --execute

Outputs go to a new ignored runs/input-comparison-TIMESTAMP directory:

- 30 full-resolution PNGs: green GT on the left, then baseline / resolution / sliced.
- baseline.jsonl, resolution.jsonl and sliced.jsonl: predictions, matches, source
  windows and per-image timings, all in original-image coordinates.
- per_image.csv: TP/FP/FN, ignored counts and timings for every image/arm.
- group_recall.csv: size and occlusion group counts and recall.
- summary.json and report.md: aggregate metrics, recovered/lost GT, extra false
  positives, timing comparisons, image links and prior-baseline consistency.
- run.json and inputs.json: status, commit/source hashes, checkpoint and sample
  hashes, parameters, runtime environment and actual CUDA input-tensor events.

Orange prediction boxes are matched, red ones are unmatched, and gray ones are
ignored under this rule. An unmatched box may be a false detection, a duplicate,
or a real car whose box does not reach IoU 0.50. Original green GT boxes remain
visible even if the ignore policy excludes them from the diagnostic denominator.

Timing includes preprocessing, prediction, NMS, CPU box copies and tile merge.
It excludes file decoding, matching and rendering. The reported median excludes
the first image of each arm because that image includes automatic warmup. Tiling
uses multiple model calls and is not an equal-compute comparison.

Saved predictions can be rendered again on Mac with Pillow:

    .venv/bin/python compare_inputs.py --render-only /absolute/path/to/run

This leaves the legacy run_inference.py entry point intact for a baseline-only
preview. Repeatedly inspected development images are not an untouched final test.

## Primary references

- [SAHI authors: slicing-aided inference](https://arxiv.org/abs/2202.06934).
  This project implements a small tile-only plus NMS comparison without adding a
  SAHI dependency or claiming to reproduce its complete pipeline.
- [VisDrone annotation fields](https://github.com/VisDrone/VisDrone2018-DET-toolkit).
- [Official ignore-region filtering](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/dropObjectsInIgr.m).
- [Official matching implementation](https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/evalRes.m).
