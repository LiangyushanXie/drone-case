# Car fine-tuning data: preparation contract

This is data preparation, **not a training run**. Mac performs lightweight data work;
MyGPU will run a later jointly configured fine-tuning experiment. No PyTorch,
Ultralytics, weights or GPU inference is needed to run this preparation script.

## Reproduce

With the original dataset and the existing Pillow environment available:

```bash
.venv/bin/python prepare_car_data.py
.venv/bin/python prepare_car_data.py --verify
```

The default output is `data/prepared/visdrone-car-v1/` (Git ignored). Preparation
refuses an existing output directory. For a new version use `--output` with a new
empty path; preserve the previous frozen view. A failed or interrupted build is
not ready: completion requires `dataset_spec.json` and successful `--verify`.
The verifier writes nothing into the frozen dataset. Both passes check every
original image and annotation against `metadata/files.sha256` and decode images.

## Bruce's reviewed selection decisions

| Original sample | Derived-view action | Basis |
| --- | --- | --- |
| train/9999985_00000_d_0000020 | Exclude whole image | One scored car has width 4, height 0. Preserve original, do not guess box. |
| train/0000239_06450_d_0000017 | Keep | Bruce reviewed the 33-car version, including distant small cars. |
| train/0000239_06950_d_0000018 | Exclude duplicate | Same pixels, 16 cars and different ignored areas. More labels is not a general selection rule. |
| train/9999950_00000_d_0000079 | Keep | Both nighttime versions have 8 cars. Bruce permits either; choose lexical first. No claim it is more accurate. |
| train/9999950_00000_d_10000080 | Exclude duplicate | Same nighttime pixels, different box geometry. No label merging. |
| val/0000023_00000_d_0000008 | Keep | Bruce selected the car-labeled version. |
| val/0000022_00000_d_0000004 | Exclude duplicate | Same pixels, different ignore labeling. |

Raw files stay unchanged. The derived membership is **6,468 train / 547 val /
1,610 test-dev**. Files never move between splits. Test-dev membership is unchanged
and is reserved for final evaluation; it is not used to select preview examples or
tune model choices. No-car images remain as valid car-negative samples. Other
scored categories (van, truck, bus, etc.) are not positive cars in this car-only task.

Two zero-height ignore regions, at train/0000293_03401_d_0000939 line 130 and
train/9999999_00590_d_0000267 line 89, are logged and preserved in original
annotation copies and sidecars. They cover zero pixels, so no mask area is added.
Any unexpected geometry problem or new exact duplicate group fails preparation
instead of silently modifying labels. Overlapping filename prefixes do not prove
scene leakage or guarantee scene independence; this view retains official split
assignments rather than inventing a scene-based resplit.

## Files and geometry

- `images/{split}/`: byte-identical original JPEG copies, no resized or gray-masked inputs.
- `annotations/{split}/`: byte-identical full original TXT copies.
- `labels/{split}/`: source category 4 with nonzero score becomes training class 0.
  YOLO rows contain normalized center-x, center-y, width, height (nine decimal places).
  Empty files represent zero scored car targets; they do not imply no objects at all.
- `ignore_masks/{split}/`: single-channel 8-bit PNG in original image dimensions;
  255 means ignored, 0 means valid. Rectangles use half-open `[x, x+w) × [y, y+h)`
  bounds; overlapping areas are unioned. Score 0 or category 0 creates ignore areas.
- `sidecars/{split}/`: all original rows plus parsed cars, ignored regions, attributes,
  source hashes, dimensions and warning/provenance records.
- `train.txt`, `val.txt`, `test-dev.txt`: ordered relative image paths, rooted at the
  dataset directory. These are manifests, not a ready-to-launch trainer configuration.
- `manifest.jsonl`: **all 8,629 original samples**, including exclusion reasons;
  `audit.json`: included counts, overlap, size/occlusion summaries and anomalies.
- `checksums.json` and `dataset_spec.json`: frozen asset checksums and data contract.
- `previews/`: annotation-only views; green is a scored car, gray is an ignored area.
  Native-pixel small/medium/large bins in the audit are descriptive (<32², <96²,
  ≥96²); they differ from the earlier baseline-640-scaled diagnostic size bins.

## Training is deliberately not ready yet

**Stock YOLO TXT loading does not implement ignore masks.** No standard YOLO
training YAML is emitted, and `training_launch_ready` is false. The next trainer
must load the mask, transform it together with images and boxes (including any
resize/crop/Mosaic), and suppress ignored **background** loss. Preserve supervised
positive cars that overlap coarse ignore regions; define and test foreground
precedence explicitly. Mask interpolation must preserve its binary meaning. Merely
saving PNG masks or dropping ignored TXT rows would not implement this behavior.

Before a full run, jointly choose the training configuration, implement/test mask
propagation and loss behavior, then run a small training smoke test on MyGPU. The
classification head's car index will be 0, unlike COCO pretrained car index 2;
evaluation needs an explicit mapping and appropriate checkpoint/head initialization.

## Matched evaluation

The 547-image cleaned validation view is a custom protocol, not the raw official
548-image benchmark. Freeze it for **both pretrained and fine-tuned** evaluation,
with the same input scheme, matching IoU, NMS, confidence thresholds and ignore
handling. Dataset-cleaning differences are not evidence of weight improvements.
Neither disputed validation image belonged to the earlier 30-image diagnostic.
That set remains useful for visual continuity, but repeated decisions on 30 images
are not independent final validation. Reserve test-dev for a later final check.
