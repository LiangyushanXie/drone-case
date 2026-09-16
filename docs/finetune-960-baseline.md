# 960 full fine-tuning baseline

Authorized experiment: pretrained YOLOv12-Turbo-S → car-only full fine-tuning,
960 square letterbox, AdamW lr 0.0003, 50 epochs, nominal/effective batch 8.
Code is edited on Mac; models run on MyGPU only. See `configs/finetune_960.json`.

## Fixed settings

- Initialize from the same pinned official COCO pretrained checkpoint used in the
  preview. Load compatible weights into a one-class head; mismatched classification
  tensors are initialized by upstream. Do not initialize from the smoke run.
- Update all learned parameters. Upstream's fixed DFL projection remains frozen;
  the runtime rejects any other frozen parameters.
- AdamW, betas (0.9,0.999), weight decay 0.0005 on upstream-selected decay group;
  3-epoch warmup from zero (including bias), cosine decay to a 0.1 LR multiplier.
- Batch 8, nominal effective batch (`nbs`) 8. If VRAM testing requires batch 4/2,
  record it and accumulate toward 8; upstream ramps accumulation during warmup.
- FP16 AMP training with GradScaler starting at 1024, gradient clipping 10;
  real optimizer steps counted, overflow skips logged, EMA advances only after a
  real optimizer update. FP32 evaluation and FP32 EMA checkpoint serialization.
- Fixed seed 20260916; complete all 50 epochs (`patience=0`), no early stopping.
- Mild HSV gains (0.01,0.2,0.2), horizontal flip probability 0.5.
  No Mosaic, MixUp, affine, crop, vertical flip, multi-scale or slicing.

## Ignore-aware loss and geometry

Load positive car labels and ignored rectangles from the hash-verified prepared
view. Append ignored rectangles as internal class -1 only during shared geometry,
then remove them from target labels before batching/loss. Square letterbox and
horizontal flip transform both sets identically. Color changes do not alter masks.
The mask union is evaluated at feature-anchor centers using half-open rectangles;
this is a continuous rectangle representation of the supplied original ignore masks.

The loss is the pinned author's v8DetectionLoss with just the negative classification
BCE weights changed. Background anchors centered in ignore regions contribute zero
classification loss. Assigned car foreground anchors take precedence, even inside
an ignore region; box/DFL losses remain unchanged. Without ignore regions the loss
and gradients must match upstream. This design requires this custom trainer;
stock YOLO training would not honor the data contract. No installed upstream file
is modified, and the frozen prepared dataset receives no runtime cache files.

## Matched evaluation and model selection

Use all 547 cleaned validation images. Test-dev is reserved, never selected/tuned on.
Both models use FP32 square960, confidence floor .001, NMS .70, class-specific car
filtering (COCO index2 vs trained index0), and maximum 1000 detections per image.
The cap increases from the old 30-image diagnostic's 300 **for both current arms**,
so current scores are not a direct replay of the old preview.

Drop scored GT and predictions whose box area is at least 50% covered by the union
of source ignore regions, matching the earlier diagnostic convention. Match in
stable confidence order one-to-one at IoUs .50,.55,...,.95. AP is mean interpolated
precision at 101 recall points. Report car AP50/AP75/AP50:95, and fixed P/R at
confidence .25/.70/.80/.90 for matching IoU .50 and .75. This is a **custom cleaned
validation protocol**, not official VisDrone AP. Per-image predictions and ignored
counts are saved for final evaluations. Report detection-cap hits.

Select `weights/best.pt` by highest custom AP50:95 across the 50 epochs; retain
`weights/last.pt` separately. Both contain FP32 EMA and optimizer state. Selection
validation uses the unfused EMA; standalone initial/best evaluation uses the same
upstream inference fusion, so tiny numerical differences from selection scores are
possible. Baseline evaluation uses its own model instance; reload the original
checkpoint before training to avoid starting from a fused inference model.

## Commands on MyGPU

```bash
.venv-runtime/bin/python check_finetune_runtime.py
.venv-runtime/bin/python run_finetune.py --execute --smoke --output runs/finetune-960-smoke-UNIQUE
.venv-runtime/bin/python run_finetune.py --execute --output runs/finetune-960-baseline-UNIQUE
```

Use a fresh explicit output directory. The entry point checks the frozen data and
checkpoint hashes, evaluates the pretrained baseline, starts from fresh pretrained
weights, trains, evaluates best.pt, verifies the 50 CSV rows/checkpoints, and writes
`run.json` and `report.md`. Default invocation without `--execute` is a dry run and
imports no ML framework. Smoke uses 64 training/16 validation samples including
crowded scenes, one epoch, and separate outputs; it is not a model-quality result.

Do not publish checkpoint weights or duplicate data. Save commit/data/config/weight
hashes, runtime settings, optimizer-update and overflow counts, per-epoch validation,
full log and final evaluation. A single-seed validation gain is evidence for this
experiment, not independent generalization evidence or proof of superiority to LoRA.

## Revised follow-up: capacity measurements, not more full training

Bruce cancelled the batch16/20 50-epoch experiments. Preserve the original batch8
run through its50epochs, final evaluation and best/last checkpoint saving, then
prioritize pretrained-versus-finetuned P/R, small/occluded misses, new false positives
and training/validation curves. Short tests must not be interpreted as detection quality.

The old scheduler was stopped/retired individually, without signaling its live batch8
child. Its original suite manifest was preserved; unstarted downstream jobs are marked
cancelled. `run_finetune_suite.py` now **attaches read-only** to the supplied existing
batch8 PID/output and waits for a completed50epoch run.json. It never starts or resumes
full training. Launch this replacement from the confirmed MyGPU SSH terminal after
Mac commit/push and MyGPU pull, using a new output directory:

```bash
.venv-runtime/bin/python run_finetune_suite.py --execute --batch8-run runs/finetune-batch-suite-20260916T1005/batch8 --batch8-pid 239745 --output runs/finetune-capacity-UNIQUE
```

After batch8 finishes, each batch benchmark runs in an isolated process from the same
original pretrained weights, with **no validation, no model-quality scores and no saved
benchmark checkpoint**. `benchmark_finetune.py` uses the same960/full backward/AdamW/AMP
pipeline, mild augmentations, workers4 and a frozen256-image training-only pool including
the64most crowded images. Batch8/16/20 are always requested; after stable20, probe24,28,…
until first OOM/instability or a bounded64. Report the highest *tested* stable batch and
first tested OOM; do not assert an exact untested limit.

Per batch:20warmup training updates, CUDA synchronization, three measured20-step blocks,
then3dense-label stress updates. Throughput includes data retrieval, preprocessing,
forward/backward and optimizer update after warmup; excludes setup/hash checks. Only
full batches count. Report each block and aggregate images/s, measured success/AMP skip
counts, peak PyTorch allocated and reserved memory, dense stress and any OOM stage.
The stability flag requires at most1timed AMP skip and all3stress updates to succeed.
Discard updated benchmark weights. Capacity is specific to current model, resolution,
precision, label density and other GPU users; prefer operating headroom over saturation.
