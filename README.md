# Drone Vision Case

A personal computer-vision learning project: start with **VisDrone2019-DET**, understand small-object detection, then evaluate on independently collected DJI imagery.

## First model preview

The agreed first experiment uses COCO-pretrained YOLOv12-S, 640-square input,
car-only predictions, 30 dispersed validation images, confidence 0.25 and NMS IoU
0.70. The root-level `model_arch.py`, `data_utils.py` and `run_inference.py` implement
this preview. Model execution belongs on MyGPU; Mac tests require no ML framework.
See [the configuration and code walkthrough](docs/first-preview.md). Runtime results
are pending until an actual GPU run is verified.

## Data ready for inspection

The three labeled detection splits are included in `data/raw/` with original image filenames and annotation files:

| Split | Images | Annotation files |
| --- | ---: | ---: |
| Train | 6,471 | 6,471 |
| Validation | 548 | 548 |
| Test-dev | 1,610 | 1,610 |
| Total | 8,629 | 8,629 |

All archives passed ZIP CRC checks, all images decoded, and image/annotation filenames matched. Checksums and the full audit are under `metadata/`.

**Known upstream issue:** three training boxes have nonpositive dimensions. They remain unchanged and must be investigated before training. Original ignored regions and category 11 (`others`) are preserved; a future YOLO conversion must handle them explicitly. No model has been trained, and no accuracy or deployment claim is made.

## Reproduce acquisition and checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 scripts/download_data.py --connections 8
.venv/bin/python scripts/verify_data.py
shasum -a 256 -c metadata/files.sha256
```

The archive downloader supports `HTTPS_PROXY`, interrupted downloads, archive hash verification and safe extraction. Archives are ignored to avoid storing a second copy of the images in Git. The exact downloaded mirror versions are pinned by SHA-256 in `metadata/sources.json`.

## First learning question

Are missed detections concentrated among the smallest objects? Establish one small YOLO baseline, inspect errors by size and occlusion, and only then choose one change. Later, collect and annotate a separate DJI flight test set, splitting by flight/location rather than randomly mixing adjacent frames.

## Source and attribution

Original release: [VisDrone / Tianjin University AISKYEYE](https://github.com/VisDrone/VisDrone-Dataset). The maintained [Ultralytics archive mirror](https://github.com/ultralytics/assets/releases/tag/v0.0.0) was used for download. This repository preserves the original detection annotations rather than executing the mirror's YOLO conversion.

Cite: Zhu et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI, 2021, DOI [10.1109/TPAMI.2021.3119563](https://doi.org/10.1109/TPAMI.2021.3119563).

Dataset rights remain with the original authors. This copy is for noncommercial research and learning; upstream use conditions apply. No new license over the underlying imagery is asserted by this repository.
