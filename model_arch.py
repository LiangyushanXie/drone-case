"""Load the agreed pretrained detector. Importing this file does not import Torch.

Model flow: RGB tensor -> convolution/area-attention backbone -> multiscale
feature fusion -> detection head -> NMS -> car boxes in original-image pixels.
The upstream YOLOv12-S architecture and pretrained parameters are preserved.
"""

import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/yolov12s_car_preview.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["image_size"] <= 0 or config["image_size"] % 32:
        raise ValueError("Input size must be a positive multiple of 32")
    if config["batch_size"] != 1:
        raise ValueError("This first preview processes one image at a time")
    for key in ("confidence", "nms_iou"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if (config["coco_class_id"], config["visdrone_class_id"], config["class_name"]) != (
        2,
        4,
        "car",
    ):
        raise ValueError(
            "The preview requires the explicit COCO-car to VisDrone-car mapping"
        )
    return config


def prediction_arguments(config):
    """Make inference choices explicit instead of inheriting hidden defaults."""
    return {
        "imgsz": config["image_size"],
        "batch": config["batch_size"],
        "rect": False,
        "device": 0,
        "conf": config["confidence"],
        "iou": config["nms_iou"],
        "classes": [config["coco_class_id"]],
        "max_det": config["max_detections"],
        "half": config["half_precision"],
        "augment": False,
        "agnostic_nms": False,
        "save": False,
        "verbose": False,
    }


def build_detector(config, weights):
    """Called only on MyGPU; refuse an unintended CPU/Mac or YOLO implementation."""
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import block

    if not torch.cuda.is_available():
        raise RuntimeError("Run this preview on MyGPU with CUDA, not on the Mac")
    direct_url = importlib.metadata.distribution("ultralytics").read_text(
        "direct_url.json"
    )
    provenance = json.loads(direct_url or "{}")
    revision = provenance.get("vcs_info", {}).get("commit_id")
    if revision != config["upstream_revision"]:
        raise RuntimeError(
            "Install the pinned author implementation from requirements-runtime.txt"
        )
    weights = Path(weights).resolve()
    if not weights.is_file() or weights.name != config["weights_name"]:
        raise ValueError("Supply the downloaded official yolov12s.pt checkpoint")

    # The author's legacy checkpoint contains a serialized model, not just tensors.
    # Enable that legacy loading only for this known official checkpoint, then restore.
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        detector = YOLO(str(weights))
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
    if detector.names.get(config["coco_class_id"]) != "car":
        raise ValueError(
            "Checkpoint class names do not match the expected COCO mapping"
        )
    environment = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "ultralytics_version": ultralytics.__version__,
        "upstream_revision": revision,
        "attention_backend": "flash_attn"
        if block.USE_FLASH_ATTN
        else "upstream_explicit_attention_fallback",
    }
    return detector, environment
