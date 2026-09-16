"""Run explicitly with MyGPU runtime; no ML dependency is installed on Mac."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss

from finetune_runtime import (
    IgnoreDataset,
    IgnoreLoss,
    background_loss_weights,
    ignored_anchor_centers,
)


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    points = torch.tensor([[0.0, 0.0], [4.0, 4.0], [8.0, 8.0]], device=device)
    boxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
    ignored = ignored_anchor_centers(points, boxes, (16, 16))
    assert ignored.tolist() == [[True, True, False]]
    foreground = torch.tensor([[False, True, False]], device=device)
    logits = torch.zeros((1, 3, 1), device=device, requires_grad=True)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none"
    )
    (loss * background_loss_weights(ignored, foreground)).sum().backward()
    assert logits.grad[0, 0, 0] == 0 and logits.grad[0, 1, 0] != 0 and logits.grad[0, 2, 0] != 0

    model = nn.Module()
    model.register_parameter("dummy", nn.Parameter(torch.zeros(1, device=device)))
    model.args = get_cfg()
    model.model = [
        SimpleNamespace(nc=1, reg_max=16, stride=torch.tensor([8, 16, 32], device=device))
    ]
    custom, original = IgnoreLoss(model), v8DetectionLoss(model)
    torch.manual_seed(5)
    feats = [torch.randn((1, 65, n, n), device=device, requires_grad=True) for n in (20, 10, 5)]
    batch = {
        "img": torch.zeros((1, 3, 160, 160), device=device),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        "ignore_boxes": [torch.empty((0, 4))],
    }
    expected, expected_parts = original(feats, batch)
    actual, actual_parts = custom(feats, batch)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_parts, expected_parts)
    expected_grads = torch.autograd.grad(expected, feats, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, feats, retain_graph=True)
    for first, second in zip(expected_grads, actual_grads):
        torch.testing.assert_close(first, second)
    batch["ignore_boxes"] = [torch.tensor([[0.5, 0.5, 1.0, 1.0]])]
    positive_loss, _ = custom(feats, batch)
    assert positive_loss > 0 and custom.last_counts["positive_anchors_in_ignore"] > 0
    batch.update(cls=torch.empty((0, 1)), bboxes=torch.empty((0, 4)), batch_idx=torch.empty((0,)))
    empty_loss, _ = custom(feats, batch)
    assert empty_loss == 0
    gradients = torch.autograd.grad(empty_loss, feats)
    assert all(torch.count_nonzero(g) == 0 for g in gradients)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for folder in ("images/train", "labels/train", "sidecars/train"):
            (root / folder).mkdir(parents=True)
        Image.new("RGB", (200, 100), "white").save(root / "images/train/example.jpg")
        (root / "labels/train/example.txt").write_text("0 0.125 0.4 0.15 0.4\n")
        (root / "sidecars/train/example.json").write_text(
            json.dumps({"size": [200, 100], "ignored": [[40, 30, 20, 10, 0, 0, 0, 0]]})
        )
        cfg = get_cfg(
            overrides={
                "imgsz": 160,
                "mosaic": 0.0,
                "mixup": 0.0,
                "fliplr": 1.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            }
        )
        dataset = IgnoreDataset(
            img_path=str(root / "images/train"),
            imgsz=160,
            batch_size=1,
            augment=True,
            hyp=cfg,
            rect=False,
            cache=False,
            data={"names": {0: "car"}, "nc": 1},
            task="detect",
        )
        sample = dataset[0]
        assert tuple(sample["img"].shape) == (3, 160, 160)
        np.testing.assert_allclose(sample["bboxes"].numpy(), [[0.875, 0.45, 0.15, 0.2]], atol=1e-6)
        np.testing.assert_allclose(
            sample["ignore_boxes"].numpy(), [[0.75, 0.425, 0.1, 0.05]], atol=1e-6
        )
        combined = dataset.collate_fn([dataset[0], dataset[0]])
        assert len(combined["ignore_boxes"]) == 2 and combined["batch_idx"].tolist() == [0, 1]
        assert not list(root.rglob("*.cache"))
    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "checks": [
                    "half-open anchor mask",
                    "zero ignored-background gradient",
                    "positive precedence",
                    "no-ignore upstream loss and gradient equivalence",
                    "all-ignore negative yields zero loss",
                    "letterbox and horizontal-flip alignment",
                    "collation",
                    "immutable dataset no cache",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
