"""MyGPU-only, pinned YOLOv12 integration. No changes to installed upstream files."""

import json
import math
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.tal import make_anchors

from comparison_utils import union_coverage
from finetune_metrics import match_overlaps, summarize_records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_sidecar(image):
    image = Path(image)
    return image.parents[2] / "sidecars" / image.parent.name / (image.stem + ".json")


class IgnoreDataset(YOLODataset):
    """Carry ignore rectangles through the same geometry as cars, then separate."""

    def get_labels(self):
        # The frozen view has already been hash/geometry verified; do not create upstream .cache files in it.
        labels = []
        self.label_files = []
        for image in self.im_files:
            sidecar = json.loads(source_sidecar(image).read_text())
            path = (
                Path(image).parents[2]
                / "labels"
                / Path(image).parent.name
                / (Path(image).stem + ".txt")
            )
            rows = [
                [float(value) for value in line.split()] for line in path.read_text().splitlines()
            ]
            array = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if len(array) and (not np.isfinite(array).all() or (array[:, 0] != 0).any()):
                raise ValueError("Invalid prepared labels")
            self.label_files.append(str(path))
            labels.append(
                {
                    "im_file": image,
                    "shape": (sidecar["size"][1], sidecar["size"][0]),
                    "cls": array[:, :1],
                    "bboxes": array[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    def update_labels_info(self, label):
        sidecar = json.loads(source_sidecar(label["im_file"]).read_text())
        width, height = sidecar["size"]
        boxes = [
            [(r[0] + r[2] / 2) / width, (r[1] + r[3] / 2) / height, r[2] / width, r[3] / height]
            for r in sidecar["ignored"]
        ]
        if boxes:
            label["bboxes"] = np.concatenate((label["bboxes"], np.asarray(boxes, dtype=np.float32)))
            label["cls"] = np.concatenate(
                (label["cls"], -np.ones((len(boxes), 1), dtype=np.float32))
            )
        return super().update_labels_info(label)

    def build_transforms(self, hyp=None):
        if hyp.mosaic or hyp.mixup or hyp.multi_scale or self.rect:
            raise ValueError("This baseline permits square letterbox, HSV and horizontal flip only")
        # No random crop/affine/perspective: rectangles and their union remain exact.
        transforms = [LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleup=True)]
        if self.augment:
            transforms += [
                RandomHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v),
                RandomFlip(p=hyp.fliplr, direction="horizontal"),
            ]
        transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0))
        return Compose(transforms)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ignored = sample["cls"].reshape(-1) < 0
        sample["ignore_boxes"] = sample["bboxes"][ignored].clone()
        for key in ("cls", "bboxes", "batch_idx"):
            sample[key] = sample[key][~ignored]
        return sample


def ignored_anchor_centers(points, boxes, size):
    """Raster-mask equivalent at feature-cell centers; half-open region bounds."""
    result = torch.zeros((len(boxes), len(points)), dtype=torch.bool, device=points.device)
    scale = points.new_tensor([size[1], size[0], size[1], size[0]])
    for image_index, image_boxes in enumerate(boxes):
        regions = ops.xywh2xyxy(image_boxes.to(points.device)) * scale
        if len(regions):
            inside = (
                (points[:, None, :] >= regions[None, :, :2])
                & (points[:, None, :] < regions[None, :, 2:])
            ).all(-1)
            result[image_index] = inside.any(-1)
    return result


def background_loss_weights(ignored, foreground):
    return (~ignored | foreground).unsqueeze(-1)


class IgnoreLoss(v8DetectionLoss):
    """Pinned v8DetectionLoss with ignored negative BCE removed; positives win."""

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [x.view(feats[0].shape[0], self.no, -1) for x in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        pred_bboxes = self.bbox_decode(anchors, pred_distri)
        _, target_bboxes, target_scores, foreground, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * strides).type(gt_bboxes.dtype),
            anchors * strides,
            gt_labels,
            gt_bboxes,
            gt_bboxes.sum(2, keepdim=True).gt_(0.0),
        )
        foreground = foreground.bool()  # Upstream returns float masks for batches with no GT.
        denominator = max(target_scores.sum(), 1)
        ignored = ignored_anchor_centers(
            anchors * strides, batch["ignore_boxes"], batch["img"].shape[-2:]
        )
        weights = background_loss_weights(ignored, foreground)
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)) * weights).sum() / denominator
        if foreground.sum():
            target_bboxes /= strides
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchors,
                target_bboxes,
                target_scores,
                denominator,
                foreground,
            )
        loss *= loss.new_tensor([self.hyp.box, self.hyp.cls, self.hyp.dfl])
        self.last_counts = {
            "ignored_background_anchors": int((ignored & ~foreground).sum()),
            "positive_anchors_in_ignore": int((ignored & foreground).sum()),
        }
        total = loss.sum() * batch_size
        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite loss; stopping rather than hiding damaged training"
            )
        return total, loss.detach()


class CarDetectionModel(DetectionModel):
    def init_criterion(self):
        return IgnoreLoss(self)


def make_dataset(args, data, image_path, batch, mode):
    return IgnoreDataset(
        img_path=image_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=mode + ": ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )


def keep_outside_ignore(boxes, regions):
    kept = []
    region_array = np.asarray(regions, dtype=float).reshape(-1, 4)
    for index, box in enumerate(boxes):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        relevant = region_array[
            (region_array[:, 0] < box[2])
            & (region_array[:, 2] > box[0])
            & (region_array[:, 1] < box[3])
            & (region_array[:, 3] > box[1])
        ]
        if not len(relevant) or union_coverage(box, relevant.tolist()) < 0.5:
            kept.append(index)
    return kept


class IgnoreValidator(DetectionValidator):
    """FP32 custom car AP; fixed geometry/ignore policy for initial and trained models."""

    def __call__(self, trainer=None, model=None):
        original_amp = trainer.amp if trainer else None
        if trainer:
            trainer.amp = False  # model selection always uses FP32, like standalone evaluation
        try:
            return super().__call__(trainer, model)
        finally:
            if trainer:
                trainer.amp = original_amp

    def build_dataset(self, img_path, mode="val", batch=None):
        return make_dataset(self.args, self.data, img_path, batch or self.args.batch, "val")

    def init_metrics(self, model):
        self.source_class = 2 if len(model.names) == 80 else 0
        if model.names.get(self.source_class) != "car":
            raise ValueError("Checkpoint car mapping mismatch")
        super().init_metrics(model)
        self.names = self.metrics.names = {0: "car"}
        self.nc = 1
        self.records = []
        self.cache = getattr(self, "cache", {})

    def postprocess(self, preds):
        outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            classes=[self.source_class],
            multi_label=False,
            agnostic=False,
            max_det=self.args.max_det,
        )
        for output in outputs:
            output[:, 5] = 0
        return outputs

    def update_metrics(self, preds, batch):
        for index, pred in enumerate(preds):
            image = batch["im_file"][index]
            if image not in self.cache:
                sidecar = json.loads(source_sidecar(image).read_text())
                regions = [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in sidecar["ignored"]]
                cars = [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in sidecar["cars"]]
                keep = keep_outside_ignore(cars, regions)
                self.cache[image] = (sidecar, regions, cars, keep)
            sidecar, regions, cars, gt_keep = self.cache[image]
            native = (
                self._prepare_pred(pred, self._prepare_batch(index, batch)).detach().float().cpu()
            )
            native = native[torch.argsort(native[:, 4], descending=True, stable=True)]
            pred_keep = keep_outside_ignore(native[:, :4].tolist(), regions)
            selected = native[pred_keep]
            gt = torch.tensor([cars[i] for i in gt_keep], dtype=torch.float32).reshape(-1, 4)
            overlaps = box_iou(selected[:, :4], gt).tolist()
            correct = match_overlaps(overlaps)
            self.records.append(
                {
                    "image": Path(image).name,
                    "evaluated_gt": len(gt),
                    "ignored_gt": len(cars) - len(gt),
                    "ignored_predictions": len(native) - len(selected),
                    "hit_prediction_cap": len(pred) >= self.args.max_det,
                    "confidence": selected[:, 4].tolist(),
                    "correct": correct,
                    "predictions": native.tolist(),
                }
            )
            self.seen += 1
            self.stats["tp"].append(
                torch.tensor(correct, dtype=torch.bool, device=self.device).reshape(-1, 10)
            )
            self.stats["conf"].append(selected[:, 4].to(self.device))
            self.stats["pred_cls"].append(torch.zeros(len(selected), device=self.device))
            self.stats["target_cls"].append(torch.zeros(len(gt), device=self.device))
            self.stats["target_img"].append(torch.zeros(int(len(gt) > 0), device=self.device))

    def get_stats(self):
        # Fill upstream display/speed structures, but select by the explicit custom AP implementation.
        super().get_stats()
        self.summary = summarize_records(self.records)
        point = self.summary["operating_points"][0]
        return {
            "metrics/precision(B)": point["precision"] or 0.0,
            "metrics/recall(B)": point["recall"] or 0.0,
            "metrics/mAP50(B)": self.summary["custom_AP50"],
            "metrics/mAP50-95(B)": self.summary["custom_AP50_95"],
            "fitness": self.summary["custom_AP50_95"],
        }


class IgnoreTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CarDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        return make_dataset(self.args, self.data, img_path, batch or self.args.batch, mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # Dense validation images can consume substantial temporary NMS/activation memory.
        return super().get_dataloader(dataset_path, 2 if mode == "val" else batch_size, rank, mode)

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return IgnoreValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _setup_train(self, world_size):
        requested_amp = self.args.amp
        self.args.amp = False  # Avoid upstream AMP checker downloading an unrelated model.
        super()._setup_train(world_size)
        self.args.amp = self.amp = requested_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp, init_scale=1024)
        self.step_attempts = self.successful_steps = self.skipped_steps = 0
        self.ignore_totals = {"ignored_background_anchors": 0, "positive_anchors_in_ignore": 0}
        frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]
        if any(".dfl." not in name for name in frozen):
            raise ValueError(f"Unexpected frozen trainable layers: {frozen}")
        self.optimizer.register_step_post_hook(self._count_step)
        write_json(
            self.save_dir / "training_runtime.json",
            {
                "batch": self.batch_size,
                "nbs": self.args.nbs,
                "accumulation_after_warmup": self.accumulate,
                "amp": self.amp,
                "eval_dtype": "float32",
                "optimizer": type(self.optimizer).__name__,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "fixed_dfl_parameters": frozen,
                "gpu": torch.cuda.get_device_name(),
                "optimizer_groups": [
                    {
                        "lr": g["lr"],
                        "weight_decay": g["weight_decay"],
                        "betas": g.get("betas"),
                        "parameter_count": sum(p.numel() for p in g["params"]),
                    }
                    for g in self.optimizer.param_groups
                ],
            },
        )

    def _count_step(self, optimizer, args, kwargs):
        self.successful_steps += 1

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if tuple(batch["img"].shape[1:]) != (3, self.args.imgsz, self.args.imgsz):
            raise ValueError(f"Unexpected training tensor: {batch['img'].shape}")
        return batch

    def optimizer_step(self):
        previous = self.successful_steps
        self.step_attempts += 1
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        if not self.amp and not torch.isfinite(norm):
            raise FloatingPointError("Non-finite full-precision gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.successful_steps > previous:
            self.ema.update(self.model)
        else:
            self.skipped_steps += 1
        if self.step_attempts >= 20 and self.successful_steps == 0:
            raise FloatingPointError("AMP has not completed any optimizer updates")
        if hasattr(self.model.criterion, "last_counts"):
            for key, value in self.model.criterion.last_counts.items():
                self.ignore_totals[key] += value

    def validate(self):
        metrics, fitness = super().validate()
        if not math.isfinite(fitness):
            raise FloatingPointError("Invalid validation fitness")
        write_json(
            self.save_dir / f"validation_epoch_{self.epoch + 1:03d}.json", self.validator.summary
        )
        write_json(
            self.save_dir / "progress.json",
            {
                "epoch": self.epoch + 1,
                "epochs": self.epochs,
                "fitness": fitness,
                "best_fitness": self.best_fitness,
                "successful_optimizer_steps": self.successful_steps,
                "skipped_amp_steps": self.skipped_steps,
                "amp_scale": self.scaler.get_scale(),
                "max_gpu_memory_allocated": torch.cuda.max_memory_allocated(),
                "ignore_counts_at_update_batches": self.ignore_totals,
            },
        )
        return metrics, fitness

    def save_model(self):
        # Preserve exact FP32 EMA and full-precision optimizer states, unlike upstream FP16 serialization.
        model = deepcopy(self.ema.ema).float().cpu()
        model.criterion = None
        checkpoint = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,
            "ema": model,
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "version": ultralytics.__version__,
            "custom_protocol": "visdrone-car-v1-ignore-aware-ap101",
        }
        temporary = self.last.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.last)
        if self.best_fitness == self.fitness:
            import shutil

            shutil.copy2(self.last, self.best)

    def final_eval(self):
        # Keep resumable checkpoints intact; the entry point performs matched standalone evaluation.
        if not self.best.exists() or not self.last.exists():
            raise RuntimeError("Missing final checkpoints")
