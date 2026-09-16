"""Run the agreed 960 full fine-tuning baseline on MyGPU; default is a dry run."""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from data_utils import sha256
from model_arch import ROOT

DEFAULT_CONFIG = ROOT / "configs/finetune_960.json"


def load_training_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["imgsz"] != 960 or config["epochs"] != 50 or config["optimizer"] != "AdamW":
        raise ValueError("This entry point is the agreed 960/50-epoch/AdamW baseline")
    if config["mosaic"] or config["mixup"] or config["effective_batch"] % config["batch"]:
        raise ValueError("Unsupported augmentation or effective batch")
    return config


def verify_prepared(config):
    root = ROOT / config["dataset"]
    if sha256(root / "dataset_spec.json") != config["dataset_spec_sha256"]:
        raise ValueError("Frozen data contract changed")
    spec = json.loads((root / "dataset_spec.json").read_text())
    for name, field in [
        ("checksums.json", "asset_checksums_sha256"),
        ("manifest.jsonl", "manifest_sha256"),
    ]:
        if sha256(root / name) != spec[field]:
            raise ValueError(f"Changed {name}")
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Prepared asset changed: {name}")
    if spec["included_counts"] != {"train": 6468, "val": 547, "test-dev": 1610}:
        raise ValueError("Unexpected split membership")
    print(
        "Frozen dataset verified: train 6468 / val 547; test-dev is not loaded for training/evaluation",
        flush=True,
    )
    return root, spec


def training_arguments(config, dataset_yaml, output, smoke=False, batch=None):
    return {
        "model": str(ROOT / config["initial_weights"]),
        "data": str(dataset_yaml),
        "task": "detect",
        "epochs": 1 if smoke else config["epochs"],
        "imgsz": config["imgsz"],
        "batch": batch or config["batch"],
        "nbs": config["effective_batch"],
        "device": 0,
        "workers": config["workers"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "cos_lr": config["cos_lr"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": 0 if smoke else config["warmup_epochs"],
        "warmup_bias_lr": config["warmup_bias_lr"],
        "seed": config["seed"],
        "deterministic": True,
        "freeze": 0,
        "amp": config["amp"],
        "patience": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": config["fliplr"],
        "hsv_h": config["hsv_h"],
        "hsv_s": config["hsv_s"],
        "hsv_v": config["hsv_v"],
        "multi_scale": False,
        "rect": False,
        "cache": False,
        "conf": config["eval_conf"],
        "iou": config["nms_iou"],
        "max_det": config["max_det"],
        "half": False,
        "save": True,
        "save_period": -1,
        "plots": False,
        "val": True,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "verbose": False,
        "single_cls": False,
    }


def create_dataset_yaml(dataset, output, smoke):
    import yaml

    train, val = dataset / "train.txt", dataset / "val.txt"
    if smoke:
        # Include crowded, ignored and negative examples; never touch original lists.
        records = [
            json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()
        ]
        for split, count in [("train", 64), ("val", 16)]:
            available = [r for r in records if r["included"] and r["split"] == split]
            ordered = sorted(available, key=lambda r: (-len(r["cars"]), r["id"]))[: count // 2]
            selected = {r["id"]: r for r in ordered}
            for r in available:
                if len(selected) >= count:
                    break
                selected[r["id"]] = r
            target = output / f"smoke_{split}.txt"
            target.write_text(
                "".join(str(dataset / r["derived_image"]) + "\n" for r in selected.values())
            )
        train, val = output / "smoke_train.txt", output / "smoke_val.txt"
    target = output / "ignore_aware_data.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": str(train),
                "val": str(val),
                "names": {0: "car"},
                "requires_ignore_aware_trainer": True,
            }
        )
    )
    return target


def evaluate(weights_or_model, dataset_yaml, output, config):
    from finetune_runtime import IgnoreValidator, write_json

    output.mkdir()
    validator = IgnoreValidator(
        save_dir=output,
        args={
            "data": str(dataset_yaml),
            "imgsz": config["imgsz"],
            "batch": 2,
            "device": 0,
            "workers": config["workers"],
            "conf": config["eval_conf"],
            "iou": config["nms_iou"],
            "max_det": config["max_det"],
            "half": False,
            "rect": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "plots": False,
            "save_json": False,
            "task": "detect",
            "split": "val",
            "verbose": False,
        },
    )
    validator(model=weights_or_model)
    write_json(output / "summary.json", validator.summary)
    with (output / "predictions.jsonl").open("w") as stream:
        for row in validator.records:
            stream.write(json.dumps(row) + "\n")
    return validator.summary


def run(config, output, smoke=False, batch=None):
    import torch

    from finetune_runtime import IgnoreTrainer, write_json
    from model_arch import build_detector, load_config

    if not torch.cuda.is_available() or os.uname().sysname != "Linux":
        raise RuntimeError("Execute on MyGPU Linux/CUDA, not Mac")
    if output.exists():
        raise FileExistsError(
            "Use a new run directory; never overwrite existing experiment evidence"
        )
    if sha256(ROOT / config["initial_weights"]) != config["initial_weights_sha256"]:
        raise ValueError("Initial checkpoint changed")
    dataset, spec = verify_prepared(config)
    output.mkdir(parents=True)
    dataset_yaml = create_dataset_yaml(dataset, output, smoke)
    started = time.monotonic()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = (
        "1"  # Only locally verified official/own checkpoints.
    )
    os.environ["WANDB_MODE"] = "disabled"
    detector, environment = build_detector(load_config(), ROOT / config["initial_weights"])
    metadata = {
        "status": "running",
        "smoke": smoke,
        "config": config,
        "actual_batch": batch or config["batch"],
        "environment": environment,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": spec["manifest_sha256"],
        "evaluation": "custom confidence-greedy AP101 IoU .50:.95, ignore union coverage >=.50, FP32 square960; not official VisDrone AP",
    }
    write_json(output / "run.json", metadata)
    baseline = evaluate(detector.model, dataset_yaml, output / "pretrained", config)
    # Standalone validation may fuse its model: reload the original checkpoint for every training run.
    del detector
    torch.cuda.empty_cache()
    detector, _ = build_detector(load_config(), ROOT / config["initial_weights"])
    arguments = training_arguments(config, dataset_yaml, output, smoke, batch)
    trainer = IgnoreTrainer(overrides=arguments)
    trainer.model = trainer.get_model(
        cfg=detector.model.yaml, weights=detector.model, verbose=False
    )
    del detector
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    if trainer.epoch + 1 != arguments["epochs"] or trainer.successful_steps <= 0:
        raise RuntimeError("Requested epoch budget or real optimizer updates not completed")
    trained = evaluate(str(trainer.best), dataset_yaml, output / "best_evaluation", config)
    checkpoints = {}
    for name, path in [("best", trainer.best), ("last", trainer.last)]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "epoch": checkpoint["epoch"] + 1,
            "selection_fitness": checkpoint["train_metrics"]["fitness"],
        }
    with (output / "results.csv").open() as stream:
        epoch_rows = list(csv.DictReader(stream))
    if len(epoch_rows) != arguments["epochs"]:
        raise RuntimeError("Incomplete epoch result table")
    metadata.update(
        status="completed",
        epochs_completed=trainer.epoch + 1,
        elapsed_seconds=time.monotonic() - started,
        checkpoints=checkpoints,
        successful_optimizer_steps=trainer.successful_steps,
        skipped_amp_steps=trainer.skipped_steps,
        baseline=baseline,
        best_evaluation=trained,
        AP50_95_delta=trained["custom_AP50_95"] - baseline["custom_AP50_95"],
    )
    write_json(output / "run.json", metadata)
    lines = [
        "# 960 full fine-tuning baseline",
        "",
        f"Commit: `{metadata['git_commit']}`. Smoke: {smoke}. Completed epochs: {trainer.epoch + 1}.",
        "",
        "| Metric | Pretrained | Best fine-tuned |",
        "| --- | ---: | ---: |",
    ]
    for key in ("custom_AP50", "custom_AP75", "custom_AP50_95"):
        lines.append(f"| {key} | {baseline[key]:.6f} | {trained[key]:.6f} |")
    lines += [
        "",
        f"Best checkpoint: `{trainer.best}` (epoch {checkpoints['best']['epoch']}).",
        "",
        "Custom cleaned validation protocol, not official VisDrone AP. Single seed; validation selected the best epoch. Test-dev remains untouched.",
        "",
        "Fixed operating points are in pretrained/summary.json and best_evaluation/summary.json. Both use identical geometry, NMS, cap, precision and ignore handling.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "best": checkpoints["best"],
                "AP50_95_delta": metadata["AP50_95_delta"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, choices=[2, 4, 8])
    args = parser.parse_args()
    config = load_training_config(args.config)
    if args.execute:
        if args.output is None:
            parser.error("--execute requires an explicit new --output directory")
        run(config, args.output.resolve(), args.smoke, args.batch)
    else:
        print(
            json.dumps({"status": "dry_run", "config": config, "execution_host": "MyGPU"}, indent=2)
        )
