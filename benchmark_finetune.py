"""Disposable warmed training-throughput/memory test; never evaluate detector quality."""

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

from data_utils import sha256
from model_arch import ROOT
from run_finetune import load_training_config, training_arguments, verify_prepared


def result_path(output):
    return Path(output) / "benchmark.json"


def benchmark(batch_size, output, warmup_steps=20, blocks=3, steps_per_block=20):
    import torch
    import yaml

    from finetune_runtime import IgnoreTrainer, write_json
    from model_arch import build_detector, load_config

    if not torch.cuda.is_available() or os.uname().sysname != "Linux":
        raise RuntimeError("Benchmark requires MyGPU Linux/CUDA")
    if output.exists():
        raise FileExistsError("Never overwrite an existing capacity trial")
    config = load_training_config()
    dataset, spec = verify_prepared(config)
    if sha256(ROOT / config["initial_weights"]) != config["initial_weights_sha256"]:
        raise ValueError("Initial checkpoint changed")
    output.mkdir(parents=True)
    result = {
        "status": "running",
        "kind": "training_throughput_not_quality",
        "batch": batch_size,
        "imgsz": 960,
        "amp": True,
        "warmup_steps": warmup_steps,
        "blocks": blocks,
        "steps_per_block": steps_per_block,
        "quality_evaluation": False,
        "initial_weights_sha256": config["initial_weights_sha256"],
        "dataset_manifest_sha256": spec["manifest_sha256"],
        "stage": "setup",
    }
    write_json(result_path(output), result)
    try:
        # Same frozen 256-image pool for every batch: dense cases plus dispersed training scenes.
        rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
        training = [row for row in rows if row["included"] and row["split"] == "train"]
        dense = sorted(training, key=lambda row: (-len(row["cars"]), row["id"]))[:64]
        selected = {row["id"]: row for row in dense}
        for i in range(256):
            if len(selected) == 256:
                break
            row = training[i * len(training) // 256]
            selected[row["id"]] = row
        for row in training:
            if len(selected) == 256:
                break
            selected[row["id"]] = row
        pool = output / "training_pool.txt"
        pool.write_text(
            "".join(
                str(dataset / row["derived_image"]) + "\n"
                for row in sorted(selected.values(), key=lambda row: row["id"])
            )
        )
        yaml_path = output / "benchmark_data.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(dataset),
                    "train": str(pool),
                    "val": str(dataset / "val.txt"),
                    "names": {0: "car"},
                }
            )
        )
        result["pool_sha256"] = sha256(pool)
        result["pool_images"] = len(selected)
        result["training_pool_ids"] = sorted(selected)
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        detector, environment = build_detector(load_config(), ROOT / config["initial_weights"])
        args = training_arguments(config, yaml_path, output, smoke=True, batch=batch_size)
        args.update(nbs=batch_size, val=False, save=False, plots=False, workers=4)
        trainer = IgnoreTrainer(overrides=args)
        trainer.model = trainer.get_model(
            cfg=detector.model.yaml, weights=detector.model, verbose=False
        )
        del detector
        trainer._setup_train(1)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        result["environment"] = environment
        result["effective_batch"] = batch_size
        result["training_dtype"] = "float16_autocast_with_float32_parameters"
        result["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        iterator = iter(trainer.train_loader)

        def next_full_batch():
            nonlocal iterator
            while True:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(trainer.train_loader)
                    continue
                if len(batch["img"]) == batch_size:
                    return batch

        def step(batch):
            with torch.autocast("cuda", dtype=torch.float16, enabled=trainer.amp):
                prepared = trainer.preprocess_batch(batch)
                loss, _ = trainer.model(prepared)
            trainer.scaler.scale(loss).backward()
            trainer.optimizer_step()

        result["stage"] = "warmup"
        for _ in range(warmup_steps):
            step(next_full_batch())
        torch.cuda.synchronize()
        result["warmup_successful_updates"] = trainer.successful_steps
        result["warmup_amp_skips"] = trainer.skipped_steps
        result["warmup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        initial_updates, initial_skips = trainer.successful_steps, trainer.skipped_steps
        rates = []
        result["stage"] = "measured"
        for index in range(blocks):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps_per_block):
                step(next_full_batch())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            rates.append(
                {
                    "block": index + 1,
                    "seconds": elapsed,
                    "images": batch_size * steps_per_block,
                    "images_per_second": batch_size * steps_per_block / elapsed,
                }
            )
            print(json.dumps({"batch": batch_size, **rates[-1]}), flush=True)
        result.update(
            measured_blocks=rates,
            images_per_second=sum(row["images"] for row in rates)
            / sum(row["seconds"] for row in rates),
            median_block_images_per_second=statistics.median(
                row["images_per_second"] for row in rates
            ),
            measured_successful_updates=trainer.successful_steps - initial_updates,
            measured_amp_skips=trainer.skipped_steps - initial_skips,
            measured_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            measured_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        result["stage"] = "dense_stress"
        ds = trainer.train_loader.dataset
        indices = sorted(range(len(ds)), key=lambda i: (-len(ds.labels[i]["cls"]), ds.im_files[i]))[
            :batch_size
        ]
        stress_updates, stress_skips = trainer.successful_steps, trainer.skipped_steps
        for _ in range(3):
            step(ds.collate_fn([ds[i] for i in indices]))
        torch.cuda.synchronize()
        result.update(
            status="completed",
            stage="done",
            stable=(
                result["measured_amp_skips"] <= 1
                and result["measured_successful_updates"] >= blocks * steps_per_block - 1
                and trainer.successful_steps - stress_updates == 3
                and trainer.skipped_steps == stress_skips
            ),
            dense_stress_steps=3,
            dense_stress_successful_updates=trainer.successful_steps - stress_updates,
            dense_stress_amp_skips=trainer.skipped_steps - stress_skips,
            dense_stress_max_cars=max(len(ds.labels[i]["cls"]) for i in indices),
            peak_allocated_bytes=max(
                result["warmup_peak_allocated_bytes"], torch.cuda.max_memory_allocated()
            ),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_successful_updates=trainer.successful_steps,
            total_amp_skips=trainer.skipped_steps,
        )
    except torch.OutOfMemoryError as error:
        result.update(
            status="oom",
            stable=False,
            error=str(error),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        print(f"Capacity OOM at batch {batch_size}, stage {result['stage']}", flush=True)
    except Exception as error:
        result.update(status="failed", stable=False, error=str(error))
        traceback.print_exc()
        raise
    finally:
        write_json(result_path(output), result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "status",
                    "batch",
                    "stable",
                    "images_per_second",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "stage",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch <= 64:
        parser.error("Bounded capacity probe supports batch1..64")
    if args.execute:
        benchmark(args.batch, args.output.resolve())
    else:
        print(
            json.dumps(
                {
                    "kind": "throughput_only",
                    "batch": args.batch,
                    "warmup_steps": 20,
                    "measured_blocks": 3,
                    "steps_per_block": 20,
                    "dense_stress_steps": 3,
                }
            )
        )
