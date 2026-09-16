"""Attach to existing batch8; after completion run short capacity probes only."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from data_utils import sha256
from model_arch import ROOT


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_jobs(root):
    return [
        {"batch": batch, "kind": "throughput_only", "output": str(root / f"batch{batch}")}
        for batch in (8, 16, 20)
    ]


def next_capacity_batch(results):
    """Bounded four-image increments after requested probes, stop on OOM/instability."""
    if len(results) < 3 or any(r["status"] != "completed" or not r.get("stable") for r in results):
        return None
    largest = max(r["batch"] for r in results)
    return largest + 4 if largest < 64 else None


def process_matches(pid, batch8_path):
    proc = Path(f"/proc/{pid}")
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        status = (proc / "status").read_text()
    except FileNotFoundError:
        return False
    return (
        "run_finetune.py" in command and str(batch8_path) in command and "\nState:\tZ" not in status
    )


def wait_for_existing_batch8(batch8_path, pid):
    """Read-only attachment: never signal, restart, resume, or overwrite the training process."""
    while True:
        manifest = batch8_path / "run.json"
        result = json.loads(manifest.read_text()) if manifest.exists() else {}
        if result.get("status") == "completed":
            if result["epochs_completed"] != 50 or result.get("smoke"):
                raise RuntimeError("Existing run is not the requested completed50epoch baseline")
            while process_matches(pid, batch8_path):
                time.sleep(1)  # Wait for CUDA context release before measuring another batch.
            return result
        if not process_matches(pid, batch8_path):
            raise RuntimeError(
                "Existing batch8 process ended without a complete result; inspect its log"
            )
        progress = batch8_path / "progress.json"
        if progress.exists():
            state = json.loads(progress.read_text())
            print(
                f"Waiting for existing batch8: {state['epoch']}/50 validated; bestAP={state['best_fitness']:.5f}",
                flush=True,
            )
        time.sleep(30)


def run_probe(job, output):
    log_path = output / f"batch{job['batch']}.log"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "benchmark_finetune.py"),
        "--execute",
        "--batch",
        str(job["batch"]),
        "--output",
        job["output"],
    ]
    print(f"Starting short throughput/memory probe batch{job['batch']}", flush=True)
    with log_path.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
        )
    path = Path(job["output"]) / "benchmark.json"
    if completed.returncode or not path.exists():
        raise RuntimeError(f"Benchmark failure; inspect {log_path}")
    result = json.loads(path.read_text())
    if result["status"] not in ("completed", "oom"):
        raise RuntimeError(f"Unexpected benchmark status: {result['status']}")
    job.update(
        status=result["status"], result=result, log=str(log_path), result_sha256=sha256(path)
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "batch",
                    "status",
                    "stable",
                    "images_per_second",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                ]
            }
        ),
        flush=True,
    )
    return result


def run(output, batch8_path, pid):
    if os.uname().sysname != "Linux":
        raise RuntimeError("Run on MyGPU Linux")
    if output.exists():
        raise FileExistsError("Choose a fresh follow-up directory")
    output.mkdir(parents=True)
    state = {
        "status": "waiting_for_batch8",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "batch8_path": str(batch8_path),
        "batch8_pid": pid,
        "jobs": build_jobs(output),
        "plan": "No new full training. Finish existing batch8, then warmed short throughput/memory probes.",
    }
    write_json(output / "followup.json", state)
    try:
        trained = wait_for_existing_batch8(batch8_path, pid)
        state["status"] = "analyzing_batch8"
        write_json(output / "followup.json", state)
        analysis_path = output / "batch8_analysis"
        with (output / "analysis.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analyze_finetune.py"),
                    "--run",
                    str(batch8_path),
                    "--output",
                    str(analysis_path),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state["analysis_path"] = str(analysis_path)
        state.update(
            status="benchmarking",
            batch8_result_sha256=sha256(batch8_path / "run.json"),
            best_model=trained["checkpoints"]["best"],
        )
        write_json(output / "followup.json", state)
        results = []
        index = 0
        while index < len(state["jobs"]):
            job = state["jobs"][index]
            job["status"] = "running"
            write_json(output / "followup.json", state)
            results.append(run_probe(job, output))
            write_json(output / "followup.json", state)
            index += 1
            if index == len(state["jobs"]):
                larger = next_capacity_batch(results)
                if larger is not None:
                    state["jobs"].append(
                        {
                            "batch": larger,
                            "kind": "throughput_only",
                            "output": str(output / f"batch{larger}"),
                        }
                    )
        stable = [r for r in results if r["status"] == "completed" and r.get("stable")]
        oom = [r["batch"] for r in results if r["status"] == "oom"]
        state.update(
            status="completed",
            highest_tested_stable_batch=max((r["batch"] for r in stable), default=None),
            first_tested_oom_batch=min(oom) if oom else None,
            fastest_tested_batch=max(stable, key=lambda r: r["images_per_second"])["batch"]
            if stable
            else None,
            quality_metrics_from_benchmarks=False,
        )
        lines = [
            "# Warmed training throughput and capacity",
            "",
            "Disposable weight updates; no validation or detection-quality interpretation.",
            "",
            "| Batch | Status | Images/s | Peak allocated GiB | Peak reserved GiB | Timed AMP skips |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for r in results:
            rate = f"{r['images_per_second']:.2f}" if "images_per_second" in r else "-"
            allocated = r.get("peak_allocated_bytes", 0) / 1024**3
            reserved = r.get("peak_reserved_bytes", 0) / 1024**3
            lines.append(
                f"| {r['batch']} | {r['status']} | {rate} | {allocated:.2f} | {reserved:.2f} | {r.get('measured_amp_skips', '-')} |"
            )
        lines += [
            "",
            "Capacity applies only to current YOLOv12-S, square960, AMP and this sampled/dense stress workload. "
            "Reserved memory includes allocator cache; desktop/driver allocations are additional. "
            "Throughput includes prepared-batch loading, forward, backward and optimizer update after20warmup steps, "
            "measured in three20step blocks; it excludes dataset verification and setup. "
            "No short-test AP is calculated; choose future batch with headroom, not just the largest fitting batch.",
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "followup.json", state)
    print(
        json.dumps(
            {
                "status": "completed",
                "highest_stable": state["highest_tested_stable_batch"],
                "first_oom": state["first_tested_oom_batch"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch8-run", type=Path, required=True)
    parser.add_argument("--batch8-pid", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run(args.output.resolve(), args.batch8_run.resolve(), args.batch8_pid)
    else:
        print(json.dumps(build_jobs(args.output.resolve()), indent=2))
