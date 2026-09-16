"""Durable sequential batch8 → batch16 → batch20 experiment queue on MyGPU."""

import argparse
import json
import os
import shutil
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
    jobs = []
    for batch in (8, 16, 20):
        # Batch8 has already passed the recorded crowded-scene smoke on this implementation.
        if batch != 8:
            jobs.append(
                {"batch": batch, "smoke": True, "output": str(root / f"probe-batch{batch}")}
            )
        jobs.append({"batch": batch, "smoke": False, "output": str(root / f"batch{batch}")})
    return jobs


def run_job(job, log_path):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "run_finetune.py"),
        "--execute",
        "--batch",
        str(job["batch"]),
        "--output",
        job["output"],
    ]
    if job["smoke"]:
        command.append("--smoke")
    environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        print(
            json.dumps(
                {
                    "event": "started",
                    "batch": job["batch"],
                    "smoke": job["smoke"],
                    "pid": process.pid,
                    "log": str(log_path),
                }
            ),
            flush=True,
        )
        while process.poll() is None:
            progress = Path(job["output"]) / "progress.json"
            if progress.exists():
                state = json.loads(progress.read_text())
                print(
                    json.dumps(
                        {"event": "progress", "batch": job["batch"], "smoke": job["smoke"], **state}
                    ),
                    flush=True,
                )
            time.sleep(20)
    job.update(
        returncode=process.returncode,
        elapsed_seconds=time.monotonic() - start,
        log=str(log_path),
        command=command,
    )
    if process.returncode:
        # Only a genuine capacity failure permits skipping this batch, not arbitrary code failures.
        tail = log_path.read_text(errors="replace")[-10000:]
        if job["smoke"] and ("torch.OutOfMemoryError" in tail or "CUDA out of memory" in tail):
            job["status"] = "skipped_oom"
            job["failure_tail"] = tail[-2000:]
            return
        job["status"] = "failed"
        raise RuntimeError(f"Job failed; inspect {log_path}")
    result = json.loads((Path(job["output"]) / "run.json").read_text())
    if result["status"] != "completed" or result["epochs_completed"] != (1 if job["smoke"] else 50):
        raise RuntimeError("Job exited without complete verified results")
    job.update(
        status="completed",
        run_manifest_sha256=sha256(Path(job["output"]) / "run.json"),
        result=result,
    )
    print(
        json.dumps(
            {
                "event": "completed",
                "batch": job["batch"],
                "smoke": job["smoke"],
                "epochs": result["epochs_completed"],
            }
        ),
        flush=True,
    )


def run(output):
    if os.uname().sysname != "Linux":
        raise RuntimeError("Run the queue on MyGPU Linux")
    if output.exists():
        raise FileExistsError("Choose a fresh suite directory")
    output.mkdir(parents=True)
    state = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "jobs": build_jobs(output),
    }
    write_json(output / "suite.json", state)
    skipped = set()
    try:
        for job in state["jobs"]:
            if job["batch"] in skipped:
                job["status"] = "skipped_after_failed_probe"
                write_json(output / "suite.json", state)
                continue
            job["status"] = "running"
            write_json(output / "suite.json", state)
            run_job(job, output / (Path(job["output"]).name + ".log"))
            if job["status"] == "skipped_oom":
                skipped.add(job["batch"])
            write_json(output / "suite.json", state)
        completed = [j for j in state["jobs"] if not j["smoke"] and j.get("status") == "completed"]
        best = max(completed, key=lambda j: j["result"]["best_evaluation"]["custom_AP50_95"])
        source = Path(best["result"]["checkpoints"]["best"]["path"])
        target = output / "best_overall.pt"
        shutil.copy2(source, target)
        if sha256(target) != best["result"]["checkpoints"]["best"]["sha256"]:
            raise RuntimeError("Best model copy failed checksum verification")
        state.update(
            status="completed",
            best_overall={
                "batch": best["batch"],
                "path": str(target),
                "sha256": sha256(target),
                "source": str(source),
            },
        )
        lines = [
            "# Batch size comparison",
            "",
            "Same seed, pretrained initialization, LR, 50 epochs and cleaned validation protocol; effective batch changes, so optimizer-update counts differ.",
            "",
            "| Batch | AP50 | AP75 | AP50:95 | Updates | AMP skips | Run hours |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for job in completed:
            result = job["result"]
            metrics = result["best_evaluation"]
            lines.append(
                f"| {job['batch']} | {metrics['custom_AP50']:.4f} | {metrics['custom_AP75']:.4f} | {metrics['custom_AP50_95']:.4f} | {result['successful_optimizer_steps']} | {result['skipped_amp_steps']} | {result['elapsed_seconds'] / 3600:.2f} |"
            )
        lines += [
            "",
            f"Best validation-selected checkpoint: batch {best['batch']}, `{target}`.",
            "",
            "Custom AP, single seed, validation-selected hyperparameters/checkpoint; not official VisDrone AP or independent test evidence. Per-run summaries include cap hits and fixed P/R thresholds.",
        ]
        (output / "report.md").write_text("\n".join(lines) + "\n")
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "suite.json", state)
    print(json.dumps({"event": "suite_completed", "best": state["best_overall"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run(args.output.resolve())
    else:
        print(json.dumps(build_jobs(args.output.resolve()), indent=2))
