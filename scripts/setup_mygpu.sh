#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
if [[ $(uname -s) != Linux ]]; then
  echo "This setup is for MyGPU Linux only; do not install ML frameworks on Mac." >&2
  exit 1
fi

base_python=${BASE_PYTHON:-/home/brucex/anaconda3/envs/torchenv/bin/python}
"$base_python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
if [[ ! -x .venv-runtime/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .venv-runtime
fi

# Install the author implementation into this project only, retaining the known
# CUDA-compatible Torch packages from the base environment without modifying it.
.venv-runtime/bin/python -m pip install --no-deps -r requirements-runtime.txt
.venv-runtime/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil

config = json.loads(Path("configs/yolov12s_car_preview.json").read_text())
directory = Path("checkpoints")
directory.mkdir(exist_ok=True)
target = directory / config["weights_name"]
receipt = target.with_suffix(".source.json")
if target.exists() and receipt.exists():
    existing = json.loads(receipt.read_text())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if existing["url"] != config["weights_url"] or existing["sha256"] != digest:
        raise SystemExit("Existing checkpoint receipt mismatch; inspect before overwriting")
else:
    if target.exists():
        raise SystemExit("Existing checkpoint has no provenance receipt; inspect before overwriting")
    temporary = target.with_suffix(".download")
    with urlopen(config["weights_url"], timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size != config["weights_bytes"]:
        raise SystemExit("Checkpoint size differs from the official release; incomplete file retained")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(target)
    receipt.write_text(json.dumps({"url": config["weights_url"], "sha256": digest,
                                   "bytes": target.stat().st_size}, indent=2) + "\n")
print("Official checkpoint ready:", target.name, digest)
PY

.venv-runtime/bin/python run_inference.py
