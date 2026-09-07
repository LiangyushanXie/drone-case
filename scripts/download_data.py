#!/usr/bin/env python3
"""Resume public ZIP downloads, verify hashes, and extract without path traversal."""

import argparse
import hashlib
import json
import shutil
import stat
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 64 * 1024 * 1024


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_chunk(url, total, start, end, path):
    expected_size = end - start + 1
    for attempt in range(8):
        present = path.stat().st_size if path.exists() else 0
        if present == expected_size:
            return
        if present > expected_size:
            raise ValueError(f"Oversized partial download: {path.name}")
        request = Request(url, headers={"Range": f"bytes={start + present}-{end}"})
        try:
            with urlopen(request, timeout=90) as response:
                expected_range = f"bytes {start + present}-{end}/{total}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise ValueError(
                        "Server did not honor the exact requested byte range"
                    )
                with path.open("ab") as output:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        if output.tell() + len(block) > expected_size:
                            raise ValueError(
                                "Server returned more bytes than requested"
                            )
                        output.write(block)
            if path.stat().st_size == expected_size:
                return
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else min(2**attempt, 60)
            time.sleep(min(max(delay, 2), 300))
            continue
        except (OSError, TimeoutError):
            pass
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Download did not complete: {path.name}")


def download_archive(record, connections):
    name = record["name"]
    if Path(name).name != name or not name.endswith(".zip"):
        raise ValueError("Archive name must be a ZIP basename")
    target = ROOT / "data/archives" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    size = record["bytes"]
    if not target.exists() or target.stat().st_size != size:
        parts = target.with_suffix(".zip.parts")
        parts.mkdir(exist_ok=True)
        # Preserve contiguous bytes from an earlier ordinary curl download.
        if target.exists():
            if target.stat().st_size > size:
                raise ValueError("Existing archive is larger than the upstream release")
            with target.open("rb") as previous:
                index = 0
                while True:
                    block = previous.read(CHUNK_SIZE)
                    if not block:
                        break
                    part = parts / f"{index:05d}.part"
                    if not part.exists() or part.stat().st_size < len(block):
                        part.write_bytes(block)
                    index += 1
        ranges = [
            (start, min(start + CHUNK_SIZE, size) - 1)
            for start in range(0, size, CHUNK_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [
                pool.submit(
                    download_chunk,
                    record["url"],
                    size,
                    start,
                    end,
                    parts / f"{index:05d}.part",
                )
                for index, (start, end) in enumerate(ranges)
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"{name}: {completed}/{len(ranges)} chunks complete", flush=True)
        assembled = target.with_suffix(".zip.assembling")
        with assembled.open("wb") as output:
            for index in range(len(ranges)):
                with (parts / f"{index:05d}.part").open("rb") as part:
                    shutil.copyfileobj(part, output, 4 * 1024 * 1024)
        if assembled.stat().st_size != size:
            raise ValueError("Assembled size does not match the release")
        assembled.replace(target)
    for algorithm, expected in record.get("hashes", {}).items():
        if file_hash(target, algorithm) != expected:
            raise ValueError(
                f"Upstream {algorithm} mismatch for {name}; preserve files for investigation"
            )
    return target


def extract_archive(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if (
                destination.resolve() not in target.parents
                and target != destination.resolve()
            ):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not accepted: {info.filename}")
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC failed: {bad_member}")
        archive.extractall(destination)
        return len(archive.infolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.connections <= 16:
        parser.error("connections must be between 1 and 16")
    sources = json.loads((ROOT / "metadata/sources.json").read_text())
    results = []
    for record in sources["archives"]:
        path = download_archive(record, args.connections)
        result = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        if not args.download_only:
            result["zip_members"] = extract_archive(path, ROOT / "data/raw")
            result["crc_check"] = "passed"
        results.append(result)
        (ROOT / "metadata/archives.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
