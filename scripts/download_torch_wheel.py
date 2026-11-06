"""Resumable ranged download of the exact official CUDA wheel for this host."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
NAME = "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"
URL = "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"
TRANSFER_URL = os.environ.get("COURSE_TORCH_DOWNLOAD_URL", URL)
SHA256 = "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0"
SIZE = 2862033275
CHUNK = 16 * 1024 * 1024
FOLDER = ROOT / ".cache" / "downloads"
PARTS = FOLDER / "torch-parts"


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = PARTS / f"{index:04d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return index
    for attempt in range(4):
        try:
            have = path.stat().st_size if path.exists() else 0
            if have > end - start + 1:
                raise ValueError("Oversized partial chunk")
            begin = start + have
            request = urllib.request.Request(TRANSFER_URL, headers={"Range": f"bytes={begin}-{end}"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{SIZE}":
                    raise ValueError("Server did not return the requested range")
                with path.open("ab") as output:
                    while block := response.read(65536):
                        output.write(block)
            if path.stat().st_size != end - start + 1:
                raise ValueError("Incomplete chunk")
            return index
        except Exception:
            if attempt == 3:
                raise


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = [pool.submit(fetch, i) for i in range(count)]
        for done, future in enumerate(as_completed(jobs), 1):
            future.result()
            if done % 5 == 0 or done == count:
                print(f"Downloaded {done}/{count} chunks", flush=True)
    dest = FOLDER / NAME
    digest = hashlib.sha256()
    with dest.open("wb") as output:
        for i in range(count):
            with (PARTS / f"{i:04d}.part").open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    assert digest.hexdigest() == SHA256, "Official wheel SHA256 mismatch"
    (FOLDER / "torch-download.json").write_text(json.dumps({"official_source": URL, "transfer_source": TRANSFER_URL, "sha256": SHA256, "size": SIZE}, indent=2))
    print(f"Verified official wheel: {dest}", flush=True)
    # Remove only the known chunks created by this downloader, after verification.
    for i in range(count):
        (PARTS / f"{i:04d}.part").unlink()
    PARTS.rmdir()


if __name__ == "__main__":
    main()
