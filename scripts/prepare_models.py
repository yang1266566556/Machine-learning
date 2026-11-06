"""Download official pretrained weights, never course data."""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coursework.runtime import WEIGHTS, configure_runtime

configure_runtime()

SOURCES = {
    "resnet34-b627a593.pth": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "yolo11n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    "yolo11s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
}
ASSETS = {
    "yolo11n-seg.pt": (196201396, 6182636),
    "yolo11s-seg.pt": (196201345, 20669228),
}


def download_yolo(name):
    asset, size = ASSETS[name]
    url = f"https://api.github.com/repos/ultralytics/assets/releases/assets/{asset}?download=1"
    folder = ROOT / ".cache" / "downloads" / name
    folder.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size

    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        part = folder / f"{index:04d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={
                    "Accept": "application/octet-stream", "Range": f"bytes={start}-{end}",
                    "User-Agent": "coursework-setup",
                })
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Unexpected official asset byte range")
                    with part.open("wb") as output:
                        while block := response.read(65536):
                            output.write(block)
                if part.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete asset chunk")
                return
            except Exception:
                if attempt == 3:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        for job in as_completed([pool.submit(fetch, i) for i in range(count)]):
            job.result()
    destination = WEIGHTS / name
    with destination.with_suffix(".pt.part").open("wb") as output:
        for i in range(count):
            output.write((folder / f"{i:04d}.part").read_bytes())
    destination.with_suffix(".pt.part").replace(destination)


def main():
    records = []
    for name, url in SOURCES.items():
        path = WEIGHTS / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            if name in ASSETS:
                download_yolo(name)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with partial.open("rb") as stream:
                downloaded_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if name.startswith("resnet34") and not downloaded_hash.startswith("b627a593"):
                raise ValueError("Downloaded ResNet34 checksum mismatch")
            partial.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if name.startswith("resnet34") and not digest.startswith("b627a593"):
            raise ValueError("ResNet34 checksum mismatch")
        if name in ASSETS and path.stat().st_size != ASSETS[name][1]:
            raise ValueError("YOLO official asset size mismatch")
        records.append({"file": name, "source": url, "bytes": path.stat().st_size, "sha256": digest})
        print(f"Ready: {name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    (ROOT / "models" / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
