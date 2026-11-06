"""Keep model caches and outputs inside this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "pretrained"


def configure_runtime():
    defaults = {
        "TORCH_HOME": ROOT / ".cache" / "torch",
        "HF_HOME": ROOT / ".cache" / "huggingface",
        "YOLO_CONFIG_DIR": ROOT / ".cache" / "ultralytics",
        "MPLCONFIGDIR": ROOT / ".cache" / "matplotlib",
    }
    for key, path in defaults.items():
        os.environ.setdefault(key, str(path))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    return ROOT
