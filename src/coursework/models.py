"""Offline model construction; these helpers do not train on course data."""
from .runtime import WEIGHTS, configure_runtime

configure_runtime()


def build_unet():
    import torch
    import segmentation_models_pytorch as smp

    checkpoint = WEIGHTS / "resnet34-b627a593.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Run scripts/prepare_models.py first.")
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state)
    return model


def build_yolo(size="n"):
    from ultralytics import YOLO

    if size not in {"n", "s"}:
        raise ValueError("Prepared sizes are n and s.")
    checkpoint = WEIGHTS / f"yolo11{size}-seg.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("Run scripts/prepare_models.py first.")
    return YOLO(str(checkpoint), task="segment")
