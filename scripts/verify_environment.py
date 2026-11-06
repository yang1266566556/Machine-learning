"""Dependency, CUDA and model checks using synthetic inputs only.

No course data is read, no optimizer is run, and no trained weights are saved.
"""
import gc
import importlib
import importlib.metadata as metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coursework.runtime import configure_runtime

configure_runtime()


def floating_tensors(value):
    import torch
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from floating_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from floating_tensors(item)


def main():
    import torch
    import torchvision
    from coursework.models import build_unet, build_yolo

    report_path = ROOT / "reports" / "environment_check.json"
    report_path.parent.mkdir(exist_ok=True)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "python": sys.version, "executable": sys.executable,
              "scope": "Synthetic input checks only; not course training or a performance benchmark."}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        modules = ["numpy", "pandas", "scipy", "sklearn", "imblearn", "openpyxl", "cv2",
                   "matplotlib", "seaborn", "jieba", "nltk", "tensorboard", "ipykernel",
                   "jupyterlab", "segmentation_models_pytorch", "ultralytics"]
        for name in modules:
            importlib.import_module(name)
        report["imports"] = modules
        report["versions"] = {n: metadata.version(n) for n in
                              ["torch", "torchvision", "segmentation-models-pytorch", "ultralytics", "numpy", "scikit-learn"]}
        assert torch.cuda.is_available(), "CUDA GPU is unavailable"
        torch.manual_seed(42)
        torch.set_num_threads(4)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
        report["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        x = torch.randn(256, 256, device="cuda")
        assert torch.isfinite(x @ x).all()
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device="cuda", dtype=torch.float32)
        kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
        assert kept.tolist() == [0], "CUDA NMS failed"
        report["cuda_matmul_and_nms"] = "passed"
        results = []
        for name in ["unet_resnet34", "yolo11n-seg", "yolo11s-seg"]:
            print(f"Checking {name} on GPU", flush=True)
            torch.cuda.reset_peak_memory_stats()
            if name == "unet_resnet34":
                net = build_unet().cuda().train()
            else:
                net = build_yolo("n" if name == "yolo11n-seg" else "s").model.cuda().float().train()
                # Ultralytics loads inference checkpoints with gradients disabled.
                # Enable the trainable parameters as its trainer does, retaining
                # the fixed DFL projection. No optimizer or updates are used here.
                for parameter_name, parameter in net.named_parameters():
                    parameter.requires_grad_(".dfl." not in parameter_name)
            # Synthetic gradient check, not task-specific training/loss validation.
            sample = torch.rand(2, 3, 128, 128, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(sample)
                tensors = list(floating_tensors(out))
                assert tensors and all(torch.isfinite(t).all() for t in tensors)
                loss = sum(t.float().square().mean() for t in tensors)
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            del out, tensors, grads, loss, sample
            net.zero_grad(set_to_none=True)
            net.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out = net(torch.rand(1, 3, 512, 512, device="cuda"))
                tensors = list(floating_tensors(out))
                assert tensors and all(torch.isfinite(t).all() for t in tensors)
                if name == "unet_resnet34":
                    assert tuple(out.shape) == (1, 1, 512, 512)
            torch.cuda.synchronize()
            results.append({"model": name, "pretrained_weights_loaded": True,
                            "synthetic_backward_128": "passed", "inference_512": "passed",
                            "parameters": sum(p.numel() for p in net.parameters()),
                            "check_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
            print(json.dumps(results[-1]), flush=True)
            del net, out, tensors
            gc.collect()
            torch.cuda.empty_cache()
        report["models"] = results
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
