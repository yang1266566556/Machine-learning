"""Create five traceable practice crops; never modify source images or labels."""
from pathlib import Path
import hashlib
import json
import tomllib

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/annotation_practice/batch_01"
SOURCE_NAME = "DJI_20260416133602_0677_D.JPG"
# Pixel coordinates in the original decoded image, not screen coordinates.
SAMPLES = [
    ("practice_001", 4200, 2100, "先练习中部细长纹理；白色标线不标为裂缝。"),
    ("practice_002", 4500, 2900, "检查细裂缝与周围沥青纹理的区别。"),
    ("practice_003", 3400, 1900, "练习分支；较宽深色痕迹可能涉及修补，需人工审核。"),
    ("practice_004", 3600, 4000, "练习多条目标；黄色标线不标为裂缝。"),
    ("practice_005", 4200, 4200, "练习标线附近的裂缝；不推测被标线遮挡的部分。"),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = tomllib.loads((ROOT / "configs/data_paths.toml").read_text("utf-8"))
    source = Path(config["sources"]["bridge_images"]) / SOURCE_NAME
    source_hash = sha256(source)
    with Image.open(source) as original:
        original.load()
        for _, x, y, _ in SAMPLES:
            assert 0 <= x < x + 512 <= original.width
            assert 0 <= y < y + 512 <= original.height
        # Refuse to overwrite an existing batch, especially user annotations.
        OUTPUT.mkdir(parents=True, exist_ok=False)
        images = OUTPUT / "images"
        images.mkdir()
        records = []
        overview = Image.new("RGB", (3 * 320, 2 * 350), "white")
        draw = ImageDraw.Draw(overview)
        for i, (name, x, y, note) in enumerate(SAMPLES):
            box = (x, y, x + 512, y + 512)
            crop = original.crop(box)
            target = images / f"{name}.png"
            crop.save(target)
            with Image.open(target) as saved:
                assert saved.size == (512, 512)
                assert saved.mode == crop.mode and saved.tobytes() == crop.tobytes()
            records.append({
                "image": f"images/{name}.png",
                "sha256": sha256(target),
                "source_file": str(source),
                "source_sha256": source_hash,
                "source_size_wh": list(original.size),
                "crop_box_xyxy_exclusive": list(box),
                "resize": False,
                "split": "practice_only",
                "physical_section_group": None,
                "label_status": "unlabeled_needs_human_review",
                "note": note,
            })
            thumb = crop.resize((320, 320))
            tx, ty = (i % 3) * 320, (i // 3) * 350
            overview.paste(thumb, (tx, ty + 30))
            draw.text((tx + 5, ty + 8), name, fill="black")
        overview.save(OUTPUT / "overview.jpg")
    assert sha256(source) == source_hash, "Source image changed during preparation"
    (OUTPUT / "manifest.json").write_text(
        json.dumps({
            "purpose": "annotation practice only, not a training/validation/test split",
            "coordinate_system": "decoded original pixels; origin top-left; right/bottom exclusive",
            "source_unchanged_verified": True,
            "saved_pixels_match_source_crops": True,
            "samples": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text("""# 第一批标注练习图

本批包含 5 张 512×512 原始像素裁图，来自同一张原图。未缩放训练图片、未修改原图、未生成任何自动标签。
overview.jpg 只是缩小后的浏览图，不要标注它。manifest.json 记录来源、坐标和文件校验值。

## 现在只标第一张

1. 在 Labelme 点击“打开文件夹”，选择本目录下的 images 文件夹。
2. 打开 practice_001.png，用“多边形”围出一小段确认的裂缝，标签统一为 crack。白色标线不标。
3. 将 JSON 保存到 images 文件夹，命名 practice_001.json，与图片同名。截图反馈后再继续。

不要在旧的整图 JSON 中继续画这些小图：整图和小图的坐标不同。
已有整图练习标注保留即可。本脚本没有读取、更改或转换它们。

## 标注规则与限制

- 多边形表示目标面积，沿一侧边缘走到末端，再沿另一侧返回闭合。
- 首次只画短段练习；正式使用时，一张图片内所有可确认的目标裂缝均需检查，不能遗漏后直接当作背景。
- 深色条带可能是裸露裂缝、灌缝或修补痕迹；有疑问先记录图片名和位置，不默认标为裂缝或背景。
- 当前没有真值标签，也不将任何图片预先认定为“无裂缝”。
- 本批仅用于学习标注，不满足正式数据规模要求。不得将这 5 张图随机拆为训练、验证和测试集。
- 正式数据先核对至少 12 张原图对应的物理桥段与重复视角，按桥段分组划分，再裁切。
- 是否纳入正式数据要在审核类别定义、标注完整性、物理桥段分组之后决定。

脚本再次运行会拒绝覆盖本批目录，保护你之后保存的 JSON。
""", encoding="utf-8")
    print(f"Created and verified {len(records)} crops: {OUTPUT}")


if __name__ == "__main__":
    main()
