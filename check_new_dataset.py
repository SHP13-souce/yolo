from pathlib import Path

root = Path("yolo11n/datasets/blackobj")

img_dir = root / "images" / "train"
lbl_dir = root / "labels" / "train"

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = {
    p.stem: p
    for p in img_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_exts
}

labels = {
    p.stem: p
    for p in lbl_dir.glob("*.txt")
}

print("图片总数:", len(images))
print("标签文件数:", len(labels))
print()

missing_labels = set(images) - set(labels)
extra_labels = set(labels) - set(images)

print("没有标签文件的图片:", len(missing_labels))
print("没有对应图片的标签:", len(extra_labels))

bad = []

for stem, label_path in labels.items():
    lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    for line_num, line in enumerate(lines, 1):

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            bad.append(
                (label_path.name, line_num, "字段数量不是5")
            )
            continue

        try:
            cls, x, y, w, h = map(float, parts)
        except ValueError:
            bad.append(
                (label_path.name, line_num, "不是数字")
            )
            continue

        if int(cls) != 0:
            bad.append(
                (label_path.name, line_num, f"class={cls}")
            )

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            bad.append(
                (
                    label_path.name,
                    line_num,
                    f"x={x}, y={y}, w={w}, h={h}"
                )
            )

print()
print("格式/坐标错误:", len(bad))

for item in bad[:20]:
    print(item)

print()
print("检查结束")