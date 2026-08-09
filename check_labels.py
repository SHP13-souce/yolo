from pathlib import Path

label_dir = Path("yolo11n/datasets/blackobj/labels/train")

total = 0
full_image = 0
bad = 0

for file in label_dir.glob("*.txt"):
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        total += 1

        values = line.split()

        if len(values) != 5:
            print("格式错误:", file, line)
            bad += 1
            continue

        cls, cx, cy, w, h = map(float, values)

        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            print("坐标错误:", file, line)
            bad += 1

        if w > 0.95 and h > 0.95:
            full_image += 1

print("标签总数:", total)
print("几乎整图框:", full_image)
print("格式/坐标异常:", bad)