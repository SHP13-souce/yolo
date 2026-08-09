from pathlib import Path
import random
import shutil

root = Path("yolo11n/datasets/blackobj")

train_images = root / "images" / "train"
train_labels = root / "labels" / "train"

val_images = root / "images" / "val"
val_labels = root / "labels" / "val"

val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

# 只选有对应标签的图片
images = []

for img in train_images.iterdir():
    if img.suffix.lower() not in image_exts:
        continue

    label = train_labels / f"{img.stem}.txt"

    if label.exists():
        images.append(img)
    else:
        print("警告：图片没有对应标签：", img.name)

# 固定随机种子，保证每次划分一致
random.seed(42)
random.shuffle(images)

# 20% 做验证集
val_count = round(len(images) * 0.2)

val_set = images[:val_count]

for img in val_set:
    label = train_labels / f"{img.stem}.txt"

    shutil.move(
        str(img),
        str(val_images / img.name)
    )

    shutil.move(
        str(label),
        str(val_labels / label.name)
    )

print()
print("划分完成")
print("总数据:", len(images))
print("训练集:", len(images) - val_count)
print("验证集:", val_count)