from pathlib import Path
import random
import shutil

root = Path("yolo11n/datasets/blackobj")

train_img_dir = root / "images" / "train"
train_lbl_dir = root / "labels" / "train"

val_img_dir = root / "images" / "val"
val_lbl_dir = root / "labels" / "val"

test_img_dir = root / "images" / "test"
test_lbl_dir = root / "labels" / "test"

for d in [
    val_img_dir,
    val_lbl_dir,
    test_img_dir,
    test_lbl_dir
]:
    d.mkdir(parents=True, exist_ok=True)

image_exts = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = [
    p for p in train_img_dir.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_exts
]

# 固定种子：重复运行时划分一致
random.seed(20260809)
random.shuffle(images)

total = len(images)

val_count = round(total * 0.15)
test_count = round(total * 0.15)

val_set = images[:val_count]

test_set = images[
    val_count:
    val_count + test_count
]

def move_pair(img, dst_img, dst_lbl):

    shutil.move(
        str(img),
        str(dst_img / img.name)
    )

    label = train_lbl_dir / f"{img.stem}.txt"

    # 有标签就移动
    # 没标签说明可能是负样本
    if label.exists():
        shutil.move(
            str(label),
            str(dst_lbl / label.name)
        )

for img in val_set:
    move_pair(
        img,
        val_img_dir,
        val_lbl_dir
    )

for img in test_set:
    move_pair(
        img,
        test_img_dir,
        test_lbl_dir
    )

print()
print("划分完成")
print("-----------------------")
print("总图片:", total)
print("train:", total - val_count - test_count)
print("val:", val_count)
print("test:", test_count)