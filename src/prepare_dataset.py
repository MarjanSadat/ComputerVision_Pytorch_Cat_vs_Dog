import random
import shutil
from pathlib import Path

# -----------------------------
# تنظیمات
# -----------------------------
SOURCE_DIR = Path("dataset")      # پوشه‌ای که cat.* و dog.* داخل آن هستند
DEST_DIR = Path("dataset")

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# -----------------------------
# ساخت پوشه‌ها
# -----------------------------
for split in ["train", "valid", "test"]:
    for cls in ["cat", "dog"]:
        (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# -----------------------------
# تابع تقسیم تصاویر
# -----------------------------
def split_and_copy(class_name):

    images = list(SOURCE_DIR.glob(f"{class_name}.*.jpg"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    for img in train_images:
        shutil.copy2(img, DEST_DIR / "train" / class_name / img.name)

    for img in valid_images:
        shutil.copy2(img, DEST_DIR / "valid" / class_name / img.name)

    for img in test_images:
        shutil.copy2(img, DEST_DIR / "test" / class_name / img.name)

    print(f"{class_name} done.")
    print(f"Train : {len(train_images)}")
    print(f"Valid : {len(valid_images)}")
    print(f"Test  : {len(test_images)}")
    print("-" * 30)

# -----------------------------
# اجرا
# -----------------------------
split_and_copy("cat")
split_and_copy("dog")

print("Dataset prepared successfully.")