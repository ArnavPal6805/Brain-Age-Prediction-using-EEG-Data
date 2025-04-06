import os
import shutil
import random

# === Configuration ===
input_folder = "/kaggle/input/eeg70to70"
output_base = "/kaggle/working/split"
output_train = os.path.join(output_base, "train")
output_test = os.path.join(output_base, "test")
split_ratio = 0.8  # 80% train, 20% test
file_extension = ".csv"

# === Create output directories ===
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# === Set seed for reproducibility ===
random.seed(42)

# === Process each label folder ===
for label in os.listdir(input_folder):
    class_folder = os.path.join(input_folder, label)
    if not os.path.isdir(class_folder):
        continue

    files = [f for f in os.listdir(class_folder) if f.endswith(file_extension)]
    if not files:
        continue

    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]

    # Paths to save train/test files
    train_label_folder = os.path.join(output_train, label)
    test_label_folder = os.path.join(output_test, label)

    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    # Copy train files
    for f in train_files:
        src = os.path.join(class_folder, f)
        dst = os.path.join(train_label_folder, f)
        shutil.copy2(src, dst)

    # Copy test files
    for f in test_files:
        src = os.path.join(class_folder, f)
        dst = os.path.join(test_label_folder, f)
        shutil.copy2(src, dst)

    print(f"📂 {label}: {len(train_files)} train | {len(test_files)} test")

print(f"\n✅ Data successfully split into:\n - Train: {output_train}\n - Test:  {output_test}")