import os
import shutil
import random


# This files only job is to take a dataset with images and labels and split them into 
# training, testing and validation folders

# -------------------------
# USER CONFIG
# -------------------------
# This is the parent folder where we expect to find all of the images and label folders
ROOT_DIR = "/home/nathaniel/Documents/GitHub/visionpipe/src/deep_lab_cut/YOLOv11/"
# This is the folder we expect all images to be in
IMAGES_DIR = f"{ROOT_DIR}yolo_images"
# This is the folder we expect the labels to be in
LABELS_DIR = f"{ROOT_DIR}yolo_labels"

# This is where the split data folder is or will be created
OUT_DIR = f"{ROOT_DIR}training_dataset"

# The ratio we are splitting our by should total exactly 1 when all values are added.
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# -------------------------

# Create directories
for t in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, "images", t), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", t), exist_ok=True)

# Collect all image files
images = [f for f in os.listdir(IMAGES_DIR)
          if f.lower().endswith((".jpg", ".png", ".jpeg"))]

images.sort()
random.shuffle(images)

total = len(images)
n_train = int(total * train_ratio)
n_val = int(total * val_ratio)

train_files = images[:n_train]
val_files = images[n_train:n_train + n_val]
test_files = images[n_train + n_val:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

def move_files(file_list, split):
    for img in file_list:
        label = os.path.splitext(img)[0] + ".txt"
        
        img_src = os.path.join(IMAGES_DIR, img)
        label_src = os.path.join(LABELS_DIR, label)

        img_dst = os.path.join(OUT_DIR, "images", split, img)
        label_dst = os.path.join(OUT_DIR, "labels", split, label)

        shutil.copy2(img_src, img_dst)

        if os.path.exists(label_src):  # skip images without labels
            shutil.copy2(label_src, label_dst)
        else:
            print(f"[WARNING] No label for image: {img}")

# Move files
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split complete!")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
