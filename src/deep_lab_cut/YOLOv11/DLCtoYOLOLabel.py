import os
import pandas as pd
import cv2
from glob import glob

# This script will take DLC styled labels and convert them into yolo labels.

#Known issue


# ---------------------------
# USER SETTINGS
# ---------------------------

# Where we look for the csv's. Includes all subdirectories.
PARENT_CSV_FOLDER = "/home/nathaniel/Documents/GitHub/visionpipe/data/raw/deep_lab_cut/osfstorage-archive/DLC_tracking/labeled-data/labeled-data"

# root folder that contains all images
IMAGES_DIR = "/home/nathaniel/Documents/GitHub/visionpipe/data/raw/deep_lab_cut/osfstorage-archive/DLC_tracking/labeled-data"

# The folder we output the label files to
YOLO_LABELS_OUT = "/home/nathaniel/Documents/GitHub/visionpipe/src/deep_lab_cut/YOLOv11/trainingsetv2"

# Important!!! must match what is in the csv otherwise it will break.
SCORER = "WG"

CLASS_ID = 0  # pig
VISIBILITY_DEFAULT = 0  # for missing keypoints
os.makedirs(YOLO_LABELS_OUT, exist_ok=True)

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def process_csv(csv_file):
    print(f"Processing: {csv_file}")
    df = pd.read_csv(csv_file, header=[0, 1, 2])

    # Extract bodyparts and number of keypoints
    bodyparts = sorted(list({bp for (_, bp, _) in df.columns[1:]}))
    num_keypoints = len(bodyparts)
    print(f"Detected {num_keypoints} keypoints:", bodyparts)

    for idx, row in df.iterrows():
        # Get frame file path
        frame_file = row[("scorer", "bodyparts", "coords")] if isinstance(row[0], str) else row[0]
        if isinstance(frame_file, float):
            continue

        frame_file = frame_file.replace("\\", "/")
        frame_file = os.path.normpath(frame_file)
        img_path = os.path.join(IMAGES_DIR, frame_file)

        if not os.path.exists(img_path):
            print(f"WARNING: Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        kpts = []
        xs, ys = [], []

        for bp in bodyparts:
            x = row[(SCORER, bp, "x")]
            y = row[(SCORER, bp, "y")]

            if pd.isna(x) or pd.isna(y):
                kpts.extend([0.0, 0.0, VISIBILITY_DEFAULT])
            else:
                kpts.extend([x / w, y / h, 1])
                xs.append(x)
                ys.append(y)

        if len(xs) == 0:
            # no visible keypoints, skip
            continue

        # Bounding box around visible keypoints
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox_xc = ((x_min + x_max) / 2) / w
        bbox_yc = ((y_min + y_max) / 2) / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h

        # Build YOLOv11 pose line
        yolo_line = f"{CLASS_ID} {bbox_xc:.6f} {bbox_yc:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        for v in kpts:
            yolo_line += f" {v:.6f}"

        # Write label file
        out_path = os.path.join(
            YOLO_LABELS_OUT,
            os.path.basename(frame_file).replace(".png", ".txt").replace(".jpg", ".txt")
        )
        with open(out_path, "w") as f:
            f.write(yolo_line)

# ---------------------------
# PROCESS ALL CSVs
# ---------------------------
all_csv_files = glob(os.path.join(PARENT_CSV_FOLDER, "**/*.csv"), recursive=True)
print(f"Found {len(all_csv_files)} CSV files.")

for csv_file in all_csv_files:
    process_csv(csv_file)

print("\nDone! YOLOv11 pose labels created at:")
print(YOLO_LABELS_OUT)
