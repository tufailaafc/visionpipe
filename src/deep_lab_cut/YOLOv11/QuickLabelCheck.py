import os

# A small script to check if there are any missing files between the image directory and label directory

img_dir = "/home/nathaniel/Documents/GitHub/visionpipe/src/deep_lab_cut/YOLOv11/yolo_images"
label_dir = "/home/nathaniel/Documents/GitHub/visionpipe/src/deep_lab_cut/YOLOv11/yolo_labels"

imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}

missing_labels = imgs - labels
missing_images = labels - imgs

print("Missing labels:", missing_labels)
print("Missing images:", missing_images)