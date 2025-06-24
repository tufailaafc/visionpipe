import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS

# gets the exif metadata from an image and prints it to console
def tagReader(path :str):
    # Read image with OpenCV
    img = cv2.imread(path)

    # Read EXIF metadata with Pillow
    image_pil = Image.open(path)
    exif_data = image_pil._getexif()

    if exif_data is not None:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag}: {value}")
    else:
        print("No EXIF metadata found")



# will take a folder and then get all of the meta data from the images inside
def getMetaDataDir(root_path: str):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            filePath = os.path.join(dirpath, filename)
            print(filePath)
            tagReader(filePath)
            


if __name__ == "__main__":
    getMetaDataDir("extracted_frames")