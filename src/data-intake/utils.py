import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import logging

logger = logging.getLogger()

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




# Will write a exif tag to the image for all supplied fields
def writeExifTag(image_path: str, author:str="", serialNumber:str="", dateTime:str="",userComment:str="", description:str=""):
    try:
        # Get existing EXIF data or create new
        exif_dict = piexif.load(image_path)


        # Modify EXIF fields if supplied

        # This will likely be the name for the camera
        if author != "":
            exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode('utf-8')
        
        # This will have the specific camera
        if serialNumber != "":
            exif_dict["0th"][piexif.ImageIFD.CameraSerialNUmber] = serialNumber.encode('utf-8')

        # This will allow us to know when the photo was taken
        if dateTime != "":
            # date time must come in the format dateTime = "2025:06:24 14:58:33" years:month:day hour:minute:seconds
            exif_dict["0th"][piexif.ImageIFD.dateTime] = dateTime.encode('utf-8')
            #tag for the visual display that is where most apps look for date created
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dateTime.encode('utf-8')

        # Any user input like purpose for the image etc
        if userComment != "":
            exif_dict["Exif"][piexif.ImageIFD.userComment] = b"ASCII\x00\x00\x00"+userComment.encode("ascii")

        # A brief description of what the picture contains, could be useful for ai model
        if description != "":
            exif_dict["0th"][piexif.ImageIFD.description] = description.encode('utf-8')


        # Convert dictionary back to bytes so that we can insert back into image
        exif_bytes = piexif.dump(exif_dict)

        # Save with the new changes to Exif tags
        piexif.insert(exif_bytes,image_path)

    except Exception as e:
        print(f"❌ Failed to write EXIF tag to {image_path}: {e}")

            


if __name__ == "__main__":
    getMetaDataDir("extracted_frames")