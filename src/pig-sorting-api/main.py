from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#from ultralytics import YOLO
from PIL import Image
#import supervision as sv
import io
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
from client import db# this contains our connection to mongoDB
from pymongo.results import InsertOneResult
import os
from bson import ObjectId
import bson
import logging
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger()


def image_to_base64(img: np.ndarray) -> str:
    # Ensure image is in BGR format for cv2.imencode as PIL loads in RGB
    if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's a 3-channel image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img # Assume it's already in a suitable format or grayscale

    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()


@app.get("/api/v1/mongoData", response_class=JSONResponse)
async def get_mongo_data():
    try:
        db_result = db["metaData"]
        result = db_result.find()

        #Parse JSON string into python objects to allow creation of a dataframe
        meta_data_json = json.loads(bson.json_util.dumps(result))

        

        return meta_data_json
            

    except Exception as e:
        logger.info(f"Error getting data :{e}")
        return JSONResponse(status_code=500, content={"error": str(e)})




# @app.post("/api/v1/predict/", response_class=JSONResponse)
# async def predict(file: UploadFile = File(...), model_id: int = Query(2), colour_corrected: bool = True, ground_truth:int=None):
#     try:
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # PIL Image
        
#         #perform colour correction if Colour correct is true
#         if colour_corrected:
#            #converting it to a numpy array becuase that is what colour corrector expects
#            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#            image = cc.correct(image_bgr)
#            image = (image * 255).astype(np.uint8)  # Ensure it's uint8 for YOLO
#         result = run_model(image, model_id)
#         print("before submit result function")
        
#         #adding tag for if it was colour corrected
#         result["colour_corrected"]=colour_corrected
        
#         #add tag for if it were classification or segmentation
#         result["type"]="classification" if model_id==1 else "segmentation"
        
#         #add groundtruth if available
#         result["ground_truth"]=ground_truth
        
#         submitResult(result, image, file.filename)
        
#         #print(f"result after submit result: {result}")
#         return result
#     except Exception as e:
#         # Log the exception for debugging purposes
#         print(f"Error in predict endpoint: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# class ImageData(BaseModel):
#     image: str
#     model_id: int = 2

# @app.post("/api/v1/predict_base64/", response_class=JSONResponse)
# async def predict_base64(data: ImageData, colour_corrected: bool = True, ground_truth:int=None):
#     try:
#         image_bytes = base64.b64decode(data.image)
#         image = Image.open(BytesIO(image_bytes)).convert("RGB") # PIL Image
        
#         #perform colour correction if Colour correct is true
#         if colour_corrected:
#             #converting it to a numpy array becuase that is what colour corrector expects
#            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#            image = cc.correct(image_bgr)
#            image = (image * 255).astype(np.uint8)  # Ensure it's uint8 for YOLO
        
#         result = run_model(image, data.model_id)
        
#         #adding tag for if it was colour corrected
#         result["colour_corrected"]=colour_corrected
        
        
#         #add tag for if it were classification or segmentation
#         result["type"] = "classification" if data.model_id == 1 else "segmentation"
        
#         #add groundtruth if available
#         result["ground_truth"]=ground_truth
        
#         submitResult(result, image)
#         #print(f"result after submit result{result}")
#         return result
#     except Exception as e:
#         # Log the exception for debugging purposes
#         print(f"Error in predict_base64 endpoint: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)
        
        
# Helper function to convert ObjectId to string
def convert_objectid_to_str(data):
    if isinstance(data, dict):
        return {key: convert_objectid_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)  # Convert ObjectId to string
    else:
        return data
        
        
#takes a json object and saves it to mongoDB, saves the image in a mounted folder
def submitResult(result, image, filename=None):
    imageLoc = saveImage(image)
    
    # Adding the file location to our result object
    result["image_path"] = imageLoc
    #adds the model used to get result
    result["class_model"] = classification_model_name
    result["segmentation_model"] = segmentation_model_name
    
    #get rid of image
    #del result["overlay"]
    
    # Convert ObjectId fields in the result to strings
    result = convert_objectid_to_str(result)
    
    
    #make a copy so we do not change the version sent to the user
    db_result_data = result.copy()
    
    #get rid of image so we do not store it inside MongoDB
    del db_result_data["overlay"]
    
    # Save the result into MongoDB
    try:
        db_result = db["result"]
        insert_result = db_result.insert_one(db_result_data)
        if isinstance(insert_result, InsertOneResult) and insert_result.acknowledged:
            print(f"[INFO] Successfully inserted result into MongoDB: {insert_result.inserted_id}")
        else:
            print(f"[ERROR] MongoDB insert failed")
    
    except Exception as e:
        print(f"[ERROR] Error inserting result into MongoDB: {e}")
    
    print(db_result_data)  # Optional: To log the inserted result for testing and debugging
	
 
# saves the images into a mounted folder, returns the save location
def saveImage(image, filename: str = None):
    #The relative path to store the images.
    IMAGE_DIR = "images"
    
    #the absoulte path within docker, used for testing and debugging
    absolute_image_dir = os.path.abspath(IMAGE_DIR)
    print(f"[INFO] Saving images to directory: {absolute_image_dir}")

    try:
        if filename is None:
            filename = f"meat_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Ensure the image directory exists
        if not os.path.exists(IMAGE_DIR):
            os.makedirs(IMAGE_DIR)

        filepath = os.path.join(IMAGE_DIR, filename)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image.save(filepath, format="JPEG")

        print(f"[INFO] Saved image to: {filepath}")
        return filepath

    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}")
        return None
        
        
        
        
        
        
        