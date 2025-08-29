from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from PIL import Image
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
import threading
import httpx
import asyncio
import aiofiles
import time
from datetime import datetime, timedelta

# Set to the log level desired, also may stop fastapi from suppressing the logs.


# Set this to uvicorn.info so that the logs will propagate
# logger = logging.getLogger('uvicorn.info')
# logger = logging.getLogger('uvicorn.info')
logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



DATA_INTAKE_URL = "http://data-intake:8301" 
PIG_SORTING_API_URL = ""
MODEL_TRAINING_URL = "http://model-training:8401/training/stream"
MODEL_DEPLOY_URL = "http://model-deploy:8601"
#DEFAULT_MODELS = ["app/models/Proper_Test/70 epochs medium run3/weights/best.pt","app/models/Proper_Test/70 epochs medium run3/weights/last.pt"]
# These can be called whatever you want.
DEFAULT_MODEL_NAMES = ["70run3Best", "70run3Last"]
# These are the relative paths inside of the model-deploy container and must be inside of the models folder.
DEFAULT_MODEL_PATHS = ["/app/models/Proper_Test/70 epochs medium run3/weights/best.pt","/app/models/Proper_Test/70 epochs medium run3/weights/last.pt"]

# A list of our cameras
class CaptureRequest(BaseModel):
    channels: list[int]



# for updating how often the cameras take pictures, 
# interval is how many frames between taking pictures and channel is the camera port it is connected to.
class FrameIntervalRequest(BaseModel):
    channel: int
    interval: int


# A custom class to ensure we get the input as expected for selecting times for monogoDB
class TimesRequest(BaseModel):
    times: List[str]


class InferenceRequest(BaseModel):
    model_name: str
    image_base64: str
    confidence: float = 0.5
    return_annotated: bool = True


# ---- Schemas (mirrors backend for model-deploy) ----
class ModelResult(BaseModel):
    model: str
    detections: List[Dict[str, Any]]
    detection_count: int

class ChainResult(BaseModel):
    chain: List[ModelResult]
    annotated_image: Optional[str] = None

class ChainPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


class ChainPredictionRequest(BaseModel):
    model_names: List[str]
    instances: List[Dict[str, str]]
    parameters: Optional[Dict[str, Any]] = None


# Define request model
class SavePredictionRequest(BaseModel):
    annotated_image: str                  # base64-encoded image
    detections: List[Dict[str, Any]]      # bounding boxes, confidences, etc.
    metadata: Optional[Dict[str, Any]] = None
    ImageName: str = None


CHAIN_PREDICTION_URL = f"{MODEL_DEPLOY_URL}/model/predict/chain"

async def load_default_models():
    """Load default models into model-deploy."""
    async with httpx.AsyncClient(timeout=300) as client:
        for name, path in zip(DEFAULT_MODEL_NAMES, DEFAULT_MODEL_PATHS):
            try:
                resp = await client.post(
                    f"{MODEL_DEPLOY_URL}/model/load",
                    json={"model_name": name, "model_path": path},
                )
                resp.raise_for_status()
                logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")

async def send_to_chain_prediction(image_path: str):
    """Read image async, encode as base64, send to chain prediction."""
    if not os.path.exists(image_path):
        logger.warning(f"Image path does not exist: {image_path}")
        return

    try:
        async with aiofiles.open(image_path, "rb") as f:
            img_bytes = await f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "model_names": DEFAULT_MODEL_NAMES,
            "instances": [{"image": img_b64}],
            "parameters": {
                "return_annotated_image": True,
                "confidence": 0.25
            }
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(CHAIN_PREDICTION_URL, json=payload)
            resp.raise_for_status()
            logger.info(f"Chain prediction completed for {image_path}")
            result = resp.json()

            # call save prediction
        # file_name = os.path.basename(image_path)
        # logger.debug(f"file_name: {file_name}")
        # listOfJson = resp.json().get("predictions", {})
        # logger.info(f"Prediction results: {listOfJson[0].get("chain")}")
        # results = resp.json().get("predictions", [])
            
        #image = results.get("annotated_image")
        #detections = resp.get("chain").get("detections",{})
        #detections = [step["detections"] for step in results.get("chain", [])]
        #metadata = [step["model"] for step in results.get("chain", [])]
                            # "detections": [step["detections"] for step in resp.get("chain", [])],
                            # "metadata": {"models": [step["model"] for step in resp.get("chain", [])]},
                        
        #save_request = SavePredictionRequest(img_b64, detections, metadata, file_name)
        #save_prediction(save_request)
        predictions = result.get("predictions", [])
        if not predictions:
            logger.warning(f"No predictions returned for {image_path}")
            return

        chain = predictions[0]["chain"]
        annotated_image = predictions[0].get("annotated_image", "")
        # detections = [step["detections"] for step in chain]
        detections = []
        for step in chain:
            model_name = step.get("model")
            for det in step.get("detections", []):
                # Add model name to each detection
                det_with_model = {**det, "model": model_name}
                detections.append(det_with_model)
        metadata = {"models": [step["model"] for step in chain]}

        # Call save_prediction
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        save_req = SavePredictionRequest(
            annotated_image=annotated_image,
            detections=detections,
            metadata=metadata,
            ImageName=file_name
        )
        #logger.info("Final save_prediction payload:\n%s", json.dumps(save_req.dict(), indent=2))

        async with httpx.AsyncClient() as client:
            save_resp = await client.post("http://pig-sorting-api:8001/save_prediction", json=save_req.dict())
            save_resp.raise_for_status()
            logger.info(f"✅ Prediction saved for {image_path}")

    except Exception as e:
        logger.error(f"Failed to send chain prediction for {image_path}: {e}")

# async def watch_mongo():
#     """Watch MongoDB for new inserts and process images."""
#     logger.info("Starting MongoDB watcher...")
#     await load_default_models()

#     try:
#         async with db.watch([{"$match": {"operationType": "insert"}}]) as stream:
#             async for change in stream:
#                 doc = change["fullDocument"]
#                 image_path = doc.get("image_path")
#                 if image_path:
#                     logger.info(f"New image detected: {image_path}")
#                     await send_to_chain_prediction(image_path)
#     except Exception as e:
#         logger.error(f"Mongo watcher failed: {e}")


# async def watch_mongo():
#     """Poll MongoDB for new images instead of using change streams."""
#     logger.info("Starting MongoDB polling...")
#     await load_default_models()
    
#     last_check = datetime.utcnow() - timedelta(minutes=1)
    
#     while True:
#         try:
#             # Find new documents since last check
#             new_docs = db["metaData"].find({
#                 "date_time": {"$gt": last_check}
#             })
#             logger.debug("checked for new images")
#             # added asyc check next time.
#             async for doc in new_docs:
#                 image_path = doc.get("image_path")
#                 if image_path:
#                     logger.info(f"New image detected: {image_path}")
#                     await send_to_chain_prediction(image_path)
            
#             last_check = datetime.utcnow()
#             await asyncio.sleep(30)  # Poll every 30 seconds
            
#         except Exception as e:
#             logger.error(f"Mongo polling failed: {e}")
#             await asyncio.sleep(60)
CATCHUP_WINDOW_MINUTES = 5
STATE_COLLECTION = "watcher_state"
STATE_ID = "mongo_poller"  # unique key for this watcher


async def get_last_checkpoint():
    """Load last checkpoint from Mongo, or fallback to catch-up window."""
    state = await db[STATE_COLLECTION].find_one({"_id": STATE_ID})
    if state and "last_check" in state:
        ts = state["last_check"]
        logger.info(f"Resuming from saved checkpoint: {ts}")
        return ts
    else:
        ts = datetime.utcnow() - timedelta(minutes=CATCHUP_WINDOW_MINUTES)
        logger.info(f"No saved checkpoint, using catch-up: {ts}")
        return ts


async def save_checkpoint(ts: datetime):
    """Save last checkpoint to Mongo."""
    await db[STATE_COLLECTION].update_one(
        {"_id": STATE_ID},
        {"$set": {"last_check": ts}},
        upsert=True
    )


#TODO change mongo to a replica database so that we can subscribe to changes rather than polling.
async def watch_mongo():
    """Poll MongoDB for new images instead of using change streams."""
    logger.info("Starting MongoDB polling...")
    await load_default_models()

    last_check = await get_last_checkpoint()

    while True:
        try:


            # if there are new images then send them through the prediction engine.
            logger.info("Trying to get metaData")
            cursor = db["metaData"].find({
                "date_time": {"$gt": last_check}
            }).sort("date_time", 1)  # sort ascending to move checkpoint correctly
            logger.info("Created cursor")
            docs = await cursor.to_list(length=100)
            logger.info("Got the documents")
            if docs:
                logger.info(f"Found {len(docs)} new docs since {last_check}")

            for doc in docs:
                image_path = doc.get("image_path")
                if image_path:
                    logger.info(f"New image detected: {image_path}")
                    await send_to_chain_prediction(image_path)

                # update checkpoint after each doc
                if "date_time" in doc:
                    last_check = doc["date_time"]
                    await save_checkpoint(last_check)

            await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Mongo polling failed: {e}")
            await asyncio.sleep(60)

def start_watcher_loop():
    logger.debug("Trying to start watcher loop")
    """Start the async watcher in an asyncio loop."""
    loop = asyncio.get_event_loop()
    loop.create_task(watch_mongo())

# Call this in your FastAPI startup event
time.sleep(10)
start_watcher_loop()









def image_to_base64(img: np.ndarray) -> str:
    # Ensure image is in BGR format for cv2.imencode as PIL loads in RGB
    if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's a 3-channel image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img # Assume it's already in a suitable format or grayscale

    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()



# Will send all of the records to the user
@app.get("/api/v1/mongoData", response_class=JSONResponse)
async def get_mongo_data():
    try:
        logger.debug("🔍 /api/v1/mongoData/ endpoint hit")
        db_result = db["metaData"]
        result = db_result.find()

        #Parse JSON string into python objects to allow creation of a dataframe
        meta_data_json = json.loads(bson.json_util.dumps(result))

        

        return meta_data_json
            

    except Exception as e:
        logger.error(f"Error getting data :{e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    


# Will send all of the images selected between certain times
@app.post("/api/v1/images/", response_class=JSONResponse)
async def get_images_by_date(request: TimesRequest):
    logger.debug("🔍 /api/v1/images/ endpoint hit")
    try:
        # Get the data into a  form that will help us query the database
        times = request.times
        logger.debug(f"time 1: {times[0]}, time 2: {times[0]}")
        start_date = datetime.fromisoformat(times[0])
        end_date = datetime.fromisoformat(times[1])

        # get all records within the provided timeframe
        db_result = db["metaData"].find({
            "date_time": {
                "$gte": start_date,
                "$lte": end_date
            }
        })

        # Iterate through all records and store them in a list
        images_base64 = []
        for doc in db_result:
            path = doc.get("image_path")
            if path and os.path.exists(path):
                with open(path, "rb") as img_file:
                    b64_str = base64.b64encode(img_file.read()).decode()
                    images_base64.append({
                        "filename": os.path.basename(path),
                        "image": b64_str,
                        "timestamp": doc.get("date_time").isoformat()
                    })


        return images_base64

    except Exception as e:
        logger.error(f"Error fetching images by date: {e}")
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

@app.get("/api/v1/training/trainer", response_class=JSONResponse)
async def run_training(
    model_path: str = Query(...),
    dataset_path: str = Query(...),
    project_name: str = Query("default_project"),
    run_name: str = Query("run_001")
):
    try:
        logger.info("Training request received")

        # Send request to model-training container as it has gpu support
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get("http://model-training:8401/training/trainer",
            params={
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_name": project_name,
            "run_name": run_name
        })
        data = response.json()

        # Log to MongoDB
        result_doc = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "model_export_path": str(data.get("model_export_path")),
            "project": project_name,
            "run": run_name,
            "train_results": str(data.get("train_results")),
            "val_results": str(data.get("val_results")),
            "date_time": datetime.utcnow()
        }

        logger.info(f"Training Logs: {result_doc}")
        db["training_logs"].insert_one(result_doc)

        return {
            "message": "Training completed!",
            "model_export_path": str(data.get("model_export_path")),
            "train_results": str(data.get("train_results")),
            "val_results": str(data.get("val_results")),
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/api/v1/training/stream_trainer")
async def proxy_training_stream(
    model_path: str = Query(...),
    dataset_path: str = Query(...),
    project_name: str = Query("default_project"),
    run_name: str = Query("run_001"),
    epochs: int = Query(1)
):
    try:
        logger.info("Streaming training request received")

        # The parameters to configure the model training
        # model_training_url = "http://model-training:8401/training/stream"
        params = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_name": project_name,
            "run_name": run_name,
            "epochs": epochs
        }

        # Async generator to yield streamed training data
        async def event_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", MODEL_TRAINING_URL, params=params) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            json_data = line.replace("data: ", "").strip()

                            try:
                                parsed = json.loads(json_data)
                                logger.info(f"Streaming event: {parsed}")

                                # Log final result to MongoDB
                                if parsed.get("status") == "done":
                                    result_doc = {
                                        "model_path": model_path,
                                        "dataset_path": dataset_path,
                                        "model_export_path": parsed.get("path"),
                                        "project": project_name,
                                        "run": run_name,
                                        "train_results": parsed.get("train_results"),  
                                        "val_results": parsed.get("val_results"),
                                        "date_time": datetime.utcnow()
                                    }
                                    db["training_logs"].insert_one(result_doc)
                                    logger.info(f"Inserted training log for run: {run_name}")

                            except Exception as e:
                                logger.warning(f"Failed to parse JSON event: {e}")

                            yield f"{line}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Streaming training failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

        
        
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
        
        
# #takes a json object and saves it to mongoDB, saves the image in a mounted folder
# def submitResult(result, image, filename=None):
#     imageLoc = saveImage(image)
    
#     # Adding the file location to our result object
#     result["image_path"] = imageLoc
#     #adds the model used to get result
#     result["class_model"] = classification_model_name
#     result["segmentation_model"] = segmentation_model_name
    
#     #get rid of image
#     #del result["overlay"]
    
#     # Convert ObjectId fields in the result to strings
#     result = convert_objectid_to_str(result)
    
    
#     #make a copy so we do not change the version sent to the user
#     db_result_data = result.copy()
    
#     #get rid of image so we do not store it inside MongoDB
#     del db_result_data["overlay"]
    
#     # Save the result into MongoDB
#     try:
#         db_result = db["result"]
#         insert_result = db_result.insert_one(db_result_data)
#         if isinstance(insert_result, InsertOneResult) and insert_result.acknowledged:
#             print(f"[INFO] Successfully inserted result into MongoDB: {insert_result.inserted_id}")
#         else:
#             print(f"[ERROR] MongoDB insert failed")
    
#     except Exception as e:
#         print(f"[ERROR] Error inserting result into MongoDB: {e}")
    
#     print(db_result_data)  # Optional: To log the inserted result for testing and debugging
	
 
# # saves the images into a mounted folder, returns the save location
# def saveImage(image, filename: str = None):
#     #The relative path to store the images.
#     IMAGE_DIR = "images"
    
#     #the absoulte path within docker, used for testing and debugging
#     absolute_image_dir = os.path.abspath(IMAGE_DIR)
#     print(f"[INFO] Saving images to directory: {absolute_image_dir}")

#     try:
#         if filename is None:
#             filename = f"meat_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

#         # Ensure the image directory exists
#         if not os.path.exists(IMAGE_DIR):
#             os.makedirs(IMAGE_DIR)

#         filepath = os.path.join(IMAGE_DIR, filename)
        
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image)
            
#         image.save(filepath, format="JPEG")

#         print(f"[INFO] Saved image to: {filepath}")
#         return filepath

#     except Exception as e:
#         print(f"[ERROR] Failed to save image: {e}")
#         return None
        
        
        
        
        
        
# @app.post("/api/v1/collector/capture")
# async def capture(channels: List[int] = Body(...)):
#     """
#     Proxy to data-collector to take manual capture(s)
#     """
#     async with httpx.AsyncClient() as client:
#         response = await client.post(f"{DATA_INTAKE_URL}/capture/", json=channels)
#         return response.json()


# @app.post("/api/v1/collector/frame_interval")
# async def set_frame_interval(interval: int = Body(..., embed=True)):
#     """
#     Proxy to data-collector to update frame interval
#     """
#     async with httpx.AsyncClient() as client:
#         response = await client.post(f"{DATA_INTAKE_URL}/frame_interval/", json={"interval": interval})
#         return response.json()

@app.post("/api/v1/collector/capture")
async def capture(request: CaptureRequest):
    """
    Proxy to data-collector to take manual capture(s)
    """
    try:
        logger.debug(f"Capture channels: {request.channels}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATA_INTAKE_URL}/capture/",
                json={"channels": request.channels}
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error sending capture request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v1/collector/frame_interval")
async def set_frame_interval(request: FrameIntervalRequest):
    """
    Proxy to data-intake to update frame interval for a specific camera
    """
    try:
        print(f"Trying to connect to: {DATA_INTAKE_URL}/frame_interval")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATA_INTAKE_URL}/frame_interval",
                json={"channel": request.channel, "interval": request.interval}
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error sending frame interval request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    







# class ChainPredictionResponse(BaseModel):
#     results: List[Dict[str, Any]]

# # ---- Forwarding Endpoint ----
# @app.post("/chain_predict", response_model=ChainPredictionResponse)
# async def forward_chain_predict(request: ChainPredictionRequest):
#     try:
#         logger.debug(f"Parameters: {request.parameters},\n model names: {request.model_names},\n images:{request.instances}")
#         async with httpx.AsyncClient(timeout=120.0) as client:
#             resp = await client.post(f"{MODEL_DEPLOY_URL}/model/predict/chain", json=request.dict())
#             resp.raise_for_status()
#             return resp.json()
#     except Exception as e:
#         logger.error(f"Middleware error: {e}")
#         return {"results": [], "error": str(e)}
    

# @app.post("/chain_predict", response_model=ChainPredictionResponse)
# async def forward_chain_predict(request: ChainPredictionRequest):
#     """
#     Forward a chain prediction request to the backend model-deploy service.
#     """
#     logger.info(f"Forwarding chain prediction: models={request.model_names}, instances={len(request.instances)}")
    
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         try:
#             resp = await client.post(
#                 f"{MODEL_DEPLOY_URL}/model/predict/chain",
#                 json=request.dict()
#             )
#             resp.raise_for_status()  # raise exception for 4xx/5xx
#             return resp.json()
        
#         except httpx.HTTPStatusError as e:
#             logger.error(f"Backend returned error: {e.response.status_code} - {e.response.text}")
#             raise HTTPException(
#                 status_code=e.response.status_code,
#                 detail=f"Backend error: {e.response.text}"
#             )
#         except Exception as e:
#             logger.error(f"Middleware error: {e}")
#             raise HTTPException(status_code=500, detail=f"Middleware error: {str(e)}")

@app.post("/chain_predict", response_model=ChainPredictionResponse)
async def forward_chain_predict(request: ChainPredictionRequest):
    """
    Forward a chain prediction request to the backend model-deploy service.
    """
    logger.info(f"Forwarding chain prediction: models={request.model_names}, instances={len(request.instances)}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{MODEL_DEPLOY_URL}/model/predict/chain",
                json=request.dict()
            )
            resp.raise_for_status()
            backend_response = resp.json()
            return backend_response  # this will validate against ChainPredictionResponse
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Backend returned error: {e.response.status_code} {e.response.text}")
            raise HTTPException(status_code=502, detail=f"Backend error: {e.response.text}")
        
        except Exception as e:
            logger.error(f"Middleware error: {e}")
            raise HTTPException(status_code=500, detail=f"Middleware error: {str(e)}")


    




@app.post("/save_prediction")
async def save_prediction(request: SavePredictionRequest):
    try:
        logger.debug(f"Hit save_prediction: {request.detections} \n {request.metadata}")
        # Decode the image
        img_bytes = base64.b64decode(request.annotated_image)

        # Build folder + file name
        now = datetime.now()
        day_str = now.strftime("%Y-%m-%d")
        ts_str = now.strftime("%H_%M_%S")
        out_dir = os.path.join("/app/images/inference", day_str)
        os.makedirs(out_dir, exist_ok=True)

        if request.ImageName == None:
            file_path = os.path.join(out_dir, f"{ts_str}_annotated.jpg")
        else:
            file_path = os.path.join(out_dir, f"{request.ImageName}_annotated.jpg")
        # Save file
        with open(file_path, "wb") as f:
            f.write(img_bytes)
        logger.info(f"Saved annotated image → {file_path}")

        # Build Mongo document
        doc = {
            "image_path": file_path,
            "detections": request.detections,
            "metadata": request.metadata or {},
            "created_at": now,
        }

        result = await db["predictions"].insert_one(doc)
        logger.info(f"Saved prediction metadata to MongoDB with id {result.inserted_id}")

        return {"status": "ok", "id": str(result.inserted_id), "path": file_path}
    except Exception as e:
        logger.error(f"Error in save_prediction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    





# # from pymongo import MongoClient
# import time

# MONGO_URI = "mongodb://mongo:27017"
# DB_NAME = "your_db_name"  # change to the database you are using
# COLLECTION_NAME = "metaData"  # or the collection where images are logged

# async def watch_mongo_for_new_images():
#     """
#     Watches MongoDB for newly inserted images and sends them to /chain_predict automatically.
#     """
#     try:
#         async with httpx.AsyncClient(timeout=300) as client:
#             response = await client.get(f"{MODEL_DEPLOY_URL}/model/load",
#             params={
#             "model_path": DEFAULT_MODEL_PATHS[0],
#             "model_name": DEFAULT_MODEL_NAMES[0]
#         })
#     except Exception as e:
#         print(f"[ERROR] Failed to load model: {e}")
#     try:
#         async with httpx.AsyncClient(timeout=300) as client:
#             response = await client.get(f"{MODEL_DEPLOY_URL}/model/load",
#             params={
#             "model_path": DEFAULT_MODEL_PATHS[1],
#             "model_name": DEFAULT_MODEL_NAMES[1]
#         })
#     except Exception as e:
#         print(f"[ERROR] Failed to load model: {e}")

#     logger.debug(f"Succesfully loaded the models: {DEFAULT_MODEL_NAMES}")


#     try:
#         with db.watch([{"$match": {"operationType": "insert"}}]) as stream:
#             print("[INFO] Watching MongoDB for new image inserts...")
#             for change in stream:
#                 doc = change["fullDocument"]
#                 file_path = doc.get("image_path")
#                 if not file_path or not os.path.exists(file_path):
#                     continue

#                 print(f"[INFO] New image detected: {file_path}")

#                 # Read image and encode
#                 with open(file_path, "rb") as f:
#                     img_bytes = f.read()
#                     img_b64 = base64.b64encode(img_bytes).decode("utf-8")

#                 payload = {
#                     "model_names": DEFAULT_MODEL_NAMES,  # or dynamically pick models
#                     "instances": [{"image": img_b64}],
#                     "parameters": {
#                         "return_annotated_image": True,
#                         "confidence": 0.25
#                     }
#                 }

#                 try:
#                     # import httpx
#                     response = httpx.post("http://pig-sorting-api:8001/chain_predict", json=payload)
#                     response.raise_for_status()
#                     print(f"[INFO] Chain prediction completed for {file_path}")
#                 except Exception as e:
#                     print(f"[ERROR] Failed to send chain prediction for {file_path}: {e}")

#     except Exception as e:
#         print(f"[ERROR] MongoDB watcher failed: {e}")


