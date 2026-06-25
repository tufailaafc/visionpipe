"""
FastAPI API for meat evaluation and image processing pipeline.

This module provides:
- Asynchronous MongoDB integration using Motor.
- Endpoints for retrieving metadata and images.
- Endpoints for triggering model training and chain predictions.
- A MongoDB watcher to automatically process newly inserted images.
- Utilities to convert images to base64, save annotated predictions, and log results.

Endpoints include:
- /api/v1/mongoData          : Fetch all metadata from MongoDB.
- /api/v1/images/            : Fetch images between given timestamps.
- /api/v1/training/trainer   : Trigger model training.
- /api/v1/training/stream_trainer : Stream training logs from GPU container.
- /chain_predict             : Forward chain prediction requests to model-deploy.
- /save_prediction           : Save annotated prediction results to MongoDB.
- /api/v1/collector/capture  : Trigger manual capture of images.
- /api/v1/collector/frame_interval : Set camera frame interval.

Dependencies:
- FastAPI
- Motor (Async MongoDB client)
- httpx (Async HTTP requests)
- PIL, OpenCV, NumPy
- pymongo, bson
"""



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
from datetime import datetime, timedelta, timezone


# Set to the log level desired, also may stop fastapi from suppressing the logs.
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
MODEL_TRAINING_URL = "http://model-training:8401/training/stream"
MODEL_DEPLOY_URL = "http://model-deploy:8601"
#DEFAULT_MODELS = ["app/models/Proper_Test/70 epochs medium run3/weights/best.pt","app/models/Proper_Test/70 epochs medium run3/weights/last.pt"]
# These can be called whatever you want.
DEFAULT_MODEL_NAMES = ["70run3Best", "70run3Last"]
# These are the relative paths inside of the model-deploy container and must be inside of the models folder.
DEFAULT_MODEL_PATHS = ["/app/models/YOLO/Proper_Test/70 epochs medium run3/weights/best.pt","/app/models/YOLO/Proper_Test/70 epochs medium run3/weights/last.pt"]


# For automatic processing of the images saved in mongo
CATCHUP_WINDOW_MINUTES = 5
STATE_COLLECTION = "watcher_state"
STATE_ID = "mongo_poller"  # unique key for this watcher




class CaptureRequest(BaseModel):
    """
    Request model for triggering manual data captures.

    Attributes:
        channels (list[int]): List of channel IDs to capture from.
        comment (str | None): Optional global comment applied to all
            captured channels.
        comments (dict[int, str] | None): Optional per-channel comments
            keyed by channel ID.
    """
    channels: list[int]
    comment: str | None = None
    comments: dict[int, str] | None = None



# for updating how often the cameras take pictures, 
# interval is how many frames between taking pictures and channel is the camera channel it is connected to.
class FrameIntervalRequest(BaseModel):
    """
    Request model for updating camera frame capture intervals.

    Attributes:
        channel (int): Camera channel identifier.
        interval (int): Number of frames between image captures.
    """
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


class PredictionInstance(BaseModel):
    image: str

    depth_image: Optional[str] = None

    depth_format: Optional[str] = None


class ChainPredictionRequest(BaseModel):
    """
    Request model for chained model predictions.

    Attributes:
        model_names (List[str]): Ordered list of model identifiers to
            execute as a prediction chain.
        instances (List[Dict[str, str]]): List of input instances to be
            passed through the model chain. Each dictionary represents
            a single instance and must conform to the backend model
            input schema which expects the List to contain base64 encoded images.
        parameters (Optional[Dict[str, Any]]): Optional configuration
            parameters forwarded to the backend prediction service.
    """
    model_names: List[str]

    instances: List[PredictionInstance]

    parameters: Optional[Dict[str, Any]] = None


# Define request model
class SavePredictionRequest(BaseModel):
    """
    Request model for saving annotated prediction results.

    Attributes:
        annotated_image (str): Base64-encoded image containing model
            annotations (e.g., bounding boxes, labels).
        detections (List[Dict[str, Any]]): List of detection results such
            as bounding boxes, class labels, and confidence scores.
        metadata (Optional[Dict[str, Any]]): Optional additional metadata
            associated with the prediction.
        ImageName (Optional[str]): Optional base name for the saved image
            file. If not provided, a timestamp-based name is used.
    """
    annotated_image: str                  # base64-encoded image
    detections: List[Dict[str, Any]]      # bounding boxes, confidences, etc.
    metadata: Optional[Dict[str, Any]] = None
    ImageName: Optional[str] = None

class ChannelsRequest(BaseModel):
    """Request model for retrieving active and connected camera ids.

    Attributes:
        username (str): The username to log into the NVR.
        password (str): The password to log into the NVR.
        nvr_ip (str): The ip to log into the NVR.
    """
    username: Optional[str] = None
    password: Optional[str] = None
    nvr_ip: Optional[str] = None


class SwineDataSave(BaseModel):
    """
    Request model for saving data produced from the swine unit.

    Attributes:
        pen (str): A string containing the idetifier for the pen.
        rfid (str): A string containing the rfid of the pig that triggered the capture.
        comment (Optional(str)): A string holding any comments for this capture.
        depth_images (list(str)): A list of depth images encoded in base64.
        depth_image_names (list(str)): A list of depth image names that correspond with depth_images.
        thermal_images (list(str)): A list of thermal images encoded in base64.
        thermal_image_names (list(str)): A list of thermal image names that correspond with thermal_images.
        rgb_images (list(str)): A list of rgb images encoded in base64.
        rgb_image_names (list(str)): A list of rgb image names that correspond with rgb_images.         
        sensor (Optional(Dict[str, Any])): Various sensor data.
        timestamp (datetime): A timestamp for when the capture happened. 
            The timezone is assumed to be utc

    """
    pen: str
    rfid: str
    comment: Optional[str] = None
    depth_images: List[str]
    depth_image_names: List[str]
    thermal_images: List[str]
    thermal_image_names: List[str]
    rgb_images: List[str]
    rgb_image_names: List[str]
    sensor: Optional[Dict[str,Any]]
    timestamp: str




CHAIN_PREDICTION_URL = f"{MODEL_DEPLOY_URL}/model/predict/chain"

async def load_default_models():
    """
    Load default models into the model-deploy service.

    This is typically called once at startup to ensure models are available for predictions.

    Returns:
        None
    """
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

# TODO UPDATE THIS FUNCTION TO ALSO HANDLE DEPTH IMAGES FROM THE MONGODB
async def send_to_chain_prediction(image_path: str):
    """
    Send an image to the chain prediction endpoint and save the annotated results.

    This function:
    - Reads the image asynchronously.
    - Encodes it as base64.
    - Sends it to the model-deploy chain prediction endpoint.
    - Flattens model detection results and attaches model names.
    - Saves the annotated image to disk.
    - Stores prediction metadata in MongoDB.

    Args:
        image_path (str): Absolute path to the input image file.

    Returns:
        None
    """
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
            logger.info(f"Prediction saved for {image_path}")

    except Exception as e:
        logger.error(f"Failed to send chain prediction for {image_path}: {e}")



async def get_last_checkpoint():
    """
    Load the last checkpoint timestamp from MongoDB.

    If no checkpoint is found, returns the current time minus the catch-up window.

    Returns:
        datetime: The timestamp of the last processed document.
    """
    state = await db[STATE_COLLECTION].find_one({"_id": STATE_ID})
    if state and "last_check" in state:
        ts = state["last_check"]
        logger.debug(f"Resuming from saved checkpoint: {ts}")
        return ts
    else:
        ts = datetime.utcnow() - timedelta(minutes=CATCHUP_WINDOW_MINUTES)
        logger.debug(f"No saved checkpoint, using catch-up: {ts}")
        return ts


async def save_checkpoint(ts: datetime):
    """
    Save the last processed timestamp to MongoDB.

    Args:
        ts (datetime): Timestamp to save as the last processed checkpoint.

    Returns:
        None
    """
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
            logger.debug("Trying to get metaData")
            cursor = db["metaData"].find({
                "date_time": {"$gt": last_check}
            }).sort("date_time", 1)  # sort ascending to move checkpoint correctly
            logger.debug("Created cursor")
            docs = await cursor.to_list(length=100)
            logger.debug("Got the documents")
            if docs:
                logger.debug(f"Found {len(docs)} new docs since {last_check}")

            for doc in docs:
                image_path = doc.get("image_path")
                if image_path:
                    logger.debug(f"New image detected: {image_path}")
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
    """
    Convert an OpenCV image (NumPy array) to a base64-encoded PNG string.

    Args:
        img (np.ndarray): Input image. Can be grayscale or RGB.

    Returns:
        str: Base64-encoded string of the PNG image.
    """
    # Ensure image is in BGR format for cv2.imencode as PIL loads in RGB
    if len(img.shape) == 3 and img.shape[2] == 3: # Check if it's a 3-channel image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img # Assume it's already in a suitable format or grayscale

    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()



# Will send all records to the user
@app.get("/api/v1/mongoData", response_class=JSONResponse)
async def get_mongo_data() -> list[dict]:
    """
    Retrieve all metadata records from the MongoDB collection.

    This endpoint queries the ``metaData`` collection in MongoDB,
    converts the result cursor into JSON-serializable Python objects,
    and returns all records to the client.

    Returns:
        list[dict]: A list of metadata records retrieved from the
        ``metaData`` MongoDB collection.

    Raises:
        HTTPException: If an unexpected error occurs while accessing
        the database or serializing the results.
    """
    try:
        logger.debug("/api/v1/mongoData/ endpoint hit")
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
async def get_images_by_date(request: TimesRequest) -> list[dict]:
    """
    Retrieve all images captured within a specified time range.

    This endpoint accepts a start and end timestamp, queries the
    ``metaData`` MongoDB collection for records within that range,
    loads the corresponding image files from disk, and returns them
    as base64-encoded strings along with metadata.

    Args:
        request (TimesRequest): Request body containing a list of two
            ISO 8601 formatted timestamps representing the start and
            end of the desired time range.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - ``filename`` (str): Name of the image file.
            - ``image`` (str): Base64-encoded image data.
            - ``timestamp`` (str): ISO 8601 formatted capture time.

    Raises:
        HTTPException: If an unexpected error occurs while querying the
        database, reading image files, or encoding the images.
    """
    logger.debug("/api/v1/images/ endpoint hit")
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



@app.get("/api/v1/training/trainer", response_class=JSONResponse)
async def run_training(
    model_path: str = Query(...),
    dataset_path: str = Query(...),
    project_name: str = Query("default_project"),
    run_name: str = Query("run_001")
):
    """
    Start a YOLO training job with the specified parameters.

    This endpoint forwards a training request to the GPU-enabled
    model-training service, waits for training to complete, and logs
    the results to the MongoDB ``training_logs`` collection.

    The following fields are logged to MongoDB:
        - ``model_path`` (str): Path to the pretrained model used.
        - ``dataset_path`` (str): Location of the training dataset.
        - ``model_export_path`` (str): Path to the newly trained model.
        - ``project`` (str): Name of the project.
        - ``run`` (str): Name of the training run.
        - ``train_results`` (str): Training metrics and results.
        - ``val_results`` (str): Validation metrics and results.
        - ``date_time`` (datetime): Timestamp when training completed.

    Args:
        model_path (str): Path to the pretrained YOLO model.
        dataset_path (str): Path to the dataset used for training.
        project_name (str): Logical name for the training project.
        run_name (str): Identifier for this specific training run.

    Returns:
        dict: A dictionary containing:
            - ``message`` (str): Confirmation that training completed.
            - ``model_export_path`` (str): Path to the trained model.
            - ``train_results`` (str): Training results.
            - ``val_results`` (str): Validation results.

    Raises:
        HTTPException: If the training service fails, times out,
        or logging to the database is unsuccessful.
    """
    try:
        logger.debug("Training request received")

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
    """
    Stream a YOLO training job and log results upon completion.

    This endpoint forwards a training request to the GPU-enabled
    model-training service and streams training progress to the client
    using Server-Sent Events (SSE). Once training completes, the final
    results are logged to the MongoDB ``training_logs`` collection.

    The following fields are logged to MongoDB:
        - ``model_path`` (str): Path to the pretrained model used.
        - ``dataset_path`` (str): Location of the training dataset.
        - ``model_export_path`` (str): Path to the newly trained model.
        - ``project`` (str): Name of the project.
        - ``run`` (str): Name of the training run.
        - ``train_results`` (dict | str): Training metrics and results.
        - ``val_results`` (dict | str): Validation metrics and results.
        - ``date_time`` (datetime): Timestamp when training completed.

    Args:
        model_path (str): Path to the pretrained YOLO model.
        dataset_path (str): Path to the dataset used for training.
        project_name (str): Logical name for the training project.
        run_name (str): Identifier for this specific training run.
        epochs (int): Number of training epochs to run.

    Returns:
        StreamingResponse: A Server-Sent Events (SSE) stream emitting
        JSON-encoded training progress updates and a final completion
        event.

    Raises:
        HTTPException: If the training service fails, the stream cannot
        be established, or logging to the database is unsuccessful.
    """
    try:
        logger.debug("Streaming training request received")

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
    """
    Recursively convert MongoDB ObjectId values to strings.

    This function traverses nested dictionaries and lists, replacing
    any instances of ``bson.ObjectId`` with their string representation.
    All other data types are returned unchanged.

    Args:
        data (Any): Input data that may contain dictionaries, lists,
            or ``ObjectId`` values.

    Returns:
        Any: The input data with all ``ObjectId`` instances converted
        to strings.
    """
    if isinstance(data, dict):
        return {key: convert_objectid_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)  # Convert ObjectId to string
    else:
        return data
        
        

@app.post("/api/v1/collector/capture")
async def capture(request: CaptureRequest):
    """
    Proxy a manual capture request to the data-collector service.

    This endpoint forwards a request to the data-collector to trigger
    one or more manual captures on the specified channels. Optional
    comments may be included globally or per channel.

    Args:
        request (CaptureRequest): Capture configuration including
            channel identifiers and optional comments.

    Returns:
        dict: JSON response returned by the data-collector service.

    Raises:
        HTTPException: If the data-collector service is unreachable
        or returns an unexpected error.
    """
    try:
        logger.debug(f"Capture channels: {request.channels}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATA_INTAKE_URL}/capture/",
                json={"channels": request.channels,
                      "comment": request.comment,
                      "comments": request.comments}
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error sending capture request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v1/collector/frame_interval")
async def set_frame_interval(request: FrameIntervalRequest):
    """
    Update the frame capture interval for a specific camera.

    This endpoint proxies a request to the data-intake service to
    configure how frequently a camera captures images, based on
    a frame interval setting.

    Args:
        request (FrameIntervalRequest): Configuration containing the
            camera channel and desired frame interval.

    Returns:
        dict: JSON response returned by the data-intake service.

    Raises:
        HTTPException: If the data-intake service is unreachable
        or returns an unexpected error.
    """
    try:
        logger.debug(f"Trying to connect to: {DATA_INTAKE_URL}/frame_interval")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATA_INTAKE_URL}/frame_interval",
                json={"channel": request.channel, "interval": request.interval}
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error sending frame interval request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    



@app.post("/chain_predict", response_model=ChainPredictionResponse)
async def forward_chain_predict(request: ChainPredictionRequest):
    """
    Forward a chained prediction request to the model-deploy service.

    This endpoint acts as a middleware layer that validates and forwards
    a chained prediction request to the backend model-deploy service,
    then returns the validated response to the client.

    Args:
        request (ChainPredictionRequest): Chained prediction request
            containing model sequence, input instances, and optional
            parameters.

    Returns:
        ChainPredictionResponse: The prediction results returned by the
        backend service after validation.

    Raises:
        HTTPException: 
            - 502 Bad Gateway: If the backend model-deploy service
              returns an error response.
            - 500 Internal Server Error: If an unexpected middleware
              error occurs.
    """
    logger.debug(f"Forwarding chain prediction: models={request.model_names}, instances={len(request.instances)}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{MODEL_DEPLOY_URL}/model/predict/chain",
                json=request.model_dump()
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
    """
    Persist annotated prediction output to disk and MongoDB.

    This endpoint decodes a base64-encoded annotated image, saves it to
    disk using a timestamped or user-provided name, and stores the
    associated prediction metadata in the MongoDB ``predictions``
    collection.

    Args:
        request (SavePredictionRequest): Annotated image data, detection
            results, and optional metadata.

    Returns:
        dict: A dictionary containing:
            - ``status`` (str): Operation status.
            - ``id`` (str): MongoDB ObjectId of the saved prediction.
            - ``path`` (str): Filesystem path to the saved annotated image.

    Raises:
        HTTPException: If image decoding, file I/O, or database insertion
        fails.
    """
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
        logger.debug(f"Saved prediction metadata to MongoDB with id {result.inserted_id}")

        return {"status": "ok", "id": str(result.inserted_id), "path": file_path}
    except Exception as e:
        logger.error(f"Error in save_prediction: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    



@app.post("/api/v1/collector/channels")
async def channels(request: ChannelsRequest):
    """
    Retrieve the list of active and connected camera channels from a network video recorder (NVR).

    This endpoint proxies a request to the data-intake service, which connects
    to the specified NVR and returns a list of currently active and connected
    camera channels. Optional authentication credentials can be provided if
    required by the NVR.

    Args:
        request (ChannelsRequest): NVR connection details and optional credentials.

    Returns:
        dict: JSON response from the data-intake service containing camera
        channel information.

    Raises:
        HTTPException: If the data-intake service is unreachable or returns an error.
    """
    logger.debug(f"Reached pig-sorting-api channel request")
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{DATA_INTAKE_URL}/channels",
                    json={"username": request.username,
                        "password": request.password,
                        "nvr_ip": request.nvr_ip})
        
        logger.debug(f"Returning values inside of pig-sorting-api: {response.json()}")
        return response.json()


@app.post("/api/v1/collector/swine_data_save")
async def save_swine_data(request: SwineDataSave):
    """
    Persist images and pen sensor readings output to disk and MongoDB.

    This endpoint decodes base64-encoded annotated images, saves it to
    disk using a timestamped or user-provided name, and stores the
    associated prediction metadata such as sensor readings and predicted weight in the MongoDB ``predictions``
    collection.

    Args:
        request(SwineDataSave): 
            - pen (str): A string containing the idetifier for the pen.
            - rfid (str): A string containing the rfid of the pig that triggered the capture.
            - comment (Optional(str)): A string holding any comments for this capture.
            - depth_images (list(str)): A list of depth images encoded in base64.
            - depth_image_names (list(str)): A list of depth image names that correspond with depth_images.
            - thermal_images (list(str)): A list of thermal images encoded in base64.
            - thermal_image_names (list(str)): A list of thermal image names that correspond with thermal_images.
            - rgb_images (list(str)): A list of rgb images encoded in base64.
            - rgb_image_names (list(str)): A list of rgb image names that correspond with rgb_images.         
            - sensor (Dict[str, Any]): Various sensor data.
            - timestamp (datetime): A timestamp for when the capture happened. The timezone is assumed to be utc

    Returns:
        dict: A dictionary containing:
            - ``status`` (str): Operation status.
            - ``id`` (str): MongoDB ObjectId of the saved prediction.
            - ``path`` (str): Filesystem path to the saved annotated image.

    Raises:
        HTTPException: If image decoding, file I/O, or database insertion
        fails.
    """
    try:
        logger.debug(f"Hit swine_data_save: {request.rfid} \n {request.pen} {request.timestamp}")
        ##################################### Decode the images and save them to the file system while keeping track of location and meta data.
        depth_image_paths:list[str] = []
        thermal_image_paths:list[str] = []
        rgb_image_paths:list[str] = []

        # Build folder to store the images
        now = datetime.now(timezone.utc)
        day_str = now.strftime("%Y-%m-%d")
        out_dir = os.path.join("/app/images/inference/swine", day_str)
        os.makedirs(out_dir, exist_ok=True)


        def save_base64_image_with_name(name, bytes, out_dir):
            #decode and save each file to the created directory
            img_bytes = base64.b64decode(bytes)

            #specify the path/name
            file_path = os.path.join(out_dir, f"{name}")
            # Save file
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            logger.info(f"Saved annotated image → {file_path}")
            return file_path

        for i in range(len(request.depth_images)):
            file_path = save_base64_image_with_name(request.depth_image_names[i], request.depth_images[i], out_dir)
            depth_image_paths.append(file_path)

        for i in range(len(request.thermal_images)):
            file_path = save_base64_image_with_name(request.thermal_image_names[i], request.thermal_images[i], out_dir)
            thermal_image_paths.append(file_path)


        for i in range(len(request.rgb_images)):
            file_path = save_base64_image_with_name(request.rgb_image_names[i], request.rgb_images[i], out_dir)
            rgb_image_paths.append(file_path)

        #hook up to model for weight prediction
        #dummy for some data replace with actual weight prediction
        import random
        predicted_weight:float = random.randint(70,120)

        #transform the timestamp into a datetime object for effecient sorting and lookup
        timestamp_object = datetime.strptime(request.timestamp,"%Y%m%d_%H%M%S_%f")


        # Build Mongo document
        doc = {
            "pen": request.pen,
            "rfid": request.rfid,
            "comment": request.comment,
            "depth_images": depth_image_paths,
            "thermal_images": thermal_image_paths,
            "rgb_images": rgb_image_paths,
            "sensor": request.sensor,
            "timestamp": timestamp_object,
            "predicted_weight": predicted_weight
        }

        #insert mongo document into the database
        result = await db["swinedata"].insert_one(doc)
        logger.debug(f"Saved prediction metadata to MongoDB with id {result.inserted_id}")

        return {"status": "ok", "id": str(result.inserted_id), "paths": f"{depth_image_paths,thermal_image_paths,rgb_image_paths}"}
    except Exception as e:
        logger.error(f"Error in swine_data_save: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    


# Will send all records to the user
@app.get("/api/v1/mongoData/swinedata", response_class=JSONResponse)
async def get_swinedata():
    """
    Retrieve all swinedata records from the MongoDB collection.

    This endpoint queries the ``swinedata`` collection in MongoDB,
    converts the result cursor into JSON-serializable Python objects,
    and returns all records to the client.

    Returns:
        list[dict]: A list of swinedata records retrieved from the
        ``swinedata`` MongoDB collection.

    Raises:
        HTTPException: If an unexpected error occurs while accessing
        the database or serializing the results.
    """
    try:
        cursor = db["swinedata"].find()
        print(f" cursor object {cursor}")
        docs = await cursor.to_list(length=5000)
        print(f"Docs {docs}")

        clean = json.loads(bson.json_util.dumps(docs))
        print(f"Cleaned docs {clean}")
        return clean

    except Exception as e:
        logger.error(f"Error getting swinedata: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})