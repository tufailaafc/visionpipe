"""
Pig Sorting YOLO Training API

This module provides FastAPI endpoints and helper functions for training and managing Ultralytics YOLO models.

Key Features:
- Train YOLO models using a provided dataset (.yaml or .zip format) with configurable hyperparameters.
- Stream training progress in real-time via Server-Sent Events (SSE).
- Export trained models to ONNX format and validate on the dataset.
- Dynamically handle datasets: unzip archives, locate .yaml configuration files.
- GPU support with logging of device and CUDA availability.
- Threaded training to prevent blocking the API server.
- Utility endpoints for retrieving available datasets and projects.

Endpoints:
- /training/trainer: Synchronous training, returns trained model path and validation metrics.
- /training/stream: Asynchronous training with real-time progress updates.
- /training/datasets: List all available datasets and projects.

Dependencies:
- ultralytics, torch, fastapi, logging, asyncio, threading, queue, zipfile, os, json
"""

import ultralytics
import logging

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os, json

import torch

from ultralytics.engine.trainer import BaseTrainer

import asyncio
import threading

from queue import Queue

import zipfile

progress_queues = {}



logger = logging.getLogger()


# Set to the log level desired, also may stop fastapi from suppressing the logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Verifying access to the gpu and the version of ultralytics
# logger.info("CUDA available:", torch.cuda.is_available())
# logger.info("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

logger.info(f"Ultralytics version: {ultralytics.__version__}")

logger.info(f"Ultralytics settings: {ultralytics.settings}")



# Setting up the app so that pig-sorting-api can talk to the trainer
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Main training function, will take the path to the training model ie yolo11n.pt and a path to the dataset 
# specifically the .yaml file in yolo11 format.
# We optionally take the project name and run name
# Returns the trained model in onnx formet as well as the validation results.
def train_model(model_path: str, 
                dataset_path: str, 
                project_name: str = "", 
                run_name: str = "", 
                epochs:int=50,
                batch:int=16,
                lr0: float = 0.002,
                lrf: float=0.01, 
                augment:bool = False):
    """
    Train a YOLO model synchronously.

    Args:
        model_path (str): Path to a pretrained YOLO model (.pt).
        dataset_path (str): Path to YOLO-formatted dataset (.yaml).
        project_name (str, optional): Directory name to save outputs.
        run_name (str, optional): Name of the training run.
        epochs (int, optional): Number of training epochs.
        batch (int, optional): Batch size.
        lr0 (float, optional): Initial learning rate.
        lrf (float, optional): Final learning rate factor.
        augment (bool, optional): Apply data augmentation.

    Returns:
        Tuple[str, dict, dict]: Path to best checkpoint, training results, validation results.
    """

    # TODO add training parameters to the output for logging


    # Making sure the project direcotry exists to store the exported model
    output_dir = f"/app/models/{project_name}" if project_name else "/app/models/default_project"
    os.makedirs(output_dir, exist_ok=True)

    # Load a pretrained YOLO model (recommended for training), will download one if set to someting like yolo11s.pt
    model = ultralytics.YOLO(model_path)


    # Train the model using the supplied dataset for n epochs
    train_results = model.train(data=dataset_path, 
                                epochs=epochs,
                                batch=batch,
                                lr0=lr0,
                                lrf=lrf,
                                augment=augment, 
                                save=True, 
                                project=output_dir, 
                                name=run_name,
                                patience=20,)

    train_results = train_results.to_json()

    # # Evaluate the model's performance on the validation set
    # val_results = model.val(verbose=True)

    # val_results = val_results.to_json()

    # # Perform object detection on an image using the model
    # # results = model("https://ultralytics.com/images/bus.jpg")


    

    # # Export the model to ONNX format
    # # Dynamic being true helps with handling images of different sizes.
    # export_model = model.export(format="onnx", dynamic=True)

    # return output_dir + run_name, train_results, val_results
    # --- VALIDATE best checkpoint ---
    best_model_path = f"{output_dir}/{run_name}/weights/best.pt"
    best_model = ultralytics.YOLO(best_model_path)
    val_results = best_model.val(verbose=True)
    val_results = val_results.to_json()

    # --- EXPORT best checkpoint to ONNX ---
    export_model = best_model.export(format="onnx", dynamic=True)

    return best_model_path, train_results, val_results



# Training function with realtime feedback.
# Main training function, will take the path to the training model ie yolo11n.pt and a path to the dataset 
# specifically the .yaml file in yolo11 format.
# We optionally take the project name and run name
# Returns the trained model in onnx formet as well as the validation results.
def train_model_stream(model_path: str, dataset_path: str,queue: Queue, project_name: str = "", run_name: str = "", epochs:int=1):
    """
    Train a YOLO model asynchronously with real-time progress updates.

    Args:
        model_path (str): Path to pretrained YOLO model (.pt).
        dataset_path (str): Path to dataset (.yaml or .zip).
        queue (Queue): Queue to stream progress messages to the caller.
        project_name (str, optional): Directory to save outputs.
        run_name (str, optional): Name of the training run.
        epochs (int, optional): Number of epochs.

    Returns:
        None

    Notes:
        - This function is typically run in a separate thread to prevent blocking.
        - Real-time epoch progress is pushed into `queue` via the `on_epoch_end` callback.
        - Once training is complete, a final message with status 'done' is pushed into the queue.
    """
    # Create a new YOLO model from scratch
    # model = ultralytics.YOLO("yolo11n.yaml")

    # Making sure the project directory exists to store the exported model
    output_dir = f"/app/models/{project_name}" if project_name else "/app/models/default_project"
    os.makedirs(output_dir, exist_ok=True)

    # Load a pretrained YOLO model (recommended for training)
    model = ultralytics.YOLO(model_path)


    # Will output data for every epoch
    def on_epoch_end(trainer):
        """
        Callback function executed at the end of each training epoch.

        This function is attached to the YOLO model during training and pushes progress
        data to a Queue, which can then be streamed to a frontend in real-time.

        Args:
            trainer (BaseTrainer): Ultralytics YOLO trainer object for the current epoch.
            queue (Queue): Queue object to push progress updates.

        Returns:
            None
        """
        epoch = trainer.epoch + 1
        total = trainer.epochs
        metrics = trainer.metrics
        data = {
            "phase": "train",
            "epoch": epoch,
            "total_epochs": total,
            "metrics": metrics
        }
        queue.put(data)


    # This will attach our function so that we will have access to the progress
    model.add_callback("on_train_epoch_end", on_epoch_end)
    

    # We make sure that the file is unzipped and return the path to the .yaml file if it exists.
    dataset_path = prepare_dataset(dataset_path)


    # Train the model using the supplied dataset for n epochs
    train_results = model.train(data=dataset_path, epochs=epochs, save=True, project=output_dir, name=run_name)

    train_results = train_results.to_json() # The models performance on the training set.


    # Evaluate the model's performance on the validation set
    val_results = model.val(verbose=True)

    val_results = val_results.to_json()


    # Perform object detection on an image using the model
    # results = model("https://ultralytics.com/images/bus.jpg")


    

    # Export the model to ONNX format
    # Dynamic being true helps with handling images of different sizes.
    #export_model = model.export(format="onnx", dynamic=True)
    export_model = model.export(format="onnx")

    
    queue.put({
        "status": "done", # will stop the connnection
        "path": os.path.join(output_dir, run_name),
        "train_results": train_results,
        "val_results": val_results
    })




# Not responsive legacy code
@app.get("/training/trainer", response_class=JSONResponse)
async def get_mongo_data(model_path:str, dataset_path:str, project_name:str, run_name:str, epochs:int=1):
    """
    Endpoint for synchronous YOLO model training.

    Args:
        model_path (str): Path to pretrained YOLO model.
        dataset_path (str): Path to dataset.
        project_name (str): Output project directory.
        run_name (str): Training run name.
        epochs (int, optional): Number of epochs.

    Returns:
        dict: Trained model path, training results, validation results.
    """
    try:
        logger.debug("/training/trainer endpoint hit")
        # model, train_results, val_results = train_model("yolo11n.pt", "coco8.yaml")
        model, train_results, val_results = train_model(model_path, dataset_path, project_name, run_name, epochs)

        # Temporary code to debug the data flow.
        if train_results != None:
            train_results = True
        else:
            train_results = False

        return {
            "train_results": train_results,
            "val_results": val_results,
            "model_export_path": str(model)
            }
            

    except Exception as e:
        logger.error(f"Error in training endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    



# Endpoint that will stream back the progress dynamically.
@app.get("/training/stream")
async def stream_training(model_path: str, dataset_path: str,
                          project_name: str = "default", run_name: str = "run001", epochs: int=1):
    """
    Endpoint for asynchronous YOLO training with real-time progress via SSE.

    Args:
        model_path (str): Path to pretrained YOLO model.
        dataset_path (str): Path to dataset.
        project_name (str, optional): Output project directory.
        run_name (str, optional): Training run name.
        epochs (int, optional): Number of epochs.

    Returns:
        StreamingResponse: Progress messages as Server-Sent Events.
    """
    # Create a new queue for this session
    q = Queue()
    progress_queues[run_name] = q

    # Run training in separate thread to avoid blocking
    threading.Thread(target=train_model_stream, args=(model_path, dataset_path, q, project_name, run_name, epochs), daemon=True).start()

    # Generate the messages to pass back to the frontend server
    async def event_gen():
        while True:
            data = q.get()
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") == "done":
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")





# Gets all available datasets and projects, returns a path to the zip folder
@app.get("/training/datasets")
def get_data_sets():
    """
    Retrieve all available datasets and projects.

    Returns:
        dict: Paths to dataset files.
    """
    try:
        root_path = "processed"
        projects = dict()

        for project, _, files in os.walk(root_path):
            for file in files:  
                projects["full_path"]=f"{project}/{file}"

        return {projects["full_path"]}   
           
    except Exception as e:
        logger.error(f"Error in datasets endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    




# Will look for the .yaml file inside supplied directory and can handle zip folders
# Returns a path to the .yaml or .yml file 
def prepare_dataset(dataset: str) -> str:
    """
    Prepare dataset for training by unzipping and locating YAML file.

    Args:
        dataset (str): Path to .yaml, .yml, or .zip dataset.

    Returns:
        str: Path to dataset YAML file.
    """
    try:
        if dataset.endswith(".zip"):
            dataset = unzip_file(dataset, ".")
        dataset = find_yaml_file(dataset)
        logger.info(f"The path to the .yaml file: {dataset}")
        return dataset
    except (FileNotFoundError, zipfile.BadZipFile) as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    

# Unzips the supplied folder to the specified directory
# Returns the path to the extracted folder
def unzip_file(zip_path: str, extract_to: str = "/tmp/unzipped") -> str:
    """
    Extract ZIP archive to a directory.

    Args:
        zip_path (str): Path to ZIP archive.
        extract_to (str, optional): Extraction directory.

    Returns:
        str: Path to extracted folder.
    """
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


# Searches for the first .yaml/.yml file in a directory
# Returns full path to the .yaml/.yml file.
def find_yaml_file(directory: str) -> str:
    """
    Locate the first YAML (.yaml or .yml) file in a directory.

    Args:
        directory (str): Directory to search.

    Returns:
        str: Path to YAML file.

    Raises:
        FileNotFoundError: If no YAML file is found.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                return os.path.join(root, file)

    raise FileNotFoundError("No .yaml file found in the directory.")