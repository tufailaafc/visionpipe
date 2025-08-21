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
logger.info("CUDA available:", torch.cuda.is_available())
logger.info("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

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

    # Create a new YOLO model from scratch
    # model = ultralytics.YOLO("yolo11n.yaml")

    # Making sure the project directory exists to store the exported model
    output_dir = f"/app/models/{project_name}" if project_name else "/app/models/default_project"
    os.makedirs(output_dir, exist_ok=True)

    # Load a pretrained YOLO model (recommended for training)
    model = ultralytics.YOLO(model_path)


    # Will output data for every epoch
    def on_epoch_end(trainer):
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
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return extract_to


# Searches for the first .yaml/.yml file in a directory
# Returns full path to the .yaml/.yml file.
def find_yaml_file(directory: str) -> str:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                return os.path.join(root, file)

    raise FileNotFoundError("No .yaml file found in the directory.")