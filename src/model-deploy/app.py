import PIL.Image
from ultralytics import YOLO
import logging
import PIL
import io
import numpy
from typing import Dict, Any

import os
import glob

# Helper functions for main.py to load and deploy models


logger = logging.getLogger()


# Set to the log level desired, also may stop fastapi from suppressing the logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# this is the folder where we look for our models, this is relative to the inside of the docker container
MODEL_ROOT = "/app/models"



# Model initialization and readiness state
model_yolo = None
_model_ready = False
models = {} 


def _initialize_model(model:str="yolo11n.pt"):
    global model_yolo, _model_ready

    try:
        # Load selected model
        model_yolo = YOLO(model)
        _model_ready = True

    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")


# Initialize model on import
_initialize_model()


# adds the model to the list so that we can use it for inference
def load_model(model_name: str, model_path: str):
    try:
        model = YOLO(model_path)
        models[model_name] = model
        return True
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return False


# Checks if the model is ready
# def is_model_ready() -> bool:
#     return _model_ready and model_yolo is not None
def is_model_ready(model_name: str) -> bool:
    return model_name in models



# Convert image from bytes to PIL RGB format
def get_image_from_bytes(binary_image: bytes) -> PIL.Image.Image:
    input_image = PIL.Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


# Convert PIL image to bytes
def get_bytes_from_image(image: PIL.Image.Image) -> bytes:
    return_image = io.BytesIO()
    image.save(return_image, format="JPEG", quality=85)
    return_image.seek(0)
    return return_image.getvalue()



# Perform inference on the given image using the selected model
def run_inference(input_image: PIL.Image.Image, model_name: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    # Select the model to perform inference with
    model = models.get(model_name)

    # ensure the model is loaded
    if not model:
        logger.warning(f"Model '{model_name}' is not loaded.")
        return {"detections": [], "results": None}

    try:

        # perform inference using the selected model
        results = model.predict(imgsz=640, source=input_image, conf=confidence_threshold, save=False, augment=False, verbose=False)

        # process the detections from the inference
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    detection = {
                        "xmin": float(xyxy[i][0]),
                        "ymin": float(xyxy[i][1]),
                        "xmax": float(xyxy[i][2]),
                        "ymax": float(xyxy[i][3]),
                        "confidence": float(conf[i]),
                        "class": int(cls[i]),
                        "name": model.names.get(cls[i], f"class_{cls[i]}"),
                    }
                    detections.append(detection)

        return {
            "detections": detections,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in YOLO detection: {e}")
        return {"detections": [], "results": None}








def get_annotated_image(results: list) -> PIL.Image.Image:
    """
    Get annotated image using Ultralytics built-in plot method.
    """
    if not results or len(results) == 0:
        raise ValueError("No results provided for annotation")
    
    result = results[0]
    return result.plot(pil=True)




def discover_models() -> Dict[str, str]:
    """
    Recursively find all YOLO .pt model files inside the models directory.
    Returns a dict: model_name -> full_path
    """
    model_paths = glob.glob(os.path.join(MODEL_ROOT, "**", "weights", "*.pt"), recursive=True)
    discovered = {}

    for path in model_paths:
        # Use relative path from MODEL_ROOT as model name
        model_name = os.path.relpath(path, MODEL_ROOT).replace("/", "_").replace(".pt", "")
        discovered[model_name] = path

    return discovered


def load_all_discovered_models():
    models_found = discover_models()
    for name, path in models_found.items():
        try:
            model = YOLO(path)
            models[name] = model
            logger.info(f"Loaded model: {name} from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model {name} at {path}: {e}")
