import PIL.Image
from ultralytics import YOLO
import logging
import PIL
import io
import numpy
from typing import Dict, Any

import os
import glob
import random

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
    """Initialize a YOLO model for inference.

    Loads the specified YOLO model and updates the global
    readiness state. Intended for internal use.

    Args:
        model (str, optional): Path or name of the YOLO model.
            Defaults to "yolo11n.pt".

    Returns:
        None

    Raises:
        Exception: If the model fails to load.
    """
    global model_yolo, _model_ready

    try:
        # Load selected model
        model_yolo = YOLO(model)
        _model_ready = True

    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")


# Initialize model on import
# _initialize_model()


# adds the model to the list so that we can use it for inference
def load_model(model_name: str, model_path: str):
    """Load a YOLO model and register it by name.

    Args:
        model_name (str): Name used to reference the model.
        model_path (str): Full path to the YOLO model file (.pt).

    Returns:
        bool: True if the model loaded successfully, False otherwise.
    """
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
    """Check if a YOLO model is loaded and ready.

    Args:
        model_name (str): The model name to check.

    Returns:
        bool: True if the model is loaded, False otherwise.
    """
    return model_name in models



# Convert image from bytes to PIL RGB format
def get_image_from_bytes(binary_image: bytes) -> PIL.Image.Image:
    """Convert raw bytes into a PIL RGB image.

    Args:
        binary_image (bytes): Image data in bytes.

    Returns:
        PIL.Image.Image: Image object in RGB mode.
    """
    input_image = PIL.Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


# Convert PIL image to bytes
def get_bytes_from_image(image: PIL.Image.Image) -> bytes:
    """Convert a PIL image into JPEG-encoded bytes.

    Args:
        image (PIL.Image.Image): The PIL image to encode.

    Returns:
        bytes: JPEG-encoded image data.
    """
    return_image = io.BytesIO()
    image.save(return_image, format="JPEG", quality=85)
    return_image.seek(0)
    return return_image.getvalue()



# Perform inference on the given image using the selected model
def run_inference(input_image: PIL.Image.Image, model_name: str, 
                  confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Perform inference on an image using a YOLO model.

    Args:
        input_image (PIL.Image.Image): Input image for detection.
        model_name (str): The registered model name to use.
        confidence_threshold (float, optional): Minimum confidence score
            for detections. Defaults to 0.5.

    Returns:
        dict: Dictionary containing:
            - detections (list[dict]): Bounding box detections with fields
              xmin, ymin, xmax, ymax, confidence, class, and name.
            - results (ultralytics.engine.results.Results | None):
              Raw YOLO results object.
    """
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
    """Draw an annotated image with YOLO detection results.

    Args:
        results (list): List of YOLO results objects.

    Returns:
        PIL.Image.Image: Image with bounding boxes and labels drawn.

    Raises:
        ValueError: If no results are provided.
    """
    if not results or len(results) == 0:
        raise ValueError("No results provided for annotation")
    
    result = results[0]
    return result.plot(pil=True)




def discover_models() -> Dict[str, str]:
    """Search the models directory for YOLO .pt files.

    Returns:
        dict: Mapping of model name to full file path.
    """
    model_paths = glob.glob(os.path.join(MODEL_ROOT, "**", "weights", "*.pt"), recursive=True)
    discovered = {}

    for path in model_paths:
        # Use relative path from MODEL_ROOT as model name
        model_name = os.path.relpath(path, MODEL_ROOT).replace("/", "_").replace(".pt", "")
        discovered[model_name] = path

    return discovered


def load_all_discovered_models():
    """Load all YOLO models discovered in the models directory.

    Scans the models directory, loads each YOLO model, and registers
    them in the global `models` dictionary.

    Returns:
        None
    """
    models_found = discover_models()
    for name, path in models_found.items():
        try:
            model = YOLO(path)
            models[name] = model
            logger.info(f"Loaded model: {name} from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model {name} at {path}: {e}")




def generate_distinct_colors(model_names):
    """Generate distinct RGB colors for each model name.

    Args:
        model_names (list): List of model names.

    Returns:
        dict: Mapping of model name to RGB color tuple.
    """
    colors = {}
    for name in model_names:
        # Generate a random RGB color as a tuple
        colors[name] = tuple(random.choices(range(50, 256), k=3))
    return colors



def annotate_image_with_detections(image: PIL.Image.Image, chain_results: list, model_colors: dict) -> PIL.Image.Image:
    """Draw bounding boxes from multiple models on a single image.

    Each model’s detections are drawn in a distinct color.

    Args:
        image (PIL.Image.Image): The image to annotate.
        chain_results (list[dict]): List of detection results. Each item should contain:
            - **model** (str): Name of the model.
            - **detections** (list[dict]): List of detections with keys:
                - **bbox** (dict): Bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`).
                - **class** (str): Class name of the detected object.
                - **confidence** (float): Confidence score of the detection.
        model_colors (dict[str, tuple[int, int, int]]): Mapping of model names to RGB colors.

    Returns:
        PIL.Image.Image: Annotated image with bounding boxes and labels.
    """
    draw = PIL.ImageDraw.Draw(image)

    try:
        font = PIL.ImageFont.load_default()
    except:
        font = None

    for model_result in chain_results:
        model_name = model_result["model"]
        detections = model_result["detections"]
        color = model_colors.get(model_name, (255, 0, 0))  # default red

        for det in detections:
            bbox = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]

            draw.rectangle(
                [(bbox["xmin"], bbox["ymin"]), (bbox["xmax"], bbox["ymax"])],
                outline=color,
                width=2
            )

            label = f"{cls_name}: {conf:.2f}"
            if font:
                draw.text((bbox["xmin"], bbox["ymin"] - 10), label, fill=color, font=font)
            else:
                draw.text((bbox["xmin"], bbox["ymin"] - 10), label, fill=color)

    return image