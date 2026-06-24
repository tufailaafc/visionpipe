# **PART OF THE DEPCRECATED VERSION**

import PIL.Image
from ultralytics import YOLO
import logging
import PIL
import io
import numpy
from typing import Dict, Any
import deeplabcut
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
# import tempfile
import shutil
import pandas as pd

import os
import glob
import random
import math

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

# The following is a version that I went to add to allow for DeepLabCut models
# Without realizing that is is deprecated, I will leave it here for now in case we want to use it in the future.
# - Jake Tensen 


# adds the model to the list so that we can use it for inference
#def load_model(model_name: str, model_path: str):
#
#    try:
#
#        # DeepLabCut model
#        if model_path.endswith("config.yaml"):
#
#           models[model_name] = {
#                "type": "DeepLabCut",
#                "config": model_path
#            }
#
#            logger.info(
#                f"Registered DeepLabCut model {model_name}"
#            )
#
#            return True
#
#        # YOLO model
#        model = YOLO(model_path)
#       models[model_name] = model
#
#        logger.info(
#            f"Loaded YOLO model {model_name}"
#        )
#
#        return True
#
#    except Exception as e:
#
#        logger.error(
#            f"Failed to load model {model_name}: {e}"
#        )
#
#        return False


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
        logger.error(f"Failed to load model {model_name} at path {model_path}: {e}")
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
    model_paths = glob.glob(os.path.join(MODEL_ROOT+"/YOLO", "**", "weights", "*.pt"), recursive=True)
    discovered = {}

    for path in model_paths:
        # Use relative path from MODEL_ROOT as model name
        model_name = os.path.relpath(path, MODEL_ROOT).replace("/", "_").replace(".pt", "")
        discovered[model_name] = path
    return discovered


def discover_deepLabCut_models() -> Dict[str, str]:
    """Search the models directory for DeepLabCut config.yml files.

    Returns:
        dict: Mapping of model name to full file path.
    """
    print("tried to load DeepLabCutModels")
    model_paths = glob.glob(os.path.join(MODEL_ROOT+"/DeepLabCut", "**", "config.yaml"), recursive=True)
    discovered = {}

    for path in model_paths:
        # Use relative path from MODEL_ROOT as model name
        model_name = os.path.relpath(path, MODEL_ROOT).replace("/", "_")
        discovered[model_name] = path
    print(f"discovered DeepLabCut: {discovered}")
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


def clear_directory(path: Path):
    """Clears all contents in the specified directory

    Args:
        path (Path): Path of the directory to delete all contents

    Returns:
        None
    """
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)




def deepLabCut_predict(image, model_name, return_annotated_image, p_cutoff=0.15):
    """Perform inference on an image using a DeepLabCut model.

    Args:
        input_image (PIL.Image.Image): Input image for detection.
        model_name (str): The registered model name to use.
        return_annotated_image (bool): Determines if we create and return and annotated image.
        confidence_threshold (float, optional): Minimum confidence score
            for detections. Defaults to 0.15.

    Returns:
        Pandas.DataFrame: contains x,y, likelihood, bodypart
        PIL.Image.Image(Optional): The annotated image 
    """
    output_dir = Path("/app/tmp/output")
    images_dir = Path("/app/tmp/input")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Optional: ensure directories start clean
    clear_directory(output_dir)
    clear_directory(images_dir)

    # Get a list of all the deeplabcut models
    deeplabmodels = discover_deepLabCut_models()
    # Get the path to the config file
    config_path = deeplabmodels.get(model_name)

    if not isinstance(config_path, str):
        raise RuntimeError("Config file not found")

    
    project_path = config_path.replace("/config.yaml", "")

    # Replacing the project path in config to ensure it works in Docker
    auxiliaryfunctions.edit_config(
        config_path,
        {"project_path": project_path}
    )

    # Save input image
    image_path = images_dir / "frame.png"
    # May need something like this for if we are recieving multiple requests
    # for i, img in enumerate(image):
    #     img.save(images_dir / f"{i:04d}.png")
    image.save(image_path)

    # Run DLC inference
    deeplabcut.analyze_images(
        config=config_path,
        images=[str(image_path)],
        destfolder=str(output_dir),
        save_as_csv=True,
        plotting=return_annotated_image,
        plot_skeleton=return_annotated_image,
        pcutoff=p_cutoff,
        #These values can be set to more explicitly select the model
        # snapshotindex=-1,
        # shuffle=0,
        # trainingsetindex=0,
    )

    # ---- READ RESULTS ----
    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No DLC output CSV found")

    results_df = pd.read_csv(csv_files[0], header=[0, 1, 2])
    print(f"return_annotated_image: {return_annotated_image}")

    # ---- LOAD ANNOTATED IMAGE IF REQUESTED ----
    annotated_image = None
    if return_annotated_image:

        # Looking for the output image directory
        labeled_dirs = [
            p for p in output_dir.iterdir()
                if p.is_dir() and p.name.startswith("LabeledImages_")
            ]

        if len(labeled_dirs) == 0:
            raise RuntimeError("DLC did not produce a LabeledImages directory")

        if len(labeled_dirs) > 1:
            raise RuntimeError(
                f"Expected 1 LabeledImages directory, found {len(labeled_dirs)}"
            )
        predicted_dir = labeled_dirs[0]
        
        annotated_candidates = list(predicted_dir.glob("*.png"))
        print(f"Trying to return one of these images: {annotated_candidates}")

        if annotated_candidates:
            annotated_image = PIL.Image.open(annotated_candidates[0]).convert("RGB")



    # ---- MANUAL CLEANUP ----
    clear_directory(output_dir)
    clear_directory(images_dir)
    
    

    return results_df, annotated_image






def format_dlc_results(df):
    """Formats the passed in results for DeepLabCut to be compatible with the rest of the system

    Args:
        df (Pandas.DataFrame): A multi-index dataframe produced by deeplabcut

    Returns:
        List[Dict]: key value pairs for: bodypart, x, y, likelihood and frame_index
    """
    # Drop the "coords" row
    df = df[df.iloc[:, 0] != "coords"]
    
    if df.empty:
        return []

    # Single-image inference → one row
    row = df.iloc[0]

    results = []

    # Skip first column (image path)
    cols = df.columns[1:]

    # Iterate in triplets: x, y, likelihood
    for i in range(0, len(cols), 3):
        try:
            x_col = cols[i]
            y_col = cols[i + 1]
            l_col = cols[i + 2]
        except IndexError:
            continue

        # bodypart is level-2 of the MultiIndex
        bodypart = x_col[2]

        x = float(row[x_col])
        y = float(row[y_col])
        likelihood = float(row[l_col])

        
        results.append({
            "bodypart": bodypart,
            "x": x,
            "y": y,
            "likelihood": likelihood,
            "frame_index": 0
        })
    print(results)
    return results

def polyline_length(segments:list[tuple[float,float]]):
    """
    Sums up the length of multiple line segments

    Args:
        segments (list[tuple[float,float]]): It expects them to be in the format [(x1,y1),(x2,y2),etc]
    
    Returns:
        total_length (float): The summation of the length of all the line segments.
    """
    total_length=0
    for i in range(len(segments)-1):
        dx = segments[i][0]-segments[i+1][0]
        dy = segments[i][1] - segments[i+1][1]
        total_length += math.sqrt(dx*dx + dy*dy)
    
    return total_length

def pig_lengths(body_parts:list[Dict]):
    """
    Takes in a dictionary of parts with xy coords and calculates the rough dimensions of the pig.
    If the dimensions are outside the measure of deviation we will return -1

    Args:
        List: A list of Dict containing at least all of the parts.
            Dict:
                - `bodypart` (str): The name of the bodypart
                - `x` (float): The x coord of the part
                - `y` (float): The y coord of the part
    Returns:
        Dict:
            - "pig_length" (float): The length of the pig in cenitmeters returns -1 if deemed invalid
            - "pig_width" (float): The width of the pig in centimeters returns -1 if deemed invalid
    """
    body_part = {}
    for i in body_parts:
        coords = (i.get("x"), i.get("y"))
        body_part[i.get("bodypart")] = coords

    # - Spine1
    # - Shoulder_left
    # - Shoulder_right
    # - Center
    # - Spine2
    # - Hip_left
    # - Hip_right
    # - Tail_base

    # Ensure that we have all of the parts.
    # Todo: add checks that allow for some flexibility for some missing parts
    required = [
        "Spine1", "Center", "Spine2", "Tail_base",
        "Shoulder_left", "Shoulder_right",
        "Hip_left", "Hip_right"
    ]

    for r in required:
        if r not in body_part:
            raise ValueError(f"Missing bodypart: {r}")


    # calculate the length of each body part in pixels
    pig_length_px = polyline_length([body_part["Spine1"], 
                                  body_part["Center"], 
                                  body_part["Spine2"], 
                                  body_part["Tail_base"]])
    

    shoulder_width_px = polyline_length([
        body_part["Shoulder_left"], 
        body_part["Shoulder_right"]])
    
    
    hip_width_px = polyline_length([
        body_part["Hip_left"],
        body_part["Hip_right"]])
    
    print(f"Pig Length in pixels: {pig_length_px}")
    print(f"Pig Shoulder width in pixels: {shoulder_width_px}")
    print(f"Pig hip width in pixels: {hip_width_px}")
    
    pig_width_px = (hip_width_px + shoulder_width_px)/2

    # convert pixel values to cm
    # the ratio is dependent on camera and distance from target
    # pixel_to_cm_ratio= 29.1
    #pixel_to_cm_ratio= 10

    # Found this pixel to cm ratio by using the wooden block in the image to find that to cover the 30 cm distance
    # between the tape markers, it takes roughly 400 pixels. So 400 pixels / 30 cm = 13.33 pixels/cm
    #pixel_to_cm_ratio = 13.33

    # Found this pixel to cm ratio by using the top of the RFID panel in the image to find that to cover the 41.5 cm distance
    # it takes roughly 460 pixels. So 460 pixels / 41.5 cm = 11.08 pixels/cm
    #pixel_to_cm_ratio = 11.08

    # Found this pixel to cm ratio by using near the middle (parallel with first screw set) of the RFID panel in the image
    # to find that to cover the 41.5 cm distance it takes roughly 350 pixels. So 350 pixels / 41.5 cm = 8.44 pixels/cm
    #pixel_to_cm_ratio = 8.44

    # Found this pixel to cm ratio by using near the middle (Below the first screw set by a bit) of the RFID panel in the image
    # to find that to cover the 41.5 cm distance it takes roughly 255 pixels. So 255 pixels / 41.5 cm = 6.14 pixels/cm
    pixel_to_cm_ratio = 6.14
    pig_length = pig_length_px / pixel_to_cm_ratio
    pig_width = pig_width_px / pixel_to_cm_ratio

    # check to see if the length and width makes some level of sense
    if pig_length > 145 or pig_length < 25:
        pig_length = -1

    if pig_width > 70 or pig_width < 10:
        pig_width = -1 

    length_to_width_ratio = pig_width/pig_length

    if length_to_width_ratio > 0.65 or length_to_width_ratio < 0.25:
        print(f"length to width ratio: {length_to_width_ratio}")
        pig_width = -1
        pig_length = -1

    return {"pig_length": pig_length, "pig_width":pig_width}

def pig_weight(pig_length:float,pig_width:float):
    """
    Calculates the weight based off of the length and width. 
    We make assumptions about density and depth.

    Args:
        pig_length (float): The length of the pig in centimeters
        pig_width (float):  The width of the pig in centimeters

    Returns:
        estimated_weight (float): The weight of the pig in grams
    """
    # length and width should be in cm
    depthfactor = 1.3
    density = 1 # Density of most organic things are pretty close to 1 gram cm^3 
    estimated_volume = pig_length*pig_width*pig_width*depthfactor
    estimated_weight = estimated_volume * density

    print(f"Estimated volume cm^3: {estimated_volume}")
    print(f"Estimated weight in grams: {estimated_weight}")


    return estimated_weight






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


def format_results_yolo(result, model_name):
    """Formats the passed in results from YOLO to be compatible with the rest of the system

    Args:
        result (Dict): A dictionary containing YOLO results
        model_name (str): The name of the model used for inference

    Returns:
        Dict: Contains the model name, the detections and the detection count.
    """
    detections = [
        {
        "class": det["name"],
        "confidence": det["confidence"],
        "bbox": {
            "xmin": det["xmin"],
            "ymin": det["ymin"],
            "xmax": det["xmax"],
            "ymax": det["ymax"],
            },
        }
        for det in result["detections"]
    ]

                    
    return {
            "model": model_name,
            "detections": detections,
            "detection_count": len(detections)
            }