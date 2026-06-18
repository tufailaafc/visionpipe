# **PART OF THE DEPRECATED VERSION**

"""
YOLO Inference API using FastAPI.

This module provides a RESTful API for performing image inference with
Ultralytics YOLO models. It supports:

- Single-model predictions (`/model/predict`)
- Chained predictions using multiple models sequentially (`/model/predict/chain`)
- Dynamic model loading (`/model/load`, `/model/load/discovered`)
- Listing available and loaded models (`/models`, `/models/discover`)
- Health check for model readiness (`/model/health`)

Key Features:
- Handles base64-encoded images in POST requests.
- Returns structured detections and optionally annotated images.
- Supports multiple models and distinct bounding box colors for chaining.
- Provides error handling with clear HTTP status codes for invalid requests.

Dependencies:
- FastAPI
- Pydantic
- Ultralytics YOLO
- Python standard libraries: base64, logging, typing, os
"""



from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from fastapi import HTTPException
from ultralytics import YOLO
from typing import Optional, Dict, Any, List
# import app as app_model
from . import app as app_model
import base64
import logging




logger = logging.getLogger()


# Set to the log level desired, also may stop fastapi from suppressing the logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



# For validating all request and responses
class PredictionRequest(BaseModel):
    model_name: str
    # list of images
    instances: list
    # for thing like confidence etc
    parameters: Optional[Dict[str, Any]] = None


# So that we can do inference with multiple models
class ChainPredictionRequest(BaseModel):
    model_names: list[str]              # list of models to run in sequence
    instances: list                     # list of base64 images
    parameters: Optional[Dict[str, Any]] = None

# One model’s output in the chain
class ModelResult(BaseModel):
    model: str
    detections: List[Dict[str, Any]]
    detection_count: int

# Full chain result for a single image
class ChainResult(BaseModel):
    chain: List[ModelResult]
    annotated_image: Optional[str] = None

class PredictionResponse(BaseModel):
    predictions: List[ChainResult]

class ModelKeyRequest(BaseModel):
    model_key: str



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# checks to see if it is available and able to take another request
@app.get("/model/health")
def health_check(model_name: str = Query(...)):
    """
    Check the readiness of a specific model.

    Args:
        model_name (str): Name of the model to check.

    Returns:
        dict: {"status": "healthy", "model": model_name} if ready.

    Raises:
        HTTPException(503): If the model is not loaded or ready.
    """
    if not app_model.is_model_ready(model_name):
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not ready")
    return {"status": "healthy", "model": model_name}




@app.post("/model/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Perform inference on one or more images using a single YOLO model.

    Args:
        request (PredictionRequest): Contains:
            - model_name (str): Name of the loaded model to use.
            - instances (list): List of dicts containing base64-encoded images.
            - parameters (dict, optional): Extra options like confidence threshold 
              and return_annotated_image flag.

    Returns:
        PredictionResponse: Contains detections and optionally annotated images.

    Raises:
        HTTPException(400): If image instances are invalid.
        HTTPException(500): On prediction or internal errors.
    """
    try:
        predictions = []


        for instance in request.instances:
            # check to ensure that we are getting an image
            if isinstance(instance, dict):
                if "image" in instance:
                    image_data = base64.b64decode(instance["image"])
                    input_image = app_model.get_image_from_bytes(image_data)
                else:
                    raise HTTPException(status_code=400, detail="Instance must contain 'image' field")
            else:
                raise HTTPException(status_code=400, detail="Invalid instance format")
            
            # Check to see which model type we are using
            if "DeepLabCut" in request.model_name:
                app_model.deepLabCut_predict(input_image, request.model_name)
            else:
                # Extract YOLO11 parameters if provided
                parameters = request.parameters or {}
                confidence_threshold = parameters.get("confidence", 0.5)
                return_annotated_image = parameters.get("return_annotated_image", False)

                # Run inference with the already loaded model
                #result = app_model.run_inference(input_image, confidence_threshold=confidence_threshold)
                model_name = request.model_name

                # Run inference
                result = app_model.run_inference(
                    input_image,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold
                )
                detection_list = result["detections"]
                # Format predictions

                detections = []

                for detection in detection_list:
                    formatted_detection = {
                        "class": detection["name"],
                        "confidence": detection["confidence"],
                        "bbox": {
                            "xmin": detection["xmin"],
                            "ymin": detection["ymin"],
                            "xmax": detection["xmax"],
                            "ymax": detection["ymax"],
                        },
                    }
                    detections.append(formatted_detection)
                
                # Build prediction response
                # prediction = {"detections": detections, "detection_count": len(detections)}
                
                prediction = {"chain": [{"model": model_name, "detections": detections, "detection_count": len(detections)}]}

                # Add annotated image if requested and detections exists
                
                if (
                    return_annotated_image
                    and result["results"]
                    and result["results"][0].boxes is not None
                    and len(result["results"][0].boxes) > 0
                ):
                    # Draw bounding boxes and labels on the image
                    annotated_image = app_model.get_annotated_image(result["results"])
                    img_bytes = app_model.get_bytes_from_image(annotated_image)
                    prediction["annotated_image"] = base64.b64encode(img_bytes).decode("utf-8")
                
                # print(f"Predictions: {prediction}")
                predictions.append(prediction)
            #logger.info(f"Processed {len(request.instances)} instances, found {sum(len(p['detections']) for p in predictions)} total detections")

        return PredictionResponse(predictions=predictions)
    
    except HTTPException:
        # Re-raise HTTPException as-is (don't catch and convert to 500)
        raise

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    



@app.post("/model/predict/chain")
async def predict_chain(request: ChainPredictionRequest):
    """
    Perform sequential inference using multiple models (chained predictions).

    Args:
        request (ChainPredictionRequest): Contains:
            - model_names (list[str]): List of loaded models to run in order.
            - instances (list): List of dicts with base64-encoded images.
            - parameters (dict, optional): Options like confidence threshold 
              and return_annotated_image flag.

    Returns:
        dict: {"predictions": [ChainResult]} with detections and optionally 
              annotated images for each input image.

    Raises:
        HTTPException(400): If image instances are invalid.
        HTTPException(404): If a model in the chain is not loaded.
        HTTPException(500): On prediction or internal errors.
    """
    try:
        logger.debug(f"Parameters: {request.parameters},\n model names: {request.model_names},\n images:{request.instances}")
        predictions = []

        # Generate a distinct color for each model in the chain
        model_colors = app_model.generate_distinct_colors(request.model_names)
        

        # Process each image passed
        for instance in request.instances:

            #Ensure that the image is valid
            if isinstance(instance, dict) and "image" in instance:
                image_data = base64.b64decode(instance["image"])
                input_image = app_model.get_image_from_bytes(image_data)
            else:
                raise HTTPException(status_code=400, detail="Invalid instance format")

            chain_results = []
            parameters = request.parameters or {}
            confidence_threshold = parameters.get("confidence", 0.5)

            
            # Use each model on the current image.
            for model_name in request.model_names:
                # Check to see which model type we are using
                if "DeepLabCut" in model_name:
                    # Check to see if the user wants an annotated image returned
                    return_annotated_image = (request.parameters or {}).get("return_annotated_image", False)

                    # Process the image with the selected model
                    results, annotated_image = app_model.deepLabCut_predict(input_image, model_name, return_annotated_image, confidence_threshold)
                    
                    
                    if annotated_image != None:
                        img_bytes = app_model.get_bytes_from_image(annotated_image)
                        annotated_image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    else:
                        annotated_image_b64 = None


                    detections = app_model.format_dlc_results(results)
                    pig_dimensions = app_model.pig_lengths(detections)
                    pig_weight = app_model.pig_weight(pig_dimensions.get("pig_length"),pig_dimensions.get("pig_width"))
                    detections.append(pig_dimensions)
                    detections.append({"weight":pig_weight})

                    formatted_results = {
                        "model": model_name,
                        "detections": detections,
                        "detection_count": len(detections),
                    }
                    chain_results.append(formatted_results)

                    

                else:
                    if not app_model.is_model_ready(model_name):
                        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")

                    # Run inference
                    result = app_model.run_inference(
                        input_image,
                        model_name=model_name,
                        confidence_threshold=confidence_threshold
                        )
                    formatted_results=app_model.format_results_yolo(result, model_name)
                    chain_results.append(formatted_results)

                    annotated_image_b64 = None
                    # Annotate image with all detections from all models
                    return_annotated_image = (request.parameters or {}).get("return_annotated_image", False)
                    if return_annotated_image:
                        annotated_image = app_model.annotate_image_with_detections(input_image.copy(), chain_results, model_colors)
                        img_bytes = app_model.get_bytes_from_image(annotated_image)
                        annotated_image_b64 = base64.b64encode(img_bytes).decode("utf-8")

                predictions.append({
                    "chain": chain_results,
                    "annotated_image": annotated_image_b64
                })

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Chained prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Chained prediction failed: {e}")








# Load a model dynamically  
@app.post("/model/load")
def load_model(model_name: str = Body(...), model_path: str = Body(...)):
    """
    Dynamically load a YOLO model into memory.

    Args:
        model_name (str): Name to assign to the loaded model.
        model_path (str): Full filesystem path to the model .pt file on the server.

    Returns:
        dict: {"status": "loaded", "model": model_name} on success.

    Raises:
        HTTPException(500): If model loading fails.
    """
    success = app_model.load_model(model_name, model_path)
    if success:
        return {"status": "loaded", "model": model_name}
    raise HTTPException(status_code=500, detail="Failed to load model")


# List all loaded models
@app.get("/models")
def list_models():
    """
    List all currently loaded YOLO models.

    Returns:
        dict: {"models": [list of loaded model names]}.
    """
    return {"models": list(app_model.models.keys())}


# List all available models
@app.get("/models/discover")
def discover_available_models():
    """
    List all YOLO model files available in the /app/models directory on the server.

    Returns:
        dict: {"models": {model_name: full_path}}.
    """
    models = app_model.discover_models()
    models.update(app_model.discover_deepLabCut_models())
    return {"models": models}

# The following is a version I had added to allow for DeepLabCut models without
# realizing this is part of the deprecated version. I will leave it here for now, but it may be removed in the future.
# - Jake Tensen

#@app.post("/model/load/discovered")
#def load_discovered_model(req: ModelKeyRequest):
#
#    all_discovered = app_model.discover_models()
#    all_discovered.update(
#        app_model.discover_deepLabCut_models()
#    )
#
#    if req.model_key not in all_discovered:
#        raise HTTPException(
#            status_code=404,
#            detail=f"Model '{req.model_key}' not found."
#        )
#
#    success = app_model.load_model(
#       req.model_key,
#        all_discovered[req.model_key]
#    )
#
#    if success:
#        return {
#            "status": "loaded",
#            "model": req.model_key
#        }
#
#    raise HTTPException(
#        status_code=500,
#        detail=f"Failed to load model: {req.model_key}"
#    )

@app.post("/model/load/discovered")
def load_discovered_model(req: ModelKeyRequest):
    """
    Load a YOLO model that has been discovered in the models directory.

    Args:
        req (ModelKeyRequest): Contains the `model_key` corresponding to the discovered model.

    Returns:
        dict: {"status": "loaded", "model": model_key} on success.

    Raises:
        HTTPException(404): If the model key is not found in discovered models.
        HTTPException(500): If model loading fails.
    """
    all_discovered = app_model.discover_models()
    

    if req.model_key not in all_discovered:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_key}' not found.")

    success = app_model.load_model(req.model_key, all_discovered[req.model_key])
    if success:
        return {"status": "loaded", "model": req.model_key}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {req.model_key}")