from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from fastapi import HTTPException
from ultralytics import YOLO
from typing import Optional, Dict, Any, List
import app as app_model
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
    annotated_image: str  

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
    if not app_model.is_model_ready(model_name):
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not ready")
    return {"status": "healthy", "model": model_name}




@app.post("/model/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
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
            prediction = {"detections": detections, "detection_count": len(detections)}

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

            predictions.append(prediction)
        logger.info(f"Processed {len(request.instances)} instances, found {sum(len(p['detections']) for p in predictions)} total detections")

        return PredictionResponse(predictions=predictions)
    
    except HTTPException:
        # Re-raise HTTPException as-is (don't catch and convert to 500)
        raise

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    


# @app.post("/model/predict/chain")
# async def predict_chain(request: ChainPredictionRequest):
#     try:
#         predictions = []

#         for instance in request.instances:
#             if isinstance(instance, dict) and "image" in instance:
#                 image_data = base64.b64decode(instance["image"])
#                 input_image = app_model.get_image_from_bytes(image_data)
#             else:
#                 raise HTTPException(status_code=400, detail="Invalid instance format")

#             # Process through each model in sequence
#             chain_results = []
#             current_image = input_image

#             for model_name in request.model_names:
#                 if not app_model.is_model_ready(model_name):
#                     raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")

#                 parameters = request.parameters or {}
#                 confidence_threshold = parameters.get("confidence", 0.5)
#                 return_annotated_image = parameters.get("return_annotated_image", False)

#                 # Run inference
#                 result = app_model.run_inference(
#                     current_image,
#                     model_name=model_name,
#                     confidence_threshold=confidence_threshold
#                 )

#                 detections = []
#                 for det in result["detections"]:
#                     detections.append({
#                         "class": det["name"],
#                         "confidence": det["confidence"],
#                         "bbox": {
#                             "xmin": det["xmin"],
#                             "ymin": det["ymin"],
#                             "xmax": det["xmax"],
#                             "ymax": det["ymax"],
#                         },
#                     })

#                 chain_step = {
#                     "model": model_name,
#                     "detections": detections,
#                     "detection_count": len(detections),
#                 }

#                 # Add annotated image if requested
#                 if (
#                     return_annotated_image
#                     and result["results"]
#                     and result["results"][0].boxes is not None
#                     and len(result["results"][0].boxes) > 0
#                 ):
#                     annotated_image = app_model.get_annotated_image(result["results"])
#                     img_bytes = app_model.get_bytes_from_image(annotated_image)
#                     chain_step["annotated_image"] = base64.b64encode(img_bytes).decode("utf-8")

#                     # update current_image → pass annotated image to next model
#                     current_image = annotated_image  

#                 chain_results.append(chain_step)

#             predictions.append({"chain": chain_results})

#         return {"predictions": predictions}

#     except Exception as e:
#         logger.error(f"Chained prediction error: {e}")
#         raise HTTPException(status_code=500, detail=f"Chained prediction failed: {e}")



@app.post("/model/predict/chain")
async def predict_chain(request: ChainPredictionRequest):
    try:
        logger.debug(f"Parameters: {request.parameters},\n model names: {request.model_names},\n images:{request.instances}")
        predictions = []

        # Generate a distinct color for each model in the chain
        model_colors = app_model.generate_distinct_colors(request.model_names)

        for instance in request.instances:
            if isinstance(instance, dict) and "image" in instance:
                image_data = base64.b64decode(instance["image"])
                input_image = app_model.get_image_from_bytes(image_data)
            else:
                raise HTTPException(status_code=400, detail="Invalid instance format")

            chain_results = []

            for model_name in request.model_names:
                if not app_model.is_model_ready(model_name):
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")

                parameters = request.parameters or {}
                confidence_threshold = parameters.get("confidence", 0.5)

                # Run inference
                result = app_model.run_inference(
                    input_image,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold
                )

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

                chain_results.append({
                    "model": model_name,
                    "detections": detections,
                    "detection_count": len(detections)
                })

            # Annotate image with all detections from all models
            annotated_image_b64 = None
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
    success = app_model.load_model(model_name, model_path)
    if success:
        return {"status": "loaded", "model": model_name}
    raise HTTPException(status_code=500, detail="Failed to load model")


# List all loaded models
@app.get("/models")
def list_models():
    return {"models": list(app_model.models.keys())}


# List all available models
@app.get("/models/discover")
def discover_available_models():
    """
    Lists all available model files found in the /app/models directory.
    """
    return {"models": app_model.discover_models()}



@app.post("/model/load/discovered")
def load_discovered_model(req: ModelKeyRequest):
    all_discovered = app_model.discover_models()

    if req.model_key not in all_discovered:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_key}' not found.")

    success = app_model.load_model(req.model_key, all_discovered[req.model_key])
    if success:
        return {"status": "loaded", "model": req.model_key}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {req.model_key}")