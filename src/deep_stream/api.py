# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepstream import add_rtsp_source  # Import your deepstream function
import logging
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class CameraRequest(BaseModel):
    rtsp_url: str


# Will add the rtsp stream as a source inside of deepstream
@app.post("/add_source")
async def add_source(camera: CameraRequest):
    logger.debug(f"Received RTSP URL: {camera.rtsp_url}")
    success = add_rtsp_source(camera.rtsp_url)
    if success:
        return {"status": "RTSP stream added"}
    raise HTTPException(status_code=400, detail="Failed to add stream")