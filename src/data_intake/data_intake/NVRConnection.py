import cv2
import os
import datetime
#from datetime import datetime
import threading
from onvif import ONVIFCamera


import time
import socket
from zeep.exceptions import Fault
from zeep import Client
from requests.exceptions import RequestException
import logging
from pydantic import BaseModel


# Custom class that contains EXIF manipulation functions
from . import utils
from .Client import image_table


#Import fast api to make the container able to communicate for scheduling etc.
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from typing import List
import json


from threading import Lock

camera_comments = {}
comment_lock = Lock()



#Try to connect to the NVR
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

manual_capture_flags = {}
#take_picture = False #A flag that we can switch to take a picture if we get an input from the website etc
# This is a fall back
g_frame_interval = 960



FRAME_CONFIG_FILE = "config.json"
# Stores frame interval per channel: { "1": 960, "2": 120, ... }
camera_frame_intervals = {}

# for updating how often the cameras take pictures interval 
# is how many frames between taking pictures and channel is the camera.
class FrameIntervalRequest(BaseModel):
    """Request model for updating camera frame intervals.

    Attributes:
        channel (int): The target camera which is tied to port number.
        interval (int): Number of frames to skip between captures for this channel.
    """
    channel: int
    interval: int



class ImageMetadata(BaseModel):
    """Model representing metadata for a captured image.

    Attributes:
        image_path (str): Path to the saved image file.
        author (str): Name or identifier of the user/system that captured the image.
        serial_number (str, optional): Serial number of the camera.
        date_time (datetime.datetime): Timestamp of when the image was captured.
        user_comment (str, optional): User-provided comment or note.
        description (str, optional): Additional description of the image.
    """
    image_path: str
    author: str
    serial_number: str = None
    date_time: datetime.datetime
    user_comment: str = None
    description: str = None

    class Config:
        arbitrary_types_allowed = True

# For requests to take a picture
class CapturePayload(BaseModel):
    """Payload model for manual image capture requests.

    Attributes:
        channels (List[int]): List of camera channels to trigger captures/pictures on.
        comment (str, optional): This will be able to hold a comment for all cameras in channels.
        comments (dict[int, str], optional): comments for individual cameras mapped by camera channels.
    """
    channels: List[int]
    comment: str | None = None
    comments: dict[int, str] | None = None

# A fast api app to allow this module to talk to the other modules
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_reachable(ip, port=80, timeout=2):
    """Check if a given host is reachable on a specific port.

    Args:
        ip (str): Target IP address.
        port (int, optional): Port to test connectivity on. Defaults to 80.
        timeout (int, optional): Connection timeout in seconds. Defaults to 2.

    Returns:
        bool: True if the host is reachable, False otherwise.
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False


# Using a logger as opposed to print to make better for scaling and multithreading.
# Set to the log level desired, also may stop fastapi from suppressing the logs.
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # Set this to uvicorn.info so that the logs will propagate
# logger = logging.getLogger('uvicorn.info')
logger = logging.getLogger("nvr")
logger.setLevel(logging.DEBUG)

# Ensure logs go to stdout
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

logger.propagate = False



#logs data about the nvr and will return a set of numbers that correlate to the connected cameras
def onvifCameraInfo(username: str, password: str, nvr_ip: str):
    """Retrieve and log ONVIF camera information from the NVR.

    Args:
        username (str): NVR username.
        password (str): NVR password.
        nvr_ip (str): IP address of the NVR.

    Logs:
        - Local and UTC system time.
        - Camera device time.
        - Manufacturer, model, firmware, serial number, and hardware ID.
        - ONVIF device service capabilities.

    Raises:
        socket.error: If network connection fails.
        Fault: If ONVIF SOAP request fails.
        RequestException: If HTTP/transport request fails.
    """

    # The container time to see if it is wrong and causing Wsse error
    logger.debug(f"Container Local time:     {time.ctime()}")

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_now = datetime.datetime.now().astimezone()

    logger.debug(f"UTC time:       {utc_now.isoformat()}")
    logger.debug(f"Local datetime: {local_now.isoformat()}")




    retries = 0
    while retries < MAX_RETRIES:
        if not is_reachable(nvr_ip, port=80):
            logger.error(f"Cannot reach {nvr_ip}. Retrying in {RETRY_DELAY}s...")
            retries += 1
            time.sleep(RETRY_DELAY)
            continue

        try:
            
            logger.info(f"Connecting to NVR ONVIF cameras at {nvr_ip}...")
            # Creating management services to get details from the NVR.
            camera = ONVIFCamera(nvr_ip, 80, username, password)
            device_service = camera.create_devicemgmt_service()

            device_time = device_service.GetSystemDateAndTime()
            logger.info(f"Camera Time: {device_time}")


            device_info = device_service.GetDeviceInformation()
            logger.info("Manufacturer: %s", device_info.Manufacturer)
            logger.info("Model: %s", device_info.Model)
            logger.info("Firmware: %s", device_info.FirmwareVersion)
            logger.info("Serial Number: %s", device_info.SerialNumber)
            logger.info("Hardware ID: %s", device_info.HardwareId)


            device_outputs = device_service.GetCapabilities({'Category': 'All'})

            logger.info("Device service capabilities: %s", device_outputs)


            return  # success

        except (socket.error, Fault, RequestException) as e:
            logger.error(f"Error connecting to ONVIF device: {e}")
            retries += 1
            time.sleep(RETRY_DELAY)
    logger.error("Failed to retrieve camera info after multiple attempts.")



def getCameraChannels(username: str, password: str, nvr_ip: str):
    """Discover available camera channels from the NVR.

    Args:
        username (str): NVR username.
        password (str): NVR password.
        nvr_ip (str): IP address of the NVR.

    Returns:
        set[str]: A set of detected camera channel ie ["3","7","9"].

    Raises:
        socket.error: If network connection fails.
        Fault: If ONVIF SOAP request fails.
        RequestException: If HTTP/transport request fails.
    """
    try:
        logger.info(f"Connecting to NVR ONVIF cameras at {nvr_ip}...")
        camera = ONVIFCamera(nvr_ip, 80, username, password)
        media_service = camera.create_media_service()

            
        profiles = media_service.GetProfiles()

        camera_channels=set()

        for idx, profile in enumerate(profiles):
            name = str(profile.Name)

            # Example Name "MediaProfile_Channel1_MainStream"
            # get the number component from the returned string to help dynamically initialize connection with the cameras
            name_components = name.split("_")
            # grab the center part
            channel_component = name_components[1]
            # grab the number
            channel_number = channel_component[7:]
            camera_channels.add(channel_number)
            logger.info(f"Detected channels {channel_number}")
            
        logger.info(f"Set of camera channels {camera_channels}")
        return camera_channels

    except (socket.error, Fault, RequestException) as e:
            logger.error(f"Error connecting to ONVIF device: {e}")
            #retries += 1
            time.sleep(RETRY_DELAY)


# Create a global stop event
stop_event = threading.Event()


def capture_camera(username, password, nvr_ip, channel, subtype=0):
    """Continuously capture frames from a camera channel, saving images and video.

    Args:
        username (str): NVR username.
        password (str): NVR password.
        nvr_ip (str): IP address of the NVR.
        channel (int): Camera channel to capture from.
        subtype (int, optional): Stream subtype (e.g., 0 = main stream, 1 = substream).
            Defaults to 0.

    Behavior:
        - Saves continuous video stream to `.avi` file.
        - Saves periodic still frames based on channel frame interval.
        - Supports manual capture triggers via `manual_capture_flags`.
        - Stops gracefully when `stop_event` is set.

    Logs:
        - Frame dimensions, FPS, capture status, saved image paths.
    """
    # This is so that we can tell one camera to take a picture.
    global manual_capture_flags
    global g_frame_interval
    global camera_frame_intervals
    logger.debug(f"capture_camera started with channel={repr(channel)} (type={type(channel)})")
    logger.debug(f"manual_capture_flags keys={list(manual_capture_flags.keys())}")

    # These are the ouptut directories for each channel/camera
    video_output_path = f"videos/{datetime.date.today()}_output_video_channel_{channel}.avi"
    frame_output_dir = f"images/extracted_frames/{datetime.date.today()}_extracted_frames_channel_{channel}"
    #frame_interval = 960 # This is how often we take a picture, if set to 60 on a 30 fps camera, it will be 
    # about every two seconds.

    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs("videos", exist_ok=True)


    # connecting to the camera through the rtsp url
    rtsp_url = f"rtsp://{username}:{password}@{nvr_ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"
    cap = cv2.VideoCapture(rtsp_url)

    # getting the information from the captured stream, to configure the video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15

    logger.info(f"Channel: {channel},  Frame Width: {frame_width}, Frame Height: {frame_height}, Fps: {fps}")

    

    # creating the writer with the correct fps and size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))


    # used for counting how often we take pictures.
    frame_count = 0


    # verify that the stream is open
    if not cap.isOpened():
        logger.error(f"Failed to open RTSP stream for channel {channel}.")

        return
    else:
        logger.info(f"RTSP stream opened for channel {channel}.")




    # main capture loop where we take the images
    while cap.isOpened():

        if stop_event.is_set():
                    logger.info(f"Stop signal received. Stopping capture on channel {channel}.")
                    break

        ret, frame = cap.read()
        if not ret:
            logger.error(f"Frame read failed for channel {channel}. Attempting to reconnect...")
            cap.release()
            time.sleep(3)  # wait before reconnecting
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logger.error(f"Reconnection failed. Exiting channel {channel}.")
                break
            continue

        # Save video frame
        video_writer.write(frame)

        # Save every nth frame, is camera specific
        frame_interval = camera_frame_intervals.get(str(channel), g_frame_interval)
        if frame_count % frame_interval == 0 or manual_capture_flags[channel].is_set():
            logger.debug(f"Camera: {channel}, frame interval {frame_interval}")
            #timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y:%m:%d %H:%M:%S")
            dt_obj = datetime.datetime.now(datetime.timezone.utc)
            timestamp_str = dt_obj.strftime("%Y:%m:%d %H:%M:%S")
            if manual_capture_flags[channel].is_set():
                frame_filename = os.path.join(frame_output_dir, f"camera_{channel}_frame_{timestamp_str}_manual.jpg")
                trigger_method = "manual"
            else:
                frame_filename = os.path.join(frame_output_dir, f"camera_{channel}_frame_{timestamp_str}.jpg")
                trigger_method = "automatic"
            logger.info(f"Saving image {frame_filename}")
            cv2.imwrite(frame_filename, frame)
            os.chmod(frame_filename, 0o777) #Grants read and write to all users to ensure that label studio can access them
            
            
            user_comment = None
            with comment_lock:
                if str(channel) in camera_comments:
                    user_comment = camera_comments[str(channel)]
                    del camera_comments[str(channel)]  # consume once


            SavePictureData(frame_filename, channel,"", dt_obj, user_comment or "No comment", "ceiling",trigger_method)

            manual_capture_flags[channel].clear()

            # Clear frame count so that it will not overflow eventually
            frame_count = 0

        frame_count += 1

        # Optional: Show frame window per camera will not work in docker
        # cv2.imshow(f'Camera {channel} Stream', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # ensure that resources are freed
    cap.release()
    video_writer.release()
    # cv2.destroyAllWindows()
    logger.info(f"Finished camera channel {channel}. Video saved to: {video_output_path}")

#This will both write the data to the exif tag and save it to the mongoDB
def SavePictureData(image_path, author, serialNumber, dateTime, userComment, description, trgger_method):
    """Write EXIF metadata to an image and store metadata in MongoDB.

    Args:
        image_path (str): Path to the saved image.
        author (str): Name or identifier of the capture source.
        serialNumber (str): Device serial number.
        dateTime (datetime.datetime): Timestamp of capture.
        userComment (str): User-provided comment.
        description (str): Description of the image content.
        trgger_method (str): How the capture was triggered (e.g., manual, interval).

    Behavior:
        - Writes EXIF metadata into the image file.
        - Inserts metadata document into MongoDB.
    
    Todo:
        add trigger_method to the database 
    """
    #images=ImageMetadata
    
    #Add the meta data to the image themselves
    utils.writeExifTag(image_path, author, serialNumber, dateTime, userComment, description)


     # Convert dateTime string to datetime object (e.g. "2025:07:30 14:41:04") to make querying easier
    # try:
    #     dt_obj = datetime.datetime.strptime(dateTime, "%Y:%m:%d %H:%M:%S")
    # except ValueError as e:
    #     logger.error(f"Invalid dateTime format: {dateTime}, Error: {e}")
    #     dt_obj = None

    # Using pydantic model to ensure data consistency
    metadata = ImageMetadata(
        image_path=image_path,
        author=author,
        serial_number=serialNumber,
        date_time=dateTime,
        user_comment=userComment,
        description=description
    )

    # Insert into MongoDB (convert to dict)
    resp = image_table.insert_one(metadata.dict())
    logger.info(f"Inserted into MongoDb: {metadata.dict()}, \n Response: {resp}")








# @app.on_event("startup")
# async def startup_event():
# # if __name__ == "__main__":
#     global manual_capture_flags
#     username = os.getenv("NVR_USERNAME")
#     password = os.getenv("NVR_PASSWORD")

#     if not username or not password:
#         raise ValueError("NVR_USERNAME and NVR_PASSWORD environment variables must be set.")

#     nvr_ip = os.getenv("NVR_IP")

#     # display NVR info.
#     onvifCameraInfo(username, password, nvr_ip)

#     #channels = [1,3] # gives all of the channels you would like to connect to.
#     channels = getCameraChannels(username, password, nvr_ip)



#     # Load frame intervals from config.json if it exists
#     if os.path.exists(FRAME_CONFIG_FILE):
#         try:
#             with open(FRAME_CONFIG_FILE, "r") as f:
#                 camera_frame_intervals = json.load(f)
#             logger.info(f"Loaded frame intervals from config.json: {camera_frame_intervals}")
#         except Exception as e:
#             logger.error(f"Failed to load frame intervals: {e}")
#             # fallback to default
#             camera_frame_intervals = {ch: g_frame_interval for ch in channels}
#     else:
#         camera_frame_intervals = {ch: g_frame_interval for ch in channels}


#     # Creating a flag for every channel to manually take pictures
#     manual_capture_flags = {ch: threading.Event() for ch in channels}
#     logger.debug(f"This is what is inside of manual_capture_flags: {manual_capture_flags}")
#     threads = []
#     try:
#     # for every channel create a thread
#         for ch in channels:
#             t = threading.Thread(target=capture_camera, args=(username, password, nvr_ip, ch))
#             t.start()
#             threads.append(t)
#     except KeyboardInterrupt:
#             logger.info("\nKeyboardInterrupt received, stopping threads...")
#             stop_event.set()  # Signal threads to stop
#     # Wait for all threads to finish
#     for t in threads:
#         t.join()

def change_frame_interval(channel: int, frame_interval: int):
    """Update the capture frame interval for a channel and persist it to disk.

    Updates the global `camera_frame_intervals` for the given `channel` (stored as a
    string key) and writes the resulting mapping to FRAME_CONFIG_FILE (`config.json`)
    as pretty-printed JSON.

    Args:
        channel (int): Camera channel identifier to update.
        frame_interval (int): New interval expressed in number of frames between saved stills.

    Globals modified:
        camera_frame_intervals (dict[str, int]): Updated with the new interval for `str(channel)`.

    Behaviour:
        - Writes the `camera_frame_intervals` mapping to FRAME_CONFIG_FILE.
        - Logs success or failure.

    Error handling:
        - Any exception raised while writing the file is caught and logged; the
          function does not re-raise exceptions.

    Returns:
        None
    """
    global camera_frame_intervals
    camera_frame_intervals[str(channel)] = frame_interval

    # Save to disk
    try:
        with open(FRAME_CONFIG_FILE, "w") as f:
            json.dump(camera_frame_intervals, f, indent=4)
        logger.info(f"Frame interval for channel {channel} set to {frame_interval} and saved to config.json")
    except Exception as e:
        logger.error(f"Failed to save config.json: {e}")


@app.on_event("startup")
async def startup_event():
    """FastAPI startup handler that initializes camera capture threads and loads config.

    This startup event reads NVR credentials from environment variables, retrieves
    basic ONVIF device information, discovers available camera channels, loads
    per-channel frame-intervals from FRAME_CONFIG_FILE (falls back to `g_frame_interval`),
    creates manual capture `threading.Event` flags for each channel, and spawns a
    daemon thread that runs `capture_camera` for each discovered channel.

    Globals modified:
        manual_capture_flags (dict[str, threading.Event]): Mapping of channel -> Event used to trigger manual captures.
        camera_frame_intervals (dict[str, int]): Mapping of channel -> frames-between-captures.

    Behavior:
        - Calls `onvifCameraInfo` and `getCameraChannels` (network/ONVIF IO).
        - May create/modify FRAME_CONFIG_FILE-backed data in memory.
        - Starts one daemon thread per channel (threads call `capture_camera`).
        - Logs info, warnings, and errors to `logger`.

    Raises:
        ValueError: If NVR_USERNAME or NVR_PASSWORD environment variables are not set.

    Returns:
        None
    """

    global manual_capture_flags
    global camera_frame_intervals

    username = os.getenv("NVR_USERNAME")
    password = os.getenv("NVR_PASSWORD")
    if not username or not password:
        raise ValueError("NVR_USERNAME and NVR_PASSWORD environment variables must be set.")

    nvr_ip = os.getenv("NVR_IP")

    # Display NVR info
    onvifCameraInfo(username, password, nvr_ip)

    # Get channel list
    channels = getCameraChannels(username, password, nvr_ip)

    # Load frame intervals for how often we take pictures
    if os.path.exists(FRAME_CONFIG_FILE):
        try:
            with open(FRAME_CONFIG_FILE, "r") as f:
                camera_frame_intervals = json.load(f)
            logger.info(f"Loaded frame intervals from config.json: {camera_frame_intervals}")
        except Exception as e:
            logger.error(f"Failed to load frame intervals: {e}")
            camera_frame_intervals = {ch: g_frame_interval for ch in channels}
    else:
        camera_frame_intervals = {ch: g_frame_interval for ch in channels}

    # Create manual capture flags
    manual_capture_flags = {ch: threading.Event() for ch in channels}
    logger.debug(f"manual_capture_flags: {manual_capture_flags}")

    # Start capture threads (daemon so they won’t block shutdown)
    for ch in channels:
        t = threading.Thread(
            target=capture_camera,
            args=(username, password, nvr_ip, ch),
            daemon=True
        )
        t.start()
        logger.info(f"Started capture thread for channel {ch}")

    logger.info("Startup complete — API is ready.")







# @app.post("/capture/")
# async def trigger_capture(payload : CapturePayload):
#     global manual_capture_flags
#     """Manually trigger image capture on one or more camera channels.

#     Args:
#         payload (CapturePayload): Payload containing list of channels to capture.

#     Returns:
#         dict: Response with status and channels where capture was triggered.

#     Raises:
#         500 Internal Server Error: If an error occurs during capture triggering.
#     """
#     try:
#         # For every channel we have passed in.
#         for ch in payload.channels:
#             #Check to see if it extists in the list
#             # Ensure that it is a string, because manual_capture_flags is {"1": event}
#             if str(ch) in manual_capture_flags:
                
#                 manual_capture_flags[str(ch)].set()
#                 logger.info(f"Manual capture triggered for channel {ch}")
#             else:
#                 logger.warning(f"Channel {ch} not found")

#         return {"status": "ok", "channels": payload.channels}
#     except Exception as e:
#         logger.error(f"Error in trigger_capture: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})




@app.post("/capture/")
async def trigger_capture(payload: CapturePayload):
    global manual_capture_flags, camera_comments

    try:
        with comment_lock:
            for ch in payload.channels:
                ch_str = str(ch)
                if ch_str in manual_capture_flags:
                    # Assign comment if provided comments for each channel
                    logger.debug(f"Payload comments: {payload.comments}, Payload comment: {payload.comment}")

                    if payload.comments and ch_str in payload.comments:
                        camera_comments[ch_str] = payload.comments[ch_str]
                        logger.info("individual comments")
                        logger.info(camera_comments)

                    # Assign a global comment to all cameras
                    elif payload.comment:
                        camera_comments[ch_str] = payload.comment
                        logger.info("global comment")
                        logger.info(camera_comments)

                    # Trigger manual capture
                    manual_capture_flags[ch_str].set()
                    logger.info(f"Manual capture triggered for channel {ch_str} "
                                f"with comment: {camera_comments.get(ch_str)}")
                else:
                    logger.warning(f"Channel {ch} not found")

        return {"status": "ok", "channels": payload.channels}
    except Exception as e:
        logger.error(f"Error in trigger_capture: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    

# @app.post("/frame_interval/")
# async def set_frame_interval(interval: int = Body(..., embed=True)):
#     """
#     Update the global frame capture interval
#     """
#     try:
#         change_frame_interval(interval)
#         logger.info(f"Frame interval set to {interval}")
#         return {"status": "ok", "frame_interval": interval}
#     except Exception as e:
#         logger.error(f"Error in set_frame_interval: {e}")
#         return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/frame_interval")
async def set_frame_interval(request: FrameIntervalRequest):
    """Update the frame capture interval for a specific camera channel.

    Args:
        request (FrameIntervalRequest): Channel and interval update request.

    Returns:
        dict: Response with status, updated channel, and new frame interval.

    Raises:
        500 Internal Server Error: If updating the interval fails.
    """
    try:
        change_frame_interval(request.channel, request.interval)
        return {"status": "ok", "channel": request.channel, "frame_interval": request.interval}
    except Exception as e:
        logger.error(f"Error in set_frame_interval: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        dict: {"status": "ok"} if service is healthy.

    Raises:
        500 Internal Server Error: If health check fails.
    """
    try:
        logger.debug("The health end point has been hit.")
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})