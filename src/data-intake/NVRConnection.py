import cv2
import os
import datetime
import threading
from onvif import ONVIFCamera


import time
import socket
from zeep.exceptions import Fault
from zeep import Client
from requests.exceptions import RequestException
import logging

# Custom class that contains EXIF manipulation functions
import utils
from Client import image_table




#Try to connect to the NVR
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

manual_capture_flags = {}
#take_picture = False #A flag that we can switch to take a picture if we get an input from the website etc

def is_reachable(ip, port=80, timeout=2):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False


# Using a logger as opposed to print to make better for scaling and multithreading.
logger = logging.getLogger()



#logs data about the nvr and will return a set of numbers that correlate to the connected cameras
def onvifCameraInfo(username: str, password: str, nvr_ip: str):

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
    # This is so that we can tell one camera to take a picture.
    global manual_capture_flags


    # These are the ouptut directories for each channel/camera
    video_output_path = f"videos/{datetime.date.today()}_output_video_channel_{channel}.avi"
    frame_output_dir = f"images/extracted_frames/{datetime.date.today()}_extracted_frames_channel_{channel}"
    frame_interval = 960 # This is how often we take a picture, if set to 60 on a 30 fps camera, it will be 
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

        # Save every nth frame
        if frame_count % frame_interval == 0 or manual_capture_flags[channel].is_set():
            timestamp = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            if manual_capture_flags[channel].is_set():
                 frame_filename = os.path.join(frame_output_dir, f"camera_{channel}_frame_{timestamp}_manual.jpg")
            else:
                frame_filename = os.path.join(frame_output_dir, f"camera_{channel}_frame_{timestamp}.jpg")
            logger.info(f"Saving image {frame_filename}")
            cv2.imwrite(frame_filename, frame)

            SavePictureData(frame_filename, channel,"",str(timestamp),"Tests","ceiling")
            manual_capture_flags[channel].clear()

            # Clear frame count so that it will not overflow eventually
            frame_count = 0

        frame_count += 1

        # Optional: Show frame window per camera 
        # cv2.imshow(f'Camera {channel} Stream', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # ensure that resources are freed
    cap.release()
    video_writer.release()
    # cv2.destroyAllWindows()
    logger.info(f"Finished camera channel {channel}. Video saved to: {video_output_path}")

# This will both write the data to the exif tag and save it to the mongoDB
def SavePictureData(image_path, author, serialNumber, dateTime, userComment, description):
    images={}
    
    #Add the meta data to the image themselves
    utils.writeExifTag(image_path, author, serialNumber, dateTime, userComment, description)


    # Create a json dict to store data into the mongoDB
    images["image_path"]=image_path
    images["author"]=author
    images["serial_number"]=serialNumber
    images["dateTime"]=dateTime
    images["userComment"]=userComment
    images["description"]=description


    #Insert metadata into the MongoDB
    resp = image_table.insert_one(images)
    logger.info(f"Inserted into MongoDb: {images}, \n Response: {resp}")





if __name__ == "__main__":

    username = os.getenv("NVR_USERNAME")
    password = os.getenv("NVR_PASSWORD")

    if not username or not password:
        raise ValueError("NVR_USERNAME and NVR_PASSWORD environment variables must be set.")

    nvr_ip = "192.168.1.5"

    # display NVR info.
    onvifCameraInfo(username,password,nvr_ip)

    #channels = [1,3] # give all of the channels you would like to connect to.
    channels = getCameraChannels(username,password,nvr_ip)

    # Creating a flag for every channel to manually take pictures
    manual_capture_flags = {ch: threading.Event() for ch in channels}
    threads = []
    try:
    # for every channel create a thread
        for ch in channels:
            t = threading.Thread(target=capture_camera, args=(username, password, nvr_ip, ch))
            t.start()
            threads.append(t)
    except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt received, stopping threads...")
            stop_event.set()  # Signal threads to stop
    # Wait for all threads to finish
    for t in threads:
        t.join()