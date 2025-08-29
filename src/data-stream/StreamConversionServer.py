from flask import Flask, Response
import cv2
import time
import os

app = Flask("mjpeg")
NVR_USERNAME = os.getenv("NVR_USERNAME")
NVR_PASSWORD = os.getenv("NVR_PASSWORD")
NVR_IP = os.getenv("NVR_IP")
# NVR_PORT = os.getenv("NVR_PORT")

# Default connection string
RTSP_BASE_CONNECTION = f"rtsp://{NVR_USERNAME}:{NVR_PASSWORD}@{NVR_IP}:554/cam/realmonitor?"

# Map camera IDs to RTSP URLs
# The camera_id must have the corresponding nvr port number it is connected to in the id, 
# if is port 3 then it must only have the number 3 in it.
# This allows us to parse it easier.
CAMERAS = {
    "cam1HighDef": f"{RTSP_BASE_CONNECTION}channel=1&subtype=0&tcp",
    "cam1LowDef": f"{RTSP_BASE_CONNECTION}channel=1&subtype=1&tcp",
    "cam3HighDef": f"{RTSP_BASE_CONNECTION}channel=3&subtype=0&tcp",
    "cam3LowDef": f"{RTSP_BASE_CONNECTION}channel=3&subtype=1&tcp",
}

def open_stream(rtsp_url):
    """
    Takes in an rtsp url and will return a connection to that camera.
    """
    cap = None
    while cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("[WARN] Failed to connect to RTSP. Retrying in 5s...")
            time.sleep(5)
    print("[INFO] Connected to RTSP stream.")
    return cap



def gen_frames(rtspstream):
    """
    Takes in an rtsp url and will yield jpegs everytime it is called
    """
    cap = open_stream(rtspstream)

    # Try to connect infinetly, this is incase the stream gets dropped becuase of internet etc
    while True:
        success, frame = cap.read()
        if success:
            print(f"Connected to: {rtspstream} for decoding.")
        if not success:
            print("[WARN] Lost connection to RTSP. Reconnecting...")
            cap.release()
            cap = open_stream(rtspstream)
            continue

        # Create the image from the rtsp frame
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

# Will yield mjpeg's for the given camera
@app.route("/stream/<camera_id>.mjpeg")
def stream(camera_id):
    #Check to see if the camera is registered
    if camera_id not in CAMERAS:
        return f"Unknown camera ID: {camera_id}", 404
    #Then get yield the frames from the rtsp stream
    rtsp_url = CAMERAS[camera_id]
    return Response(gen_frames(rtsp_url),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8801, threaded=True)

