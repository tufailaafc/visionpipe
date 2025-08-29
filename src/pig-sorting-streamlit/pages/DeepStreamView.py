import streamlit as st
import cv2
import numpy as np
import requests
import time

# Need to be changed to not local host but the server ip when not running locally
mjpeg_url = "http://localhost:8801/stream.mjpeg"
# NOT FINISHED YET


st.title("DeepStream Camera Connector")

rtsp_url = st.text_input("Enter RTSP URL")

if st.button("Connect Camera"):
    try:
        response = requests.post(
            "http://deep-stream:8701/add_source", 
            json={"rtsp_url": rtsp_url}
        )
        if response.status_code == 200:
            st.success("RTSP stream sent to DeepStream!")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect: {e}")



st.title("Live DeepStream MJPEG Video")



# User picks from predefined camera IDs
camera_id = st.multiselect("Choose camera", ["cam1HighDef", "cam1LowDef","cam3HighDef", "cam3LowDef"])
if camera_id and st.button("Connect to selected cameras"):
    for cam in camera_id:
        if cam:
            mjpeg_url = f"http://localhost:8801/stream/{cam}.mjpeg"
            st.image(mjpeg_url)

# Define RTP GStreamer pipeline (as OpenCV VideoCapture input)
# gst_str = (
#     "pig-sorting-streamlit=8701 "
#     "! application/x-rtp, encoding-name=H264, payload=96 "
#     "! rtph264depay "
#     "! h264parse "
#     "! avdec_h264 "
#     "! videoconvert "
#     "! appsink"
# )

# for _ in range(25):
#     cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
#     if cap.isOpened():
#         break
#     time.sleep(5)



# # # Open GStreamer pipeline
# # cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

# if not cap.isOpened():
#     st.error("Failed to open video stream.")
# else:
#     stframe = st.empty()
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Stream ended or no frame received.")
#             break
#         # Show frame in Streamlit
#         stframe.image(frame, channels="BGR")