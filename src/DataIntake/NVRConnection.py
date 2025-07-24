import cv2
import os
import datetime
import threading
from onvif import ONVIFCamera

# Create a global stop event
stop_event = threading.Event()


def capture_camera(username, password, nvr_ip, channel, subtype=0):
    video_output_path = f"video/output_video_channel_{channel}.avi"
    frame_output_dir = f"extracted_frames/{datetime.date.today()}_extracted_frames_channel_{channel}"
    frame_interval = 120

    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs("video", exist_ok=True)


    # connecting to the camera through the rtsp url
    rtsp_url = f"rtsp://{username}:{password}@{nvr_ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"
    cap = cv2.VideoCapture(rtsp_url)

    # getting the information from the captured stream
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15


    # creating the writer with the correct fps and size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))


    # used for counting how often we take pictures.
    frame_count = 0


    # verify that the stream is open
    if not cap.isOpened():
        print(f"❌ Failed to open RTSP stream for channel {channel}.")
        return
    else:
        print(f"✅ RTSP stream opened for channel {channel}.")



    # main capture loop where we take the images
    while cap.isOpened():

        if stop_event.is_set():
                    print(f"🛑 Stop signal received. Stopping capture on channel {channel}.")
                    break

        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Stream ended or frame read failed for channel {channel}.")
            break

        # Save video frame
        video_writer.write(frame)

        # Save every nth frame
        if frame_count % frame_interval == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = os.path.join(frame_output_dir, f"camera_{channel}_frame_{timestamp}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

        # Optional: Show frame window per camera 
        # cv2.imshow(f'Camera {channel} Stream', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # ensure that resources are freed
    cap.release()
    video_writer.release()
    # cv2.destroyAllWindows()

    print(f"✅ Finished camera channel {channel}. Video saved to: {video_output_path}")


def onvifCameraInfo(username :str,password:str,nvr_ip:str):
    # Connect to the camera
    camera = ONVIFCamera(nvr_ip, 80, username, password)

    # Create device management service
    device_service = camera.create_devicemgmt_service()
    device_time = device_service.GetSystemDateAndTime()
    print(device_time)

    # Get device information
    device_info = device_service.GetDeviceInformation()
    print("Manufacturer:", device_info.Manufacturer)
    print("Model:", device_info.Model)
    print("FirmwareVersion:", device_info.FirmwareVersion)
    print("SerialNumber:", device_info.SerialNumber)
    print("HardwareId:", device_info.HardwareId)

if __name__ == "__main__":
    username = "admin"
    password = "bottle123"
    nvr_ip = "192.168.0.2"

    # display camera info.
    onvifCameraInfo(username,password,nvr_ip)

    channels = [3,5] 
    threads = []
    try:
    # for every channel create a thread
        for ch in channels:
            t = threading.Thread(target=capture_camera, args=(username, password, nvr_ip, ch))
            t.start()
            threads.append(t)
    except KeyboardInterrupt:
            print("\n⚠️ KeyboardInterrupt received, stopping threads...")
            stop_event.set()  # Signal threads to stop
    # Wait for all threads to finish
    for t in threads:
        t.join()