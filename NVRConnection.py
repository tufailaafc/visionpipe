import cv2
import os
import datetime
import threading


def capture_camera(username, password, nvr_ip, channel, subtype=0):
    video_output_path = f"output_video_channel_{channel}.avi"
    frame_output_dir = f"extracted_frames/{datetime.date.today()}_extracted_frames_channel_{channel}"
    frame_interval = 120

    os.makedirs(frame_output_dir, exist_ok=True)

    rtsp_url = f"rtsp://{username}:{password}@{nvr_ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"
    cap = cv2.VideoCapture(rtsp_url)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    if not cap.isOpened():
        print(f"❌ Failed to open RTSP stream for channel {channel}.")
        return
    else:
        print(f"✅ RTSP stream opened for channel {channel}.")

    while cap.isOpened():
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

    cap.release()
    video_writer.release()
    # cv2.destroyAllWindows()

    print(f"✅ Finished camera channel {channel}. Video saved to: {video_output_path}")

if __name__ == "__main__":
    username = "admin"
    password = "bottle123"
    nvr_ip = "192.168.0.2"

    channels = [3] 
    threads = []

# for every channel create a thread
    for ch in channels:
        t = threading.Thread(target=capture_camera, args=(username, password, nvr_ip, ch))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()