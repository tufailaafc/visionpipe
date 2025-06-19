import cv2
import os

# ==== CONFIGURATION ====
username = "admin"
password = "bottle123"
nvr_ip = "192.168.1.5"  # Your NVR IP
channel = 3              # Camera channel (1, 2, 3, ...)
subtype = 0              # 0 = Main stream, 1 = Sub stream

# === OUTPUT SETTINGS ===
video_output_path = "output_video.avi"
frame_output_dir = "extracted_frames"
frame_interval = 30  # Save every 30th frame

# Create frame output directory if it doesn't exist
os.makedirs(frame_output_dir, exist_ok=True)

# Construct RTSP URL
rtsp_url = f"rtsp://{username}:{password}@{nvr_ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 15  # Fallback

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

if not cap.isOpened():
    print("❌ Failed to open RTSP stream.")
else:
    print("✅ RTSP stream opened. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or frame read failed.")
        break

    cv2.imshow('Lorex Camera Stream', frame)
    video_writer.write(frame)

    # Save every nth frame
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(frame_output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"✅ Finished. Video saved to: {video_output_path}")
print(f"✅ Frames saved in: {frame_output_dir}")