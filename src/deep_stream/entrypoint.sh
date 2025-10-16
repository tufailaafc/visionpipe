#!/bin/bash
set -e

# Build RTSP URL from env vars
RTSP_URL="rtsp://${NVR_USERNAME}:${NVR_PASSWORD}@${NVR_IP}:${NVR_PORT}/cam/realmonitor?channel=1&subtype=1&tcp"

echo "Launching DeepStream with RTSP URL: $RTSP_URL"

# Run your DeepStream Python app with the URL
exec python3 deepstream.py "$RTSP_URL"