import streamlit as st
import requests
import threading
import sseclient
import json
from urllib.parse import urlencode
import time

st.set_page_config(page_title="YOLO Trainer", layout="centered")

st.title("📦 Train YOLOv8 via FastAPI")

# fields for users to configure the training
model_path = st.text_input("Model Path", value="yolo11n.pt")
dataset_path = st.text_input("Dataset YAML Path", value="coco8.yaml")
project_name = st.text_input("Project Name", value="my_project")
run_name = st.text_input("Run Name", value="run001")
epochs = st.number_input("Epochs", min_value=1)


# Initialize session state to allow for rendering the progress bar
if "progress_data" not in st.session_state:
    st.session_state.progress_data = None
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False



# Will start the training.
if st.button("🚀 Start Training"):
    st.info("Training started...")

    # Will display to the user how many epochs they have been through
    progress_bar = st.progress(0)
    log = st.empty()

    try:
        base_url = "http://model-training:8401/training/stream"
        params = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_name": project_name,
            "run_name": run_name,
            "epochs": epochs
        }
        full_url = f"{base_url}?{urlencode(params)}"

        # Allow the front end to recive Server Sent Events to display progress updates
        client = sseclient.SSEClient(full_url)

        for event in client:
            data = json.loads(event.data)

            # Check to see if we are getting valid data
            if "error" in data:
                st.error(f"Error during training: {data['error']}")
                break

            # Calculate progress bar and display metrics
            elif "epoch" in data:
                epoch = data["epoch"]
                total = data.get("total_epochs", 1)
                progress = epoch / total
                progress_bar.progress(progress)
                log.text(f"Epoch {epoch}/{total} — metrics: {data['metrics']}")

            elif data.get("status") == "done":
                progress_bar.progress(1.0)
                st.success(f"✅ Training finished! Model path: {data.get('path', 'unknown')}")
                break

    except Exception as e:
        st.error(f"Streaming error: {e}")