import streamlit as st
import requests
import threading
import sseclient
import json
from urllib.parse import urlencode
import time


# Returns a path to the datasets that are stored on the server which will be a zip folder 
def get_datasets():
    try:
        response = requests.get(
            "http://model-training:8401/training/datasets"
        )
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")









st.set_page_config(page_title="YOLO Trainer", layout="centered")

st.title("📦 Train YOLO via FastAPI")

# fields for users to configure the training
model_path = st.text_input("Model Path", value="yolo11n.pt")

dataset_path = st.selectbox("Dataset path ie to the zip folder", get_datasets())

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
        base_url = "http://pig-sorting-api:8001/api/v1/training/stream_trainer"
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





def get_datasets():
    try:
        response = requests.get(
            #works because the backend is on the same docker network. If not replace (pig_sorting_api) with the server ip
            "http://model-training:8401/training/datasets",  # Replace with Jetson IP or server IP
            #params={"model_id": model_id, "colour_corrected":colour_corrected, "ground_truth":ground_truth},
            #files={"file": uploaded_file}
        )
        if response.status_code == 200:
            result = response.json()
            st.json(result)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")


if st.button("Available Projects"):
    get_datasets()
