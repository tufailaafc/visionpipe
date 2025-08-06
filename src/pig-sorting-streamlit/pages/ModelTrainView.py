import streamlit as st
import requests

st.set_page_config(page_title="YOLO Trainer", layout="centered")

st.title("📦 Train YOLOv8 via FastAPI")


model_path = st.text_input("Model Path", value="yolo11n.pt")
dataset_path = st.text_input("Dataset YAML Path", value="coco8.yaml")
project_name = st.text_input("Project Name", value="my_project")
run_name = st.text_input("Run Name", value="run001")

# Sends request to pig-sorting-api to start the model training with the set params
if st.button("🚀 Start Training"):
    st.info("Sending request to backend...")
    try:
        url = "http://pig-sorting-api:8001/api/v1/training/trainer"
        params = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_name": project_name,
            "run_name": run_name
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        st.success("Training completed successfully!")

        # Display the results from the training
        st.json(res.json())
    except Exception as e:
        st.error(f"Failed to start training: {e}")
