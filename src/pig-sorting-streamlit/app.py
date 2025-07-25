import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AI-assisted Pigs Evaluation", layout="centered")
st.title("AI-assisted Pigs Evaluation")
st.subheader("Dr. Muhammad Tufail, Nathaniel Yeo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    if st.button("Predict"):
        try:
            response = requests.post(
                #works because the backend is on the same docker network. If not replace (pig_sorting_api) with the server ip
                "http://pig_sorting_api:8001/api/v1/predict/",  # Replace with Jetson IP or server IP
                #params={"model_id": model_id, "colour_corrected":colour_corrected, "ground_truth":ground_truth},
                files={"file": uploaded_file}
            )

            if response.status_code == 200:
                result = response.json()
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
