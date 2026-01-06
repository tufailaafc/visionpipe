#Currently mostly depreciated. Instead use app.py

import streamlit as st
import requests
import base64
from PIL import Image
import io

# Backend API base URL
API_URL = "http://model-deploy:8601"

st.title("YOLO Model Inference")

# Fetch available discovered models
@st.cache_data
def get_discovered_models():
    try:
        response = requests.get(f"{API_URL}/models/discover")
        response.raise_for_status()
        return response.json().get("models", {})
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        return {}

models = get_discovered_models()

if not models:
    st.warning("No models discovered in the backend.")
    st.stop()

if st.button("Refresh model list"):
    models = get_discovered_models()

# Model selection
model_keys = list(models.keys())
selected_model_key = st.selectbox("Select Model", model_keys)

if st.button("Load Selected Model"):
    with st.spinner("Loading model..."):
        try:
            res = requests.post(f"{API_URL}/model/load/discovered", json={"model_key": selected_model_key})
            res.raise_for_status()
            st.success(f"Model '{selected_model_key}' loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Confidence threshold slider
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Toggle for annotated image
return_annotated = st.checkbox("Return Annotated Image", value=True)


# take the user uploaded image and prepare it
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "model_name": selected_model_key,
        "instances": [{"image": img_base64}],
        "parameters": {
            "confidence": confidence,
            "return_annotated_image": return_annotated
        }
    }




    # sends image to back end to perform inference
    if st.button("Run Inference"):

        # to display that something is happening
        with st.spinner("Running prediction..."):
            try:
                res = requests.post(f"{API_URL}/model/predict", json=payload)
                res.raise_for_status()
                result = res.json()

                # Display detections
                # for i, pred in enumerate(result["predictions"]):
                #     st.subheader(f"Detections (Image {i+1})")
                #     for det in pred["detections"]:
                #         st.write(f"- **{det['class']}** with {det['confidence']*100:.1f}% confidence")

                #     if return_annotated and "annotated_image" in pred:
                #         annotated_bytes = base64.b64decode(pred["annotated_image"])
                #         annotated_img = Image.open(io.BytesIO(annotated_bytes))
                #         st.image(annotated_img, caption="Annotated Image", use_column_width=True)
                for i, pred in enumerate(result["predictions"]):
                    st.subheader(f"Detections (Image {i+1})")

                    # NEW: iterate through chain
                    for step in pred["chain"]:
                        st.markdown(f"**Model:** `{step['model']}`")

                        for det in step["detections"]:
                            st.write(
                                f"- **{det['class']}** with {det['confidence']*100:.1f}% confidence"
                            )

                    if return_annotated and pred.get("annotated_image"):
                        annotated_bytes = base64.b64decode(pred["annotated_image"])
                        annotated_img = Image.open(io.BytesIO(annotated_bytes))
                        st.image(
                            annotated_img,
                            caption="Annotated Image",
                            use_column_width=True
                        )

            except Exception as e:
                st.error(f"Inference failed: {e}")
