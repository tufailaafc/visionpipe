import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO



MIDDLEWARE_URL = "http://pig-sorting-api:8001"
MODEL_DEPLOY_URL = "http://model-deploy:8601"



st.set_page_config(page_title="AI-assisted Pigs Evaluation", layout="centered")
st.title("AI-assisted Pigs Evaluation")
st.subheader("Dr. Muhammad Tufail, Nathaniel Yeo")





#This will turn the selected camera_id into a form that we can use to tell the nvr which cameras we want to take pictures
def parse_channels(camera_ids):
    channels = set()
    for camera_id in camera_ids:
        channel_num = "".join(filter(str.isdigit, camera_id))
        channels.add(channel_num)
    return list(channels)


# --- Run chain prediction ---
def chain_prediction(uploaded_files, models_to_chain, return_annotated, confidence):
    """
    Sends request to perform prediction utilizing model(s) and sets the results in session state "prediction_results"

    :noindex:

    Parameters:
        uploaded_files   (list[UploadedFile]): A list of all the images
        models_to_chain  (list[Dict]):         A list of all the models to perform inference with order matters. starts with first element
        return_annotated (bool):               A flag to tell us if we should return a image, this image will have annotations
        confidence       (float):              This is the threshold for what classes we consider valid and will return 0.5 = 50%  

    Returns:
        None  
    """
    if st.button("Run Chain Prediction"):
        if not uploaded_files:
            st.error("Please upload at least one image.")
        elif not models_to_chain:
            st.error("Please select at least one model for chaining.")
        else:
            instances = []
            for file in uploaded_files:
                img_bytes = file.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                instances.append({"image": img_b64})

            payload = {
                "model_names": models_to_chain,
                "instances": instances,
                "parameters": {"return_annotated_image": return_annotated,
                            "confidence": confidence,}
            }

            with st.spinner("Running chain prediction..."):
                st.json(payload)
                try:
                    resp = requests.post(f"{MIDDLEWARE_URL}/chain_predict", json=payload)
                    resp.raise_for_status()
                    results = resp.json().get("predictions", [])
                    
                    # Store results in session state
                    st.session_state.prediction_results = results
                    
                except requests.exceptions.HTTPError as e:
                    st.error(f"Backend returned error: {e.response.status_code} - {e.response.text}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")



################################################Select Cameras and time frame############################################
# Need to be changed to not local host but the server ip when not running locally
mjpeg_url = "http://localhost:8801/stream.mjpeg"


st.title("Live DeepStream MJPEG Video")


def camera_select():
    # User picks from predefined camera IDs
    # The camera_id must have the corresponding nvr port number it is connected to in the id, 
    # if is port 3 then it must only have the number 3 in it.
    # This allows us to parse it easier.
    # camera_ids = st.multiselect("Choose cameras", ["cam1HighDef", "cam1LowDef","cam3HighDef", "cam3LowDef"])
    try:
        response = requests.post(f"{MIDDLEWARE_URL}/api/v1/collector/channels", 
                                json={"username": None,
                                    "password": None,
                                    "nvr_ip": None})

        camera_ids = {}
        # st.json(response.json())
        res_val_list = response.json()["channel_ids"]
        
        # for res in response.json().values():
        # st.dataframe(res_val_list)
        for res in res_val_list:
            camera_ids[f"cam{res}HighDef"] = res 
            camera_ids[f"cam{res}LowDef"] = res 
    
        camera_ids = st.multiselect("Choose cameras", camera_ids)


        return camera_ids
    # if camera_id and st.button("Connect to selected cameras"):
    #     for cam in camera_id:
    #         if cam:
    #             mjpeg_url = f"http://localhost:8801/stream/{cam}.mjpeg"
    #             st.image(mjpeg_url)
    except Exception as e:
        st.write(e)


def display_camera_stream(camera_ids):
    if camera_ids and st.button("Connect to selected cameras"):
        for cam in camera_ids:
            if cam:
                mjpeg_url = f"http://localhost:8801/stream/{cam}.mjpeg"
                st.image(mjpeg_url)


# Select how often you want it to take pictures
def get_user_frame_interval():
    frame_between_pic = st.number_input("How many frames between taking screenshots (The Lorex cameras operate at 15fps)",
                                        min_value=0,
                                        value=300)
    return frame_between_pic


def set_camera_interval(camera_ids, frame_between_pic):
    if st.button("Set Frame Interval for Selected Cameras"):
        if camera_ids:
            channels = parse_channels(camera_ids)
            errors = []
            for ch in channels:
                try:
                    response = requests.post(
                        f"{MIDDLEWARE_URL}/api/v1/collector/frame_interval",
                        json={"channel": ch, "interval": frame_between_pic}
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    errors.append(str(e))
            if not errors:
                st.success(f"Frame interval set to {frame_between_pic} for channels: {channels}")
            else:
                st.error(f"Errors occurred: {errors}")

# batch_size = st.number_input("How many pictures do you want to take for the selected camera?",
#                             min_value=0,
#                             value=5)

def trigger_picture(camera_ids, models_to_chain, return_annotated, confidence):
    #trigger picture once
    if st.button("Take one picture now"):
        if camera_ids:
            channels = parse_channels(camera_ids)
            print(f"channels: {channels}")

            #test declaration TOBE GOTTEN RID OF
            comment = "Testing for global comments"
            comments = None
            
            try:
                response = requests.post(
                    f"{MIDDLEWARE_URL}/api/v1/collector/capture",
                    json={"channels": channels,
                          "comment": comment,
                          "comments": comments}
                )
                response.raise_for_status()
                st.success(f"Took picture(s) for channels: {channels}")
                chain_prediction(response.images, models_to_chain, return_annotated, confidence)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to take picture: {e}")


def set_inference_variables():
    uploaded_files = st.file_uploader("Upload pig images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    # --- Choose models to chain ---
    models_to_chain = st.multiselect("Select models to chain (order matters)", st.session_state.models)

    # Confidence threshold slider
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # --- Annotated image toggle ---
    return_annotated = st.checkbox("Return annotated image", value=True)

    return uploaded_files, models_to_chain, return_annotated, confidence

def load_models():
    # Fetch available discovered models
    @st.cache_data
    def get_discovered_models():
        try:
            response = requests.get(f"{MODEL_DEPLOY_URL}/models/discover")
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

    st.session_state.models = model_keys


    if st.button("Load Selected Model"):
        with st.spinner("Loading model..."):
            try:
                res = requests.post(f"{MODEL_DEPLOY_URL}/model/load/discovered", json={"model_key": selected_model_key})
                res.raise_for_status()
                st.success(f"Model '{selected_model_key}' loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")



# Display results if they exist (either from current run or previous session state)
def display_prediction():
    """
    Looks in the session state for prediction_results and displays the results with the option to save.

    """
    if "prediction_results" in st.session_state and st.session_state.prediction_results != None:
        st.subheader("Prediction Results")
        st.json(st.session_state.prediction_results)
        # Add a clear results button
        if st.button("Clear Results"):
            st.session_state.prediction_results = None
            st.rerun()
        
        for i, res in enumerate(st.session_state.prediction_results):
            st.subheader(f"Image {i+1}")
            annotated_img_b64 = res.get("annotated_image")
            if annotated_img_b64:

                # --- Display chain results ---
                chain = res.get("chain", [])
                for step_idx, step in enumerate(chain):
                    st.markdown(f"**Step {step_idx+1}: Model `{step['model']}`**")
                    
                    detections = step.get("detections", [])
                    if detections:
                        for det in detections:
                            # Display Yolo Classification formatted predictions
                            label = det.get("class")
                            conf = det.get("confidence", 0)
                            if label != None:
                                st.write(f"- `{label}` (confidence: {conf:.2f})")

                            # Display DeepLabCut formatted predicitions
                            bodypart = det.get("bodypart") 
                            x_coord = det.get("x")
                            y_coord = det.get("y")
                            likelihood = det.get("likelihood")
                            if bodypart != None:
                                st.write(f"- `{bodypart}` (confidence: {likelihood:.2f}) `Coords(x,y)` ({x_coord:.2f},{y_coord:.2f})")

                    else:
                        st.write("_No detections_")


                img_bytes = base64.b64decode(annotated_img_b64)
                st.image(img_bytes, caption="Annotated Result", use_container_width=True)

                

                # Save button with unique key
                button_key = f"save_{i}_{hash(str(res))}"  # More unique key
                if st.button(f"💾 Save Image {i+1}", key=button_key):
                    save_payload = {
                        "annotated_image": annotated_img_b64,
                        "detections": sum([step["detections"] for step in res.get("chain", [])], []),
                        "metadata": {"models": [step["model"] for step in res.get("chain", [])]},
                    }
                    try:
                        save_resp = requests.post(f"{MIDDLEWARE_URL}/save_prediction", json=save_payload)
                        save_resp.raise_for_status()
                        st.success(f"Image {i+1} saved to DB at {save_resp.json().get('path')}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")


camera_ids = camera_select()
display_camera_stream(camera_ids)
frame_interval = get_user_frame_interval()
set_camera_interval(camera_ids, frame_interval)

# _, models_to_chain, return_annotated, confidence = set_inference_variables()

# trigger_picture(camera_ids, models_to_chain, return_annotated, confidence)

load_models()

uploaded_files, models_to_chain, return_annotated, confidence = set_inference_variables()

trigger_picture(camera_ids, models_to_chain, return_annotated, confidence)




chain_prediction(uploaded_files, models_to_chain, return_annotated, confidence)


display_prediction()




# if st.button("Start the batch job"):
#     st.write("Starting the batch job")







################################################confiugre models############################################
# Backend API base URL
# MODEL_DEPLOY_URL = "http://model-deploy:8601"

# st.title("YOLO Model Inference")

# # Fetch available discovered models
# @st.cache_data
# def get_discovered_models():
#     try:
#         response = requests.get(f"{MODEL_DEPLOY_URL}/models/discover")
#         response.raise_for_status()
#         return response.json().get("models", {})
#     except Exception as e:
#         st.error(f"Failed to fetch models: {e}")
#         return {}

# models = get_discovered_models()

# if not models:
#     st.warning("No models discovered in the backend.")
#     st.stop()

# if st.button("Refresh model list"):
#     models = get_discovered_models()

# # Model selection
# model_keys = list(models.keys())
# selected_model_key = st.selectbox("Select Model", model_keys)

# if st.button("Load Selected Model"):
#     with st.spinner("Loading model..."):
#         try:
#             res = requests.post(f"{MODEL_DEPLOY_URL}/model/load/discovered", json={"model_key": selected_model_key})
#             res.raise_for_status()
#             st.success(f"Model '{selected_model_key}' loaded successfully.")
#         except Exception as e:
#             st.error(f"Failed to load model: {e}")

# st.markdown("---")

# Upload image
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Confidence threshold slider
# confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Toggle for annotated image
# return_annotated = st.checkbox("Return Annotated Image", value=True)



########################################

# st.title("🐷 Pig Sorting Demo")

# Upload images




# uploaded_files, models_to_chain, return_annotated, confidence = set_inference_variables()






# chain_prediction(uploaded_files, models_to_chain, return_annotated, confidence)

# def inspect_types_streamlit(obj, indent=0):
#     """Recursively inspect the type of an object and display in Streamlit."""
#     prefix = "  " * indent

#     # Show the object type and value (for simple objects)
#     if isinstance(obj, (int, float, str, bool, type(None))):
#         st.write(f"{prefix}{type(obj).__name__}: {repr(obj)}")
#     else:
#         st.write(f"{prefix}{type(obj).__name__}")

#     # Handle nested structures
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             with st.expander(f"{prefix}Key ({type(key).__name__}): {repr(key)}", expanded=False):
#                 inspect_types_streamlit(value, indent + 1)

#     elif isinstance(obj, (list, tuple, set)):
#         for i, item in enumerate(obj):
#             with st.expander(f"{prefix}Index {i}", expanded=False):
#                 inspect_types_streamlit(item, indent + 1)




# display_prediction()



