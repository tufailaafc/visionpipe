import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO
import pymongo
import pandas as pd


# Initialize connection
# Use st.cache_resource to only run once.

# @st.cache_resource
# def init_connection():
#     return pymongo.MongoClient(**st.secrets["mongo"])

# client = init_connection()


# Use st.cache_data to only rerun when the query changes or after 600 seconds.
# @st.cache_data(ttl=600)
# def get_data():
#     db = client.visionPipe
#     items = db.metaData.find()
#     items = list(items) # Make hashable for st.cache_data
#     return items

# items = get_data()

# for item in items:
#     st.write(f"{item}")


if st.button("Load Data from MongoDB"):
        try:
            response = requests.get(
                #works because the backend is on the same docker network. If not replace (pig_sorting_api) with the server ip
                "http://pig-sorting-api:8001/api/v1/mongoData",  # Replace with Jetson IP or server IP
                #params={"model_id": model_id, "colour_corrected":colour_corrected, "ground_truth":ground_truth},
                #files={"file": uploaded_file}
            )

            if response.status_code == 200:
                result = response.json()
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
                #st.image(uploaded_file, caption="Original Image", use_column_width=True)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

