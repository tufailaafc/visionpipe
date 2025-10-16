import requests
import streamlit as st
import base64
from PIL import Image
from io import BytesIO
#import pymongo
import pandas as pd



st.set_page_config(page_title="Mongo Data", layout="wide")
st.title("Mongo Data Viewer", width="stretch")
st.subheader("Dr. Muhammad Tufail, Nathaniel Yeo", width="stretch")



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

df = None

# Creating a button in order to request for a json object and display it to the user in a data frame.
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
                st.dataframe(df, use_container_width=True, height=800)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# if st.button("Fetch Images"):
#         try:
#             response = requests.post(
#                 #works because the backend is on the same docker network. If not replace (pig_sorting_api) with the server ip
#                 "http://pig-sorting-api:8001/api/v1/images/",  # Replace with Jetson IP or server IP
#                 params={"times": time},
#                 #files={"file": uploaded_file}
#             )

#             if response.status_code == 200:
#                 result = response.json()
#                 st.image(uploaded_file, caption="Original Image", use_column_width=True)
#             else:
#                 st.error(f"API Error: {response.status_code} - {response.text}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Connection error: {e}")


from datetime import datetime

start_date = st.date_input("Start date", value=datetime(2025, 7, 30))
end_date = st.date_input("End date", value=datetime(2025, 7, 30))

if st.button("Fetch Images"):
    try:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        #times = [start_date_str, end_date_str]
        times = [str(start_date), str(end_date)]
        response = requests.post(
            "http://pig-sorting-api:8001/api/v1/images/",
            json={"times": times}
        )

        if response.status_code == 200:
            images = response.json()
            # for img in images:
            #     st.image(
            #         Image.open(BytesIO(base64.b64decode(img["image"]))),
            #         caption=f"{img['filename']} - {img['timestamp']}",
            #         use_container_width=True
            #     )
            # Number of images per row
            cols_per_row = 4

            # Loop over images in chunks
            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, img_data in enumerate(images[i:i+cols_per_row]):
                    with cols[j]:
                        image = Image.open(BytesIO(base64.b64decode(img_data["image"])))
                        st.image(image, caption=f"{img_data['filename']}\n{img_data['timestamp']}", use_container_width='always')
    
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error fetching images: {e}")





