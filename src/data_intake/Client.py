from pymongo import MongoClient
from datetime import datetime
import os
"""
MongoDB connection setup for data-intake.

This module establishes a connection to the MongoDB server and initializes
the `visionPipe` database along with the `metaData` collection.

Environment Variables:
    MONGO_INITDB_ROOT_USERNAME (str): Username for the MongoDB root user.
    MONGO_INITDB_ROOT_PASSWORD (str): Password for the MongoDB root user.

Attributes:
    client (MongoClient): Active MongoDB client instance.
    db (Database): Reference to the `visionPipe` database.
    image_table (Collection): Reference to the `metaData` collection in the database.
"""





username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")


host = "mongo" #should match the image name in the docker-compose.yml
port = 27017 #port on which mongoDB is exposed

# Connect to MongoDB server
client = MongoClient(f"mongodb://{username}:{password}@{host}:{port}/")

# Connect to database
db = client["visionPipe"]

# Connect to the table
image_table = db["metaData"]





###############################################################
# import os
# from pymongo import MongoClient
# from pymongo.errors import ServerSelectionTimeoutError
# import time

# # Load credentials
# username = os.getenv("MONGO_APP_USER") or os.getenv("MONGO_INITDB_ROOT_USERNAME")
# password = os.getenv("MONGO_APP_PASS") or os.getenv("MONGO_INITDB_ROOT_PASSWORD")
# host = "mongo"
# port = 27017
# replica_set = "rs0"
# auth_source = "admin"

# uri = f"mongodb://{username}:{password}@{host}:{port}/?authSource={auth_source}&replicaSet={replica_set}"

# # Retry loop for initial connection
# max_retries = 10
# delay = 3

# for attempt in range(max_retries):
#     try:
#         client = MongoClient(uri, serverSelectionTimeoutMS=5000)
#         client.admin.command("ping")  # test connection
#         print("✅ Connected to MongoDB!")
#         break
#     except ServerSelectionTimeoutError:
#         print(f"⚠️ MongoDB not ready, retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
#         time.sleep(delay)
# else:
#     raise Exception("❌ Failed to connect to MongoDB after multiple attempts")

# # Connect to database and collection
# db = client["visionPipe"]
# db_result = db["metaData"]
