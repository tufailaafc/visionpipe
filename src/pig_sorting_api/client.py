"""
Asynchronous MongoDB client module using Motor.

This module sets up an asynchronous connection to a MongoDB instance using
credentials from environment variables and exposes:

- `client`: The AsyncIOMotorClient instance connected to the MongoDB server.
- `db`: The `visionPipe` database.
- `db_result`: The `metaData` collection within the `visionPipe` database.

Environment variables required:
- MONGO_INITDB_ROOT_USERNAME
- MONGO_INITDB_ROOT_PASSWORD

Usage:
    from client_async import db_result

    # Example asynchronous query
    result = await db_result.find_one({"some_field": "value"})
"""
# from pymongo import MongoClient
# #from dotenv import load_dotenv
# import os




# #load_dotenv()

# #get credentials from .env
# username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
# password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")


# host = "mongo" #should match the image name in the docker-compose.yml
# port = 27017 #port on which mongoDB is exposed

# uri = f"mongodb://{username}:{password}@{host}:{port}/"

# #connect to the server
# client = MongoClient(uri)

# # #connect to/create the database
# # db = client["meat_evaluation"]

# # #connect to/create the table
# # db_result = db["result"]

# # Connect to database
# db = client["visionPipe"]

# # Connect to the table
# db_result = db["metaData"]


# #def saveResult(result):
# #	result = result.insert_one(result)

# client_async.py
import os
from motor.motor_asyncio import AsyncIOMotorClient

username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")

host = "mongo"
port = 27017

uri = f"mongodb://{username}:{password}@{host}:{port}/"

# Connect to MongoDB asynchronously
client = AsyncIOMotorClient(uri)

# Connect to database
db = client["visionPipe"]

# Connect to collection
db_result = db["metaData"]

