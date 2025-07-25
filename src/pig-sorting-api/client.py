from pymongo import MongoClient
from dotenv import load_dotenv
import os




load_dotenv()

#get credentials from .env
username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")


host = "mongo" #should match the image name in the docker-compose.yml
port = 27017 #port on which mongoDB is exposed

uri = f"mongodb://{username}:{password}@{host}:{port}/"

#connect to the server
client = MongoClient(uri)

#connect to/create the database
db = client["meat_evaluation"]

#connect to/create the table
db_result = db["result"]


#def saveResult(result):
#	result = result.insert_one(result)

