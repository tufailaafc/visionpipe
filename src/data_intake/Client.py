from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Connect to database
db = client["visionPipe"]

# Connect to the table
images = db["metaData"]

# Example image metadata document for testing connection
image_metadata = {
    "filename": "photo1.jpg",
    "uploadDate": datetime.utcnow(),
    "format": "jpg",
    "dimensions": {"width": 1920, "height": 1080},
    "camera": {
        "make": "Canon",
        "model": "EOS 5D",
        "exposure": "1/200",
        "aperture": "f/2.8"
    },
    "gps": {"type": "Point", "coordinates": [-74.0060, 40.7128]},  # GeoJSON format
    "tags": ["vacation", "newyork", "summer"],
    "url": "https://your-storage.example.com/photo1.jpg"
}

# Insert the document into MongoDB
result = images.insert_one(image_metadata)
print(f"Inserted document id: {result.inserted_id}")