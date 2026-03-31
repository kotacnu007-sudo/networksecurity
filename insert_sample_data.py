import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

client = pymongo.MongoClient(MONGO_DB_URL)
db = client["Network_Security"]
collection = db["NetworkData"]

sample_data = [
    {"src_ip": "192.168.1.1", "dst_ip": "192.168.1.2", "protocol": "TCP", "size": 1500},
    {"src_ip": "192.168.1.3", "dst_ip": "192.168.1.4", "protocol": "UDP", "size": 500}
]

collection.insert_many(sample_data)
print("Inserted sample records")
