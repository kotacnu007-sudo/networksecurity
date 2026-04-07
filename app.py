import sys
import os
import certifi

from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]  # You can specify allowed origins here, e.g., ["http://localhost:3000"] for a React app running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        config = TrainingPipelineConfig()   
        pipeline = TrainingPipeline(training_pipeline_config=config)  
        pipeline.run_pipeline()
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    

@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Load the trained model
        preprocessor_path = "final_model/preprocessor.pkl"
        preprocessor = load_object(file_path=preprocessor_path)

        model_path = "final_model/model.pkl"  # Update with your actual model path
        model = load_object(file_path=model_path)

        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df["predicted_label"] = y_pred
        print(df['predicted_label'])
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(index=False,classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e



if __name__=="__main__":
    app_run(app, host="localhost", port=8000)
