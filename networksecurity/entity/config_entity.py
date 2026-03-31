from datetime import datetime
import os
from networksecurity.constant import training_pipeline
from networksecurity.constant.training_pipeline import TARGET_COLUMN, PIPELINE_NAME, ARTIFACT_DIR, FILE_NAME, TRAIN_FILE_NAME, TEST_FILE_NAME, DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR, DATA_INGESTION_INGESTED_DIR, DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

print(training_pipeline.ARTIFACT_DIR)
print(training_pipeline.PIPELINE_NAME)


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
        self.ingested_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR)
        self.train_file_name = training_pipeline.TRAIN_FILE_NAME
        self.test_file_name = training_pipeline.TEST_FILE_NAME
        self.train_file_path = os.path.join(self.ingested_dir, self.train_file_name)
        self.test_file_path = os.path.join(self.ingested_dir, self.test_file_name)
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.test_size = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.feature_store_file_path = os.path.join(self.feature_store_dir, "network_data.csv")