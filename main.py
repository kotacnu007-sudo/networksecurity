from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from networksecurity.constant import training_pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataValidationConfig

import sys

if __name__=='__main__':
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.__dict__)
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact.__dict__)
        train_dataframe=DataValidation.read_data(data_ingestion_artifact.train_file_path)
        print(train_dataframe.columns.tolist())

        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Starting data validation")
        data_validation.initiate_data_validation()
        logging.info("Data validation completed successfully")
    except Exception as e:
        raise NetworkSecurityException(e, sys)