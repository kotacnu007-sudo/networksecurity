import os
import sys

from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity. components.data_ingestion import DataIngestion
from networksecurity. components.data_validation import DataValidation
from networksecurity. components.data_transformation import DataTransformation
from networksecurity. components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

class TrainingPipeline:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.training_pipeline_config=training_pipeline_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion_config=DataIngestionConfig(
                root_dir=self.training_pipeline_config.root_dir,
                source_URL=self.training_pipeline_config.source_URL,
                local_data_file=self.training_pipeline_config.local_data_file,
                unzip_dir=self.training_pipeline_config.unzip_dir
            )
            data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config=DataValidationConfig(
                root_dir=self.training_pipeline_config.root_dir,
                report_file_path=self.training_pipeline_config.report_file_path,
                missing_threshold=self.training_pipeline_config.missing_threshold
            )
            data_validation=DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation_config=DataTransformationConfig(
                root_dir=self.training_pipeline_config.root_dir,
            )
            data_transformation=DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")
            model_trainer_config=ModelTrainerConfig(
                root_dir=self.training_pipeline_config.root_dir,
                trained_model_file_path=self.training_pipeline_config.trained_model_file_path,
                expected_score=self.training_pipeline_config.expected_score,
                overfitting_underfitting_threshold=self.training_pipeline_config.overfitting_underfitting_threshold
            )
            model_trainer=ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initiate_model_trainer()   
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def sync_artifact_dir_to_s3(self):
        try:
            logging.info("Syncing artifact directory to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
            logging.info("Artifact directory synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def sync_saved_model_dir_to_s3(self):
        try:
            logging.info("Syncing saved model directory to S3")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/saved_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.saved_model_dir, aws_bucket_url=aws_bucket_url)
            logging.info("Saved model directory synced to S3 successfully")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e