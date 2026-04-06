import os
import sys
from xml.parsers.expat import model

from sklearn.svm import SVC

from networksecurity. exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, log_loss, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
AdaBoostClassifier,
GradientBoostingClassifier,
RandomForestClassifier,
)
import numpy as np
from imblearn.over_sampling import SMOTE
import mlflow
mlflow.set_tracking_uri("file:./mlruns")


import dagshub

dagshub.init(repo_owner='kotacnu007-sudo', repo_name='networksecurity', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    def track_mlflow(self,best_model,classification_metric):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            accuracy = classification_metric.accuracy
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)   
            mlflow.log_metric("recall_score", recall_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self,x_train,y_train,x_test,y_test):
        models = {
        "Random Forest": RandomForestClassifier(verbose=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        "Logistic Regression": LogisticRegression(verbose=1),
        "AdaBoost": AdaBoostClassifier(),
        }
        params={
                "Decision Tree": {
                'criterion' : ['gini', 'entropy', 'log_loss'],
                # 'splitter': ['best','random' ],
                # 'max_features': ['sqrt','log2'],
                },

                "Random Forest":{
                # 'criterion' : ['gini', 'entropy', 'log_loss'],

                # 'max_features': ['sqrt','log2' , None],
                'n_estimators': [8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                # 'loss': ['log_loss', 'exponential' ],
                'learning_rate': [.1,.01,.05,.001],
                'subsample' : [0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion' : ['squared_error', 'friedman_mse'],
                # 'max_features' : ['auto','sqrt' , 'log2'],
                'n_estimators': [8,16,32,64,128,256]
                },

                "Logistic Regression": {},
                "AdaBoost":{
                'learning_rate': [.1,.01,0.5,.001],
                'n_estimators': [8,16,32,64,128,256]}
                }

        model_report: dict = evaluate_models(x_train, y_train, x_test, y_test, models=models, param=params)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = [model_name for model_name, score in model_report.items() if score == best_model_score][0]
        best_model = models[best_model_name]

        logging.info(f"Best found model on both training and testing dataset is: {best_model_name}")

        #training the best model
        best_model.fit(x_train, y_train)

        #calculating metric on training and testing data
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)
        classification_train_metric_artifact = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        ##Track the mlflow metric for training data
        classification_test_metric_artifact = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_train_metric_artifact)
        self.track_mlflow(best_model, classification_test_metric_artifact)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        Network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)

        model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                best_model_name=best_model_name,
                best_model_score=best_model_score,
                train_metric_artifact=classification_train_metric_artifact,
                test_metric_artifact=classification_test_metric_artifact
            )


        logging.info(f"Model trainer artifact: {model_trainer_artifact}")            
        return model_trainer_artifact
   

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # ✅ Fallback checks
            if x_train.shape[0] < 2:
                logging.warning("Training set too small. Switching to anomaly detection.")
                from sklearn.svm import OneClassSVM
                model = OneClassSVM(gamma="auto")
                model.fit(x_train)

                save_object(self.model_trainer_config.trained_model_file_path, model)
                #model pusher
                save_object("final_model/model.pkl", model)


                return ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    best_model_name="OneClassSVM",
                    best_model_score=None  # anomaly detection doesn’t use accuracy
                )

            if len(np.unique(y_train)) < 2:
                logging.warning("Only one class present. Switching to anomaly detection.")
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(random_state=42)
                model.fit(x_train)

                save_object(self.model_trainer_config.trained_model_file_path, model)

                return ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    best_model_name="IsolationForest",
                    best_model_score=None
                )

            # ✅ Normal path: apply SMOTE and train classifiers
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
