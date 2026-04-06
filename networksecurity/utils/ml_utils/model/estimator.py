from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys
from networksecurity. exception. exception import NetworkSecurityException
from networksecurity.logging. logger import logging

class NetworkModel:
    def __init__(self,preprocessor, model):
        self.preprocessor=preprocessor
        self.model=model
    
    def predict(self,X):
        X = self.preprocessor.transform(X)
        return self.model.predict(X)
    
    def __str__(self) -> str:
        return f"{type(self.model).__name__}()"
    
    def save_model(self, model_dir: str = SAVED_MODEL_DIR, model_file_name: str = MODEL_FILE_NAME) -> None:
        try:
            model_file_path=os.path.join(model_dir,model_file_name)
            dir_path=os.path.dirname(model_file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(model_file_path, "wb") as file_obj:
                import pickle
                pickle.dump(self.model, file_obj)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e