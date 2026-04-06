import os, sys, pickle, yaml
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to file: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to file: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)  
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    report = {}
    try:
        n_samples = x_train.shape[0]
        n_classes = len(np.unique(y_train))

        for model_name, model in models.items():
            logging.info(f"Training {model_name} model")
            para = param.get(model_name, {})

            # ✅ Skip if only one class
            if n_classes < 2:
                logging.warning(f"Skipping {model_name}: only one class in training data.")
                report[model_name] = None
                continue

            # ✅ Skip if too few samples
            if n_samples < 2:
                logging.warning(f"Skipping {model_name}: too few samples ({n_samples}) to train.")
                report[model_name] = None
                continue

            # ✅ Dynamic CV folds (never exceed n_samples)
            cv_folds = min(3, n_samples)
            if para and n_samples >= cv_folds:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                gs = GridSearchCV(model, para, cv=cv, n_jobs=-1)
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(x_train, y_train)

            y_test_pred = best_model.predict(x_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        # ✅ Filter out None values
        valid_scores = {k: v for k, v in report.items() if v is not None}
        if not valid_scores:
            raise ValueError("No valid models were trained. Please check dataset size and class distribution.")

        return valid_scores

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    


