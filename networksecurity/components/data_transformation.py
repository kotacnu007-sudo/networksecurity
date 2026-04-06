import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from imblearn.over_sampling import SMOTE

from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS, TARGET_COLUMN
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_columns = ["size"]
            categorical_columns = ["src_ip", "dst_ip"]

            num_pipeline = Pipeline(steps=[
                ("knn_imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # ✅ Load validated train and test files
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            full_df = pd.concat([train_df, test_df], axis=0)

            X = full_df.drop(columns=[TARGET_COLUMN])
            y = full_df[TARGET_COLUMN]

            # ✅ Check class counts
            class_counts = y.value_counts()
            logging.info(f"Class distribution before split: {class_counts.to_dict()}")

            if class_counts.min() < 2:
                logging.warning("One class has fewer than 2 samples. Falling back to simple split.")
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                for train_idx, test_idx in sss.split(X, y):
                    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # ✅ Apply preprocessing pipeline
            preprocessor = self.get_data_transformer_object()
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)

            # ✅ Apply SMOTE only on transformed training set
            if len(np.unique(y_train)) > 1 and x_train_transformed.shape[0] >= 2:
                smote = SMOTE(random_state=42)
                x_train_transformed, y_train = smote.fit_resample(x_train_transformed, y_train)

            # ✅ Save arrays
            train_arr = np.c_[x_train_transformed, y_train]
            test_arr = np.c_[x_test_transformed, y_test]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # ✅ Save the transformer object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info(f"Train set size after SMOTE: {train_arr.shape[0]}")
            logging.info(f"Class distribution after SMOTE: {np.unique(y_train, return_counts=True)}")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
