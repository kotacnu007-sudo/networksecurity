import sys

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_accuracy = accuracy_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred, average="weighted")
        model_recall_score = recall_score(y_true, y_pred, average="weighted")
        model_f1_score = f1_score(y_true, y_pred, average="weighted")

        classification_metric_artifact = ClassificationMetricArtifact(
            accuracy=model_accuracy,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score
        )

        return classification_metric_artifact
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
 