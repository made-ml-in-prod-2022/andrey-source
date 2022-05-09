import numpy as np
import pandas as pd

from typing import Dict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, params) -> Dict[str, float]:
    metrics = {}
    if params.metrics.roc_auc:
        if params.svm_params.soft_classification:
            metrics['roc_auc'] = np.round(roc_auc_score(y_true.values, y_pred[:, 1]), 3)
            y_pred = y_pred.argmax(axis=1)
        else:
            raise ValueError("for calculate roc_auc need model.soft_classification True")
    if params.metrics.accuracy:
        metrics['accuracy'] = np.round(accuracy_score(y_true, y_pred), 3)
    if params.metrics.recall:
        metrics['recall'] = np.round(recall_score(y_true, y_pred), 3)
    if params.metrics.precision:
        metrics['precision'] = np.round(precision_score(y_true, y_pred), 3)
        return metrics
