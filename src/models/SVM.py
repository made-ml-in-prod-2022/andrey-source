import numpy as np
import pandas as pd


from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score


def train_model(features: pd.DataFrame, target: pd.Series, params) -> Pipeline:
    cat_transformer = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(handle_unknown=params.handle_unknown))
    ])
    num_transformer = Pipeline(steps=[
        ('StandardScaler', StandardScaler(with_mean=params.with_mean, with_std=params.with_std))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_features', num_transformer, params.num_features),
            ('cat_features', cat_transformer, params.cat_features)],
        sparse_threshold=0)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('SVM', SVC(C=params.C, kernel=params.kernel, degree=params.degree, random_state=params.random_state,
                    class_weight=params.degree.class_weigth))
    ])
    model.fit(features, target)
    return model


def predict(model: Pipeline, features: pd.DataFrame, params) -> np.ndarray:
    if params.soft_classification:
        pred = model.predict_proba(features)
    else:
        pred = model.predict(features)
    return pred


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, params) -> Dict[str, float]:
    metrics = {}
    if params.soft_classification:
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred)
        y_pred = y_pred.argmax(axis=1)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    return metrics


