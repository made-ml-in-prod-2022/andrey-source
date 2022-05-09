import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fit_model(features: pd.DataFrame, target: pd.Series, params) -> Pipeline:
    handle_unknown = params.transformers.cat_transformers.one_hot_encoder.handle_unknown
    with_mean = params.transformers.num_transformers.standard_scaler.with_mean
    with_std = params.transformers.num_transformers.standard_scaler.with_std
    feature_params = params.features
    cat_transformer = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(handle_unknown=handle_unknown))
    ])
    num_transformer = Pipeline(steps=[
        ('StandardScaler', StandardScaler(with_mean=with_mean, with_std=with_std))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_features', num_transformer, feature_params.num_features),
            ('cat_features', cat_transformer, feature_params.cat_features)],
        sparse_threshold=0)

    model_params = params.svm_params
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('SVM', SVC(C=model_params.C, kernel=model_params.kernel, degree=model_params.degree, random_state=model_params.random_state,
                    class_weight=model_params.class_weight, probability=model_params.soft_classification))
    ])
    model.fit(features, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame, params) -> np.ndarray:
    if params.svm_params.soft_classification:
        pred = model.predict_proba(features)
    else:
        pred = model.predict(features)
    return pred





