import yaml
from marshmallow_dataclass import class_schema
from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    cat_features: List[str]
    num_features: List[str]


@dataclass()
class SplitParams:
    train_size: float = field(default=0.75)
    random_state: int = field(default=42)
    stratify: bool = field(default=True)
    shuffle: bool = field(default=True)


@dataclass()
class OneHotEncoder:
    handle_unknown: str = field(default='ignore')
    custom_one_hot_encoder: bool = field(default=True)
    custom_C: float = field(default=1)


@dataclass()
class StandardScaler:
    with_mean: bool = field(default=True)
    with_std: bool = field(default=True)


@dataclass()
class CatTransformers:
    one_hot_encoder: OneHotEncoder


@dataclass()
class NumTransformers:
    standard_scaler: StandardScaler


@dataclass()
class Transformers:
    cat_transformers: CatTransformers
    num_transformers: NumTransformers


@dataclass()
class SVMParams:
    C: int = field(default=1)
    kernel: str = field(default='rbf')
    degree: int = field(default=3)
    random_state: int = field(default=42)
    class_weight: str = field(default='balanced')
    soft_classification: bool = field(default=True)


@dataclass()
class MetricsParams:
    accuracy: bool = field(default=True)
    recall: bool = field(default=True)
    precision: bool = field(default=True)
    roc_auc: bool = field(default=True)


@dataclass()
class Params:
    data_train_path: str
    data_test_path: str
    data_predict: str
    model_path: str
    train_log_path: str
    predict_log_path: str
    features: FeatureParams
    target: str
    split_params: SplitParams
    transformers: Transformers
    svm_params: SVMParams
    metrics: MetricsParams


def read_params(path: str):
    with open(path) as stream:
        model_schema = class_schema(Params)
        schema = model_schema()
        return schema.load(yaml.safe_load(stream))

