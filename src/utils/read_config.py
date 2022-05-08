import yaml
from marshmallow_dataclass import class_schema
from dataclasses import dataclass, field
from typing import List


@dataclass()
class SplitParams:
    train_size: int = field(default=0.75)
    random_state: int = field(default=42)
    stratify: bool = field(default=True)


@dataclass()
class CatTransformers:
    handle_unknown: str = field(default='ignore')


@dataclass()
class NumTransformers:
    with_mean: bool = field(default=True)
    with_std: bool = field(default=True)


@dataclass()
class FeatureParams:
    cat_features: List[str]
    num_features: List[str]
    target: List[str]
    split_params: SplitParams
    cat_transformer: CatTransformers
    num_transformer: NumTransformers


@dataclass()
class ModelParams:
    C: int = field(default=1)
    kernel: str = field(default='rbf')
    degree: int = field(default=3)
    random_state: int = field(default=42)
    class_weight: str = field(default='balanced')
    soft_classification: bool = field(default=True)


@dataclass()
class Params:
    data_path: str
    features: FeatureParams
    model: ModelParams


model_schema = class_schema(Params)
with open('../../configs/config.yml', 'r') as stream:
    schema = model_schema()
    test = schema.load(yaml.safe_load(stream))

print(test.features)

