import os
import click
import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@click.command('airflow-predict')
@click.option('--inference_path')
@click.option('--prediction_path')
@click.option('--model_path')
def predict(inference_path: str, prediction_path: str, model_path: str):
    os.makedirs(prediction_path, exist_ok=True)
    model = load_model(os.path.join(model_path, 'model.pkl'))
    data = pd.read_csv(os.path.join(inference_path, 'inference.csv'), index_col=0)
    pred = model.predict(data.values)
    np.savetxt(os.path.join(prediction_path, 'prediction.csv'), pred, delimiter=',')


def load_model(path: str) -> Pipeline:
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model


if __name__ == '__main__':
    predict()
