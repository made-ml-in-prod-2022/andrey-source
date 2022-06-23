import os
import datetime
import pickle
import json
import click
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.command('airflow-validation')
@click.option('--train_path')
@click.option('--metrics_path')
@click.option('--model_path')
def valid(train_path: str, metrics_path: str, model_path: str):
    os.makedirs(metrics_path, exist_ok=True)
    model = load_model(os.path.join(model_path, 'model.pkl'))
    data = pd.read_csv(os.path.join(train_path, 'test.csv'))
    y = data.target.values
    x = data.drop(['target'], axis=1).values
    y_pred_proba = model.predict_proba(x)
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_pred_proba, average='macro', multi_class='ovr')
    metrics = {
        'Date': str(datetime.datetime.now()),
        'acc': acc,
        'f1': f1,
        'auc': auc
    }
    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)


def load_model(path: str) -> Pipeline:
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model


if __name__ == '__main__':
    valid()
