import os
import click
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@click.command('airflow-train')
@click.option('--train_path')
@click.option('--model_path')
def train(train_path: str, model_path: str):
    data = pd.read_csv(os.path.join(train_path, 'train.csv'))
    y = data.target.values
    x = data.drop(['target'], axis=1).values
    model = Pipeline([
        ('SS', StandardScaler()),
        ('LR', LogisticRegression())
    ])
    model.fit(x, y)
    os.makedirs(model_path, exist_ok=True)
    path = os.path.join(model_path, 'model.pkl')
    with open(path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train()
