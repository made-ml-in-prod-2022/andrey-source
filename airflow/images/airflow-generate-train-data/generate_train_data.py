import os
import numpy as np
import click
from sklearn import datasets


@click.command('get_train_data')
@click.option('--train_path')
def generate_iris_data(train_path: str):
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    nan_mask = np.random.random(X.shape) < 0.05
    X[nan_mask] = np.nan
    os.makedirs(train_path, exist_ok=True)
    X.to_csv(os.path.join(train_path, 'data.csv'))
    y.to_csv(os.path.join(train_path, 'target.csv'))


if __name__ == '__main__':
    generate_iris_data()
