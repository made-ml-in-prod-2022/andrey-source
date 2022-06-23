import os
import click
from sklearn import datasets


@click.command('get_inference_data')
@click.option('--inference_path')
def generate_iris_data(inference_path: str):
    X, _ = datasets.load_iris(return_X_y=True, as_frame=True)
    os.makedirs(inference_path, exist_ok=True)
    X.to_csv(os.path.join(inference_path, 'inference.csv'))


if __name__ == '__main__':
    generate_iris_data()
