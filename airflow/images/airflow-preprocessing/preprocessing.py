import os
import click
import pandas as pd


@click.command('preprocessor')
@click.option('--train_path')
def preprocess_data(train_path: str):
    X = pd.read_csv(os.path.join(train_path, 'data.csv'), index_col=0)
    y = pd.read_csv(os.path.join(train_path, 'target.csv'), index_col=0)
    data = X.merge(y, right_index=True, left_index=True)
    data = data.dropna()
    data.to_csv(os.path.join(train_path, 'prepared_data.csv'), index=False)


if __name__ == '__main__':
    preprocess_data()

