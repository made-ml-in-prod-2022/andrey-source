import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import NoReturn


@click.command('airflow-split')
@click.option('--train_path')
@click.option('--random_state', default=42)
@click.option('--test_size', default=0.25)
def split(train_path: str,  random_state: int, test_size: float) -> NoReturn:
    data = pd.read_csv(os.path.join(train_path, 'prepared_data.csv'))
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    os.makedirs(train_path, exist_ok=True)
    train.to_csv(os.path.join(train_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(train_path, 'test.csv'), index=False)


if __name__ == '__main__':
    split()
