import click
import pandas as pd
import logging
import pickle

from utils.read_config import read_params
from utils.metrics import calculate_metrics
from models.svm import fit_model, predict_model
from sklearn.model_selection import train_test_split



@click.command(name='train')
@click.argument('config_path')
def train(config_path: str):
    params = read_params(config_path)
    logging.basicConfig(filename=params.train_log_path, level='INFO')
    logger = logging.getLogger('Train')
    logger.info('read config at {}'.format(config_path))
    df = pd.read_csv(params.data_train_path)
    logger.info('read dataset at {}'.format(params.data_train_path))
    target = df.condition
    feature_names = params.features.num_features + params.features.cat_features
    split_params = params.split_params
    if split_params.stratify:
        x_train, x_valid, y_train, y_valid = train_test_split(df[feature_names], target,
                                                              train_size=split_params.train_size,
                                                              random_state=split_params.random_state,
                                                              shuffle=split_params.shuffle, stratify=target)
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(df[feature_names], target,
                                                              train_size=split_params.train_size,
                                                              random_state=split_params.random_state,
                                                              shuffle=split_params.shuffle)
    logger.info('Split dataset\n\t{}'.format(split_params))
    model = fit_model(x_train, y_train, params)

    logger.info('features\n\t{}'.format(params.features))
    logger.info('Preprocessing\n\t{}'.format(params.transformers))

    pred_train = predict_model(model, x_train, params)
    pred_valid = predict_model(model, x_valid, params)
    train_metrics = calculate_metrics(y_train, pred_train, params)
    valid_metrics = calculate_metrics(y_valid, pred_valid, params)
    logger.info('train metrics: \n\t{}'.format(train_metrics))
    logger.info('valid_metrics \n\t{}'.format(valid_metrics))

    with open(params.model_path, 'wb') as file:
        pickle.dump(model, file)
        logger.info('save model at {}'.format(params.model_path))


if __name__ == '__main__':
    train()
