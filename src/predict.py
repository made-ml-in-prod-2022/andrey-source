import click
import pandas as pd
import logging
import pickle
from utils.read_config import read_params
from models.svm import predict_model


@click.command(name="predict")
@click.argument("config_path")
def predict(config_path: str):
    params = read_params(config_path)
    logging.basicConfig(filename=params.predict_log_path, level="INFO")
    logger = logging.getLogger("Predict")
    logger.info("read config at {}".format(config_path))
    df = pd.read_csv(params.data_test_path)
    logger.info("read dataset at {}".format(params.data_test_path))
    with open(params.model_path, "rb") as file:
        model = pickle.load(file)
        logger.info("load model at {}".format(params.model_path))
    features = params.features.num_features + params.features.cat_features
    pred = predict_model(model, df[features], params)
    pred_df = pd.DataFrame(pred)
    pred_df.to_csv(params.data_predict)
    logger.info("The forecast by address {}".format(params.data_test_path))


if __name__ == "__main__":
    predict()
