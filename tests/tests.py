import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.utils import read_params, calculate_metrics
from src.models import fit_model, predict_model, CustomOneHotEncoder


class TestProject(unittest.TestCase):

    def test_read_params(self):
        params = read_params('tests/data_test/config.yml')
        self.assertEqual(params.data_train_path, 'tests/data_test/heart_cleveland_upload.csv')
        self.assertEqual(params.data_test_path, 'tests/data_test/heart_cleveland_upload_test.csv')
        self.assertEqual(params.data_predict, 'tests/data_test/artifacts/submission.csv')
        self.assertEqual(params.model_path, 'tests/data_test/artifacts/svm_model.pkl')
        self.assertEqual(params.train_log_path, 'tests/data_test/artifacts/train_log.log')
        self.assertEqual(params.predict_log_path, 'tests/data_test/artifacts/predict_log.log')
        self.assertListEqual(params.features.cat_features, ['cp', 'slope', 'thal'])
        self.assertListEqual(params.features.num_features, ['age', 'sex', 'trestbps', 'chol', 'fbs',
                                                   'thalach', 'exang', 'oldpeak', 'ca'])
        self.assertEqual(params.target, 'condition')
        self.assertEqual(params.split_params.train_size, 0.75)
        self.assertEqual(params.split_params.random_state, 42)
        self.assertEqual(params.split_params.stratify, True)
        self.assertEqual(params.split_params.shuffle, True)
        self.assertEqual(params.transformers.cat_transformers.one_hot_encoder.custom_one_hot_encoder, True)
        self.assertEqual(params.transformers.cat_transformers.one_hot_encoder.custom_C, 1)
        self.assertEqual(params.transformers.cat_transformers.one_hot_encoder.handle_unknown, 'ignore')
        self.assertEqual(params.transformers.num_transformers.standard_scaler.with_mean, True)
        self.assertEqual(params.transformers.num_transformers.standard_scaler.with_std, True)
        self.assertEqual(params.svm_params.C, 1)
        self.assertEqual(params.svm_params.kernel, 'rbf')
        self.assertEqual(params.svm_params.degree, 3)
        self.assertEqual(params.svm_params.random_state, 42)
        self.assertEqual(params.svm_params.class_weight, 'balanced')
        self.assertEqual(params.svm_params.soft_classification, True)
        self.assertEqual(params.metrics.accuracy, True)
        self.assertEqual(params.metrics.recall, True)
        self.assertEqual(params.metrics.precision, True)
        self.assertEqual(params.metrics.roc_auc, True)
        with self.assertRaises(Exception):
            self.assertEqual(params.data_train_path, 'tests/dta/heart_clevelad_upload.csv')
            self.assertEqual(params.data_test_path, 'tests/daa/heart_cleveland_upload_test.cv')
            self.assertEqual(params.data_predict, 'tests/data_test/artfacts/submission.csv')
            self.assertEqual(params.model_path, 'tests/data_test/artfacts/svm_model.pkl')
            self.assertEqual(params.train_log_path, 'tests/data_test/artifacts/trai_log.log')
            self.assertEqual(params.predict_log_path, 'tests/data_test/artifats/predict_log.log')
            self.assertListEqual(params.features.cat_features, ['cp', 'thalach', 'slope', 'thal'])
            self.assertListEqual(params.features.num_features, ['age', 'sex', 'trestbps', 'chol', 'fbs',
                                                                'exang', 'oldpeak', 'ca'])
            self.assertEqual(params.target, 'condition')
            self.assertEqual(params.split_params.train_size, 0.7)
            self.assertEqual(params.split_params.random_state, 2)
            self.assertEqual(params.split_params.stratify, False)
            self.assertEqual(params.split_params.shuffle, 'True')
            self.assertEqual(params.transformers.cat_transformers.one_hot_encoder.handle_unknown, 'inore')
            self.assertEqual(params.transformers.num_transformers.standard_scaler.with_mean, 1)
            self.assertEqual(params.transformers.num_transformers.standard_scaler.with_std, False)
            self.assertEqual(params.svm_params.C, 10)
            self.assertEqual(params.svm_params.kernel, 'linear')
            self.assertEqual(params.svm_params.degree, 0)
            self.assertEqual(params.svm_params.random_state, 43)
            self.assertEqual(params.svm_params.class_weight, 'balancd')
            self.assertEqual(params.svm_params.soft_classification, 'True')
            self.assertEqual(params.metrics.accuracy, 0.5)
            self.assertEqual(params.metrics.recall, 'e')
            self.assertEqual(params.metrics.precision, 1)
            self.assertEqual(params.metrics.roc_auc, False)

    def test_fit_predict_model(self):
        params = read_params('tests/data_test/config.yml')
        df = pd.read_csv(params.data_train_path)
        feature_names = params.features.num_features + params.features.cat_features
        y_train = df.condition
        x_train = df[feature_names]
        params.svm_params.soft_classification = False
        model = fit_model(x_train, y_train, params)
        self.assertTrue(isinstance(model, Pipeline))
        hard_predict = predict_model(model, x_train, params)
        self.assertEqual(hard_predict.shape[0], x_train.shape[0])
        n_classes = len(np.unique(y_train))
        with self.assertRaises(Exception):
            self.assertEqual(hard_predict.shape[1], n_classes)
        params.svm_params.soft_classification = True
        model = fit_model(x_train, y_train, params)
        self.assertTrue(isinstance(model, Pipeline))
        soft_predict = predict_model(model, x_train, params)
        self.assertEqual(soft_predict.shape[0], x_train.shape[0])
        self.assertEqual(soft_predict.shape[1], n_classes)

    def test_calculate_metrics(self):
        params = read_params('tests/data_test/config.yml')
        df = pd.read_csv(params.data_train_path)
        feature_names = params.features.num_features + params.features.cat_features
        y_train = df.condition
        x_train = df[feature_names]
        model = fit_model(x_train, y_train, params)
        params.svm_params.soft_classification = True
        soft_predict = predict_model(model, x_train, params)
        metrics = calculate_metrics(y_train, soft_predict, params)
        roc_correct = np.abs(metrics['roc_auc'] - 0.963) / 0.963 < 0.05
        acc_correct = np.abs(metrics['accuracy'] - 0.909) / 0.909 < 0.05
        recall_correct = np.abs(metrics['recall'] - 0.891) / 0.891 < 0.05
        precision_correct = np.abs(metrics['precision'] - 0.91) / 0.91 < 0.05
        self.assertTrue(roc_correct)
        self.assertTrue(acc_correct)
        self.assertTrue(recall_correct)
        self.assertTrue(precision_correct)
        params.svm_params.soft_classification = False
        model = fit_model(x_train, y_train, params)
        hard_predict = predict_model(model, x_train, params)
        with self.assertRaises(Exception):
            metrics = calculate_metrics(y_train, hard_predict, params)
        params.metrics.roc_auc = False
        metrics = calculate_metrics(y_train, hard_predict, params)
        acc_correct = np.abs(metrics['accuracy'] - 0.906) / 0.906 < 0.05
        recall_correct = np.abs(metrics['recall'] - 0.898) / 0.898 < 0.05
        precision_correct = np.abs(metrics['precision'] - 0.898) / 0.898 < 0.05
        self.assertTrue(acc_correct)
        self.assertTrue(recall_correct)
        self.assertTrue(precision_correct)


    def test_CustomOneHotEncoder(self):
        params = read_params('tests/data_test/config.yml')
        df = pd.read_csv(params.data_train_path)
        cat_features = params.features.cat_features
        y_train = df.condition
        x_train = df[cat_features]
        custom_one_hot = CustomOneHotEncoder()
        base_one_hot = OneHotEncoder()
        base_pred = base_one_hot.fit_transform(x_train)
        custom_pred = custom_one_hot.fit_transform(x_train, y_train)
        self.assertTrue(base_pred.shape[1] > custom_pred.shape[1])
        self.assertEqual(custom_pred.shape[1], 6)


if __name__ == '__main__':
    unittest.main()