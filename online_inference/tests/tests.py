import unittest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from online_inference.app import app


class MyTestCase(unittest.TestCase):

    def test_health(self):
        with TestClient(app) as client:
            response = client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json())

    def test_predict(self):
        true_response = [
            [{'id': 0, 'target': 0}],
            [{'id': 1, 'target': 0}],
        ]

        with TestClient(app) as client:
            data = pd.read_csv('data/test_data.csv')
            data['id'] = data.index.values
            request_features = list(data.columns)
            request_data0 = [x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()]
            request_data1 = [x.item() if isinstance(x, np.generic) else x for x in data.iloc[1].tolist()]
            response0 = client.get(
                '/predict',
                json={'data': [request_data0], 'features': request_features},
            )
            response1 = client.get(
                '/predict',
                json={'data': [request_data1], 'features': request_features},
            )
            self.assertEqual(response0.status_code, 200)
            self.assertEqual(response1.status_code, 200)
            self.assertDictEqual(true_response[0][0], response0.json()[0])
            self.assertDictEqual(true_response[1][0], response1.json()[0])

    def test_bad_command(self):
        with TestClient(app) as client:
            response = client.get('/badcommand')
            self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
