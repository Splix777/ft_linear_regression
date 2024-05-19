import json
import os
import unittest
import warnings
from unittest.mock import MagicMock, patch

import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from srcs.train_model_class import LinearRegressionModel
from srcs.linear_regression_predictor import LinearRegressionPredictor

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class TestLinearRegression(unittest.TestCase):
    """
    A class for testing the linear regression model and predictor.
    """
    def setUp(self) -> None:
        """
        Set up test data and objects.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.json = os.path.join(base_dir, "json_files/model.json")
        self.data_path = os.path.join(base_dir, "csv_files/data.csv")
        data = pd.read_csv(self.data_path)
        self.x_feature = 'km'
        self.y_feature = 'price'
        self.km = data[self.x_feature].values
        self.price = data[self.y_feature].values

        if not os.path.exists(self.json):
            self.data_analysis = LinearRegressionModel()
            self.data_analysis.load_data(self.data_path, self.x_feature, self.y_feature)
            self.data_analysis.fit()
            self.data_analysis.save_model(self.json)

        self.predictor = LinearRegressionPredictor(self.json)
        self.sklearn_model = LinearRegression().fit(
            np.array(self.km).reshape(-1, 1),
            np.array(self.price).reshape(-1, 1))

        self.km_list = list(range(0, 1_000_000, 10_000))

    def test_custom_model(self) -> None:
        """
        Test the custom linear regression model against sklearn.
        """
        for km in self.km_list:
            with self.subTest(km=km):
                custom_model = self.predictor.predict(km)
                sklearn_model = self.sklearn_model.predict(
                    np.array([[km]])).flatten()[0]
                np.testing.assert_almost_equal(
                    custom_model,
                    sklearn_model,
                    decimal=1)

        print("My model and sklearn model are equal within 1 decimal place.")

    def test_error_handling(self) -> None:
        """
        Test error handling of predictor.
        """
        with self.assertRaises(ValueError):
            self.predictor.predict(-1)
        with self.assertRaises(ValueError):
            self.predictor.predict(1_000_001)
        with self.assertRaises(ValueError):
            self.predictor.predict('a')

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        pass


if __name__ == '__main__':
    unittest.main()
