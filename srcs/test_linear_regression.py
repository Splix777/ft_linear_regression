import unittest
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from train_model_class import DataAnalysisClass
from linear_regression_predictor import LinearRegressionPredictor

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class TestLinearRegression(unittest.TestCase):
    """
    A class for testing the linear regression model and predictor.
    """
    def setUp(self) -> None:
        """
        Set up test data and objects.
        """
        self.data_path = '/home/splix/Desktop/ft_linear_regression/csv_files/'
        self.data_csv = 'data.csv'
        self.pkl_pth = '/home/splix/Desktop/ft_linear_regression/pickle_files/'
        self.pickle_model = 'model.pkl'
        self.bonus = False
        self.learning_rate = 0.1
        self.iterations = 100
        self.data_analysis = DataAnalysisClass(
            data_dir=self.data_path,
            csv_file=self.data_csv,
            pickle_path=self.pkl_pth,
            pickle_model=self.pickle_model,
            bonus=self.bonus,
            learning_rate=self.learning_rate,
            iterations=self.iterations
        )
        self.predictor = LinearRegressionPredictor(
            self.pkl_pth + self.pickle_model)
        self.km_list = list(range(0, 1_000_000, 10_000))

    def test_custom_model(self) -> None:
        """
        Test the custom linear regression model against sklearn.
        """
        for km in self.km_list:
            with self.subTest(km=km):
                custom_model = self.predictor.predict(km)

                reg = LinearRegression().fit(
                    np.array(self.data_analysis.km).reshape(-1, 1),
                    np.array(self.data_analysis.price).reshape(-1, 1))
                sklearn_model = reg.predict(np.array([[km]]))[0][0]
                np.testing.assert_almost_equal(
                    custom_model,
                    sklearn_model,
                    decimal=1)

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
