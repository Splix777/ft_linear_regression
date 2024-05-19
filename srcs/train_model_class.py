import json
import os

import numpy as np
import pandas as pd
import tqdm
from matplotlib import animation

from srcs.plotter_class import PlottingClass


class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, iterations=1500, stop_threshold=1e-6, bonus=False):
        self.precision = None
        self.json_path = None
        self.plotter = None
        self.y_feature = None
        self.x_feature = None
        self.data_path = None
        self.alpha = learning_rate
        self.iter = iterations
        self.stop_threshold = stop_threshold
        self.bonus = bonus
        self.theta = np.zeros(2)

    def load_data(self, data_path, x_feature, y_feature):
        self.data_path = data_path
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.__load_data()
        self.__normalize_features()
        self.plotter = PlottingClass(x_feature=self.x, x_name=x_feature,
                                     y_feature=self.y, y_name=y_feature,
                                     learning_rate=self.alpha,
                                     iterations=self.iter,
                                     stop_threshold=self.stop_threshold)

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.data_path} not found") from e

        try:
            self.data = self.data.dropna().drop_duplicates()
            self.x = self.data[self.x_feature].values.astype(float)
            self.y = self.data[self.y_feature].values.astype(float)
        except ValueError as e:
            raise ValueError(e) from e

        if len(self.x) < 2 or len(self.y) < 2:
            raise ValueError("Not enough data to train the model")

    def __normalize_features(self):
        self.mean_x = np.mean(self.x)
        self.std_x = np.std(self.x)
        self.x_normalized = (self.x - self.mean_x) / self.std_x
        self.x_normalized = np.column_stack((self.x_normalized, np.ones_like(self.x_normalized)))

    def load_model(self, json_path):
        self.json_path = json_path
        self.__load_model()

    def __load_model(self):
        try:
            with open(self.json_path, 'r') as f:
                data_loaded = json.load(f)
            self.theta = np.array(data_loaded["theta"])
            self.mean_x = data_loaded["mean_x"]
            self.std_x = data_loaded["std_x"]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.json_path} not found") from e

    def save_model(self, json_path):
        self.json_path = json_path
        self.__save_model()

    def __save_model(self):
        try:
            # Check if directory exists. If not, create it
            if not os.path.exists(os.path.dirname(self.json_path)):
                os.makedirs(os.path.dirname(self.json_path))
            data_to_save = {
                "theta": self.theta.tolist(),
                "mean_x": self.mean_x,
                "std_x": self.std_x
            }
            with open(self.json_path, 'w') as f:
                json.dump(data_to_save, f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.json_path} not found") from e

    def __save_animation(self, ani):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest = os.path.join(base_dir, 'srcs/training_animation.gif')
        self.plotter.save_animation(ani, dest)

    def fit(self):
        self.__train_dataset()

    def __train_dataset(self):
        m = len(self.y)
        x = self.x_normalized
        y = self.y
        theta = np.zeros(2)

        self.losses = []
        self.theta0_history = []
        self.theta1_history = []
        self.__initialize_plot()
        self.pbar = tqdm.tqdm(total=self.iter, desc="Training Model", position=0, leave=True)

        def update(frame) -> iter:
            nonlocal theta

            hypothesis = self.__hypothesis(theta, x)
            error = hypothesis - y
            gradient = np.dot(x.T, error) / m
            theta -= self.alpha * gradient

            self.losses.append(self.__cost_function(hypothesis, y))
            self.theta0_history.append(theta[0])
            self.theta1_history.append(theta[1])

            self.plotter.update_plots(self.theta0_history,
                                      self.theta1_history,
                                      self.losses, hypothesis)

            self.pbar.update(1)

            if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]):
                self.ani.event_source.stop()
                self.iter = frame
                return

        self.ani = animation.FuncAnimation(self.plotter.fig, update, frames=self.iter, repeat=False)
        self.__save_animation(self.ani)
        self.pbar.close()
        self.plotter.show()
        self.theta = theta

    @staticmethod
    def __hypothesis(theta, x):
        return np.dot(x, theta)

    @staticmethod
    def __cost_function(hypothesis, y):
        m = len(y)
        return np.sum((hypothesis - y) ** 2) / (2 * m)

    def __initialize_plot(self):
        self.plotter.initialize_plots()

    def predict(self, x: list) -> list:
        if not isinstance(x, list):
            raise ValueError("Please enter a list of values to predict")
        for value in x:
            if not isinstance(value, (int, float)):
                raise ValueError("Please enter a list of numbers to predict")
        # Load the model if it hasn't been loaded
        if self.theta[0] == 0 and self.theta[1] == 0:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(base_dir, 'json_files/model.json')
            self.load_model(json_path)
        results = []
        for i in range(len(x)):
            result = (self.theta[0] * (x[i] - self.mean_x) / self.std_x + self.theta[1])
            x[i] = round(result, 2)
            results.append(x[i])
        return results

    def plot_residuals(self):
        predictions = self.__hypothesis(self.theta, self.x_normalized)
        residuals = self.y - predictions
        self.plotter.plot_residuals(self.x, residuals, self.x_feature)

    def plot_data(self):
        hypothesis = self.__hypothesis(self.theta, self.x_normalized)
        self.plotter.plot_data(self.x, self.y, hypothesis, self.x_feature, self.y_feature)

    def calculate_precision(self):
        predictions = self.__hypothesis(self.theta, self.x_normalized)
        mae = np.mean(np.abs(predictions - self.y))
        mse = np.mean((predictions - self.y) ** 2)
        r_squared = 1 - (np.sum((predictions - self.y) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2))
        self.precision = r_squared * 100

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R²): {r_squared}")
        if self.precision < 70:
            print(f"Model Precision (R²): {self.precision:.2f}%")
            print("The model is not accurate enough")
        else:
            print(f"Model Precision (R²): {self.precision:.2f}%")
            print("Anything above 70% is considered a good model")


def premade_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'csv_files/data.csv')
    json_path = os.path.join(base_dir, 'json_files/model.json')
    bonus = True
    learning_rate = 0.01
    iterations = 500
    stop_threshold = 1e-6
    x_feature = 'km'
    y_feature = 'price'

    model = LinearRegressionModel(learning_rate=learning_rate, iterations=iterations, stop_threshold=stop_threshold,
                                  bonus=bonus)
    model.load_data(data_path=data_path, x_feature=x_feature, y_feature=y_feature)
    model.fit()
    model.save_model(json_path=json_path)
    model.calculate_precision()
    model.plot_residuals()
    model.plot_data()

    predictions = [240000, 139800, 150500, 185530, 176000]
    print(model.predict(predictions))


if __name__ == "__main__":
    premade_data()
