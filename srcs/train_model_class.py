import json
import os

import numpy as np
import pandas as pd
import tqdm
from matplotlib import animation

from srcs.plotter_class import PlottingClass


class LinearRegressionModel:
    """
    Class for training a linear regression model.

    Includes methods for loading and preprocessing data, fitting the model, saving and loading the model,
    making predictions, plotting data, calculating precision, and more.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1500,
                 stop_threshold: float = 1e-6, bonus: bool = False) -> None:
        """
        Initialize the training model with specified hyperparameters.

        Initializes the training model with the given learning rate, number of iterations, stop threshold,
        and bonus flag. Sets default values for other attributes like precision, paths, features, and theta.
        """
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

    def load_data(self, data_path: str, x_feature: str, y_feature: str) -> None:
        """
        Load and preprocess the training data.

        Loads the data from a CSV file located at the specified path, extracts the specified
        features for input (x_feature) and output (y_feature), and prepares the data for training.
        """
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

    def __load_data(self) -> None:
        """
        Load and preprocess the training data.

        Loads the data from a CSV file located at the specified path, removes any rows with missing
        values or duplicates, and extracts the features and target values. If there is not enough
        data to train the model, a ValueError is raised.

        Raises:
            FileNotFoundError: If the CSV file at the specified path is not found.
            ValueError: If there is not enough data to train the model or if a value error occurs during data processing.
        """
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

    def __normalize_features(self) -> None:
        """
        Normalize the features for training.

        Calculates the mean and standard deviation of the features, then normalizes the features
        by subtracting the mean and dividing by the standard deviation. If the standard deviation
        of the feature to predict is 0, a ZeroDivisionError is raised.

        Raises:
            ZeroDivisionError: If the standard deviation of the feature to predict is 0.
        """
        self.mean_x = np.mean(self.x)
        self.std_x = np.std(self.x)
        try:
            self.x_normalized = (self.x - self.mean_x) / self.std_x
            self.x_normalized = np.column_stack((self.x_normalized, np.ones_like(self.x_normalized)))
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Standard deviation of the feature to predict is 0") from e

    def load_model(self, json_path: str) -> None:
        """
        Load the model from the json file

        Args:
            json_path: Path to the json file

        Returns:
            None
        """
        self.json_path = json_path
        self.__load_model()

    def __load_model(self) -> None:
        """
        Load the model parameters from a JSON file.

        Loads the model parameters from a JSON file located at the specified path.
        If the file is not found, a FileNotFoundError is raised.

        Raises:
            FileNotFoundError: If the JSON file at the specified path is not found.
        """
        try:
            with open(self.json_path, 'r') as f:
                data_loaded = json.load(f)
            self.theta = np.array(data_loaded["theta"])
            self.mean_x = data_loaded["mean_x"]
            self.std_x = data_loaded["std_x"]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.json_path} not found") from e

    def save_model(self, json_path: str) -> None:
        """
        Save the model to a JSON file.

        Saves the model parameters (theta, mean_x, std_x) to a JSON file at the specified path.
        """
        self.json_path = json_path
        self.__save_model()

    def __save_model(self) -> None:
        """
        Save the model parameters to a JSON file.

        Saves the model parameters (theta, mean_x, std_x) to a JSON file at the specified path.
        If the directory for the file does not exist, it creates the directory.
        """
        try:
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

    def __save_animation(self, ani: animation.FuncAnimation) -> None:
        """
        Save the training animation as a GIF.

        Saves the training animation (ani) to a specified destination as a GIF file.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest = os.path.join(base_dir, 'srcs/training_animation.gif')
        self.plotter.save_animation(ani, dest)

    def fit(self) -> None:
        """
        Fit the linear regression model.

        Calls the private method __train_dataset() to train the linear regression model.
        """
        self.__train_dataset()

    def __train_dataset(self) -> None:
        """
        Train the linear regression model with the dataset.

        Performs the training of the linear regression model using the normalized dataset.
        It iteratively updates the model parameters (theta) based on the gradient descent algorithm.
        """
        self.theta = np.zeros(2)

        self.losses = []
        self.theta0_history = []
        self.theta1_history = []
        self.__initialize_plot()
        self.__initialize_progress_bar()

        self.ani = animation.FuncAnimation(self.plotter.fig, self.__update, frames=self.iter, repeat=False)
        self.__save_animation(self.ani)
        self.pbar.close()
        self.plotter.show()

    def __initialize_progress_bar(self) -> None:
        """
        Initialize the progress bar for training.

        Creates and initializes a progress bar for tracking the training progress of the linear regression model.
        """
        self.pbar = tqdm.tqdm(total=self.iter, desc="Training Model", position=0, leave=True)

    def __update(self, frame: int) -> iter:
        """
        Update the model parameters during training.

        Calculates the hypothesis, error, and gradient for updating the model parameters (theta)
        using the gradient descent algorithm. Updates the loss history and model parameter history,
        and triggers plot updates during training.
        """
        hypothesis = self.__hypothesis(self.theta, self.x_normalized)
        error = hypothesis - self.y
        gradient = np.dot(self.x_normalized.T, error) / len(self.y)
        self.theta -= self.alpha * gradient

        self.losses.append(self.__cost_function(hypothesis, self.y))
        self.theta0_history.append(self.theta[0])
        self.theta1_history.append(self.theta[1])

        self.plotter.update_plots(self.theta0_history, self.theta1_history, self.losses, hypothesis)
        self.pbar.update(1)

        if len(self.losses) > 1 and abs(self.losses[-1] - self.losses[-2]):
            self.ani.event_source.stop()
            self.iter = frame
            return

    @staticmethod
    def __hypothesis(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calculate the hypothesis for linear regression.

        Calculates the hypothesis values based on the model parameters (theta) and input features (x).
        """
        return np.dot(x, theta)

    @staticmethod
    def __cost_function(hypothesis: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the cost function for linear regression.

        Calculates the cost function value based on the hypothesis values and actual target values.
        """
        m = len(y)
        return np.sum((hypothesis - y) ** 2) / (2 * m)

    def __initialize_plot(self) -> None:
        """
        Initialize the plotting for the training process.

        Initializes the plots for visualizing the training progress of the linear regression model.
        """
        self.plotter.initialize_plots()

    def predict(self, x: list) -> list:
        """
        Predict the target values using the linear regression model.

        Given a list of input values (x), predicts the target values using the trained linear regression model.
        If the model has not been loaded, it loads the model from a JSON file before making predictions.
        Returns a list of predicted target values.
        Raises:
            ValueError: If the input values are not in the correct format or if there is an issue loading the model.
        """
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

    def plot_residuals(self) -> None:
        """
        Plot the residuals of the linear regression model.

        Calculates the residuals by comparing the actual target values (y) with the predicted values
        from the model. Plots the residuals against the input features (x) for visualization.
        """
        predictions = self.__hypothesis(self.theta, self.x_normalized)
        residuals = self.y - predictions
        self.plotter.plot_residuals(self.x, residuals, self.x_feature)

    def plot_data(self) -> None:
        """
        Plot the data points and the linear regression model.

        Calculates the hypothesis values using the model parameters and input features, then plots
        the actual data points along with the linear regression model's predictions for visualization.
        """
        hypothesis = self.__hypothesis(self.theta, self.x_normalized)
        self.plotter.plot_data(self.x, self.y, hypothesis, self.x_feature, self.y_feature)

    def calculate_precision(self) -> None:
        """
        Calculate the precision metrics of the linear regression model.

        Calculates and prints the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²)
        values to evaluate the model's performance. Updates the precision attribute with the R-squared value
        multiplied by 100 to represent the model's precision percentage.
        """
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


def premade_data() -> None:
    """
    Prepare and train the linear regression model with predefined data.

    Loads predefined data from a CSV file, sets up the model with specified hyperparameters,
    trains the model, and optionally saves the model, calculates precision metrics, and plots data.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'csv_files/data.csv')
    json_path = os.path.join(base_dir, 'json_files/model.json')
    bonus = True
    learning_rate = 0.01
    iterations = 1500
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
