import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataAnalysisClass:
    """
    A class to perform data analysis and linear regression.

    Attributes:
        data_path (str): The directory path where the data file is located.
        data_csv (str): The filename of the CSV data file.
        pickle_path (str): The dir path where the pickle file will be saved.
        pickle_model (str): The filename of the pickle model.
        bonus (bool): Flag indicating whether to include bonus features.
        alpha (float): The learning rate for gradient descent.
        iter (int): The number of iterations for gradient descent.
        theta (numpy.ndarray): The parameters for the linear regression model.
        losses (list): List to store the loss values during training.
        biases (list): List to store the bias values during training.
        weights (list): List to store the weight values during training.
    """
    def __init__(self, data_dir: str, csv_file: str, pickle_path: str,
                 pickle_model: str, bonus: bool = False,
                 learning_rate: float = 0.01, iterations: int = 1500):
        """
        Initializes the DataAnalysisClass.

        Args:
            data_dir (str): The directory path where the data file is located.
            csv_file (str): The filename of the CSV data file.
            pickle_path (str): Dir path where the pickle file will be saved.
            pickle_model (str): The filename of the pickle model.
            bonus (bool, optional): Flag indicating whether to include bonus.
            learning_rate (float, optional): Learning rate gradient descent.
            iterations (int, optional): Iterations for gradient descent.
        """
        self.data_path = data_dir
        self.data_csv = csv_file
        self.pickle_path = pickle_path
        self.pickle_model = pickle_model
        self.bonus = bonus
        self.alpha = learning_rate
        self.iter = iterations
        self.theta = np.zeros(2)
        self.__load_data()
        self.__standardize_data()
        self.__train_dataset()
        self.__save_model()
        self.__plot_data()
        if self.bonus:
            self.__plot_residuals()
            self.__calculate_precision()

    def __load_data(self) -> None:
        """
        Loads the data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.data_path + self.data_csv)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.data_path} not found") from e

        self.data = self.data.dropna().drop_duplicates()
        arr = self.data.to_numpy(copy=True)
        self.km = arr[:, 0]
        self.price = arr[:, 1]
        if len(self.km) < 2 or len(self.price) < 2:
            raise ValueError("Not enough data to train the model")

    def __standardize_data(self) -> None:
        """
        Standardizes the data by calculating mean
        and standard deviation of kilometers.
        """
        try:
            self.mean_km = np.mean(self.km)
            self.std_km = np.std(self.km)
            self.x_normalized = (self.km - self.mean_km) / self.std_km
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Division by zero") from e

    @staticmethod
    def __hypothesis(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calculates the hypothesis of the linear regression model.

        Args:
            theta (numpy.ndarray): Linear regression model.
            x (numpy.ndarray): The input features.

        Returns:
            numpy.ndarray: The predicted values.
        """
        return np.dot(x, theta)

    def __cost_function(self, theta: np.ndarray, x: np.ndarray,
                        y: np.ndarray) -> float:
        """
        Calculates the cost function for linear regression.

        Args:
            theta (numpy.ndarray): Linear regression model.
            x (numpy.ndarray): The input features.
            y (numpy.ndarray): The target values.

        Returns:
            float: The cost value.
        """
        try:
            m = len(y)
            return np.sum((self.__hypothesis(theta, x) - y) ** 2) / (2 * m)
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Division by zero") from e

    def __gradient_descent(self, theta: np.ndarray, x: np.ndarray,
                           y: np.ndarray, alpha: float = 0.01,
                           epochs: int = 1500) -> np.ndarray:
        """
        Performs gradient descent to optimize the parameters.

        Args:
            theta (numpy.ndarray): Theta values.
            x (numpy.ndarray): The input features.
            y (numpy.ndarray): The target values.
            alpha (float, optional): The learning rate. Defaults to 0.01.
            epochs (int, optional): The number of iterations. Defaults to 1500.

        Returns:
            numpy.ndarray: The optimized parameters.
        """
        m = len(y)
        try:
            for _ in range(epochs):
                hypothesis = self.__hypothesis(theta, x)
                loss = hypothesis - y
                gradient = x.T.dot(loss) / m
                theta = theta - alpha * gradient
                if np.linalg.norm(gradient) < 1e-6:
                    break
            return theta
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Division by zero") from e

    def __train_dataset(self) -> None:
        """
        Trains the linear regression model using gradient descent.
        """
        self.losses = []
        self.biases = []
        self.weights = []
        x = np.column_stack(
            (self.x_normalized, np.ones_like(self.x_normalized)))
        theta = np.zeros(2)

        for _ in range(self.iter):
            cost = self.__cost_function(theta, x, self.price)
            self.losses.append(cost)
            self.biases.append(theta[0])
            self.weights.append(theta[1])

            theta = self.__gradient_descent(theta, x, self.price,
                                            self.alpha, self.iter)
            if (len(self.losses) > 1
                    and abs(self.losses[-1] - self.losses[-2]) < 1e-6):
                break

        self.theta = theta

    def __save_model(self) -> None:
        """
        Saves the trained model parameters to a pickle file.
        """
        try:
            with open(self.pickle_path + self.pickle_model, 'wb') as f:
                data_to_save = {"theta": self.theta,
                                "mean_km": self.mean_km,
                                "std_km": self.std_km}
                pickle.dump(data_to_save, f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.pickle_path} not found") from e

    def __plot_data(self) -> None:
        """
        Plots the data points and the linear regression line.
        """
        plt.scatter(self.km, self.price, color='blue')
        plt.plot(
            self.km,
            self.__hypothesis(
                self.theta,
                np.column_stack((
                    self.x_normalized,
                    np.ones_like(self.km)))))
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear regression'])
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.show()

    def __plot_residuals(self) -> None:
        """
        Plots residuals to check for any patterns.
        """
        x = np.column_stack((self.x_normalized,
                             np.ones_like(self.x_normalized)))
        predictions = self.__hypothesis(self.theta, x)
        residuals = self.price - predictions

        plt.scatter(self.km, residuals, color='red')
        plt.axhline(y=0, color='blue', linestyle='--')
        plt.title('Residuals Plot')
        plt.xlabel('km')
        plt.ylabel('Residuals')
        plt.grid()
        plt.show()

    def __calculate_precision(self) -> None:
        """
        Calculates the precision of the model.
        """
        x = np.column_stack((self.x_normalized,
                             np.ones_like(self.x_normalized)))
        predictions = self.__hypothesis(self.theta, x)
        mae = np.mean(np.abs(predictions - self.price))
        mse = np.mean((predictions - self.price) ** 2)
        r_squared = 1 - (np.sum((predictions - self.price) ** 2)
                         / np.sum((self.price - np.mean(self.price)) ** 2))
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


if __name__ == "__main__":
    data_path = '/home/splix/Desktop/ft_linear_regression/csv_files/'
    data_csv = 'data.csv'
    pickle_file = '/home/splix/Desktop/ft_linear_regression/pickle_files/'
    pickle_model = 'model.pkl'
    bonus = True
    learning_rate = 0.01
    iterations = 500
    data_analysis = DataAnalysisClass(
        data_dir=data_path,
        csv_file=data_csv,
        pickle_path=pickle_file,
        pickle_model=pickle_model,
        bonus=bonus,
        learning_rate=learning_rate,
        iterations=iterations
    )
