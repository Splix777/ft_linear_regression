import os
import pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataAnalysisClass:
    """
    A class to perform data analysis and linear regression.

    Attributes:
        data_path (str): The directory path where the data file is located.
        pickle_path (str): The dir path where the pickle file will be saved.
        bonus (bool): Flag indicating whether to include bonus features.
        alpha (float): The learning rate for gradient descent.
        iter (int): The number of iterations for gradient descent.
        stop_threshold (float): The threshold to stop gradient descent.
        theta (numpy.ndarray): The parameters for the linear regression model.
        losses (list): List to store the loss values during training.
        theta0_history (list): List to store the theta0 values during training.
        theta1_history (list): List to store the theta1 values during training.
    """

    def __init__(self, data_path: str, pickle_path: str, bonus: bool = False,
                 learning_rate: float = 0.01, iterations: int = 1500,
                 stop_threshold: float = 1e-6):
        """
        Initializes the DataAnalysisClass.

        Args:
            data_path (str): The Dir path where the data file is located.
            pickle_path (str): Dir path where the pickle file will be saved.
            bonus (bool, optional): Flag indicating whether to include bonus.
            learning_rate (float, optional): Learning rate gradient descent.
            iterations (int, optional): Iterations for gradient descent.
        """
        self.data_path = data_path
        self.pickle_path = pickle_path
        self.bonus = bonus
        self.alpha = learning_rate
        self.iter = iterations
        self.stop_threshold = stop_threshold
        self.theta = np.zeros(2)
        self.__load_data()
        self.__normalize_features()
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
            self.data = pd.read_csv(self.data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.data_path} not found") from e

        try:
            self.data = self.data.dropna().drop_duplicates()
            self.km = self.data['km'].values.astype(float)
            self.price = self.data['price'].values.astype(float)
        except ValueError as e:
            raise ValueError(e)
    
        if len(self.km) < 2 or len(self.price) < 2:
            raise ValueError("Not enough data to train the model")

    def __normalize_features(self) -> None:
        """
        Standardizes the data by calculating mean
        and standard deviation of kilometers.
        """
        self.mean_km = np.mean(self.km)
        self.std_km = np.std(self.km)
        self.x_normalized = (self.km - self.mean_km) / self.std_km
        self.x_normalized = np.column_stack(
            (self.x_normalized, np.ones_like(self.x_normalized)))

    def __initialize_plot(self) -> None:
        """
        Initializes the plot for the training animation.
        """
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 16))
        plt.subplots_adjust(hspace=0.5)

        self.main_title = self.fig.suptitle(f'Linear Regression Training\nLearning Rate: {self.alpha}, Epochs: 0')

        # Plot for data points and regression line
        self.ax1.scatter(self.km, self.price, color='blue')
        self.line1, = self.ax1.plot(self.km, self.__hypothesis(self.theta, self.x_normalized), 'r-')
        self.ax1.set_xlabel('km')
        self.ax1.set_ylabel('price')
        self.ax1.set_title('Linear Regression Training')
        self.ax1.legend(['Data', 'Linear Regression Line'])
        self.ax1.grid(True)

        # Plot for Theta 0 values
        self.theta0_vals, = self.ax2.plot([], [], 'b-', label='Theta 0 (Weight)')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Theta 0 Value')
        self.ax2.set_title('Theta 0 Values over Iterations')
        self.ax2.grid(True)

        # Plot for Theta 1 values
        self.theta1_vals, = self.ax3.plot([], [], 'r-', label='Theta 1 (Bias)')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Theta 1 Value')
        self.ax3.set_title('Theta 1 Values over Iterations')
        self.ax3.grid(True)

        # Plot for Cost function values
        self.loss_vals, = self.ax4.plot([], [], 'g-', label='Cost Function')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Cost')
        self.ax4.set_title('Cost Function over Iterations')
        self.ax4.grid(True)

    def __save_animation(self) -> None:
        """
        Saves the training animation to a GIF file.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest = os.path.join(base_dir, 'srcs/training_animation.gif')
        self.ani.save(dest, writer='imagemagick', fps=60)

    def __rescale_theta(self, theta: np.ndarray) -> tuple[float, float]:
        """
        Rescales the theta values back to the original data scale.

        Args:
            theta (numpy.ndarray): The trained coefficients.

        Returns:
            Tuple[float, float]: The rescaled coefficients.
        """
        mean_price = np.mean(self.price)
        std_price = np.std(self.price)
        original_theta1 = theta[1] * std_price / self.std_km
        original_theta0 = (theta[0]
                           - (original_theta1 * self.mean_km / self.std_km)
                           + mean_price)
        return original_theta0, original_theta1

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
        m = len(y)
        return np.sum((self.__hypothesis(theta, x) - y) ** 2) / (2 * m)

    def __train_dataset(self) -> None:
        """
        Trains the linear regression model using gradient descent.
        """
        m = len(self.price)
        x = self.x_normalized
        y = self.price
        theta = np.zeros(2)

        self.__initialize_plot()
        self.losses = []
        self.theta0_history = []
        self.theta1_history = []

        def update(frame) -> iter:
            nonlocal theta
            hypothesis = self.__hypothesis(theta, x)
            error = hypothesis - y
            gradient = np.dot(x.T, error) / m
            theta -= self.alpha * gradient
            self.losses.append(self.__cost_function(theta, x, y))

            self.theta0_history.append(theta[0])
            self.theta1_history.append(theta[1])
            self.line1.set_ydata(self.__hypothesis(theta, x))
            self.theta0_vals.set_data(
                range(len(self.theta0_history)), self.theta0_history)
            self.theta1_vals.set_data(
                range(len(self.theta1_history)), self.theta1_history)
            self.loss_vals.set_data(range(len(self.losses)), self.losses)

            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax3.relim()
            self.ax3.autoscale_view()
            self.ax4.relim()
            self.ax4.autoscale_view()

            self.ax2.legend().remove()
            self.ax2.legend([f'Theta 0: {self.theta0_history[-1]:.8f}'])
            self.ax3.legend().remove()
            self.ax3.legend([f'Theta 1: {self.theta1_history[-1]:.8f}'])
            self.ax4.legend().remove()
            self.ax4.legend([f'Cost: {self.losses[-1]:.8f}'])

            epoch = len(self.theta0_history)
            self.main_title.set_text(f'Linear Regression Training\nLearning Rate: {self.alpha}, Epochs: {epoch}')

            if len(self.losses) > 1:
                if (abs(self.losses[-1]
                        - self.losses[-2])
                        < self.stop_threshold):
                    return

        self.ani = animation.FuncAnimation(self.fig, update,
                                           frames=self.iter,
                                           repeat=False)
        self.__save_animation()
        plt.ioff()
        plt.show()
        self.theta = theta

    def __save_model(self) -> None:
        """
        Saves the trained model parameters to a pickle file.
        """
        try:
            with open(self.pickle_path, 'wb') as f:
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
                self.x_normalized),
            color='red')
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear Regression Line'])
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.show()

    def __plot_residuals(self) -> None:
        """
        Plots residuals to check for any patterns.
        """
        predictions = self.__hypothesis(self.theta, self.x_normalized)
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
        predictions = self.__hypothesis(self.theta, self.x_normalized)
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


def premade_data() -> None:
    """
    Function to run the DataAnalysisClass with premade data.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'csv_files/data.csv')
    pickle_path = os.path.join(base_dir, 'pickle_files/model.pkl')
    bonus = True
    learning_rate = 0.01
    iterations = 1500
    stop_threshold = 1e-6
    DataAnalysisClass(
        data_path=data_path,
        pickle_path=pickle_path,
        bonus=bonus,
        learning_rate=learning_rate,
        iterations=iterations,
        stop_threshold=stop_threshold
    )


if __name__ == "__main__":
    premade_data()
