import matplotlib.pyplot as plt
import numpy as np


class PlottingClass:
    """
    Class for plotting and visualizing the linear regression training process.

    Includes methods for initializing plots, updating plots with training progress, showing plots,
    saving animations, plotting residuals, and plotting data.
    """

    def __init__(self, x_feature: np.ndarray, x_name: str, y_feature: np.ndarray,
                 y_name: str, learning_rate: float, iterations: int, stop_threshold: float):
        """
        Initialize the plotting class for visualizing linear regression training.

        Initializes the plotting class with the specified features, learning rate, number of iterations,
        and stop threshold for plotting the training progress.
        """
        self.plt = plt
        self.main_title = None
        self.fig = None
        self.loss_vals = None
        self.theta1_vals = None
        self.theta0_vals = None
        self.line1 = None
        self.x_feature = x_feature
        self.x_name = x_name
        self.y_feature = y_feature
        self.y_name = y_name
        self.alpha = learning_rate
        self.iter = iterations
        self.stop_threshold = stop_threshold

    def initialize_plots(self) -> None:
        """
        Initialize the plots for visualizing linear regression training.

        Sets up the subplots for displaying data points, regression lines, theta values, and cost function values
        over the training iterations.
        """
        # self.plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 16))
        self.plt.subplots_adjust(hspace=0.5)

        self.main_title = self.fig.suptitle(f'Linear Regression Training\nLearning Rate: {self.alpha}, Epochs: 0')

        # Plot for data points and regression line
        self.ax1.scatter(self.x_feature, self.y_feature, color='blue')
        self.line1, = self.ax1.plot(self.x_feature, np.zeros_like(self.x_feature), 'r-')
        self.ax1.set_xlabel(f'{self.x_name}')
        self.ax1.set_ylabel(f'{self.y_name}')
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

    def update_plots(self, theta0_history: np.ndarray, theta1_history: np.ndarray,
                     losses: np.ndarray, hypothesis: np.ndarray, epoch: int) -> None:
        """
        Update the plots with the latest training progress.

        Updates the plots with the latest values of hypothesis, theta 0, theta 1, and cost function
        over the training iterations. Adjusts the axes and legends accordingly to reflect the
        current state of the training.
        """
        self.line1.set_ydata(hypothesis)
        # self.line1.set_data(self.x_feature, hypothesis)
        self.theta0_vals.set_data(range(len(theta0_history)), theta0_history)
        self.theta1_vals.set_data(range(len(theta1_history)), theta1_history)
        self.loss_vals.set_data(range(len(losses)), losses)

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()

        if epoch > 1:
            self.__update_legends(theta0_history, theta1_history, losses)
        self.main_title.set_text(f'Linear Regression Training\nLearning Rate: {self.alpha}, Epochs: {epoch}')

    def __update_legends(self, theta0_history, theta1_history, losses):
        self.ax2.legend().remove()
        self.ax2.legend([f'Theta 0: {theta0_history[-1]:.8f}'])
        self.ax3.legend().remove()
        self.ax3.legend([f'Theta 1: {theta1_history[-1]:.8f}'])
        self.ax4.legend().remove()
        self.ax4.legend([f'Cost: {losses[-1]:.8f}'])

    def show(self) -> None:
        """
        Display the plots.

        Turns off interactive mode and displays the plots for visualization.
        """
        self.plt.ioff()
        self.plt.show()

    @staticmethod
    def save_animation(ani: plt, dest: str) -> None:
        """
        Save the animation as a GIF.

        Saves the provided animation object as a GIF file at the specified
        destination using the 'ffmpeg' writer with a frame rate of 15 fps.
        """
        try:
            ani.save(dest, writer='ffmpeg', fps=15)
        except Exception as e:
            print(f"Error saving the animation: {e}")

    @staticmethod
    def plot_residuals(feature: np.ndarray, residuals: np.ndarray, feature_name: str) -> None:
        """
        Plot the residuals of the linear regression model.

        Creates a scatter plot of residuals against a specific feature, with the residuals colored in red.
        Includes a horizontal line at y=0, representing the zero residual line.
        Args:
            feature: An array of feature values.
            residuals: An array of residuals corresponding to the feature values.
            feature_name: The name of the feature being analyzed.
        Returns:
            None
        """
        plt.scatter(feature, residuals, color='red')
        plt.axhline(y=0, color='blue', linestyle='--')
        plt.title('Residuals Plot')
        plt.xlabel(f'{feature_name}')
        plt.ylabel('Residuals')
        plt.grid()
        plt.show()

    @staticmethod
    def plot_data(feature1: np.ndarray, feature2: np.ndarray, hypothesis: np.ndarray,
                  feature1_name: str, feature2_name: str) -> None:
        """
        Plot the data points and linear regression line.

        Creates a scatter plot of data points with a linear regression line based on the provided features.
        The data points are shown in blue, and the linear regression line is displayed in red.
        Args:
            feature1: An array of values for the first feature.
            feature2: An array of values for the second feature.
            hypothesis: An array of predicted values based on the linear regression model.
            feature1_name: The name of the first feature.
            feature2_name: The name of the second feature.
        Returns:
            None
        """
        plt.scatter(feature1, feature2, color='blue')
        plt.plot(feature1, hypothesis, color='red')
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear Regression Line'])
        plt.grid()
        plt.xlabel(f'{feature1_name}')
        plt.ylabel(f'{feature2_name}')
        plt.show()

    @staticmethod
    def plot_cost(costs: np.ndarray) -> None:
        """
        Plot the cost function values over the range of weights.

        Creates a line plot of the cost function values against a range of weight values.
        Args:
            weights: An array of weight values.
            costs: An array of cost function values corresponding to the weight values.
        Returns:
            None
        """
        plt.plot(range(len(costs)), costs, color='green')
        plt.title('Costs Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid()
        plt.show()
