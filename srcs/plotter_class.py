import matplotlib.pyplot as plt
import numpy as np


class PlottingClass:
    """
    Class for plotting and visualizing the linear
    regression training process.

    Includes methods for initializing plots, updating plots
    with training progress, showing plots,
    saving animations, plotting residuals, and plotting data.
    """
    def __init__(self, x_feature: np.ndarray, x_name: str,
                 y_feature: np.ndarray, y_name: str, learning_rate: float,
                 iterations: int, stop_threshold: float):
        """
        Initialize the plotting class for visualizing
        linear regression training.

        Initializes the plotting class with the specified features,
        learning rate, number of iterations,
        and stop threshold for plotting the training progress.
        """
        self.plt = plt
        self.x_feature = x_feature
        self.x_name = x_name
        self.y_feature = y_feature
        self.y_name = y_name
        self.learning_rate = learning_rate
        self.iter = iterations
        self.stop_threshold = stop_threshold

        self.main_title = None
        self.fig = None
        self.loss_vals = None
        self.slope_vals = None
        self.intercept_vals = None
        self.line1 = None

    def initialize_gif_plot(self) -> None:
        """
        Initialize the plots for visualizing linear regression
        training. Sets up the subplots for displaying data points,
        regression lines, theta values, and cost function values
        over the training iterations.
        """
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) \
            = plt.subplots(4, 1, figsize=(10, 16))
        self.plt.subplots_adjust(hspace=0.5)

        self.main_title = self.fig.suptitle(
            f'Linear Regression Training\nLearning Rate: '
            f'{self.learning_rate}, Epochs: 0'
        )

        # Plot for data points and regression line
        self.ax1.scatter(self.x_feature, self.y_feature, color='blue')
        self.line1, = self.ax1.plot(
            self.x_feature,
            np.zeros_like(self.x_feature), 'r-'
        )
        self.ax1.set_xlabel(f'{self.x_name}')
        self.ax1.set_ylabel(f'{self.y_name}')
        self.ax1.set_title('Linear Regression Training')
        self.ax1.legend(['Data', 'Linear Regression Line'])
        self.ax1.grid(True)

        # Plot for Intercept values
        self.intercept_vals, = self.ax2.plot([], [], 'b-', label='Intercept')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Theta 0 Value')
        self.ax2.set_title('Theta 0 Values over Iterations')
        self.ax2.grid(True)

        # Plot for Slope values
        self.slope_vals, = self.ax3.plot([], [], 'r-', label='Slope')
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

    def update_plots(self, inter_history: np.ndarray,
                     slope_history: np.ndarray, loss_history: np.ndarray,
                     predictions: np.ndarray, epoch: int) -> None:
        """
        Update the plots with the latest training progress.

        Args:
            inter_history: An array of historical values for theta0.
            slope_history: An array of historical values for theta1.
            loss_history: An array of historical cost function values.
            predictions: An array of predicted values based on the
                linear regression model.
            epoch: The current epoch number.

        Returns:
            None
        """
        self.line1.set_ydata(predictions)
        self.intercept_vals.set_data(range(len(inter_history)), inter_history)
        self.slope_vals.set_data(range(len(slope_history)), slope_history)
        self.loss_vals.set_data(range(len(loss_history)), loss_history)

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()

        if epoch > 1:
            self._update_legends(inter_history, slope_history, loss_history)

        self.main_title.set_text(
            f'Linear Regression Training\nLearning Rate: '
            f'{self.learning_rate}, Epochs: {epoch}'
        )

    def _update_legends(self, inter_history: np.ndarray,
                        slope_history: np.ndarray, loss_history: np.ndarray):
        """
        Update the legends in the plots with the latest values of
        theta0, theta1, and the cost.

        Args:
            inter_history: List of historical values for theta0.
            slope_history: List of historical values for theta1.
            loss_history: List of historical cost values.

        Returns:
            None
        """
        self.ax2.legend().remove()
        self.ax2.legend([f'Theta 0: {inter_history[-1]:.4f}'])
        self.ax3.legend().remove()
        self.ax3.legend([f'Theta 1: {slope_history[-1]:.4f}'])
        self.ax4.legend().remove()
        self.ax4.legend([f'Cost: {loss_history[-1]:.4f}'])

    @staticmethod
    def save_animation(ani: plt, dest: str):
        """
        Save the animation as a GIF.

        Saves the provided animation object as a GIF
        file at the specified destination using the 'ffmpeg'
        writer with a frame rate of 15 fps.

        Args:
            ani: The animation object to save.
            dest: The destination path to save the animation.

        Returns:
            None

        Raises:
            Exception: If an error occurs while saving the animation
        """
        try:
            ani.save(dest, writer='ffmpeg', fps=15)

        except Exception as e:
            print(f"Error saving the animation: {e}")

    @staticmethod
    def plot_residuals(feature: np.ndarray, residuals: np.ndarray,
                       feature_name: str):
        """
        Plot the residuals of the linear regression model.

        Creates a scatter plot of residuals against a specific feature,
        with the residuals colored in red.
        Includes a horizontal line at y=0, representing the
        zero residual lines.

        Args:
            feature: An array of feature values.
            residuals: An array of residuals corresponding
                to the feature values.
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
        plt.savefig('images/my_residuals.png')
        plt.close()

    @staticmethod
    def plot_data(feature1: np.ndarray, feature2: np.ndarray,
                  predictions: np.ndarray, feature1_name: str,
                  feature2_name: str):
        """
        Plot the data points and linear regression line.

        Creates a scatter plot of data points with a linear regression
        line based on the provided features.
        The data points are shown in blue, and the linear regression
        line is displayed in red.

        Args:
            feature1: An array of values for the first feature.
            feature2: An array of values for the second feature.
            predictions: An array of predicted values based on
                the linear regression model.
            feature1_name: The name of the first feature.
            feature2_name: The name of the second feature.

        Returns:
            None
        """
        plt.scatter(feature1, feature2, color='blue')
        plt.plot(feature1, predictions, color='red')
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear Regression Line'])
        plt.grid()
        plt.xlabel(f'{feature1_name}')
        plt.ylabel(f'{feature2_name}')
        plt.savefig('images/my_linear.png')
        plt.close()

    @staticmethod
    def plot_cost(costs: np.ndarray):
        """
        Plot the cost function values over the range of weights.

        Creates a line plot of the cost function values against
        a range of weight values.

        Args:
            costs: An array of cost function values
                corresponding to the weight values.

        Returns:
            None
        """
        plt.plot(range(len(costs)), costs, color='green')
        plt.title('Costs Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid()
        plt.savefig('images/my_costs.png')
        plt.close()
