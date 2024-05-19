import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class PlottingClass:
    def __init__(self, x_feature, x_name, y_feature, y_name, learning_rate, iterations, stop_threshold):
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
        self.initialize_plots()

    def initialize_plots(self):
        self.plt.ion()
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

    def update_plots(self, theta0_history, theta1_history, losses, hypothesis):
        self.line1.set_ydata(hypothesis)
        self.theta0_vals.set_data(range(len(theta0_history)), theta0_history)
        self.theta1_vals.set_data(range(len(theta1_history)), theta1_history)
        self.loss_vals.set_data(range(len(losses)), losses)

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()

        self.ax2.legend().remove()
        self.ax2.legend([f'Theta 0: {theta0_history[-1]:.8f}'])
        self.ax3.legend().remove()
        self.ax3.legend([f'Theta 1: {theta1_history[-1]:.8f}'])
        self.ax4.legend().remove()
        self.ax4.legend([f'Cost: {losses[-1]:.8f}'])

        epoch = len(theta0_history)
        self.main_title.set_text(f'Linear Regression Training\nLearning Rate: {self.alpha}, Epochs: {epoch}')

    def show(self):
        self.plt.ioff()
        self.plt.show()

    @staticmethod
    def save_animation(ani, dest):
        try:
            ani.save(dest, writer='ffmpeg', fps=15)
        except Exception as e:
            print(f"Error saving the animation: {e}")

    @staticmethod
    def plot_residuals(feature, residuals, feature_name):
        plt.scatter(feature, residuals, color='red')
        plt.axhline(y=0, color='blue', linestyle='--')
        plt.title('Residuals Plot')
        plt.xlabel(f'{feature_name}')
        plt.ylabel('Residuals')
        plt.grid()
        plt.show()

    @staticmethod
    def plot_data(feature1, feature2, hypothesis, feature1_name, feature2_name):
        plt.scatter(feature1, feature2, color='blue')
        plt.plot(feature1, hypothesis, color='red')
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear Regression Line'])
        plt.grid()
        plt.xlabel(f'{feature1_name}')
        plt.ylabel(f'{feature2_name}')
        plt.show()
