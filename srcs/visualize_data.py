import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dotenv import load_dotenv
from srcs.utils import error_decorator


def plot_linear_regression(data, theta0, theta1):
    """
    Plot the data along with the linear regression line.

    Args:
        data (DataFrame): A DataFrame containing the mileage and price of cars.
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.
    """
    plt.scatter(data['km'], data['price'], color='blue', label='Data points')
    plt.plot(data['km'], theta0 + theta1 * data['km'], color='red', label='Linear regression')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


def load_parameters(file_path):
    """
    Load the trained parameters theta0 and theta1 from a file.

    Args:
        file_path (str): The file path to load the parameters from.

    Returns:
        tuple: A tuple containing the loaded parameters theta0 and theta1.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        theta0 = float(lines[0].split('=')[1].strip())
        theta1 = float(lines[1].split('=')[1].strip())
    return theta0, theta1


@error_decorator(debug=False)
def visualize_data():
    load_dotenv()
    csv_path = Path(__file__).parent.parent / os.getenv("CSV_PATH")
    data = pd.read_csv(csv_path)

    parameters_file_path = Path(__file__).parent.parent / os.getenv("OUTPUT_FILE_PATH")
    theta0, theta1 = load_parameters(str(parameters_file_path))

    plot_linear_regression(data, theta0, theta1)


if __name__ == "__main__":
    visualize_data()
