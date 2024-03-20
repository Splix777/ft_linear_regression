import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from srcs.utils import error_decorator


def train_linear_regression(data):
    """
    Train a linear regression model and save the parameters to a file.

    Args:
        data (DataFrame): A DataFrame containing the mileage and price of cars.

    Returns:
        tuple: A tuple containing the trained parameters theta0 and theta1.
    """
    mean_mileage = data['km'].mean()
    mean_price = data['price'].mean()

    num = ((data['km'] - mean_mileage) * (data['price'] - mean_price)).sum()
    den = ((data['km'] - mean_mileage) ** 2).sum()
    theta1 = num / den
    theta0 = mean_price - (theta1 * mean_mileage)

    return theta0, theta1


def save_parameters(theta0, theta1, file_path):
    """
    Save the trained parameters theta0 and theta1 to a file.

    Args:
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.
        file_path (str): The file path to save the parameters.
    """
    with open(file_path, 'w') as f:
        f.write(f"theta0={theta0}\n")
        f.write(f"theta1={theta1}")


@error_decorator(debug=False)
def train():
    load_dotenv()
    csv_path = Path(__file__).parent.parent / os.getenv("CSV_PATH")
    data = pd.read_csv(csv_path)

    theta0, theta1 = train_linear_regression(data)

    output_file_path = Path(__file__).parent.parent / os.getenv("OUTPUT_FILE_PATH")
    save_parameters(theta0, theta1, str(output_file_path))


if __name__ == "__main__":
    train()
