import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from srcs.predict_price import predict_price, load_parameters


def calculate_precision(data, theta0, theta1):
    """
    Calculate the precision of the linear regression algorithm.

    Args:
        data (DataFrame): A DataFrame containing the mileage and price of cars.
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.

    Returns:
        float: The precision of the algorithm.
    """
    predicted_prices = predict_price(theta0, theta1, data['km'])
    actual_prices = data['price']
    error = actual_prices - predicted_prices
    mean_absolute_error = abs(error).mean()
    return 1 - (mean_absolute_error / actual_prices.mean())


def precision_of_algorithm():
    load_dotenv()
    csv_path = Path(__file__).parent.parent / os.getenv("CSV_PATH")
    data = pd.read_csv(csv_path)

    parameters_file_path = Path(__file__).parent.parent / os.getenv("OUTPUT_FILE_PATH")
    theta0, theta1 = load_parameters(str(parameters_file_path))

    precision = calculate_precision(data, theta0, theta1)
    print(f"The precision of the algorithm is: {precision:.2%}")


if __name__ == "__main__":
    precision_of_algorithm()
