import os

from pathlib import Path
from dotenv import load_dotenv
from srcs.utils import error_decorator


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


def predict_price(theta0, theta1, mileage):
    """
    Predict the price of a car based on its mileage using a linear regression model.

    Args:
        theta0 (float): Intercept parameter.
        theta1 (float): Slope parameter.
        mileage (float): The mileage of the car.

    Returns:
        float: The predicted price of the car.
    """
    return theta0 + theta1 * mileage


@error_decorator(debug=True)
def main():
    load_dotenv()

    mileage = float(input("Enter the mileage of the car: "))
    if mileage < 0 or mileage > 1_000_000:
        raise ValueError("The mileage must be between 0 and 1,000,000.")

    parameters_file_path = Path(__file__).parent.parent / os.getenv("OUTPUT_FILE_PATH")
    theta0, theta1 = load_parameters(str(parameters_file_path))

    price = predict_price(theta0, theta1, mileage)
    print(f"The estimated price for a car with {mileage} miles is: ${price:.2f}.")


if __name__ == "__main__":
    main()
