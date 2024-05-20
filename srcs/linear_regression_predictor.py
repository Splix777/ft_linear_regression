import os
import json


class LinearRegressionPredictor:
    def __init__(self, model_path: str) -> None:
        """
        Load a linear regression model from a pickle file.
        """
        self.model_path = model_path
        self.__load_model()

    def __load_model(self) -> None:
        """
        Load the model from the pickle file.
        """
        try:
            with open(self.model_path, 'rb') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.model_path} not found") from e

        self.theta = data["theta"]

    def predict(self, km: float) -> float:
        """
        Predict the price of a car given its kilometers.
        """
        if not isinstance(km, int) or km < 0 or km > 1_000_000:
            raise ValueError("Please enter a number between 0 and 1_000_000")
        results = float(self.theta[0]) + float(self.theta[1]) * km
        return round(results, 2)


def get_user_input() -> int:
    """
    Get user input for the kilometers of the car.

    Returns:
        int: The kilometers of the car.

    Raises:
        ValueError: If the input is not a valid number.
    """
    km = 0
    try:
        print('Please input your cars kilometers to get a price estimate.',
              'Values should be between 0 and 1_000_000.',
              sep='\n')

        km = int(input("Cars Kilometers: "))
        if km < 0 or km > 1_000_000:
            raise ValueError("Please enter a number between 0 and 1_000_000")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pickle_path = os.path.join(base_dir, "json_files/model.json")
        predictor = LinearRegressionPredictor(pickle_path)
        price = predictor.predict(km)
        if price < 0:
            print(f"Price for a car with {km} km is: $0 ({price})")
        else:
            print(f"Price for a car with {km} km is: ${price}")
    except ValueError as e:
        raise ValueError("Please enter a valid number") from e

    return km


if __name__ == "__main__":
    get_user_input()
