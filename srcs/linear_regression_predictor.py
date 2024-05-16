import pickle


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
                data = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.model_path} not found") from e

        self.theta = data["theta"]
        self.mean_km = data["mean_km"]
        self.std_km = data["std_km"]

    def predict(self, km: int) -> float:
        """
        Predict the price of a car given its kilometers.
        """
        if not isinstance(km, int) or km < 0 or km > 1_000_000:
            raise ValueError("Please enter a number between 0 and 1_000_000")
        price = (self.theta[0] * (km - self.mean_km)
                 / self.std_km + self.theta[1])
        return round(price, 2)


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
    except ValueError as e:
        raise ValueError("Please enter a valid number") from e

    return km


if __name__ == "__main__":
    pickle_dir = '/home/splix/Desktop/ft_linear_regression/pickle_files/'
    pickle_model = 'model.pkl'
    predictor = LinearRegressionPredictor(pickle_dir + pickle_model)

    car_km = get_user_input()
    print(f"Price for a car with {car_km} km is: ${predictor.predict(car_km)}")
