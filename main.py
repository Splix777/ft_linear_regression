import argparse

from srcs.model.model import LinearRegressionModel
from srcs.utils.utils import (
    verify_args,
    verify_prediction_request,
    add_arguments
)


def train_or_predict(model: LinearRegressionModel, mode: str):
    """
    Train or predict the linear regression model.

    Args:
        - model: LinearRegressionModel: The linear regression model.
        - mode: str: The mode to run the model.

    Returns:
        None
    """
    if mode == "train":
        model.fit()
        response = input("Do you want to evaluate the model? (y/n): ")
        if response.lower() == "y":
            model.evaluate()
    elif mode == "predict":
        response = input("Enter a value to predict: ")
        model.load_model()
        prediction = model.predict(verify_prediction_request(response))
        print(f"The predicted value is: ${prediction:.2f}")
    elif mode == "evaluate":
        model.load_model()
        model.evaluate()


def main():
    """
    Main function to run the linear regression
    model training and prediction.

    Args:
        - csv_path: str: The path to the CSV file.
        - model_dir: str: The directory to save the model.
        - target: str: The target column in the CSV file.
        - pred: str: The feature to predict the target.
        - mode: str: The mode to run the model.
        - bonus: bool: Whether to run the bonus part.

    Returns:
        None
    """
    args = add_arguments(args=argparse.ArgumentParser())
    parsed_args = verify_args(args=args.parse_args())

    model = LinearRegressionModel(
        csv_path=parsed_args.csv_path,
        model_dir=parsed_args.model_dir,
        bonus=parsed_args.bonus,
        learn_rate=0.01,
        loss_thresh=1e-6,
        epochs=1000,
        patience=10,
        target=parsed_args.target or "price",
        pred=parsed_args.pred or "km",
    )
    train_or_predict(model=model, mode=parsed_args.mode)


if __name__ == "__main__":
    """📈 Linear Regression Model 📉"""
    try:
        main()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
