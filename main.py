import argparse

from srcs.train_model import LinearRegressionModel
from srcs.utils import verify_args, verify_prediction_request


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
    args = argparse.ArgumentParser()
    args.add_argument("--csv_path", type=str, required=True,
                      help="The path to the CSV file.")
    args.add_argument("--model_dir", type=str, required=True,
                      help="The directory to save or load the model.")
    args.add_argument("--mode", type=str, required=True,
                      help="The mode to run the model (train or predict).")
    args.add_argument("--target", type=str, required=False,
                      help="The target column in the CSV file. (optional)")
    args.add_argument("--pred", type=str, required=False,
                      help="The feature to predict the target. (optional)")
    args.add_argument("--bonus", action="store_true",
                      help="Run the bonus part of the model. (default: False)")
    args = args.parse_args()

    csv_path, model_dir, mode, bonus = verify_args(args)
    model = LinearRegressionModel(
        csv_path=csv_path,
        model_dir=model_dir,
        bonus=bonus,
        target=args.target or "price",
        pred=args.pred or "km",
    )
    train_or_predict(model=model, mode=mode)


if __name__ == "__main__":
    """📈 Linear Regression Model 📉"""
    try:
        main()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
