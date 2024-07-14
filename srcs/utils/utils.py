import os
import argparse

import pandas as pd


def verify_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Verify the arguments passed to the program.

    Args:
        - args: argparse.Namespace: The args passed to the program.

    Returns:
        - argparse.Namespace: The verified arguments.

    Raises:
        - FileNotFoundError: If the data file is not found.
        - ValueError: If the data file is not a CSV file
            or the mode is invalid.
        - PermissionError: If the data file or model directory
            is not accessible.
    """
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError("Data file not found.")
    if not args.csv_path.endswith(".csv"):
        raise ValueError("Data file must be a CSV file")
    if not os.access(args.csv_path, os.R_OK):
        raise PermissionError("CSV file is not readable.")

    if not os.path.exists(args.model_dir):
        try:
            os.makedirs(args.model_dir, exist_ok=True)
        except Exception as e:
            raise e
    elif not os.access(args.model_dir, os.W_OK):
        raise PermissionError("Model directory is not writable.")

    if args.mode not in ["train", "predict", "evaluate"]:
        raise ValueError("Mode must be either 'train', 'predict', 'evaluate'.")
    if args.mode in ["predict", "evaluate"]:
        model_file = os.path.join(args.model_dir, "model.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError("Model file not found.")

    if bool(args.target) != bool(args.pred):
        raise ValueError("Both 'target' and 'pred' must be provided together.")

    return args


def verify_prediction_request(response: str) -> float:
    """
    Verify the prediction request.

    Args:
        - response: str: The response from the user.

    Returns:
        - float: The verified prediction request.

    Raises:
        - ValueError: If the response is not a number.
    """
    try:
        number = float(response)
    except ValueError as e:
        raise ValueError("Prediction request must be a number") from e

    if number < 0 or number > 2_000_000:
        raise ValueError("Prediction must be between 0 and 2,000,000.")

    return number


def add_arguments(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add the arguments to the program.

    Args:
        - args: argparse.ArgumentParser: The argument parser object.

    Returns:
        - argparse.ArgumentParser: The argument parser
            object with added arguments.
    """
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

    return args


def verify_project_csv(csv_path: str) -> pd.DataFrame:
    """
    Verify the project CSV file.

    Args:
        - csv_path: str: The path to the CSV file.

    Returns:
        data: pd.DataFrame: The data from the CSV file.

    Raises:
        FileNotFoundError: If the CSV file is not found.
        ValueError: If the CSV file is not a CSV file.
        PermissionError: If the CSV file is not readable.
        ValueError: If the CSV file contains missing values.
        ValueError: If the CSV file is empty.
        ValueError: If the CSV file does not contain exactly 2 columns.
        ValueError: If the CSV file is not a 2D array.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Data file not found.")
    if not csv_path.endswith(".csv"):
        raise ValueError("Data file must be a CSV file.")
    if not os.access(csv_path, os.R_OK):
        raise PermissionError("CSV file is not readable.")

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise e

    if data.isnull().values.any():
        raise ValueError("Data file contains missing values.")
    if data.empty:
        raise ValueError("Data file is empty.")
    if data.shape[1] != 2:
        raise ValueError("Data file must contain exactly 2 columns.")
    if data.ndim != 2:
        raise ValueError("Data file must be a 2D array.")

    return data
