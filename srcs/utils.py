import os
import argparse


def verify_args(args: argparse.Namespace) -> tuple[str, str, str, bool]:
    """
    Verify the arguments passed to the program.

    Args:
        - args: argparse.Namespace: The args passed to the program.

    Returns:
        - tuple[str, str, str, bool]: The verified arguments.

    Raises:
        - FileNotFoundError: If the data file is not found.
        - ValueError: If the data file is not a CSV file
            or the mode is invalid.
        - PermissionError: If the data file or model directory
            is not accessible.
    """
    # Check if the CSV file exists and is accessible
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError("Data file not found.")
    if not args.csv_path.endswith(".csv"):
        raise ValueError("Data file must be a CSV file")
    if not os.access(args.csv_path, os.R_OK):
        raise PermissionError("CSV file is not readable.")

    # Check if the model directory exists or create it
    if not os.path.exists(args.model_dir):
        try:
            os.makedirs(args.model_dir, exist_ok=True)
        except Exception as e:
            raise e
    elif not os.access(args.model_dir, os.W_OK):
        raise PermissionError("Model directory is not writable.")

    # Check if the mode is valid
    if args.mode not in ["train", "predict"]:
        raise ValueError("Mode must be either 'train' or 'predict'.")
    if args.mode == "predict":
        model_file = os.path.join(args.model_dir, "model.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError("Model file not found.")

    return args.csv_path, args.model_dir, args.mode, args.bonus


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
        if 0 > number <= 2_000_000:
            raise ValueError("Prediction must be between 0 and 2,000,000.")
        return number

    except ValueError as e:
        raise ValueError("Prediction request must be a number.") from e
