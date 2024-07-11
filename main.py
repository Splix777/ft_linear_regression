import argparse

from srcs.train_model import LinearRegressionModel
from srcs.utils import verify_args, verify_prediction_request


def train_or_predict(model, mode):
    if mode == "train":
        model.fit()
        model.evaluate()
    elif mode == "predict":
        response = input("Enter a value to predict: ")
        model.load_model()
        pred = model.predict(verify_prediction_request(response))
        print(f"The predicted value is: ${pred:.2f}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--csv_path", type=str, required=True)
    args.add_argument("--model_dir", type=str, required=True)
    args.add_argument("--mode", type=str, required=True)
    args.add_argument("--bonus", action="store_true")

    args = args.parse_args()

    csv_path, model_dir, mode, bonus = verify_args(args)
    model = LinearRegressionModel(
        csv_path=csv_path,
        model_dir=model_dir,
        bonus=bonus
    )
    train_or_predict(model, mode)


if __name__ == "__main__":
    main()
