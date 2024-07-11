import argparse

from srcs.train_model import LinearRegressionModel
from srcs.utils import verify_args, verify_prediction_request


def train_or_predict(model, mode):
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
    args = argparse.ArgumentParser()
    args.add_argument("--csv_path", type=str, required=True)
    args.add_argument("--model_dir", type=str, required=True)
    args.add_argument("--target", type=str, required=False)
    args.add_argument("--pred", type=str, required=False)
    args.add_argument("--mode", type=str, required=True)
    args.add_argument("--bonus", action="store_true")
    args = args.parse_args()

    csv_path, model_dir, mode, bonus = verify_args(args)
    model = LinearRegressionModel(
        csv_path=csv_path,
        model_dir=model_dir,
        bonus=bonus,
        target=args.target or "price",
        pred=args.pred or "km",
    )
    train_or_predict(model, mode)


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
