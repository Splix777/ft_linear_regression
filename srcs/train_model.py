import os
import json
import time
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import animation
from dataclasses import dataclass, field

from srcs.plotter_class import PlottingClass


@dataclass
class LinearRegressionModel:
    """
    Dataclass to store the linear regression model.
    """
    csv_path: str
    model_dir: str
    bonus: bool

    learn_rate: float = field(default=0.01)
    loss_thresh: float = field(default=1e-6)
    epochs: int = field(default=5_000)
    patience: int = field(default=10)
    slope: float = field(default=0)
    intercept: float = field(default=0)
    loss: list = field(default_factory=list)
    history: dict = field(default_factory=dict)
    original_data: pd.DataFrame = field(default=None)

    target: str = "price"
    pred: str = "km"

    def _load_data(self):
        """
        Load the data from the csv file.
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            if self.data.shape[1] != 2:
                raise ValueError("Data must have only 2 columns.")

        except FileNotFoundError as e:
            raise FileNotFoundError("Data file not found.") from e
        except Exception as e:
            raise e

    def _save_model(self):
        """
        Save the model to a JSON file.
        """
        try:
            model_file = os.path.join(self.model_dir, "model.json")
            with open(model_file, "w") as f:
                json.dump(
                    {
                        "slope": self.slope,
                        "intercept": self.intercept,
                        "target": self.target,
                        "pred": self.pred,
                    },
                    f,
                )

        except FileNotFoundError as e:
            raise FileNotFoundError("Cannot save model.") from e
        except Exception as e:
            raise e

    def load_model(self):
        """
        Load the model from a JSON file.
        """
        try:
            model_file = os.path.join(self.model_dir, "model.json")
            with open(model_file, "r") as f:
                model = json.load(f)
                self.slope = model["slope"]
                self.intercept = model["intercept"]
                self.target = model["target"]
                self.pred = model["pred"]

        except FileNotFoundError as e:
            raise FileNotFoundError("Model file not found.") from e
        except Exception as e:
            raise e

    def _standardize_data(self):
        """
        Standardize the data.
        """
        try:
            if self.bonus:
                self.original_data = self.data.copy()
            self._set_standards()

        except KeyError as e:
            raise KeyError("Column not found in data. "
                           "Please set indicate target and pred cols") from e
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Cannot divide by zero.") from e
        except Exception as e:
            raise e

    def _set_standards(self):
        """
        Set the standardization values.
        """
        self.pred_mean = self.data[self.pred].mean()
        self.pred_std = self.data[self.pred].std()
        self.target_mean = self.data[self.target].mean()
        self.target_std = self.data[self.target].std()

        self.data[self.pred] = (
                (self.data[self.pred] - self.pred_mean)
                / self.pred_std
        )
        self.data[self.target] = (
                (self.data[self.target] - self.target_mean)
                / self.target_std
        )

    def _de_standardize_weights(self):
        """
        De-standardize the slope and intercept to the original scale.
        """
        self.slope = self.slope * (self.target_std / self.pred_std)
        self.intercept = self.target_mean - (self.slope * self.pred_mean)

        if self.bonus:
            inter = self.history["intercept"]
            slope = self.history["slope"]
            for idx in range(len(inter)):
                slope[idx] = slope[idx] * (self.target_std / self.pred_std)
                inter[idx] = self.target_mean - (slope[idx] * self.pred_mean)

    def fit(self):
        """
        Fit the linear regression model.
        """
        self._load_data()
        self._standardize_data()

        pbar = tqdm(total=self.epochs, desc="Fitting model", colour='green')
        for epoch in range(self.epochs):
            self._gradient_descent()
            pbar.update(1)

            if self._early_stop():
                pbar.close()
                self.epochs = epoch + 1
                print(f"Early stopping at epoch {epoch}")
                break

        pbar.close()
        self._de_standardize_weights()
        self._save_model()

        if self.bonus:
            self.plot()

    def predict(self, x):
        """
        Predict the target value.
        """
        return self.slope * x + self.intercept

    def _early_stop(self):
        """
        Check if early stopping criteria are met.
        """
        if len(self.loss) < self.patience:
            return False

        if abs(self.loss[-1] - self.loss[-self.patience]) < self.loss_thresh:
            return True

        return np.mean(self.loss[-self.patience:]) < self.loss_thresh

    def _gradient_descent(self):
        """
        Perform gradient descent.
        """
        predictions = self.predict(self.data[self.pred])
        error = predictions - self.data[self.target]
        self.loss.append(np.mean(error ** 2))
        self._update_weights(error)

        if self.bonus:
            self._update_history()

    def _update_weights(self, error):
        """
        Update the weights.
        """
        self.slope -= self.learn_rate * np.mean(error * self.data[self.pred])
        self.intercept -= self.learn_rate * np.mean(error)

    def _update_history(self):
        """
        Update the history dictionary.
        """
        self.history.setdefault("slope", []).append(self.slope)
        self.history.setdefault("intercept", []).append(self.intercept)
        self.history.setdefault("loss", []).append(self.loss[-1])

    def evaluate(self):
        """
        Get MAE (Mean Absolute Error), RMSE (Root Mean Squared Error),
        MSE (Mean Squared Error), and R2 Score.
        """
        self._load_data()
        self.load_model()
        predictions = self.predict(self.data[self.pred])
        error = predictions - self.data[self.target]

        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum(error ** 2) / np.sum(
            (self.data[self.target] - np.mean(self.data[self.target])) ** 2))

        return (
            f"Mean Absolute Error: {mae:.2f}\n"
            f"Mean Squared Error: {mse:.2f}\n"
            f"Root Mean Squared Error: {rmse:.2f}\n"
            f"R2 Score: {r2:.2f}\n",
            f"Model Precision: {r2 * 100:.2f}%\n"
            f"Intercept: {self.intercept:.4f}\n"
            f"Slope: {self.slope:.4f}\n"
        )

    def plot(self):
        """
        Plot the linear regression model.
        """
        try:
            print('Plotting...')
            plotter = PlottingClass(
                x_feature=self.original_data[self.pred],
                x_name=self.pred,
                y_feature=self.original_data[self.target],
                y_name=self.target,
                learning_rate=self.learn_rate,
                iterations=self.epochs,
                stop_threshold=self.loss_thresh,
            )
            plotter.plot_cost(np.array(self.loss))
            predictions = self.predict(self.original_data[self.pred])
            plotter.plot_data(
                feature1=self.original_data[self.pred],
                feature2=self.original_data[self.target],
                predictions=predictions,
                feature1_name=self.pred,
                feature2_name=self.target,
            )
            plotter.plot_residuals(
                feature=self.original_data[self.pred],
                residuals=predictions - self.original_data[self.target],
                feature_name=self.pred,
            )

            def _create_gif(frame) -> iter:
                """
                Create a frame for a GIF animation by updating the
                plots with historical data up to the current frame.

                Args:
                    frame: The current frame number.

                Returns:
                    iter
                """
                def _predict(x):
                    """
                    Predict the target value based on the linear
                    regression model parameters at a specific frame.

                    Args:
                        x: The input value for prediction.

                    Returns:
                        The predicted target value.
                    """
                    return (self.history["intercept"][frame]
                            + self.history["slope"][frame] * x)

                plotter.update_plots(
                    inter_history=np.array(self.history["intercept"][:frame]),
                    slope_history=np.array(self.history["slope"][:frame]),
                    loss_history=np.array(self.history["loss"][:frame]),
                    predictions=_predict(self.original_data[self.pred]),
                    epoch=frame,
                )
                time.sleep(0.01)

            plotter.initialize_gif_plot()
            ani = animation.FuncAnimation(
                fig=plotter.fig,
                func=_create_gif,
                frames=self.epochs,
                repeat=False
            )
            plotter.save_animation(ani, "images/training_animation.gif")

        except Exception as e:
            logging.error(f"Error plotting: {e}")
