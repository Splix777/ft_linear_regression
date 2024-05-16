import pandas as pd
import numpy as np

class LinearRegression:
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.intercept_ = None
    self.slope_ = None

  def fit(self, X, y):
    """
    Trains the linear regression model using gradient descent.

    Args:
      X: A numpy array of shape (m, 1) representing the features.
      y: A numpy array of shape (m,) representing the target values.
    """

    m = len(X)  # Number of training examples
    self.intercept_ = 0  # Initialize intercept
    self.slope_ = 0  # Initialize slope

    for _ in range(1000):  # You might need to adjust the number of iterations
      y_predicted = self.predict(X)
      error = y - y_predicted
      self.slope_ -= self.learning_rate * (1/m) * np.sum(X * error)
      self.intercept_ -= self.learning_rate * (1/m) * np.sum(error)

  def predict(self, X):
    """
    Predicts the target values for new features.

    Args:
      X: A numpy array of shape (m, 1) representing the new features.

    Returns:
      A numpy array of shape (m,) containing the predicted target values.
    """

    return self.intercept_ + self.slope_ * X

# Example usage
data = pd.read_csv("cars.csv")  # Load data using pandas
X = data["km"].values.reshape(-1, 1)  # Features (km) as numpy array
y = data["price"].values  # Target (price) as numpy array

model = LinearRegression(learning_rate=0.005)  # Set custom learning rate
model.fit(X, y)

predicted_price = model.predict(np.array([50000]).reshape(-1, 1))
print(f"Estimated price for a car with {50000} km: €{predicted_price[0]:.2f}")
