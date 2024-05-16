import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataAnalysisClass:
    def __init__(self, data_dir, csv_file, pickle_path, pickle_model,
                 bonus=False, learning_rate=0.01, iterations=1500):
        self.data_path = data_dir
        self.data_csv = csv_file
        self.pickle_path = pickle_path
        self.pickle_model = pickle_model
        self.bonus = bonus
        self.alpha = learning_rate
        self.iter = iterations
        self.theta = np.array([0, 0])
        self.__load_data()
        self.__standardize_data()
        self.__train_dataset()
        self.__save_model()
        self.__plot_data()

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.data_path + self.data_csv)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.data_path} not found") from e

        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()
        arr = self.data.to_numpy(copy=True)
        self.km = arr[:, 0]
        self.price = arr[:, 1]
        if len(self.km) < 2 or len(self.price) < 2:
            raise ValueError("Not enough data to train the model")

    def __standardize_data(self):
        try:
            self.mean_km = np.mean(self.km)
            self.std_km = np.std(self.km)
            self.x_normalized = (self.km - self.mean_km) / self.std_km
        except ZeroDivisionError as e:
            raise ZeroDivisionError("Division by zero") from e

    @staticmethod
    def __hypothesis(theta, x):
        return np.dot(x, theta)

    def __cost_function(self, theta, x, y):
        m = len(y)
        return np.sum((self.__hypothesis(theta, x) - y) ** 2) / (2 * m)

    def __gradient_descent(self, theta, x, y, alpha=0.01, iter=1500):
        m = len(y)
        for _ in range(iter):
            hypothesis = self.__hypothesis(theta, x)
            loss = hypothesis - y
            gradient = x.T.dot(loss) / m
            theta = theta - alpha * gradient
        return theta

    def __train_dataset(self):
        self.losses = []
        self.biases = []
        self.weights = []
        x = np.column_stack(
            (self.x_normalized, np.ones_like(self.x_normalized)))
        theta = np.zeros(2)

        for _ in range(self.iter):
            cost = self.__cost_function(theta, x, self.price)
            self.losses.append(cost)
            self.biases.append(theta[0])
            self.weights.append(theta[1])

            theta = self.__gradient_descent(theta, x, self.price,
                                            self.alpha, self.iter)

        self.theta = theta

    def __save_model(self):
        try:
            with open(self.pickle_path + self.pickle_model, 'wb') as f:
                data_to_save = {"theta": self.theta,
                                "mean_km": self.mean_km,
                                "std_km": self.std_km}
                pickle.dump(data_to_save, f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.pickle_path} not found") from e

    def __plot_data(self):
        plt.scatter(self.km, self.price, color='blue')
        plt.plot(
            self.km,
            self.__hypothesis(
                self.theta,
                np.column_stack((
                    self.x_normalized,
                    np.ones_like(self.km)))))
        plt.title('Linear Regression')
        plt.legend(['Data', 'Linear regression'])
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.show()


if __name__ == "__main__":
    data_path = '/home/splix/Desktop/ft_linear_regression/csv_files/'
    data_csv = 'data.csv'
    pickle_file = '/home/splix/Desktop/ft_linear_regression/pickle_files/'
    pickle_model = 'model.pkl'
    bonus = False
    learning_rate = 0.01
    iterations = 500
    data_analysis = DataAnalysisClass(
        data_dir=data_path,
        csv_file=data_csv,
        pickle_path=pickle_file,
        pickle_model=pickle_model,
        bonus=bonus,
        learning_rate=learning_rate,
        iterations=iterations
    )
