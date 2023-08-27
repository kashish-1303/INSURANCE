import numpy as np


class LinearRegression:
    def _init_(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zero(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X.T, self.weights) + self.bias

            dw = (1/n_samples)* np.dot(X, (y_pred - y))
            db = (1 / n_samples) * np.dot(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred