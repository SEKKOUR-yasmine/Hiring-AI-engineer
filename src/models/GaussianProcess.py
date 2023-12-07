import numpy as np
from typing import Tuple


class GaussianProcess:
    def __init__(self, kernel, noise: float = 1e-5) -> None:
        """
        Initialize a Gaussian Process with a specified kernel and noise level.

        Parameters:
        kernel: An instance of a kernel class.
        noise (float): The noise level in the Gaussian Process.
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process model to training data.

        Parameters:
        X (np.ndarray): The input training data.
        y (np.ndarray): The target training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the mean and covariance of the Gaussian Process at specified input points.

        Parameters:
        X (np.ndarray): The input points for which predictions are made.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mean and covariance of the predictions.
        """
        # Covariance between new input points
        K_ss = self.kernel.compute(X[:, np.newaxis], X)

        # Covariance between new input points and training data
        K_s = self.kernel.compute(X[:, np.newaxis], self.X_train)

        # Covariance between training data points
        K = self.kernel.compute(self.X_train[:, np.newaxis], self.X_train)

        K_inv = np.linalg.inv(K + self.noise * np.identity(K.shape[0]))

        mu_s = K_s.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        return mu_s, cov_s
