from typing import List

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.functional import F

from config import config, Batch


class SimpleLinear(torch.nn.Module):
    """Simple model for simple purposes"""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Make it flat
        x = self.linear(x)
        x = torch.relu(x)

        output = F.log_softmax(x, dim=1)

        return output


class SimpleCNN(torch.nn.Module):
    """Simple model for simple purposes"""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net:
    """
    Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.
    """

    def __init__(self, X: np.array, y: np.array):
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def predict(self, X) -> np.array:
        """Use model"""
        return X.dot(self.weights)

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def compute_gradient(self) -> np.array:
        """
        Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient


# Currently used model to import from trainer
Model = SimpleLinear

