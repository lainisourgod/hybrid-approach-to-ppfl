from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch.optim import Adam
from torch.functional import F

from config import config, Batch


class Model(torch.nn.Module):
    """Simple model for simple purposes"""

    optimizer: torch.optim.Optimizer

    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.linear = Linear(in_size, out_size)
        self.optimizer = Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(  # SOTA activation
            self.linear(  # SOTA model
                x.view(x.size(0), -1)  # Make it flat
            )
        )

    def training_step(self, batch) -> List[Parameter]:
        """Forward and backward pass"""
        features, target = batch
        self.optimizer.zero_grad()
        pred = self.forward(features)
        loss: Tensor = F.nll_loss(pred, target)
        loss.backward()
        self.optimizer.step()

        return list(self.parameters())


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

