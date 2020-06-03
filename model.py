from __future__ import annotations

from typing import List, Union

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

        return x


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
        return x


class SimpleRNN(nn.Module):
    """
    Inputs:
        in_size = size of vocabulary
        out_size = num of categories
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(in_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, inputs):
        #  inputs = inputs.view(-1, 28, 28)

        embedded = self.embedding(inputs)

        self.rnn.flatten_parameters()
        out, _ = self.rnn(embedded)

        out = self.out(out)
        out = out[:, -1, :]  # at last timestep

        out = F.log_softmax(out, dim=1)

        return out


# Currently used model to import from trainer
Model = Union[SimpleLinear, SimpleCNN, SimpleRNN]

