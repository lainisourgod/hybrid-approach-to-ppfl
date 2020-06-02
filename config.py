from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class Config:
    n_clients: int = 3
    threshold: int = 1
    batch_size: int = 1000
    key_length: int = 128
    n_epochs: int = 300
    learning_rate: float = 0.05
    hidden_size: int = 64
    test_every: int = 3
    device: torch.device = torch.device('cuda')


config = Config()

Batch = List[Tuple[Tensor, int]]

