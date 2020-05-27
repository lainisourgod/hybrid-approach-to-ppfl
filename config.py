from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class Config:
    n_clients: int = 3
    threshold: int = 1
    batch_size: int = 64
    key_length: int = 128
    n_epochs: int = 10
    learning_rate: float = 0.0001
    print_every: int = 10
    test_every: int = 30
    device: torch.device = torch.device('cuda')


config = Config()

Batch = List[Tuple[Tensor, int]]

