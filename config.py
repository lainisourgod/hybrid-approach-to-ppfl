from dataclasses import dataclass
from typing import List, Tuple

from torch import Tensor


@dataclass
class Config:
    n_clients: int = 5
    batch_size: int = 16
    key_length: int = 128
    n_epochs: int = 10
    learning_rate: float = 0.001


config = Config()

Batch = List[Tuple[Tensor, int]]

