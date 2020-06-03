from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class Config:
    n_clients: int = 3
    threshold: int = 1
    batch_size: int = 2000
    key_length: int = 128
    n_epochs: int = 500
    learning_rate: float = 0.03
    hidden_size: int = 32
    test_every: int = 5
    device: torch.device = torch.device('cuda')
    use_dp: bool = False
    use_he: bool = True


config = Config()

Batch = List[Tuple[Tensor, int]]

