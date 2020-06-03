from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class Config:
    n_parties: int = 10
    threshold: int = 3
    batch_size: int = 2000
    key_length: int = 256
    n_epochs: int = 400
    min_loss: float = 0.1
    learning_rate: float = 0.03
    hidden_size: int = 32
    test_every: int = 5
    device: torch.device = torch.device('cuda')
    use_dp: bool = True
    use_he: bool = True
    dataset: str = 'names'
    run_name: str = (
        f"{dataset}" +
        ("+dp" if use_dp else "") +
        ("+he" if use_he else "") +
        (f"+key={key_length}" if use_he else "") +
        f"+n_parties={n_parties}"
        f"+threshold={threshold}"
    )


config = Config()

Batch = List[Tuple[Tensor, int]]

