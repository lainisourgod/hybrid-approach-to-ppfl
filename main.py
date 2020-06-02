import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import model
# Set model before any other import so all scripts use the same class
model.Model = model.SimpleRNN

from config import config, Batch
from train import Trainer


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


def configure_dataloaders(data_dir: Path) -> Tuple[DataLoader, DataLoader]:
    def create_loader(is_train_loader):
        return DataLoader(
            MNIST(
                data_dir,
                train=is_train_loader,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            # yield batches for every client
            batch_size=config.n_clients * config.batch_size,
        )

    return (create_loader(True), create_loader(False))


if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data/'
    data_dir.mkdir(parents=True, exist_ok=True)

    loaders = configure_dataloaders(data_dir)
    trainer = Trainer(
        train_loader=loaders[0],
        valid_loader=loaders[1],
    )

    try:
        trainer.fit()
    except KeyboardInterrupt:
        exit(0)

