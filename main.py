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

from rnn_data import NamesDataset, transform_batch
from model import SimpleRNN, SimpleLinear
from config import config, Batch
from train import Trainer


@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


def configure_dataloaders(data_dir: Path) -> Tuple[DataLoader, DataLoader]:
    if config.dataset == 'mnist':
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
                batch_size=config.n_parties * config.batch_size,
            )
    else:
        def create_loader(is_train_loader):
            return DataLoader(
                NamesDataset(),
                # yield batches for every client
                batch_size=config.n_parties * config.batch_size,
                collate_fn=transform_batch
            )

    return (create_loader(True), create_loader(False))


def configure_model() -> torch.nn.Module:
    if config.dataset == 'mnist':
        model = SimpleLinear(in_size=28 * 28, out_size=10)
    else:
        num_langs = len(loaders[0].dataset.langs)
        vocab_size = len(loaders[0].dataset.char2index)

        model = SimpleRNN(in_size=vocab_size, hidden_size=config.hidden_size, out_size=num_langs)

    return model


if __name__ == '__main__':
    data_dir = Path(__file__).parent / 'data/'
    data_dir.mkdir(parents=True, exist_ok=True)

    loaders = configure_dataloaders(data_dir)

    model = configure_model()

    trainer = Trainer(
        model=model,
        train_loader=loaders[0],
        valid_loader=loaders[1],
    )

    try:
        trainer.fit()
    except KeyboardInterrupt:
        exit(0)

