import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import random
import math
from pathlib import Path
from typing import Iterator, List, Tuple


# Set model before any other import so all scripts use the same class
import model
model.Model = model.SimpleRNN

from config import config
from train import Trainer

import glob
import unicodedata
import string


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


def find_files(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# Build the category_lines dictionary, a list of lines per category
all_categories = []
samples: List[Tuple[str, str]] = []

for filename in find_files('data/names/*.txt'):
    category: str = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines: List[str] = readLines(filename)

    samples.extend((line, category) for line in lines)

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(IterableDataset):
    def __init__(self, samples):
        super().__init__()
        self.samples: List[Tuple[str, str]] = samples
        self.current = 0

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        features = []
        targets = []

        for _ in range(config.n_clients):
            sample = self.samples[self.current]
            line, category = sample

            category_tensor = torch.LongTensor([all_categories.index(category)])
            line_tensor = lineToTensor(line)

            features.append(line_tensor)
            targets.append(category_tensor)

            self.current += 1

        yield (features, targets)

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == "__main__":
    data_loaders = (
        DataLoader(NamesDataset(samples), batch_size=1),
        DataLoader(NamesDataset(samples), batch_size=1)
    )

    trainer = Trainer(
        train_loader=data_loaders[0],
        valid_loader=data_loaders[1],
    )

    try:
        trainer.fit()
    except KeyboardInterrupt:
        exit(0)

