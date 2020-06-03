import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import math
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from config import config
from train import Trainer

import string
import unicodedata


all_letters = string.ascii_letters + " .,;'-"


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


class NamesDataset(Dataset):
    samples: List[Tuple[str, str]]
    char2index: Dict[str, int]
    index2char: Dict[int, str]
    langs: List[str]

    def __init__(self):
        super().__init__()
        self.samples = []
        self.char2index = {'PAD': 0}
        self.index2char = {0: 'PAD'}
        self.langs = []

        self.read_samples()
        self.index_samples()

        random.shuffle(self.samples)

    def read_samples(self):
        dataset_path: Path = Path(__file__).parent / 'data' / 'names'

        # Autocompletion hack
        file: Path

        for file in dataset_path.iterdir():
            # Get language
            lang: str = file.stem
            self.langs.append(lang)

            # Autocompletion hack
            line: str

            # Get all names of this language
            for line in file.open('r'):
                # Remove whitespace
                name = line.strip()

                # Remove unicode symbols
                name = unicode_to_ascii(name)

                # Store sample
                sample = (name, lang)
                self.samples.append(sample)

    def index_samples(self):
        for (name, lang) in self.samples:
            chars = set(name)
            for char in chars:
                if char not in self.char2index:
                    index = len(self.char2index)
                    self.char2index[char] = index
                    self.index2char[index] = char

    def __getitem__(self, index: int) -> Tuple[int, int]:
        sample = self.samples[index]
        sequence = [self.char2index[char] for char in sample[0]]
        lang_id = self.langs.index(sample[1])

        return (sequence, lang_id)

    def __len__(self) -> int:
        return len(self.samples)


# Common DataLoader utils
def pad_seq(seq, max_length):
    """
    Return padded sequences of one size.
    """
    seq += [0 for i in range(max_length - len(seq))]
    return seq


def transform_batch(samples: List[Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
    """
    1. Sort sequences by length (may be not of equal sized)
    2. Pad sequences to max length
    """
    seqs = [sample[0] for sample in samples]
    langs = [sample[1] for sample in samples]

    # For sequences in batch, get array of lengths and pad with 0   1
    seq_lengths = [len(s) for s in seqs]
    seqs_padded = [pad_seq(s, max(seq_lengths)) for s in seqs]

    # Turn padded array into (batch_size x max_len) tensor
    seqs_tensor = torch.LongTensor(seqs_padded)
    langs_tensor = torch.LongTensor(langs)
    #  lengths_tensor = torch.LongTensor(seq_lengths)

    # Send to device (cpu or cuda)
    seqs_tensor = seqs_tensor.to(config.device)
    langs_tensor = langs_tensor.to(config.device)
    #  lengths_tensor = lengths_tensor.to(config.device)

    return seqs_tensor, langs_tensor

