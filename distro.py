# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
import random
from multiprocess import Pool
from typing import Iterable, List

import numpy as np
import phe
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer

from distro_paillier.source import distributed_paillier
from distro_paillier.source.distributed_paillier import generate_shared_paillier_key

from config import config
from model import Model


pool = Pool()


class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self):
        Key, _, _, _, PublicKey, _, _, SecretKeyShares, theta = generate_shared_paillier_key(
            keyLength=config.key_length,
            n=config.n_clients,
        )

        self.prikey = Key
        self.pubkey = PublicKey
        self.shares = SecretKeyShares
        self.theta = theta

    def aggregate_params(self, gradients_of_parties: np.ndarray) -> np.ndarray:
        """
        Take an array of encrypted parameters of models from all parties. Shape: (n_parties, ...)
        Return array of mean encrypted params.
        """
        return np.mean(gradients_of_parties, axis=0)

    def decrypt_aggregate_params(self, aggregate_params: np.ndarray) -> torch.Tensor:
        """
        Take encrypted aggregate params.
        Return decrypted params.
        """
        def decrypt_number(num: np.int64):
            return self.prikey.decrypt(
                num, config.n_clients, distributed_paillier.CORRUPTION_THRESHOLD, self.pubkey, self.shares, self.theta
            )

        # Firstly, flatten the tensor so we can decrypt every number separately
        flattened = aggregate_params.view(-1)

        # Decrypt every number in parallel
        decrypted_flat = Tensor(
            pool.map(decrypt_number, flattened),  # XXX: maybe should make flattened a list?
            dtype=torch.float64,
        )

        # Restore gradient shape
        decrypted = decrypted_flat.view_as(aggregate_params.shape)

        return decrypted


class Party:
    """
    Using public key can encrypt locally trained model.
    """
    def __init__(self, pubkey: phe.PaillierPublicKey, model: Model):
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=0.02)

        self.pubkey = pubkey

    def add_noise_to_param(self, param: Tensor):
        """
        Differential privacy simulation xD
        param: 1-d tensor
        """
        return torch.Tensor([num + random.random() for num in param])

    def train_one_epoch(self, batch) -> np.ndarray:
        """
        1. Take model parameters.
        2. Add noise to them.
        3. Encrypt them.

        Return List[np.ndarray[phe.EncryptedNumber]].
        """
        self.model.training_step(batch)

        params = self.model.parameters()
        encrypted_params: List[np.ndarray] = []

        # TODO: maybe map all this operations to pool?
        for param in params:
            flattened = param.data.view(-1)
            noised = self.add_noise_to_param()
            encrypted = np.ndarray(
                pool.map(self.pubkey.encrypt, noised),
                dtype=np.object,
            )
            encrypted_params.append(encrypted)

        return encrypted_params

    def update_params(self, params: torch.Tensor) -> None:
        with torch.no_grad():
            self.model.parameters = params

