# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
import random
from multiprocessing import Pool
from typing import Iterable, List

import numpy as np
import phe
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam, Optimizer

from distro_paillier.source import distributed_paillier
from distro_paillier.source.distributed_paillier import generate_shared_paillier_key

from config import config
from model import Model


EncryptedParameter = np.ndarray  # [phe.EncryptedNumber]


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

    def decrypt_number(self, num: phe.EncryptedNumber):
        return self.prikey.decrypt(
            num, config.n_clients, distributed_paillier.CORRUPTION_THRESHOLD, self.pubkey, self.shares, self.theta
        )

    def decrypt_aggregate_params(self, aggregate_params: np.ndarray) -> List[Tensor]:
        """
        Take encrypted aggregate params.
        Return decrypted params.
        """
        decrypted_params: List[Parameter] = []

        for param in aggregate_params:
            # To list so we can use decrypt on it
            flattened = param.tolist()

            decrypted_param = Tensor(
                [self.decrypt_number(num) for num in flattened],
                #  dtype=torch.float64,
            )

            decrypted_params.append(decrypted_param)

        return decrypted_params


class Party:
    """
    Using public key can encrypt locally trained model.
    """
    def __init__(self, pubkey: phe.PaillierPublicKey, model: Model):
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=0.02)

        self.pubkey = pubkey

    def add_noise_to_param(self, param: Parameter) -> Parameter:
        """
        Differential privacy simulation xD
        param: 1-d tensor
        """
        noise = torch.Tensor([random.random() for _ in range(len(param))])
        return param + noise

    def train_one_epoch(self, batch) -> List[EncryptedParameter]:
        """
        1. Train one epoch (including backward pass).
        2. Take parameters of model after epoch.
        3. Flatten parameters so we can apply transformations on every element of param.
        3. Add noise to them.
        4. Encrypt them.

        Return list of flattened encrypted params.
        """
        self.model.training_step(batch)

        params = self.model.parameters()
        encrypted_params: List[np.ndarray] = []

        # TODO: maybe map all this operations to pool?
        for param in params:
            # Flatten
            flattened = param.data.view(-1)

            # Add noise for diffential privacy
            noised = self.add_noise_to_param(flattened)

            # Convert to list so phe can work with it
            noised = noised.tolist()

            # Encrypt in multiprocessing
            encrypted: EncryptedParameter = np.array([self.pubkey.encrypt(num) for num in noised])
            encrypted_params.append(encrypted)

        return encrypted_params

    def update_params(self, new_params: Tensor) -> None:
        """
        Copy data from new parameters into party's model.
        Async because party can be somewhere far away...
        """
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data)

