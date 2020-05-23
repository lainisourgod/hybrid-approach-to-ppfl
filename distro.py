# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
import copy
import random
from multiprocess import Pool, cpu_count
from typing import Callable, Iterable, List

import diffprivlib as dp
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


n_cpus = cpu_count()

pool = Pool(processes=n_cpus - 3)
EncryptedParameter = np.ndarray  # [phe.EncryptedNumber]

use_pool = True


decrypt: Callable


class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self):
        Key, _, _, _, PublicKey, _, _, SecretKeyShares, theta = generate_shared_paillier_key(
            keyLength=config.key_length,
            n=config.n_clients,
            t=config.threshold,
        )

        self.prikey = Key
        self.pubkey = PublicKey
        self.shares = SecretKeyShares
        self.theta = theta

        global decrypt
        decrypt = lambda cipher: Key.decrypt(
                    Ciphertext=cipher,
                    n=config.n_clients,
                    t=config.threshold,
                    PublicKey=PublicKey,
                    SecretKeyShares=SecretKeyShares,
                    theta=theta
                )

    def aggregate_params(self, gradients_of_parties: np.ndarray) -> np.ndarray:
        """
        Take an array of encrypted parameters of models from all partieprime_threshold)
        Return array of mean encrypted params.
        """
        return np.mean(gradients_of_parties, axis=0)

    def decrypt_param(self, param):
        if use_pool:
            return pool.map(decrypt, param)
        else:
            return [decrypt(num) for num in param]

    def decrypt_aggregate_params(self, aggregate_params: np.ndarray) -> List[Tensor]:
        """
        Take encrypted aggregate params.
        Return decrypted params.
        """
        decrypted_params: List[Parameter] = []

        for param in aggregate_params:
            # To list so we can use decrypt on it
            flattened = param.tolist()

            decrypted_param = Tensor(self.decrypt_param(flattened))

            decrypted_params.append(decrypted_param)

        return decrypted_params


class Party:
    """
    Using public key can encrypt locally trained model.
    """
    def __init__(self, pubkey: phe.PaillierPublicKey, model: Model):
        self.model: Model = copy.deepcopy(model).to(config.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.02)

        self.pubkey = pubkey
        self.randomiser = dp.mechanisms.Gaussian().set_epsilon_delta(0.5, 0.001).set_sensitivity(0.1)

    def add_noise_to_param(self, param: Parameter) -> Tensor:
        """
        Add noise from diffential privacy mechanism.
        param: 1-D (flattened) Parameter
        return Tensor of param's data with applied DP.
        """
        param_mean = param.data.mean()
        param_std = param.data.std()

        # Scale to normal distribution as distribution of randomiser is normal
        param_in_normal_distribution = (param.data - param_mean) / param_std

        randomised = [self.randomiser.randomise(num.item()) for num in param_in_normal_distribution]
        randomised_tensor = torch.Tensor(randomised).to(config.device)

        # Rescale results back
        randomised_tensor = randomised_tensor * param_std + param_mean

        return randomised_tensor

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
            print(f"diff: {((noised - flattened).abs() / flattened).mean():.3}")

            # Convert to list so phe can work with it
            noised = noised.tolist()

            # Encrypt in multiprocessing
            encrypted: EncryptedParameter = self.encrypt_param(noised)
            encrypted_params.append(encrypted)

        return encrypted_params

    def encrypt_param(self, param):
        if use_pool:
            return np.array(pool.map(self.pubkey.encrypt, param))
        else:
            return np.array([self.pubkey.encrypt(num) for num in param])

    def update_params(self, new_params: Tensor) -> None:
        """
        Copy data from new parameters into party's model.
        Async because party can be somewhere far away...
        """
        self.model.update_params(new_params)

