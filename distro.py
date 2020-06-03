# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
import copy
import random
from functools import partial
from multiprocess import Pool, cpu_count
from typing import Callable, Iterable, List, Tuple

import diffprivlib as dp
import numpy as np
import phe
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.functional import F
from torch.optim import Adam, Optimizer

from distro_paillier.source import distributed_paillier
from distro_paillier.source.distributed_paillier import generate_shared_paillier_key

from config import config
from model import Model


n_cpus = cpu_count()

pool = Pool(processes=n_cpus - 3)
EncryptedParameter = np.ndarray  # [phe.EncryptedNumber]

use_pool = True


class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self):
        if not config.use_he:
            # pubkey mock
            self.pubkey = phe.PaillierPublicKey(1)
            return

        Key, _, _, _, PublicKey, _, _, SecretKeyShares, theta = generate_shared_paillier_key(
            keyLength=config.key_length,
            n=config.n_parties,
            t=config.threshold,
        )

        self.prikey = Key
        self.pubkey = PublicKey

        # decrypt takes one argument -- ciphertext to decode
        self.decrypt = partial(
            Key.decrypt,
            n=config.n_parties,
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

    def decrypt_param(self, param: List[phe.EncryptedNumber]) -> List[float]:
        if not config.use_he:
            return param

        if use_pool:
            return pool.map(self.decrypt, param, chunksize=100)
        else:
            return [self.decrypt(num) for num in param]

    def decrypt_aggregate_params(self, aggregate_params: np.ndarray) -> List[Tensor]:
        """
        Take encrypted aggregate params.
        Return decrypted params.
        """
        decrypted_params: List[Tensor] = []

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
    optimizer: torch.optim.Optimizer
    model: Model
    pubkey: phe.PaillierPublicKey
    randomiser: dp.mechanisms.Gaussian

    def __init__(self, pubkey: phe.PaillierPublicKey, model: Model):
        self.model: Model = copy.deepcopy(model).to(config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)

        self.pubkey = pubkey
        self.randomiser = dp.mechanisms.Gaussian().set_epsilon_delta(1, 1).set_sensitivity(0.1)

    def train_one_epoch(self, batch) -> List[EncryptedParameter]:
        """
        1. Train one epoch (including backward pass).
        2. Take parameters of model after epoch.
        3. Flatten parameters so we can apply transformations on every element of param.
        3. Add noise to them.
        4. Encrypt them.

        Return list of flattened encrypted params.
        """
        # Train for one epoch
        self.training_step(batch)

        # Get params after one epoch
        params = self.model.parameters()
        encrypted_params: List[np.ndarray] = []

        for param in params:
            # Flatten
            flattened = param.data.view(-1)

            # Add noise for diffential privacy
            noised = self.add_noise_to_param(flattened)

            # Convert to list so phe can work with it
            noised = noised.tolist()

            # Encrypt in multiprocessing
            encrypted: EncryptedParameter = self.encrypt_param(noised)
            encrypted_params.append(encrypted)

        return encrypted_params

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> List[Parameter]:
        """Forward and backward pass"""
        features, target = batch
        features, target = features.to(config.device), target.to(config.device)
        self.optimizer.zero_grad()

        pred = self.model(features)

        loss: Tensor = F.nll_loss(pred, target)

        loss.backward()
        self.optimizer.step()

    def add_noise_to_param(self, param: Parameter) -> Tensor:
        """
        Add noise from diffential privacy mechanism.
        param: 1-D (flattened) Parameter
        return Tensor of param's data with applied DP.
        """
        # DP mock
        if not config.use_dp:
            return param.data

        param_mean = param.data.mean()
        param_std = param.data.std()

        # Scale to normal distribution as distribution of randomiser is normal
        param_in_normal_distribution = (param.data - param_mean) / param_std

        randomised = [self.randomiser.randomise(num.item()) for num in param_in_normal_distribution]
        randomised_tensor = torch.Tensor(randomised).to(config.device)

        # Rescale results back
        randomised_tensor = randomised_tensor * param_std + param_mean

        # XXX: Make it less noisy to emulate that we sample from gaussian with lower std
        if config.use_he:
            randomised_tensor = (param.data * 0.9 + randomised_tensor * 1.1) / 2

        # Count difference between noised and original parameter data for debug
        diff_abs = randomised_tensor - param.data
        diff_rel = (diff_abs / param.data).abs()
        mean_diff = diff_rel.mean()

        #  print(f"diff: {mean_diff:.3}")

        return randomised_tensor

    def encrypt_param(self, param: List[float]) -> EncryptedParameter:
        # HE mock
        if not config.use_he:
            return np.array(param)

        encrypt = partial(self.pubkey.encrypt)
        if use_pool:
            return np.array(pool.map(encrypt, param, chunksize=300))
        else:
            return np.array([encrypt(num) for num in param])

    def update_params(self, new_params: Tensor) -> None:
        """Copy data from new parameters into party's model."""
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data).to(config.device)

