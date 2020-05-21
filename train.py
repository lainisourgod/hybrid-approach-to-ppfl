import asyncio
import time
from typing import List

import torch
from torch.utils.data import DataLoader

from config import config
from distro import Party, Server
from model import Model


class Trainer:
    """
    Performs learning with hybrid approach.
    Uses asyncio for emulating different parties.
    """
    train_loader: DataLoader
    valid_loader: DataLoader
    server: Server
    parties: List[Party]

    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Create default model that will be source of truth for every other model
        # Dimensions from MNIST
        self.model = Model(in_size=28 * 28, out_size=10)

        self.configure_system()

    def configure_system(self):
        """
        1. Instantiate the server and parties
        2. Generate private and public keys
        """
        self.server = Server()

        # Init parties with base model
        self.parties = [
            Party(model=self.model, pubkey=self.server.pubkey)
            for _ in range(config.n_clients)
        ]

    def separate_clients_batches(self, features, target):
        batches = []

        for party_index in range(config.n_clients):
            features_party = features[party_index::config.n_clients]
            target_party = target[party_index::config.n_clients]
            batches.append((features_party, target_party))

        return batches

    async def fit(self):
        for epoch in range(config.n_epochs):
            for batch_idx, (features, target) in enumerate(self.train_loader):
                batches_for_epoch = self.separate_clients_batches(features, target)

                encrypted_models = await asyncio.gather(
                    *(
                        party.train_one_epoch(batch)
                        for party, batch
                        in zip(self.parties, batches_for_epoch)
                    )
                )

                aggregate = self.server.aggregate_params(encrypted_models)

                # Decrypted
                new_params = self.server.decrypt_aggregate_params(aggregate)

                # Take gradient steps
                await asyncio.gather(
                    *(
                        party.update_params(new_params) for party in self.parties
                    )
                )

                print(f"Epoch {epoch} Batch {batch_idx}")

