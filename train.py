import asyncio
import time
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from config import config
from distro import Party, Server
from model import Model, SimpleRNN


class Trainer:
    """
    Performs learning with hybrid approach.
    Uses asyncio for emulating different parties.
    """
    model: Model
    train_loader: DataLoader
    valid_loader: DataLoader
    server: Server
    parties: List[Party]

    def __init__(self, model: Model, train_loader: DataLoader, valid_loader: DataLoader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model

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

    def fit(self):
        for epoch in range(config.n_epochs):
            epoch_start = time.time()
            for batch_idx, (features, target) in enumerate(self.train_loader):
                print(f"Epoch {epoch} Batch {batch_idx} Time {time.time() - epoch_start:.4}")

                # Divide one big batch into parties' batches
                batches_for_epoch = self.separate_clients_batches(features, target)

                # Train loop on parties
                encrypted_models = [
                    party.train_one_epoch(batch)
                    for party, batch
                    in zip(self.parties, batches_for_epoch)
                ]

                # Get mean params
                aggregate = self.server.aggregate_params(encrypted_models)

                # Decrypted
                new_params = self.server.decrypt_aggregate_params(aggregate)

                # Update before next epoch
                for party in self.parties:
                    party.update_params(new_params)

                # Test
                if batch_idx % config.test_every == 0:
                    # Update local model for test
                    self.update_params(new_params)
                    self.test_model()

    def test_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for features, target in self.valid_loader:
                features, target = features.to(config.device), target.to(config.device)

                output = self.model(features)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.valid_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.valid_loader.dataset),
            100. * correct / len(self.valid_loader.dataset)))

    def update_params(self, new_params: Tensor) -> None:
        """Copy data from new parameters into party's model."""
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data).to(config.device)

