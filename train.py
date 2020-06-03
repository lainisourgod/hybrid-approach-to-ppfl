import asyncio
import time
from typing import List
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
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
    start_time: float = 0
    current_epoch: int = 0
    train_id: str = f'{config.run_name}-{str(uuid4())[:6]}'
    all_losses: List[float] = []
    f1_scores: List[float] = []

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
            for _ in range(config.n_parties)
        ]

    def separate_clients_batches(self, features, target):
        batches = []

        for party_index in range(config.n_parties):
            features_party = features[party_index::config.n_parties]
            target_party = target[party_index::config.n_parties]
            batches.append((features_party, target_party))

        return batches

    def fit(self):
        self.start_time = time.time()

        for epoch in range(1, config.n_epochs + 1):
            self.current_epoch = epoch

            self.fit_on_batch()

            # Test
            if epoch % config.test_every == 0:
                # Update local model for test
                self.test_model()
                # Plot
                self.plot()
                # End by loss
                if self.all_losses[-1] < config.min_loss:
                    break

    def fit_on_batch(self):
        for (features, target) in self.train_loader:
            # Divide one big batch into parties' batches
            batches_for_epoch = self.separate_clients_batches(features, target)

            # Train loop on parties
            encrypted_models = [
                party.train_one_epoch(batch)
                for party, batch
                in zip(self.parties, batches_for_epoch)
            ]

            # Get mean params
            aggregate: np.ndarray = self.server.aggregate_params(encrypted_models)

            # Decrypted
            new_params: List[Tensor] = self.server.decrypt_aggregate_params(aggregate)

            # Update before next epoch
            for party in self.parties:
                party.update_params(new_params)

            self.update_params(new_params)

    def test_model(self):
        self.model.eval()

        all_true = []
        all_pred = []

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for features, target in self.valid_loader:
                features, target = features.to(config.device), target.to(config.device)

                output = self.model(features)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                all_true.extend(target.tolist())
                all_pred.extend(pred.tolist())

        total = len(self.valid_loader.dataset)
        test_loss /= total

        self.all_losses.append(test_loss)

        percent_correct = 100. * correct / total

        f1 = f1_score(all_true, all_pred, average='weighted')
        self.f1_scores.append(f1)

        current_time = time.time() - self.start_time

        report = (
            f"Epoch: {self.current_epoch} "
            f"Test set: "
            f"Average loss: {test_loss:.4f} "
            f"Accuracy: {correct}/{total} ({percent_correct:.0f}%) "
            f"F1 score: {f1:.2f} "
            f"Time: {current_time:.0f}\n"
        )
        print(report)

    def plot(self):
        plt.figure()

        plt.xlabel('epoch')
        plt.ylabel('loss')

        epochs = list(range(1, self.current_epoch + 1, config.test_every))
        plt.plot(epochs, self.f1_scores)

        plt.savefig(Path(__file__).parent / 'experiment' / (self.train_id + '.png'))

        plt.close()


    def update_params(self, new_params: Tensor) -> None:
        """Copy data from new parameters into party's model."""
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data).to(config.device)

