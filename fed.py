import asyncio
import random
import time
from multiprocessing import pool
from typing import Iterable

import numpy as np
from sklearn.datasets import load_diabetes

import phe as paillier

from distro_paillier.source import distributed_paillier
from distro_paillier.source.distributed_paillier import generate_shared_paillier_key

seed = 43
np.random.seed(seed)

pool = pool(


def get_data(n_clients):
    """
    Import the dataset via sklearn, shuffle and split train/test.
    Return training, target lists for `n_clients` and a holdout test set
    """
    print("Loading data")
    diabetes = load_diabetes()
    y = diabetes.target
    X = diabetes.data
    # Add constant to emulate intercept
    X = np.c_[X, np.ones(X.shape[0])]

    # The features are already preprocessed
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm, :], y[perm]

    # Select test at random
    test_size = 50
    test_idx = np.random.choice(X.shape[0], size=test_size, replace=False)
    train_idx = np.ones(X.shape[0], dtype=bool)
    train_idx[test_idx] = False
    X_test, y_test = X[test_idx, :], y[test_idx]
    X_train, y_train = X[train_idx, :], y[train_idx]

    # Split train among multiple clients.
    # The selection is not at random. We simulate the fact that each client
    # sees a potentially very different sample of patients.
    X, y = [], []
    step = int(X_train.shape[0] / n_clients)
    for c in range(n_clients):
        X.append(X_train[step * c: step * (c + 1), :])
        y.append(y_train[step * c: step * (c + 1)])

    return X, y, X_test, y_test


def mean_square_error(y_pred, y):
    """ 1/m * \sum_{i=1..m} (y_pred_i - y_i)^2 """
    return np.mean((y - y_pred) ** 2)


class Server:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self, key_length, n_clients):
        Key, _, _, _, PublicKey, _, _, SecretKeyShares, theta = generate_shared_paillier_key(keyLength = key_length)

        self.prikey = Key
        self.pubkey = PublicKey
        self.shares = SecretKeyShares
        self.theta = theta

        self.n_clients = n_clients

    def aggregate_gradients(self, gradients: Iterable[np.array]):
        return np.sum(gradients, axis=0) / self.n_clients

    def decrypt_gradient(self, gradient):
        dec = np.array([
            self.prikey.decrypt(
                num, self.n_clients, distributed_paillier.CORRUPTION_THRESHOLD, self.pubkey, self.shares, self.theta
            )
            for num in gradient
        ], dtype=np.float64)
        return dec


class Net:
    """
    Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.
    """

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.weights = np.zeros(X.shape[1])

    def predict(self, X):
        """Use model"""
        return X.dot(self.weights)

    def fit(self, n_iter, eta=0.01):
        """Linear regression for n_iter"""
        for _ in range(n_iter):
            gradient = self.compute_gradient()
            self.gradient_step(gradient, eta)

    def compute_gradient(self):
        """
        Compute the gradient of the current model using the training set
        """
        delta = self.predict(self.X) - self.y
        return delta.dot(self.X) / len(self.X)

    def gradient_step(self, gradient, eta=0.01):
        """Update the model with the given gradient"""
        self.weights -= eta * gradient


class Party:
    """
    Using public key can encrypt locally computed gradients.
    """
    def __init__(self, name, X, y, pubkey):
        self.name = name
        self.model = Net(X, y)
        self.pubkey = pubkey

    def get_noise(self):
        """
        Differential privacy simulation xD
        """
        return random.random() * 0.01

    async def compute_partial_gradient(self):
        """
        1. Compute gradient
        2. Add noise to it
        3. Encrypt it
        """
        gradient = self.model.compute_gradient()
        noisy_gradient = gradient + self.get_noise()
        sy.pool().map(public_key.encrypt, inputs)
        encrypted_gradient = np.array([self.pubkey.encrypt(i) for i in noisy_gradient])
        return encrypted_gradient


async def hybrid_learning(X, y, X_test, y_test, config):
    """
    Performs learning with hybrid approach.
    Uses asyncio for emulating different parties.
    """
    n_clients = config['n_clients']
    n_iter = config['n_iter']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the server and generate private and public keys
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    server = Server(key_length=config['key_length'], n_clients=n_clients)

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = [
        Party(name, train_data, target_data, server.pubkey)
        for name, train_data, target_data in zip(names, X, y)
    ]

    # The federated learning with gradient descent
    print(f'Running distributed gradient aggregation for {n_iter} iterations')

    for i in range(n_iter):
        gradients = await asyncio.gather(
            *(
                client.compute_partial_gradient() for client in clients
            )
        )

        aggregate = server.aggregate_gradients(gradients)

        # Decrypted
        aggregate = server.decrypt_gradient(aggregate)

        # Take gradient steps
        for c in clients:
            c.model.gradient_step(aggregate, config['eta'])

        if i % 10 == 1:
            print(f'Epoch {i}')

    print('Error (MSE) that each client gets after running the protocol:')
    for c in clients:
        y_pred = c.model.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


def local_learning(X, y, X_test, y_test, config):
    n_clients = config['n_clients']
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]

    # Instantiate the clients.
    # Each client gets the public key at creation and its own local dataset
    clients = []
    for i in range(n_clients):
        clients.append(Party(names[i], X[i], y[i], None))

    # Each client trains a linear regressor on its own data
    print('Error (MSE) that each client gets on test set by '
          'training only on own local data:')
    for c in clients:
        c.model.fit(config['n_iter'], config['eta'])
        y_pred = c.model.predict(X_test)
        mse = mean_square_error(y_pred, y_test)
        print('{:s}:\t{:.2f}'.format(c.name, mse))


from contextlib import contextmanager
@contextmanager
def timer():
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[elapsed time: %.2f s]' % (time.perf_counter() - time0))


if __name__ == '__main__':
    config = {
        'n_clients': distributed_paillier.NUMBER_PLAYERS,
        'key_length': distributed_paillier.DEFAULT_KEYSIZE,
        'n_iter': 100,
        'eta': 1.5,
    }

    # load data, train/test split and split training data between clients
    X, y, X_test, y_test = get_data(n_clients=config['n_clients'])

    # first each hospital learns a model on its respective dataset for comparison.
    with timer():
        local_learning(X, y, X_test, y_test, config)

    with timer():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hybrid_learning(X, y, X_test, y_test, config))

