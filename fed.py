import asyncio
import random
import time
from multiprocess import Pool
from typing import Iterable

import numpy as np
from sklearn.datasets import load_diabetes

import phe as paillier

from distro_paillier.source import distributed_paillier
from distro_paillier.source.distributed_paillier import generate_shared_paillier_key

from model import Net


seed = 43
np.random.seed(seed)

# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
pool = Pool()


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

    def decrypt_gradient(self, gradient: np.array):
        def decrypt_number(num: np.int64):
            return self.prikey.decrypt(
                num, self.n_clients, distributed_paillier.CORRUPTION_THRESHOLD, self.pubkey, self.shares, self.theta
            )

        decrypted = np.array(list(pool.map(decrypt_number, gradient)), dtype=np.float64)
        #  decrypted = np.array([
            #  self.prikey.decrypt(
                #  num, self.n_clients, distributed_paillier.CORRUPTION_THRESHOLD, self.pubkey, self.shares, self.theta
            #  )
            #  for num in gradient
        #  ], dtype=np.float64)
        return decrypted

def encrypt_vector(pubkey, vector: np.array) -> np.array:
    assert not np.array([np.isnan(val) for val in vector]).any()
    results = pool.map(pubkey.encrypt, vector)
    return np.array(results)


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
        #  encrypted_gradient_global = encrypt_vector(self.pubkey, noisy_gradient)
        encrypted_gradient_with_self = np.array(
            list(
                pool.map(self.pubkey.encrypt, noisy_gradient)
            )
        )
        #  encrypted_gradient_one_process = np.array([self.pubkey.encrypt(i) for i in noisy_gradient])

        encrypted_gradient = encrypted_gradient_with_self

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
        print("Computing gradients")
        with timer():
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
        'n_iter': 10,
        'eta': 1.5,
    }

    # load data, train/test split and split training data between clients
    X, y, X_test, y_test = get_data(n_clients=config['n_clients'])

    # first each hospital learns a model on its respective dataset for comparison.
    #  with timer():
        #  local_learning(X, y, X_test, y_test, config)

    with timer():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hybrid_learning(X, y, X_test, y_test, config))

