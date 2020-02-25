# PoC Hybrid Approach

### What is done
* `fed.py` is a federated learning example that employes (approximate) hybrid approach from https://arxiv.org/pdf/1812.03224.pdf
* It is based on example from [python-paillier](https://github.com/data61/python-paillier/blob/master/examples/federated_learning_with_encryption.py)
* Model is just np.array of weights, simple SGD is implemented
* For Threshold Homomorphic Encryption, [this library](https://github.com/TNO/Distributed-Paillier-Cryptosystem) is used. I've done minor changes to `decrypt` function for it to work correctly with float numbers.
* Currently encrypted training for 100 iterations between 5 parties with key_size = 1024 is completed by 25s. Unencrypted training is 0.01s. very bad.
* May be done faster via proper multiprocessing and vector operations. But for now it's trash.
* For example: Crypten benchmarks now show that encrypted training is 50 times slower and 2 times less accurate)) Maybe I still have a chance
