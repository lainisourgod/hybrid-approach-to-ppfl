# PoC Hybrid Approach

### What is done
* `fed.py` is a federated learning example that employes (approximate) hybrid approach from https://arxiv.org/pdf/1812.03224.pdf
* It is based on example from [python-paillier](https://github.com/data61/python-paillier/blob/master/examples/federated_learning_with_encryption.py)
* Model is just np.array of weights, simple SGD is implemented
* For Threshold Homomorphic Encryption, [this library](https://github.com/TNO/Distributed-Paillier-Cryptosystem) is used. I've done minor changes to `decrypt` function for it to work correctly with float numbers.
* Currently encrypted training for 100 iterations between 5 parties with key_size = 1024 is completed by 25s. Unencrypted training is 0.01s. very bad.
* May be done faster via proper multiprocessing and vector operations. But for now it's trash.
* For example: Crypten benchmarks now show that encrypted training is 50 times slower and 2 times less accurate)) Maybe I still have a chance


## Original description of fed.py from TNO/Distributed-Paillier-Cryptosystem
This example involves learning using sensitive medical data from multiple hospitals
to predict diabetes progression in patients. The data is a standard dataset from
sklearn[1].

Recorded variables are:
- age,
- gender,
- body mass index,
- average blood pressure,
- and six blood serum measurements.

The target variable is a quantitative measure of the disease progression.
Since this measure is continuous, we solve the problem using linear regression.

The patients' data is split between 3 hospitals, all sharing the same features
but different entities. We refer to this scenario as horizontally partitioned.

The objective is to make use of the whole (virtual) training set to improve
upon the model that can be trained locally at each hospital.

50 patients will be kept as a test set and not used for training.

An additional agent is the 'server' who facilitates the information exchange
among the hospitals under the following privacy constraints:

1) The individual patient's record at each hospital cannot leave the premises,
   not even in encrypted form.
2) Information derived (read: gradients) from any hospital's dataset
   cannot be shared, unless it is first encrypted.
3) None of the parties (hospitals AND server) should be able to infer WHERE
   (in which hospital) a patient in the training set has been treated.

Note that we do not protect from inferring IF a particular patient's data
has been used during learning. Differential privacy could be used on top of
our protocol for addressing the problem. For simplicity, we do not discuss
it in this example.

In this example linear regression is solved by gradient descent. The server
creates a paillier public/private keypair and does not share the private key.
The hospital clients are given the public key. The protocol works as follows.
Until convergence: hospital 1 computes its gradient, encrypts it and sends it
to hospital 2; hospital 2 computes its gradient, encrypts and sums it to
hospital 1's; hospital 3 does the same and passes the overall sum to the
server. The server obtains the gradient of the whole (virtual) training set;
decrypts it and sends the gradient back - in the clear - to every client.
The clients then update their respective local models.

From the learning viewpoint, notice that we are NOT assuming that each
hospital sees an unbiased sample from the same patients' distribution:
hospitals could be geographically very distant or serve a diverse population.
We simulate this condition by sampling patients NOT uniformly at random,
but in a biased fashion.
The test set is instead an unbiased sample from the overall distribution.

From the security viewpoint, we consider all parties to be "honest but curious".
Even by seeing the aggregated gradient in the clear, no participant can pinpoint
where patients' data originated. This is true if this RING protocol is run by
at least 3 clients, which prevents reconstruction of each others' gradients
by simple difference.

This example was inspired by Google's work on secure protocols for federated
learning[2].

[1]: http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
[2]: https://research.googleblog.com/2017/04/federated-learning-collaborative.html

Dependencies: numpy, sklearn

