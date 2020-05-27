# PoC Hybrid Approach

## Project structure
* `main.py` -- script for MNIST experiment (either Linear or CNN model). Defines dataset and dataloaders for trainer
* `rnn_main.py` -- script for RNN experiment
* `trainer.py` -- Trainer class that rules training and encryption stuff
* `distro.py` -- Party and Server classes that knows about model and about encryption and do it
* `distro_paillier/` -- sources of Threshold Paillier Homomorphic Encryption.
[this library](https://github.com/TNO/Distributed-Paillier-Cryptosystem) is used.
I've done minor changes to `decrypt` function for it to work correctly with float numbers.
* `config.py` -- containes singleton config object used by all modules.

## How to run
1. Install system dependencies for crypto stuff: `sudo apt install libgmp-dev libmpfr-dev libmpc-dev`.
2. Install python dependencies: `pip install -r requirements.txt`.
3. Run MNIST experiment `python main.py`.
4. Run RNN experiment `python rnn_main.py`.

