# Distributed Key Generation for the Paillier Cryptosystem

This repository contains the code for the distributed key generation protocol for the Paillier cryptosystem developed by TNO within the [Shared Research Program Cyber Security](https://www.tno.nl/en/collaboration/partners-of-tno/shared-research-program-srp-cyber-security/).


## Current Status

The code is functional, but it is for now only a "local" Python prototype for illustration purposes: it consists of a single process that simply emulates behavior of the players.

The actually distributed version, with different processes per player, is still under testing, and is therefore not published here yet.


## Dependencies

Python 3 is required.

Python dependencies are listed in the requirements.txt file;
`pip install -r requirements.txt` (or possibly `pip3 install -r requirements.txt`) will install them.

The gmpy Python library depends on a few GNU multiple precision arithmetic libraries: you can get them with `sudo apt install libgmp-dev libmpfr-dev libmpc-dev`.


## Instructions

The main source code file, contains an executable "main": `cd` to source and run `python3 distributed_paillier.py` to witness the key generation, encryption (by means of the public key) of 14, and decryption (by means of the secret key shares) of the resulting ciphertext.

Relevant functions of distributed_paillier are:

- generate_shared_paillier_key(), to generate a Paillier public key, shares of the secret key, and public masking value theta;
- the PublicKey obtained in this way is an instance of the PaillierPublicKey class of the Python "phe" library, you can use it to encrypt a value with PublicKey.encrypt(value);
- decryption of a given Ciphertext can be performed with Key.decrypt(Ciphertext, n, t, PublicKey, SecretKeyShares, theta), where n and t denote the total number of players and the privacy threshold, respectively, and PublicKey, SecretKeyShares and theta are obtain by the above key generation.
