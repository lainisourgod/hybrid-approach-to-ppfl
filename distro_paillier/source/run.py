################################################################################
################################################################################
from distributed_paillier import NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PRIME_THRESHOLD, STATISTICAL_SECURITY_SECRET_SHARING, CORRECTNESS_PARAMETER_BIPRIMALITY
from distributed_paillier import generate_shared_paillier_key

if __name__ == "__main__":
    lengths = [256]
    
    for length in lengths:
        print('Next iteration')
        print('Number of players:', NUMBER_PLAYERS)
        print('Corruption threshold:', CORRUPTION_THRESHOLD)
        print('Small prime test threshold:', PRIME_THRESHOLD)
        print('Statistical security parameter of integer secret sharing:', STATISTICAL_SECURITY_SECRET_SHARING)
        print('Correctness parameter biprimality test:', CORRECTNESS_PARAMETER_BIPRIMALITY)

        Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta = generate_shared_paillier_key(keyLength = length)

        print('Key generation complete')
        
        import numpy as np

        message = np.array([14.3 ** 5, 14.3 ** 2]) # Of course 14!
        message2 = np.array([10.3 ** 3, 10 ** 4])
        expected = message + message2 * 2 + 1
        print('Encrypting test message', expected)

        Ciphertext = np.array([PublicKey.encrypt(elem) for elem in message]) 
        Ciphertext2 = np.array([PublicKey.encrypt(elem) for elem in message2]) 
        computed = Ciphertext + Ciphertext2 * 2 + 1
        print('Decrypting obtained ciphertext')

        decryption = np.array([
            Key.decrypt(elem, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta)
            for elem in computed
        ])

        if (expected == decryption).all():
            print('Hurray! Correctly decrypted encryption of', expected)

    print('Halleluja!')
    print('\n')
