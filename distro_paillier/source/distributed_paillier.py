#!/usr/bin/env/python3
# TNO, G. Spini, T. Attema
"""
    Code to distributely generate a shared Paillier keypair
    and perform shared decryption.
    Notice that this is a "local" version only for illustration purposes.
    The actually distributed version, with different processes that can be run
    on different machines, is still work-in-progress.
"""

import math
import time
import copy

from fractions import gcd
import secrets
import sympy
from sympy.ntheory import jacobi_symbol
from gmpy2 import powmod, invert
import phe
from phe.util import invert

from .shamir_secret_sharing import ShamirSecretSharingScheme as Shamir
from .shamir_secret_sharing_integers import ShamirSecretSharingIntegers as IntegerShamir
from .shamir_secret_sharing_integers import mult_list


DEFAULT_KEYSIZE = 1024 # Bit-length of p and q
NUMBER_PLAYERS = 4
CORRUPTION_THRESHOLD = 1
PRIME_THRESHOLD = 20000 # We will check p and q for prime factors up to THRESHOLD. This value should be chosen to minimize runtime.
MAX_ITERATIONS = 10 # Maximum number of times we will check if rq=0. This value should be chosen to minimize runtime.
CORRECTNESS_PARAMETER_BIPRIMALITY = 100 # Probability that public key is not biprime = 2^-(STATISTICAL_SECURITY)
STATISTICAL_SECURITY_SECRET_SHARING = 40 # Statistical security parameter for Shamir over the integers


################################################################################
"""
    Main script
"""
################################################################################

def generate_shared_paillier_key(
    keyLength=DEFAULT_KEYSIZE,
    n=NUMBER_PLAYERS,
    t=CORRUPTION_THRESHOLD,
    prime_threshold=PRIME_THRESHOLD,
    it=MAX_ITERATIONS,
    correctParamPrime=CORRECTNESS_PARAMETER_BIPRIMALITY,
    statSecShamir=STATISTICAL_SECURITY_SECRET_SHARING
):
    """
    Main code to obtain a Paillier public key N
    and shares of the private key.
    """
    primeLength = keyLength // 2
    length = 2 * (primeLength + math.ceil((math.log2(n))))
    shamirP = sympy.nextprime(2**length)

    smallPrimeTest = True

    if primeLength < math.log(prime_threshold,2):
        prime_threshold = 1

    print('Bit-length of primes: ', primeLength)

    Key = PaillierSharedKey(primeLength, n, t, prime_threshold, it )

    print('Starting generation of p, q and N...')
    success = 0
    counter_smallPrime = 0
    counter_biprime = 0
    startTime = time.time()

    # Here we define the small primes considered in small prime test.
    # The generated p and q are both 3 mod 4, hence we do not have to check for divisibility by 2.
    primeList = [p for p in sympy.primerange(3, prime_threshold + 1)]

    while success == 0:
        # Generate the factors p and q
        pShares = Key.generate_prime_vector(n, primeLength)
        qShares = Key.generate_prime_vector(n, primeLength)

        # Then obtain N
        N = compute_product(pShares, qShares, n, t, shamirP)

        # Test bi-primality of N
        [success, smallPrimeTestFail] = is_biprime(n, N, pShares, qShares, correctParamPrime, smallPrimeTest, primeList)

        if smallPrimeTestFail:
            counter_smallPrime += 1
        elif not(success):
            counter_biprime += 1

    print('Generation successful.')
    print('Number of insuccesfull trials small prime test: ', counter_smallPrime)
    print('Number of insuccesfull trials biprime test: ', counter_biprime)
    print('Elapsed time: ', math.floor(time.time()-startTime), 'seconds.')

    PublicKey = phe.PaillierPublicKey(N)

    # So long for the public key. Let's look at the secret key(s):

    # Instatiate secret-sharing scheme for lambda and beta.
    Int_SS_Scheme = IntegerShamir(statSecShamir, N, n, t)

    LambdaShares = obtain_lambda_shares(Int_SS_Scheme, pShares, qShares, N)

    success = 0
    while success == 0:
        # Generate shares of random mask
        BetaShares = share_random_element(N, Int_SS_Scheme)

        # Obtain shares of secret key.
        # Notice that here we mask lambda over the integers:
        # since we will only reveal this product modulo N, this product perfectly masks lambda mod N.
        SecretKeyShares = LambdaShares * BetaShares

        success, theta = compute_theta(SecretKeyShares, N)

    return Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta


################################################################################
"""
    Classes
"""
################################################################################

class PaillierSharedKey():
    """
    Class containing relevant attributes and methods of a shared paillier key.
    """

    def __init__(self, keyLength, n, t, threshold, iterations):
        # The keyLength is the size (in bits) of the RSA-modulus N.
        # Note that N is the product of two primes.
        # Hence, the size of the primes is half the keyLength.
        self.keyLength = keyLength
        self.n = n
        self.t = t
        self.threshold = threshold
        self.iterations = iterations

    def generate_prime_vector(self, n, length):
        """
         Everty party picks a random share in [2^(length-1),2^length -1].
         Party 1 randomly samples an integer equal to 3 modulo 4.
         The other parties sample integers equal to 0 modulo 4.
         The resulting integers are additive shares of an integer equal to
         3 modulo 4.
         """
        primeArray = {i:1 for i in range(1,n+1)}

        while primeArray[1] % 4 != 3:
            primeArray[1] = 2**(length-1)+secrets.randbits(length-1)
        for i in range(2,n+1):
            while primeArray[i] % 4 != 0:
                primeArray[i] = 2**(length-1)+secrets.randbits(length-1)
        return primeArray

    def decrypt(self, Ciphertext, n, t, PublicKey, SecretKeyShares, theta):
        """
        Decryption functionality (no decoding).
        It is assumed that Ciphertext and PublicKey are class instances from
        the phe library, and that SecretKeyShares is a Shares class instance
        from Shamir secret sharing over the integers.
        NB: Now assuming reconstruction set = {1, ..., t+1}
        NB: this code perfoms both local partial decryption
        AND combination of them.
        The two will need to be separated to securely run the code on several
        machines.
        """
        nFac = math.factorial(n)

        c = Ciphertext.ciphertext()
        N = PublicKey.n
        NSquare = N**2
        shares: dict = SecretKeyShares.shares
        degree: int = SecretKeyShares.degree

        # NB: Here the reconstruction set is implicity defined, but any
        # large enough subset of shares will do.
        # shares: {1: 1254t4, 2: 1182471924, 3: 381284}
        # move to list because WTF we use dict with int keys here???
        shares = list(shares.values())
        reconstruction_shares = shares[:degree+1]

        lagrange_interpol_enum = [
            mult_list([j for j in range(1, len(reconstruction_shares) + 1) if i != j])
            for i in range(1, len(reconstruction_shares) + 1)
        ]
        lagrange_interpol_denom = [
            mult_list([(j - i) for j in range(1, len(reconstruction_shares) + 1) if i != j])
            for i in range(1, len(reconstruction_shares) + 1)
        ]
        exponents = [
            (nFac * lagrange_interpol_enum[i] * reconstruction_shares[i]) // lagrange_interpol_denom[i]
            for i in range(len(reconstruction_shares))
        ]

        # Notice that the partial decryption is already raised to the power given
        # by the Lagrange interpolation coefficient
        partialDecryptions = [ powmod(c, exp, NSquare) for exp in exponents ]
        combinedDecryption = mult_list(partialDecryptions) % NSquare

        if (combinedDecryption-1) % N != 0:
            print("Error, combined decryption minus one not divisible by N")

        message = ( (combinedDecryption-1)//N * invert(theta, N) ) %N

        # Decode (bug of float numbers)
        encoded = phe.EncodedNumber(PublicKey, message, Ciphertext.exponent)
        decoded = encoded.decode()

        return decoded


################################################################################
"""
    High-level functions
"""
################################################################################

def compute_product(dict1, dict2, n, t, shamirP):
    """
    dict1 and dict2 are interpreted as additive share vectors.
    Code will convert them to Shamir,
    then compute and reveal their product.
    """
    Scheme = Shamir(shamirP, n, t)
    # Shamir-share each share and add them (addition done in reshare)
    shares1 = reshare(dict1, Scheme)
    shares2 = reshare(dict2, Scheme)
    # Then multiply and open
    shares = shares1 * shares2
    value = shares.reconstruct_secret()
    return value


def is_biprime(n, N, pShares, qShares, statSec, smallPrimeTest, primeList):
    """
    Distributed bi-primality test.
    """

    if smallPrimeTest == True:
        for P in primeList:
            if N % P == 0:
                smallPrimeTestFail = True
                return [False, smallPrimeTestFail]

    smallPrimeTestFail = False

    is_biprime_out = True
    counter = 0
    startTime = time.time()
    while is_biprime_out and counter < statSec:
        testValue = sum([secrets.randbelow(N) for _ in range(n)]) % N
        if jacobi_symbol(testValue, N) == 1:
            is_biprime_out = is_biprime_parametrized(N, pShares, qShares, testValue)
            if is_biprime_out:
                counter += 1
    return [is_biprime_out, smallPrimeTestFail]


def obtain_lambda_shares(SSScheme, pShares, qShares, N):
    """
    Obtain Shamir shares over the integers of lambda = (p-1)(q-1),
    where N=pq and p=sum(pShares), q=sum(qShares)
    """

    lambdaAdditiveShares = {1:N - pShares[1] - qShares[1] +1}
    for i in range(1, SSScheme.n):
        lambdaAdditiveShares[i+1]=-pShares[i+1] - qShares[i+1]

    lambdaShares = reshare(lambdaAdditiveShares, SSScheme)
    return lambdaShares

def compute_theta(SecretKeyShares, N):
    n = len(SecretKeyShares.shares)

    SharesModN = copy.deepcopy(SecretKeyShares)
    SharesModN.shares = { key: (value % N) for key,value in SharesModN.shares.items() }

    y = SharesModN.reconstruct_secret(modulus=N)

    theta = (y * math.factorial(n)**3) % N

    if gcd(theta, N) == 0:
        return 0, None, None

    return 1, theta


################################################################################
"""
    Auxiliary functions
"""
################################################################################

def reshare(inputDict, Scheme):
    """
    Convert additive (n-out-of-n) sharing to secret sharing given by Scheme.
    """
    shares = Scheme.share_secret(0)
    for i in inputDict.keys():
        shares += Scheme.share_secret(inputDict[i])
    return shares


def share_random_element(modulus, Scheme):
    """
    Obtain Shamir secret-sharing of random element mod modulus.
    """
    randomList = [secrets.randbelow(modulus) for _ in range(Scheme.n)]

    randomShares = Scheme.share_secret(randomList[0])
    for i in range(1,Scheme.n):
        randomShares += Scheme.share_secret(randomList[i])
    return randomShares


def is_biprime_parametrized(N, pShares, qShares, testValue):
    """
    Test whether N is the product of two primes.
    If so, this test will succeed with probability 1.
    If N is not the product of two primes, the test will fail with at least
    probability 1/2.
    """

    values = [ int(powmod(testValue, x//4, N)) for x in [sum(y) for y in zip(list(pShares.values())[1:], list(qShares.values())[1:])] ]
    values = [ int(powmod(testValue, (N - pShares[1] - qShares[1] +1)//4, N)) ] + values
    product = mult_list(values[1:])
    if values[0] % N == product % N or values[0] % N == -product % N:
        # Test succeeds, so N is "probably" the product of two primes.
        return True

    # Test fails, so N is definitely not the product of two primes.
    return False


def mult_list(L):
    out=1
    for l in L:
        out=out*l
    return out


