#!/usr/bin/env/python3
# TNO, G. Spini, T. Attema
"""
Utility for Shamir secret sharing over the integers.
"""

import secrets
import math
from gmpy2 import invert

kappa = 40         # Statistical security (e.g. 40)
Max = 5            # Maximum integer size
n = 10             # Number of players
t = 4              # Reconstructrion threshold, at least t+1 players are needed to reconstruct the secret

class ShamirSecretSharingIntegers(object):
    
    def __init__(self,kappa,Max,n,t):
        self.kappa = kappa
        self.Max = Max
        self.n = n
        self.t = t
        # Random polynomial coefficients are sampled uniformly at random from the interval [-A,A], with A as follows:
        self.randomness_interval = (math.factorial(n)**2)*(2**kappa)*Max

        # Vandermonde matrix for evaluation of polynomials at points [1,..,n]
        self.Vm = [[pow(i+1,j) for j in range(t+1)] for i in range(n)]

    def share_secret(self,s):
        #Sample random polynomial of degree t with constant coeffiecient
        N = self.randomness_interval
        secret_poly = [math.factorial(self.n)*s] + [secrets.randbelow(2*N+1)-N for _ in range(self.t)]
        # print("secret poly", secret_poly) #Debug
        # Create an array of all the shares
        # Player IDs are equal to the points of evaluation. 
        shares = {ind+1: sum([self.Vm[ind][i]*secret_poly[i] for i in range(self.t+1)]) for ind in range(self.n)}
        scaling = math.factorial(self.n)
        Sharing = IntegerShares(self,shares,self.t,scaling)
        return Sharing

class IntegerShares(object):

    def __init__(self,ShamirSSS,shares,degree,scaling):
        self.scheme = ShamirSSS
        self.shares = shares
        self.degree = degree                       # The degree of the polynomial used for sharing the secret, i.e. at least degree+1 shares are required to reconstruct.
        self.n = len(self.shares)
        self.nFac = math.factorial(self.n)
        self.scaling = scaling

    def reconstruct_secret(self, modulus=0):
        if len(self.shares)< self.degree+1:
            raise ValueError('Too little shares to reconstruct.')

        # We will use the first self.degree+1 shares to reconstruct. This can be any subset.
        # Hence, here the reconstruction set is implicitely defined.
        reconstruction_shares = {key: self.shares[key] for key in list(self.shares.keys())[:self.degree+1]}

        # We precompute some values so that we can do the Langrage interpolation.
        lagrange_interpol_enum = {i:mult_list([j for j in reconstruction_shares.keys() if i != j]) for i in reconstruction_shares.keys()}
        lagrange_interpol_denom = {i:mult_list([ (j-i) for j in reconstruction_shares.keys() if i!=j ]) for i in reconstruction_shares.keys()}

        # modulus=0 is treated as standard reconstruction.
        # Otherwise, reconstruct modulo modulus
        if modulus == 0:
            partial_recon = [(lagrange_interpol_enum[i]*self.nFac*reconstruction_shares[i])//lagrange_interpol_denom[i] for i in reconstruction_shares.keys()] # The fractions in this list are all integral.
            secret = sum(partial_recon)//(self.scaling * self.nFac)     # The scaling factor is a divisor of the enumerator. 
        else:
            if self.scaling % modulus == 0:
                raise ValueError("Scaling is not divisible mod modulus")
            partial_recon = [((lagrange_interpol_enum[i]*self.nFac*(reconstruction_shares[i] % modulus))//lagrange_interpol_denom[i]) % modulus for i in reconstruction_shares.keys()] # The fractions in this list are all integral.
            secret = ( sum(partial_recon) * int(invert(self.scaling * self.nFac, modulus)) ) % modulus
        return secret

    def __add__(self,other):
        if (self.scheme != other.scheme):
            raise ValueError("Different secret sharing schemes have been used, i.e. shares are incompatible.")
        
        if (self.scaling != other.scaling):
            raise ValueError("Incompatible sharess, different scaling factors.")
        
        shares= {i:(self.shares[i]+other.shares[i]) for i in self.shares.keys()}
        degree = max(self.degree,other.degree)
        scaling = self.scaling
        return IntegerShares(self.scheme,shares,degree,scaling)

    def __mul__(self,other):
        if (self.scheme != other.scheme):
            # If self is multiplied (from the right) by another object, we redirect to the __rmul__  method of that
            # object. Only implemented when other is a shamir secret sharing. In this case all shares are reduced modulo
            # the Shamir modulus and a shamir sharing is returned.
            return NotImplemented
        shares= {i:(self.shares[i]*other.shares[i]) for i in self.shares.keys()}
        degree = self.degree + other.degree
        scaling = self.scaling * other.scaling
        return IntegerShares(self.scheme,shares,degree,scaling)

    def __rmul__(self,other):

        if isinstance(other,int):
            # Scalar multiplication from the left by an integer
            shares = {i: (other * self.shares[i]) for i in self.shares.keys()}
            degree = self.degree
            scaling = self.scaling
            return IntegerShares(self.scheme, shares, degree, scaling)
        else:
            # We redirect to the __rmul__ functionality of other.
            return self*other

def sign(a):
    return 2*(a>=0)-1

def mult_list(L):
    out=1
    for l in L:
        out=out*l
    return out

# Debugging
#ShamirSSS = ShamirSecretSharingIntegers(kappa,Max,n,t)
#a=ShamirSSS.share_secret(20)
#print(a.shares)
#print((a*a).reconstruct_secret())
#print('\n')
