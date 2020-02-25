#!/usr/bin/env/python3
# TNO, G. Spini, T. Attema
"""
Utility for Shamir secret sharing.
"""

import sympy as sp
from gmpy2 import invert
import secrets
import warnings
from .shamir_secret_sharing_integers import IntegerShares

class ShamirSecretSharingScheme(object):

    def __init__(self,P,n,t):
        self.P = P
        self.n = n
        self.t = t
        # Vandermonde matrix for evaluation of polynomials at points [1,..,n]
        self.Vm = [[pow(i+1,j,P) for j in range(t+1)] for i in range(n)]

    def share_secret(self,s):
        #Sample random polynomial of degree t with constant coeffiecient
        secret_poly = [s] + [secrets.randbelow(self.P) for _ in range(self.t)]
        # Create an array of all the shares
        # Player IDs are equal to the points of evaluation. 
        shares = {ind+1: sum([self.Vm[ind][i]*secret_poly[i] % self.P for i in range(self.t+1)]) for ind in range(self.n)}
        Sharing = ShamirShares(self,shares,self.t)
        return Sharing

class ShamirShares(object):

    def __init__(self,ShamirSSS,shares,degree):
        self.scheme = ShamirSSS
        self.shares = shares
        self.degree = degree                       # The degree of the polynomial used for sharing the secret, i.e. at least degree+1 shares are required to reconstruct.

    def reconstruct_secret(self):
        if len(self.shares)< self.degree+1:
            raise ValueError('Too little shares to reconstruct.')
        
        # We will use the first self.degree+1 shares to reconstruct. This can be any subset.
        # Hence, here the reconstruction set is implicitely defined.
        reconstruction_shares = {key: self.shares[key] for key in list(self.shares.keys())[:self.degree+1]}

        # We precompute some values so that the Langrage interpolation.
        # We assume that we can always use shares f(1), ... f(self.degree+1)

        lagrange_interpol_enum = {i: mult_list([j for j in reconstruction_shares.keys() if i != j],self.scheme.P) for i in reconstruction_shares.keys()}
        lagrange_interpol_denom = {i: mult_list([(j - i) for j in reconstruction_shares.keys() if i != j], self.scheme.P) for i in reconstruction_shares.keys()}

        secret = int(sum(lagrange_interpol_enum[i]*invert(lagrange_interpol_denom[i],self.scheme.P)*reconstruction_shares[i] % self.scheme.P for i in reconstruction_shares.keys()) % self.scheme.P)
        return secret

    def __add__(self,other):
        if (self.scheme != other.scheme):
            raise ValueError("Different secret sharing schemes have been used, i.e. shares are incompatible.")

        shares= {i:(self.shares[i]+other.shares[i]) % self.scheme.P for i in self.shares.keys()}
        degree = max(self.degree,other.degree)
        return ShamirShares(self.scheme,shares,degree)

    def __mul__(self,other):
        if (self.scheme != other.scheme):
            # If self is multiplied (from the right) by another object, we redirect to the __rmul__  method of that
            # object. Only implemented when other is a shamir secret sharing over the integers. In this case all shares
            # are reduced modulo the Shamir modulus and a shamir sharing is returned.
            return NotImplemented

        shares= {i:(self.shares[i]*other.shares[i]) % self.scheme.P for i in self.shares.keys()}
        degree = self.degree + other.degree
        return ShamirShares(self.scheme,shares,degree)

    def __rmul__(self,other):

        if isinstance(other,int):
            # Scalar multiplication from the left by an integer
            shares= {i:(other*self.shares[i]) % self.scheme.P for i in self.shares.keys()}
            degree = self.degree
            return ShamirShares(self.scheme,shares,degree)
        elif isinstance(other,IntegerShares):
            # Multiply by a sharing over the integers and return a Shamir Sharing
            # NB: This operation returns a Shamir sharing which inherits the statistical security of the integer
            # sharing and should therefore only be used with caution.
            warnings.warn("Caution multiplying integer shares by shamir shares.")

            shares = {i: (self.shares[i] * other.shares[i] * invert(other.scaling,self.scheme.P)) % self.scheme.P for i in self.shares.keys()}
            degree = self.degree + other.degree
            return ShamirShares(self.scheme, shares, degree)
        else:
            raise ValueError("Different secret sharing schemes have been used, i.e. shares are incompatible.")


def sign(a):
    return 2*(a>=0)-1

def mult_list(L,modulus):
    out=1
    for l in L:
        out=out*l % modulus
    return out

# Debugging

#P = 73               #sp.nextprime(2**5)              # prime field size
#n = 3               # Number of players
#t = 1               # Reconstructrion threshold, at least t+1 players are needed to reconstruct the secret

#ShamirSSS = ShamirSecretSharingScheme(P,n,t)
#a=ShamirSSS.share_secret(3)
#print(a.shares)
#print(a.reconstruct_secret())
#print('\n')
#SS_int=ShamirSecretSharingIntegers(40,P,n,t)
#Shares_int=SS_int.share_secret(17)
#b=Shares_int*a
#print(b.reconstruct_secret())

#print('end')
