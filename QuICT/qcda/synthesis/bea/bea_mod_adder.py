#!/usr/bin/env python
# -*- coding:utf8 -*-

from .._synthesis import Synthesis
from QuICT.core import Circuit, CX, CCX, Swap, X

def FourierAdderMod(a, N, phib, c, low):
    """ use FourierAdd to calculate (a+b)%N in Fourier space

    a,N: (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)


    Args:
        a(int):   
        N(int):   
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 2,
        low(Qureg):  the clean ancillary qubit, length is 1,
    """
    #TODO

class BEAAdderModModel(BEAModel):
    """ 
    a circuit calculate `Φ((a+b)%N)`, `Φ(b)` are gotten from some qubits.
    `a` and `N` are hardwired.

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)

    Quregs:
        phib: the qureg stores b,        length is n+1,
        c:    the control qubits,        length is 2,
        low:  the clean ancillary qubit, length is 1,

        
    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    def __call__(self,a,N,n):
        """ Overload the function __call__,

        Args:
            a: the classical number
            N: the modulus
            n: the length of b. n+1 qubits will be used to avoid overflow
        Returns:
            BEAAdderModModel: the model filled by parameters.
        """

        self.pargs = [a,N,n]
        return self