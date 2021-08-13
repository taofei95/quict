"""
Encoders transform ladder operators to CompositeGate on zero state of qubits.
"""

from abc import ABC, abstractclassmethod
from qubit_operator import QubitOperator
from fermion_operator import FermionOperator
import numpy as np

class Encoder(ABC):
    """
    Abstract class of encoding methods.
    """
    def encoder(cls, fermion_operator, maxN):
        """
        Encoders transform ladder operators to Qubit Operators.

        Args:
            fermion_operator(FermionOperator): FermionOperator to be transformed.
            maxN(integer): the number of all the fermions.

        Returns:
            QubitOperator: The corresponding operators on qubits.
        """
        ans = QubitOperator(0)
        for mono_f in fermion_operator.operators:
            mono_q = QubitOperator([],mono_f[1])
            for operator in mono_f[0]:
                mono_q *= cls.encoder_single(operator[0], operator[1], maxN)
            ans += mono_q
        return ans
    
    @abstractclassmethod
    def encoder_single(cls, target, kind, maxN):
        '''
        Encode a single ladder operator (To be overrided).

        Args:
            latter_operator(tuple): a single ladder operator to be transformed.
            maxN(integer): the number of all the fermions.

        Returns:
            QubitOperator: The corresponding operators on qubits.
        '''
        raise Exception("The encoder is not realized.")
    
def Trans_01(target):
    '''
    Construct the operator |0><1| on the target qubit.
    The next three functions are similar.

    Args:
        target(integer): id of the target qubit.
    
    Returns:
        QubitOperator: '0.5X+0.5iY' on the target qubit
    '''
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], 0.5j)

def Trans_10(target):
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], -0.5j)

def Trans_00(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], 0.5)

def Trans_11(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], -0.5)



class JordanWigner(Encoder):
    """
    Implement the Jordan-Wigner encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, maxN):
        Zlist = [(i, 3) for i in range(target)]
        ans = FermionOperator(Zlist)
        if kind == 0:               #annihilation
            ans *= Trans_01(target)
        else:                       #creation
            ans *= Trans_10(target)
        return ans

class Parity(Encoder):
    """
    Implement the parity encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, maxN):
        Xlist = [(i, 1) for i in range(target+1, maxN)]
        ans = FermionOperator(Xlist)
        if kind == 0:               #annihilation
            if target == 0:
                ans *= Trans_01(target)
            else:
                ans *= Trans_00(target - 1) * Trans_01(target) - Trans_11(target - 1) * Trans_10(target)
        else:                       #creation
            if target == 0:
                ans *= Trans_10(target)
            else:
                ans *= Trans_00(target - 1) * Trans_10(target) - Trans_11(target - 1) * Trans_01(target)
        return ans

class BravyiKitaev(Encoder):
    """
    Implement the Bravyi-Kitaev encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, maxN):
        pass
