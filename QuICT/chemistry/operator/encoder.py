"""
Encoders transform ladder operators to CompositeGate on zero state of qubits.
"""

from QuICT.chemistry.operator.qubit_operator import QubitOperator
import numpy as np

class Encoder:
    """
    Superclass of encoding methods.
    """
    @classmethod
    def encoder(cls, fermion_operator, n_fermions):
        """
        Encoders transform ladder operators to Qubit Operators.

        Args:
            fermion_operator(FermionOperator): FermionOperator to be transformed.
            n_fermions(integer): the number of all the fermions.

        Returns:
            QubitOperator: The corresponding operators on qubits.
        """
        ans = QubitOperator(0)
        for mono_f in fermion_operator.operators:
            mono_q = QubitOperator([],mono_f[1])
            for operator in mono_f[0]:
                mono_q *= cls.encoder_single(operator[0], operator[1], n_fermions)
            ans += mono_q
        return ans
    
    @classmethod
    def encoder_single(cls, target, kind, n_fermions):
        """
        Encode a single ladder operator (To be overrided).

        Args:
            latter_operator(tuple): a single ladder operator to be transformed.
            n_fermions(integer): the number of all the fermions.

        Returns:
            QubitOperator: The corresponding operators on qubits.
        """
        raise Exception("The encoder is not realized.")
    
def trans_01(target):
    """
    Construct the operator |0><1| on the target qubit.
    The next three functions are similar.

    Args:
        target(integer): id of the target qubit.
    
    Returns:
        QubitOperator: '0.5X+0.5iY' on the target qubit
    """
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], 0.5j)

def trans_10(target):
    return QubitOperator([(target, 1)], 0.5) + QubitOperator([(target, 2)], -0.5j)

def trans_00(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], 0.5)

def trans_11(target):
    return QubitOperator([], 0.5) + QubitOperator([(target, 3)], -0.5)



class JordanWigner(Encoder):
    """
    Implement the Jordan-Wigner encoding method
    """         
    @classmethod
    def encoder_single(cls, target, kind, n_fermions):
        Zlist = [(i, 3) for i in range(target)]
        ans = QubitOperator(Zlist)
        if kind == 0:               #annihilation
            ans *= trans_01(target)
        else:                       #creation
            ans *= trans_10(target)
        return ans

class Parity(Encoder):
    """
    Implement the parity encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, n_fermions):
        Xlist = [(i, 1) for i in range(target+1, n_fermions)]
        ans = QubitOperator(Xlist)
        if kind == 0:               #annihilation
            if target == 0:
                ans *= trans_01(target)
            else:
                ans *= trans_00(target - 1) * trans_01(target) - trans_11(target - 1) * trans_10(target)
        else:                       #creation
            if target == 0:
                ans *= trans_10(target)
            else:
                ans *= trans_00(target - 1) * trans_10(target) - trans_11(target - 1) * trans_01(target)
        return ans

def flip(x, n_fermions):
    """
    Construct a list of indexes involved in flipping n_x

    Args:
        x(integer): id of the target qubit.
        n_fermions(integer): the number of all the fermions.
    
    Returns:
        list: the indexes involved in flipping n_x
    """
    index = []
    x += 1
    while (x <= n_fermions):
        index.append(x - 1)
        x += x&(-x)
    return index

def sumup(x):
    """
    Construct a list of indexes involved in summation n_1,...,n_x

    Args:
        x(integer): id of the target qubit.
    
    Returns:
        list: the indexes involved in summation n_1,...,n_x
    """
    index = []
    x += 1
    while (x > 0):
        index.append(x - 1)
        x -= x&(-x)
    return index[::-1]

class BravyiKitaev(Encoder):
    """
    Implement the Bravyi-Kitaev encoding method
    """
    @classmethod
    def encoder_single(cls, target, kind, n_fermions):
        Xlist = [(i,1) for i in flip(target, n_fermions)]
        ans = QubitOperator(Xlist, 0.5)
        Zlist1 = [(i,3) for i in sumup(target - 1)]
        Zlist2 = [(i,3) for i in sumup(target)]
        ans *= QubitOperator(Zlist1) - QubitOperator(Zlist2, (-1)**kind)
        return ans
